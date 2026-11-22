"""Offline checks for the URDF-based kinematics preview."""

import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from websockets.sync.server import serve

from claude_policy import ClaudeEndPosePlanner, hold_pose_decision
from cube_depth import localize_cube
from direct_claude import CAMERA_TOPICS, DirectClaudeControl
from jog_console import DesktopJogger
from kinematics import arm_chain, jog_target, preview
from server import PromptController, start_console_http_server
from sim_cpu_env import TabletopState, build_observation_payload


class TestKinematicsPreview(unittest.TestCase):
    def setUp(self) -> None:
        base = "base_link"
        joints = []
        for index, axis in enumerate(("0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"), 1):
            child = f"left_arm_link{index}"
            joints.append(
                f'<joint name="left_arm_joint{index}" type="revolute">'
                f'<parent link="{base}"/><child link="{child}"/>'
                f'<origin xyz="0.12 0 0"/><axis xyz="{axis}"/>'
                '<limit lower="-2" upper="2"/></joint>'
            )
            base = child
        joints.append(
            '<joint name="left_gripper" type="fixed">'
            f'<parent link="{base}"/><child link="left_arm_gripper_base_link"/>'
            '<origin xyz="0.08 0 0"/></joint>'
        )
        joints.append(
            '<joint name="left_arm_gripper" type="revolute">'
            '<parent link="left_arm_gripper_base_link"/><child link="left_gripper_motor"/>'
            '<limit lower="0" upper="1.526"/></joint>'
        )
        left = "".join(joints)
        self.urdf = '<robot name="test">' + left + left.replace("left", "right") + "</robot>"
        self.joints = {f"left_arm_joint{i}": 0.0 for i in range(1, 7)}

    def test_forward_and_inverse_kinematics_round_trip(self) -> None:
        origin = preview(self.urdf, "left", self.joints, None)["pose"]
        np.testing.assert_allclose(origin, [0.80, 0, 0, 0, 0, 0], atol=1e-9)
        desired = {**self.joints, "left_arm_joint1": 0.15, "left_arm_joint2": 0.25}
        target = preview(self.urdf, "left", desired, None)["pose"]
        result = preview(self.urdf, "left", self.joints, target)
        self.assertTrue(result["converged"])
        self.assertLessEqual(result["position_error_m"], 0.002)
        self.assertLessEqual(result["rotation_error_rad"], 0.02)
        self.assertTrue(all(-2 <= angle <= 2 for angle in result["joints"].values()))

    def test_unreachable_target_has_no_joint_solution(self) -> None:
        result = preview(self.urdf, "left", self.joints, [4, 0, 0, 0, 0, 0])
        self.assertFalse(result["converged"])
        self.assertEqual(result["joints"], {})

    def test_requires_complete_joint_feedback(self) -> None:
        self.joints.pop("left_arm_joint4")
        with self.assertRaisesRegex(ValueError, "all six joints"):
            preview(self.urdf, "left", self.joints, None)

    def test_joint_jog_offsets_only_the_selected_joint(self) -> None:
        chain = arm_chain(self.urdf, "left")
        target = jog_target(chain, self.joints, "j3", -1, 1)
        self.assertEqual(target, {**self.joints, "left_arm_joint3": -np.deg2rad(1)})
        with self.assertRaisesRegex(ValueError, "joint limit"):
            jog_target(chain, {**self.joints, "left_arm_joint3": 2}, "j3", 1, 1)

    def test_cartesian_jog_uses_ik_and_preserves_orientation(self) -> None:
        seed = {**self.joints, "left_arm_joint2": 0.35, "left_arm_joint3": -0.60}
        start = preview(self.urdf, "left", seed, None)["pose"]
        joints = jog_target(arm_chain(self.urdf, "left"), seed, "z", 1, 2)
        end = preview(self.urdf, "left", joints, None)["pose"]
        np.testing.assert_allclose(np.subtract(end[:3], start[:3]), [0, 0, 0.002], atol=0.00025)
        np.testing.assert_allclose(end[3:], start[3:], atol=0.003)

    def test_live_jog_sends_one_target_to_selected_arm_and_rejects_rapid_repeat(self) -> None:
        robot = Mock()
        robot.robot_control.get_manipulator_control_mode.return_value.mode = 1
        robot.left_arm.get_joint_states.side_effect = [
            SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)),
                name=list(self.joints), position=list(self.joints.values()), velocity=[0] * 6,
            )
            for stamp in (1, 2)
        ]
        robot.left_arm.set_joint_positions.return_value.is_success = True
        jogger = DesktopJogger(robot, 1, SimpleNamespace, self.urdf)
        target = jogger("left", "j2", 1, 1)
        self.assertAlmostEqual(target["left_arm_joint2"], np.deg2rad(1))
        robot.left_arm.set_joint_positions.assert_called_once_with(
            SimpleNamespace(positions=[0, np.deg2rad(1), 0, 0, 0, 0]), timeout=1
        )
        robot.right_arm.set_joint_positions.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "settle"):
            jogger("left", "j2", 1, 1)

    def test_home_and_gripper_use_selected_sdk_actuator(self) -> None:
        for action in ("home", "open", "close"):
            with self.subTest(action=action):
                robot = Mock()
                robot.robot_control.get_manipulator_control_mode.return_value.mode = 1
                robot.right_arm.get_joint_states.side_effect = [
                    SimpleNamespace(
                        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)),
                        name=[name.replace("left", "right") for name in self.joints],
                        position=list(self.joints.values()), velocity=[0] * 6,
                    )
                    for stamp in (1, 2)
                ]
                robot.right_arm.reset.return_value.is_success = True
                robot.right_gripper.set_position.return_value.is_success = True
                jogger = DesktopJogger(robot, 1, SimpleNamespace, self.urdf, SimpleNamespace)
                self.assertIn("SDK accepted", jogger.action("right", action))
                if action == "home":
                    robot.right_arm.reset.assert_called_once_with(timeout=15)
                    robot.right_gripper.set_position.assert_not_called()
                else:
                    robot.right_gripper.set_position.assert_called_once_with(
                        SimpleNamespace(position=1.526 if action == "open" else 0), timeout=2
                    )
                    robot.right_arm.reset.assert_not_called()
                robot.left_arm.reset.assert_not_called()
                robot.left_gripper.set_position.assert_not_called()

    def test_manual_console_rejects_phantom_prompts_and_checks_action_origin(self) -> None:
        motion = Mock(return_value="SDK accepted")
        server = start_console_http_server(
            "127.0.0.1", 0, PromptController(), live_action=motion, prompts_enabled=False
        )
        origin = f"http://127.0.0.1:{server.server_port}"

        def post(path, data, origin_header):
            request = urllib.request.Request(
                origin + path,
                data=json.dumps(data).encode(),
                headers={"Origin": origin_header, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.load(response)

        try:
            with self.assertRaises(urllib.error.HTTPError) as bad_prompt:
                post("/api/prompt", {"prompt": "move"}, origin)
            self.assertEqual(bad_prompt.exception.code, 409)
            bad_prompt.exception.close()
            with self.assertRaises(urllib.error.HTTPError) as bad_origin:
                post("/api/manual", {"arm": "right", "action": "home"}, "http://other.test")
            self.assertEqual(bad_origin.exception.code, 403)
            bad_origin.exception.close()
            motion.assert_not_called()
            self.assertEqual(post("/api/manual", {"arm": "right", "action": "home"}, origin),
                             {"message": "SDK accepted"})
            motion.assert_called_once_with("right", "home")
        finally:
            server.shutdown()
            server.server_close()

    def test_direct_claude_runs_left_arm_through_the_live_sdk_endpoint(self) -> None:
        robot = Mock()
        robot.robot_control.get_manipulator_control_mode.return_value.mode = 1
        seed = {**self.joints, "left_arm_joint2": 0.35, "left_arm_joint3": -0.60}
        left_positions = list(seed.values())
        ticks = iter(range(1, 100))

        def feedback(arm):
            return SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=next(ticks))),
                name=[name.replace("left", arm) for name in seed],
                position=left_positions.copy() if arm == "left" else list(seed.values()), velocity=[0] * 6,
            )

        robot.left_arm.get_joint_states.side_effect = lambda **_: feedback("left")
        robot.right_arm.get_joint_states.side_effect = lambda **_: feedback("right")
        for gripper in (robot.left_gripper, robot.right_gripper):
            gripper.get_position.return_value.position = 1.0
        def accept_target(command, **_):
            left_positions[:] = command.positions
            return SimpleNamespace(is_success=True)

        robot.left_arm.set_joint_positions.side_effect = accept_target
        jogger = DesktopJogger(robot, 1, SimpleNamespace, self.urdf, SimpleNamespace)
        images = build_observation_payload(TabletopState(), "hold")["observation"]["images"]
        key_planner = ClaudeEndPosePlanner(api_key="test-key")
        status = PromptController()
        finished = threading.Event()
        original_finish = status.finish_direct

        def finish(*args):
            original_finish(*args)
            finished.set()

        status.finish_direct = finish
        planner = Mock()
        planner.plan_every = 4
        arrived = threading.Event()
        resume = threading.Event()

        def decide(observation):
            if planner.decide.call_count == 1:
                arrived.set()
                if not resume.wait(3):
                    raise RuntimeError("Test did not release the planner.")
                decision = hold_pose_decision(observation, stage="lift", status="in_progress")
                decision.left_target_pos[2] += 0.03
                return decision
            return hold_pose_decision(observation)

        planner.decide.side_effect = decide
        def rosbridge(socket):
            for _ in CAMERA_TOPICS:
                request = json.loads(socket.recv(timeout=2))
                topic = request["topic"]
                socket.send(json.dumps({
                    "op": "publish", "topic": topic,
                    "msg": {"format": "jpeg", "data": images[CAMERA_TOPICS[topic]]},
                }))

        camera_server = serve(rosbridge, "127.0.0.1", 0, close_timeout=0.5)
        threading.Thread(target=camera_server.serve_forever, daemon=True).start()
        control = DirectClaudeControl(jogger, key_planner, status, f"ws://127.0.0.1:{camera_server.socket.getsockname()[1]}/")
        server = start_console_http_server(
            "127.0.0.1", 0, status, live_jog=jogger, prompts_enabled=False,
            key_planner=key_planner, direct_control=control,
        )
        origin = f"http://127.0.0.1:{server.server_port}"

        def post(path, payload):
            request = urllib.request.Request(
                origin + path, data=json.dumps(payload).encode(),
                headers={"Origin": origin, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.load(response)

        try:
            with patch("direct_claude.ClaudeEndPosePlanner", return_value=planner):
                self.assertTrue(post("/api/prompt", {"prompt": "Lift the cube"})["ok"])
                self.assertTrue(arrived.wait(3))
                self.assertTrue(post("/api/direct/heartbeat", {})["ok"])
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    jogger("left", "j1", 1, 1)
                resume.set()
                self.assertTrue(finished.wait(5))
                self.assertEqual(status.snapshot()["status"], "completed")
                previous_step = status.snapshot()["motion_feedback"]["previous_step"]
                self.assertIn("accepted", previous_step["command"])
                self.assertAlmostEqual(previous_step["measured_left_delta_xyz_mm"][2], 30, delta=1.5)
                self.assertEqual(planner.decide.call_args.args[0].perception["last_sdk_step"], previous_step)
                calls = robot.left_arm.set_joint_positions.call_args_list
                self.assertGreaterEqual(len(calls), 2)
                prev_q = np.array(list(seed.values()), dtype=float)
                for call in calls:
                    next_q = np.array(call.args[0].positions, dtype=float)
                    self.assertLessEqual(float(np.max(np.abs(next_q - prev_q))), np.deg2rad(0.65))
                    prev_q = next_q
                robot.right_arm.set_joint_positions.assert_not_called()
                robot.left_gripper.set_position.assert_not_called()
                robot.right_gripper.set_position.assert_not_called()
                command = calls[-1].args[0].positions
                initial = preview(self.urdf, "left", seed, None)["pose"]
                target = preview(self.urdf, "left", dict(zip(seed, command)), None)["pose"]
                np.testing.assert_allclose(np.subtract(target[:3], initial[:3]), [0, 0, 0.03], atol=0.0012)

                finished.clear()
                arrived.clear()
                resume.clear()
                planner.reset_mock()
                robot.left_arm.set_joint_positions.reset_mock()
                self.assertTrue(post("/api/prompt", {"prompt": "Lift the cube again", "use_cube_depth": False})["ok"])
                self.assertTrue(arrived.wait(3))
                first_instruction = planner.decide.call_args.args[0].instruction
                self.assertIn("RGB-only mode", first_instruction)
                self.assertNotIn("local_perception.cube_depth reports", first_instruction)
                self.assertTrue(post("/api/direct/stop", {})["ok"])
                resume.set()
                self.assertTrue(finished.wait(5))
                self.assertEqual(status.snapshot()["status"], "stopped")
                robot.left_arm.set_joint_positions.assert_not_called()
        finally:
            control.stop()
            resume.set()
            server.shutdown()
            server.server_close()
            camera_server.shutdown()

    def test_localize_cube_in_base_link_from_depth_and_color(self) -> None:
        depth_array = np.full((120, 160), 600, np.uint16)
        depth_array[52:69, 72:89] = 545
        png = io.BytesIO()
        Image.fromarray(depth_array).save(png, format="PNG")
        depth = {
            "header": {"frame_id": "depth_optical"},
            "format": "16UC1; compressedDepth png",
            "data": base64.b64encode(bytes(12) + png.getvalue()).decode(),
        }
        rgb = {"header": {"frame_id": "color_optical"}, "data": "mock-rgb"}
        info = {
            "width": 160, "height": 120, "distortion_model": "rational_polynomial",
            "k": [160, 0, 80, 0, 160, 60, 0, 0, 1], "d": [0] * 8,
        }
        mount = {
            "header": {"frame_id": "base_link"},
            "transform": {
                "rotation": {"x": 1, "y": 0, "z": 0, "w": 0},
                "translation": {"x": 0.25, "y": 0.10, "z": 0.60},
            },
        }
        quad = [[72/160, 52/120], [88/160, 52/120], [88/160, 68/120], [72/160, 68/120]]
        with patch("cube_depth.detect_cube_cv", return_value={
            "visible_faces": [{"face": "U", "known_stickers": 9, "quad": quad}, {"face": "F"}],
        }):
            result = localize_cube(
                rgb, depth, {**info, "header": rgb["header"]},
                {**info, "header": depth["header"]},
                {"color_optical": mount, "depth_optical": mount}, np.eye(4),
            )
        self.assertEqual(result["cube_top_center_m"], [0.25, 0.1, 0.055])
        self.assertAlmostEqual(result["table_z_m"], 0)
