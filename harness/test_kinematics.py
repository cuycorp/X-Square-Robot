"""Offline checks for the URDF-based kinematics preview."""

import json
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from jog_console import DesktopJogger
from kinematics import arm_chain, jog_target, preview
from server import PromptController, start_console_http_server


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
