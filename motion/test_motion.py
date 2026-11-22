"""Offline tests for kinematics, safety checks and command parsing. No robot, no Nebius calls.
Run: .venv/bin/python -m unittest -v motion/test_motion.py
Tests marked 'needs URDF' use the cached robot description (.cache/robot_description.urdf, created by the web UI)."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "perception"))

import io  # noqa: E402

from PIL import Image  # noqa: E402

import command_parser  # noqa: E402
import grasp  # noqa: E402
import kinematics as K  # noqa: E402
import safety  # noqa: E402
import vision3d as V  # noqa: E402

URDF_PATH = HERE.parent / ".cache" / "robot_description.urdf"
needs_urdf = unittest.skipUnless(URDF_PATH.exists(), "needs URDF: run the web UI once while the robot is reachable")
# A pose measured on the real robot (arms raised, grippers ~12 cm apart), in radians.
LIVE_Q = {"left": np.radians([-33.3, 59.4, -32.2, 2.2, 5.3, 7.9]),
          "right": np.radians([36.5, 63.2, -40.4, 4.8, -1.1, -5.5])}


class RotationMathTests(unittest.TestCase):
    def test_rpy_round_trip(self):
        for rpy in ([0.1, -0.4, 2.0], [-1.2, 0.3, -0.7], [0, 0, 0]):
            np.testing.assert_allclose(K.matrix_to_rpy(K.rpy_to_matrix(*rpy)), rpy, atol=1e-9)

    def test_quat_round_trip(self):
        R = K.rpy_to_matrix(0.5, -0.2, 1.1)
        q = K.matrix_to_quat(R)
        np.testing.assert_allclose(K.quat_to_matrix(*q), R, atol=1e-9)

    def test_rotation_error_is_zero_for_same_rotation(self):
        R = K.rpy_to_matrix(0.3, 0.2, 0.1)
        np.testing.assert_allclose(K.rotation_error(R, R), np.zeros(3), atol=1e-9)


@needs_urdf
class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = URDF_PATH.read_text()
        cls.arm = K.Arm(cls.urdf, "left")

    def test_chain_has_six_joints_with_limits(self):
        self.assertEqual(len(self.arm.names), 6)
        self.assertTrue(np.all(self.arm.lower < self.arm.upper))

    def test_neutral_pose_matches_robot(self):
        # Measured: FK(0) of link6 in the arm base is (0.1185, 0, 0.181) with no rotation.
        T0 = self.arm.fk(np.zeros(6))
        np.testing.assert_allclose(T0[:3, 3], [0.1185, 0.0, 0.181], atol=1e-3)
        np.testing.assert_allclose(T0[:3, :3], np.eye(3), atol=1e-6)

    def test_ik_round_trip(self):
        q0 = LIVE_Q["left"]
        T = self.arm.fk(q0)
        T[:3, 3] += [0.02, -0.01, 0.015]
        q1, pos_err, rot_err = self.arm.ik(T, q0)
        self.assertLess(pos_err, 1e-3)
        self.assertLess(np.degrees(rot_err), 0.5)

    def test_fingertip_offset(self):
        np.testing.assert_allclose(K.fingertip_offset(self.urdf, "left"), [0.217, 0.0, -0.01], atol=2e-3)


@needs_urdf
class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = safety.SafetyPlanner(URDF_PATH.read_text(), safety.load_config())
        cls.state = {s: {"q": list(LIVE_Q[s]), "gripper": 0.0} for s in K.ARMS}

    def plan(self, *moves, state=None):
        r = self.P.plan(list(moves), state or self.state)
        return r, {c["name"]: c for c in r["checks"]}

    def test_small_move_is_allowed(self):
        r, c = self.plan(safety.Move("left", np.array([0, 0, 0.02])))
        self.assertTrue(r["ok"], [x for x in r["checks"] if not x["ok"]])
        self.assertIn("left", r["targets"])

    def test_big_step_is_blocked(self):
        r, c = self.plan(safety.Move("left", np.array([0, 0, 0.08])))
        self.assertFalse(r["ok"])
        self.assertFalse(c["step size"]["ok"])

    def test_big_rotation_is_blocked(self):
        r, c = self.plan(safety.Move("left", rotate_rad=np.radians([0, 0, 40])))
        self.assertFalse(c["rotation size"]["ok"])

    def test_arms_too_close_is_blocked(self):
        # Grippers start ~12 cm apart; moving both 3 cm toward each other brings them under 10 cm.
        r, c = self.plan(safety.Move("left", np.array([0, -0.03, 0])), safety.Move("right", np.array([0, 0.03, 0])))
        self.assertFalse(c["arm separation (path)"]["ok"], c["arm separation (path)"]["detail"])

    def test_moving_apart_is_allowed_even_when_close(self):
        r, c = self.plan(safety.Move("left", np.array([0, 0.02, 0])))
        self.assertTrue(c["arm separation (path)"]["ok"])

    def test_moving_away_from_a_joint_limit_is_allowed(self):
        # Folded rest pose: joint2 = 0 and joint3 = 0 sit exactly on their limits.
        q0 = np.zeros(6)
        check = self.P._joint_limits("left", q0, q0 + np.array([0, 0.05, -0.05, 0, 0, 0]))
        self.assertTrue(check["ok"], check["detail"])
        self.assertIn("moving away", check["detail"])

    def test_moving_into_a_joint_limit_is_blocked(self):
        a = self.P.arms["left"]
        q0 = np.array(LIVE_Q["left"])
        q1 = q0.copy()
        q1[3] = a.upper[3] - np.radians(1)
        check = self.P._joint_limits("left", q0, q1)
        self.assertFalse(check["ok"])

    def test_one_move_per_arm(self):
        r, _ = self.plan(safety.Move("left", np.array([0, 0, 0.01])), safety.Move("left", np.array([0, 0, 0.01])))
        self.assertFalse(r["ok"])

    def test_preview_is_relative_to_neutral(self):
        r, _ = self.plan(safety.Move("left", np.array([0, 0, 0.02]), gripper="open"))
        msgs = r["targets"]["left"]["ros_preview"]
        self.assertEqual(msgs[0]["msg"]["header"]["frame_id"], "")
        self.assertEqual(msgs[1]["topic"], "/left_gripper_controller/commands")


class ContactTests(unittest.TestCase):
    def test_contact_alert_on_force_change(self):
        alerts = safety.contact_alerts({"left": [44, 9, 41], "right": [0, 0, 0]},
                                       {"left": [44, 9, 20], "right": [0, 0, 1]}, 15.0)
        self.assertEqual(len(alerts), 1)
        self.assertIn("left", alerts[0])


class ParserTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        text = '```json\n{"moves": [{"arm": "left", "translate_cm": {"z": 2}, "gripper": "open"}], "explanation": "up"}\n```'
        out = command_parser.parse_reply(text)
        self.assertEqual(len(out["moves"]), 1)
        m = out["moves"][0]
        np.testing.assert_allclose(m.translate_m, [0, 0, 0.02])
        self.assertEqual(m.gripper, "open")

    def test_clarification(self):
        out = command_parser.parse_reply('{"moves": [], "clarification": "Which arm?"}')
        self.assertEqual(out["moves"], [])
        self.assertEqual(out["clarification"], "Which arm?")

    def test_rejects_bad_input(self):
        for bad in ("no json here", '{"moves": [{"arm": "middle"}]}',
                    '{"moves": [{"arm": "left", "translate_cm": {"x": "far"}}]}',
                    '{"moves": [{"arm": "left", "gripper": "smash"}]}',
                    '{"moves": [{"arm": "left", "translate_cm": {"x": 1}}, {"arm": "left", "translate_cm": {"x": 1}}]}'):
            with self.assertRaises(ValueError, msg=bad):
                command_parser.parse_reply(bad)

    def test_empty_move_becomes_clarification(self):
        out = command_parser.parse_reply('{"moves": [{"arm": "left"}]}')
        self.assertEqual(out["moves"], [])
        self.assertTrue(out["clarification"])


class PickParserTests(unittest.TestCase):
    def test_pick_command(self):
        out = command_parser.parse_reply('{"moves": [], "pick": {"object": "the Rubik\'s cube", "arm": null}, "explanation": "pick"}')
        self.assertEqual(out["pick"], {"object": "the Rubik's cube", "arm": None})
        self.assertIsNone(out["clarification"])

    def test_pick_bad_arm(self):
        with self.assertRaises(ValueError):
            command_parser.parse_reply('{"moves": [], "pick": {"object": "cube", "arm": "third"}}')


class BboxTests(unittest.TestCase):
    def test_normalised_to_pixels(self):
        box = grasp.parse_bbox('```json\n{"bbox_2d": [486, 574, 562, 735]}\n```', 1280, 720)
        np.testing.assert_allclose(box, [622.08, 413.28, 719.36, 529.2])

    def test_not_visible(self):
        self.assertIsNone(grasp.parse_bbox('{"bbox_2d": null}', 1280, 720))

    def test_invalid(self):
        for bad in ("nothing", '{"bbox": [1, 2, 3]}', '{"bbox_2d": [500, 10, 400, 20]}', '{"bbox_2d": [0, 0, 1200, 10]}'):
            with self.assertRaises(ValueError, msg=bad):
                grasp.parse_bbox(bad, 1280, 720)


class DepthAndCameraTests(unittest.TestCase):
    def test_decode_compressed_depth(self):
        depth = (np.arange(20, dtype=np.uint16).reshape(4, 5) * 100)
        buf = io.BytesIO()
        Image.fromarray(depth).save(buf, "PNG")
        out = V.decode_compressed_depth(b"\x00" * 12 + buf.getvalue())
        np.testing.assert_array_equal(out, depth)

    def test_project_deproject_round_trip(self):
        Kmat = np.array([[150.0, 0, 160], [0, 150.0, 100], [0, 0, 1]])
        T = K.transform(K.rpy_to_matrix(-np.pi / 2 - 0.8, 0, -np.pi / 2), np.array([0.5, 0.0, 0.53]))
        cam = V.Camera(320, 200, Kmat, T)
        depth = np.zeros((200, 320), np.float32)
        depth[120, 200] = 0.6
        pts, px = cam.deproject(depth)
        uv, z = cam.project(pts)
        np.testing.assert_allclose(uv[0], [200, 120], atol=1e-6)
        self.assertAlmostEqual(float(z[0]), 0.6, places=6)


@needs_urdf
class PickPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = safety.SafetyPlanner(URDF_PATH.read_text(), safety.load_config())
        cls.rest = {s: {"q": [0.0, 0.005, -0.02, 0.02, -0.01, 0.0], "gripper": 0.0} for s in K.ARMS}
        # The Rubik's cube as measured live (base_link, metres).
        cls.cube = V.Located(np.array([0.892, -0.012, 0.043]), 0.043, -0.013, 0.056, np.array([0.044, 0.053]), 300, [0, 0, 1, 1])

    def test_interpolation_respects_limits(self):
        T0 = K.transform(np.eye(3), np.zeros(3))
        T1 = K.transform(K.rpy_to_matrix(0, 1.0, 0.3), np.array([0.2, -0.1, 0.05]))
        poses = list(grasp._interpolate(T0, T1, 0.03, np.radians(15), 0.9))
        prev = T0
        for T in poses:
            self.assertLessEqual(np.linalg.norm(T[:3, 3] - prev[:3, 3]), 0.03)
            self.assertLessEqual(np.degrees(np.linalg.norm(K.rotation_error(T[:3, :3], prev[:3, :3]))), 15)
            prev = T
        np.testing.assert_allclose(poses[-1], T1, atol=1e-9)

    def test_pick_cube_from_rest(self):
        res = grasp.plan_pick(self.P, self.rest, self.cube)
        self.assertTrue(res["ok"], res["attempts"])
        steps = res["chosen"]["steps"]
        self.assertTrue(all(s["ok"] for s in steps))
        self.assertEqual([s["phase"] for s in steps][0], "open gripper")
        self.assertIn("close gripper", [s["phase"] for s in steps])
        lowest = min(s["tip"][2] for s in steps)
        self.assertGreaterEqual(lowest, self.cube.table_z + 0.01 - 1e-3)   # never closer than 1 cm to the table
        self.assertLessEqual(lowest, self.cube.top_z)                       # actually reaches down to the cube

    def test_requested_arm_is_used(self):
        res = grasp.plan_pick(self.P, self.rest, self.cube, arm="left")
        self.assertTrue(all(a["arm"] == "left" for a in res["attempts"]))

    def test_unreachable_object_is_blocked(self):
        far = V.Located(np.array([1.5, 0.0, 0.05]), 0.05, 0.0, 0.05, np.array([0.05, 0.05]), 300, [0, 0, 1, 1])
        self.assertFalse(grasp.plan_pick(self.P, self.rest, far)["ok"])


class FakeWS:
    """Stands in for the rosbridge socket: records what would be sent; never touches the robot."""

    def __init__(self, incoming=()):
        self.sent, self.incoming = [], list(incoming)

    def send(self, text):
        import json
        self.sent.append(json.loads(text))

    def recv(self, timeout=None):
        import json
        if self.incoming:
            return json.dumps(self.incoming.pop(0))
        raise TimeoutError


@needs_urdf
class SenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sender
        cls.S = sender
        cfg = safety.load_config()
        cls.P = safety.SafetyPlanner(URDF_PATH.read_text(), cfg)
        cls.q = np.radians([-33.3, 59.4, -32.2, 2.2, 5.3, 7.9])

    def make(self, convention="relative_to_neutral", live=True, backend="ros_pose_cmd"):
        snd = self.S.LiveSender("ws://unused", self.P, live)
        snd.cfg = {**snd.cfg, "pose_cmd_convention": convention, "step_timeout_s": 0.3, "republish_hz": 100,
                   "foreign_traffic_window_s": 0.05, "interp_hz": 1000}
        snd.backend = backend
        return snd

    def test_locked_until_convention_verified_and_live(self):
        self.assertTrue(self.make("unverified").blockers())
        self.assertTrue(self.make(live=False).blockers())
        self.assertFalse(self.make().blockers())
        with self.assertRaises(RuntimeError):
            self.S.pose_message(self.P, "left", self.q, "unverified")

    # ----- SDK backend, with a fake bridge (nothing reaches the robot) -----

    class FakeBridge:
        def __init__(self, status=None, follow=True):
            self.status = status or {"ok": True, "operation_mode": "sdk", "estop": False, "control_mode": "joint_positions"}
            self.sent, self.follow, self.q = [], follow, None

        def request(self, req, timeout=5.0):
            self.sent.append(req)
            if req["cmd"] == "status":
                return self.status
            if req["cmd"] == "joints" and self.follow:
                self.q = req["q"]
            return {"ok": True}

    def sdk_sender(self, bridge):
        snd = self.make(backend="sdk")
        snd.sdk_request = bridge.request
        return snd

    def test_joint_path_respects_step_limit(self):
        q0, q1 = np.zeros(6), np.radians([3.0, -1.2, 0.4, 0, 0, 2.0])
        path = self.S.joint_path(q0, q1, np.radians(0.5))
        prev = q0
        for q in path:
            self.assertLessEqual(np.degrees(np.max(np.abs(q - prev))), 0.5 + 1e-9)
            prev = q
        np.testing.assert_allclose(path[-1], q1)

    def test_sdk_refuses_outside_sdk_mode_or_estop(self):
        for status, msg in (({"operation_mode": "idle", "estop": False}, "not in SDK mode"),
                            ({"operation_mode": "sdk", "estop": True}, "emergency stop")):
            snd = self.sdk_sender(self.FakeBridge(status={"ok": True, **status}))
            with self.assertRaisesRegex(RuntimeError, msg):
                snd._prepare_sdk(["left"], None, self._state(self.q))

    def test_sdk_mode_change_that_moves_the_arm_aborts(self):
        bridge = self.FakeBridge(status={"ok": True, "operation_mode": "sdk", "estop": False, "control_mode": None})
        snd = self.sdk_sender(bridge)
        calls = {"n": 0}

        def drifting(hub):
            calls["n"] += 1
            q = self.q + (np.radians([5, 0, 0, 0, 0, 0]) if calls["n"] > 1 else 0)
            return {s: {"q": list(q), "gripper": 0.0, "force": [0, 0, 0]} for s in K.ARMS}
        with self.assertRaisesRegex(RuntimeError, "moved .* by itself"):
            snd._prepare_sdk(["left"], None, drifting)
        self.assertEqual(bridge.sent[1], {"cmd": "set_mode", "mode": "joint_positions"})

    def test_sdk_move_sends_small_interpolated_targets(self):
        bridge = self.FakeBridge()
        snd = self.sdk_sender(bridge)
        target = self.q + np.radians([0, 2.0, -1.0, 0, 0, 0])
        state = {"q": list(self.q)}

        def follows(hub):
            if bridge.q is not None:
                state["q"] = bridge.q
            return {s: {"q": list(state["q"]), "gripper": 0.0, "force": [0, 0, 0]} for s in K.ARMS}
        snd._move_sdk(None, follows, {"arm": "left", "q": list(target)}, {})
        joint_cmds = [r["q"] for r in bridge.sent if r["cmd"] == "joints"]
        self.assertGreaterEqual(len(joint_cmds), 4)                      # 2° in ≤ 0.5° steps
        prev = self.q
        for q in joint_cmds[:4]:
            self.assertLessEqual(np.degrees(np.max(np.abs(np.asarray(q) - prev))), 0.5 + 1e-9)
            prev = np.asarray(q)

    def test_relative_convention_matches_end_pose(self):
        m = self.S.pose_message(self.P, "left", self.q, "relative_to_neutral")
        expect = np.linalg.inv(self.P.neutral["left"]) @ self.P.arms["left"].fk(self.q)
        p = m["pose"]["position"]
        np.testing.assert_allclose([p["x"], p["y"], p["z"]], expect[:3, 3], atol=1e-9)
        a = self.S.pose_message(self.P, "left", self.q, "arm_base")["pose"]["position"]
        np.testing.assert_allclose([a["x"] - p["x"], a["z"] - p["z"]], [0.1185, 0.181], atol=2e-3)

    def _state(self, q, force=(0, 0, 0)):
        return lambda hub: {s: {"q": list(q), "gripper": 0.0, "force": list(force)} for s in K.ARMS}

    def test_refuses_stale_or_drifted_plan(self):
        import time
        snd = self.make()
        old = {"created": time.monotonic() - 999, "steps": [], "start_q": {"left": list(self.q)}}
        with self.assertRaisesRegex(RuntimeError, "too old"):
            snd.start(old, None, self._state(self.q))
        drifted = {"created": time.monotonic(), "steps": [], "start_q": {"left": list(self.q + 0.1)}}
        with self.assertRaisesRegex(RuntimeError, "moved"):
            snd.start(drifted, None, self._state(self.q))

    def test_foreign_traffic_blocks_takeover(self):
        ws = FakeWS([{"op": "publish", "topic": "/left_arm_cartesian_controller/pose_cmd", "msg": {}}])
        with self.assertRaisesRegex(RuntimeError, "another program"):
            self.make()._preflight_ros(ws, ["left"])
        self.make()._preflight_ros(FakeWS(), ["left"])   # quiet: no error

    def test_move_returns_when_arrived(self):
        ws = FakeWS()
        self.make()._move_ros(ws, None, self._state(self.q), {"arm": "left", "q": list(self.q)}, {})
        self.assertEqual(ws.sent[0]["topic"], "/left_arm_cartesian_controller/pose_cmd")

    def test_move_times_out_if_arm_does_not_follow(self):
        target = list(self.q + np.array([0, 0.05, 0, 0, 0, 0]))
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            self.make()._move_ros(FakeWS(), None, self._state(self.q), {"arm": "left", "q": target}, {})

    def test_unexpected_force_stops(self):
        target = list(self.q + np.array([0, 0.05, 0, 0, 0, 0]))
        with self.assertRaisesRegex(RuntimeError, "unexpected force"):
            self.make()._move_ros(FakeWS(), None, self._state(self.q, force=(30, 0, 0)), {"arm": "left", "q": target},
                                  {"left": [0, 0, 0], "right": [0, 0, 0]})


@needs_urdf
class LivePlanStorageTests(unittest.TestCase):
    def test_two_arm_plans_are_not_executable(self):
        import service
        svc = service.MotionService(URDF_PATH.read_text(), "m", live=True)
        state = {s: {"q": [0.0] * 6, "gripper": 0.0} for s in K.ARMS}
        two = svc._store_plan([{"arm": "left", "q": [0] * 6, "gripper": None, "phase": "move"},
                               {"arm": "right", "q": [0] * 6, "gripper": None, "phase": "move"}], state)
        self.assertFalse(two["executable"])
        one = svc._store_plan([{"arm": "left", "q": [0] * 6, "gripper": None, "phase": "move"}], state)
        self.assertIn("plan_id", one)
        self.assertTrue(one["executable"])      # convention verified from recordings + server in live mode

    def test_nothing_is_executable_without_live_flag(self):
        import service
        svc = service.MotionService(URDF_PATH.read_text(), "m", live=False)
        state = {s: {"q": [0.0] * 6, "gripper": 0.0} for s in K.ARMS}
        one = svc._store_plan([{"arm": "left", "q": [0] * 6, "gripper": None, "phase": "move"}], state)
        self.assertFalse(one["executable"])
        self.assertIn("--live", one["why_not"])

    def test_config_convention_is_a_known_value(self):
        import sender
        self.assertIn(safety.load_config()["live"]["pose_cmd_convention"], sender.CONVENTIONS + ("unverified",))


class RecorderAnalysisTests(unittest.TestCase):
    """Synthetic recordings: the arm follows pose_cmd 80 ms later."""

    def _write(self, tmp, offset):
        import json
        import tempfile
        rows = []
        for i in range(200):
            t = i * 0.005
            z = 0.05 * np.sin(t * 3)
            cmd = {"header": {"frame_id": "left_arm_neutral"}, "pose": {"position": {"x": 0.0, "y": 0.0, "z": z + offset},
                                                                       "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}
            rows.append({"t": t, "topic": "/left_arm_cartesian_controller/pose_cmd", "msg": cmd})
            act = {"header": {"frame_id": "left_arm_neutral"}, "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.05 * np.sin((t - 0.08) * 3)},
                                                                       "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}
            rows.append({"t": t, "topic": "/left_arm/end_pose", "msg": act})
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, dir=tmp)
        f.write("\n".join(json.dumps(r) for r in rows))
        f.close()
        return Path(f.name)

    def test_detects_end_pose_convention_and_lag(self):
        import tempfile
        import record_commands as R
        with tempfile.TemporaryDirectory() as tmp:
            rep = R.analyze(self._write(tmp, 0.0))
        m = rep["arms"]["left"]["match_as_end_pose_frame"]
        self.assertLess(m["median_mm"], 0.5)
        self.assertAlmostEqual(m["lag_ms"], 80, delta=10)
        self.assertIn("CONFIRMED", rep["verdict"][0])
        self.assertIn("no pose_cmd", rep["verdict"][1])

    def test_static_commands_are_inconclusive(self):
        import json
        import tempfile
        import record_commands as R
        rows = []
        for i in range(100):
            for topic in ("/left_arm_cartesian_controller/pose_cmd", "/left_arm/end_pose"):
                rows.append({"t": i * 0.01, "topic": topic, "msg": {"header": {"frame_id": ""}, "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "static.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows))
            rep = R.analyze(p)
        self.assertIn("INCONCLUSIVE", rep["verdict"][0])

    def test_mismatch_is_not_confirmed(self):
        import tempfile
        import record_commands as R
        with tempfile.TemporaryDirectory() as tmp:
            rep = R.analyze(self._write(tmp, 0.25))   # 25 cm off in z: neither convention
        self.assertNotIn("CONFIRMED", rep["verdict"][0])


if __name__ == "__main__":
    unittest.main()
