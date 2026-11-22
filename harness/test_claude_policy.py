"""Offline unit and loopback tests for Claude EndPose autonomous control."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from websockets.asyncio.server import serve

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "harness"
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from claude_policy import (
    AuthError,
    ClaudeEndPosePlanner,
    HeuristicOfflinePlanner,
    extract_json_object,
)
from harness import VisionAgent
from protocol import ACTION_HORIZON, ProtocolError, validate_request
from server import DualArmPolicyEngine, PromptController, serve_connection, start_console_http_server
from sim_cpu_env import TabletopState, build_observation_payload, run_episode, shift_chunk, step_physics
from trajectory import WorkspaceLimits, matrix_to_rpy, rpy_to_matrix, synthesize_arm_chunk
import launch as launcher


class TestProtocolAndTrajectory(unittest.TestCase):
    def test_rpy_matrix_roundtrip(self) -> None:
        rpy = [0.31, -0.42, 0.77]
        mat = rpy_to_matrix(rpy)
        self.assertAlmostEqual(float(np.linalg.det(mat)), 1.0, places=6)
        recovered = matrix_to_rpy(mat)
        for a, b in zip(rpy, recovered):
            self.assertAlmostEqual(a, b, places=5)

    def test_accepts_numpy_arrays_and_scalars_from_ros_client(self) -> None:
        state = TabletopState()
        payload = build_observation_payload(state, "Pick up the spoon")
        for arm in ("left_ee", "right_ee"):
            payload["observation"]["state"][arm]["position"] = np.array([0.3, 0.1, 0.2], dtype=np.float32)
            payload["observation"]["state"][arm]["rotation_rpy"] = np.array([0.0, 0.2, 0.0], dtype=np.float32)
            payload["observation"]["state"][arm]["gripper"] = np.float32(0.9)
        req = validate_request(payload)
        self.assertEqual(req.embodiment, "quanta_x1_pro/gripper-g")
        self.assertAlmostEqual(req.left_ee.gripper, 0.9, places=5)

    def test_unknown_embodiment_rejected(self) -> None:
        payload = build_observation_payload(TabletopState(), "Pick up")
        payload["embodiment"] = "quanta_x1_pro/gripper-G"
        with self.assertRaises(ProtocolError) as ctx:
            validate_request(payload)
        self.assertEqual(ctx.exception.code, "unknown_embodiment")

    def test_zero_delta_never_snaps_outside_box(self) -> None:
        state = TabletopState()
        payload = build_observation_payload(state, "Hold")
        payload["observation"]["state"]["left_ee"]["position"] = [0.30, 0.10, -0.55]
        req = validate_request(payload)
        chunk = synthesize_arm_chunk(
            current=req.left_ee,
            target_pos=req.left_ee.position,
            target_rpy=req.left_ee.rotation_rpy,
            target_gripper=req.left_ee.gripper,
            gripper_range=req.gripper_range,
            limits=WorkspaceLimits(),
        )
        self.assertAlmostEqual(chunk["position"][-1][2], -0.55, places=5)
        self.assertEqual(chunk["position"][30], chunk["position"][31])

    def test_rtc_prefix_anchoring(self) -> None:
        req = validate_request(build_observation_payload(TabletopState(), "Lift"))
        first = synthesize_arm_chunk(
            current=req.left_ee,
            target_pos=[0.35, 0.12, 0.25],
            target_rpy=[0.0, 0.3, 0.1],
            target_gripper=0.2,
            gripper_range=req.gripper_range,
            limits=WorkspaceLimits(),
        )
        shifted = shift_chunk({"left_ee": first, "right_ee": first}, executed=6)
        second = synthesize_arm_chunk(
            current=req.left_ee,
            target_pos=[0.36, 0.08, 0.27],
            target_rpy=[0.0, 0.3, 0.1],
            target_gripper=0.2,
            gripper_range=req.gripper_range,
            limits=WorkspaceLimits(),
            rtc_prefix_arm=shifted["left_ee"],
            inference_delay=4,
        )
        self.assertEqual(len(second["position"]), ACTION_HORIZON)
        self.assertEqual(second["position"][0], first["position"][6])
        self.assertEqual(second["position"][3], first["position"][9])


class TestClaudePlannerAndHarness(unittest.TestCase):
    def test_claude_endpose_planner_and_completion_hold(self) -> None:
        req = validate_request(build_observation_payload(TabletopState(), "Approach red block"))
        planner = ClaudeEndPosePlanner(api_key="test-key", model="claude-sonnet-5-5", plan_every=1)
        body = planner.build_request_body(req)
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertNotIn("temperature", body)
        img_blocks = [b for b in body["messages"][0]["content"] if b["type"] == "image"]
        self.assertEqual(len(img_blocks), 3)

        replies = [
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "stage": "approach",
                                "status": "in_progress",
                                "reason": "Moving left arm forward",
                                "left_ee": {"delta_position": [0.05, 0.0, -0.05], "delta_rpy": [0, 0, 0], "gripper": 1.2},
                                "right_ee": {"delta_position": [0, 0, 0], "delta_rpy": [0, 0, 0], "gripper": 1.2},
                            }
                        ),
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "stage": "done",
                                "status": "completed",
                                "reason": "Object is inside bowl",
                                "left_ee": {"delta_position": [0.09, 0.0, 0.0], "delta_rpy": [0, 0, 0], "gripper": 1.2},
                                "right_ee": {"delta_position": [0, 0, 0], "delta_rpy": [0, 0, 0], "gripper": 1.2},
                            }
                        ),
                    }
                ],
            },
        ]

        class FakeResp:
            def __init__(self, obj: dict[str, Any]) -> None:
                self.raw = json.dumps(obj).encode("utf-8")

            def read(self) -> bytes:
                return self.raw

            def __enter__(self) -> "FakeResp":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        captured_headers = []

        def fake_urlopen(request: Any, timeout: float = 10.0) -> FakeResp:
            captured_headers.append(dict(request.header_items()))
            return FakeResp(replies.pop(0))

        with patch("claude_policy.urllib.request.urlopen", side_effect=fake_urlopen):
            d1 = planner.decide(req)
            self.assertEqual(d1.stage, "approach")
            self.assertAlmostEqual(d1.left_target_pos[0], req.left_ee.position[0] + 0.05, places=6)
            d2 = planner.decide(req)
            self.assertEqual(d2.status, "completed")
            # Completed state must hold current pose safely
            self.assertEqual(d2.left_target_pos, req.left_ee.position)
            # Subsequent call must hold without hitting API again
            d3 = planner.decide(req)
            self.assertEqual(d3.status, "completed")
            self.assertEqual(len(captured_headers), 2)
            self.assertEqual(captured_headers[0]["X-api-key"], "test-key")
            self.assertEqual(captured_headers[0]["Anthropic-version"], "2023-06-01")

    def test_waypoints_follow_measured_feedback_in_simulation(self) -> None:
        state = TabletopState()
        planner = ClaudeEndPosePlanner(api_key="test-key")
        engine = DualArmPolicyEngine(planner)
        req = build_observation_payload(state, "Move the left arm forward and up")
        targets = ([0.34, 0.14, 0.30], [0.34, 0.14, 0.35])
        waypoints = [
            {"left_ee": {"position": target, "rotation_rpy": [0.0, 0.25, 0.0], "gripper": 1.2}}
            for target in targets
        ]
        replies = [
            {"status": "in_progress", "stage": "approach", "waypoints": waypoints},
            {"status": "completed", "stage": "done", "waypoints": []},
        ]
        with patch.object(planner, "_post_messages", side_effect=[
            {"content": [{"type": "text", "text": json.dumps(reply)}]} for reply in replies
        ]) as model:
            first = engine.handle_payload(req)
            stalled = engine.handle_payload(req)
            self.assertEqual(first, stalled)
            self.assertEqual(engine.prompt_ctrl.snapshot()["waypoint_index"], 1)
            for index, target in enumerate(targets, 1):
                if index == 1:
                    actions = first["actions"]
                else:
                    actions = engine.handle_payload(build_observation_payload(state, req["instruction"]))["actions"]
                self.assertEqual(engine.prompt_ctrl.snapshot()["waypoint_index"], index)
                self.assertEqual(engine.prompt_ctrl.snapshot()["waypoint_count"], 2)
                np.testing.assert_allclose(actions["left_ee"]["position"][-1], target)
                np.testing.assert_allclose(actions["right_ee"]["position"][-1], state.right_pos)
                for left_p, left_r, left_g, right_p, right_r, right_g in zip(
                    actions["left_ee"]["position"], actions["left_ee"]["rotation_rpy"], actions["left_ee"]["gripper"],
                    actions["right_ee"]["position"], actions["right_ee"]["rotation_rpy"], actions["right_ee"]["gripper"],
                ):
                    step_physics(state, left_p, left_r, left_g, right_p, right_r, right_g)
            self.assertEqual(model.call_count, 1)
            final = engine.handle_payload(build_observation_payload(state, req["instruction"]))
            summary = json.loads(model.call_args.args[0]["messages"][0]["content"][0]["text"])
            self.assertEqual(summary["recent_history"][-1]["reached_waypoints"], 2)
            self.assertEqual(engine.prompt_ctrl.snapshot()["status"], "completed")
            self.assertEqual(final["actions"]["left_ee"]["position"][-1], list(targets[-1]))

    def test_invalid_waypoint_does_not_execute_a_partial_plan(self) -> None:
        state = TabletopState()
        planner = ClaudeEndPosePlanner(api_key="test-key")
        engine = DualArmPolicyEngine(planner)
        valid = {"left_ee": {"position": [0.35, 0.14, 0.30], "rotation_rpy": [0, 0.25, 0], "gripper": 1.2}}
        invalid = {"left_ee": {**valid["left_ee"], "gripper": float("nan")}}
        reply = {"status": "in_progress", "waypoints": [valid, invalid]}
        with patch.object(planner, "_post_messages", return_value={
            "content": [{"type": "text", "text": json.dumps(reply)}]
        }):
            actions = engine.handle_payload(build_observation_payload(state, "Move forward"))["actions"]
        self.assertEqual(engine.prompt_ctrl.snapshot()["stage"], "error_hold")
        self.assertEqual(actions["left_ee"]["position"][-1], state.left_pos.tolist())

    def test_plan_arriving_after_prompt_change_holds_instead(self) -> None:
        state = TabletopState()
        planner = ClaudeEndPosePlanner(api_key="test-key")
        engine = DualArmPolicyEngine(planner)

        def new_prompt_while_planning(_body: dict[str, Any]) -> dict[str, Any]:
            engine.prompt_ctrl.set_prompt("Wait")
            reply = {"status": "in_progress", "waypoints": [
                {"left_ee": {"position": [0.35, 0.14, 0.30], "rotation_rpy": [0, 0.25, 0], "gripper": 1.2}}
            ]}
            return {"content": [{"type": "text", "text": json.dumps(reply)}]}

        with patch.object(planner, "_post_messages", side_effect=new_prompt_while_planning):
            actions = engine.handle_payload(build_observation_payload(state, "Move forward"))["actions"]
        self.assertEqual(engine.prompt_ctrl.snapshot()["effective_instruction"], "Wait")
        self.assertEqual(actions["left_ee"]["position"][-1], state.left_pos.tolist())

    def test_claude_auth_error_raises_immediately(self) -> None:
        req = validate_request(build_observation_payload(TabletopState(), "Pick"))
        planner = ClaudeEndPosePlanner(api_key="bad-key")
        err = urllib.error.HTTPError("https://api.anthropic.com/v1/messages", 401, "Unauthorized", {}, None)
        with patch("claude_policy.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(AuthError):
                planner.decide(req)

    def test_local_key_setup_verifies_access_and_enables_prompt(self) -> None:
        import urllib.request
        from unittest.mock import MagicMock

        planner = ClaudeEndPosePlanner(model="claude-sonnet-5-5")
        planner.api_key = ""
        controller = PromptController()
        server = start_console_http_server("127.0.0.1", 0, controller, key_planner=planner)
        origin = f"http://127.0.0.1:{server.server_port}"
        urlopen = urllib.request.urlopen

        def post(path: str, data: dict[str, str]) -> dict:
            request = urllib.request.Request(
                origin + path, data=json.dumps(data).encode(),
                headers={"Origin": origin, "Content-Type": "application/json"}, method="POST",
            )
            with urlopen(request, timeout=2) as response:
                return json.load(response)

        try:
            with self.assertRaises(urllib.error.HTTPError) as blocked:
                post("/api/prompt", {"prompt": "Move the arm"})
            self.assertEqual(blocked.exception.code, 409)
            blocked.exception.close()
            with patch("claude_policy.urllib.request.urlopen", return_value=MagicMock()) as cloud:
                result = post("/api/anthropic-key", {"key": "test-key"})
                request = cloud.call_args.args[0]
                self.assertTrue(request.full_url.endswith("/v1/models/claude-sonnet-5-5"))
                self.assertEqual(request.get_method(), "GET")
            self.assertEqual(result, {"key_ready": True})
            self.assertTrue(post("/api/prompt", {"prompt": "Move the arm"})["ok"])
            with urlopen(origin + "/api/status", timeout=2) as response:
                status = json.load(response)
            self.assertTrue(status["key_ready"])
            self.assertEqual(status["active_prompt"], "Move the arm")
            self.assertNotIn("test-key", json.dumps(status))
        finally:
            server.shutdown()
            server.server_close()

    def test_vision_agent_anthropic_mode_in_harness(self) -> None:
        obs = build_observation_payload(TabletopState(), "Place spoon in bowl")
        agent = VisionAgent(key="test-anthropic-key", provider="anthropic")

        class FakeResp:
            def read(self) -> bytes:
                return json.dumps(
                    {
                        "content": [
                            {"type": "text", "text": json.dumps({"instruction": "Grasp the spoon", "reason": "Visible"})}
                        ]
                    }
                ).encode("utf-8")

            def __enter__(self) -> "FakeResp":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        with patch("harness.urllib.request.urlopen", return_value=FakeResp()) as mock_open:
            out = agent.decide(obs, "Place spoon in bowl", "", [])
            self.assertEqual(out, "Grasp the spoon")
            req_obj = mock_open.call_args.args[0]
            self.assertEqual(req_obj.full_url, "https://api.anthropic.com/v1/messages")

    def test_validate_task_script_passes_on_prompt_task_json(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "validate_task",
            REPO_ROOT / ".claude" / "skills" / "robot-task" / "scripts" / "validate_task.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        msgs = mod.check(REPO_ROOT / "tasks" / "local_claude_prompt_benchmark.json")
        errs = [m for m in msgs if m.startswith("ERROR")]
        self.assertEqual(errs, [], msg=f"Unexpected task validation errors: {msgs}")

    def test_prompt_controller_and_http_console(self) -> None:
        import urllib.request

        ctrl = PromptController("Initial prompt")
        srv = start_console_http_server("127.0.0.1", 0, ctrl)
        port = srv.server_address[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as page_resp:
                html_text = page_resp.read().decode("utf-8")
                self.assertIn('id="code-version"', html_text)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as status_resp:
                status_body = json.loads(status_resp.read().decode("utf-8"))
                self.assertRegex(status_body["code_version"], r"^v\d+")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/prompt",
                data=json.dumps({"prompt": "Stack red block on blue block"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(body["ok"])
            eff, _ = ctrl.get_effective_instruction("WebUI fallback")
            self.assertEqual(eff, "Stack red block on blue block")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_extract_json_object(self) -> None:
        parsed = extract_json_object('```json\n{"stage": "grasp", "status": "in_progress"}\n```')
        self.assertEqual(parsed["stage"], "grasp")

    def test_build_task_dict_no_secrets(self) -> None:
        d = launcher.build_task_dict("ws://10.19.20.98:8001", "Pick up cup", gripper="h")
        self.assertEqual(d["model"]["infer_params"]["async_params"]["embodiment"], "quanta_x1_pro/gripper-h")
        self.assertEqual(d["data"][0]["instruction"], "Pick up cup")


class TestWebSocketLoopback(unittest.TestCase):
    def test_offline_loopback_episode_and_prompt_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "chunks.jsonl"
            ctrl = PromptController("Pick up the red object and place it into the blue bowl.")
            engine = DualArmPolicyEngine(
                planner=HeuristicOfflinePlanner(),
                log_path=log_file,
                prompt_ctrl=ctrl,
            )
            port_holder: dict[str, int] = {}
            ready = threading.Event()
            stop_holder: dict[str, Any] = {}

            def run_bg() -> None:
                async def runner() -> None:
                    stop_evt = asyncio.Event()
                    stop_holder["loop"] = asyncio.get_running_loop()
                    stop_holder["evt"] = stop_evt
                    async with serve(
                        lambda c: serve_connection(c, engine),
                        "127.0.0.1",
                        0,
                        ping_interval=None,
                        compression=None,
                        max_size=64 * 1024 * 1024,
                    ) as srv:
                        port_holder["port"] = srv.sockets[0].getsockname()[1]
                        ready.set()
                        await stop_evt.wait()

                asyncio.run(runner())

            t = threading.Thread(target=run_bg, daemon=True)
            t.start()
            self.assertTrue(ready.wait(timeout=5))
            ws_url = f"ws://127.0.0.1:{port_holder['port']}"

            try:
                final_state = run_episode(
                    ws_url=ws_url,
                    instruction="Ignored WebUI text",
                    cycles=9,
                    use_rtc=False,
                    save_dir=Path(tmp) / "frames",
                )
                dist_to_bowl = float(np.linalg.norm(final_state.obj_pos[:2] - final_state.bowl_pos[:2]))
                self.assertLess(dist_to_bowl, 0.03)
                lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
                self.assertEqual(len(lines), 9)
                self.assertTrue(all(x["instruction"] == "Pick up the red object and place it into the blue bowl." for x in lines))
            finally:
                stop_holder["loop"].call_soon_threadsafe(stop_holder["evt"].set)
                t.join(timeout=3)


class TestStopReasonDiagnostics(unittest.TestCase):
    def test_stop_reason_refusal_and_max_tokens_diagnostics(self) -> None:
        req = validate_request(build_observation_payload(TabletopState(), "Grasp the cube"))
        planner = ClaudeEndPosePlanner(api_key="test-key", model="claude-sonnet-5-5", plan_every=1)
        self.assertGreaterEqual(planner.build_request_body(req)["max_tokens"], 2048)

        with patch.object(planner, "_post_messages", return_value={
            "stop_reason": "refusal",
            "content": [{"type": "text", "text": "Cannot command arm near person"}],
        }):
            with self.assertRaisesRegex(RuntimeError, "stop_reason=refusal.*Cannot command arm"):
                planner.decide(req)

        with patch.object(planner, "_post_messages", return_value={
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": '{"stage": "approach", "status":'}],
        }):
            with self.assertRaisesRegex(RuntimeError, "stop_reason=max_tokens"):
                planner.decide(req)

    def test_compact_jpeg_downscales_oversized_frames(self) -> None:
        import base64
        import io
        from PIL import Image

        big_buf = io.BytesIO()
        Image.new("RGB", (1280, 720), (40, 120, 200)).save(big_buf, format="JPEG")
        big_b64 = base64.b64encode(big_buf.getvalue()).decode("ascii")
        compacted = ClaudeEndPosePlanner._compact_jpeg_b64(big_b64)
        with Image.open(io.BytesIO(base64.b64decode(compacted))) as out_img:
            self.assertEqual(out_img.size, (768, 432))

        small_buf = io.BytesIO()
        Image.new("RGB", (640, 480), (40, 120, 200)).save(small_buf, format="JPEG")
        small_b64 = base64.b64encode(small_buf.getvalue()).decode("ascii")
        self.assertEqual(ClaudeEndPosePlanner._compact_jpeg_b64(small_b64), small_b64)


if __name__ == "__main__":
    unittest.main()
