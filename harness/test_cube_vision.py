"""Offline unit and HTTP integration tests for Rubik's cube vision and 3D state."""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image, ImageDraw

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

from claude_policy import ClaudeEndPosePlanner, HeuristicOfflinePlanner
from cube_state import CubeStateManager, FACES, rotate_grid_cw
from cube_vision import classify_rgb_sticker, detect_cube_cv, scan_cube_with_claude
from server import DualArmPolicyEngine, PromptController, start_console_http_server
from sim_cpu_env import TabletopState, build_observation_payload

PALETTE = {
    "W": (245, 245, 245),
    "Y": (240, 215, 20),
    "R": (205, 25, 25),
    "O": (245, 125, 20),
    "B": (20, 75, 215),
    "G": (25, 175, 55),
}


def render_cube_image(
    grid: list[list[str]],
    *,
    w: int = 640,
    h: int = 480,
    cx: float = 320.0,
    cy: float = 240.0,
    size: float = 150.0,
    angle_deg: float = 0.0,
    perspective: float = 0.0,
    stickerless: bool = False,
    bg: tuple[int, int, int] = (95, 92, 88),
    clutter: bool = False,
) -> Image.Image:
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    if clutter:
        draw.rectangle([20, 30, 120, 140], fill=(185, 65, 55))
        draw.ellipse([500, 340, 610, 440], fill=(45, 110, 195))
        draw.rectangle([40, 380, 220, 440], fill=(220, 220, 215))

    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)

    def xform(u: float, v: float) -> tuple[float, float]:
        scale_v = 1.0 + perspective * u
        lx, ly = u * size, v * size * scale_v
        return (lx * ca - ly * sa + cx, lx * sa + ly * ca + cy)

    if not stickerless:
        draw.polygon(
            [xform(-0.52, -0.52), xform(0.52, -0.52), xform(0.52, 0.52), xform(-0.52, 0.52)],
            fill=(22, 22, 24),
        )
        gap = 0.035
    else:
        gap = 0.012
    for r in range(3):
        for c in range(3):
            u0, u1 = -0.5 + c / 3.0 + gap, -0.5 + (c + 1) / 3.0 - gap
            v0, v1 = -0.5 + r / 3.0 + gap, -0.5 + (r + 1) / 3.0 - gap
            draw.polygon([xform(u0, v0), xform(u1, v0), xform(u1, v1), xform(u0, v1)], fill=PALETTE[grid[r][c]])
    return img


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestCubeVision(unittest.TestCase):
    def test_color_classifier_separates_red_orange_yellow_white_blue_green(self) -> None:
        for code, rgb in PALETTE.items():
            got, conf = classify_rgb_sticker(rgb)
            self.assertEqual(got, code, f"Expected {code} for {rgb}, got {got}")
            self.assertGreater(conf, 0.45)

    def test_detect_cube_cv_both_backends(self) -> None:
        cases = [
            ("frontal", [["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]], dict(angle_deg=0.0)),
            ("tilted_22", [["O", "B", "W"], ["G", "R", "Y"], ["W", "O", "B"]], dict(angle_deg=22.0, cx=260, cy=210)),
            ("tilted_42", [["R", "Y", "B"], ["O", "W", "G"], ["B", "R", "Y"]], dict(angle_deg=42.0, cx=380, cy=260)),
            ("perspective", [["G", "W", "Y"], ["R", "B", "O"], ["Y", "G", "W"]], dict(angle_deg=-15.0, perspective=0.28)),
            ("white_majority", [["W", "W", "W"], ["W", "W", "R"], ["W", "B", "W"]], dict(angle_deg=10.0)),
            ("all_white", [["W", "W", "W"], ["W", "W", "W"], ["W", "W", "W"]], dict(angle_deg=5.0)),
            ("stickerless", [["R", "G", "B"], ["Y", "W", "O"], ["B", "R", "G"]], dict(bg=(235, 235, 232), stickerless=True, angle_deg=12.0)),
            ("cluttered", [["Y", "O", "B"], ["G", "R", "W"], ["B", "Y", "G"]], dict(clutter=True, angle_deg=-18.0, size=130.0)),
        ]
        for label, grid, kwargs in cases:
            img = render_cube_image(grid, **kwargs)
            for use_cv2 in (True, False):
                with self.subTest(case=label, use_cv2=use_cv2):
                    res = detect_cube_cv(img, source="front", use_cv2=use_cv2)
                    self.assertTrue(res["detected"])
                    self.assertEqual(res["grid"], grid)
                    self.assertEqual(len(res["quad"]), 4)
                    self.assertEqual(res["estimated_pose"]["frame"], "camera_optical_uncalibrated")

    def test_detects_small_48px_tabletop_and_low_saturation_cube(self) -> None:
        grid = [["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]]
        # 48px cube at 640x480 (realistic wide-angle robot head camera scale)
        small_img = render_cube_image(grid, size=48.0, angle_deg=15.0, clutter=True)
        for use_cv2 in (True, False):
            res = detect_cube_cv(small_img, source="front", use_cv2=use_cv2)
            self.assertTrue(res["detected"], f"Failed small_48px with use_cv2={use_cv2}")
            self.assertEqual(res["grid"], grid)

        # Low-saturation washed-out lighting render
        washed = Image.blend(
            render_cube_image(grid, size=90.0, angle_deg=-8.0, bg=(140, 140, 140)),
            Image.new("RGB", (640, 480), (175, 175, 175)),
            alpha=0.45,
        )
        res_w = detect_cube_cv(washed, source="front", use_cv2=True)
        self.assertTrue(res_w["detected"])

    def test_configure_api_key_falls_back_on_404_model(self) -> None:
        planner = ClaudeEndPosePlanner(model="claude-sonnet-5-5")
        err404 = urllib.error.HTTPError("https://api.anthropic.com/v1/models/claude-sonnet-5-5", 404, "Not Found", {}, None)

        class FakeModelsResp:
            def read(self) -> bytes:
                return json.dumps({"data": [{"id": "claude-sonnet-4-5-20250929"}]}).encode("utf-8")

            def __enter__(self) -> "FakeModelsResp":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        with patch("claude_policy.urllib.request.urlopen", side_effect=[err404, FakeModelsResp()]):
            planner.configure_api_key("sk-ant-valid-key-123456")
        self.assertEqual(planner.api_key, "sk-ant-valid-key-123456")
        self.assertEqual(planner.model, "claude-sonnet-4-5-20250929")

    def test_no_cube_returns_undetected(self) -> None:
        img = Image.new("RGB", (640, 480), (95, 92, 88))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 30, 120, 140], fill=(185, 65, 55))
        draw.ellipse([500, 340, 610, 440], fill=(45, 110, 195))
        for use_cv2 in (True, False):
            res = detect_cube_cv(img, source="front", use_cv2=use_cv2)
            self.assertFalse(res["detected"])

    def test_scan_cube_with_claude_parses_multiface_json(self) -> None:
        img_b64 = img_to_b64(render_cube_image([["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]]))
        reply = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "detected": True,
                            "confidence": 0.92,
                            "reason": "Front and Right faces visible",
                            "primary_quad": [[0.3, 0.3], [0.6, 0.3], [0.6, 0.7], [0.3, 0.7]],
                            "faces": {
                                "F": [["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]],
                                "R": [["O", "B", "W"], ["G", "B", "Y"], ["R", "O", "G"]],
                            },
                        }
                    ),
                }
            ]
        }

        class FakeResp:
            def read(self) -> bytes:
                return json.dumps(reply).encode("utf-8")

            def __enter__(self) -> "FakeResp":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        with patch("cube_vision.urllib.request.urlopen", return_value=FakeResp()) as mock_open:
            out = scan_cube_with_claude({"Front camera": img_b64}, api_key="sk-ant-test")
            req = mock_open.call_args.args[0]
            body = json.loads(req.data.decode("utf-8"))
            self.assertNotIn("temperature", body)
            self.assertEqual(dict(req.header_items())["Anthropic-version"], "2023-06-01")
            self.assertTrue(out["detected"])
            self.assertIn("F", out["faces"])
            self.assertIn("R", out["faces"])


class TestCubeStateAndConsoleAPI(unittest.TestCase):
    def test_state_validation_rotation_and_facelet_string(self) -> None:
        mgr = CubeStateManager()
        solved = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
        for face, col in solved.items():
            mgr.capture_face(face=face, grid=[[col] * 3 for _ in range(3)], source="test")
        snap = mgr.snapshot()
        self.assertTrue(snap["validation"]["ready_to_solve"])
        self.assertEqual(snap["validation"]["known_stickers"], 54)
        self.assertEqual(snap["validation"]["facelet_string"], "".join(f * 9 for f in FACES))

        # Modify one sticker to cause an over-9 count error
        bad = mgr.set_sticker("F", 0, 0, "W")
        self.assertFalse(bad["validation"]["ready_to_solve"])
        self.assertTrue(any("More than 9" in msg for msg in bad["validation"]["issues"]))

        # Verify clockwise rotation helper
        g = [["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]]
        self.assertEqual(rotate_grid_cw(g, 4), g)
        mgr.capture_face(face="F", grid=g)
        rot_snap = mgr.rotate_face("F", 1)
        self.assertEqual(rot_snap["faces"]["F"]["grid"], rotate_grid_cw(g, 1))

    def test_console_cube_endpoints_and_passive_inference_detection(self) -> None:
        grid = [["W", "R", "G"], ["B", "Y", "O"], ["G", "W", "R"]]
        img_b64 = img_to_b64(render_cube_image(grid))
        planner = ClaudeEndPosePlanner(model="claude-sonnet-5-5")
        planner.api_key = "test-key"
        ctrl = PromptController("Scan the cube")
        srv = start_console_http_server("127.0.0.1", 0, ctrl, key_planner=planner)
        origin = f"http://127.0.0.1:{srv.server_port}"

        def call(path: str, data: dict[str, Any] | None = None, origin_hdr: str | None = origin) -> dict[str, Any]:
            headers = {"Content-Type": "application/json"}
            if origin_hdr:
                headers["Origin"] = origin_hdr
            req = urllib.request.Request(
                origin + path,
                data=(json.dumps(data).encode("utf-8") if data is not None else None),
                headers=headers if data is not None else {},
                method="POST" if data is not None else "GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return json.load(resp)

        try:
            st0 = call("/api/cube/state")
            self.assertTrue(st0["claude_scan_ready"])
            self.assertFalse(st0["live"]["detected"])

            det_snap = call("/api/cube/detect", {"image_b64": img_b64, "source": "/camera_head_front/color/image_raw/compressed"})
            self.assertTrue(det_snap["live"]["detected"])
            self.assertEqual(det_snap["live"]["grid"], grid)

            cap = call("/api/cube/capture", {"face": "AUTO"})
            self.assertEqual(cap["captured_face"], "F")
            self.assertEqual(cap["state"]["faces"]["F"]["grid"], grid)

            rot = call("/api/cube/rotate", {"face": "F", "turns": 1})
            self.assertEqual(rot["state"]["faces"]["F"]["grid"], rotate_grid_cw(grid, 1))

            stk = call("/api/cube/sticker", {"face": "F", "row": 0, "col": 0, "color": "B"})
            self.assertEqual(stk["faces"]["F"]["grid"][0][0] if "faces" in stk else stk["state"]["faces"]["F"]["grid"][0][0], "B")

            # Enforce local origin check on /api/cube/claude-scan when key_planner is active
            with self.assertRaises(urllib.error.HTTPError) as forbidden:
                call("/api/cube/claude-scan", {"images_b64": {"front": img_b64}}, origin_hdr="http://evil.example")
            self.assertEqual(forbidden.exception.code, 403)
            forbidden.exception.close()

            with patch("server.scan_cube_with_claude", return_value={
                "detected": True,
                "confidence": 0.91,
                "reason": "Two faces visible",
                "primary_quad": None,
                "faces": {"R": [["R"] * 3 for _ in range(3)]},
            }):
                scanned = call("/api/cube/claude-scan", {"images_b64": {"front": img_b64}, "auto_capture": True})
                self.assertEqual(scanned["state"]["faces"]["R"]["grid"][1][1], "R")

            reset = call("/api/cube/reset", {})
            self.assertEqual(reset["state"]["validation"]["scanned_faces"], 0)
        finally:
            srv.shutdown()
            srv.server_close()

        # Verify passive cube detection when inference frames arrive on port 8001
        ctrl2 = PromptController("Hold")
        engine = DualArmPolicyEngine(planner=HeuristicOfflinePlanner(), prompt_ctrl=ctrl2)
        payload = build_observation_payload(TabletopState(), "Hold")
        payload["observation"]["images"]["front"] = img_b64
        engine.handle_payload(payload)
        live = ctrl2.cube_state.snapshot()["live"]
        self.assertTrue(live["detected"])
        self.assertEqual(live["source"], "inference:front")
        self.assertEqual(live["grid"], grid)


if __name__ == "__main__":
    unittest.main()
