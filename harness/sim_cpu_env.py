"""Lightweight 100% CPU tabletop dual-arm simulator & public_v2 WebSocket client.

Renders three synchronized JPEG camera streams (front, left_wrist, right_wrist)
with the robot's right arm appearing on the left side of the front image,
matching quanta_x1_pro visual conventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np
from PIL import Image, ImageDraw
from websockets.sync.client import connect

from protocol import encode_jpeg_base64


def pack_frame(value: Any) -> bytes:
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def unpack_frame(frame: bytes) -> Any:
    return msgpack.unpackb(frame, raw=False, object_hook=msgpack_numpy.decode)


@dataclass
class TabletopState:
    left_pos: np.ndarray = field(default_factory=lambda: np.array([0.29, 0.14, 0.30], dtype=np.float64))
    left_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.25, 0.0], dtype=np.float64))
    left_grip: float = 1.20
    right_pos: np.ndarray = field(default_factory=lambda: np.array([0.29, -0.14, 0.30], dtype=np.float64))
    right_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.25, 0.0], dtype=np.float64))
    right_grip: float = 1.20

    obj_pos: np.ndarray = field(default_factory=lambda: np.array([0.36, 0.10, 0.06], dtype=np.float64))
    bowl_pos: np.ndarray = field(default_factory=lambda: np.array([0.36, -0.04, 0.05], dtype=np.float64))
    held_by: str | None = None


def _project_point(p: np.ndarray, width: int, height: int) -> tuple[int, int]:
    depth = max(0.20, 0.85 - float(p[0]) * 0.65)
    u = int(width * 0.5 - (float(p[1]) / depth) * (width * 0.48))
    v = int(height * 0.78 - (float(p[2]) / depth) * (height * 0.52))
    return u, v


def render_front_camera(state: TabletopState, width: int = 320, height: int = 240) -> Image.Image:
    img = Image.new("RGB", (width, height), (236, 240, 244))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, int(height * 0.52), width, height], fill=(210, 196, 175))

    bu, bv = _project_point(state.bowl_pos, width, height)
    draw.ellipse([bu - 24, bv - 10, bu + 24, bv + 12], fill=(60, 110, 180), outline=(30, 60, 110), width=2)

    ou, ov = _project_point(state.obj_pos, width, height)
    draw.rectangle([ou - 9, ov - 9, ou + 9, ov + 9], fill=(210, 60, 55), outline=(120, 25, 25), width=2)

    for label, pos, grip, color in (
        ("R", state.right_pos, state.right_grip, (45, 55, 70)),
        ("L", state.left_pos, state.left_grip, (70, 85, 110)),
    ):
        base = np.array([0.10, -0.22 if label == "R" else 0.22, 0.42])
        su, sv = _project_point(base, width, height)
        eu, ev = _project_point(pos, width, height)
        draw.line([(su, sv), (eu, ev)], fill=color, width=6)
        jaw = int(4 + grip * 10)
        draw.rectangle([eu - jaw, ev - 4, eu - jaw + 5, ev + 10], fill=(35, 35, 40))
        draw.rectangle([eu + jaw - 5, ev - 4, eu + jaw, ev + 10], fill=(35, 35, 40))
        draw.text((eu - 4, ev - 18), label, fill=(20, 20, 20))

    return img


def render_wrist_camera(
    arm_pos: np.ndarray,
    arm_grip: float,
    state: TabletopState,
    width: int = 224,
    height: int = 224,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (218, 205, 185))
    draw = ImageDraw.Draw(img)

    for pos, radius, fill in (
        (state.bowl_pos, 28, (60, 110, 180)),
        (state.obj_pos, 14, (210, 60, 55)),
    ):
        dx = float(pos[0] - arm_pos[0])
        dy = float(pos[1] - arm_pos[1])
        dz = max(0.05, float(arm_pos[2] - pos[2]))
        u = int(width * 0.5 - (dy / dz) * 80.0)
        v = int(height * 0.5 - (dx / dz) * 80.0)
        r = max(5, int(radius / (dz * 3.0)))
        draw.ellipse([u - r, v - r, u + r, v + r], fill=fill)

    gap = int(12 + arm_grip * 28)
    draw.rectangle([width // 2 - gap - 10, height - 35, width // 2 - gap, height], fill=(40, 45, 55))
    draw.rectangle([width // 2 + gap, height - 35, width // 2 + gap + 10, height], fill=(40, 45, 55))
    return img


def step_physics(
    state: TabletopState,
    left_pos: list[float],
    left_rpy: list[float],
    left_grip: float,
    right_pos: list[float],
    right_rpy: list[float],
    right_grip: float,
) -> None:
    state.left_pos = np.asarray(left_pos, dtype=np.float64)
    state.left_rpy = np.asarray(left_rpy, dtype=np.float64)
    state.left_grip = float(left_grip)
    state.right_pos = np.asarray(right_pos, dtype=np.float64)
    state.right_rpy = np.asarray(right_rpy, dtype=np.float64)
    state.right_grip = float(right_grip)

    grasp_thresh = 0.25
    dist_l = float(np.linalg.norm(state.left_pos - state.obj_pos))
    dist_r = float(np.linalg.norm(state.right_pos - state.obj_pos))

    if state.held_by == "left":
        if state.left_grip > 0.65:
            state.held_by = None
            state.obj_pos[2] = 0.06
        else:
            state.obj_pos = state.left_pos.copy()
    elif state.held_by == "right":
        if state.right_grip > 0.65:
            state.held_by = None
            state.obj_pos[2] = 0.06
        else:
            state.obj_pos = state.right_pos.copy()
    else:
        if dist_l < 0.08 and state.left_grip < grasp_thresh:
            state.held_by = "left"
            state.obj_pos = state.left_pos.copy()
        elif dist_r < 0.08 and state.right_grip < grasp_thresh:
            state.held_by = "right"
            state.obj_pos = state.right_pos.copy()


def build_observation_payload(
    state: TabletopState,
    instruction: str,
    embodiment: str = "quanta_x1_pro/gripper-g",
) -> dict[str, Any]:
    front = render_front_camera(state)
    lw = render_wrist_camera(state.left_pos, state.left_grip, state)
    rw = render_wrist_camera(state.right_pos, state.right_grip, state)
    return {
        "embodiment": embodiment,
        "normalization": "provided",
        "instruction": instruction,
        "observation": {
            "images": {
                "front": encode_jpeg_base64(front),
                "left_wrist": encode_jpeg_base64(lw),
                "right_wrist": encode_jpeg_base64(rw),
            },
            "state": {
                "left_ee": {
                    "position": state.left_pos.tolist(),
                    "rotation_rpy": state.left_rpy.tolist(),
                    "gripper": state.left_grip,
                },
                "right_ee": {
                    "position": state.right_pos.tolist(),
                    "rotation_rpy": state.right_rpy.tolist(),
                    "gripper": state.right_grip,
                },
            },
        },
    }


def shift_chunk(actions: dict[str, Any], executed: int = 6) -> dict[str, Any]:
    chunk: dict[str, Any] = {}
    for arm, block in actions.items():
        chunk[arm] = {key: [frames[0], *frames[executed:]] for key, frames in block.items()}
    return chunk


def run_episode(
    ws_url: str,
    instruction: str,
    cycles: int = 6,
    use_rtc: bool = True,
    save_dir: Path | None = None,
) -> TabletopState:
    state = TabletopState()
    prev_actions: dict[str, Any] | None = None
    executed = 6

    with connect(ws_url, max_size=64 * 1024 * 1024, open_timeout=30) as ws:
        metadata = unpack_frame(ws.recv(timeout=30))
        assert metadata["model"] == "wall-x"
        assert metadata["model_version"] == "2610-dualhead-v2"

        for cycle in range(cycles):
            payload = build_observation_payload(state, instruction)
            if use_rtc and prev_actions is not None:
                payload["use_rtc"] = True
                payload["inference_delay"] = 4
                payload["prev_action_chunk"] = shift_chunk(prev_actions, executed=executed)

            ws.send(pack_frame(payload))
            reply = unpack_frame(ws.recv(timeout=60))
            if "error" in reply:
                raise RuntimeError(str(reply["error"]))

            actions = reply["actions"]
            n_exec = executed if use_rtc else 31
            for idx in range(n_exec):
                step_physics(
                    state,
                    actions["left_ee"]["position"][idx],
                    actions["left_ee"]["rotation_rpy"][idx],
                    actions["left_ee"]["gripper"][idx],
                    actions["right_ee"]["position"][idx],
                    actions["right_ee"]["rotation_rpy"][idx],
                    actions["right_ee"]["gripper"][idx],
                )
            prev_actions = actions
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                render_front_camera(state).save(save_dir / f"cycle_{cycle:02d}_front.jpg")

            dist_bowl = float(np.linalg.norm(state.obj_pos[:2] - state.bowl_pos[:2]))
            print(
                f"Cycle {cycle}: obj={np.round(state.obj_pos, 3).tolist()} "
                f"held_by={state.held_by} dist_to_bowl_xy={dist_bowl:.3f}"
            )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU tabletop dual-arm simulator client")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8001")
    parser.add_argument("--instruction", default="Pick up the red object and place it into the blue bowl.")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--no-rtc", action="store_true")
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    run_episode(args.ws_url, args.instruction, cycles=args.cycles, use_rtc=not args.no_rtc, save_dir=save_dir)


if __name__ == "__main__":
    main()
