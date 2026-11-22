"""Run the Claude EndPose planner against rosbridge and the local desktop SDK."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np
from websockets.sync.client import connect

from claude_policy import ClaudeEndPosePlanner
from cube_depth import DEPTH_TOPICS, FRONT_COLOR, localize_cube
from protocol import validate_request


CAMERA_TOPICS = {
    FRONT_COLOR: "front",
    "/camera3/usb_cam3/image_raw/image_compressed": "left_wrist",
    "/camera1/usb_cam1/image_raw/image_compressed": "right_wrist",
}
EMBODIMENT = "quanta_x1_pro/gripper-g"
OPERATOR_TIMEOUT = 6.0
TASK_CONTEXT = (
    "Use only the left arm and omit right_ee from every waypoint. "
    "Reported poses are gripper-base poses in the robot base_link frame. "
    "Use last_sdk_step measured feedback to tell whether a command moved the arm. "
    "A fresh local_perception.cube_depth cube_top_center_m measures the top face, not a gripper target. "
    "Use its base_link x/y for alignment and the nearby table_z_m to reason about vertical clearance. "
    "This measurement does not check arm collisions or assure a clear path. "
    "Keep approach and gripper changes in separate waypoints. "
    "Return retry if the target, approach or tabletop clearance is uncertain. "
)


def _stamp(message: dict) -> int:
    stamp = message.get("header", {}).get("stamp", {})
    return int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nanosec", 0))


def read_camera_snapshot(
    url: str, stop: threading.Event, front_mount: np.ndarray | None = None, static_cache: dict | None = None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Start fresh subscriptions so queued frames cannot carry over from planning."""
    images = {}
    depth_messages = {}
    static = static_cache if static_cache is not None else {}
    rgb_frames = deque(maxlen=8)
    depth_frames = deque(maxlen=8)
    tested_rgb = 0
    depth_failure = "No synchronized front RGB and depth with matching calibration."
    deadline = time.monotonic() + 5
    with connect(url, open_timeout=3, close_timeout=0.5, max_size=4 * 1024 * 1024, proxy=None) as socket:
        topics = [*CAMERA_TOPICS, *(DEPTH_TOPICS if front_mount is not None else ())]
        for topic in topics:
            options = {"queue_length": 1, "throttle_rate": 300}
            if topic == "/tf_static":
                options = {"qos": {"durability": "transient_local", "reliability": "reliable", "depth": 10}}
            socket.send(json.dumps({"op": "subscribe", "topic": topic, **options}))
        while True:
            if stop.is_set():
                raise RuntimeError("Direct Claude control stopped.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if len(images) == len(CAMERA_TOPICS):
                    return images, {"status": "unavailable", "reason": depth_failure}
                raise RuntimeError("Timed out waiting for all three rosbridge cameras.")
            try:
                update = json.loads(socket.recv(timeout=min(0.5, remaining)))
            except TimeoutError:
                continue
            if update.get("op") != "publish":
                continue
            message = update.get("msg") or {}
            topic = update.get("topic")
            camera = CAMERA_TOPICS.get(topic)
            if camera is not None:
                data = message.get("data")
                if isinstance(data, str) and data and "jpeg" in str(message.get("format", "")).lower():
                    images[camera] = data
                    if camera == "front":
                        rgb_frames.append(message)
            elif front_mount is not None and topic in DEPTH_TOPICS:
                kind = DEPTH_TOPICS[topic]
                if kind == "static":
                    static.update({entry["child_frame_id"]: entry for entry in message.get("transforms", [])})
                elif kind == "depth":
                    depth_frames.append(message)
                else:
                    depth_messages[kind] = message

            if len(images) != len(CAMERA_TOPICS):
                continue
            if front_mount is None:
                return images, None
            if not {"color_info", "depth_info"} <= depth_messages.keys() or not static:
                continue
            pairs = [
                (abs(_stamp(rgb) - _stamp(depth)), rgb, depth)
                for rgb in rgb_frames for depth in depth_frames if _stamp(rgb) and _stamp(depth)
            ]
            if not pairs:
                continue
            pairs = [pair for pair in pairs if pair[0] <= 120_000_000 and _stamp(pair[1]) > tested_rgb]
            if not pairs:
                continue
            _, rgb, depth = min(pairs, key=lambda pair: pair[0])
            try:
                measurement = localize_cube(rgb, depth, depth_messages["color_info"],
                                            depth_messages["depth_info"], static, front_mount)
            except (ValueError, KeyError, OSError) as exc:
                depth_failure = str(exc)
                if not depth_failure.startswith("Missing fixed transform"):
                    tested_rgb = _stamp(rgb)
                continue
            images["front"] = rgb["data"]
            return images, {"status": "measured", **measurement}


class DirectClaudeControl:
    """Own the local motion session and submit at most one bounded target per observation."""

    def __init__(
        self, jogger: Any, key_planner: ClaudeEndPosePlanner, status: Any, rosbridge: str,
        *, front_mount: np.ndarray | None = None,
    ) -> None:
        self.jogger = jogger
        self.key_planner = key_planner
        self.status = status
        self.rosbridge = rosbridge
        self.front_mount = front_mount
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._operator_expires_at = 0.0

    def start(self, task: str, use_cube_depth: bool = True) -> None:
        if not isinstance(task, str) or not task.strip() or len(task) > 1500:
            raise ValueError("Enter a task of 1–1500 characters.")
        if not self.key_planner.api_key:
            raise RuntimeError("Verify the Anthropic key before starting direct control.")
        with self._lock:
            if self._stop is not None:
                raise RuntimeError("Direct Claude control is already active.")
            stop = threading.Event()
            self.jogger.reserve_direct(stop)
            self._stop = stop
            self._operator_expires_at = time.monotonic() + OPERATOR_TIMEOUT
            self.status.set_prompt(task)
            self.status.set_connected(True)
            worker = threading.Thread(target=self._run, args=(task.strip(), stop, use_cube_depth), daemon=True)
            try:
                worker.start()
            except RuntimeError:
                self._stop = None
                self.jogger.finish_direct(stop)
                self.status.finish_direct("error", "Could not start the direct control worker.")
                raise

    def stop(self) -> None:
        with self._lock:
            if self._stop is not None:
                self._stop.set()

    def heartbeat(self) -> None:
        with self._lock:
            if self._stop is not None and not self._stop.is_set() and time.monotonic() < self._operator_expires_at:
                self._operator_expires_at = time.monotonic() + OPERATOR_TIMEOUT

    def _run(self, task: str, stop: threading.Event, use_cube_depth: bool) -> None:
        deadline = time.monotonic() + 150
        outcome, reason = "stopped", "Stopped issuing SDK targets."
        origin = None
        last_measurement = None
        static_cache = {}
        previous_left = None
        previous_action = None
        is_cube_task = use_cube_depth and "cube" in task.casefold()
        try:
            planner = ClaudeEndPosePlanner(
                api_key=self.key_planner.api_key, base_url=self.key_planner.base_url,
                model=self.key_planner.model, timeout_s=14,
            )
            for index in range(1, 101):
                if stop.is_set():
                    break
                if time.monotonic() >= self._operator_expires_at:
                    outcome, reason = "stopped", "The operator console disconnected."
                    break
                if time.monotonic() >= deadline:
                    outcome, reason = "limit", "Reached the direct control time limit."
                    break
                started = time.monotonic()
                measure_now = is_cube_task and (index == 1 or (index - 1) % planner.plan_every == 0)
                front_mount = self.front_mount if measure_now else None
                images, measured = read_camera_snapshot(self.rosbridge, stop, front_mount, static_cache)
                sampled_at = started
                if index == 1 and front_mount is not None and measured["status"] != "measured":
                    sampled_at = time.monotonic()
                    images, measured = read_camera_snapshot(self.rosbridge, stop, front_mount, static_cache)
                    if measured["status"] != "measured":
                        raise RuntimeError("Cube depth unavailable. " + measured["reason"])
                if stop.is_set():
                    break
                states = self.jogger.observe_direct(stop)
                observation = validate_request({
                    "embodiment": EMBODIMENT, "normalization": "provided", "instruction": TASK_CONTEXT + task,
                    "observation": {"images": images, "state": states},
                })
                if measured and measured["status"] == "measured":
                    last_measurement = {"observation": index, **measured}
                context = {}
                if measured is not None or last_measurement is not None:
                    context["cube_depth"] = measured if measured and measured["status"] == "measured" else {
                        "status": "no fresh depth measurement",
                        "current_measurement": measured,
                        "last_seen": last_measurement,
                    }
                if previous_left is not None:
                    displacement = [
                        round(1000 * (actual - previous), 2)
                        for actual, previous in zip(observation.left_ee.position, previous_left.position)
                    ]
                    context["last_sdk_step"] = {
                        "command": previous_action,
                        "measured_left_delta_xyz_mm": displacement,
                        "measured_left_gripper_delta_rad": round(observation.left_ee.gripper - previous_left.gripper, 4),
                    }
                observation.perception = context or None
                if origin is None:
                    origin = list(observation.left_ee.position)
                decision = planner.decide(observation)
                if stop.is_set():
                    break
                if time.monotonic() >= self._operator_expires_at:
                    outcome, reason = "stopped", "The operator console disconnected."
                    break
                action = "hold"
                if decision.status not in ("completed", "retry"):
                    expiry = min(deadline, sampled_at + 20, self._operator_expires_at)
                    action = self.jogger.apply_direct(observation, decision, origin, stop, expiry)
                self.status.record_step(
                    chunk_index=index, instruction=task, decision=decision,
                    left_ee=asdict(observation.left_ee), right_ee=asdict(observation.right_ee),
                    images_b64=observation.images_b64, latency_ms=(time.monotonic() - started) * 1000,
                    motion_feedback={"latest_command": action, "previous_step": context.get("last_sdk_step"),
                                     "cube_depth": context.get("cube_depth")},
                )
                if decision.status in ("completed", "retry"):
                    outcome = decision.status
                    reason = decision.reason or ("Task completed." if outcome == "completed" else "Claude was unsure.")
                    break
                previous_left = observation.left_ee
                previous_action = action
                if stop.wait(0.65):
                    break
            else:
                outcome, reason = "limit", "Reached the direct control observation limit."
        except Exception as exc:
            if not stop.is_set():
                outcome, reason = "error", str(exc) or type(exc).__name__
        finally:
            with self._lock:
                self.jogger.finish_direct(stop)
                self.status.finish_direct(outcome, reason)
                self._stop = None
