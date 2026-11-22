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
BASE_TASK_CONTEXT = (
    "Use only the left arm and omit right_ee from every waypoint. "
    "Reported poses are gripper-base poses in the robot base_link frame. "
    "Sequential waypoints in your waypoints list (up to 4, up to 50 mm and 20 deg per waypoint, up to 20 cm total per call) execute continuously in one unbroken trajectory before fresh camera frames are read. "
    "Maximize speed by chaining 3 to 4 waypoints whenever the path is clear (for example [approach_1, approach_2, align_above] or [descend, grasp, lift]), and only stop early right before fine alignment or grasp verification. "
    "Use last_sdk_step and since_last_plan measured displacement feedback together with the updated cameras to verify arm motion. "
    "Keep approach and gripper changes in separate waypoints. "
)
CUBE_DEPTH_CONTEXT = (
    "local_perception.cube_depth reports cube_top_center_m (the cube's top face, not a gripper target) and nearby table_z_m in base_link. "
    "Use its x/y for alignment and table_z_m to maintain vertical clearance. "
    "If last_seen is present while the moving arm briefly blocks the front view, rely on last_seen for the stationary tabletop cube and chain waypoints directly toward it. "
    "If cube_depth reports status='unavailable' and last_seen is null while left_wrist sees the cube, do not abort: chain 4 to 5 cm waypoints along the clear approach direction and use 2 to 3 cm steps only for final centering right above the cube. "
    "Return retry only if the target is held, missing from all views, or unsafe. "
)
RGB_ONLY_CONTEXT = (
    "RGB-only mode (no depth measurement is attached): do not wait for cube_depth or table_z_m. "
    "Use the front and left_wrist RGB views to chain 2 to 4 waypoints (up to 5 cm each) along clear approach paths while keeping safe clearance above the table, then center finely over the target before chaining [descend, grasp, lift]. "
    "Return retry only if the target object is missing, held or unsafe, or all camera views are blocked. "
)
TASK_CONTEXT = BASE_TASK_CONTEXT + CUBE_DEPTH_CONTEXT


def build_task_instruction(task: str, *, use_cube_depth: bool) -> str:
    extra = CUBE_DEPTH_CONTEXT if use_cube_depth else RGB_ONLY_CONTEXT
    return BASE_TASK_CONTEXT + extra + task


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
    depth_grace_deadline: float | None = None
    with connect(url, open_timeout=3, close_timeout=0.5, max_size=4 * 1024 * 1024, proxy=None) as socket:
        topics = [*CAMERA_TOPICS, *(DEPTH_TOPICS if front_mount is not None else ())]
        for topic in topics:
            if topic == "/tf_static" and static:
                continue
            options = {"queue_length": 1, "throttle_rate": 40}
            if topic == "/tf_static":
                options = {"qos": {"durability": "transient_local", "reliability": "reliable", "depth": 10}}
            socket.send(json.dumps({"op": "subscribe", "topic": topic, **options}))
        while True:
            if stop.is_set():
                raise RuntimeError("Direct Claude control stopped.")
            now = time.monotonic()
            if depth_grace_deadline is not None and now >= depth_grace_deadline and len(images) == len(CAMERA_TOPICS):
                return images, {"status": "unavailable", "reason": depth_failure}
            remaining = deadline - now
            if remaining <= 0:
                if len(images) == len(CAMERA_TOPICS):
                    return images, {"status": "unavailable", "reason": depth_failure}
                raise RuntimeError("Timed out waiting for all three rosbridge cameras.")
            wait_s = min(0.25, remaining)
            if depth_grace_deadline is not None:
                wait_s = min(wait_s, max(0.02, depth_grace_deadline - now))
            try:
                update = json.loads(socket.recv(timeout=wait_s))
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
            if depth_grace_deadline is None:
                depth_grace_deadline = min(deadline, time.monotonic() + 0.9)
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
                    depth_grace_deadline = min(depth_grace_deadline, time.monotonic() + 0.35)
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
        last_plan_left = None
        last_plan_count = 0
        is_cube_task = use_cube_depth and "cube" in task.casefold()
        instruction = build_task_instruction(task, use_cube_depth=is_cube_task)
        try:
            planner = ClaudeEndPosePlanner(
                api_key=self.key_planner.api_key, base_url=self.key_planner.base_url,
                model=self.key_planner.model, timeout_s=18, plan_every=2,
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
                if (
                    index == 1
                    and front_mount is not None
                    and measured["status"] != "measured"
                    and str(measured.get("reason", "")).startswith(("Missing fixed transform", "No synchronized"))
                ):
                    sampled_at = time.monotonic()
                    images, measured = read_camera_snapshot(self.rosbridge, stop, front_mount, static_cache)
                if stop.is_set():
                    break
                states = self.jogger.observe_direct(stop)
                observation = validate_request({
                    "embodiment": EMBODIMENT, "normalization": "provided", "instruction": instruction,
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
                if last_plan_left is not None:
                    context["since_last_plan"] = {
                        "measured_left_delta_xyz_mm": [
                            round(1000 * (actual - previous), 2)
                            for actual, previous in zip(observation.left_ee.position, last_plan_left.position)
                        ],
                        "measured_left_gripper_delta_rad": round(observation.left_ee.gripper - last_plan_left.gripper, 4),
                    }
                observation.perception = context or None
                if origin is None:
                    origin = list(observation.left_ee.position)
                if callable(getattr(type(planner), "decide_sequence", None)):
                    decisions = planner.decide_sequence(observation)
                else:
                    decisions = [planner.decide(observation)]
                decision = decisions[-1]
                history = getattr(planner, "history", None)
                history_len = len(history) if isinstance(history, list) else 0
                if history_len > last_plan_count or last_plan_left is None:
                    last_plan_count = history_len
                    last_plan_left = observation.left_ee
                if stop.is_set():
                    break
                if time.monotonic() >= self._operator_expires_at:
                    outcome, reason = "stopped", "The operator console disconnected."
                    break
                action = "hold"
                if decision.status not in ("completed", "retry"):
                    expiry = min(deadline, sampled_at + 35, self._operator_expires_at)
                    if callable(getattr(type(self.jogger), "apply_direct_sequence", None)):
                        action = self.jogger.apply_direct_sequence(observation, decisions, origin, stop, expiry)
                    else:
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
                if stop.wait(0.05):
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
