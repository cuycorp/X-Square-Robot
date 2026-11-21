"""WebSocket policy server and live prompt console for quanta_x1_pro (public_v2)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import html
import http.server
import json
import logging
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from claude_policy import (
    AuthError,
    ClaudeEndPosePlanner,
    HeuristicOfflinePlanner,
    PlannerDecision,
    hold_pose_decision,
)
from kinematics import arm_chain, jog_target, preview as preview_kinematics
from protocol import (
    ProtocolError,
    handshake_metadata,
    validate_request,
    validate_response,
)
from trajectory import WorkspaceLimits, synthesize_arm_chunk

LOGGER = logging.getLogger("claude_policy_server")
MAX_FRAME_BYTES = 64 * 1024 * 1024


def unpack_frame(frame: bytes) -> Any:
    return msgpack.unpackb(frame, raw=False, object_hook=msgpack_numpy.decode)


def pack_frame(value: Any) -> bytes:
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


class PromptController:
    """Thread-safe holder for the live natural-language task prompt and telemetry."""

    def __init__(self, initial_prompt: str | None = None) -> None:
        self._lock = threading.Lock()
        self._prompt: str | None = initial_prompt.strip() if initial_prompt and initial_prompt.strip() else None
        self._prompt_version = 1 if self._prompt else 0
        self._status: dict[str, Any] = {
            "active_prompt": self._prompt,
            "webui_instruction": None,
            "effective_instruction": self._prompt,
            "prompt_version": self._prompt_version,
            "chunk_index": 0,
            "stage": "idle",
            "waypoint_index": None,
            "waypoint_count": 0,
            "status": "waiting_for_robot",
            "reason": "Waiting for robot WebUI inference session to connect.",
            "connected": False,
            "left_ee": None,
            "right_ee": None,
            "latency_ms": None,
            "last_error": "",
            "updated_at": time.time(),
        }
        self._history: deque[dict[str, Any]] = deque(maxlen=40)
        self._images_b64: dict[str, str] = {}

    def set_prompt(self, prompt: str | None) -> int:
        with self._lock:
            cleaned = prompt.strip() if prompt and prompt.strip() else None
            self._prompt = cleaned
            self._prompt_version += 1
            self._status["active_prompt"] = cleaned
            self._status["effective_instruction"] = cleaned or self._status.get("webui_instruction")
            self._status["prompt_version"] = self._prompt_version
            self._status["waypoint_index"] = None
            self._status["waypoint_count"] = 0
            if cleaned:
                self._status["status"] = "queued"
                self._status["stage"] = "new_prompt"
                self._status["reason"] = f"New prompt queued: {cleaned}"
                self._status["last_error"] = ""
            self._status["updated_at"] = time.time()
            return self._prompt_version

    def get_effective_instruction(self, webui_instruction: str) -> tuple[str, int]:
        with self._lock:
            self._status["webui_instruction"] = webui_instruction
            effective = self._prompt if self._prompt else webui_instruction
            self._status["effective_instruction"] = effective
            return effective, self._prompt_version

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self._status["connected"] = connected
            if not connected and self._status["status"] != "auth_error":
                self._status["status"] = "waiting_for_robot"
                self._status["stage"] = "idle"
            if not connected:
                self._status["waypoint_index"] = None
                self._status["waypoint_count"] = 0
            self._status["updated_at"] = time.time()

    def record_step(
        self,
        *,
        chunk_index: int,
        instruction: str,
        decision: PlannerDecision,
        left_ee: dict[str, Any],
        right_ee: dict[str, Any],
        images_b64: dict[str, str],
        latency_ms: float,
        last_error: str = "",
    ) -> None:
        with self._lock:
            self._status.update(
                {
                    "effective_instruction": instruction,
                    "chunk_index": chunk_index,
                    "stage": decision.stage,
                    "waypoint_index": decision.waypoint_index,
                    "waypoint_count": decision.waypoint_count,
                    "status": decision.status,
                    "reason": decision.reason,
                    "connected": True,
                    "left_ee": left_ee,
                    "right_ee": right_ee,
                    "latency_ms": round(latency_ms, 1),
                    "last_error": last_error,
                    "updated_at": time.time(),
                }
            )
            self._images_b64 = dict(images_b64)
            self._history.append(
                {
                    "chunk": chunk_index,
                    "stage": decision.stage,
                    "status": decision.status,
                    "reason": decision.reason,
                    "latency_ms": round(latency_ms, 1),
                }
            )

    def record_error(self, msg: str, auth: bool = False) -> None:
        with self._lock:
            self._status["last_error"] = msg
            if auth:
                self._status["status"] = "auth_error"
            self._status["updated_at"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = copy.deepcopy(self._status)
            data["history"] = list(self._history)
            data["images_b64"] = dict(self._images_b64)
            return data


class DualArmPolicyEngine:
    def __init__(
        self,
        planner: ClaudeEndPosePlanner | HeuristicOfflinePlanner,
        limits: WorkspaceLimits | None = None,
        log_path: Path | None = None,
        prompt_ctrl: PromptController | None = None,
        max_image_dim: int = 640,
        max_consecutive_errors: int = 4,
    ) -> None:
        self.planner = planner
        self.limits = limits or WorkspaceLimits()
        self.log_path = log_path
        self.prompt_ctrl = prompt_ctrl or PromptController()
        self.max_image_dim = max_image_dim
        self.max_consecutive_errors = max_consecutive_errors
        self.chunk_index = 0
        self._last_prompt_version = -1
        self._consecutive_errors = 0
        self._completed_announced = False
        self.session_connected = False

    def handle_payload(self, raw_payload: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            req = validate_request(raw_payload, max_image_dim=self.max_image_dim)
            webui_instruction = req.instruction
            effective_goal, pver = self.prompt_ctrl.get_effective_instruction(webui_instruction)
            req.instruction = effective_goal
            if pver != self._last_prompt_version:
                self._last_prompt_version = pver
                self._completed_announced = False
                if hasattr(self.planner, "reset_for_goal"):
                    self.planner.reset_for_goal(effective_goal)

            err_msg = ""
            try:
                decision = self.planner.decide(req)
                self._consecutive_errors = 0
            except AuthError as auth_err:
                self.prompt_ctrl.record_error(str(auth_err), auth=True)
                self._write_log({"event": "auth_error", "message": str(auth_err)})
                raise
            except Exception as exc:
                self._consecutive_errors += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                self.prompt_ctrl.record_error(err_msg)
                self._write_log({"event": "planner_error", "attempt": self._consecutive_errors, "message": err_msg})
                if self._consecutive_errors > self.max_consecutive_errors:
                    raise
                decision = hold_pose_decision(
                    req,
                    stage="error_hold",
                    status="retry",
                    reason=f"Holding pose safely after transient planner error ({err_msg}).",
                )

            current_goal, current_version = self.prompt_ctrl.get_effective_instruction(webui_instruction)
            if current_version != pver:
                req.instruction = current_goal
                decision = hold_pose_decision(
                    req, stage="new_prompt", status="queued", reason="Waiting for a fresh plan for the new prompt."
                )

            left_prefix = req.prev_action_chunk.get("left_ee") if req.use_rtc and req.prev_action_chunk else None
            right_prefix = req.prev_action_chunk.get("right_ee") if req.use_rtc and req.prev_action_chunk else None

            left_chunk = synthesize_arm_chunk(
                current=req.left_ee,
                target_pos=decision.left_target_pos,
                target_rpy=decision.left_target_rpy,
                target_gripper=decision.left_gripper,
                gripper_range=req.gripper_range,
                limits=self.limits,
                rtc_prefix_arm=left_prefix,
                inference_delay=req.inference_delay,
            )
            right_chunk = synthesize_arm_chunk(
                current=req.right_ee,
                target_pos=decision.right_target_pos,
                target_rpy=decision.right_target_rpy,
                target_gripper=decision.right_gripper,
                gripper_range=req.gripper_range,
                limits=self.limits,
                rtc_prefix_arm=right_prefix,
                inference_delay=req.inference_delay,
            )
            response = {
                "embodiment": req.embodiment,
                "actions": {
                    "left_ee": left_chunk,
                    "right_ee": right_chunk,
                },
            }
            validate_response(response, req.embodiment)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.chunk_index += 1

            if decision.status == "completed" and not self._completed_announced:
                self._completed_announced = True
                print(
                    f"\n[COMPLETED] {decision.reason}\n"
                    "Holding current pose safely. Enter a new prompt below, or click "
                    "'End Evaluation Inference Mode' in the robot WebUI to stop & home.\n",
                    flush=True,
                )

            self.prompt_ctrl.record_step(
                chunk_index=self.chunk_index,
                instruction=req.instruction,
                decision=decision,
                left_ee={
                    "position": [round(v, 4) for v in req.left_ee.position],
                    "gripper": round(req.left_ee.gripper, 4),
                },
                right_ee={
                    "position": [round(v, 4) for v in req.right_ee.position],
                    "gripper": round(req.right_ee.gripper, 4),
                },
                images_b64=req.images_b64,
                latency_ms=dt_ms,
                last_error=err_msg,
            )
            self._write_log(
                {
                    "event": "chunk",
                    "chunk_index": self.chunk_index,
                    "instruction": req.instruction,
                    "stage": decision.stage,
                    "status": decision.status,
                    "reason": decision.reason,
                    "use_rtc": req.use_rtc,
                    "latency_ms": round(dt_ms, 2),
                }
            )
            return response
        except ProtocolError as err:
            self._write_log({"event": "protocol_error", "code": err.code, "message": err.message})
            return err.to_dict()

    def _write_log(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry)
        LOGGER.info(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


async def serve_connection(conn: Any, engine: DualArmPolicyEngine) -> None:
    if engine.session_connected:
        await conn.close(code=1013, reason="A robot policy session is already connected")
        return
    engine.session_connected = True
    engine._last_prompt_version = -1
    engine._consecutive_errors = 0
    engine.prompt_ctrl.set_connected(True)
    try:
        await conn.send(pack_frame(handshake_metadata()))
        async for raw_msg in conn:
            if isinstance(raw_msg, str):
                err = ProtocolError("missing_field", "Expected binary MessagePack frame, got text").to_dict()
                await conn.send(pack_frame(err))
                continue
            payload = unpack_frame(raw_msg)
            reply = await asyncio.to_thread(engine.handle_payload, payload)
            await conn.send(pack_frame(reply))
    except ConnectionClosed:
        LOGGER.info("Robot client disconnected; confirm session state in WebUI.")
    except AuthError as err:
        LOGGER.error("%s", err)
        await conn.close(code=1011, reason="Anthropic authentication error")
    except Exception as err:
        LOGGER.error("Policy connection closed on error: %s", type(err).__name__)
        try:
            await conn.close(code=1011, reason="Policy server error; stop in WebUI")
        except Exception:
            pass
    finally:
        engine.session_connected = False
        engine.prompt_ctrl.set_connected(False)


CONSOLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quanta X1 Pro - Claude Autonomous Prompt Console</title>
<style>
  * { box-sizing: border-box; }
  body { font: 13px system-ui, -apple-system, sans-serif; margin: 0; padding: 8px clamp(8px, 1.3vw, 20px); background: #f2f5fa; color: #1a1d24; }
  .wrap { max-width: 1920px; margin: 0 auto; }
  h1 { margin: 0; font-size: 18px; }
  .card { min-width: 0; background: #fff; border: 1px solid #dce3ee; border-radius: 8px; padding: 8px; margin-bottom: 8px; }
  .dashboard { display: grid; grid-template-columns: minmax(0, 2.5fr) minmax(235px, 1fr); gap: 8px; align-items: start; }
  .header, .camera-header, .jog-header, .dialog-header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 4px 8px; margin-bottom: 5px; }
  .header .hint, .camera-header h2, .jog-header h2, .dialog-header h2 { margin: 0; }
  .row { display: flex; gap: 4px; flex-wrap: wrap; }
  .row[hidden] { display: none; }
  input[type=text], input[type=url], input[type=password] { flex: 1; min-width: min(130px, 100%); padding: 5px 7px; font: inherit; border: 1px solid #cfd6e4; border-radius: 5px; }
  button { min-height: 26px; padding: 4px 8px; font: inherit; font-weight: 600; border: 0; border-radius: 5px; cursor: pointer; background: #4f44e1; color: #fff; }
  button.sec { background: #eef1f7; color: #1a1d24; }
  button:disabled { opacity: .55; cursor: default; }
  h2 { font-size: 14px; margin: 0 0 5px; }
  .camera-controls { align-items: center; }
  .camera-controls label { font-size: 13px; }
  .camera-controls input { min-width: 0; width: min(210px, 54vw); }
  .camera-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 5px; }
  .camera-tile { position: relative; min-width: 0; margin: 0; overflow: hidden; border-radius: 5px; background: #111722; color: #f5f7fb; }
  .camera-view { position: relative; height: clamp(72px, calc(100dvh - 430px), 330px); }
  .camera-view img { display: block; width: 100%; height: 100%; object-fit: contain; }
  .camera-view p { position: absolute; inset: 0; display: grid; place-items: center; margin: 0; padding: 8px; color: #c7d1df; text-align: center; }
  .camera-view [hidden] { display: none; }
  .camera-tile figcaption { position: absolute; top: 4px; left: 4px; padding: 2px 5px; border-radius: 3px; background: #111722eb; font-size: 11px; pointer-events: none; }
  #camera-status { color: #5b6475; margin: 0; }
  .hint { color: #5b6475; font-size: 12px; margin: 4px 0; }
  .hint:empty { display: none; }
  .joint-table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  .joint-table th, .joint-table td { padding: 1px 4px; text-align: left; border-bottom: 1px solid #e2e6ee; line-height: 1.25; }
  .pose-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin: 12px 0; }
  .pose-fields label { display: grid; gap: 4px; font-size: 14px; }
  .pose-fields input { min-width: 0; padding: 8px; border: 1px solid #cfd6e4; border-radius: 8px; font: inherit; }
  #ik-result { white-space: pre-wrap; padding-top: 4px; }
  .jog-settings { align-items: center; }
  .jog-settings select { margin-left: 4px; font: inherit; }
  .arm-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin-bottom: 5px; }
  .arm-controls { min-width: 0; border: 1px solid #dce3ee; border-radius: 6px; padding: 6px; }
  .arm-controls h3 { margin: 0; font-size: 14px; }
  .arm-controls h4 { margin: 4px 0 0; font-size: 12px; }
  .jog-buttons { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 3px; margin: 3px 0 5px; }
  .jog-pair { display: flex; gap: 2px; }
  .jog-pair button { flex: 1; min-width: 0; padding: 2px; white-space: nowrap; }
  .jog-output { white-space: pre-wrap; font: 11px ui-monospace, monospace; }
  dialog { width: min(900px, calc(100vw - 24px)); max-height: calc(100dvh - 24px); overflow: auto; color: inherit; }
  dialog.card { margin: auto; padding: 12px; }
  dialog::backdrop { background: #11172299; }
  #key-form label { display: flex; flex: 1; flex-wrap: wrap; align-items: center; gap: 4px; }
  #prompt { flex-basis: 100%; }
  .cams { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
  .cam img { width: 100%; border-radius: 8px; border: 1px solid #e2e6ee; background: #111; min-height: 150px; object-fit: contain; }
  .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 14px; }
  .kv b { display: block; color: #5b6475; font-size: 12px; text-transform: uppercase; }
  .log { font-family: ui-monospace, monospace; font-size: 12px; max-height: 220px; overflow: auto; background: #f3f5f9; padding: 10px; border-radius: 8px; }
  .warn { background: #fff4e5; border: 1px solid #f5c26b; padding: 7px 10px; border-radius: 8px; font-size: 13px; margin: 6px 0 10px; }
  .jog-output { display: flex; flex-wrap: wrap; gap: 4px 12px; }
  .jog-output span { white-space: nowrap; }
  @media (max-width: 820px) {
    .dashboard { display: block; }
    .sidebar { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .camera-view { height: clamp(62px, calc(100dvh - 680px), 240px); }
  }
  @media (max-width: 480px) {
    .arm-grid, .sidebar { grid-template-columns: minmax(0, 1fr); }
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="header">
    <h1>Quanta X1 Pro Console</h1>
    <p class="hint" id="camera-status" role="status">Telemetry disconnected</p>
  </header>
  <section class="card" aria-labelledby="live-camera-title">
    <div class="camera-header">
      <h2 id="live-camera-title">Live Robot Cameras</h2>
      <form id="camera-form" class="row camera-controls">
        <label>Rosbridge <input id="camera-url" type="url" required placeholder="ws://192.168.77.2:9090/" spellcheck="false"></label>
        <button id="camera-connect" type="submit">Connect</button>
        <button id="camera-disconnect" class="sec" type="button" hidden>Disconnect</button>
      </form>
    </div>
    <div class="camera-grid" id="live-cameras"></div>
  </section>
  <main class="dashboard">
  <section class="card" aria-labelledby="jog-title">
    <div class="jog-header">
      <h2 id="jog-title">Incremental Jog</h2>
    <div class="row jog-settings">
      <label>Translation step
        <select id="jog-linear"><option value="1">1 mm</option><option value="2" selected>2 mm</option><option value="5">5 mm</option></select>
      </label>
      <label>Rotation and joint step
        <select id="jog-angular"><option value="0.5">0.5°</option><option value="1" selected>1°</option><option value="2">2°</option></select>
      </label>
    </div>
    </div>
    <p id="jog-mode" class="hint" data-live="false">Preview only. Connect to rosbridge for measured joint angles.</p>
    <div id="arm-controls" class="arm-grid" data-manual="false"></div>
    <template id="arm-template">
      <section class="arm-controls">
        <h3></h3>
        <h4>Cartesian <small class="hint">in base_link</small></h4>
        <div class="jog-buttons cartesian-jog"></div>
        <h4>Motor joints</h4>
        <div class="jog-buttons joint-jog"></div>
        <div class="row">
          <button type="button" data-action="home">Home</button>
          <button class="sec" type="button" data-action="open">Open Gripper</button>
          <button class="sec" type="button" data-action="close">Close Gripper</button>
        </div>
        <p class="jog-status hint" role="status"></p>
        <output class="jog-output"></output>
      </section>
    </template>
    <div class="row">
      <button class="sec" type="button" data-dialog="ik-dialog">Advanced IK</button>
      <button class="sec" type="button" data-dialog="inference-dialog">Inference Details</button>
      <div id="manual-release" hidden>
        <button id="release-sdk" class="sec" type="button">Release Manual Control</button>
      </div>
    </div>
    <p id="release-status" class="hint" role="status"></p>
    <dialog class="card" id="ik-dialog" aria-labelledby="ik-title">
    <div class="dialog-header">
      <h2 id="ik-title">Advanced pose preview</h2>
      <button type="button" class="sec" data-close>Close</button>
    </div>
    <p class="hint">Enter a gripper-base pose relative to <code>base_link</code>. The preview uses the robot model and live joint angles. It checks joint limits, and collision checks are outside this calculation.</p>
    <form id="ik-form">
      <label>Arm <select id="ik-arm"><option value="left">Left</option><option value="right">Right</option></select></label>
      <div class="pose-fields" id="ik-pose">
        <label>X (m)<input type="number" step="any" required></label>
        <label>Y (m)<input type="number" step="any" required></label>
        <label>Z (m)<input type="number" step="any" required></label>
        <label>Roll (rad)<input type="number" step="any" required></label>
        <label>Pitch (rad)<input type="number" step="any" required></label>
        <label>Yaw (rad)<input type="number" step="any" required></label>
      </div>
      <div class="row">
        <button id="ik-current" type="button" class="sec">Use Current Pose</button>
        <button id="ik-solve" type="submit">Calculate IK</button>
      </div>
    </form>
    <p id="ik-status" class="hint" role="status">Waiting for the robot model and measured joint angles.</p>
    <output id="ik-result" for="ik-arm ik-solve"></output>
    </dialog>
  </section>
  <aside class="sidebar">
    <section class="card" aria-labelledby="joints-title">
      <h2 id="joints-title">Measured Joint Angles</h2>
      <p id="joints-status" class="hint" role="status">Connect to rosbridge above to read joint states.</p>
      <table class="joint-table">
        <thead><tr><th scope="col">Joint</th><th scope="col">Left (rad)</th><th scope="col">Right (rad)</th></tr></thead>
        <tbody id="joint-values"></tbody>
        <tfoot><tr><th scope="row">Gripper</th><td id="left-gripper">-</td><td id="right-gripper">-</td></tr></tfoot>
      </table>
    </section>
    <section class="card" aria-labelledby="prompt-title">
      <h2 id="prompt-title"><label for="prompt">Claude Zero-Shot Task</label></h2>
      <form id="key-form" class="row" hidden>
        <label>Anthropic API key
          <input id="anthropic-key" type="password" required autocomplete="off" spellcheck="false">
        </label>
        <button type="submit">Verify Key</button>
      </form>
      <p id="key-status" class="hint" role="status"></p>
      <form id="pform" class="row" data-enabled="true">
        <input id="prompt" type="text" required placeholder="Pick up the spoon and place it into the blue bowl.">
        <button type="submit">Run Zero-Shot Prompt</button>
        <button type="button" class="sec" onclick="sendPrompt('Hold current pose safely.')">Hold Pose</button>
        <button id="use-webui-case" type="button" class="sec" onclick="sendPrompt('')">Use WebUI Case</button>
      </form>
      <p id="prompt-status" class="hint" role="status"></p>
      <p id="conn" class="hint">Waiting for WebUI Start</p>
    </section>
  </aside>
  </main>
  <dialog class="card" id="inference-dialog" aria-labelledby="inference-title">
    <div class="dialog-header">
      <h2 id="inference-title">Inference details</h2>
      <button type="button" class="sec" data-close>Close</button>
    </div>
    <p class="warn">Use <b>End Evaluation Inference Mode</b> in the robot WebUI to end an inference run.</p>
  <div class="meta">
    <div class="kv"><b>Status / Stage</b><span id="st">-</span></div>
    <div class="kv"><b>Waypoint</b><span id="waypoint">-</span></div>
    <div class="kv"><b>Chunk / Latency</b><span id="lat">-</span></div>
    <div class="kv"><b>Active Goal</b><span id="goal">-</span></div>
  </div>
  <div class="card">
    <b>Claude Reasoning</b>
    <p id="reason" style="margin:6px 0 0">-</p>
    <p id="err" style="margin:6px 0 0;color:#c0392b"></p>
  </div>
  <div class="card" id="inference-frames" hidden>
    <h2>Frames used by the planner</h2>
    <div class="cams">
      <div class="cam"><b>Front (Right arm on left)</b><img id="img_front" alt="front"></div>
      <div class="cam"><b>Left Wrist</b><img id="img_left_wrist" alt="left_wrist"></div>
      <div class="cam"><b>Right Wrist</b><img id="img_right_wrist" alt="right_wrist"></div>
    </div>
  </div>
  <div class="card">
    <b>Recent Steps</b>
    <div id="log" class="log">Waiting for observations...</div>
  </div>
  </dialog>
</div>
<script>
for (const button of document.querySelectorAll('[data-dialog]')) {
  button.addEventListener('click', () => document.getElementById(button.dataset.dialog).showModal());
}
for (const button of document.querySelectorAll('[data-close]')) {
  button.addEventListener('click', () => button.closest('dialog').close());
}
const cameraForm = document.getElementById('camera-form');
const cameraUrl = document.getElementById('camera-url');
const cameraConnect = document.getElementById('camera-connect');
const cameraDisconnect = document.getElementById('camera-disconnect');
const cameraStatus = document.getElementById('camera-status');
const jointsStatus = document.getElementById('joints-status');
const jointValues = document.getElementById('joint-values');
const ikStatus = document.getElementById('ik-status');
const ikResult = document.getElementById('ik-result');
const ikArm = document.getElementById('ik-arm');
const jogMode = document.getElementById('jog-mode');
const jogLive = jogMode.dataset.live === 'true';
const armControls = document.getElementById('arm-controls');
const manualEnabled = armControls.dataset.manual === 'true';
const releaseStatus = document.getElementById('release-status');
const releaseButton = document.getElementById('release-sdk');
const promptForm = document.getElementById('pform');
const promptsEnabled = promptForm.dataset.enabled === 'true';
const manualPromptMessage = 'Zero-shot runs in the Claude policy console after the manual SDK session ends.';
const keyForm = document.getElementById('key-form');
const keyStatus = document.getElementById('key-status');
let keyReady = keyForm.hidden;
let policyConnected = false;
const poseInputs = [...document.querySelectorAll('#ik-pose input')];
const ikButtons = ['ik-current', 'ik-solve'].map(id => document.getElementById(id));
let cameraSocket;
let cameraTimer;
let jointTimer;
let measuredJoints;
let robotDescription;
let ikPending = false;
let jogPending = false;
let manualReleased = false;
const armPanels = new Map();
const feeds = new Map([
  ['/camera_head_front/color/image_raw/compressed', 'Front camera'],
  ['/camera1/usb_cam1/image_raw/image_compressed', 'Camera 1'],
  ['/camera3/usb_cam3/image_raw/image_compressed', 'Camera 3']
].map(([topic, label]) => {
  const tile = document.createElement('figure');
  tile.className = 'camera-tile';
  const view = document.createElement('div');
  view.className = 'camera-view';
  const image = document.createElement('img');
  image.alt = label;
  image.hidden = true;
  const message = document.createElement('p');
  message.textContent = 'Connect to view';
  const caption = document.createElement('figcaption');
  caption.textContent = label;
  view.append(image, message);
  tile.append(view, caption);
  document.getElementById('live-cameras').append(tile);
  const feed = { image, message, timer: undefined };
  image.addEventListener('load', () => {
    if (!cameraSocket || !image.naturalWidth) return;
    image.hidden = false;
    message.hidden = true;
    watchForCameraFrames(feed);
  });
  return [topic, feed];
}));

const jointRows = Array.from({ length: 6 }, (_, index) => {
  const row = jointValues.insertRow();
  const heading = document.createElement('th');
  heading.scope = 'row';
  heading.textContent = index + 1;
  row.append(heading);
  return [row.insertCell(), row.insertCell()];
});

function ikReady(arm = ikArm.value) {
  return Boolean(robotDescription) && jointRows.every((_, index) =>
    Number.isFinite(measuredJoints?.[arm + '_arm_joint' + (index + 1)])
  );
}

function renderJointAngles() {
  for (const [index, cells] of jointRows.entries()) {
    for (const [column, side] of ['left', 'right'].entries()) {
      const value = measuredJoints?.[side + '_arm_joint' + (index + 1)];
      cells[column].textContent = Number.isFinite(value) ? value.toFixed(4) : '-';
    }
  }
  for (const side of ['left', 'right']) {
    const value = measuredJoints?.[side + '_arm_gripper'];
    document.getElementById(side + '-gripper').textContent = Number.isFinite(value) ? value.toFixed(4) : '-';
  }
  const ready = ikReady();
  for (const button of ikButtons) button.disabled = !ready || ikPending;
  releaseButton.disabled = jogPending || manualReleased;
  for (const [arm, panel] of armPanels) {
    const disabled = !ikReady(arm) || jogPending || manualReleased;
    for (const button of panel.buttons) {
      button.disabled = disabled || (Boolean(button.dataset.action) && !manualEnabled);
    }
  }
}

function reportIKAvailability() {
  ikStatus.textContent = ikReady()
    ? 'Ready to calculate a pose preview.'
    : 'Waiting for the robot model and measured joint angles.';
}

function recordRobotTelemetry(socket, update) {
  if (update.topic === '/joint_states') {
    const { name, position } = update.msg || {};
    if (!Array.isArray(name) || !Array.isArray(position)) return;
    const wasReady = ikReady();
    measuredJoints = Object.fromEntries(name.map((joint, index) => [joint, position[index]]));
    jointsStatus.textContent = 'Live · /joint_states';
    renderJointAngles();
    if (wasReady !== ikReady()) reportIKAvailability();
    clearTimeout(jointTimer);
    jointTimer = setTimeout(() => {
      measuredJoints = undefined;
      jointsStatus.textContent = 'No recent joint states.';
      ikResult.textContent = '';
      renderJointAngles();
      reportIKAvailability();
    }, 5000);
  }
  if (update.topic === '/robot_description' && !robotDescription && typeof update.msg?.data === 'string') {
    robotDescription = update.msg.data;
    socket.send(JSON.stringify({ op: 'unsubscribe', topic: '/robot_description' }));
    renderJointAngles();
    reportIKAvailability();
  }
}

async function requestKinematics(useCurrentPose) {
  const socket = cameraSocket;
  const arm = ikArm.value;
  if (!ikReady() || ikPending) return;
  const target = useCurrentPose ? null : poseInputs.map(input => input.valueAsNumber);
  if (target && !target.every(Number.isFinite)) {
    ikStatus.textContent = 'Enter six finite pose values.';
    return;
  }
  ikPending = true;
  renderJointAngles();
  ikStatus.textContent = 'Calculating…';
  ikResult.textContent = '';
  try {
    const response = await fetch('/api/ik', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ arm, urdf: robotDescription, measured: measuredJoints, target })
    });
    const result = await response.json();
    if (socket !== cameraSocket || arm !== ikArm.value || !ikReady()) return;
    if (!response.ok) throw new Error(result.error || 'The IK preview failed.');
    if (useCurrentPose) {
      result.pose.forEach((value, index) => { poseInputs[index].value = value.toFixed(6); });
      ikStatus.textContent = 'Loaded the current gripper-base pose.';
    } else {
      const residual = result.position_error_m.toFixed(4) + ' m / ' + result.rotation_error_rad.toFixed(4) + ' rad';
      ikStatus.textContent = (result.converged ? 'Solution found. Residual ' : 'No solution from this seed. Residual ') + residual + '.';
      ikResult.textContent = Object.entries(result.joints).map(([name, angle]) => name + '  ' + angle.toFixed(4) + ' rad').join('\n');
    }
  } catch (error) {
    if (socket === cameraSocket && arm === ikArm.value) ikStatus.textContent = error.message;
  } finally {
    ikPending = false;
    renderJointAngles();
  }
}

async function requestJog(arm, axis, direction) {
  if (!ikReady(arm) || jogPending || manualReleased) return;
  const socket = cameraSocket;
  const { jogStatus, jogResult } = armPanels.get(arm);
  const step = Number(document.getElementById(
    ['x', 'y', 'z'].includes(axis) ? 'jog-linear' : 'jog-angular'
  ).value);
  jogPending = true;
  renderJointAngles();
  jogStatus.textContent = jogLive ? 'Sending one increment…' : 'Calculating one increment…';
  jogResult.textContent = '';
  try {
    const body = { arm, axis, direction, step };
    if (!jogLive) Object.assign(body, { urdf: robotDescription, measured: measuredJoints });
    const response = await fetch('/api/jog', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    const result = await response.json();
    if (socket !== cameraSocket) return;
    if (!response.ok) throw new Error(result.error || 'The jog failed.');
    jogStatus.textContent = result.sent ? 'SDK accepted the target. Check measured angles for motion.' : 'Preview calculated. No motion sent.';
    jogResult.replaceChildren(...Object.entries(result.joints).map(([name, angle]) => {
      const value = document.createElement('span');
      value.textContent = 'J' + name.slice(-1) + ' ' + angle.toFixed(4) + ' rad';
      return value;
    }));
  } catch (error) {
    if (socket === cameraSocket) jogStatus.textContent = error.message;
  } finally {
    jogPending = false;
    renderJointAngles();
  }
}

async function requestManualAction(arm, action) {
  if (!manualEnabled || !ikReady(arm) || jogPending || manualReleased) return;
  const socket = cameraSocket;
  const { jogStatus, jogResult } = armPanels.get(arm);
  jogPending = true;
  renderJointAngles();
  jogResult.textContent = '';
  jogStatus.textContent = 'Sending ' + action + ' request for the ' + arm + ' side…';
  try {
    const response = await fetch('/api/manual', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ arm, action })
    });
    const result = await response.json();
    if (socket !== cameraSocket) return;
    if (!response.ok) throw new Error(result.error || 'The manual request failed.');
    jogStatus.textContent = result.message;
  } catch (error) {
    if (socket === cameraSocket) jogStatus.textContent = error.message;
  } finally {
    jogPending = false;
    renderJointAngles();
  }
}

releaseButton.addEventListener('click', async () => {
  if (jogPending || manualReleased) return;
  jogPending = true;
  renderJointAngles();
  try {
    const response = await fetch('/api/manual/release', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not release manual control.');
    manualReleased = true;
    jogMode.textContent = 'Manual SDK control released.';
    releaseStatus.textContent = result.message;
  } catch (error) {
    releaseStatus.textContent = error.message;
  } finally {
    jogPending = false;
    renderJointAngles();
  }
});

for (const [arm, label] of [['left', 'Left Arm'], ['right', 'Right Arm']]) {
  const panel = document.getElementById('arm-template').content.firstElementChild.cloneNode(true);
  const title = panel.querySelector('h3');
  title.id = arm + '-controls-title';
  title.textContent = label;
  panel.setAttribute('aria-labelledby', title.id);
  const buttons = [...panel.querySelectorAll('button')];
  panel.querySelector('[data-action="home"]').setAttribute('aria-label', 'Home ' + label);
  for (const button of buttons) {
    button.addEventListener('click', () => requestManualAction(arm, button.dataset.action));
  }
  for (const [selector, axes] of [
    ['.cartesian-jog', ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']],
    ['.joint-jog', ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']]
  ]) {
    const container = panel.querySelector(selector);
    for (const axis of axes) {
      const pair = document.createElement('div');
      pair.className = 'jog-pair';
      for (const direction of [-1, 1]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'sec';
        button.textContent = axis + (direction < 0 ? ' −' : ' +');
        button.addEventListener('click', () => requestJog(arm, axis.toLowerCase(), direction));
        pair.append(button);
        buttons.push(button);
      }
      container.append(pair);
    }
  }
  for (const button of buttons) {
    button.addEventListener('keydown', event => { if (event.repeat) event.preventDefault(); });
  }
  armPanels.set(arm, {
    buttons, jogStatus: panel.querySelector('.jog-status'), jogResult: panel.querySelector('.jog-output')
  });
  armControls.append(panel);
}

document.getElementById('ik-form').addEventListener('submit', event => {
  event.preventDefault();
  requestKinematics(false);
});
document.getElementById('ik-current').addEventListener('click', () => requestKinematics(true));
ikArm.addEventListener('change', () => {
  for (const input of poseInputs) input.value = '';
  ikResult.textContent = '';
  renderJointAngles();
  reportIKAvailability();
});
renderJointAngles();

if (sessionStorage.getItem('robot-rosbridge')) {
  cameraUrl.value = sessionStorage.getItem('robot-rosbridge');
} else if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
  cameraUrl.value = 'ws://' + location.hostname + ':9090/';
}

function clearCameraFrame(feed, detail) {
  clearTimeout(feed.timer);
  feed.image.hidden = true;
  feed.image.removeAttribute('src');
  feed.message.textContent = detail;
  feed.message.hidden = false;
}

function disconnectCamera(status, message) {
  const socket = cameraSocket;
  cameraSocket = undefined;
  clearTimeout(cameraTimer);
  clearTimeout(jointTimer);
  measuredJoints = undefined;
  robotDescription = undefined;
  socket?.close();
  cameraUrl.disabled = false;
  cameraConnect.hidden = false;
  cameraDisconnect.hidden = true;
  cameraStatus.textContent = status;
  for (const feed of feeds.values()) clearCameraFrame(feed, message);
  jointsStatus.textContent = 'Connect to rosbridge above to read joint states.';
  ikResult.textContent = '';
  for (const { jogStatus, jogResult } of armPanels.values()) {
    jogResult.textContent = '';
    jogStatus.textContent = '';
  }
  renderJointAngles();
  reportIKAvailability();
}

function watchForCameraFrames(feed) {
  clearTimeout(feed.timer);
  feed.timer = setTimeout(() => {
    clearCameraFrame(feed, 'No recent frames');
  }, 5000);
}

cameraForm.addEventListener('submit', event => {
  event.preventDefault();
  if (cameraSocket) return;
  let socket;
  try {
    const url = new URL(cameraUrl.value);
    if (!['ws:', 'wss:'].includes(url.protocol)) throw new Error();
    socket = new WebSocket(url);
  } catch {
    cameraStatus.textContent = 'Enter a valid rosbridge WebSocket URL.';
    return;
  }
  cameraSocket = socket;
  cameraUrl.disabled = true;
  cameraConnect.hidden = true;
  cameraDisconnect.hidden = false;
  cameraStatus.textContent = 'Connecting to rosbridge';
  for (const feed of feeds.values()) clearCameraFrame(feed, 'Waiting for frames');
  cameraTimer = setTimeout(() => {
    disconnectCamera('Connection timed out', 'Check the robot address and Ethernet connection.');
  }, 5000);
  socket.addEventListener('open', () => {
    if (socket !== cameraSocket) return;
    clearTimeout(cameraTimer);
    sessionStorage.setItem('robot-rosbridge', cameraUrl.value);
    cameraStatus.textContent = 'Telemetry connected';
    socket.send(JSON.stringify({ op: 'subscribe', topic: '/joint_states', throttle_rate: 350, queue_length: 1 }));
    socket.send(JSON.stringify({ op: 'subscribe', topic: '/robot_description', queue_length: 1 }));
    for (const [topic, feed] of feeds) {
      socket.send(JSON.stringify({ op: 'subscribe', topic, throttle_rate: 400, queue_length: 1 }));
      watchForCameraFrames(feed);
    }
  });
  socket.addEventListener('message', event => {
    if (socket !== cameraSocket) return;
    let update;
    try {
      update = JSON.parse(event.data);
    } catch {
      return;
    }
    if (update.op !== 'publish') return;
    recordRobotTelemetry(socket, update);
    const feed = feeds.get(update.topic);
    if (!feed) return;
    const { format, data } = update.msg || {};
    if (typeof format !== 'string' || !/jpeg|jpg/i.test(format) || typeof data !== 'string' || !data) return;
    feed.image.src = 'data:image/jpeg;base64,' + data;
  });
  socket.addEventListener('error', () => {
    if (socket === cameraSocket) {
      disconnectCamera('Connection failed', 'Check the robot address and Ethernet connection.');
    }
  });
  socket.addEventListener('close', () => {
    if (socket === cameraSocket) {
      disconnectCamera('Connection closed', 'Reconnect to resume viewing.');
    }
  });
});

cameraDisconnect.addEventListener('click', () => {
  disconnectCamera('Telemetry disconnected', 'Connect to view');
});

async function sendPrompt(val) {
  const feedback = document.getElementById('prompt-status');
  if (!promptsEnabled) {
    feedback.textContent = manualPromptMessage;
    return;
  }
  if (!keyReady) return;
  try {
    const response = await fetch('/api/prompt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt: val })
    });
    if (!response.ok) throw new Error('The console did not accept the prompt.');
    if (val) document.getElementById('prompt').value = val;
    feedback.textContent = val
      ? (policyConnected ? 'Prompt updated for the next observation.' : 'Prompt armed. Start evaluation in the robot WebUI.')
      : 'Using the task from the robot WebUI.';
    poll();
  } catch (error) {
    feedback.textContent = error.message;
  }
}
function updatePromptAvailability() {
  for (const input of promptForm.elements) input.disabled = !promptsEnabled || !keyReady;
}
updatePromptAvailability();
if (!keyReady) keyStatus.textContent = 'Verify an Anthropic key to enable prompts.';
if (!keyForm.hidden) document.getElementById('use-webui-case').hidden = true;
if (!promptsEnabled) {
  document.getElementById('prompt-status').textContent = manualPromptMessage;
}
keyForm.addEventListener('submit', async event => {
  event.preventDefault();
  const input = document.getElementById('anthropic-key');
  const button = keyForm.querySelector('button');
  button.disabled = true;
  keyStatus.textContent = 'Checking model access with Anthropic…';
  try {
    const response = await fetch('/api/anthropic-key', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: input.value })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not verify the key.');
    keyReady = true;
    input.value = '';
    keyStatus.textContent = 'Anthropic accepted the key. It is held in memory for this session.';
    updatePromptAvailability();
  } catch (error) {
    keyStatus.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
promptForm.onsubmit = event => {
  event.preventDefault();
  const input = document.getElementById('prompt');
  const prompt = input.value.trim();
  if (!prompt) {
    document.getElementById('prompt-status').textContent = 'Enter a task for Claude.';
    return;
  }
  sendPrompt(prompt);
};
async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    policyConnected = d.connected;
    if (typeof d.key_ready === 'boolean' && keyReady !== d.key_ready) {
      keyReady = d.key_ready;
      keyStatus.textContent = keyReady ? 'Anthropic key configured.' : 'Verify an Anthropic key to enable prompts.';
      updatePromptAvailability();
    }
    document.getElementById('conn').textContent = promptsEnabled
      ? (d.connected ? 'Robot Connected' : 'Waiting for WebUI Start')
      : '';
    document.getElementById('st').textContent = (d.status || '-') + ' / ' + (d.stage || '-');
    document.getElementById('waypoint').textContent = d.waypoint_index ? d.waypoint_index + ' / ' + d.waypoint_count : '-';
    document.getElementById('lat').textContent = (d.chunk_index || 0) + ' (' + (d.latency_ms ?? '-') + ' ms)';
    document.getElementById('goal').textContent = d.effective_instruction || '(from WebUI case)';
    document.getElementById('reason').textContent = d.reason || '-';
    document.getElementById('err').textContent = d.last_error || '';
    document.getElementById('inference-frames').hidden = !Object.values(d.images_b64 || {}).some(Boolean);
    for (const k of ['front','left_wrist','right_wrist']) {
      if (d.images_b64 && d.images_b64[k]) document.getElementById('img_' + k).src = 'data:image/jpeg;base64,' + d.images_b64[k];
    }
    if (d.history && d.history.length) {
      document.getElementById('log').textContent = d.history.slice().reverse().map(
        x => `#${x.chunk} [${x.status}/${x.stage}] (${x.latency_ms} ms): ${x.reason}`
      ).join('\n');
    }
  } catch (_) {}
}
setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""


def start_console_http_server(
    host: str,
    port: int,
    prompt_ctrl: PromptController,
    *,
    live_jog: Callable[[str, str, int, float], dict[str, float]] | None = None,
    live_action: Callable[[str, str], str] | None = None,
    release_manual: Callable[[], str] | None = None,
    prompts_enabled: bool = True,
    key_planner: ClaudeEndPosePlanner | None = None,
) -> http.server.ThreadingHTTPServer:
    if any(item is not None for item in (live_jog, live_action, release_manual, key_planner)) and host != "127.0.0.1":
        raise ValueError("Live SDK controls and cloud key entry require a localhost-only console")
    page = CONSOLE_HTML
    if live_jog is not None:
        page = page.replace(
            'data-live="false">Preview only. Connect to rosbridge for measured joint angles.',
            'data-live="true">Live SDK mode. Each click sends one target to the selected robot arm.',
        )
    if live_action is not None:
        page = page.replace('id="arm-controls" class="arm-grid" data-manual="false"', 'id="arm-controls" class="arm-grid" data-manual="true"')
    if release_manual is not None:
        page = page.replace('id="manual-release" hidden', 'id="manual-release"')
    if key_planner is not None:
        page = page.replace('id="key-form" class="row" hidden', 'id="key-form" class="row"')
    if not prompts_enabled:
        page = page.replace('id="pform" class="row" data-enabled="true"', 'id="pform" class="row" data-enabled="false"')
    page_body = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def reply_json(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def require_local_json(self) -> None:
            port = self.server.server_port
            origin_host = self.headers.get("Host")
            if (
                origin_host not in (f"localhost:{port}", f"127.0.0.1:{port}")
                or self.headers.get("Origin") != f"http://{origin_host}"
                or self.headers.get_content_type() != "application/json"
            ):
                raise PermissionError("This request must come from the local console.")

        def do_GET(self) -> None:
            if self.path == "/api/status":
                data = prompt_ctrl.snapshot()
                if key_planner is not None:
                    data["key_ready"] = bool(key_planner.api_key)
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page_body)))
            self.end_headers()
            self.wfile.write(page_body)

        def do_POST(self) -> None:
            if self.path in ("/api/anthropic-key", "/api/manual/release"):
                try:
                    self.require_local_json()
                    if self.path == "/api/manual/release" and release_manual is not None:
                        self.reply_json(200, {"message": release_manual()})
                    elif self.path == "/api/anthropic-key" and key_planner is not None:
                        if prompt_ctrl.snapshot()["connected"]:
                            raise RuntimeError("End evaluation before changing the Anthropic key.")
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 4096:
                            raise ValueError("Invalid Anthropic key request.")
                        key = json.loads(self.rfile.read(length))["key"]
                        if not isinstance(key, str):
                            raise ValueError("Invalid Anthropic key.")
                        key_planner.configure_api_key(key)
                        self.reply_json(200, {"key_ready": True})
                    else:
                        self.reply_json(404, {"error": "Unavailable in this console."})
                except PermissionError as exc:
                    self.reply_json(403, {"error": str(exc)})
                except (ValueError, KeyError, TypeError) as exc:
                    self.reply_json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self.reply_json(409, {"error": str(exc)})
                return
            if self.path in ("/api/ik", "/api/jog", "/api/manual"):
                is_jog = self.path == "/api/jog"
                is_manual = self.path == "/api/manual"
                sender = live_jog if is_jog else None
                if is_manual and live_action is None:
                    self.reply_json(409, {"error": "Homing and gripper controls require the live SDK console."})
                    return
                try:
                    if sender is not None or is_manual:
                        self.require_local_json()
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 262_144:
                        raise ValueError("Expected a request under 256 KiB")
                    request = json.loads(self.rfile.read(length))
                    if not isinstance(request, dict):
                        raise ValueError("Expected a JSON object")
                    if is_manual:
                        result = {"message": live_action(request["arm"], request["action"])}
                    elif is_jog:
                        arm, axis, direction, step = (request[key] for key in ("arm", "axis", "direction", "step"))
                        if sender is not None:
                            joints = sender(arm, axis, direction, step)
                        else:
                            joints = jog_target(
                                arm_chain(request["urdf"], arm), request["measured"], axis, direction, step
                            )
                        result = {"joints": joints, "sent": sender is not None}
                    else:
                        result = preview_kinematics(
                            request["urdf"], request["arm"], request["measured"], request["target"]
                        )
                    status = 200
                except PermissionError as exc:
                    result = {"error": str(exc)}
                    status = 403
                except (AttributeError, KeyError, TypeError, ValueError, ET.ParseError, OverflowError) as exc:
                    error = str(exc) if (is_jog or is_manual) and isinstance(exc, ValueError) else "Invalid manual or kinematics request."
                    result = {"error": error}
                    status = 400
                except (RuntimeError, OSError) as exc:
                    result = {"error": str(exc)}
                    status = 409
                self.reply_json(status, result)
                return
            if self.path == "/api/prompt":
                if not prompts_enabled:
                    self.reply_json(409, {"error": "Zero-shot prompts require the Claude policy console."})
                    return
                if key_planner is not None:
                    try:
                        self.require_local_json()
                    except PermissionError as exc:
                        self.reply_json(403, {"error": str(exc)})
                        return
                    if not key_planner.api_key:
                        self.reply_json(409, {"error": "Verify the Anthropic key before arming a prompt."})
                        return
                ln = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(ln) if ln > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                ver = prompt_ctrl.set_prompt(payload.get("prompt"))
                out = json.dumps({"ok": True, "prompt_version": ver, "prompt": payload.get("prompt")}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(404)
            self.end_headers()

    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


async def run_server(
    host: str,
    port: int,
    engine: DualArmPolicyEngine,
    *,
    key_planner: ClaudeEndPosePlanner | None = None,
    robot_ip: str | None = None,
) -> None:
    async def connection(conn: Any) -> None:
        if robot_ip and (conn.remote_address is None or conn.remote_address[0] != robot_ip):
            await conn.close(code=1008, reason="This policy is reserved for the assigned robot")
            return
        if key_planner is not None and not (
            key_planner.api_key and engine.prompt_ctrl.snapshot()["active_prompt"]
        ):
            await conn.close(code=1008, reason="Verify the Anthropic key and arm a prompt in the laptop console")
            return
        await serve_connection(conn, engine)

    LOGGER.info("Starting public_v2 policy server on ws://%s:%d", host, port)
    async with serve(
        connection,
        host,
        port,
        ping_interval=None,
        compression=None,
        max_size=MAX_FRAME_BYTES,
    ):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Autonomous Dual-Arm Policy Server (public_v2)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--console-port", type=int, default=8002)
    parser.add_argument("--console-host", default=None, help="HTTP console bind address (defaults to --host)")
    parser.add_argument("--local-key-setup", action="store_true", help="Verify a Claude key and arm a prompt on localhost")
    parser.add_argument("--robot-ip", help="Accept policy sessions only from this robot IP")
    parser.add_argument("--mode", choices=("claude", "offline"), default="claude")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--plan-every", type=int, default=4, help="Maximum observation chunks between Claude plans")
    parser.add_argument("--max-step-m", type=float, default=0.08)
    parser.add_argument("--max-step-rad", type=float, default=0.45)
    parser.add_argument("--log-file", default="logs/chunks.jsonl")
    args = parser.parse_args()
    console_host = args.console_host or args.host
    if args.local_key_setup and (args.mode != "claude" or console_host != "127.0.0.1" or args.console_port <= 0):
        parser.error("--local-key-setup requires Claude mode and an enabled 127.0.0.1 console")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctrl = PromptController(initial_prompt=args.prompt or None)

    planner: ClaudeEndPosePlanner | HeuristicOfflinePlanner
    if args.mode == "offline":
        planner = HeuristicOfflinePlanner()
    else:
        planner = ClaudeEndPosePlanner(model=args.model or None, plan_every=args.plan_every)

    key_planner = planner if args.local_key_setup and isinstance(planner, ClaudeEndPosePlanner) else None
    if args.console_port > 0:
        start_console_http_server(console_host, args.console_port, ctrl, key_planner=key_planner)
    limits = WorkspaceLimits(max_step_pos_m=args.max_step_m, max_step_rot_rad=args.max_step_rad)
    engine = DualArmPolicyEngine(planner=planner, limits=limits, log_path=Path(args.log_file), prompt_ctrl=ctrl)
    asyncio.run(run_server(args.host, args.port, engine, key_planner=key_planner, robot_ip=args.robot_ip))


if __name__ == "__main__":
    main()
