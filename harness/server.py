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
from collections import deque
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

    def handle_payload(self, raw_payload: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            req = validate_request(raw_payload, max_image_dim=self.max_image_dim)
            effective_goal, pver = self.prompt_ctrl.get_effective_instruction(req.instruction)
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
        engine.prompt_ctrl.set_connected(False)


CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quanta X1 Pro - Claude Autonomous Prompt Console</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; background: #f8f9fb; color: #1a1d24; }
  .wrap { max-width: 1020px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 22px; }
  .sub { color: #5b6475; margin: 0 0 18px; font-size: 14px; }
  .card { background: #fff; border: 1px solid #e2e6ee; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; }
  input[type=text] { flex: 1; min-width: 240px; padding: 10px 12px; font-size: 15px; border: 1px solid #cfd6e4; border-radius: 8px; }
  button { padding: 10px 16px; font-size: 14px; font-weight: 600; border: 0; border-radius: 8px; cursor: pointer; background: #4f44e1; color: #fff; }
  button.sec { background: #eef1f7; color: #1a1d24; }
  .cams { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
  .cam img { width: 100%; border-radius: 8px; border: 1px solid #e2e6ee; background: #111; min-height: 150px; object-fit: contain; }
  .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; font-size: 14px; }
  .kv b { display: block; color: #5b6475; font-size: 12px; text-transform: uppercase; }
  .log { font-family: ui-monospace, monospace; font-size: 12px; max-height: 220px; overflow: auto; background: #f3f5f9; padding: 10px; border-radius: 8px; }
  .warn { background: #fff4e5; border: 1px solid #f5c26b; padding: 10px 12px; border-radius: 8px; font-size: 13px; margin-bottom: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Quanta X1 Pro — Claude Autonomous Console</h1>
  <p class="sub">Enter a natural-language task prompt below. The running robot session switches immediately on the next observation chunk.</p>
  <div class="warn">Hardware safety: Stopping, homing, and scoring always happen in the robot WebUI (<b>End Evaluation Inference Mode</b>).</div>
  <div class="card">
    <form id="pform" class="row">
      <input id="prompt" type="text" placeholder="e.g. Pick up the spoon and place it into the blue bowl." />
      <button type="submit">Run Prompt</button>
      <button type="button" class="sec" onclick="sendPrompt('Hold current pose safely.')">Hold Pose</button>
      <button type="button" class="sec" onclick="sendPrompt('')">Use WebUI Case</button>
    </form>
  </div>
  <div class="card meta">
    <div class="kv"><b>Connection</b><span id="conn">-</span></div>
    <div class="kv"><b>Status / Stage</b><span id="st">-</span></div>
    <div class="kv"><b>Chunk / Latency</b><span id="lat">-</span></div>
    <div class="kv"><b>Active Goal</b><span id="goal">-</span></div>
  </div>
  <div class="card">
    <b>Claude Reasoning</b>
    <p id="reason" style="margin:6px 0 0">-</p>
    <p id="err" style="margin:6px 0 0;color:#c0392b"></p>
  </div>
  <div class="card cams">
    <div class="cam"><b>Front (Right arm on left)</b><img id="img_front" alt="front"></div>
    <div class="cam"><b>Left Wrist</b><img id="img_left_wrist" alt="left_wrist"></div>
    <div class="cam"><b>Right Wrist</b><img id="img_right_wrist" alt="right_wrist"></div>
  </div>
  <div class="card">
    <b>Recent Steps</b>
    <div id="log" class="log">Waiting for observations...</div>
  </div>
</div>
<script>
async function sendPrompt(val) {
  await fetch('/api/prompt', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt: val}) });
  if (val) document.getElementById('prompt').value = val;
  poll();
}
document.getElementById('pform').onsubmit = (e) => { e.preventDefault(); sendPrompt(document.getElementById('prompt').value); };
async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('conn').textContent = d.connected ? 'Robot Connected' : 'Waiting for WebUI Start';
    document.getElementById('st').textContent = (d.status || '-') + ' / ' + (d.stage || '-');
    document.getElementById('lat').textContent = (d.chunk_index || 0) + ' (' + (d.latency_ms ?? '-') + ' ms)';
    document.getElementById('goal').textContent = d.effective_instruction || '(from WebUI case)';
    document.getElementById('reason').textContent = d.reason || '-';
    document.getElementById('err').textContent = d.last_error || '';
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


def start_console_http_server(host: str, port: int, prompt_ctrl: PromptController) -> http.server.ThreadingHTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/api/status":
                body = json.dumps(prompt_ctrl.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = CONSOLE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path == "/api/prompt":
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


async def run_server(host: str, port: int, engine: DualArmPolicyEngine) -> None:
    LOGGER.info("Starting public_v2 policy server on ws://%s:%d", host, port)
    async with serve(
        lambda conn: serve_connection(conn, engine),
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
    parser.add_argument("--mode", choices=("claude", "offline"), default="claude")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--plan-every", type=int, default=1)
    parser.add_argument("--max-step-m", type=float, default=0.08)
    parser.add_argument("--max-step-rad", type=float, default=0.45)
    parser.add_argument("--log-file", default="logs/chunks.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctrl = PromptController(initial_prompt=args.prompt or None)
    if args.console_port > 0:
        start_console_http_server(args.host, args.console_port, ctrl)

    planner: ClaudeEndPosePlanner | HeuristicOfflinePlanner
    if args.mode == "offline":
        planner = HeuristicOfflinePlanner()
    else:
        planner = ClaudeEndPosePlanner(model=args.model or None, plan_every=args.plan_every)

    limits = WorkspaceLimits(max_step_pos_m=args.max_step_m, max_step_rot_rad=args.max_step_rad)
    engine = DualArmPolicyEngine(planner=planner, limits=limits, log_path=Path(args.log_file), prompt_ctrl=ctrl)
    asyncio.run(run_server(args.host, args.port, engine))


if __name__ == "__main__":
    main()
