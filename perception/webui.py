#!/usr/bin/env python3
"""Local web UI: live robot camera + ask a Nebius vision model about it. Read-only: never commands the robot.

  .venv/bin/python perception/webui.py          # then open http://127.0.0.1:8080
"""
import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.sync.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "motion"))

import camera  # noqa: E402
import nebius_vision  # noqa: E402
import prompts  # noqa: E402
import service as motion_service  # noqa: E402  (dry-run planning; never publishes)

STATIC = Path(__file__).resolve().parent / "static"
STREAM_FPS = 4          # rosbridge throttle and browser frame rate. At 8 fps the head JPEG stream starved
                        # rosbridge (joint states dropped from ~12 Hz to <1 Hz), so keep this low.
CAMERA_IDLE_S = 15      # unsubscribe a camera nobody has used for this long
CAMERA_SILENT_S = 5     # re-subscribe a wanted camera that has sent nothing for this long
MAX_FRAME_AGE_S = 3.0   # refuse to ask about a stale frame
MAX_QUESTION_CHARS = 500


class FrameHub:
    """One read-only rosbridge connection: latest JPEG per camera (subscribed on first use) and the latest
    message of each state topic (joint states, grippers, wrench). Only ever subscribes."""

    def __init__(self, url: str = camera.ROSBRIDGE_URL, state_topics: dict | None = None):
        self.url = url
        self.frames = {}            # name -> (jpeg_bytes, monotonic_time, seq)
        self.wanted = set()         # cameras subscribed on demand and dropped when idle
        self.last_use = {}          # camera -> monotonic time it was last asked for
        self.state_topics = state_topics or {}
        self.states = {}            # name -> (msg_dict, monotonic_time)
        self.connected = False
        self.last_error = ""
        self._cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def state(self, name: str):
        with self._cond:
            return self.states.get(name)

    def want(self, name: str):
        if name not in camera.CAMERAS:
            raise KeyError(name)
        with self._cond:
            self.wanted.add(name)
            self.last_use[name] = time.monotonic()

    def latest(self, name: str):
        with self._cond:
            self.last_use[name] = time.monotonic()
            self.wanted.add(name)   # re-subscribe if it was dropped as idle
            return self.frames.get(name)

    def wait_newer(self, name: str, seq: int, timeout: float = 2.0):
        with self._cond:
            self.last_use[name] = time.monotonic()
            self.wanted.add(name)
            self._cond.wait_for(lambda: self.frames.get(name, (0, 0, 0))[2] > seq, timeout)
            return self.frames.get(name)

    def _run(self):
        backoff = 1.0
        while True:
            subscribed = set()
            try:
                with connect(self.url, open_timeout=5, max_size=None) as ws:
                    self.connected, self.last_error, backoff = True, "", 1.0
                    for name, (topic, typ) in self.state_topics.items():
                        ws.send(json.dumps({"op": "subscribe", "id": f"st-{name}", "topic": topic, "type": typ,
                                            "throttle_rate": 100, "queue_length": 1}))
                    by_topic = {topic: name for name, (topic, _) in self.state_topics.items()}
                    sub_time = {}
                    while True:
                        with self._cond:
                            now = time.monotonic()
                            idle = {n for n in subscribed if now - self.last_use.get(n, 0) > CAMERA_IDLE_S}
                            self.wanted -= idle
                            # Watchdog: a wanted camera silent for CAMERA_SILENT_S gets re-subscribed (seen live:
                            # the subscription went quiet for minutes while the camera itself kept publishing).
                            silent = {n for n in subscribed - idle
                                      if now - max(self.frames.get(n, (0, 0.0, 0))[1], sub_time.get(n, 0.0)) > CAMERA_SILENT_S}
                            todo = (self.wanted - subscribed) | silent
                        for name in idle | silent:
                            ws.send(json.dumps({"op": "unsubscribe", "id": f"ui-{name}", "topic": camera.CAMERAS[name][0]}))
                            subscribed.discard(name)
                        for name in todo:
                            ws.send(json.dumps({"op": "subscribe", "id": f"ui-{name}", "topic": camera.CAMERAS[name][0],
                                                "type": "sensor_msgs/msg/CompressedImage",
                                                "throttle_rate": int(1000 / STREAM_FPS), "queue_length": 1}))
                            subscribed.add(name)
                            sub_time[name] = time.monotonic()
                        try:
                            msg = json.loads(ws.recv(timeout=0.5))
                        except TimeoutError:
                            continue
                        if msg.get("op") != "publish":
                            continue
                        if msg.get("topic") in by_topic:
                            with self._cond:
                                self.states[by_topic[msg["topic"]]] = (msg["msg"], time.monotonic())
                            continue
                        name = next((n for n, (t, _) in camera.CAMERAS.items() if t == msg.get("topic")), None)
                        if name is None:
                            continue
                        jpeg = camera.decode_compressed_image(msg["msg"])
                        with self._cond:
                            seq = self.frames.get(name, (0, 0, 0))[2] + 1
                            self.frames[name] = (jpeg, time.monotonic(), seq)
                            self._cond.notify_all()
            except Exception as e:  # noqa: BLE001 - keep retrying; the UI shows the error
                self.connected, self.last_error = False, f"{type(e).__name__}: {e}"[:200]
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)


def ask_about(hub, question: str, cameras: list, think: bool, model: str) -> dict:
    """Answer a question using the latest frame of each camera. Returns a JSON-able dict."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    cameras = [c for c in (cameras or ["head"]) if c in camera.CAMERAS] or ["head"]
    frames = {}
    for name in cameras:
        hub.want(name)
        f = hub.latest(name) or hub.wait_newer(name, 0, timeout=MAX_FRAME_AGE_S)
        if not f or time.monotonic() - f[1] > MAX_FRAME_AGE_S:
            raise RuntimeError(f"No fresh frame from the {name} camera. Is the robot connected?")
        frames[name] = f[0]
    labelled = {camera.CAMERAS[n][1]: jpeg for n, jpeg in frames.items()}
    t0 = time.monotonic()
    result = nebius_vision.ask(labelled, question, model=model, system=prompts.SYSTEM, think=think)
    return {
        "answer": result["text"],
        "model": result["model"],
        "model_ms": round(1000 * (time.monotonic() - t0)),
        "usage": result.get("usage", {}),
        "cameras": cameras,
        "frames": {n: base64.b64encode(j).decode("ascii") for n, j in frames.items()},
    }


def make_handler(hub, model: str, motion=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotEyes/1.0"

        def log_message(self, format, *args):  # noqa: A002 - quiet: no per-frame request logs
            del format, args

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/architecture"):
                page = "architecture.html" if path == "/architecture" else "index.html"
                body = (STATIC / page).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                now = time.monotonic()
                ages = {n: round(now - f[1], 2) for n, f in list(hub.frames.items())}
                self._json(200, {"connected": hub.connected, "error": hub.last_error, "frame_age_s": ages,
                                 "model": model,
                                 "cameras": {n: d for n, (_, d) in camera.CAMERAS.items()}})
            elif path.startswith("/stream/"):
                self._stream(path.removeprefix("/stream/"))
            elif path == "/exec_status":
                if motion is None:
                    return self._json(503, {"error": "motion planning is not available"})
                self._json(200, motion.exec_status())
            elif path == "/robot_mode":
                if motion is None:
                    return self._json(503, {"error": "motion planning is not available"})
                try:
                    self._json(200, motion.robot_mode())
                except Exception as e:  # noqa: BLE001
                    self._json(503, {"error": str(e)[:200]})
            elif path == "/arm_state":
                if motion is None:
                    return self._json(503, {"error": "motion planning is not available"})
                try:
                    self._json(200, motion.arm_state(hub))
                except RuntimeError as e:
                    self._json(503, {"error": str(e)})
            else:
                self._json(404, {"error": "not found"})

        def _stream(self, name: str):
            try:
                hub.want(name)
            except KeyError:
                return self._json(404, {"error": f"unknown camera {name}"})
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while True:
                    f = hub.wait_newer(name, seq, timeout=2.0)
                    if not f or f[2] == seq:
                        continue
                    jpeg, _, seq = f
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser switched camera or closed the tab

        def do_POST(self):
            if self.path not in ("/ask", "/plan_move", "/reset_force_baseline", "/execute", "/stop",
                                 "/sdk_mode", "/robot_estop"):
                return self._json(404, {"error": "not found"})
            if self.path != "/ask" and motion is None:
                return self._json(503, {"error": "motion planning is not available"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(min(n, 10_000)) or b"{}")
                if self.path == "/ask":
                    out = ask_about(hub, req.get("question", ""), req.get("cameras"), bool(req.get("think")), model)
                elif self.path == "/stop":
                    assert motion is not None
                    out = motion.stop()
                elif self.path == "/sdk_mode":
                    assert motion is not None
                    if not req.get("confirmed"):
                        raise ValueError("mode change must be confirmed")
                    out = motion.set_sdk_mode(bool(req.get("on")))
                elif self.path == "/robot_estop":
                    assert motion is not None
                    out = motion.robot_estop(recover=bool(req.get("recover")))
                elif self.path == "/execute":
                    assert motion is not None
                    if not req.get("confirmed"):
                        raise ValueError("execution must be confirmed")
                    out = motion.execute(hub, str(req.get("plan_id", "")))
                elif self.path == "/plan_move":
                    assert motion is not None
                    out = motion.plan_command(hub, req.get("command", ""))
                else:
                    assert motion is not None
                    motion.reset_force_baseline(hub)
                    out = {"ok": True}
                self._json(200, out)
            except (ValueError, RuntimeError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"[:300]})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="default: this laptop only (protects your credits)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=nebius_vision.DEFAULT_MODEL)
    ap.add_argument("--live", action="store_true",
                    help="allow confirmed, fully checked plans to be SENT to the robot (needs a verified pose_cmd convention)")
    a = ap.parse_args()
    nebius_vision.load_key()  # fail fast if the key is missing (never printed)
    hub = FrameHub(state_topics=motion_service.STATE_TOPICS)
    try:
        motion = motion_service.MotionService(motion_service.load_urdf(hub.url), a.model, rosbridge_url=hub.url,
                                              live=a.live)
        if a.live:
            print("LIVE MODE: confirmed plans can move the robot. Keep a hand on the red E-stop.", flush=True)
    except Exception as e:  # noqa: BLE001 - the camera/ask features still work without it
        print(f"Motion planning disabled: {e}", flush=True)
        motion = None
    try:
        server = ThreadingHTTPServer((a.host, a.port), make_handler(hub, a.model, motion))
    except OSError as e:
        if e.errno == 98:  # address already in use
            print(f"Robot Eyes is already running on port {a.port} (probably in another terminal).\n"
                  f"Stop it there with Ctrl+C, or run:  pkill -INT -f 'perception/webui.py'\n"
                  f"then start this command again.", flush=True)
            return 1
        raise
    server.daemon_threads = True
    print(f"Robot eyes UI: http://{a.host}:{a.port}  (rosbridge {hub.url}, model {a.model}). Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
