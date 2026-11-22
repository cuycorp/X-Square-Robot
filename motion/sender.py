"""LIVE execution of already-checked plans. This is the only module that sends commands to the robot.

Backend "sdk" (default): the official X2Robot SDK (v1.1.2, gRPC :50051) through motion/sdk_bridge.py running in
.venv-sdk. Joint targets from safety.SafetyPlanner go to <arm>.set_joint_positions, interpolated in steps of at
most max_joint_step_deg at interp_hz (gentle even if the target is right).
Backend "ros_pose_cmd": publish /{side}_arm_cartesian_controller/pose_cmd via rosbridge (only works when the
platform is not commanding the arm itself; convention verified from recordings).

Refuses or stops when:
  - the server was not started with --live, or the convention / backend is not usable;
  - SDK backend: the robot is not in SDK mode, or its E-stop is active;
  - ROS backend: another program sent arm commands in the last second;
  - the plan is older than max_plan_age_s, or the arm moved since planning (> max_start_drift_deg);
  - setting the control mode makes the arm move by itself;
  - the arm does not reach a step within step_timeout_s, or the external force jumps.
Stopping = command the arm to hold its measured pose. The red E-stop on the robot is always the emergency stop.
"""
import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import kinematics as K
import safety

CONVENTIONS = ("relative_to_neutral", "arm_base")
ROOT = Path(__file__).resolve().parent.parent
SDK_PYTHON = ROOT / ".venv-sdk" / "bin" / "python"
BRIDGE = Path(__file__).resolve().parent / "sdk_bridge.py"


def pose_message(planner: safety.SafetyPlanner, side: str, q, convention: str) -> dict:
    """PoseStamped for /{side}_arm_cartesian_controller/pose_cmd that puts link6 at FK(q) (ROS backend)."""
    if convention not in CONVENTIONS:
        raise RuntimeError("pose_cmd convention is not verified, so refusing to send")
    T = planner.arms[side].fk(np.asarray(q, float))
    if convention == "relative_to_neutral":
        T = np.linalg.inv(planner.neutral[side]) @ T
    quat = K.matrix_to_quat(T[:3, :3])
    now = time.time()
    # frame_id is empty, exactly as the platform's own commands are (seen in the 2026-11-05 recordings).
    return {"header": {"stamp": {"sec": int(now), "nanosec": int((now % 1) * 1e9)}, "frame_id": ""},
            "pose": {"position": dict(zip("xyz", map(float, T[:3, 3]))), "orientation": dict(zip("xyzw", map(float, quat)))}}


def joint_path(q0, q1, max_step_rad: float) -> list:
    """Intermediate joint targets from q0 (exclusive) to q1 (inclusive), no joint moving more than max_step_rad."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(1, math.ceil(float(np.max(np.abs(q1 - q0))) / max_step_rad))
    return [q0 + (i / n) * (q1 - q0) for i in range(1, n + 1)]


class SdkBridge:
    """Client for motion/sdk_bridge.py (one persistent subprocess, JSON lines)."""

    def __init__(self, address: str):
        if not SDK_PYTHON.exists():
            raise RuntimeError(".venv-sdk with the X2Robot SDK v1.1.2 is not installed")
        self.proc = subprocess.Popen([str(SDK_PYTHON), str(BRIDGE), address], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.stdin, self.stdout = self.proc.stdin, self.proc.stdout
        self._lock = threading.Lock()
        # A reader thread feeds JSON replies into a queue. (select() on the pipe is wrong here: readline()
        # buffers several lines at once, so select would wait for data that already arrived.)
        self._replies = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        ready = self._read(20.0)
        if not ready.get("ready"):
            raise RuntimeError(f"SDK bridge did not start: {ready}")

    def _reader(self):
        for line in self.stdout:
            try:
                self._replies.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # the SDK prints its own non-JSON log lines
        self._replies.put(None)  # bridge exited

    def _read(self, timeout: float) -> dict:
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("SDK bridge timed out") from None
        if reply is None:
            raise RuntimeError("SDK bridge exited")
        return reply

    def request(self, req: dict, timeout: float = 5.0) -> dict:
        with self._lock:
            self.stdin.write(json.dumps(req) + "\n")
            self.stdin.flush()
            return self._read(timeout)

    def close(self):
        try:
            self.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class LiveSender:
    def __init__(self, url: str, planner: safety.SafetyPlanner, live_enabled: bool, sdk_address: str = ""):
        self.url, self.planner, self.live_enabled = url, planner, live_enabled
        self.cfg = planner.cfg.get("live", {})
        self.backend = self.cfg.get("backend", "sdk")
        self.sdk_address = sdk_address or self.cfg.get("sdk_address", "x2://192.168.77.2:50051")
        self.bridge = None
        self.status = {"state": "idle", "step": 0, "total": 0, "message": "", "log": []}
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---------- gates ----------

    def blockers(self) -> list:
        out = []
        if not self.live_enabled:
            out.append("server not started with --live")
        if self.backend == "sdk":
            if not SDK_PYTHON.exists():
                out.append(".venv-sdk (X2Robot SDK v1.1.2) not installed")
        elif self.backend == "ros_pose_cmd":
            if self.cfg.get("pose_cmd_convention") not in CONVENTIONS:
                out.append("pose_cmd_convention is 'unverified' in safety_config.json")
        else:
            out.append(f"unknown backend {self.backend!r}")
        return out

    def _log(self, msg: str):
        with self._lock:
            self.status["message"] = msg
            self.status["log"] = (self.status["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-40:]

    def _sdk(self) -> SdkBridge:
        if self.bridge is None or self.bridge.proc.poll() is not None:
            self.bridge = SdkBridge(self.sdk_address)
        return self.bridge

    def sdk_request(self, req: dict, timeout: float = 5.0) -> dict:
        return self._sdk().request(req, timeout)

    # ---------- execution ----------

    def start(self, plan: dict, hub, read_state) -> dict:
        """plan: {"created": monotonic, "steps": [{"arm", "q", "gripper", "phase"}], "start_q": {side: q}}."""
        if self.status["state"] == "running":
            raise RuntimeError("already executing a plan")
        if b := self.blockers():
            raise RuntimeError("live sending is locked: " + "; ".join(b))
        if time.monotonic() - plan["created"] > self.cfg.get("max_plan_age_s", 60):
            raise RuntimeError("plan is too old; plan again")
        state = read_state(hub)
        drift = max(float(np.degrees(np.max(np.abs(np.asarray(state[s]["q"]) - np.asarray(q)))))
                    for s, q in plan["start_q"].items())
        if drift > self.cfg.get("max_start_drift_deg", 1.0):
            raise RuntimeError(f"the arm moved {drift:.1f}° since planning; plan again")
        self._stop.clear()
        with self._lock:
            self.status = {"state": "running", "step": 0, "total": len(plan["steps"]), "message": "starting", "log": []}
        threading.Thread(target=self._run, args=(plan, hub, read_state), daemon=True).start()
        return self.status

    def stop(self):
        self._stop.set()

    def _run(self, plan, hub, read_state):
        sides = sorted({st["arm"] for st in plan["steps"]})
        try:
            if self.backend == "sdk":
                self._prepare_sdk(sides, hub, read_state)
                send_move, send_gripper = self._move_sdk, self._gripper_sdk
                ws = None
            else:
                from websockets.sync.client import connect
                ws = connect(self.url, open_timeout=5, max_size=None)
                self._preflight_ros(ws, sides)
                for s in sides:
                    ws.send(json.dumps({"op": "advertise", "id": f"adv-{s}", "topic": f"/{s}_arm_cartesian_controller/pose_cmd",
                                        "type": "geometry_msgs/msg/PoseStamped"}))
                    ws.send(json.dumps({"op": "advertise", "id": f"advg-{s}", "topic": f"/{s}_gripper_controller/commands",
                                        "type": "std_msgs/msg/Float64MultiArray"}))
                send_move = lambda h, rs, st, b: self._move_ros(ws, h, rs, st, b)       # noqa: E731
                send_gripper = lambda h, rs, st, b: self._gripper_ros(ws, h, rs, st, b)  # noqa: E731
            baseline = {s: v["force"] for s, v in read_state(hub).items() if v["force"] is not None}
            for i, st in enumerate(plan["steps"], 1):
                if self._stop.is_set():
                    raise RuntimeError("stopped by user")
                with self._lock:
                    self.status["step"] = i
                self._log(f"step {i}/{len(plan['steps'])}: {st['phase']} ({st['arm']} arm)")
                if st.get("gripper") is not None:
                    send_gripper(hub, read_state, st, baseline)
                else:
                    send_move(hub, read_state, st, baseline)
            self._log("plan finished")
            with self._lock:
                self.status["state"] = "done"
        except Exception as e:  # noqa: BLE001 - every failure ends in a hold + visible status
            self._log(f"STOPPED: {e}")
            self._hold(hub, read_state, sides)
            with self._lock:
                self.status["state"] = "stopped"

    # ---------- SDK backend ----------

    def _prepare_sdk(self, sides, hub, read_state):
        st = self.sdk_request({"cmd": "status"})
        if st.get("operation_mode") != "sdk":
            raise RuntimeError(f"robot is not in SDK mode (it is in {st.get('operation_mode')!r}); enter SDK mode first")
        if st.get("estop"):
            raise RuntimeError("the robot's emergency stop is active")
        if st.get("control_mode") != "joint_positions":
            before = {s: np.asarray(read_state(hub)[s]["q"]) for s in sides}
            res = self.sdk_request({"cmd": "set_mode", "mode": "joint_positions"}, timeout=10.0)
            if not res.get("ok"):
                raise RuntimeError(f"could not set joint-position control: {res.get('error')}")
            self._log("control mode set to joint positions; checking the arm stays still")
            time.sleep(0.8)
            after = read_state(hub)
            moved = max(float(np.degrees(np.max(np.abs(np.asarray(after[s]["q"]) - before[s])))) for s in sides)
            if moved > self.cfg.get("max_start_drift_deg", 1.0):
                raise RuntimeError(f"the arm moved {moved:.1f}° by itself after the mode change")

    def _move_sdk(self, hub, read_state, st, baseline):
        side, q_t = st["arm"], np.asarray(st["q"])
        q0 = np.asarray(read_state(hub)[side]["q"])
        period = 1.0 / max(1, self.cfg.get("interp_hz", 50))
        for k, q in enumerate(joint_path(q0, q_t, math.radians(self.cfg.get("max_joint_step_deg", 0.5)))):
            if self._stop.is_set():
                raise RuntimeError("stopped by user")
            res = self.sdk_request({"cmd": "joints", "arm": side, "q": [float(v) for v in q]})
            if not res.get("ok"):
                raise RuntimeError(f"SDK refused the joint target: {res.get('error') or res.get('code')}")
            if k % 5 == 0:
                self._check_force(read_state, hub, baseline)
            time.sleep(period)
        self._wait_arrival(hub, read_state, side, q_t, baseline,
                           resend=lambda: self.sdk_request({"cmd": "joints", "arm": side, "q": [float(v) for v in q_t]}))

    def _gripper_sdk(self, hub, read_state, st, baseline):
        res = self.sdk_request({"cmd": "gripper", "arm": st["arm"], "value": float(st["gripper"])})
        if not res.get("ok"):
            raise RuntimeError(f"SDK refused the gripper command: {res.get('error') or res.get('code')}")
        self._wait_gripper(hub, read_state, st, baseline, resend=None)

    # ---------- ROS backend ----------

    def _preflight_ros(self, ws, sides):
        """Refuse if anything else is commanding these arms right now."""
        topics = [f"/{s}_arm_cartesian_controller/pose_cmd" for s in sides] + [f"/{s}_gripper_controller/commands" for s in sides]
        for t in topics:
            ws.send(json.dumps({"op": "subscribe", "id": f"pf{t}", "topic": t}))
        seen, end = 0, time.monotonic() + self.cfg.get("foreign_traffic_window_s", 1.0)
        while time.monotonic() < end:
            try:
                m = json.loads(ws.recv(timeout=0.2))
            except TimeoutError:
                continue
            seen += m.get("op") == "publish"
        for t in topics:
            ws.send(json.dumps({"op": "unsubscribe", "id": f"pf{t}", "topic": t}))
        if seen:
            raise RuntimeError(f"another program sent {seen} arm commands in the last second; not taking control")

    def _move_ros(self, ws, hub, read_state, st, baseline):
        side, q_t = st["arm"], np.asarray(st["q"])
        msg = pose_message(self.planner, side, q_t, self.cfg["pose_cmd_convention"])
        topic = f"/{side}_arm_cartesian_controller/pose_cmd"
        self._wait_arrival(hub, read_state, side, q_t, baseline,
                           resend=lambda: ws.send(json.dumps({"op": "publish", "topic": topic, "msg": msg})))

    def _gripper_ros(self, ws, hub, read_state, st, baseline):
        topic = f"/{st['arm']}_gripper_controller/commands"
        self._wait_gripper(hub, read_state, st, baseline,
                           resend=lambda: ws.send(json.dumps({"op": "publish", "topic": topic, "msg": {"data": [float(st["gripper"])]}})))

    # ---------- shared ----------

    def _check_force(self, read_state, hub, baseline):
        st = read_state(hub)
        alerts = safety.contact_alerts({s: v["force"] for s, v in st.items() if v["force"] is not None},
                                       baseline, self.planner.cfg["contact_force_delta_n"])
        if alerts:
            raise RuntimeError("unexpected force: " + "; ".join(alerts))
        return st

    def _wait_arrival(self, hub, read_state, side, q_t, baseline, resend):
        T_t = self.planner.arms[side].fk(q_t)
        period = 1.0 / max(1, self.cfg.get("republish_hz", 20))
        end = time.monotonic() + self.cfg.get("step_timeout_s", 4.0)
        while True:
            if self._stop.is_set():
                raise RuntimeError("stopped by user")
            if resend:
                resend()
            state = self._check_force(read_state, hub, baseline)
            T_now = self.planner.arms[side].fk(np.asarray(state[side]["q"]))
            pos_err = float(np.linalg.norm(T_now[:3, 3] - T_t[:3, 3]))
            rot_err = float(np.degrees(np.linalg.norm(K.rotation_error(T_t[:3, :3], T_now[:3, :3]))))
            if pos_err <= self.cfg.get("arrival_tolerance_m", 0.004) and rot_err <= self.cfg.get("arrival_tolerance_deg", 2.0):
                return
            if time.monotonic() > end:
                raise RuntimeError(f"{side} arm did not reach step (still {1000 * pos_err:.0f} mm / {rot_err:.1f}° away)")
            time.sleep(period)

    def _wait_gripper(self, hub, read_state, st, baseline, resend):
        side, target = st["arm"], float(st["gripper"])
        end = time.monotonic() + self.cfg.get("gripper_timeout_s", 3.0)
        last, still = None, 0
        while time.monotonic() < end:
            if self._stop.is_set():
                raise RuntimeError("stopped by user")
            if resend:
                resend()
            pos = self._check_force(read_state, hub, baseline)[side]["gripper"]
            if abs(pos - target) < 0.05:
                return
            still = still + 1 if last is not None and abs(pos - last) < 0.002 else 0
            if still >= 5 and target < 0.1:   # closing stopped early: it is holding something
                self._log(f"gripper stopped at {pos:.3f} (holding an object)")
                return
            last = pos
            time.sleep(0.1)
        raise RuntimeError(f"{side} gripper did not reach {target:.2f}")

    def _hold(self, hub, read_state, sides):
        """Command each arm to stay where it is now."""
        try:
            state = read_state(hub)
            if self.backend == "sdk":
                for s in sides:
                    self.sdk_request({"cmd": "joints", "arm": s, "q": [float(v) for v in state[s]["q"]]})
            else:
                from websockets.sync.client import connect
                with connect(self.url, open_timeout=3, max_size=None) as ws:
                    for s in sides:
                        ws.send(json.dumps({"op": "publish", "topic": f"/{s}_arm_cartesian_controller/pose_cmd",
                                            "msg": pose_message(self.planner, s, state[s]["q"], self.cfg["pose_cmd_convention"])}))
            self._log("holding the current pose")
        except Exception as e:  # noqa: BLE001
            self._log(f"could not send hold ({e}); use the E-stop if the arm is still moving")


