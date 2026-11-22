"""Glue between the web UI, live robot state, the command parser, the safety planner and the live sender.

Plans are always checked here. Only sender.LiveSender commands the arms (via the official SDK by default), and
only when the server runs with --live and the user confirms a stored, single-arm, fully checked plan. Robot mode
changes (SDK on/off) and the SDK E-stop also happen only on the user's click.
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np

import command_parser
import grasp
import kinematics as K
import nebius_vision
import safety
import sender
import vision3d

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "robot_description.urdf"
MAX_STATE_AGE_S = 1.0

# name -> (topic, type) read from rosbridge by the web UI's hub
STATE_TOPICS = {
    **{f"{s}_joints": (f"/{s}_arm/joint_states", "sensor_msgs/msg/JointState") for s in K.ARMS},
    **{f"{s}_gripper": (f"/{s}_gripper/joint_states", "sensor_msgs/msg/JointState") for s in K.ARMS},
    **{f"{s}_wrench": (f"/{s}_arm/wrench_ext_world", "geometry_msgs/msg/WrenchStamped") for s in K.ARMS},
}


def load_urdf(rosbridge_url: str, timeout: float = 5.0) -> str:
    """Fetch /robot_description (read-only) and cache it; fall back to the cache when the robot is offline."""
    try:
        from websockets.sync.client import connect
        with connect(rosbridge_url, open_timeout=timeout, max_size=None) as ws:
            ws.send(json.dumps({"op": "subscribe", "topic": "/robot_description", "type": "std_msgs/msg/String", "id": "urdf"}))
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                m = json.loads(ws.recv(timeout=max(0.1, end - time.monotonic())))
                if m.get("op") == "publish" and m.get("topic") == "/robot_description":
                    urdf = m["msg"]["data"]
                    CACHE.parent.mkdir(exist_ok=True)
                    CACHE.write_text(urdf)
                    return urdf
    except Exception:  # noqa: BLE001 - fall back to the cache below
        pass
    if CACHE.exists():
        return CACHE.read_text()
    raise RuntimeError("robot description unavailable: robot offline and no cached copy")


class MotionService:
    def __init__(self, urdf_xml: str, model: str, config: dict | None = None, rosbridge_url: str = "", live: bool = False):
        self.cfg = config or safety.load_config()
        self.planner = safety.SafetyPlanner(urdf_xml, self.cfg)
        self.model = model
        self.force_baseline = {}
        self.rosbridge_url = rosbridge_url
        self.cameras = None          # (color, depth, correction) — loaded lazily for picks
        self.urdf_xml = urdf_xml
        self.sender = sender.LiveSender(rosbridge_url, self.planner, live)
        self.plans = {}              # plan_id -> executable plan (only fully passing ones)

    # ---------- live execution ----------

    def _store_plan(self, steps: list, state: dict) -> dict:
        """Keep a passing plan for execution; returns what the UI needs to offer (or refuse) execution."""
        arms = sorted({s["arm"] for s in steps})
        if len(arms) != 1:
            return {"executable": False, "why_not": "live execution moves one arm per plan (the checks assume a "
                                                    "single moving arm); plan each arm separately"}
        plan_id = uuid.uuid4().hex[:12]
        self.plans = {plan_id: {"created": time.monotonic(), "steps": steps,
                                "start_q": {arms[0]: list(state[arms[0]]["q"])}}}   # only the newest plan is executable
        blockers = self.sender.blockers()
        return {"executable": not blockers, "plan_id": plan_id, "why_not": "; ".join(blockers), "steps": len(steps)}

    def execute(self, hub, plan_id: str) -> dict:
        plan = self.plans.get(plan_id)
        if plan is None:
            raise ValueError("unknown or replaced plan; plan again")
        return self.sender.start(plan, hub, self.read_state)

    def stop(self):
        self.sender.stop()
        return self.sender.status

    def exec_status(self) -> dict:
        return {**self.sender.status, "blockers": self.sender.blockers(), "backend": self.sender.backend,
                "convention": self.cfg.get("live", {}).get("pose_cmd_convention")}

    # ---------- robot mode (shared robot: only on the user's click) ----------

    TASK_NAMES = {0: "base", 1: "teleoperation", 3: "data collection", 4: "inference", 5: "SDK", 6: "homing",
                  8: "debug", 10: "RL", 11: "sleep"}

    def _rosbridge(self):
        from websockets.sync.client import connect
        return connect(self.rosbridge_url, open_timeout=5, max_size=None)

    def robot_mode(self) -> dict:
        """The SDK's own operation mode when live (that is what the arm controller obeys; SDK mode belongs to the
        connection that entered it, i.e. this server's bridge), else the platform task (read-only)."""
        if self.sender.live_enabled and not self.sender.blockers():
            try:
                st = self.sender.sdk_request({"cmd": "status"})
                name = st.get("operation_mode") or "unknown"
                return {"task_type": None, "name": "SDK" if name == "sdk" else name, "status": None,
                        "control_mode": st.get("control_mode"), "estop": st.get("estop")}
            except Exception:  # noqa: BLE001 - fall back to the platform reading
                pass
        with self._rosbridge() as ws:
            ws.send(json.dumps({"op": "subscribe", "id": "mode", "topic": "/application/task_info"}))
            end = time.monotonic() + 3
            while time.monotonic() < end:
                try:
                    m = json.loads(ws.recv(timeout=0.5))
                except TimeoutError:
                    continue
                if m.get("op") == "publish":
                    t = int(m["msg"].get("type", -1))
                    return {"task_type": t, "name": self.TASK_NAMES.get(t, f"type {t}"), "status": m["msg"].get("status")}
        return {"task_type": None, "name": "unknown", "status": None}

    def set_sdk_mode(self, on: bool) -> dict:
        """Enter/exit SDK mode through the SDK's own system.set_work_mode (the WebUI-style platform task alone
        leaves the SDK server saying "Sdk mode is not running"). Only with --live, only on a user click."""
        if not self.sender.live_enabled:
            raise RuntimeError("robot mode changes need the server started with --live")
        if self.sender.status.get("state") == "running":
            raise RuntimeError("a plan is executing; stop it first")
        res = self.sender.sdk_request({"cmd": "work_mode", "mode": "sdk" if on else "idle"}, timeout=20.0)
        if not res.get("ok"):
            raise RuntimeError(f"SDK refused: {res.get('error') or res.get('code')}")
        return {"ok": True, "mode": "SDK" if on else "idle"}

    def set_platform_task(self, on: bool) -> dict:
        """START/STOP the SDK task the way the robot's WebUI does (platform level). Kept for recovery."""
        if not self.sender.live_enabled:
            raise RuntimeError("robot mode changes need the server started with --live")
        with self._rosbridge() as ws:
            ws.send(json.dumps({"op": "call_service", "id": "sdkmode", "service": "/application/request/task_operation",
                                "args": {"operation_type": 1 if on else 2, "task_type": 5, "is_from_remote": False}}))
            end = time.monotonic() + 20
            while time.monotonic() < end:
                m = json.loads(ws.recv(timeout=max(0.1, end - time.monotonic())))
                if m.get("id") == "sdkmode":
                    v = m.get("values") or {}
                    if not v.get("res"):
                        raise RuntimeError(f"robot refused: {v.get('message') or v.get('error_code')}")
                    return {"ok": True, "mode": "SDK" if on else "base"}
        raise RuntimeError("robot did not answer the mode request")

    def robot_estop(self, recover: bool = False) -> dict:
        """The SDK's emergency stop (or its recovery). The physical red E-stop always works too."""
        self.sender.stop()
        res = self.sender.sdk_request({"cmd": "recover_estop" if recover else "estop"}, timeout=5.0)
        if not res.get("ok"):
            raise RuntimeError(f"SDK: {res.get('error') or res.get('code')}")
        return {"ok": True}

    def _head_cameras(self):
        if self.cameras is None:
            calib = vision3d.fetch_calibration(self.rosbridge_url)
            color = vision3d.Camera.from_calib(calib, self.urdf_xml, "color")
            depth = vision3d.Camera.from_calib(calib, self.urdf_xml, "depth")
            corr = vision3d.camera_correction(calib, self.urdf_xml, color)
            self.cameras = (vision3d.corrected(color, corr["shift"]), vision3d.corrected(depth, corr["shift"]), corr)
        return self.cameras

    # ---------- live state ----------

    def read_state(self, hub) -> dict:
        """{side: {"q", "gripper", "force"}} from the hub; raises if any reading is missing or stale."""
        now, out = time.monotonic(), {}
        for s in K.ARMS:
            js, gr, wr = (hub.state(f"{s}_{k}") for k in ("joints", "gripper", "wrench"))
            for name, item in ((f"{s} joints", js), (f"{s} gripper", gr)):
                if item is None or now - item[1] > MAX_STATE_AGE_S:
                    raise RuntimeError(f"No fresh {name} reading from the robot.")
            pos = dict(zip(js[0]["name"], js[0]["position"]))
            f = wr[0]["wrench"]["force"] if wr else None
            out[s] = {"q": [pos[n] for n in self.planner.arms[s].names],
                      "gripper": float(gr[0]["position"][0]),
                      "force": None if f is None else [f["x"], f["y"], f["z"]]}
        return out

    def arm_state(self, hub) -> dict:
        st = self.read_state(hub)
        forces = {s: st[s]["force"] for s in K.ARMS if st[s]["force"] is not None}
        for s, f in forces.items():
            self.force_baseline.setdefault(s, f)
        arms = {}
        for s in K.ARMS:
            q = np.asarray(st[s]["q"])
            tip = self.planner.points(s, q)["fingertip"]
            f0 = self.force_baseline.get(s)
            delta = None if st[s]["force"] is None or f0 is None else float(np.linalg.norm(np.subtract(st[s]["force"], f0)))
            arms[s] = {
                "joints": [{"name": n.removeprefix(f"{s}_arm_"), "deg": round(float(np.degrees(v)), 1),
                            "margin_deg": round(float(m), 1)}
                           for n, v, m in zip(self.planner.arms[s].names, q, self.planner.joint_margins_deg(s, q))],
                "fingertip_cm": [round(float(v) * 100, 1) for v in tip],
                "gripper": round(st[s]["gripper"], 3),
                "force_change_n": None if delta is None else round(delta, 1),
            }
        alerts = safety.contact_alerts(forces, self.force_baseline, self.cfg["contact_force_delta_n"])
        return {"arms": arms, "contact_alerts": alerts, "joint_margin_deg": self.cfg["joint_margin_deg"],
                "table_z_verified": self.cfg.get("table_z_verified", False)}

    def reset_force_baseline(self, hub):
        st = self.read_state(hub)
        self.force_baseline = {s: st[s]["force"] for s in K.ARMS if st[s]["force"] is not None}

    # ---------- planning ----------

    def summary(self, state: dict) -> str:
        lines = []
        for s in K.ARMS:
            tip = self.planner.points(s, state[s]["q"])["fingertip"]
            lines.append(f"{s} arm fingertip at x={tip[0]:.2f} y={tip[1]:.2f} z={tip[2]:.2f} m; gripper {state[s]['gripper']:.2f}")
        return "\n".join(lines)

    def plan_command(self, hub, command: str) -> dict:
        command = (command or "").strip()
        if not command:
            raise ValueError("Type a movement command first.")
        if len(command) > 300:
            raise ValueError("Command is too long (max 300 characters).")
        state = self.read_state(hub)
        t0 = time.monotonic()
        reply = nebius_vision.complete_text(command_parser.build_prompt(command, self.summary(state)),
                                            model=self.model, system=command_parser.SYSTEM)
        t1 = time.monotonic()
        parsed = command_parser.parse_reply(reply["text"])
        result = {"dry_run": True, "command": command, "explanation": parsed["explanation"],
                  "clarification": parsed["clarification"],
                  "moves": [command_parser.describe(m) for m in parsed["moves"]],
                  "model": reply["model"], "model_ms": round(1000 * (t1 - t0))}
        if parsed["pick"]:
            result.update(self.plan_pick(hub, state, parsed["pick"]))
            return result
        if parsed["moves"]:
            plan = self.planner.plan(parsed["moves"], state)
            result.update(ok=plan["ok"], checks=plan["checks"], targets=plan["targets"],
                          plan_ms=round(1000 * (time.monotonic() - t1)))
            if plan["ok"]:
                steps = []
                for s, t in plan["targets"].items():
                    moved = any(abs(v) > 1e-6 for v in t["joint_change_deg"])
                    if moved:
                        steps.append({"arm": s, "q": t["q"], "gripper": None, "phase": "move"})
                    if t["gripper"] is not None:
                        steps.append({"arm": s, "q": t["q"], "gripper": t["gripper"], "phase": "gripper"})
                result["live"] = self._store_plan(steps, state)
        else:
            result.update(ok=False, checks=[], targets={})
        return result

    def plan_pick(self, hub, state: dict, pick: dict) -> dict:
        """Locate the object (vision model box + depth) and plan a checked pick sequence. DRY-RUN."""
        color, depth_cam, corr = self._head_cameras()
        hub.want("head")
        frame = hub.latest("head")
        if not frame or time.monotonic() - frame[1] > 3.0:
            raise RuntimeError("No fresh head-camera frame.")
        jpeg = frame[0]
        t0 = time.monotonic()
        depth = vision3d.grab_depth(self.rosbridge_url)
        r = nebius_vision.ask({"head camera": jpeg}, grasp.LOCATE_PROMPT.format(obj=pick["object"]), model=self.model)
        bbox = grasp.parse_bbox(r["text"], color.width, color.height)
        t1 = time.monotonic()
        out = {"pick": {"object": pick["object"], "arm_requested": pick["arm"], "camera_correction": corr},
               "locate_ms": round(1000 * (t1 - t0)), "checks": [], "targets": {}}
        if bbox is None:
            out.update(ok=False, clarification=f"I can't see {pick['object']} in the head camera.",
                       overlay=grasp.overlay(jpeg, color, None, None, []))
            return out
        try:
            loc = vision3d.locate(bbox, depth, color, depth_cam)
        except RuntimeError as e:
            out.update(ok=False, clarification=f"Found it in the image but couldn't measure it in 3D: {e}",
                       overlay=grasp.overlay(jpeg, color, bbox, None, []))
            return out
        plan = grasp.plan_pick(self.planner, state, loc, pick["arm"])
        t2 = time.monotonic()
        chosen = plan["chosen"] or {"steps": [], "arm": None, "orientation": None, "failure": None}
        phases = {}
        for st in chosen["steps"]:
            ph = phases.setdefault(st["phase"], {"steps": 0, "ok": True})
            ph["steps"] += 1
            ph["ok"] = ph["ok"] and st["ok"]
        out["pick"].update(
            bbox_px=[round(v) for v in bbox],
            object_cm={"x": round(100 * loc.center[0], 1), "y": round(100 * loc.center[1], 1),
                       "top_z": round(100 * loc.top_z, 1), "height": round(100 * loc.height, 1),
                       "size_xy": [round(100 * v, 1) for v in loc.size_xy], "table_z": round(100 * loc.table_z, 1)},
            arm=chosen["arm"], orientation=chosen["orientation"], phases=phases,
            attempts=plan["attempts"], failure=chosen["failure"],
            first_messages=chosen["steps"][1]["ros_preview"] if len(chosen["steps"]) > 1 else [])
        out.update(ok=plan["ok"], plan_ms=round(1000 * (t2 - t1)),
                   overlay=grasp.overlay(jpeg, color, bbox, loc, chosen["steps"]))
        if plan["ok"]:
            out["live"] = self._store_plan([{k: st[k] for k in ("arm", "q", "gripper", "phase")} for st in chosen["steps"]],
                                           state)
        return out
