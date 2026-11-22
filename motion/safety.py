"""Our own safety checks for planned arm moves. They run on every plan, whatever the robot's settings.

Conventions:
- Positions are in base_link (x forward, y robot-left, z up), metres.
- Moves act on the FINGERTIP (the tool point between the finger tips): translations move it, rotations
  pivot around it. Step size is measured at the fingertip.
- "Don't make it worse": a check that already fails at the current pose (e.g. the arms rest right at joint
  limits) passes if the move keeps that quantity the same or improves it.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import kinematics as K

CONFIG_PATH = Path(__file__).resolve().parent / "safety_config.json"
SEPARATION_LINKS = ("link3", "link4", "link5", "link6", "fingertip")
CLEARANCE_LINKS = ("link4", "link5", "link6", "fingertip")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


@dataclass
class Move:
    arm: str
    translate_m: np.ndarray = field(default_factory=lambda: np.zeros(3))   # fingertip, base_link axes
    rotate_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))    # roll, pitch, yaw about base axes, at the fingertip
    gripper: str | None = None                                             # "open" | "close" | None


@dataclass
class GraspZone:
    """Lets the fingertip go below the table-clearance height, but only near the object being grasped."""
    center_xy: np.ndarray
    radius_m: float
    min_tip_z: float


def check(name: str, ok: bool, detail: str, arm: str | None = None) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "arm": arm}


class SafetyPlanner:
    def __init__(self, urdf_xml: str, config: dict):
        self.cfg = config
        self.arms = {s: K.Arm(urdf_xml, s) for s in K.ARMS}
        self.world = {s: K.arm_base_in_world(urdf_xml, s) for s in K.ARMS}
        self.tip = {s: K.transform(np.eye(3), K.fingertip_offset(urdf_xml, s)) for s in K.ARMS}
        self.neutral = {s: self.arms[s].fk(np.zeros(6)) for s in K.ARMS}

    # ---------- geometry ----------

    def tip_pose(self, side: str, q) -> np.ndarray:
        """Fingertip pose in base_link."""
        return self.world[side] @ self.arms[side].fk(q) @ self.tip[side]

    def link6_goal(self, side: str, T_tip_world: np.ndarray) -> np.ndarray:
        """Link6 pose in the arm base frame that puts the fingertip at T_tip_world."""
        return np.linalg.inv(self.world[side]) @ T_tip_world @ np.linalg.inv(self.tip[side])

    def points(self, side: str, q) -> dict:
        """base_link positions of the links used by the checks."""
        W, out = self.world[side], {}
        for name, T in self.arms[side].fk_all(np.asarray(q, float)):
            short = name.removeprefix(f"{side}_arm_")
            if short in ("link3", "link4", "link5", "link6"):
                out[short] = (W @ T)[:3, 3]
            if short == "link6":
                out["fingertip"] = (W @ T @ self.tip[side])[:3, 3]
        return out

    def joint_margins_deg(self, side: str, q) -> np.ndarray:
        a = self.arms[side]
        return np.degrees(np.minimum(np.asarray(q) - a.lower, a.upper - np.asarray(q)))

    # ---------- planning ----------

    def plan(self, moves: list, state: dict) -> dict:
        """moves: [Move]; state: {side: {"q": [6], "gripper": float}}."""
        sides = [m.arm for m in moves]
        if not moves or len(moves) > 2 or len(set(sides)) != len(sides) or any(s not in K.ARMS for s in sides):
            return {"ok": False, "checks": [check("moves", False, "Plan one move per arm (left and/or right).")],
                    "targets": {}}
        goals = {}
        for mv in moves:
            T_tip = self.tip_pose(mv.arm, np.asarray(state[mv.arm]["q"], float))
            T_goal = K.transform(K.rpy_to_matrix(*mv.rotate_rad) @ T_tip[:3, :3], T_tip[:3, 3] + mv.translate_m)
            goals[mv.arm] = (T_goal, mv.gripper)
        return self.plan_goals(goals, state)

    def plan_goals(self, goals: dict, state: dict, grasp_zone: GraspZone | None = None) -> dict:
        """goals: {side: (T_tip_goal in base_link, gripper "open"/"close"/None)}. Runs every check and returns
        targets plus the ROS messages that *would* be sent (never sent from here)."""
        cfg, checks, targets = self.cfg, [], {}
        q_now = {s: np.asarray(state[s]["q"], float) for s in K.ARMS}
        q_goal = dict(q_now)
        for s, (T_goal, gripper) in goals.items():
            arm = self.arms[s]
            T_now = self.tip_pose(s, q_now[s])
            step = float(np.linalg.norm(T_goal[:3, 3] - T_now[:3, 3]))
            rot = float(np.degrees(np.linalg.norm(K.rotation_error(T_goal[:3, :3], T_now[:3, :3]))))
            checks.append(check("step size", step <= cfg["max_step_m"] + 1e-6,
                                f"fingertip moves {100 * step:.1f} cm (max {100 * cfg['max_step_m']:.0f} cm)", s))
            checks.append(check("rotation size", rot <= cfg["max_rotation_deg"] + 1e-6,
                                f"{rot:.1f}° (max {cfg['max_rotation_deg']}°)", s))
            q1, pos_err, rot_err = arm.ik(self.link6_goal(s, T_goal), q_now[s])
            ik_ok = pos_err <= cfg["ik_position_tolerance_m"] and np.degrees(rot_err) <= cfg["ik_rotation_tolerance_deg"]
            checks.append(check("reachable (IK)", ik_ok,
                                f"solution error {1000 * pos_err:.1f} mm / {np.degrees(rot_err):.2f}°", s))
            checks.append(self._joint_limits(s, q_now[s], q1))
            dq = float(np.max(np.abs(q1 - q_now[s])))
            checks.append(check("joint change", dq <= cfg["max_joint_change_rad"],
                                f"largest joint change {np.degrees(dq):.1f}° (max {np.degrees(cfg['max_joint_change_rad']):.0f}°)", s))
            checks.extend(self._workspace_checks(s, q_now[s], q1))
            tip = self.points(s, q1)["fingertip"]
            g_value = None
            if gripper is not None:
                g_value = cfg["gripper_open"] if gripper == "open" else cfg["gripper_closed"]
                lo, hi = cfg["gripper_range"]
                checks.append(check("gripper range", lo <= g_value <= hi, f"{gripper} → {g_value:.3f} (range {lo}–{hi})", s))
            q_goal[s] = q1
            targets[s] = {
                "q": [round(float(v), 4) for v in q1],
                "joint_change_deg": [round(float(v), 2) for v in np.degrees(q1 - q_now[s])],
                "fingertip_world": [round(float(v), 4) for v in tip],
                "gripper": g_value,
                "ros_preview": self._ros_preview(s, np.linalg.inv(self.neutral[s]) @ arm.fk(q1), gripper),
            }
        checks.extend(self._path_checks(q_now, q_goal, list(goals), grasp_zone))
        return {"ok": all(c["ok"] for c in checks), "checks": checks, "targets": targets,
                "q_goal": {s: q_goal[s].tolist() for s in K.ARMS}}

    def _box_excess(self, tip) -> float:
        """How far (m) the fingertip is outside the workspace box (0 when inside)."""
        cfg = self.cfg
        (x0, x1), (y0, y1) = cfg["workspace_x"], cfg["workspace_y"]
        return float(max(0.0, x0 - tip[0]) + max(0.0, tip[0] - x1) + max(0.0, y0 - tip[1]) + max(0.0, tip[1] - y1)
                     + max(0.0, tip[2] - cfg["workspace_z_max"]))

    def _workspace_checks(self, s, q0, q1) -> list:
        """Workspace box and reach, with the don't-make-it-worse rule (an arm already outside may move back in)."""
        cfg = self.cfg
        p0, p1 = self.points(s, q0), self.points(s, q1)
        tip = p1["fingertip"]
        e0, e1 = self._box_excess(p0["fingertip"]), self._box_excess(tip)
        box_ok = e1 == 0 or e1 <= e0 + 1e-4
        note = "" if e1 == 0 else f"; {100 * e1:.1f} cm outside, " + ("not getting worse" if box_ok else "getting worse")
        base = self.world[s][:3, 3]
        r0, r1 = (float(np.linalg.norm(p["link6"] - base)) for p in (p0, p1))
        reach_ok = r1 <= cfg["max_reach_m"] or r1 <= r0 + 1e-4
        return [check("workspace box", box_ok, f"fingertip at ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m{note}", s),
                check("reach", reach_ok, f"{100 * r1:.0f} cm from the arm base (max {100 * cfg['max_reach_m']:.0f} cm)", s)]

    def plan_joint_goal(self, side: str, q_goal, state: dict) -> dict:
        """Check a joint-space move of one arm (used by "go home"): limits, joint change, workspace and the path."""
        cfg = self.cfg
        q_now = {s: np.asarray(state[s]["q"], float) for s in K.ARMS}
        q1 = np.asarray(q_goal, float)
        dq = float(np.max(np.abs(q1 - q_now[side])))
        checks = [self._joint_limits(side, q_now[side], q1),
                  check("joint change", dq <= cfg["max_joint_change_rad"],
                        f"largest joint change {np.degrees(dq):.1f}° (max {np.degrees(cfg['max_joint_change_rad']):.0f}°)", side),
                  *self._workspace_checks(side, q_now[side], q1)]
        checks.extend(self._path_checks(q_now, {**q_now, side: q1}, [side], None))
        return {"ok": all(c["ok"] for c in checks), "checks": checks, "q_goal": {**{s: q_now[s].tolist() for s in K.ARMS},
                                                                                 side: q1.tolist()}}

    def _joint_limits(self, s, q0, q1) -> dict:
        a, m = self.arms[s], np.radians(self.cfg["joint_margin_deg"])
        bad, notes = [], []
        for i, name in enumerate(a.names):
            d0 = min(q0[i] - a.lower[i], a.upper[i] - q0[i])
            d1 = min(q1[i] - a.lower[i], a.upper[i] - q1[i])
            short = name.removeprefix(f"{s}_arm_")
            if d1 >= m:
                continue
            if d1 >= 0 and d1 >= d0 - 1e-4:
                notes.append(f"{short} near limit but moving away ({np.degrees(d0):.1f}° → {np.degrees(d1):.1f}°)")
            else:
                bad.append(f"{short} would be {np.degrees(d1):.1f}° from its limit")
        detail = "; ".join(bad or notes) or f"all joints ≥ {self.cfg['joint_margin_deg']}° from limits"
        return check("joint limits", not bad, detail, s)

    def _path_checks(self, q_now, q_goal, sides, zone: GraspZone | None) -> list:
        cfg = self.cfg
        n = max(2, int(cfg["path_samples"]))
        floor = cfg["table_z"] + cfg["table_clearance_m"]

        def clearance(qs):  # lowest point of the moving arms above its allowed floor (negative = too low)
            worst = np.inf
            for s in sides:
                for k, p in self.points(s, qs[s]).items():
                    if k not in CLEARANCE_LINKS:
                        continue
                    f = floor
                    if zone is not None and k == "fingertip" and np.linalg.norm(p[:2] - zone.center_xy) <= zone.radius_m:
                        f = zone.min_tip_z
                    worst = min(worst, p[2] - f)
            return worst

        def separation(qs):
            L, R = self.points("left", qs["left"]), self.points("right", qs["right"])
            return min(float(np.linalg.norm(L[a] - R[b])) for a in SEPARATION_LINKS for b in SEPARATION_LINKS)

        path = [{s: q_now[s] + t * (q_goal[s] - q_now[s]) for s in K.ARMS} for t in np.linspace(0, 1, n)]
        c0, s0 = clearance(path[0]), separation(path[0])
        c_min, s_min = min(clearance(p) for p in path), min(separation(p) for p in path)
        verified = "" if cfg.get("table_z_verified") else " (table height NOT confirmed)"
        zone_note = f"; grasp zone allows the fingertip down to {100 * zone.min_tip_z:.1f} cm near the object" if zone else ""
        c_ok = c_min >= 0 or c_min >= c0 - 1e-3
        s_ok = s_min >= cfg["min_arm_distance_m"] or s_min >= s0 - 1e-3
        return [
            check("table clearance (path)", c_ok,
                  f"closest approach to the allowed floor {100 * c_min:+.1f} cm (floor = table + "
                  f"{100 * cfg['table_clearance_m']:.0f} cm{zone_note}){verified}"),
            check("arm separation (path)", s_ok,
                  f"closest approach {100 * s_min:.1f} cm (min {100 * cfg['min_arm_distance_m']:.0f} cm)"),
        ]

    def _ros_preview(self, s, T_cmd, gripper) -> list:
        """The messages a live sender would publish. Format NOT yet verified with the organizers."""
        q = K.matrix_to_quat(T_cmd[:3, :3])
        msgs = [{"topic": f"/{s}_arm_cartesian_controller/pose_cmd", "type": "geometry_msgs/msg/PoseStamped",
                 "msg": {"header": {"frame_id": ""},  # as the platform sends
                         "pose": {"position": dict(zip("xyz", [round(float(v), 5) for v in T_cmd[:3, 3]])),
                                  "orientation": dict(zip("xyzw", [round(float(v), 5) for v in q]))}}}]
        if gripper is not None:
            g = self.cfg["gripper_open"] if gripper == "open" else self.cfg["gripper_closed"]
            msgs.append({"topic": f"/{s}_gripper_controller/commands", "type": "std_msgs/msg/Float64MultiArray",
                         "msg": {"data": [g]}})
        return msgs


def contact_alerts(wrench_now: dict, wrench_baseline: dict, threshold_n: float) -> list:
    """Runtime monitor: arms whose external force changed more than threshold since the baseline."""
    alerts = []
    for s, f in wrench_now.items():
        if s in wrench_baseline:
            delta = float(np.linalg.norm(np.asarray(f) - np.asarray(wrench_baseline[s])))
            if delta > threshold_n:
                alerts.append(f"{s} arm: external force changed by {delta:.0f} N (limit {threshold_n:.0f} N)")
    return alerts
