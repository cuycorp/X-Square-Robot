"""Pick planning (DRY-RUN): locate an object with the vision model + depth, then break a top-down grasp into
small steps that each pass every safety check. Nothing here sends commands to the robot.

Sequence: open gripper -> move above the object -> descend -> close gripper -> lift.
"""
import base64
import io
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

import kinematics as K
import safety

LOCATE_PROMPT = ('Find {obj} in the image. Return ONLY JSON {{"bbox_2d": [x1, y1, x2, y2]}} with coordinates '
                 'normalised to 0-1000 (x to the right, y down). If it is not visible, return {{"bbox_2d": null}}.')
_BOX = re.compile(r'"bbox(?:_2d)?"\s*:\s*(null|\[[^\]]*\])', re.S)


@dataclass
class PickParams:
    approach_m: float = 0.10        # height above the object top before descending
    grasp_depth_m: float = 0.02     # how far below the top the fingertips go (capped at half the object height)
    table_gap_m: float = 0.01       # fingertips never closer than this to the table
    zone_radius_m: float = 0.05     # where the fingertip may go below the normal table clearance
    step_fraction: float = 0.9      # use 90 % of the step/rotation limits per step


def parse_bbox(text: str, width: int, height: int):
    """Model reply -> pixel bbox [x1, y1, x2, y2] or None (not visible). Raises ValueError if unparseable."""
    m = _BOX.search(text or "")
    if not m:
        raise ValueError("the model did not return a bounding box")
    if m.group(1) == "null":
        return None
    vals = json.loads(m.group(1))
    if len(vals) != 4 or not all(isinstance(v, (int, float)) for v in vals):
        raise ValueError("bounding box must be four numbers")
    x1, y1, x2, y2 = (float(v) for v in vals)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError(f"bounding box out of range: {vals}")
    return [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]


def grasp_orientations(side_base_xy, obj_xy):
    """Candidate fingertip orientations, best first: straight down (two finger directions), then tilted
    toward the object from the arm's side."""
    toward = math.atan2(obj_xy[1] - side_base_xy[1], obj_xy[0] - side_base_xy[0])
    yield "straight down", K.rpy_to_matrix(0, math.pi / 2, 0)
    yield "straight down, fingers turned 90°", K.rpy_to_matrix(0, math.pi / 2, math.pi / 2)
    for pitch in (75, 60):
        yield f"tilted {pitch}° toward the object", K.rpy_to_matrix(0, math.radians(pitch), toward)


def _interpolate(T0, T1, max_step, max_rot_rad, frac):
    """Fingertip poses from T0 (exclusive) to T1 (inclusive) in steps under the limits."""
    e = K.rotation_error(T1[:3, :3], T0[:3, :3])
    angle, dist = float(np.linalg.norm(e)), float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    n = max(1, math.ceil(max(dist / (frac * max_step), angle / (frac * max_rot_rad))))
    axis = e / angle if angle > 1e-9 else np.array([0.0, 0.0, 1.0])
    for i in range(1, n + 1):
        t = i / n
        yield K.transform(K.axis_angle(axis, angle * t) @ T0[:3, :3], T0[:3, 3] + t * (T1[:3, 3] - T0[:3, 3]))


def _sequence(planner: safety.SafetyPlanner, side: str, R, located, state: dict, p: PickParams):
    """Plan one full pick with one arm and orientation. Returns (ok, steps, failure)."""
    cfg = planner.cfg
    c = located.center
    grasp_z = max(located.top_z - min(p.grasp_depth_m, located.height / 2), located.table_z + p.table_gap_m)
    above = K.transform(R, np.array([c[0], c[1], located.top_z + p.approach_m]))
    at = K.transform(R, np.array([c[0], c[1], grasp_z]))
    zone = safety.GraspZone(np.array(c[:2]), p.zone_radius_m, located.table_z + p.table_gap_m)
    sim = {s: {"q": list(state[s]["q"]), "gripper": state[s]["gripper"]} for s in K.ARMS}
    steps = []

    def run(phase, T, gripper=None, use_zone=False):
        res = planner.plan_goals({side: (T, gripper)}, sim, zone if use_zone else None)
        tip = planner.tip_pose(side, res["q_goal"][side])[:3, 3]
        steps.append({"phase": phase, "ok": res["ok"], "tip": [round(float(v), 4) for v in tip],
                      "failed": [f"{x['name']}: {x['detail']}" for x in res["checks"] if not x["ok"]],
                      "ros_preview": res["targets"][side]["ros_preview"],
                      "arm": side, "q": [float(v) for v in res["q_goal"][side]],
                      "gripper": res["targets"][side]["gripper"]})
        if res["ok"]:
            sim[side]["q"] = res["q_goal"][side]
        return res["ok"]

    def move(phase, T_goal, use_zone=False):
        T_now = planner.tip_pose(side, np.asarray(sim[side]["q"]))
        for T in _interpolate(T_now, T_goal, cfg["max_step_m"], math.radians(cfg["max_rotation_deg"]), p.step_fraction):
            if not run(phase, T, use_zone=use_zone):
                return False
        return True

    T_start = planner.tip_pose(side, np.asarray(sim[side]["q"]))
    ok = (run("open gripper", T_start, "open")
          and move("move above the object", above)
          and move("descend", at, use_zone=True)
          and run("close gripper", planner.tip_pose(side, np.asarray(sim[side]["q"])), "close", use_zone=True)
          and move("lift", above, use_zone=True))
    failure = None if ok else steps[-1]
    return ok, steps, failure


def plan_pick(planner: safety.SafetyPlanner, state: dict, located, arm: str | None = None, params: PickParams | None = None):
    """Try arms (nearest first, or the requested one) and grasp orientations until a full sequence passes."""
    p = params or PickParams()
    obj_xy = np.asarray(located.center[:2])
    sides = [arm] if arm in K.ARMS else sorted(K.ARMS, key=lambda s: np.linalg.norm(planner.world[s][:2, 3] - obj_xy))
    attempts, best = [], None
    for s in sides:
        for label, R in grasp_orientations(planner.world[s][:2, 3], obj_xy):
            ok, steps, failure = _sequence(planner, s, R, located, state, p)
            attempt = {"arm": s, "orientation": label, "ok": ok, "steps": steps, "failure": failure,
                       "progress": sum(st["ok"] for st in steps)}
            attempts.append({k: attempt[k] for k in ("arm", "orientation", "ok", "progress", "failure")})
            if ok:
                return {"ok": True, "chosen": attempt, "attempts": attempts}
            if best is None or attempt["progress"] > best["progress"]:
                best = attempt
    return {"ok": False, "chosen": best, "attempts": attempts}


def overlay(jpeg: bytes, color_cam, bbox_px, located, steps) -> str:
    """Camera frame with the object box, its 3D centre and the planned fingertip path. Returns base64 JPEG."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    d = ImageDraw.Draw(img)
    if bbox_px:
        d.rectangle(bbox_px, outline=(255, 200, 0), width=3)
    if located is not None:
        (u, v), = color_cam.project(np.array([located.center]))[0]
        d.ellipse([u - 7, v - 7, u + 7, v + 7], outline=(255, 0, 255), width=3)
    if steps:
        uv, _ = color_cam.project(np.array([s["tip"] for s in steps]))
        pts = [tuple(map(float, x)) for x in uv if np.all(np.isfinite(x))]
        if len(pts) > 1:
            d.line(pts, fill=(0, 220, 120), width=3)
        for (u, v), s in zip(uv, steps):
            if np.all(np.isfinite([u, v])):
                col = (0, 220, 120) if s["ok"] else (255, 40, 40)
                d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=col)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")
