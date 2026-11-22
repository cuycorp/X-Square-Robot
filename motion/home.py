"""'Go home': bring one arm back to a known ready pose in small, fully checked joint-space steps (DRY plan;
motion/sender.py executes it like any other plan).

Picks are planned in small Cartesian steps, which can be impossible from an awkward pose (e.g. a wrist joint
near its limit). From the ready pose they work, so "go home" is the reset between tasks.
"""
import math

import numpy as np

import safety

# Folded rest pose, but 6° away from the joint2/joint3 limits (margin is 5°) it normally rests on.
HOME_Q_DEG = (0.0, 6.0, -6.0, 0.0, 0.0, 0.0)


def plan_home(planner: safety.SafetyPlanner, state: dict, side: str, max_step_deg: float = 10.0) -> dict:
    """Joint-space path from the current pose to HOME, split so no joint moves more than max_step_deg per step.
    Every step passes plan_joint_goal (limits, joint change, workspace, table clearance and arm separation)."""
    q0 = np.asarray(state[side]["q"], float)
    q1 = np.radians(HOME_Q_DEG)
    n = max(1, math.ceil(float(np.degrees(np.max(np.abs(q1 - q0)))) / max_step_deg))
    sim = {s: {"q": list(state[s]["q"]), "gripper": state[s].get("gripper", 0.0)} for s in state}
    steps = []
    for i in range(1, n + 1):
        q = q0 + (i / n) * (q1 - q0)
        res = planner.plan_joint_goal(side, q, sim)
        steps.append({"phase": "go home", "ok": res["ok"], "arm": side, "q": [float(v) for v in q], "gripper": None,
                      "failed": [f"{c['name']}: {c['detail']}" for c in res["checks"] if not c["ok"]],
                      "checks": res["checks"]})
        if not res["ok"]:
            return {"ok": False, "steps": steps, "failure": steps[-1]}
        sim[side]["q"] = [float(v) for v in q]
    return {"ok": True, "steps": steps, "failure": None}


def plan_home_arms(planner: safety.SafetyPlanner, state: dict, arms: list, max_step_deg: float = 10.0) -> dict:
    """Send one or more arms home ONE AFTER THE OTHER. Each arm's steps are checked with the other arm where it
    will really be (still at its start pose, or already home), which is exactly how the plan is executed.
    For two arms both orders are tried; the first one that passes every check is used."""
    orders = [list(arms)] if len(arms) == 1 else [["left", "right"], ["right", "left"]]
    best = None
    for order in orders:
        sim = {s: {"q": list(state[s]["q"]), "gripper": state[s].get("gripper", 0.0)} for s in state}
        steps, failure = [], None
        for side in order:
            r = plan_home(planner, sim, side, max_step_deg)
            steps += [{**st, "phase": f"{side} arm home"} for st in r["steps"]]
            if not r["ok"]:
                failure = {**r["failure"], "phase": f"{side} arm home"}
                break
            sim[side]["q"] = list(r["steps"][-1]["q"])
        result = {"ok": failure is None, "order": order, "steps": steps, "failure": failure}
        if result["ok"]:
            return result
        if best is None or len(steps) > len(best["steps"]):
            best = result
    assert best is not None  # orders is never empty
    return best
