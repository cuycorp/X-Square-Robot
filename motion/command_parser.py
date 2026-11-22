"""Natural-language movement command -> structured Move(s), using a Nebius language model.

The model only *proposes* a move. safety.SafetyPlanner decides whether it is allowed, and the user confirms.
"""
import json
import re

import numpy as np

from safety import Move

SYSTEM = """You convert movement commands for a dual-arm tabletop robot into JSON. Output ONLY a JSON object.

Directions are from the ROBOT's point of view, in its base frame:
  forward = +x (away from the robot, toward the far side of the table), back = -x,
  left = +y (robot's left), right = -y, up = +z, down = -z.
"left arm" / "right arm" are the robot's own arms. Rotations are about those axes at the gripper:
  roll about x, pitch about y (positive tips the gripper down), yaw about z (positive turns it to the robot's left).

Schema:
{"moves": [{"arm": "left"|"right",
            "translate_cm": {"x": number, "y": number, "z": number},
            "rotate_deg": {"roll": number, "pitch": number, "yaw": number},
            "gripper": "open"|"close"|null}],
 "pick": null | {"object": "short description of the object to pick", "arm": "left"|"right"|null},
 "home": null | {"arm": "left"|"right"|"both"},
 "explanation": "one short sentence of what will happen",
 "clarification": null | "question to ask the user"}

Rules:
- At most one entry per arm. "both arms" -> two entries.
- Use 0 for axes that are not mentioned. If no distance is given, use 2 cm; if no angle is given, use 10 degrees.
- Never make a move bigger than the user asked for. Do not add moves the user did not ask for.
- If the user asks an arm to go home / reset / return to rest / go back to the start, return "moves": [] and
  "home": {"arm": ...} ("both" if they say both arms or name no arm).
- If the user asks to pick up / grab / take an object, return "moves": [] and fill "pick" with a short visual
  description of the object (e.g. "the Rubik's cube") and the arm only if the user named one.
- If the command refers to objects or places in any other way you cannot turn into a direction and distance
  (e.g. "move to the cup"), or the arm is ambiguous, return "moves": [] and ask in "clarification".
"""

_JSON = re.compile(r"\{.*\}", re.S)


def build_prompt(command: str, arm_summary: str) -> str:
    return f"Current state (for context only):\n{arm_summary}\n\nCommand: {command.strip()}"


def parse_reply(text: str) -> dict:
    """Model reply -> {"moves": [Move], "explanation", "clarification", "raw"}. Raises ValueError if malformed."""
    m = _JSON.search(text or "")
    if not m:
        raise ValueError("the model did not return JSON")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"the model returned invalid JSON ({e.msg})") from None
    moves, seen = [], set()
    for item in data.get("moves") or []:
        arm = str(item.get("arm", "")).lower()
        if arm not in ("left", "right"):
            raise ValueError(f"unknown arm {item.get('arm')!r}")
        if arm in seen:
            raise ValueError(f"two moves for the {arm} arm")
        seen.add(arm)
        t, r = item.get("translate_cm") or {}, item.get("rotate_deg") or {}
        try:
            translate = np.array([float(t.get(k, 0) or 0) for k in "xyz"]) / 100.0
            rotate = np.radians([float(r.get(k, 0) or 0) for k in ("roll", "pitch", "yaw")])
        except (TypeError, ValueError):
            raise ValueError("non-numeric distance or angle") from None
        if not np.all(np.isfinite(translate)) or not np.all(np.isfinite(rotate)):
            raise ValueError("non-finite distance or angle")
        gripper = item.get("gripper")
        if gripper not in (None, "open", "close"):
            raise ValueError(f"unknown gripper action {gripper!r}")
        if not np.any(translate) and not np.any(rotate) and gripper is None:
            continue  # nothing to do for this arm
        moves.append(Move(arm, translate, rotate, gripper))
    pick = None
    raw_pick = data.get("pick")
    if isinstance(raw_pick, dict) and str(raw_pick.get("object") or "").strip():
        arm = raw_pick.get("arm")
        if arm not in (None, "left", "right"):
            raise ValueError(f"unknown arm {arm!r} for pick")
        pick = {"object": str(raw_pick["object"]).strip()[:100], "arm": arm}
        moves = []  # a pick is planned on its own
    home = None
    raw_home = data.get("home")
    if isinstance(raw_home, dict) and not pick:
        arm = raw_home.get("arm") or "both"
        if arm not in ("left", "right", "both"):
            raise ValueError(f"unknown arm {arm!r} for home")
        home = {"arm": arm}
        moves = []
    clarification = data.get("clarification") or None
    if not moves and not pick and not home and not clarification:
        clarification = "I couldn't find a movement in that command. Try e.g. 'left arm up 2 cm'."
    return {"moves": moves, "pick": pick, "home": home, "explanation": str(data.get("explanation") or ""),
            "clarification": None if (pick or home) else clarification}


def describe(move: Move) -> dict:
    """JSON-friendly view of a Move for the UI."""
    return {"arm": move.arm,
            "translate_cm": dict(zip("xyz", [round(float(v) * 100, 2) for v in move.translate_m])),
            "rotate_deg": dict(zip(("roll", "pitch", "yaw"), [round(float(v), 1) for v in np.degrees(move.rotate_rad)])),
            "gripper": move.gripper}
