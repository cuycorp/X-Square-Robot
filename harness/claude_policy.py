"""Autonomous Vision-to-EndPose planners using Anthropic Claude (zero local GPU).

Uses only Python stdlib urllib.request so it runs in minimal robot environments.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from protocol import ValidatedRequest
from trajectory import rotation_distance


DEFAULT_CLAUDE_KEY_FILE = Path.home() / ".config" / "anthropic" / "api_key"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
MAX_WAYPOINTS = 4

SYSTEM_PROMPT = """You autonomously control a bimanual desktop robot (quanta_x1_pro) via EndPose targets.

Coordinate frame & camera layout:
- Robot base frame in meters and radians: +X is forward away from the robot torso, +Y is robot's left, +Z is up.
- Rotation is RPY [roll, pitch, yaw] with R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
- CAMERA ORIENTATION WARNING: In the 'front' image, the robot's RIGHT arm appears on the LEFT side of the image, and the robot's LEFT arm appears on the RIGHT side of the image.
- 'left_wrist' and 'right_wrist' are eye-in-hand cameras mounted on each respective gripper.
- Gripper range: 0.0 is fully closed, max is 1.526 (gripper-g) or 5.5 (gripper-h). Typical open is ~0.85 * max, typical grasp close is ~0.05 * max.

Control guidelines:
1. Break any natural-language task into smooth sequential stages:
   approach_above -> align_xy -> descend -> grasp (hold position while closing gripper) -> lift -> transport -> place -> release (hold position while opening gripper) -> retreat -> completed.
2. Return one to four sequential waypoints for the current stage. Use absolute base-frame positions in meters and RPY in radians. Keep each leg short (roughly <= 0.08 m and <= 0.40 rad), clear of the tabletop, both arms and visible obstacles. Give each arm's full pose and gripper when commanding it. Omit any idle arm.
3. Use feedback from all three cameras and the measured end-effector poses. Stop the current plan at a point where new visual evidence is required, for example before closing on an unverified target or after lifting.
4. If the scene is ambiguous, the path cannot be estimated from the views, or a grasp is uncertain, return status="retry" and an empty waypoints array. Explain what blocked planning.
5. Return status="completed" with an empty waypoints array only when fresh observations show the task goal is satisfied. The controller will hold the measured pose.

Respond strictly with ONE valid JSON object (no markdown fences or commentary):
{
  "stage": "short_stage_label",
  "status": "in_progress | completed | retry",
  "reason": "brief observation-grounded explanation",
  "waypoints": [
    {
      "left_ee": {
        "position": [x, y, z],
        "rotation_rpy": [roll, pitch, yaw],
        "gripper": 1.2
      }
    }
  ]
}"""


class AuthError(RuntimeError):
    """Raised when the API key is missing or rejected (HTTP 401/403)."""


@dataclass
class PlannerDecision:
    stage: str
    status: str
    reason: str
    left_target_pos: list[float]
    left_target_rpy: list[float]
    left_gripper: float
    right_target_pos: list[float]
    right_target_rpy: list[float]
    right_gripper: float
    raw_json: dict[str, Any] = field(default_factory=dict)
    waypoint_index: int | None = None
    waypoint_count: int = 0


def load_anthropic_api_key(explicit_key: str | None = None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key
    if DEFAULT_CLAUDE_KEY_FILE.exists():
        return DEFAULT_CLAUDE_KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_anthropic_api_key(api_key: str, path: Path = DEFAULT_CLAUDE_KEY_FILE) -> Path:
    clean = api_key.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(clean + "\n")
    os.chmod(path, 0o600)
    return path


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace:
            cleaned = brace.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Claude response must be a JSON object")
    return parsed


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Pose values must be finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Pose values must be finite numbers")
    return result


def _validated_waypoints(parsed: dict[str, Any], gripper_range: tuple[float, float]) -> list[dict[str, Any]]:
    waypoints = parsed["waypoints"]
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= MAX_WAYPOINTS:
        raise ValueError(f"Expected 1 to {MAX_WAYPOINTS} waypoints")
    lo, hi = gripper_range
    for point in waypoints:
        if not isinstance(point, dict) or not point or set(point) - {"left_ee", "right_ee"}:
            raise ValueError("A waypoint must specify at least one arm")
        for arm in point.values():
            if not isinstance(arm, dict) or set(arm) != {"position", "rotation_rpy", "gripper"}:
                raise ValueError("A waypoint arm needs position, rotation_rpy and gripper")
            for field_name in ("position", "rotation_rpy"):
                values = arm[field_name]
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError(f"{field_name} must have three components")
                arm[field_name] = [_finite_number(value) for value in values]
            gripper = _finite_number(arm["gripper"])
            if not lo <= gripper <= hi:
                raise ValueError("Waypoint gripper is outside the configured range")
            arm["gripper"] = gripper
    return waypoints


def _apply_arm_command(
    current_pos: list[float],
    current_rpy: list[float],
    current_grip: float,
    cmd: dict[str, Any] | None,
    gripper_range: tuple[float, float],
) -> tuple[list[float], list[float], float]:
    if not isinstance(cmd, dict):
        return list(current_pos), list(current_rpy), float(current_grip)

    if "position" in cmd and isinstance(cmd["position"], (list, tuple)) and len(cmd["position"]) == 3:
        tgt_pos = [_finite_number(value) for value in cmd["position"]]
    else:
        dp = cmd.get("delta_position", [0.0, 0.0, 0.0])
        if not isinstance(dp, (list, tuple)) or len(dp) != 3:
            dp = [0.0, 0.0, 0.0]
        tgt_pos = [_finite_number(current + _finite_number(delta)) for current, delta in zip(current_pos, dp)]

    if "rotation_rpy" in cmd and isinstance(cmd["rotation_rpy"], (list, tuple)) and len(cmd["rotation_rpy"]) == 3:
        tgt_rpy = [_finite_number(value) for value in cmd["rotation_rpy"]]
    else:
        dr = cmd.get("delta_rpy", [0.0, 0.0, 0.0])
        if not isinstance(dr, (list, tuple)) or len(dr) != 3:
            dr = [0.0, 0.0, 0.0]
        tgt_rpy = [_finite_number(current + _finite_number(delta)) for current, delta in zip(current_rpy, dr)]

    lo, hi = gripper_range
    g_raw = cmd.get("gripper", current_grip)
    tgt_grip = max(lo, min(hi, _finite_number(g_raw)))
    return tgt_pos, tgt_rpy, tgt_grip


def _decision_from_parsed(req: ValidatedRequest, parsed: dict[str, Any]) -> PlannerDecision:
    lp, lr, lg = _apply_arm_command(
        req.left_ee.position,
        req.left_ee.rotation_rpy,
        req.left_ee.gripper,
        parsed.get("left_ee"),
        req.gripper_range,
    )
    rp, rr, rg = _apply_arm_command(
        req.right_ee.position,
        req.right_ee.rotation_rpy,
        req.right_ee.gripper,
        parsed.get("right_ee"),
        req.gripper_range,
    )
    return PlannerDecision(
        stage=str(parsed.get("stage", "step")),
        status=str(parsed.get("status", "in_progress")),
        reason=str(parsed.get("reason", "")),
        left_target_pos=lp,
        left_target_rpy=lr,
        left_gripper=lg,
        right_target_pos=rp,
        right_target_rpy=rr,
        right_gripper=rg,
        raw_json=parsed,
    )


def _state_summary(req: ValidatedRequest, history: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "task_prompt": req.instruction,
            "embodiment": req.embodiment,
            "gripper_range": list(req.gripper_range),
            "left_ee": {
                "position": [round(v, 4) for v in req.left_ee.position],
                "rotation_rpy": [round(v, 4) for v in req.left_ee.rotation_rpy],
                "gripper": round(req.left_ee.gripper, 4),
            },
            "right_ee": {
                "position": [round(v, 4) for v in req.right_ee.position],
                "rotation_rpy": [round(v, 4) for v in req.right_ee.rotation_rpy],
                "gripper": round(req.right_ee.gripper, 4),
            },
            "recent_history": history[-6:],
            **({"local_perception": req.perception} if req.perception is not None else {}),
        }
    )


def _reached_waypoint(req: ValidatedRequest, decision: PlannerDecision) -> bool:
    grip_tolerance = 0.02 * (req.gripper_range[1] - req.gripper_range[0])
    arms = (
        (req.left_ee, decision.left_target_pos, decision.left_target_rpy, decision.left_gripper),
        (req.right_ee, decision.right_target_pos, decision.right_target_rpy, decision.right_gripper),
    )
    return all(
        math.dist(arm.position, pos) <= 0.01
        and rotation_distance(arm.rotation_rpy, rpy) <= 0.05
        and abs(arm.gripper - grip) <= grip_tolerance
        for arm, pos, rpy, grip in arms
    )


def hold_pose_decision(req: ValidatedRequest, stage: str = "hold", status: str = "completed", reason: str = "") -> PlannerDecision:
    return PlannerDecision(
        stage=stage,
        status=status,
        reason=reason,
        left_target_pos=list(req.left_ee.position),
        left_target_rpy=list(req.left_ee.rotation_rpy),
        left_gripper=req.left_ee.gripper,
        right_target_pos=list(req.right_ee.position),
        right_target_rpy=list(req.right_ee.rotation_rpy),
        right_gripper=req.right_ee.gripper,
    )


class ClaudeEndPosePlanner:
    """Autonomous EndPose planner powered by Anthropic Claude Vision Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 25.0,
        plan_every: int = 4,
    ) -> None:
        self.api_key = load_anthropic_api_key(api_key)
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL
        self.timeout_s = timeout_s
        self.plan_every = max(1, int(plan_every))
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self._cached_decision: PlannerDecision | None = None
        self._active_goal: str | None = None
        self._plan_step = 0
        self._plan: dict[str, Any] = {}
        self._waypoints: list[dict[str, Any]] = []
        self._waypoint_index = 0

    def _pick_available_model(self, key: str) -> str | None:
        req = urllib.request.Request(
            f"{self.base_url}/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=min(self.timeout_s, 8)) as resp:
            payload = json.load(resp)
        ids = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
        for prefix in ("claude-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-opus", "claude-"):
            for mid in ids:
                if mid.startswith(prefix):
                    return mid
        return ids[0] if ids else None

    def configure_api_key(self, key: str) -> None:
        """Check model access with a metadata request before accepting the session key."""
        key = key.strip()
        if not key or len(key) > 512 or any(character.isspace() for character in key):
            raise ValueError("Enter a valid Anthropic API key.")
        model = urllib.parse.quote(self.model, safe="")
        request = urllib.request.Request(
            f"{self.base_url}/v1/models/{model}",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 8)):
                pass
        except urllib.error.HTTPError as error:
            if error.code == 404:
                try:
                    fallback = self._pick_available_model(key)
                except urllib.error.HTTPError as list_err:
                    raise ValueError(f"Anthropic couldn't verify the API key (HTTP {list_err.code}).") from list_err
                except OSError as net_err:
                    raise RuntimeError("Could not reach the Anthropic API to verify the key.") from net_err
                if not fallback:
                    raise ValueError(f"Anthropic couldn't verify model access (HTTP {error.code}).") from error
                self.model = fallback
            else:
                raise ValueError(f"Anthropic couldn't verify model access (HTTP {error.code}).") from error
        except OSError as error:
            raise RuntimeError("Could not reach the Anthropic API to verify the key.") from error
        self.api_key = key

    def reset_for_goal(self, goal: str) -> None:
        self._active_goal = goal
        self.step_count = 0
        self.history.clear()
        self._cached_decision = None
        self._plan_step = 0
        self._waypoints.clear()
        self._plan.clear()
        self._waypoint_index = 0

    def _waypoint_decision(self, req: ValidatedRequest) -> PlannerDecision:
        decision = _decision_from_parsed(req, {**self._plan, **self._waypoints[self._waypoint_index]})
        decision.waypoint_index = self._waypoint_index + 1
        decision.waypoint_count = len(self._waypoints)
        return decision

    def build_request_body(self, req: ValidatedRequest) -> dict[str, Any]:
        # Claude 5.5 omits temperature, budget_tokens, and assistant prefill.
        content = [
            {"type": "text", "text": _state_summary(req, self.history)},
            {"type": "text", "text": "Camera 1: front (robot RIGHT arm is on LEFT of image, LEFT arm on RIGHT)"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": req.images_b64["front"]},
            },
            {"type": "text", "text": "Camera 2: left_wrist"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": req.images_b64["left_wrist"]},
            },
            {"type": "text", "text": "Camera 3: right_wrist"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": req.images_b64["right_wrist"]},
            },
        ]
        return {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        }

    def _post_messages(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(2):
            req = urllib.request.Request(f"{self.base_url}/v1/messages", data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if err.code in (401, 403):
                    raise AuthError(f"Anthropic API authentication failed (HTTP {err.code}). Check ANTHROPIC_API_KEY.") from err
                if err.code in (429, 500, 502, 503, 529) and attempt == 0:
                    time.sleep(0.75)
                    last_err = RuntimeError(f"HTTP {err.code}: {err_body[:300]}")
                    continue
                raise RuntimeError(f"Anthropic API error HTTP {err.code}: {err_body[:300]}") from err
            except Exception as exc:
                last_err = exc
                if attempt == 0:
                    time.sleep(0.4)
                    continue
        raise RuntimeError(f"Anthropic API call failed: {last_err}")

    def decide(self, req: ValidatedRequest) -> PlannerDecision:
        if req.instruction != self._active_goal:
            self.reset_for_goal(req.instruction)

        if self._cached_decision is not None and self._cached_decision.status == "completed":
            return hold_pose_decision(
                req,
                stage=self._cached_decision.stage,
                status="completed",
                reason=self._cached_decision.reason or "Task completed; holding current pose.",
            )

        self.step_count += 1
        if self.step_count - self._plan_step < self.plan_every:
            if self._waypoints:
                while self._waypoint_index < len(self._waypoints):
                    decision = self._waypoint_decision(req)
                    if not _reached_waypoint(req, decision):
                        return decision
                    self.history[-1]["reached_waypoints"] = decision.waypoint_index
                    self._waypoint_index += 1
            elif self._cached_decision is not None and self._cached_decision.status == "in_progress":
                return self._cached_decision

        # Check the active waypoint against the latest observation even when the
        # replanning interval has expired.
        if self._waypoint_index < len(self._waypoints):
            decision = self._waypoint_decision(req)
            if _reached_waypoint(req, decision):
                self.history[-1]["reached_waypoints"] = decision.waypoint_index
        self._cached_decision = None
        self._waypoints.clear()
        self._waypoint_index = 0

        if not self.api_key:
            raise AuthError("ANTHROPIC_API_KEY is not set.")

        body = self._post_messages(self.build_request_body(req))
        if body.get("stop_reason") in {"refusal", "max_tokens"}:
            raise RuntimeError("Claude did not return a complete plan")

        text_parts = [blk.get("text", "") for blk in body.get("content", []) if blk.get("type") == "text"]
        parsed = extract_json_object("\n".join(text_parts))
        status = parsed.get("status", "in_progress")
        if status in ("completed", "retry"):
            decision = hold_pose_decision(
                req, stage=str(parsed.get("stage", "hold")), status=status, reason=str(parsed.get("reason", ""))
            )
        elif status == "in_progress" and "waypoints" in parsed:
            self._waypoints = _validated_waypoints(parsed, req.gripper_range)
            self._plan = {key: parsed[key] for key in ("stage", "status", "reason") if key in parsed}
            decision = self._waypoint_decision(req)
        elif status == "in_progress" and any(arm in parsed for arm in ("left_ee", "right_ee")):
            decision = _decision_from_parsed(req, parsed)
        else:
            raise ValueError("Claude returned an invalid plan status or no arm targets")
        self._plan_step = self.step_count
        self.history.append(
            {
                "step": self.step_count,
                "stage": decision.stage,
                "status": decision.status,
                "reason": decision.reason,
                "reached_waypoints": 0,
                "planned_waypoints": decision.waypoint_count,
            }
        )
        self._cached_decision = decision
        return decision

    def decide_sequence(self, req: ValidatedRequest) -> list[PlannerDecision]:
        """Return all planned sequential waypoints for unbroken direct-SDK execution."""
        first = self.decide(req)
        if first.status in ("completed", "retry") or not self._waypoints:
            return [first]
        decisions = [first]
        for idx in range(self._waypoint_index + 1, len(self._waypoints)):
            item = _decision_from_parsed(req, {**self._plan, **self._waypoints[idx]})
            item.waypoint_index = idx + 1
            item.waypoint_count = len(self._waypoints)
            decisions.append(item)
        self._waypoint_index = len(self._waypoints)
        if self.history:
            self.history[-1]["reached_waypoints"] = len(self._waypoints)
        return decisions


class HeuristicOfflinePlanner:
    """Zero-API offline planner for testing the WebSocket + CPU simulator loop."""

    def __init__(self) -> None:
        self.step_count = 0
        self._active_goal: str | None = None

    def decide(self, req: ValidatedRequest) -> PlannerDecision:
        if req.instruction != self._active_goal:
            self._active_goal = req.instruction
            self.step_count = 0

        self.step_count += 1
        _lo, hi = req.gripper_range
        open_g = 0.85 * hi
        close_g = 0.05 * hi
        instr = req.instruction.lower()

        if "hold" in instr or "stop" in instr:
            return hold_pose_decision(req, stage="hold", status="completed", reason="Holding current dual-arm poses.")

        phase = (self.step_count - 1) % 9
        if phase == 0:
            return PlannerDecision(
                stage="approach_above",
                status="in_progress",
                reason="Moving left arm above target object and opening gripper.",
                left_target_pos=[0.36, 0.10, 0.23],
                left_target_rpy=[0.0, 0.4, 0.0],
                left_gripper=open_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 1:
            return PlannerDecision(
                stage="descend_mid",
                status="in_progress",
                reason="Lowering left gripper toward object.",
                left_target_pos=[0.36, 0.10, 0.16],
                left_target_rpy=[0.0, 0.4, 0.0],
                left_gripper=open_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 2:
            return PlannerDecision(
                stage="descend_object",
                status="in_progress",
                reason="Positioning left gripper around object.",
                left_target_pos=[0.36, 0.10, 0.10],
                left_target_rpy=[0.0, 0.4, 0.0],
                left_gripper=open_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 3:
            return PlannerDecision(
                stage="grasp_object",
                status="in_progress",
                reason="Closing left gripper to grasp object.",
                left_target_pos=[0.36, 0.10, 0.10],
                left_target_rpy=[0.0, 0.4, 0.0],
                left_gripper=close_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 4:
            return PlannerDecision(
                stage="lift_object",
                status="in_progress",
                reason="Lifting grasped object safely.",
                left_target_pos=[0.36, 0.06, 0.16],
                left_target_rpy=[0.0, 0.35, 0.0],
                left_gripper=close_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 5:
            return PlannerDecision(
                stage="transport_object",
                status="in_progress",
                reason="Translating object toward receptacle.",
                left_target_pos=[0.36, -0.01, 0.17],
                left_target_rpy=[0.0, 0.35, 0.0],
                left_gripper=close_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 6:
            return PlannerDecision(
                stage="align_bowl",
                status="in_progress",
                reason="Aligning object directly over bowl center.",
                left_target_pos=[0.36, -0.04, 0.14],
                left_target_rpy=[0.0, 0.35, 0.0],
                left_gripper=close_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        if phase == 7:
            return PlannerDecision(
                stage="place_and_release",
                status="in_progress",
                reason="Opening gripper to release object into bowl.",
                left_target_pos=[0.36, -0.04, 0.14],
                left_target_rpy=[0.0, 0.35, 0.0],
                left_gripper=open_g,
                right_target_pos=list(req.right_ee.position),
                right_target_rpy=list(req.right_ee.rotation_rpy),
                right_gripper=open_g,
            )
        return PlannerDecision(
            stage="retreat",
            status="completed",
            reason="Retracting left gripper after release.",
            left_target_pos=[0.32, 0.02, 0.20],
            left_target_rpy=[0.0, 0.25, 0.0],
            left_gripper=open_g,
            right_target_pos=list(req.right_ee.position),
            right_target_rpy=list(req.right_ee.rotation_rpy),
            right_gripper=open_g,
        )
