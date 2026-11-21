"""Autonomous Vision-to-EndPose planners using Anthropic Claude (zero local GPU).

Uses only Python stdlib urllib.request so it runs in minimal robot environments.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from protocol import ValidatedRequest


DEFAULT_CLAUDE_KEY_FILE = Path.home() / ".config" / "anthropic" / "api_key"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"

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
2. Keep per-step translation deltas modest (<= 0.08 m) and rotation deltas modest (<= 0.40 rad). Each output commands ~1.0 s (32 frames).
3. Use visual feedback from all 3 cameras + current EE poses. If a grasp missed (visible gap in wrist camera), set status to "retry" and re-align before lifting.
4. Use the closer/most appropriate arm (or both arms for bimanual tasks). Keep an idle arm stationary (delta [0, 0, 0]).
5. Only output status="completed" when the goal is visually satisfied and the object is stably placed/released (or held as requested). Do not claim hardware has powered off; the server will safely hold current pose while waiting for the next prompt or operator stop in WebUI.

Respond strictly with ONE valid JSON object (no markdown fences or commentary):
{
  "stage": "short_stage_label",
  "status": "in_progress | completed | retry",
  "reason": "brief observation-grounded explanation",
  "left_ee": {
    "delta_position": [dx, dy, dz],
    "delta_rpy": [droll, dpitch, dyaw],
    "gripper": 1.2
  },
  "right_ee": {
    "delta_position": [dx, dy, dz],
    "delta_rpy": [droll, dpitch, dyaw],
    "gripper": 1.2
  }
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
    return json.loads(cleaned)


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
        tgt_pos = [float(cmd["position"][i]) for i in range(3)]
    else:
        dp = cmd.get("delta_position", [0.0, 0.0, 0.0])
        if not isinstance(dp, (list, tuple)) or len(dp) != 3:
            dp = [0.0, 0.0, 0.0]
        tgt_pos = [float(current_pos[i]) + float(dp[i]) for i in range(3)]

    if "rotation_rpy" in cmd and isinstance(cmd["rotation_rpy"], (list, tuple)) and len(cmd["rotation_rpy"]) == 3:
        tgt_rpy = [float(cmd["rotation_rpy"][i]) for i in range(3)]
    else:
        dr = cmd.get("delta_rpy", [0.0, 0.0, 0.0])
        if not isinstance(dr, (list, tuple)) or len(dr) != 3:
            dr = [0.0, 0.0, 0.0]
        tgt_rpy = [float(current_rpy[i]) + float(dr[i]) for i in range(3)]

    lo, hi = gripper_range
    g_raw = cmd.get("gripper", current_grip)
    tgt_grip = max(lo, min(hi, float(g_raw)))
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
        }
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
        plan_every: int = 1,
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

    def reset_for_goal(self, goal: str) -> None:
        self._active_goal = goal
        self.step_count = 0
        self.history.clear()
        self._cached_decision = None

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
            "max_tokens": 512,
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
        if self._cached_decision is not None and (self.step_count - 1) % self.plan_every != 0:
            return self._cached_decision

        if not self.api_key:
            raise AuthError("ANTHROPIC_API_KEY is not set.")

        body = self._post_messages(self.build_request_body(req))
        if body.get("stop_reason") == "refusal":
            raise RuntimeError("Claude refused request")

        text_parts = [blk.get("text", "") for blk in body.get("content", []) if blk.get("type") == "text"]
        parsed = extract_json_object("\n".join(text_parts))
        decision = _decision_from_parsed(req, parsed)
        if decision.status == "completed":
            decision = hold_pose_decision(req, stage=decision.stage, status="completed", reason=decision.reason)
        self.history.append(
            {
                "step": self.step_count,
                "stage": decision.stage,
                "status": decision.status,
                "reason": decision.reason,
            }
        )
        self._cached_decision = decision
        return decision


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
