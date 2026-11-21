"""Protocol validation and formatting for the quanta_x1_pro public_v2 WebSocket API."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    HAS_PIL = False

SUPPORTED_EMBODIMENTS: dict[str, tuple[float, float]] = {
    "quanta_x1_pro": (0.0, 1.526),
    "quanta_x1_pro/gripper-g": (0.0, 1.526),
    "quanta_x1_pro/gripper-h": (0.0, 5.5),
}

REQUIRED_CAMERAS = ("front", "left_wrist", "right_wrist")
REQUIRED_ARMS = ("left_ee", "right_ee")
ACTION_HORIZON = 32
ROW_DT_S = 0.0337
VALID_RTC_DELAYS = {1, 2, 3, 4}


class ProtocolError(Exception):
    """Error matching the public_v2 server error schema."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass
class ArmState:
    position: list[float]
    rotation_rpy: list[float]
    gripper: float


@dataclass
class ValidatedRequest:
    embodiment: str
    gripper_range: tuple[float, float]
    instruction: str
    images_b64: dict[str, str]
    left_ee: ArmState
    right_ee: ArmState
    use_rtc: bool = False
    inference_delay: int | None = None
    prev_action_chunk: dict[str, Any] | None = None


def handshake_metadata() -> dict[str, str]:
    return {
        "model": "wall-x",
        "model_version": "2610-dualhead-v2",
        "backend": "claude-cpu-endpose",
    }


def _strip_data_uri(b64_str: str) -> str:
    if "," in b64_str and b64_str.lstrip().startswith("data:image/"):
        return b64_str.split(",", 1)[1]
    return b64_str


def normalize_jpeg_base64(raw: Any, camera_name: str, max_dim: int = 0) -> str:
    """Validate JPEG stream (accepting base64 str, data URI, or raw JPEG bytes)."""
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)
        if not raw_bytes:
            raise ProtocolError("missing_field", f"Empty camera stream: {camera_name}")
        if HAS_PIL:
            try:
                img = Image.open(io.BytesIO(raw_bytes))
                img.load()
                img = img.convert("RGB")
                if max_dim > 0 and max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                    return encode_jpeg_base64(img, quality=80)
            except Exception as exc:
                raise ProtocolError("image_decode_failed", f"Camera {camera_name} invalid image bytes: {exc}") from exc
        return base64.b64encode(raw_bytes).decode("ascii")

    if not isinstance(raw, str) or not raw.strip():
        raise ProtocolError("missing_field", f"Missing or empty camera stream: {camera_name}")
    payload = _strip_data_uri(raw.strip())
    try:
        img_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ProtocolError("image_decode_failed", f"Camera {camera_name} invalid base64: {exc}") from exc

    if HAS_PIL:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.load()
            img = img.convert("RGB")
            if max_dim > 0 and max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim))
                return encode_jpeg_base64(img, quality=80)
            return payload
        except Exception as exc:
            raise ProtocolError("image_decode_failed", f"Camera {camera_name} invalid JPEG: {exc}") from exc

    if len(img_bytes) < 4 or img_bytes[:2] != b"\xff\xd8":
        raise ProtocolError("image_decode_failed", f"Camera {camera_name} does not start with JPEG SOI header")
    return payload


def encode_jpeg_base64(img: Any, quality: int = 85) -> str:
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to encode images")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_arm_state(raw_arm: Any, arm_name: str, gripper_range: tuple[float, float]) -> ArmState:
    if not isinstance(raw_arm, dict):
        raise ProtocolError("missing_field", f"Missing state dict for {arm_name}")
    if "rotation_6d" in raw_arm:
        raise ProtocolError("normalization_conflict", f"{arm_name} must use rotation_rpy, not rotation_6d")

    try:
        pos = np.asarray(raw_arm.get("position"), dtype=float)
        rpy = np.asarray(raw_arm.get("rotation_rpy"), dtype=float)
        grip_arr = np.asarray(raw_arm.get("gripper"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("missing_field", f"Invalid numeric state for {arm_name}: {exc}") from exc

    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ProtocolError("missing_field", f"{arm_name}.position must be a finite 3-element vector")
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ProtocolError("missing_field", f"{arm_name}.rotation_rpy must be a finite 3-element vector")
    if grip_arr.shape != () or not np.isfinite(grip_arr):
        raise ProtocolError("missing_field", f"{arm_name}.gripper must be a finite scalar")

    lo, hi = gripper_range
    g_val = float(max(lo, min(hi, float(grip_arr))))
    return ArmState(
        position=[float(pos[0]), float(pos[1]), float(pos[2])],
        rotation_rpy=[float(rpy[0]), float(rpy[1]), float(rpy[2])],
        gripper=g_val,
    )


def validate_request(payload: Any, max_image_dim: int = 640) -> ValidatedRequest:
    if not isinstance(payload, dict):
        raise ProtocolError("missing_field", "Request payload must be a dictionary")

    if "norm_key" in payload or "views" in payload:
        raise ProtocolError("mixed_schema", "Do not mix norm_key or views with embodiment/observation schema")
    if "normalizer" in payload or "dof_mask" in payload or "robot_id" in payload:
        raise ProtocolError("normalization_conflict", "Do not pass normalizer, dof_mask, or robot_id")

    embodiment = payload.get("embodiment")
    if not isinstance(embodiment, str) or embodiment not in SUPPORTED_EMBODIMENTS:
        raise ProtocolError("unknown_embodiment", f"Unsupported embodiment: {embodiment!r}")
    gripper_range = SUPPORTED_EMBODIMENTS[embodiment]

    norm = payload.get("normalization")
    if norm != "provided":
        raise ProtocolError("normalization_conflict", "normalization must be set to 'provided'")

    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProtocolError("missing_field", "instruction must be a non-empty string")

    obs = payload.get("observation")
    if not isinstance(obs, dict):
        raise ProtocolError("missing_field", "Missing observation object")

    images_raw = obs.get("images")
    if not isinstance(images_raw, dict):
        raise ProtocolError("missing_field", "Missing observation.images object")

    clean_b64: dict[str, str] = {}
    for cam in REQUIRED_CAMERAS:
        if cam not in images_raw:
            raise ProtocolError("missing_field", f"Missing required camera stream: {cam}")
        clean_b64[cam] = normalize_jpeg_base64(images_raw[cam], cam, max_dim=max_image_dim)

    state_raw = obs.get("state")
    if not isinstance(state_raw, dict):
        raise ProtocolError("missing_field", "Missing observation.state object")

    left_ee = _parse_arm_state(state_raw.get("left_ee"), "left_ee", gripper_range)
    right_ee = _parse_arm_state(state_raw.get("right_ee"), "right_ee", gripper_range)

    use_rtc = bool(payload.get("use_rtc", False))
    inference_delay = payload.get("inference_delay")
    prev_action_chunk = payload.get("prev_action_chunk")

    if use_rtc:
        if not isinstance(inference_delay, int) or inference_delay not in VALID_RTC_DELAYS:
            raise ProtocolError(
                "rtc_delay_exceeds_training",
                "inference_delay must be an integer in {1, 2, 3, 4} when use_rtc is true",
            )
        if not isinstance(prev_action_chunk, dict):
            raise ProtocolError("missing_field", "prev_action_chunk is required when use_rtc is true")
        for arm in REQUIRED_ARMS:
            block = prev_action_chunk.get(arm)
            if not isinstance(block, dict):
                raise ProtocolError("missing_field", f"prev_action_chunk missing {arm}")
            for key in ("position", "rotation_rpy", "gripper"):
                seq = block.get(key)
                if seq is None or len(seq) < 2:
                    raise ProtocolError("missing_field", f"prev_action_chunk.{arm}.{key} needs at least 2 frames")

    return ValidatedRequest(
        embodiment=embodiment,
        gripper_range=gripper_range,
        instruction=instruction.strip(),
        images_b64=clean_b64,
        left_ee=left_ee,
        right_ee=right_ee,
        use_rtc=use_rtc,
        inference_delay=inference_delay,
        prev_action_chunk=prev_action_chunk,
    )


def validate_response(response: dict[str, Any], embodiment: str) -> None:
    if response.get("embodiment") != embodiment:
        raise ValueError("Response embodiment mismatch")
    actions = response.get("actions")
    if not isinstance(actions, dict):
        raise ValueError("Missing actions dict")
    for arm in REQUIRED_ARMS:
        block = actions[arm]
        pos = np.asarray(block["position"])
        rpy = np.asarray(block["rotation_rpy"])
        grip = np.asarray(block["gripper"])
        if pos.shape != (ACTION_HORIZON, 3) or rpy.shape != (ACTION_HORIZON, 3) or grip.shape != (ACTION_HORIZON,):
            raise ValueError(f"Unexpected action shape for {arm}")
        if not (np.array_equal(pos[30], pos[31]) and np.array_equal(rpy[30], rpy[31]) and grip[30] == grip[31]):
            raise ValueError(f"Frame 31 must repeat frame 30 for {arm}")
