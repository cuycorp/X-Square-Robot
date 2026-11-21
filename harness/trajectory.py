"""CPU SE(3) and gripper trajectory synthesis matching quanta_x1_pro conventions.

Rotation convention from the public_v2 specification:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
Each chunk contains 32 frames at row_dt_s = 0.0337 s, where frame 31 repeats frame 30.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from protocol import ACTION_HORIZON, ArmState


@dataclass
class WorkspaceLimits:
    # Generous base-frame bounds so different URDF base origins do not cause spurious snaps;
    # per-chunk relative step limits enforce smooth, safe motion.
    x_min: float = -0.25
    x_max: float = 0.90
    y_min: float = -0.80
    y_max: float = 0.80
    z_min: float = -0.40
    z_max: float = 0.90
    max_step_pos_m: float = 0.08
    max_step_rot_rad: float = 0.45


def rpy_to_matrix(rpy: list[float] | np.ndarray) -> np.ndarray:
    """Convert [roll, pitch, yaw] to 3x3 rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_rpy(r: np.ndarray) -> list[float]:
    """Extract [roll, pitch, yaw] from R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    sy = -float(np.clip(r[2, 0], -1.0, 1.0))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-6:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        yaw = 0.0
    return [roll, pitch, yaw]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _so3_log(r: np.ndarray) -> np.ndarray:
    """Logarithm map on SO(3) returning axis-angle vector omega."""
    tr = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    theta = math.acos(cos_theta)
    if theta < 1e-7:
        return np.array(
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            dtype=np.float64,
        ) * 0.5
    if abs(math.pi - theta) < 1e-4:
        diag = np.maximum(0.0, np.diag(r) + 1.0)
        axis = np.sqrt(0.5 * diag)
        if r[2, 1] - r[1, 2] < 0:
            axis[0] = -abs(axis[0])
        if r[0, 2] - r[2, 0] < 0:
            axis[1] = -abs(axis[1])
        if r[1, 0] - r[0, 1] < 0:
            axis[2] = -abs(axis[2])
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            return theta * (axis / norm)
    factor = theta / (2.0 * math.sin(theta))
    return factor * np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    )


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) via Rodrigues formula."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) + _skew(omega)
    k = omega / theta
    kx = _skew(k)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def _min_jerk(tau: float) -> float:
    t = max(0.0, min(1.0, tau))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def clamp_pose(
    pos: list[float],
    rpy: list[float],
    gripper: float,
    ref_state: ArmState,
    gripper_range: tuple[float, float],
    limits: WorkspaceLimits,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Clamp target pose to safe workspace bounds and max per-chunk delta."""
    ref_p = np.asarray(ref_state.position, dtype=np.float64)
    tgt_p = np.asarray(pos, dtype=np.float64)
    delta_p = tgt_p - ref_p
    dist = float(np.linalg.norm(delta_p))
    if dist > limits.max_step_pos_m and dist > 1e-9:
        tgt_p = ref_p + delta_p * (limits.max_step_pos_m / dist)

    # Preserve current position if already outside configured box and no further violation occurs
    x_lo = min(limits.x_min, float(ref_p[0]))
    x_hi = max(limits.x_max, float(ref_p[0]))
    y_lo = min(limits.y_min, float(ref_p[1]))
    y_hi = max(limits.y_max, float(ref_p[1]))
    z_lo = min(limits.z_min, float(ref_p[2]))
    z_hi = max(limits.z_max, float(ref_p[2]))
    tgt_p[0] = float(np.clip(tgt_p[0], x_lo, x_hi))
    tgt_p[1] = float(np.clip(tgt_p[1], y_lo, y_hi))
    tgt_p[2] = float(np.clip(tgt_p[2], z_lo, z_hi))

    r_ref = rpy_to_matrix(ref_state.rotation_rpy)
    r_tgt = rpy_to_matrix(rpy)
    omega = _so3_log(r_ref.T @ r_tgt)
    ang = float(np.linalg.norm(omega))
    if ang > limits.max_step_rot_rad and ang > 1e-9:
        omega = omega * (limits.max_step_rot_rad / ang)
        r_tgt = r_ref @ _so3_exp(omega)

    lo, hi = gripper_range
    g_clamped = float(np.clip(gripper, lo, hi))
    return tgt_p, r_tgt, g_clamped


def synthesize_arm_chunk(
    current: ArmState,
    target_pos: list[float],
    target_rpy: list[float],
    target_gripper: float,
    gripper_range: tuple[float, float],
    limits: WorkspaceLimits,
    rtc_prefix_arm: dict[str, Any] | None = None,
    inference_delay: int | None = None,
) -> dict[str, list[Any]]:
    """Generate a 32-frame trajectory (frames 0..30 smooth, frame 31 == frame 30)."""
    n_exec = ACTION_HORIZON - 1
    prefix_len = 0
    positions: list[list[float]] = []
    rpys: list[list[float]] = []
    grippers: list[float] = []

    if rtc_prefix_arm is not None and inference_delay is not None:
        unexecuted_pos = rtc_prefix_arm["position"][1:]
        unexecuted_rpy = rtc_prefix_arm["rotation_rpy"][1:]
        unexecuted_grip = rtc_prefix_arm["gripper"][1:]
        prefix_len = min(inference_delay, len(unexecuted_pos), n_exec - 1)
        for i in range(prefix_len):
            positions.append([float(v) for v in unexecuted_pos[i]])
            rpys.append([float(v) for v in unexecuted_rpy[i]])
            grippers.append(float(unexecuted_grip[i]))

    if prefix_len > 0:
        anchor_state = ArmState(
            position=positions[-1],
            rotation_rpy=rpys[-1],
            gripper=grippers[-1],
        )
    else:
        anchor_state = current

    p0 = np.asarray(anchor_state.position, dtype=np.float64)
    r0 = rpy_to_matrix(anchor_state.rotation_rpy)
    g0 = float(anchor_state.gripper)

    p1, r1, g1 = clamp_pose(
        target_pos, target_rpy, target_gripper, anchor_state, gripper_range, limits
    )
    omega = _so3_log(r0.T @ r1)

    remaining = n_exec - prefix_len
    for idx in range(remaining):
        tau = (idx + 1) / float(remaining)
        s = _min_jerk(tau)
        p_t = p0 + s * (p1 - p0)
        r_t = r0 @ _so3_exp(s * omega)
        g_t = g0 + s * (g1 - g0)
        positions.append([round(float(v), 6) for v in p_t])
        rpys.append([round(float(v), 6) for v in matrix_to_rpy(r_t)])
        grippers.append(round(float(g_t), 6))

    positions.append(list(positions[-1]))
    rpys.append(list(rpys[-1]))
    grippers.append(grippers[-1])

    return {
        "position": positions,
        "rotation_rpy": rpys,
        "gripper": grippers,
    }
