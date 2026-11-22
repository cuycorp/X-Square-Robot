"""Forward/inverse kinematics for the desktop robot arms, built from the robot's own URDF. Pure numpy.

Frames (verified against the live robot, 0.0 mm error):
  base_link ──(fixed)──> {side}_arm_base_link ──joints 1..6──> {side}_arm_link6 (gripper base)
  /{side}_arm/end_pose  = FK(0)^-1 · FK(q)   (pose relative to the zero-joint "neutral" pose; frame
                          {side}_arm_neutral, whose axes equal the arm base axes)
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ARMS = ("left", "right")


# ---------- rotation helpers (R = Rz(yaw)·Ry(pitch)·Rx(roll), as in the Wall-X API) ----------

def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) < 1e-9:  # gimbal lock: put everything in yaw
        return np.array([0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def quat_to_matrix(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    """Returns [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i], q[j], q[k], q[3] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    return q


def transform(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def axis_angle(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rotation_error(R_target, R_current):
    """Axis-angle vector (rad) rotating R_current onto R_target, in the base frame."""
    R = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(axis)
    if s < 1e-9:  # angle ~ pi
        axis = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        return axis / np.linalg.norm(axis) * angle
    return axis / s * angle


# ---------- URDF ----------

@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray        # 4x4, parent -> joint frame at q = 0
    axis: np.ndarray
    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = 0.0


def _floats(text: str | None, default: str) -> list:
    return [float(v) for v in (text or default).split()]


def urdf_joints(urdf_xml: str) -> dict:
    """{child_link: Joint} for the top-level <joint> elements (ros2_control <joint> tags are skipped)."""
    joints = {}
    for j in ET.fromstring(urdf_xml).findall("joint"):
        parent, child = j.find("parent"), j.find("child")
        if parent is None or child is None:
            continue
        o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
        xyz = _floats(o.get("xyz") if o is not None else None, "0 0 0")
        rpy = _floats(o.get("rpy") if o is not None else None, "0 0 0")
        joints[child.get("link", "")] = Joint(
            name=j.get("name", ""), type=j.get("type", "fixed"),
            parent=parent.get("link", ""), child=child.get("link", ""),
            origin=transform(rpy_to_matrix(*rpy), np.array(xyz)),
            axis=np.array(_floats(ax.get("xyz") if ax is not None else None, "0 0 1")),
            lower=float(lim.get("lower", "-inf")) if lim is not None else -np.inf,
            upper=float(lim.get("upper", "inf")) if lim is not None else np.inf,
            velocity=float(lim.get("velocity", "0")) if lim is not None else 0.0)
    return joints


def static_transform(urdf_xml: str, ancestor: str, link: str) -> np.ndarray:
    """Pose of `link` in `ancestor` with every joint in between at q = 0."""
    joints, T = urdf_joints(urdf_xml), np.eye(4)
    while link != ancestor:
        if link not in joints:
            raise ValueError(f"{link} is not below {ancestor}")
        T = joints[link].origin @ T
        link = joints[link].parent
    return T


def arm_base_in_world(urdf_xml: str, side: str) -> np.ndarray:
    return static_transform(urdf_xml, "base_link", f"{side}_arm_base_link")


def fingertip_offset(urdf_xml: str, side: str) -> np.ndarray:
    """Fingertip centre in the link6 frame: tip x/z from the finger link, centred between the fingers (y = 0)."""
    p = static_transform(urdf_xml, f"{side}_arm_link6", f"{side}_arm_gripper_left_tip_link")[:3, 3]
    return np.array([p[0], 0.0, p[2]])


class Arm:
    """Serial chain from <side>_arm_base_link to <side>_arm_link6."""

    def __init__(self, urdf_xml: str, side: str):
        if side not in ARMS:
            raise ValueError(side)
        self.side, self.base, self.tip = side, f"{side}_arm_base_link", f"{side}_arm_link6"
        joints, chain, link = urdf_joints(urdf_xml), [], self.tip
        while link != self.base:
            if link not in joints:
                raise ValueError(f"no chain from {self.base} to {self.tip}")
            if joints[link].type not in ("revolute", "continuous", "fixed"):
                raise ValueError(f"unsupported joint type {joints[link].type}")
            chain.append(joints[link])
            link = joints[link].parent
        self.chain = chain[::-1]
        self.joints = [j for j in self.chain if j.type != "fixed"]
        self.names = [j.name for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])

    def fk_all(self, q):
        """[(link_name, 4x4 pose in the arm base frame)] for every link along the chain."""
        T, out, i = np.eye(4), [], 0
        for j in self.chain:
            T = T @ j.origin
            if j.type != "fixed":
                T = T @ transform(axis_angle(j.axis, q[i]), np.zeros(3))
                i += 1
            out.append((j.child, T.copy()))
        return out

    def fk(self, q):
        return self.fk_all(np.asarray(q, float))[-1][1]

    def jacobian(self, q, eps=1e-6):
        q = np.asarray(q, float)
        T0, J = self.fk(q), np.zeros((6, len(q)))
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += eps
            T1 = self.fk(dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = rotation_error(T1[:3, :3], T0[:3, :3]) / eps
        return J

    def ik(self, T_target, q_seed, iters=200, damping=0.05, pos_tol=1e-3, rot_tol=np.radians(0.5)):
        """Damped least squares IK, clamped to joint limits. Returns (q, pos_err_m, rot_err_rad)."""
        q = np.clip(np.asarray(q_seed, float), self.lower, self.upper)
        for _ in range(iters):
            T = self.fk(q)
            e = np.concatenate([T_target[:3, 3] - T[:3, 3], rotation_error(T_target[:3, :3], T[:3, :3])])
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
                break
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lower, self.upper)
        T = self.fk(q)
        return (q, float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])),
                float(np.linalg.norm(rotation_error(T_target[:3, :3], T[:3, :3]))))
