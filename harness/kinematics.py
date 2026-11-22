"""Read-only Cartesian IK preview for the desktop arm chains in a ROS URDF."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from trajectory import _so3_exp, _so3_log, matrix_to_rpy, rpy_to_matrix


@dataclass(frozen=True)
class Joint:
    name: str
    transform: np.ndarray
    axis: np.ndarray | None
    bounds: tuple[float, float] = (0.0, 0.0)


def _vector(values: Any, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {count} finite numbers")
    return array


def arm_chain(urdf: str, arm: str) -> list[Joint]:
    if arm not in ("left", "right"):
        raise ValueError("Select the left or right arm")
    if not isinstance(urdf, str) or "<!" in urdf.replace("<!--", ""):
        raise ValueError("Expected a robot URDF without external declarations")
    root = ET.fromstring(urdf)
    if root.tag != "robot":
        raise ValueError("Expected a robot URDF")
    by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            name = child.attrib["link"]
            if name in by_child:
                raise ValueError("Each URDF link must have one parent")
            by_child[name] = joint

    link = f"{arm}_arm_gripper_base_link"
    chain = []
    visited = set()
    while link != "base_link":
        if link in visited or link not in by_child:
            raise ValueError(f"No connected {arm} arm chain from base_link to the gripper base")
        visited.add(link)
        joint = by_child[link]
        parent = joint.find("parent").attrib["link"]
        origin = joint.find("origin")
        attributes = origin.attrib if origin is not None else {}
        transform = np.eye(4)
        transform[:3, 3] = _vector(attributes.get("xyz", "0 0 0").split(), 3)
        transform[:3, :3] = rpy_to_matrix(_vector(attributes.get("rpy", "0 0 0").split(), 3))
        axis = None
        bounds = (0.0, 0.0)
        if joint.attrib["type"] == "revolute":
            axis_tag = joint.find("axis")
            axis = _vector(axis_tag.attrib["xyz"].split() if axis_tag is not None else [1, 0, 0], 3)
            norm = np.linalg.norm(axis)
            if not 0 < norm < math.inf:
                raise ValueError("Invalid joint axis")
            axis /= norm
            limit = joint.find("limit")
            low, high = _vector([limit.attrib["lower"], limit.attrib["upper"]], 2)
            if low >= high:
                raise ValueError("Invalid revolute joint limits")
            bounds = (low, high)
        elif joint.attrib["type"] != "fixed":
            raise ValueError("The preview supports fixed and revolute links on the arm chain")
        chain.append(Joint(joint.attrib["name"], transform, axis, bounds))
        link = parent

    chain.reverse()
    names = [joint.name for joint in chain if joint.axis is not None]
    if names != [f"{arm}_arm_joint{i}" for i in range(1, 7)]:
        raise ValueError("The selected arm must have the six expected desktop joints")
    return chain


def _forward(chain: list[Joint], angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.eye(4)
    axes, pivots = [], []
    for joint in chain:
        transform = transform @ joint.transform
        if joint.axis is not None:
            rotation = _so3_exp(joint.axis * angles[len(axes)])
            axes.append(transform[:3, :3] @ joint.axis)
            pivots.append(transform[:3, 3].copy())
            transform[:3, :3] = transform[:3, :3] @ rotation
    jacobian = np.vstack((np.cross(axes, transform[:3, 3] - pivots).T, np.asarray(axes).T))
    return transform, jacobian


def _joint_state(chain: list[Joint], measured: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    movable = [joint for joint in chain if joint.axis is not None]
    names = [joint.name for joint in movable]
    lower, upper = np.asarray([joint.bounds for joint in movable]).T
    if not isinstance(measured, dict) or any(name not in measured for name in names):
        raise ValueError("The latest joint readings must include all six joints")
    angles = _vector([measured[name] for name in names], len(names))
    return names, angles, lower, upper


def _solve(
    chain: list[Joint],
    names: list[str],
    angles: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    desired: np.ndarray,
    goal_rotation: np.ndarray,
    position_tolerance: float = 0.002,
    rotation_tolerance: float = 0.02,
) -> dict[str, Any]:
    angles = np.clip(angles, lower, upper)
    weights = np.array([1, 1, 1, 0.2, 0.2, 0.2])
    damping = 0.025 ** 2 * np.eye(6)
    for iteration in range(81):
        current, jacobian = _forward(chain, angles)
        linear = desired - current[:3, 3]
        angular = _so3_log(goal_rotation @ current[:3, :3].T)
        position_error = float(np.linalg.norm(linear))
        rotation_error = float(np.linalg.norm(angular))
        converged = position_error <= position_tolerance and rotation_error <= rotation_tolerance
        if converged or iteration == 80:
            break
        weighted_jacobian = jacobian * weights[:, None]
        residual = np.concatenate((linear, angular)) * weights
        step = weighted_jacobian.T @ np.linalg.solve(weighted_jacobian @ weighted_jacobian.T + damping, residual)
        angles = np.clip(angles + np.clip(step, -0.25, 0.25), lower, upper)

    return {
        "converged": converged,
        "joints": dict(zip(names, angles.tolist())) if converged else {},
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
    }


def preview(urdf: str, arm: str, measured: dict[str, float], target: list[float] | None) -> dict[str, Any]:
    chain = arm_chain(urdf, arm)
    names, angles, lower, upper = _joint_state(chain, measured)
    if target is None:
        current, _ = _forward(chain, angles)
        return {"pose": [*current[:3, 3].tolist(), *matrix_to_rpy(current[:3, :3])]}
    desired = _vector(target, 6)
    return _solve(chain, names, angles, lower, upper, desired[:3], rpy_to_matrix(desired[3:]))


def measured_pose(chain: list[Joint], measured: dict[str, float]) -> tuple[list[float], list[float]]:
    _, angles, _, _ = _joint_state(chain, measured)
    transform, _ = _forward(chain, angles)
    return transform[:3, 3].tolist(), matrix_to_rpy(transform[:3, :3])


def small_pose_target(
    chain: list[Joint], measured: dict[str, float], position: np.ndarray, rotation: np.ndarray
) -> dict[str, float]:
    names, angles, lower, upper = _joint_state(chain, measured)
    solution = _solve(chain, names, angles, lower, upper, position, rotation, 0.00025, 0.003)
    if not solution["converged"]:
        raise ValueError("No joint solution for this increment")
    joints = solution["joints"]
    if max(abs(joints[name] - angle) for name, angle in zip(names, angles)) > math.radians(3):
        raise ValueError("This Cartesian step requires more than 3 degrees at a joint. Try a smaller increment.")
    return joints


def jog_target(
    chain: list[Joint], measured: dict[str, float], axis: str, direction: int, step: float
) -> dict[str, float]:
    names, angles, lower, upper = _joint_state(chain, measured)
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("Choose a positive or negative increment")
    if type(step) not in (int, float):
        raise ValueError("Choose an increment from the step menu")
    if axis in ("x", "y", "z"):
        if step not in (1, 2, 5):
            raise ValueError("Choose a translation step of 1, 2 or 5 mm")
        increment = direction * step / 1000
    else:
        if step not in (0.5, 1, 2):
            raise ValueError("Choose an angular step of 0.5, 1 or 2 degrees")
        increment = math.radians(direction * step)

    if axis in ("j1", "j2", "j3", "j4", "j5", "j6"):
        target = angles.copy()
        target[int(axis[1]) - 1] += increment
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError("This step would exceed a joint limit")
        return dict(zip(names, target.tolist()))

    axes = ("x", "y", "z", "rx", "ry", "rz")
    if axis not in axes:
        raise ValueError("Unknown jog axis")
    current, _ = _forward(chain, angles)
    position, rotation = current[:3, 3].copy(), current[:3, :3]
    axis_index = axes.index(axis)
    if axis_index < 3:
        position[axis_index] += increment
    else:
        rotation = _so3_exp(np.eye(3)[axis_index - 3] * increment) @ rotation
    return small_pose_target(chain, measured, position, rotation)
