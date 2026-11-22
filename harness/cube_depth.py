"""Localize the cube with synchronized front RGB-D and the station transforms."""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from PIL import Image

from cube_vision import detect_cube_cv
from trajectory import rpy_to_matrix


FRONT_COLOR = "/camera_head_front/color/image_raw/compressed"
DEPTH_TOPICS = {
    "/camera_head_front/color/camera_info": "color_info",
    "/camera_head_front/depth/camera_info": "depth_info",
    "/camera_head_front/depth/image_raw/compressedDepth": "depth",
    "/tf_static": "static",
}


def _transform(rotation: np.ndarray, translation: list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def camera_mount(urdf: str) -> np.ndarray:
    root = ET.fromstring(urdf)
    for joint in root.findall("joint"):
        if (
            joint.get("type") == "fixed"
            and joint.find("parent").get("link") == "base_link"
            and joint.find("child").get("link") == "left_arm_base_link"
        ):
            origin = joint.find("origin")
            xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
            rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
            return _transform(rpy_to_matrix(rpy), xyz)
    raise ValueError("The station URDF has no fixed left arm mount in base_link.")


def _optical_transform(frame: str, static: dict, mount: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    visited = set()
    while frame not in ("base_link", "left_arm_base_link"):
        if frame in visited or frame not in static:
            raise ValueError("Missing fixed transform for the front camera.")
        visited.add(frame)
        item = static[frame]
        tf = item["transform"]
        quaternion = np.array([tf["rotation"][key] for key in ("x", "y", "z", "w")])
        norm = np.linalg.norm(quaternion)
        if not 0.99 <= norm <= 1.01:
            raise ValueError("Invalid camera transform rotation.")
        x, y, z, w = quaternion / norm
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        transform = _transform(rotation, [tf["translation"][key] for key in ("x", "y", "z")]) @ transform
        frame = item["header"]["frame_id"]
    return mount @ transform if frame == "left_arm_base_link" else transform


def _depth_points(depth: dict, info: dict) -> np.ndarray:
    if depth["format"] != "16UC1; compressedDepth png":
        raise ValueError("Expected a 16-bit PNG depth stream in millimetres.")
    raw = base64.b64decode(depth["data"], validate=True)
    if raw[12:20] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Missing compressed depth PNG.")
    with Image.open(io.BytesIO(raw[12:])) as image:
        measured = np.asarray(image, dtype=float) / 1000.0
    if measured.shape != (info["height"], info["width"]) or any(abs(d) > 1e-8 for d in info["d"]):
        raise ValueError("Expected the station's rectified depth stream and matching intrinsics.")
    valid = (measured >= 0.15) & (measured <= 2.0)
    v, u = np.nonzero(valid)
    z = measured[valid]
    k = info["k"]
    return np.column_stack(((u - k[2]) * z / k[0], (v - k[5]) * z / k[4], z))


def _project_color(points: np.ndarray, info: dict) -> np.ndarray:
    x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
    r2 = x*x + y*y
    d = info["d"]
    if info["distortion_model"] not in ("rational_polynomial", "plumb_bob") or len(d) not in (5, 8):
        raise ValueError("Unsupported front camera distortion model.")
    k1, k2, p1, p2, k3, k4, k5, k6 = [*d, *([0.0] * (8 - len(d)))]
    radial = (1 + k1*r2 + k2*r2**2 + k3*r2**3) / (1 + k4*r2 + k5*r2**2 + k6*r2**3)
    k = info["k"]
    u = k[0] * (x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)) + k[2]
    v = k[4] * (y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y) + k[5]
    return np.column_stack((u / info["width"], v / info["height"]))


def localize_cube(rgb: dict, depth: dict, color_info: dict, depth_info: dict, static: dict, mount: np.ndarray) -> dict[str, Any]:
    """Measure the upper face and nearby table in base_link. Raise on weak evidence."""
    if rgb["header"]["frame_id"] != color_info["header"]["frame_id"] or depth["header"]["frame_id"] != depth_info["header"]["frame_id"]:
        raise ValueError("Camera calibration does not match image frame IDs.")
    detection = detect_cube_cv(rgb["data"], source="front")
    top = next((face for face in detection.get("visible_faces", []) if face["face"] == "U"), None)
    if top is None or top["known_stickers"] < 6 or len(detection["visible_faces"]) < 2:
        raise ValueError("Could not verify the cube's top and side in the front view.")
    base_from_depth = _optical_transform(depth["header"]["frame_id"], static, mount)
    base_from_color = _optical_transform(rgb["header"]["frame_id"], static, mount)
    color_from_depth = np.linalg.solve(base_from_color, base_from_depth)
    points = _depth_points(depth, depth_info)
    in_color = points @ color_from_depth[:3, :3].T + color_from_depth[:3, 3]
    visible = in_color[:, 2] > 0.1
    image_uv = _project_color(in_color[visible], color_info)
    base_points = points[visible] @ base_from_depth[:3, :3].T + base_from_depth[:3, 3]
    quad = np.asarray(top["quad"])
    quad = quad * 0.78 + quad.mean(axis=0) * 0.22
    edges = np.roll(quad, -1, axis=0) - quad
    relative = image_uv[:, None, :] - quad
    cross = edges[:, 0] * relative[:, :, 1] - edges[:, 1] * relative[:, :, 0]
    inside = (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)
    samples = base_points[inside]
    if len(samples) < 8:
        raise ValueError("Too few depth samples on the cube's top face.")
    top_z = float(np.median(samples[:, 2]))
    inliers = samples[np.abs(samples[:, 2] - top_z) <= 0.012]
    if len(inliers) < max(8, len(samples) * 0.7):
        raise ValueError("Depth samples disagree on the height of the cube.")
    top_center = np.median(inliers, axis=0)
    radius = np.linalg.norm(base_points[:, :2] - top_center[:2], axis=1)
    table = base_points[
        (radius >= 0.06) & (radius <= 0.12)
        & (base_points[:, 2] < top_z - 0.02) & (base_points[:, 2] > top_z - 0.10), 2
    ]
    if len(table) < 24 or float(np.ptp(np.percentile(table, [20, 80]))) > 0.018:
        raise ValueError("Could not verify the local tabletop height.")
    table_z = float(np.median(table))
    if not 0.035 <= top_z - table_z <= 0.078:
        raise ValueError("Measured cube height is inconsistent with a tabletop Rubik's Cube.")
    return {
        "frame": "base_link",
        "cube_top_center_m": np.round(top_center, 3).tolist(),
        "table_z_m": round(table_z, 3),
        "depth_samples": len(inliers),
    }
