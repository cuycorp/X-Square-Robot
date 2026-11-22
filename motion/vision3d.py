"""Head-camera geometry: pixel <-> 3D in base_link, depth decoding, and locating objects in 3D.

Read-only. Frames (from the robot's /tf_static and camera_info):
  base_link -> left_arm_base_link -> camera_head_front_link (= depth_frame) -> color_frame -> *_optical_frame
"""
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import kinematics as K

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "head_camera.json"
COLOR_INFO = "/camera_head_front/color/camera_info"
DEPTH_INFO = "/camera_head_front/depth/camera_info"
DEPTH_TOPIC = "/camera_head_front/depth/image_raw/compressedDepth"
COLOR_OPTICAL = "camera_head_front_color_optical_frame"
DEPTH_OPTICAL = "camera_head_front_depth_optical_frame"


# ---------- fetching calibration (read-only, cached) ----------

def _recv_until(ws, pred, timeout):
    end = time.monotonic() + timeout
    while True:
        m = json.loads(ws.recv(timeout=max(0.1, end - time.monotonic())))
        if pred(m):
            return m


def _one(ws, topic, typ, timeout=8.0):
    ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": typ, "id": f"v3d{topic}"}))
    try:
        return _recv_until(ws, lambda m: m.get("op") == "publish" and m.get("topic") == topic, timeout)["msg"]
    finally:
        ws.send(json.dumps({"op": "unsubscribe", "topic": topic, "id": f"v3d{topic}"}))


def fetch_calibration(rosbridge_url: str) -> dict:
    """Camera intrinsics + static transforms from the robot; cached so it also works offline."""
    from websockets.sync.client import connect
    try:
        with connect(rosbridge_url, open_timeout=5, max_size=None) as ws:
            info = {k: _one(ws, t, "sensor_msgs/msg/CameraInfo") for k, t in (("color", COLOR_INFO), ("depth", DEPTH_INFO))}
            ws.send(json.dumps({"op": "subscribe", "topic": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "id": "v3dtf"}))
            tfs, end = {}, time.monotonic() + 3
            while time.monotonic() < end:
                try:
                    m = json.loads(ws.recv(timeout=0.5))
                except TimeoutError:
                    continue
                if m.get("topic") == "/tf_static":
                    for t in m["msg"]["transforms"]:
                        tr, r = t["transform"]["translation"], t["transform"]["rotation"]
                        tfs[t["child_frame_id"]] = {"parent": t["header"]["frame_id"],
                                                    "t": [tr["x"], tr["y"], tr["z"]], "q": [r["x"], r["y"], r["z"], r["w"]]}
            ws.send(json.dumps({"op": "unsubscribe", "topic": "/tf_static", "id": "v3dtf"}))
            hand_eye = {}
            for s in K.ARMS:
                for key, topic in ((f"{s}_head", f"/calibration/extrinsic/from_{s}_gripper_camera_to_head_front_RGBD_camera"),
                                   (f"{s}_action", f"/calibration/extrinsic/from_{s}_gripper_action_to_{s}_gripper_camera")):
                    try:
                        hand_eye[key] = json.loads(_one(ws, topic, "std_msgs/msg/String", 4.0)["data"])["T"]
                    except Exception:  # noqa: BLE001 - optional; correction is skipped without it
                        pass
        calib = {"camera_info": {k: {"width": v["width"], "height": v["height"], "k": v["k"], "d": v["d"]}
                                 for k, v in info.items()}, "tf_static": tfs, "hand_eye": hand_eye}
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(calib))
        return calib
    except Exception:  # noqa: BLE001 - fall back to the cache
        if CACHE.exists():
            return json.loads(CACHE.read_text())
        raise


def frame_in_base(calib: dict, urdf_xml: str, frame: str) -> np.ndarray:
    """Pose of `frame` in base_link, following /tf_static parents up to an arm base, then the URDF."""
    T, tfs = np.eye(4), calib["tf_static"]
    while frame in tfs and not frame.endswith("_arm_base_link"):
        e = tfs[frame]
        T = K.transform(K.quat_to_matrix(*e["q"]), np.array(e["t"])) @ T
        frame = e["parent"]
    if frame == "base_link":
        return T
    return K.static_transform(urdf_xml, "base_link", frame) @ T


# ---------- camera model ----------

@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray            # 3x3
    T_base_cam: np.ndarray   # 4x4 optical frame in base_link

    @classmethod
    def from_calib(cls, calib, urdf_xml, which: str):
        ci = calib["camera_info"][which]
        frame = COLOR_OPTICAL if which == "color" else DEPTH_OPTICAL
        return cls(ci["width"], ci["height"], np.array(ci["k"], float).reshape(3, 3),
                   frame_in_base(calib, urdf_xml, frame))

    def project(self, pts_base: np.ndarray):
        """(N,3) base_link points -> (N,2) pixels and (N,) depth along the optical axis."""
        T = np.linalg.inv(self.T_base_cam)
        p = (T[:3, :3] @ np.asarray(pts_base, float).T + T[:3, 3:4]).T
        z = p[:, 2]
        uv = (self.K @ (p.T / np.where(z > 1e-6, z, np.nan))).T[:, :2]
        return uv, z

    def deproject(self, depth_m: np.ndarray):
        """(H,W) depth in metres -> (M,3) base_link points and the (M,2) pixel each came from (valid pixels only)."""
        v, u = np.nonzero(depth_m > 0)
        z = depth_m[v, u]
        x = (u - self.K[0, 2]) * z / self.K[0, 0]
        y = (v - self.K[1, 2]) * z / self.K[1, 1]
        cam = np.stack([x, y, z])
        pts = (self.T_base_cam[:3, :3] @ cam + self.T_base_cam[:3, 3:4]).T
        return pts, np.stack([u, v], axis=1)


def camera_correction(calib: dict, urdf_xml: str, color: "Camera", tol_m: float = 0.02, tol_deg: float = 5.0) -> dict:
    """Correct the head camera's position using the robot's own hand-eye calibration.

    The robot stores head-camera -> gripper-camera and gripper-camera -> gripper-base transforms (calibrated at
    the zero-joint pose). Composing them gives where each gripper really is relative to the camera; comparing
    with the arm model gives the camera position error. Measured on this robot: ~+7.3 cm in x, both arms
    agreeing within 0.5 cm. Applied only when both arms agree.
    """
    he = calib.get("hand_eye") or {}
    diffs, rots = [], []
    for s in K.ARMS:
        if f"{s}_head" not in he or f"{s}_action" not in he:
            return {"shift": [0.0, 0.0, 0.0], "applied": False, "reason": "hand-eye calibration not available"}
        T_cal = color.T_base_cam @ np.linalg.inv(np.array(he[f"{s}_head"])) @ np.array(he[f"{s}_action"])
        T_mod = static_link6_zero(urdf_xml, s)
        diffs.append(T_mod[:3, 3] - T_cal[:3, 3])
        rots.append(float(np.degrees(np.linalg.norm(K.rotation_error(T_cal[:3, :3], T_mod[:3, :3])))))
    spread = float(np.linalg.norm(diffs[0] - diffs[1]))
    if spread > tol_m or max(rots) > tol_deg:
        return {"shift": [0.0, 0.0, 0.0], "applied": False,
                "reason": f"arms disagree ({100 * spread:.1f} cm, {max(rots):.1f}°), so no correction"}
    shift = np.mean(diffs, axis=0)
    return {"shift": shift.tolist(), "applied": True,
            "reason": f"from hand-eye calibration; arms agree within {100 * spread:.1f} cm"}


def static_link6_zero(urdf_xml: str, side: str) -> np.ndarray:
    return K.arm_base_in_world(urdf_xml, side) @ K.Arm(urdf_xml, side).fk(np.zeros(6))


def corrected(cam: "Camera", shift) -> "Camera":
    T = cam.T_base_cam.copy()
    T[:3, 3] += np.asarray(shift, float)
    return Camera(cam.width, cam.height, cam.K, T)


def decode_compressed_depth(raw: bytes) -> np.ndarray:
    """ROS compressedDepth (16UC1, 12-byte header + PNG) -> (H,W) uint16 depth in millimetres."""
    start = raw.find(b"\x89PNG")
    if start < 0:
        raise ValueError("compressedDepth frame has no PNG payload")
    img = Image.open(io.BytesIO(raw[start:]))
    arr = np.array(img)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr


def grab_depth(rosbridge_url: str, timeout: float = 8.0) -> np.ndarray:
    """One depth frame (metres, float32)."""
    from websockets.sync.client import connect
    with connect(rosbridge_url, open_timeout=5, max_size=None) as ws:
        msg = _one(ws, DEPTH_TOPIC, "sensor_msgs/msg/CompressedImage", timeout)
    data = msg["data"]
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
    return decode_compressed_depth(raw).astype(np.float32) / 1000.0


# ---------- locating an object ----------

@dataclass
class Located:
    center: np.ndarray        # (3,) base_link: x, y of the object's top centre; z = top surface
    top_z: float
    table_z: float
    height: float
    size_xy: np.ndarray       # (2,) extent of the top points in x and y
    n_points: int
    bbox_px: list


def locate(bbox_px, depth_m: np.ndarray, color: Camera, depth_cam: Camera, min_points: int = 8) -> Located:
    """3D position of the object inside a colour-image bounding box [x1, y1, x2, y2] (pixels)."""
    x1, y1, x2, y2 = bbox_px
    pts, _ = depth_cam.deproject(depth_m)
    uv, z = color.project(pts)
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2) & (z > 0)
    # A ring around the box gives the table height under the object.
    pad_x, pad_y = 0.6 * (x2 - x1) + 15, 0.6 * (y2 - y1) + 15
    ring = ((uv[:, 0] >= x1 - pad_x) & (uv[:, 0] <= x2 + pad_x) & (uv[:, 1] >= y1 - pad_y) & (uv[:, 1] <= y2 + pad_y)
            & ~inside & (z > 0))
    if ring.sum() < min_points:
        raise RuntimeError("not enough depth around the object to measure the table")
    table_z = float(np.median(pts[ring, 2]))
    obj = pts[inside]
    obj = obj[obj[:, 2] > table_z + 0.008]  # ignore table pixels inside the box
    if len(obj) < min_points:
        raise RuntimeError(f"only {len(obj)} depth points on the object (it may be too small, shiny or too close)")
    top_z = float(np.percentile(obj[:, 2], 90))
    top = obj[obj[:, 2] >= top_z - 0.015]
    center = np.array([float(np.median(top[:, 0])), float(np.median(top[:, 1])), top_z])
    size = np.percentile(obj[:, :2], 95, axis=0) - np.percentile(obj[:, :2], 5, axis=0)
    return Located(center, top_z, table_z, top_z - table_z, size, int(len(obj)), list(map(float, bbox_px)))
