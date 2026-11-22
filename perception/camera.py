"""Grab camera frames from the robot over rosbridge. Read-only: subscribe, take one frame, unsubscribe."""
import base64
import json
import os
import time

from websockets.sync.client import connect

ROSBRIDGE_URL = os.environ.get("ROSBRIDGE_URL", "ws://192.168.77.2:9090")

# name -> (topic, description). The head camera looks down at the table; the wrist cameras are
# wide-angle and often see people in the room, so they are only sent when asked for explicitly.
CAMERAS = {
    "head": ("/camera_head_front/color/image_raw/compressed", "head camera, looking down at the table"),
    "cam1": ("/camera1/usb_cam1/image_raw/image_compressed", "wrist camera 1 (arm side to be confirmed)"),
    "cam3": ("/camera3/usb_cam3/image_raw/image_compressed", "wrist camera 3 (arm side to be confirmed)"),
}
JPEG_MAGIC = b"\xff\xd8\xff"


def decode_compressed_image(msg: dict) -> bytes:
    """Return JPEG bytes from a rosbridge sensor_msgs/CompressedImage message."""
    if "jpeg" not in str(msg.get("format", "")).lower():
        raise ValueError(f"unsupported image format {msg.get('format')!r}; expected jpeg")
    data = msg.get("data")
    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data or b"")
    if not raw.startswith(JPEG_MAGIC):
        raise ValueError("frame is not a JPEG")
    return raw


def grab_frames(names, url: str = ROSBRIDGE_URL, timeout: float = 10.0) -> dict:
    """Return {camera_name: jpeg_bytes}, one fresh frame per requested camera."""
    unknown = [n for n in names if n not in CAMERAS]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; choose from {sorted(CAMERAS)}")
    frames = {}
    with connect(url, open_timeout=5, max_size=None) as ws:
        for name in names:
            topic = CAMERAS[name][0]
            ws.send(json.dumps({"op": "subscribe", "id": name, "topic": topic,
                                "type": "sensor_msgs/msg/CompressedImage",
                                "throttle_rate": 0, "queue_length": 1}))
            deadline = time.monotonic() + timeout
            try:
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(f"no frame from {topic} within {timeout:.0f}s")
                    msg = json.loads(ws.recv(timeout=left))
                    if msg.get("op") == "publish" and msg.get("topic") == topic:
                        frames[name] = decode_compressed_image(msg["msg"])
                        break
            finally:
                ws.send(json.dumps({"op": "unsubscribe", "id": name, "topic": topic}))
    return frames
