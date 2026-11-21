#!/usr/bin/env python3
"""Check WALLX handshake (and optionally one inference). Never controls the robot."""
import argparse
import json
import os
import sys
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.client import connect

msgpack_numpy.patch()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("WALLX_ENDPOINT", "wss://maas-api.x2robot.com/v1/vla/wallx"))
    ap.add_argument("--headers-file", default=os.environ.get("WALLX_HEADERS_FILE"))
    ap.add_argument("--embodiment", default=os.environ.get("WALLX_EMBODIMENT", "quanta_x1_pro/gripper-g"))
    ap.add_argument("--observation", help="observation.json with images + state (real, live data)")
    ap.add_argument("--instruction", default=os.environ.get("WALLX_INSTRUCTION", ""))
    ap.add_argument("--out", default="actions.msgpack")
    a = ap.parse_args()

    headers = json.loads(Path(os.path.expanduser(a.headers_file)).read_text()) if a.headers_file else None
    print(f"connecting to {a.endpoint} (auth headers: {'yes' if headers else 'no'})")
    with connect(a.endpoint, additional_headers=headers, open_timeout=30, max_size=64 * 1024 * 1024) as ws:
        meta = msgpack.unpackb(ws.recv(timeout=60), raw=False)
        if not isinstance(meta, dict) or "error" in meta:
            print("handshake failed:", meta if not isinstance(meta, dict) else meta.get("error"))
            return 1
        print("handshake OK  model:", meta.get("model"), " version:", meta.get("model_version"))
        print("metadata keys:", sorted(meta.keys()))
        if not a.observation:
            return 0
        if not a.instruction.strip():
            print("--instruction required with --observation")
            return 2
        payload = {"embodiment": a.embodiment, "normalization": "provided",
                   "instruction": a.instruction.strip(),
                   "observation": json.loads(Path(a.observation).read_text())}
        ws.send(msgpack.packb(payload, use_bin_type=True))
        reply = msgpack.unpackb(ws.recv(timeout=60), raw=False)
        if "error" in reply:
            print("model error:", reply["error"])
            return 1
        assert reply["embodiment"] == payload["embodiment"], "embodiment mismatch"
        for arm in ("left_ee", "right_ee"):
            b = reply["actions"][arm]
            shapes = [np.asarray(b[k]).shape for k in ("position", "rotation_rpy", "gripper")]
            print(arm, "shapes", shapes)
            assert shapes == [(32, 3), (32, 3), (32,)], "unexpected action shape"
        Path(a.out).write_bytes(msgpack.packb(reply, use_bin_type=True))
        print("saved reply to", a.out, "(not executed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
