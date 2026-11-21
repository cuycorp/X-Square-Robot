#!/usr/bin/env python3
"""Validate X-Square benchmark task JSON files against the participant-guide rules."""
import json
import re
import sys
from pathlib import Path

EMBODIMENTS = {"quanta_x1_pro", "quanta_x1_pro/gripper-g", "quanta_x1_pro/gripper-h"}
SECRET_HINTS = ("api_key", "apikey", "authorization", "token", "secret", "bearer")


def check(path: Path) -> list[str]:
    errs, warns = [], []
    m = re.fullmatch(r"local_([a-z0-9_-]+)\.json", path.name)
    if not m:
        errs.append("filename must be local_<task_id>.json with task_id in [a-z0-9_-]")
    try:
        raw = path.read_text()
        task = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return [f"ERROR invalid JSON: {e}"]
    low = raw.lower()
    for hint in SECRET_HINTS:
        if hint in low:
            errs.append(f"possible credential field '{hint}' in task file — keys belong in the WebUI only")
    if task.get("mode") != "benchmark":
        errs.append("mode must be 'benchmark'")
    if not task.get("task_name"):
        errs.append("task_name missing")
    cid = task.get("collection_id")
    if cid is not None and cid != path.stem:
        errs.append(f"collection_id '{cid}' != filename stem '{path.stem}'")
    model = task.get("model") or {}
    for k, v in (("robot_type", "DESKTOP"), ("mode_type", "EndPose"), ("infer_type", "sync")):
        if model.get(k) != v:
            warns.append(f"model.{k} = {model.get(k)!r} (guide uses {v!r})")
    addr = model.get("model_address", "")
    if not re.match(r"^wss?://", addr or ""):
        errs.append("model.model_address must be a ws:// or wss:// URL")
    if re.search(r"//(localhost|127\.0\.0\.1)", addr or ""):
        warns.append("model_address uses localhost — the robot client may not reach it; use the LAN IP")
    ip = model.get("infer_params") or {}
    ap = ip.get("async_params") or {}
    if ip.get("action_horizon") != 32:
        warns.append(f"action_horizon={ip.get('action_horizon')} (guide: 32)")
    if ip.get("control_dt") != 0.005:
        warns.append(f"control_dt={ip.get('control_dt')} (guide: 0.005)")
    if ap.get("api_schema") != "public_v2":
        errs.append("async_params.api_schema must be 'public_v2'")
    if ap.get("embodiment") not in EMBODIMENTS:
        errs.append(f"async_params.embodiment must be one of {sorted(EMBODIMENTS)}")
    if ap.get("normalization") != "provided":
        errs.append("async_params.normalization must be 'provided'")
    if ap.get("control_context") != "masked":
        warns.append("async_params.control_context is not 'masked'")
    data = task.get("data")
    if not isinstance(data, list) or not data:
        errs.append("data must be a non-empty list of Cases")
        data = []
    seen = set()
    for i, case in enumerate(data):
        p = f"data[{i}]"
        cid_ = case.get("case_id")
        if cid_ is None or cid_ in seen:
            errs.append(f"{p}.case_id missing or duplicate ({cid_!r})")
        seen.add(cid_)
        ins = case.get("instruction")
        if not isinstance(ins, str) or not ins.strip():
            errs.append(f"{p}.instruction must be a non-empty string")
        sd = case.get("scores_dict", {})
        if not isinstance(sd, dict) or any(not isinstance(v, (int, float)) or v < 0 for v in sd.values()):
            errs.append(f"{p}.scores_dict must map names to non-negative numbers")
        t = case.get("timeout")
        if not isinstance(t, (int, float)) or t <= 0:
            errs.append(f"{p}.timeout must be a positive number of seconds")
        img = case.get("image_path", "")
        if img and not img.startswith("http") and not (Path("/home/xr/diversity") / img).exists():
            warns.append(f"{p}.image_path '{img}' not found under /home/xr/diversity/")
        if not isinstance(case.get("env_description", {}), dict):
            errs.append(f"{p}.env_description must be an object")
    return [f"ERROR {e}" for e in errs] + [f"WARN  {w}" for w in warns]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_task.py local_<id>.json [...]")
        return 2
    bad = 0
    for arg in sys.argv[1:]:
        msgs = check(Path(arg))
        status = "FAIL" if any(m.startswith("ERROR") for m in msgs) else "OK"
        bad += status == "FAIL"
        print(f"[{status}] {arg}")
        for msg in msgs:
            print("   ", msg)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
