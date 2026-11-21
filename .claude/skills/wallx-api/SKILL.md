---
name: wallx-api
description: WALLX VLA WebSocket protocol reference (public_v2 request/response, msgpack framing, handshake, error codes) plus a probe script to verify connectivity/auth. Use when writing code that talks to WALLX directly, debugging handshake/401/403/unknown_embodiment/missing_field errors, or checking action shapes.
---

# WALLX API (public_v2, sync)

Endpoint: `wss://maas-api.x2robot.com/v1/vla/wallx` (cloud) or on-site `ws://host:port`.
Auth: HTTP handshake header `Authorization: Bearer <WALLX key>` (via `WALLX_HEADERS_FILE`).
WALLX is NOT a chat endpoint; the vision agent uses its own Chat Completions URL.

## Framing
Binary MessagePack, `msgpack_numpy`. Sequence: connect → **recv metadata first** (dict; has `model`,
`model_version`; `error` key = failure) → loop { send 1 request, recv 1 reply }.
Allow ~60 s on the first reply (model warm-up). Max frame ~32–64 MB.

## Request
```python
{
  "embodiment": "quanta_x1_pro/gripper-g",   # or quanta_x1_pro, quanta_x1_pro/gripper-h (lowercase)
  "normalization": "provided",
  "instruction": "non-empty python str (not numpy str array)",
  "observation": {
    "images": {"front": b64, "left_wrist": b64, "right_wrist": b64},  # JPEG base64, same timestamp, all 3 required; data:image/jpeg;base64, prefix ok; raw bytes NOT ok
    "state": {"left_ee":  {"position": [x,y,z], "rotation_rpy": [r,p,y], "gripper": g},
              "right_ee": {"position": [...],   "rotation_rpy": [...],   "gripper": g}},
  },
  # client extension fields like "control_context" — preserve untouched
}
```
Frame: robot base, meters / radians, `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. Gripper G ∈ [0, 1.526], H ∈ [0, 5.5].
Sync mode: no RTC fields (`use_rtc`, `prev_action_chunk`, `inference_delay`).

## Response
`{"embodiment": ..., "actions": {"left_ee": {...}, "right_ee": {...}}, ...timing/diagnostics}`
Each arm: `position[32,3]`, `rotation_rpy[32,3]`, `gripper[32]`; same index = same timestep;
row 31 repeats row 30. Check `embodiment` matches the request. Don't use extra fields to decide task completion.
`error` may be a string or object → log it, don't execute anything from that round.

## Errors
| error | fix |
|---|---|
| unknown_embodiment | match physical gripper; lowercase `gripper-g`/`gripper-h` |
| missing_field / image_decode_failed | instruction + 3 JPEG b64 images + both arm states |
| mixed_schema / normalization_conflict | exact fields above, `normalization: provided` |
| 401 / 403 | key / header format |
| closed / timeout | confirm platform stopped in WebUI → reconnect → re-read metadata → fresh observation; **never replay cached actions** |

## Auth file setup
Key is the organizer-issued WALLX/competition API key (also entered in the WebUI model config).
User runs (interactive, alone on its line — never paste multi-line blocks around a hidden prompt):
`python3 .claude/skills/wallx-api/scripts/set_wallx_key.py` → writes `~/.config/wallx/headers.json` (600).
Claude must not read or extract the key itself.

## Probe (no robot motion)
Verified 2026-11-21: handshake OK, model `wall-x`, version `2610-dualhead-v2`.
Metadata keys include action_dim, action_dt, action_horizon, control_context, robot_type, rtc, valid_rows.
```bash
WALLX_HEADERS_FILE=~/.config/wallx/headers.json .venv/bin/python .claude/skills/wallx-api/scripts/wallx_probe.py
# optional single inference from a captured observation (saves reply, executes nothing):
.venv/bin/python .claude/skills/wallx-api/scripts/wallx_probe.py --observation observation.json --instruction "pick up the spoon"
```
