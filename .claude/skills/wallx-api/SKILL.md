---
name: wallx-api
description: Wall-X VLA protocol reference (public_v2 request/response, msgpack handshake, error codes, RTC) and a probe script. Use when writing or debugging code that talks to Wall-X, checking action shapes, or diagnosing handshake, 401/403, unknown_embodiment or missing_field errors.
---

# Wall-X API (public_v2)

Sources: `docs/API Tutorial & User Guide - 2610 (Condensed Version).pdf` (redact `sk-…` when
extracting) and `docs/Hackathon Participant Guide.pdf`.

- Endpoints: on-site `ws://HOST:33677` (ask the organizers for HOST) or cloud `wss://maas-api.x2robot.com/v1/vla/wallx`.
- Auth: HTTP handshake header `Authorization: Bearer <key>` from `WALLX_HEADERS_FILE`
  (`~/.config/wallx/headers.json`). Verified earlier against the cloud endpoint.
- Wall-X is an action model, not a chat endpoint.

## Framing

Binary MessagePack + `msgpack_numpy`. Connect → **receive the handshake first** (dict with `model`
`wall-x`, `model_version` e.g. `2610-dualhead-v2`; an `error` key means failure) → then strictly
one request, one reply. The first reply includes CUDA warm-up, so allow ~60 s. Use `max_size=None` (or ≥ 64 MB).

## Request

```python
{
  "embodiment": "quanta_x1_pro/gripper-g",  # = quanta_x1_pro; H: quanta_x1_pro/gripper-h (lowercase)
  "normalization": "provided",
  "instruction": "Pick up the cup.",        # non-empty Python str
  "observation": {
    "images": {"front": b64, "left_wrist": b64, "right_wrist": b64},  # JPEG base64, same timestamp, all 3
    "state": {"left_ee":  {"position": [x, y, z], "rotation_rpy": [r, p, y], "gripper": g},
              "right_ee": {"position": [...],    "rotation_rpy": [...],    "gripper": g}},
  },
  # client extension fields (e.g. control_context) pass through untouched
}
```

- Absolute poses in the robot body frame, metres and radians, `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
- Gripper G `[0, 1.526]`, H `[0, 5.5]`. Send only the left/right arms (no head, waist or base) and no
  `dof_mask`, `normalizer` or `robot_id`.
- The robot's right arm appears on the **left** of the front image.

## Reply

`{"embodiment", "actions": {"left_ee", "right_ee"}}`. Each arm has `position[32,3]`,
`rotation_rpy[32,3]` and `gripper[32]`; the same index is the same timestep, frames are **33.7 ms**
apart, and frame 31 repeats 30. Check that `embodiment` echoes the request. An `error` (string or
`{code, message}`) means nothing from that round is used. Server crash → close code `1011`, so reconnect.

## RTC (streaming; the robot client's job in sync mode)

Execute frames 0–5 (~0.2 s), then request again with `use_rtc: true`, `inference_delay: 4` (1–4
allowed; 5+ → `rtc_delay_exceeds_training`) and `prev_action_chunk` = the last reply with frame 0
kept and frames ≥ 6 appended (`[f[0], *f[6:]]` for every key). First request: no RTC fields.

## Errors

| code | fix |
|---|---|
| `unknown_embodiment` | match the physical gripper; lowercase `gripper-g` / `gripper-h` |
| `missing_field` / `image_decode_failed` | instruction + 3 JPEG b64 images + both arms' position/rotation_rpy/gripper |
| `mixed_schema` / `normalization_conflict` | exactly the fields above, `normalization: provided`, no `rotation_6d` |
| 401 / 403 | key or header format |
| closed / timeout | confirm the stop in the WebUI → reconnect → re-read handshake → fresh observation |

## Probe (handshake only; never moves the robot)

```bash
WALLX_HEADERS_FILE=~/.config/wallx/headers.json WALLX_ENDPOINT='<endpoint>' \
  .venv/bin/python .claude/skills/wallx-api/scripts/wallx_probe.py
# one inference from a saved observation (reply saved, not executed):
.venv/bin/python .claude/skills/wallx-api/scripts/wallx_probe.py --observation observation.json --instruction "Pick up the cup."
```

To replace the key, the user runs `python3 .claude/skills/wallx-api/scripts/set_wallx_key.py`
(hidden prompt, writes `headers.json` with mode 600).
