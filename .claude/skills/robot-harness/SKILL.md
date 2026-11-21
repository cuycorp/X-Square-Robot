---
name: robot-harness
description: Run, test and extend harness/harness.py, the relay between the robot client and Wall-X, including typed-command override, the browser dashboard (cameras, arm state, latency) and the optional vision agent. Use for passthrough runs, harness code changes, GUI work, logs, or harness errors.
---

# Harness: robot client ⇄ laptop ⇄ Wall-X

```
robot client (192.168.77.2) ──ws──▶ harness.py on 192.168.77.18:8001 ──ws/wss──▶ Wall-X
                                      └─ GUI on 127.0.0.1:8080 (cameras, state, command box)
```

The robot client captures the 3 cameras and arm state, sends a `public_v2` request, and executes the
32-frame reply. The harness **changes only `instruction`**. Observation, `control_context`, handshake
and the model reply pass through unchanged: frames are forwarded as-is unless the instruction changes,
in which case the request is unpacked, edited and re-packed (`pack(observation)`). Protocol details:
the `wallx-api` skill.

## Setup and tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt
(cd harness && ../.venv/bin/python -m unittest -v test_harness.py)   # offline: no robot, no API
export WALLX_ENDPOINT='<on-site ws://HOST:33677 or wss://maas-api.x2robot.com/v1/vla/wallx>'
export WALLX_HEADERS_FILE=~/.config/wallx/headers.json
```

Probe Wall-X first (the `wallx-api` probe, handshake only). The robot must reach port 8001 on the
laptop: `sudo ufw allow from 192.168.77.0/24 to any port 8001` when ufw is active (the user runs sudo).

## Run

Bind to `192.168.77.18`, the address the robot reaches. Run the harness in the background and watch
for `event=chunk` lines (one per round trip).

```bash
cd harness && ../.venv/bin/python harness.py --host 192.168.77.18 --port 8001 --passthrough
```

The user then sets, in the WebUI's Edit Model Configuration, `model_address = ws://192.168.77.18:8001`,
`EndPose`, `sync`, with the same `infer_params` as the direct run, and starts the Case from the WebUI.
Stop order: WebUI "End Evaluation Inference Mode" → submit scores → Ctrl+C the harness.

## Code map (`harness/harness.py`)

- `Harness.handle(downstream)`: per-connection loop. Receives the handshake from upstream and sends it
  down, then for each request: validate → choose instruction → forward → return the reply.
- `validate_request`, `instruction_text`: input checks.
- `VisionAgent.decide(observation, goal, previous, history) -> str`: the optional planner,
  configured by `AGENT_CHAT_URL` / `AGENT_MODEL` / `AGENT_API_KEY`; enabled when `--passthrough` is
  dropped. `--agent-every N` replans every N model requests. Agent timeout 10 s; Wall-X timeout 60 s.
- An empty or invalid instruction, or a timeout, closes the session (1011); the operator stops in the WebUI.

## Extending (plan Phases 4–5)

- **Typed command**: a shared `CommandState` (current, pending, history). Set `request["instruction"]`
  from it just before forwarding. Non-empty, ≤ 200 chars. The switch policy (next request / after chunk /
  on Apply) is the user's choice; check `plan.md`.
- **Timing**: record robot→harness receive, the Wall-X round trip and the reply send per request,
  and flag the warm-up call. Add the fields to the `event=chunk` log line.
- **GUI**: `aiohttp` HTTP + WebSocket on `127.0.0.1:8080` in the same process. Push the 3 JPEGs
  (throttled ~5 fps), both arms' pose and gripper, the active instruction, Wall-X's first frame and
  the timings; accept `{"type": "command", "text": …}`. A single `gui/static/index.html` with no build
  step. It carries a permanent banner "Stop = WebUI → End Evaluation Inference Mode"; stopping lives in the WebUI only.
- The GUI only updates while a Case runs, because the client only sends during inference. Show "last update N s ago".
- Every change: re-run the offline tests, plus new tests for the change (command override leaves all
  other fields identical; bad input rejected; a GUI disconnect leaves the relay running).

## Evaluate

Same Case, layout and `infer_params` for direct vs harness. Record the task file, case_id, harness
version, commands sent, success/scores, time, Wall-X calls and latency, and the failure reason. No keys in logs.
