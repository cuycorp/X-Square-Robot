---
name: robot-harness
description: Set up, test, run and extend the synchronous WALLX WebSocket harness (harness/harness.py) that sits between the robot client and the WALLX VLA API, including the vision Agent that rewrites sub-instructions. Use for passthrough tests, agent integration, replanning logic, logs, or harness errors.
---

# WALLX harness

Architecture: `robot client (x2robot_infer_node) ──ws──> harness (this repo) ──wss──> WALLX API`.
The client captures 3 cameras + arm state, sends a `public_v2` request, executes returned actions.
The harness may change **only `instruction`**; everything else (observation, control_context,
metadata, model reply bytes) is forwarded unchanged.

## One-time setup
```bash
# python3.12-venv (ensurepip) is NOT installed on this host and sudo needs a password,
# so create the venv without pip and drive it with the system pip. Never pip-install into
# the system python3 — the ROS robot nodes use it.
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install -r harness/requirements.txt
(cd harness && ../.venv/bin/python -m unittest -v test_harness.py)   # offline, no robot/API
```
Auth headers file (key comes from the user or from the WebUI override — never echo it):
```bash
mkdir -p ~/.config/wallx && umask 077
python3 - <<'EOF'
import json, pathlib
o = json.load(open("/home/xr/diversity/local/state/model_overrides.json"))
key = next(v["infer_params"]["async_params"].get("api_key") for v in o.values()
           if v.get("infer_params", {}).get("async_params", {}).get("api_key"))
p = pathlib.Path.home()/".config/wallx/headers.json"
p.write_text(json.dumps({"Authorization": f"Bearer {key}"})); p.chmod(0o600)
print("wrote", p)
EOF
```
Env: `export WALLX_ENDPOINT=wss://maas-api.x2robot.com/v1/vla/wallx WALLX_HEADERS_FILE=~/.config/wallx/headers.json`

Check WALLX reachability first with the `wallx-api` skill's probe (handshake only, no motion).

## Run
Bind to the LAN IP the robot client can reach (this host: `10.19.20.98`; the client may run in a
container, so don't rely on 127.0.0.1). Run in background and watch for `event=chunk` lines.

1. **Passthrough** (baseline, no agent):
   `cd harness && ../.venv/bin/python harness.py --host 10.19.20.98 --port 8001 --passthrough`
2. **With vision agent** (OpenAI-compatible Chat Completions with image_url support):
   ```bash
   export AGENT_CHAT_URL=https://.../v1/chat/completions AGENT_MODEL=<vision-model>
   read -rs AGENT_API_KEY; export AGENT_API_KEY      # user types it; never pass on cmdline
   ../.venv/bin/python harness.py --host 10.19.20.98 --port 8001 --agent-every 4
   ```
   `--agent-every N` = replan every N model requests (not control steps).
   Agent HTTP timeout 10 s; WALLX send/recv timeout 60 s.

Then the **user** sets in WebUI → Edit Model Configuration: `model_address = ws://10.19.20.98:8001`,
`EndPose`, `sync`, same infer_params as direct run. Start a Case from the WebUI.

## Stopping (critical)
User clicks **End Evaluation Inference Mode** → enters scores → **Submit Scores** → *then* Ctrl+C / stop
the harness. Closing the harness does NOT stop the arm; the client may reconnect. Never claim an
agent "stop"/"complete" output halts hardware.

## Extending `VisionAgent.decide()`
Signature `decide(observation, goal, previous, history) -> str` (non-empty instruction; set `self.reason`).
Empty/invalid/timeout → harness closes the session (code 1011) and the operator must stop in WebUI.
Agent sees: goal (task instruction), 3 images (front/left_wrist/right_wrist), state, history (last 8).
`env_description`/`scores_dict` are NOT passed automatically — load the task JSON yourself if useful.

Improvement directions from the guide: sub-goal selection from observations (don't restart task),
memory (completed stages, failures, retry counts), failure-type recovery with retry limits,
call scheduling to cut latency, completion detection, structured logs for reproducibility.
After any harness change, re-run the offline tests and add tests for new behavior.

## Evaluate
Same Case, same initial layout, same infer_params for direct vs harness. Record: task file, case_id,
harness version, agent model + prompt version, success/scores, time, #agent and #VLA calls,
latency, instructions sent, retries/recovery, failure reason. Keep instructions & timings in logs; no keys.
