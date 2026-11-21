# X-Square Robot hackathon workspace

Goal: type a command → the **Wall-X** model moves the TeleTwin Desktop arms (dual 6-DOF, G gripper),
with a browser dashboard for the cameras and arm readings.

**`plan.md` is the plan and the progress record.** Read it before any robot work. Work phase by phase
and tick its checkboxes as items are verified.

## Machines

- **This machine is the laptop**, not the robot: user `francesca`, `192.168.77.18` on Ethernet
  (`enx…`); Wi-Fi `10.19.19.179` is a different network.
- **Robot**: `xr@192.168.77.2`. SSH needs a password, so the user runs robot-side commands and pastes
  results back. WebUI: `http://192.168.77.2/`.
- The robot's own client drives the motors. Our code runs on the laptop: `harness/harness.py`
  listens on `192.168.77.18:8001`, the robot connects to it, and it forwards to Wall-X.

## Safety (hard rules)

- **The WebUI is the stop.** Stopping, homing and scoring happen only in the WebUI. Stop order:
  "End Evaluation Inference Mode" → submit scores → then Ctrl+C the harness. Closing the harness,
  a socket or the GUI leaves the arm running, so say this whenever a run is started.
- Motion starts only when the user explicitly asks for it in that turn.
- On the robot, observe only (`ps`, `ros2 topic list/echo/hz`). Starting, stopping or relaunching
  control processes needs the user's explicit OK; the guide forbids a second control process.
- After a disconnect: confirm the stop in the WebUI, reconnect, re-read the handshake, use fresh
  observations. Old action chunks are discarded.

## Secrets

- The Wall-X key lives only in `~/.config/wallx/headers.json` (`600`), used via `WALLX_HEADERS_FILE`.
- `docs/API Tutorial & User Guide - 2610 (Condensed Version).pdf` contains the key in plain text.
  When extracting it, pipe through `sed -E 's/sk-[A-Za-z0-9_-]+/sk-<redacted>/g'`.
- Keys stay out of task JSON, logs, the GUI, commits and chat. `*.pdf` and `*headers*.json` are git-ignored.

## Layout

- `docs/`: organizer PDFs (extract with `pdftotext -layout`) and the original demo in `participant-guide-demo (1)/`
- `harness/`: relay; becomes the typed-command + GUI server (plan Phases 3–5)
- `tasks/`: task JSON drafts `local_<id>.json`; deployed copies go to the robot's `/home/xr/diversity/`
- `.venv/`: harness venv (`python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt`)
- Skills: `robot-task` (task JSON + deploy), `robot-harness` (relay, typed commands, GUI), `wallx-api` (protocol + probe)
