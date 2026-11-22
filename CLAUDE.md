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

- **The robot is SHARED with another team.** Before any mode change or live move, the user confirms it is their
  turn and the other people are clear of the arms. Never leave the robot in a mode we switched to; switch back
  when done.

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
- The Nebius Token Factory key lives only in `~/.config/nebius/api_key` (`600`), set by the user with
  `python3 perception/set_nebius_key.py`. It is separate from the Wall-X key.
- Keys stay out of task JSON, logs, the GUI, commits and chat. `*.pdf` and `*headers*.json` are git-ignored.

## Layout

- `docs/`: organizer PDFs (extract with `pdftotext -layout`) and the original demo in `participant-guide-demo (1)/`
- `harness/`: relay; becomes the typed-command + GUI server (plan Phases 3–5)
- `perception/`: read-only "what does the camera see": rosbridge frame grab (`camera.py`) → Nebius
  Token Factory vision model (`nebius_vision.py`); CLI `perception/see.py`; browser UI
  `perception/webui.py` → `http://127.0.0.1:8080` (localhost only; the page is `perception/static/index.html`). Default camera is `head`;
  the wrist cameras capture people, so send them only when asked.
- `motion/`: **dry-run** movement planning: Nebius parses a typed command → `safety.py` checks it
  (IK from the robot's URDF, joint limits, step size, workspace, table clearance, arm separation along the path)
  → the UI shows the result and the message that *would* be sent. Limits: `motion/safety_config.json`.
  **Live sending** (user's decision, 2026-11-22, without organizer approval): `motion/sender.py` is the ONLY
  publisher. It is locked unless the UI runs with `--live`. Default backend **"sdk"**: `motion/sdk_bridge.py`
  (runs in `.venv-sdk`) sends checked joint targets via `set_joint_positions`, interpolated ≤ 0.5° per command at
  50 Hz; it requires SDK mode (UI buttons Enter/Exit SDK mode), sets `MANIPULATOR_JOINT_POSITIONS` control and aborts
  if that makes the arm move. UI also has a Robot E-stop (SDK `emergency_stop`) + Recover. It executes stored, fully checked, single-arm plans
  after a confirm click; refuses if another program is commanding the arm, the plan is > 60 s old, or the arm
  drifted; stops on unexpected force or a missed step, then holds the pose.
  **pose_cmd convention VERIFIED = `relative_to_neutral`, `frame_id: ""`** (robot recordings 2026-11-05 14:04–14:40:
  arm follows within 1–4 mm vs 215–218 mm for the arm-base alternative; see `_evidence` in safety_config.json).
  Recordings live in `.cache/bags/`; `motion/record_commands.py --analyze <dir>` re-checks one.
  Verified: `/{side}_arm/end_pose` = FK(0)⁻¹·FK(q) in `{side}_arm_neutral` (0.0 mm error).
  Moves act on the fingertip (0.217 m beyond link6); rotations pivot around it.
  Picks (`grasp.py`): vision model box (Qwen returns **0–1000 normalised** coords) → depth (`vision3d.py`,
  320×200 compressedDepth PNG) → 3D in base_link → open / above / descend / close / lift, each step ≤ 3 cm / 15°
  and fully checked; the fingertip may go below table clearance only inside a 5 cm "grasp zone".
  **Head camera correction:** `/tf_static` places the camera ~7.3 cm off in x. `vision3d.camera_correction`
  fixes it from the robot's own hand-eye calibration (`/calibration/extrinsic/*`, matrices read as `A⁻¹·B`),
  applied only when both arms agree. The table measures at z ≈ −1 cm in base_link.
- Python deps for our code: `requirements.txt` (adds Pillow).
- `motion/record_commands.py`: **read-only** recorder of the platform's own arm commands (`pose_cmd`, gripper
  commands) vs. where the arm goes. Its verdict confirms or refutes our `pose_cmd` format before any sender exists.
  Recordings go to `.cache/recordings/` (git-ignored).
- **Official SDK** (gRPC, robot port 50051): client **v1.1.2** in `.venv-sdk/` (git-ignored; the v2.0.0 client gets
  `UNIMPLEMENTED` from this robot's v1 server). `connect("x2://192.168.77.2:50051")` → model `desktop`. API:
  `left_arm/right_arm.{get_joint_states, get_end_pose, set_joint_positions, set_end_pose, reset}`,
  `left_gripper.set_position`, `robot_control.{set_manipulator_control_mode, emergency_stop, recover_emergency_stop,
  homing}`, `get_robot_status()`. Arm commands need `set_manipulator_control_mode` first (reads "Control mode is not
  set" until then). Installing packages needs the user's approval; the user ran the install.
- **Robot modes:** `/application/request/task_operation` (`protocol/srv/TaskOperation`), as the WebUI does it:
  `{operation_type: 1 START | 2 STOP, task_type: 5 SDK, is_from_remote: false}`; current mode in
  `/application/task_info` (`type`, `status`). In base mode the platform ignores external `pose_cmd`; in SDK mode
  `sdk2_server` streams hold commands, so direct `pose_cmd` would conflict (the sender refuses).
- **rosbridge is fragile:** it relays joint states at only ~10–15 Hz, and a heavy camera stream starves it
  (8 fps head JPEG dropped joint states to < 1 Hz). Keep camera subscriptions at ≤ 4 fps and on demand.
  Note that `pkill -f <pattern>` also matches the shell running it; run it as its own command.
- `tasks/`: task JSON drafts `local_<id>.json`; deployed copies go to the robot's `/home/xr/diversity/`
- `.venv/`: harness venv (`python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt`)
- Skills: `robot-task` (task JSON + deploy), `robot-harness` (relay, typed commands, GUI), `wallx-api` (protocol + probe)
