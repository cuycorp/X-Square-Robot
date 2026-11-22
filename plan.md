# Plan: type a command → Wall-X moves the arms, with a live browser dashboard

## Goal

1. Configure the robot to use **Wall-X**, the organizers' model trained for this robot
   (`quanta_x1_pro/gripper-g`). It works zero-shot, needs no GPU, and the organizers host it.
2. Type a plain-language command (e.g. `Pick up the cup and put it in the bowl.`) and have Wall-X move
   the TeleTwin Desktop arms.
3. Open a **browser dashboard** showing the three camera feeds, the arm/gripper readings and the latency.

No other models, no cloud GPU, no Nebius.

## Architecture

```
 Browser ── http://localhost:8080 ──┐ cameras · arm state · command box · latency · log
                                    ▼
 Robot 192.168.77.2          Laptop 192.168.77.18                        Wall-X
 ┌──────────────────┐  WS   ┌──────────────────────────────┐   WS     ┌───────────────────────┐
 │ organizer client │──────▶│ harness.py :8001             │─────────▶│ cloud (no on-site one) │
 │ public_v2, sync  │◀──────│  • instruction ← your text   │◀─────────│  or cloud wss://maas-  │
 │ executes poses   │ 32 fr │  • taps images + state → GUI │          │  api.x2robot.com/…     │
 └──────────────────┘       │  • timing per hop            │          └───────────────────────┘
                            └──────────────────────────────┘
```

- The **robot's client** reads the cameras and arm state, asks the "model" for actions, and executes them. We never drive the motors.
- The **harness** sits in the middle. It swaps in your typed command as `instruction`, forwards everything
  else unchanged, and copies the images and state to the GUI.
- **Start / stop / home happen only in the robot WebUI** (`http://192.168.77.2/`). Closing the harness or the GUI does **not** stop the arm.

Facts from the docs:
- Each request carries 3 JPEGs (`front`, `left_wrist`, `right_wrist`) and, per arm, absolute
  `position` (m), `rotation_rpy` (rad, `R = Rz·Ry·Rx`) and `gripper` (G: `[0, 1.526]`).
- Each reply has 32 frames per arm, 33.7 ms apart (frame 31 repeats 30).
- The first call does a GPU warm-up, so expect tens of seconds once, then it gets faster.

---

## Phase 0 — Secrets and repo (10 min)

- [x] **`docs/API Tutorial & User Guide - 2610 (Condensed Version).pdf` contains the Wall-X key in plain text.**
      `*.pdf` is now in `.gitignore`. Checked: the key is in no commit and no repo text file.
- [x] The key is in `~/.config/wallx/headers.json` (`600`, folder `700`) as
      `{"Authorization": "Bearer <key>"}`. **Verified**: the cloud handshake succeeds from the laptop
      (`wall-x` `2610-dualhead-v2`) via the `wallx-api` probe.
- [x] Claude config: `CLAUDE.md` + skills `robot-task`, `robot-harness`, `wallx-api`; `.venv` created,
      offline harness tests pass; `tasks/` holds task drafts.
- [ ] Decide what to do with the remaining staged deletions on branch `robot-env-setup` (old `CLAUDE.md`,
      skills, `tasks/`, `tools/`, root PDF). Nothing is committed yet.
- [x] Layout: `harness/` restored (byte-identical to the demo in `docs/`). `gui/` gets created in Phase 5.
  ```
  docs/      organizer PDFs (git-ignored) + demo
  harness/   harness.py, test_harness.py, requirements.txt  (copied from docs/participant-guide-demo)
  gui/       server.py + static/index.html
  plan.md
  ```

## Phase 1 — Ask the organizers (5 min, saves hours)

Laptop-side checks (2026-11-21):
- WebUI `http://192.168.77.2/` → HTTP 200 (16 ms); robot SSH port 22 open.
- Laptop **ufw is active**, so the Phase 3 rule for port 8001 is required.
- Cloud Wall-X: connect + handshake ~1.3 s; **WebSocket RTT median ~290 ms (213–1423 ms)** before
  uploading 3 JPEGs or any inference. That adds ~0.3 s to every ~1.1 s chunk.

- [x] Which Wall-X address? **There is no on-site endpoint.** Use the cloud
      `wss://maas-api.x2robot.com/v1/vla/wallx` (auth `Bearer`, already verified). The ~290 ms network RTT is fixed,
      so reduce latency elsewhere: one persistent connection per run, and keep the harness hop on wired Ethernet.
- [x] Auth for the on-site address: not applicable.
- [ ] Confirm our gripper is **G** (`quanta_x1_pro/gripper-g`) and not H. Fallback: check in Phase 2 that the
      gripper reading stays ≤ 1.526 when fully open.

## Phase 2 — Configure Wall-X directly in the WebUI (no code)

> **Blocker (2026-11-21):** `http://192.168.77.2/robot/task-manage` shows "X2Robot Operator Login" (PD cloud
> account). This WebUI build has **no "Local Environment"** option: the environment dropdown offers only
> 正式/开发/测试/自定义 (production/dev/test/custom). Ask the organizers for an operator account or the local-mode
> procedure. Discovered on the robot: **rosbridge (Tornado) on `:9090`** (useful read-only for the Phase 5
> dashboard) and a Python `websockets` server on `:8888`.

This proves the robot, the model and the auth work before we add anything.

- [ ] `ping 192.168.77.2` from the laptop (already OK, ~2 ms over Ethernet).
- [ ] Open `http://192.168.77.2/` → login → **Local Environment** → Task Management.
      Check the 3 camera feeds and robot status.
- [ ] Create a task file `local_cup_test.json` (based on `docs/participant-guide-demo (1)/participant-guide/local_my_task_benchmark.json`):
      `mode: benchmark`, `robot_type: DESKTOP`, `mode_type: EndPose`, `infer_type: sync`,
      `infer_params: {action_horizon: 32, control_dt: 0.005, interpolate_multiplier: 8,
      async_params: {api_schema: public_v2, embodiment: quanta_x1_pro/gripper-g,
      normalization: provided, control_context: masked}}`, one Case with a simple instruction.
      **No API key in this file.**
- [ ] Validate and deploy:
  ```bash
  python3 -m json.tool local_cup_test.json > /dev/null
  scp local_cup_test.json xr@192.168.77.2:/home/xr/diversity/local_cup_test.json
  ```
- [ ] WebUI → refresh the task list → open the task → **Edit Model Configuration**: model address =
      the Wall-X address from Phase 1, `EndPose`, `sync`, the params above, and the auth from Phase 1. Save.
- [ ] Clear the workspace → **Start Evaluation Inference Mode** → watch → **End Evaluation Inference Mode** → submit scores.
- [ ] Note the baseline: success, time, and how long the first vs later chunks took.

If something fails: `unknown_embodiment` → wrong gripper suffix or capitalisation. `401/403` → auth.
Old address still used → a WebUI override in `/home/xr/diversity/local/state/model_overrides.json`
(Restore Default may wipe the key).

## Phase 2b — "What does the camera see" with Nebius (read-only, no login needed)

Runs alongside the Phase 2 blocker: rosbridge `:9090` serves the cameras with no WebUI login.

- [x] `perception/camera.py`: one JPEG per camera via rosbridge (head 1280×720, wrist 640×480; < 70 ms).
- [x] `perception/nebius_vision.py`: OpenAI-compatible call to `https://api.tokenfactory.nebius.com/v1`.
- [x] `perception/see.py` CLI + 10 offline tests (`.venv/bin/python -m unittest -v perception/test_perception.py`).
- [x] User: create a Token Factory API key → `python3 perception/set_nebius_key.py`.
- [x] Model chosen by side-by-side test: **`Qwen/Qwen3.8-27B`, thinking off (~1.2 s)**; runner-up
      `nvidia/Nemotron-3.5-Super-VL` (~1.8 s). The reasoning models return empty text if `max_tokens` is
      spent on thinking. Gemma-3 reads images at low resolution; Kimi ignores the thinking switch.
- [x] First live run on the head camera: objects listed correctly, and pen-vs-book position verified against the saved frame.
- [ ] **User:** write `prompts.scene_json_prompt()` (the scene schema), then `see.py --json`.
- [ ] Confirm which arm `cam1` / `cam3` belong to (move one arm, watch which image changes).
- [x] Browser UI `perception/webui.py` (`http://127.0.0.1:8080`): live camera (Head / Wrist 1 / Wrist 3), question
      box, answer box, "model saw" thumbnails, history. 9 more offline tests (21 total). Verified live in a headless browser.
- [ ] Later: merge into the Phase 5 dashboard (arm readings via rosbridge `/left_arm/end_pose`, `/joint_states`).

## Phase 2c — Natural-language movement with our own safety checks (DRY-RUN)

- [x] `motion/kinematics.py`: FK/IK from the robot's URDF; matches `/{side}_arm/end_pose` exactly.
- [x] `motion/safety.py`: step ≤ 3 cm / 15°, IK reachable, joint limits (5° margin, "don't make worse"),
      joint change ≤ 34°, workspace box, reach, table clearance + arm separation checked at 6 points along the path,
      gripper range, contact-force monitor. Limits in `motion/safety_config.json`.
- [x] `motion/command_parser.py`: Nebius turns a command into JSON moves, or asks a clarifying question.
- [x] UI: live Arms panel (joints with limit bars, fingertip, gripper, force change) + Move panel (interpretation,
      every check ✓/✗, message preview). The send button is locked. 42 offline tests in total.
- [x] Pick planning (dry-run): "pick the cube in front of you" → vision-model box → depth → 3D (Rubik's cube
      measured 5.6–5.7 cm tall, true 5.7) → 23-step plan (right arm, 60° tilt; straight-down grips hit joint4's
      limit). Overlay shows the box and planned fingertip path. 53 offline tests.
- [x] Head-camera position corrected by +7.3 cm from the robot's hand-eye calibration; projected fingertips
      verified on the image.
- [ ] **User:** confirm the table height. Depth measures it at z ≈ −1 cm, so `table_z: 0.0` is about right but
      unconfirmed. Confirm the gripper direction (is 1.526 open?) and that a 5.7 cm cube fits the open gripper;
      then set the `*_verified` flags in `safety_config.json`.
- [x] Read-only command recorder `motion/record_commands.py` (+ tests). Fixed the UI starving rosbridge
      (camera stream now 4 fps and on demand; joint states back to 10–16 Hz).
- [ ] **Record the platform's own commands** while someone moves the arm (leader arm / teleop / Wall-X run):
      `.venv/bin/python motion/record_commands.py --seconds 15`. Need the verdict "CONFIRMED" for each arm.
- [x] Live sender built (`motion/sender.py`, UI Execute/Stop, `--live` flag); locked until the convention is verified.
      63 offline tests (sender tested against a fake socket).
- [x] **`pose_cmd_convention` verified from the robot's own recordings:** `relative_to_neutral`, empty frame_id
      (12 moving arm sessions: 1–4 mm vs 215–218 mm). Static recordings are reported as inconclusive. 66 tests.
- [x] First attempt via raw pose_cmd: the arm didn't move (platform in base mode ignores external commands).
- [x] Official SDK v1.1.2 (gRPC :50051) connected; sender now uses `set_joint_positions` in SDK mode; UI has
      Enter/Exit SDK mode + Robot E-stop. 70 offline tests; bridge verified read-only on the robot.
- [x] **First live move via SDK works (2026-11-22):** typed command → checked plan → `set_joint_positions` → arm
      moved. Lessons: SDK mode must be entered by the server's own connection (UI button; it ends when that
      connection closes), and the arm settles ~2° when SDK mode takes over, so plan right before Execute.
- [x] Architecture page in the UI (`/architecture`, "Architecture" button) + shared artifact.
- [x] Pick execution: commands now paced by the arm's real progress (≤ 8 mm / 4° lead) because the controller's
      jump detection (2 cm / 0.2 rad) slowed the arm to ~1 %. The arm then moved through pick steps.
- [x] "Go home" command (checked joint steps ≤ 10° to a ready pose 6° off the limits); workspace check now lets an
      arm that is outside move back in. 79 tests.
- [x] Contact check now uses sudden jumps (> 12 N in 0.3 s) or a large total (> 40 N): pose-dependent drift of the
      force estimate (18 N over a long go-home) caused a false stop.
- [x] Left arm didn't move: its SDK targets reached `/left_arm_joint_controller/commands` while that controller
      was inactive. The sender now always re-requests joint control and names an inactive controller when an arm
      doesn't follow. Camera subscription watchdog added. Exit SDK mode goes through the platform STOP.
- [x] "both arms go home" + a "Both arms home" button: arms move one after the other, each checked against the
      other's real position; both orders tried. 82 tests.
- [ ] Next: Both arms home → "grab the cube with the left arm". Watch: the USB Ethernet adapter dropped 3 times.
- [ ] When done: **Exit SDK mode** (shared robot).
- [ ] First live test (only after both of the above): left arm +1 cm in z, hand on the E-stop; expect
      `end_pose` z +1.0 cm and nothing else.
- [ ] Only after approval: add a sender (one move at a time, confirm click, live joint/contact monitor → `/safety_controller/halt`).

## Phase 3 — Put the harness in the middle (passthrough)

- [ ] On the laptop:
  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r harness/requirements.txt
  python -m unittest -v harness/test_harness.py
  export WALLX_ENDPOINT='<address from Phase 1>'
  export WALLX_HEADERS_FILE=~/.config/wallx/headers.json
  python harness/harness.py --host 192.168.77.18 --port 8001 --passthrough
  ```
- [ ] Allow the robot to reach the laptop: `sudo ufw allow from 192.168.77.0/24 to any port 8001` (if ufw is on).
- [ ] WebUI → Edit Model Configuration → model address `ws://192.168.77.18:8001`. Same params.
      Auth now comes from the harness, not the WebUI.
- [ ] Start a Case. Each `event=chunk` line in the terminal is one round trip. Behaviour should match Phase 2.
- [ ] Stop order: **WebUI End Evaluation → submit scores → Ctrl+C harness.**

## Phase 4 — Typed commands + timing (harness changes)

- [ ] `CommandState`: current instruction, pending instruction, history. Initialised from the task's instruction.
- [ ] In `Harness.handle()`, set `request["instruction"] = CommandState.current` before forwarding.
      Keep everything else (observation, control_context, handshake, reply) byte-for-byte.
- [ ] Validate typed text: non-empty, ≤ 200 chars. Suggest the "Verb + object." style.
- [ ] **Your decision, the switch policy:** a new command applies on the *next request*, *after the current
      chunk* (shown as pending → active), or only when you press *Apply*.
- [ ] Time each request: receive from robot → send to Wall-X → reply → send to robot. Log it in the
      `event=chunk` JSON, and flag the first (warm-up) call separately.
- [ ] Tests in `test_harness.py`: the typed command replaces `instruction`; nothing else changes;
      empty/oversized text is rejected; a GUI disconnect doesn't break the relay.

## Phase 5 — Browser dashboard

### Server (`gui/server.py`, same process as the harness)
- [ ] `aiohttp` HTTP + WebSocket on **`127.0.0.1:8080`** (localhost only, so only you can send commands).
- [ ] On every robot request, push to the browser:
  - the 3 JPEGs (already base64, so send as-is; throttle to ~5 fps);
  - `left_ee` / `right_ee`: position, rpy, gripper;
  - the active instruction, Wall-X's first returned frame per arm, and the timing per hop.
- [ ] Receive `{"type":"command","text":"…"}` → `CommandState`.
- [ ] Status: robot client connected? Wall-X reachable? Last error, "last update N s ago".

### Page (`gui/static/index.html`, single file, no build step)
- [ ] Three camera tiles (front / left wrist / right wrist). The robot's right arm appears on the **left** of the front image.
- [ ] Per-arm panel: x/y/z (m), roll/pitch/yaw (rad and °), gripper bar `0–1.526`, rolling 30 s plots.
- [ ] Latency panel: per-hop times, a sparkline, and the warm-up call marked.
- [ ] Command box + Send. Active/pending instruction. History list.
- [ ] Event log (chunks, errors).
- [ ] Permanent banner: **"Stop = WebUI → End Evaluation Inference Mode"** with a link to `http://192.168.77.2/`. No fake stop button.

Note: the dashboard updates only while a Case is running, because the client only sends data then.

## Phase 6 — Use it

- [ ] Clear workspace; keep the WebUI stop in reach.
- [ ] `python harness/harness.py --host 192.168.77.18 --port 8001 --passthrough` (with GUI) → open `http://localhost:8080`.
- [ ] WebUI → select the Case → Start Evaluation Inference Mode.
- [ ] Type commands as each stage finishes, e.g. `Pick up the cup.` → `Put the cup in the bowl.`
- [ ] End **in the WebUI** → submit scores → Ctrl+C the harness.
- [ ] Compare with the Phase 2 baseline (success, time, latency).

## Later, optional

- Always-on readings between runs: a read-only ROS 2 bridge on the robot (ask the organizers first).
- Simple direct moves ("left arm up 3 cm", "open gripper") generated by the harness, for low-latency jogging.
- RTC smoothing (API Tutorial §2), if the robot client's config supports it.

## Risks

| Risk | Mitigation |
|---|---|
| API key leaks via the `docs/` PDF | `.gitignore` in Phase 0; check `git status` before every commit |
| Latency | On-site address; ignore the first warm-up call; the GUI shows where time goes |
| Command changes mid-grasp → odd motion | Switch policy (Phase 4); short "Verb + object." commands |
| Treating closing the GUI/harness as a stop | Banner + stop order; stop only in the WebUI |
| WebUI override hides config changes | Check `model_overrides.json`; re-enter config after Restore Default |
