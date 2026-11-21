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
 │ organizer client │──────▶│ harness.py :8001             │─────────▶│ on-site ws://HOST:33677│
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

- [ ] Which Wall-X address should we use: the **on-site `ws://HOST:33677`** (low latency, preferred) or the cloud `wss://maas-api.x2robot.com/v1/vla/wallx`?
- [ ] The auth header format for the on-site address (the cloud one already works with `Bearer`).
- [ ] Confirm our gripper is **G** (`quanta_x1_pro/gripper-g`) and not H.

## Phase 2 — Configure Wall-X directly in the WebUI (no code)

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
