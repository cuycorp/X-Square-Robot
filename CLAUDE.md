# X-Square Robot hackathon workspace

This machine **is the robot host** (user `xr`, LAN IP `10.19.20.98`). The robot stack is
already running: `ros2 launch x2robot_infer infer_with_dispatcher.launch.py`
(`x2robot_infer_node` + `x2robot_action_dispatcher_cpp`) and `moveit_servo_desktop`.

Source of truth: `Hackathon Participant Guide.pdf` (extract with `pdftotext -layout`).

## Layout
- `harness/` – organizer demo: `harness.py` (sync WebSocket relay + VisionAgent), `test_harness.py` (offline), `requirements.txt`
- `tasks/` – task JSON drafts (`local_<task_id>.json`); deployed copies live in `/home/xr/diversity/`
- `.claude/skills/` – `robot-task`, `robot-harness`, `wallx-api`
- `.venv/` – Python venv for the harness (not committed)

## Safety rules (hard)
- **Never start another control process** (`start_desktop.sh`, a second `infer_with_dispatcher`, etc.) while the existing nodes run. Use `--attach` if terminal debugging is required, and only after the user confirms.
- **Stopping, homing and scoring happen only in the WebUI** (`http://10.19.20.98/`). Killing the harness, closing a WebSocket, or an agent saying "stop"/"done" does NOT stop the arm. Always tell the user to click "End Evaluation Inference Mode" first, then Ctrl+C the harness.
- Never replay cached actions after a disconnect; rebuild session and fetch fresh observations.
- Do not send anything that makes the arm move without the user explicitly asking in that turn.

## Secrets
- The WALLX key lives in `/home/xr/diversity/local/state/model_overrides.json` (`async_params.api_key`). Never print it, commit it, or put it in task JSON. Redact when displaying that file.
- Harness auth: `WALLX_HEADERS_FILE` → JSON `{"Authorization": "Bearer <key>"}`, mode 600, outside the repo (`~/.config/wallx/headers.json`).
- Agent key (`AGENT_API_KEY`) is separate from the WALLX key.
