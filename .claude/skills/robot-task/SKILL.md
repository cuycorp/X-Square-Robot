---
name: robot-task
description: Create, edit, validate and deploy X-Square desktop robot benchmark task JSON files (local_<task_id>.json in /home/xr/diversity). Use when the user wants a new robot task, new test Cases, scoring criteria, or to fix a task that doesn't show up / shows stale config in the WebUI.
---

# Robot task JSON (benchmark mode)

The WebUI loads inference-evaluation tasks from `/home/xr/diversity/local_<task_id>.json`
(this machine is the robot host, so deploy is a local copy — no scp needed).
Drafts live in `tasks/` in this repo. Template: `tasks/local_my_task_benchmark.json`.
Existing reference task: `/home/xr/diversity/local_desktop_power_cord_insert_benchmark.json`.

## Rules
- Filename `local_<task_id>.json`; task_id = `[a-z0-9_-]+`. `collection_id` = filename minus `.json` (omit it, or it must match exactly).
- `mode` = `"benchmark"`. `task_name` = display name (free).
- `model`: `robot_type: DESKTOP`, `mode_type: EndPose`, `infer_type: sync`, `deploy: lan`.
  - `model_address`: `wss://maas-api.x2robot.com/v1/vla/wallx` for direct, or `ws://10.19.20.98:8001` when going through the harness.
  - `infer_params` (Desktop G gripper):
    ```json
    {"action_horizon": 32, "control_dt": 0.005, "interpolate_multiplier": 8,
     "async_params": {"api_schema": "public_v2", "embodiment": "quanta_x1_pro/gripper-g",
                      "normalization": "provided", "control_context": "masked"}}
    ```
    (The pre-installed power-cord task uses `interpolate_multiplier: 12` and `"algorithm": "sync"` — keep whatever the on-site deployment uses, and keep it identical between direct and harness runs.)
  - H gripper → `quanta_x1_pro/gripper-h`. Must match the physical gripper.
- `data`: non-empty list of Cases, each with unique `case_id`, non-empty `instruction`, optional `image_path` (relative like `images/x.jpg` under `/home/xr/diversity/images/`, or URL), `env_description` (dict), `scores_dict` (name → non-negative number), `timeout` (seconds).
- **Never put API keys in task JSON.** Keys are entered via the WebUI "Edit Model Configuration".
- Only `instruction` reaches the model. `env_description`/`scores_dict`/image are for humans; the harness must load them itself if the agent needs them.

## Design guidance
Define objects, initial positions, expected outcome, timeout first. Success criteria should be
outcome-based ("spoon is inside the bowl and released"), not a single momentary frame.
Start with single-step tasks the VLA does reliably; one Case per initial-condition variant.

## Workflow
1. Draft/edit `tasks/local_<id>.json`.
2. Validate: `python3 .claude/skills/robot-task/scripts/validate_task.py tasks/local_<id>.json`
3. Deploy (confirm with user first — it changes what the WebUI shows):
   `cp tasks/local_<id>.json /home/xr/diversity/`
4. Tell the user: in WebUI Task Management, stop any current run, refresh the list, re-select the task.

## Model overrides gotcha
WebUI edits are stored per collection_id in `/home/xr/diversity/local/state/model_overrides.json`
and **override the task JSON's `model` block**. If a model address change "doesn't take",
check that file (redact `api_key` when showing it). "Restore Default" in the WebUI clears the override **and the API key**.
New tasks do not inherit keys from other tasks — the user must enter the key in the WebUI.
To inspect safely:
```bash
python3 .claude/skills/robot-task/scripts/show_overrides.py
```
