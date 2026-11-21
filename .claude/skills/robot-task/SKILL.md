---
name: robot-task
description: Create, validate and deploy X-Square benchmark task JSON (local_<task_id>.json) to the robot. Use for new tasks or Cases, scoring criteria, WebUI model configuration, or a task that is missing or shows stale config in the WebUI.
---

# Robot task JSON (benchmark mode)

The robot WebUI loads evaluation tasks from the **robot's** `/home/xr/diversity/local_<task_id>.json`.
Drafts live in `tasks/` here. Template: `tasks/local_my_task_benchmark.json`.

## Rules

- Filename `local_<task_id>.json`, task_id `[a-z0-9_-]+`. `collection_id` = filename minus `.json`
  (omit it, or match exactly). `mode: "benchmark"`. `task_name` is the display name.
- `model`: `robot_type: DESKTOP`, `mode_type: EndPose`, `infer_type: sync`, `deploy: lan`.
  - `model_address`: the Wall-X endpoint for direct runs (on-site `ws://HOST:33677` or
    `wss://maas-api.x2robot.com/v1/vla/wallx`); `ws://192.168.77.18:8001` when going through the harness.
  - `infer_params` (G gripper):
    ```json
    {"action_horizon": 32, "control_dt": 0.005, "interpolate_multiplier": 8,
     "async_params": {"api_schema": "public_v2", "embodiment": "quanta_x1_pro/gripper-g",
                      "normalization": "provided", "control_context": "masked"}}
    ```
    The H gripper uses `quanta_x1_pro/gripper-h`, matching the physical gripper. Keep `infer_params`
    identical between direct and harness runs so results compare.
- `data`: non-empty list of Cases, each with a unique `case_id`, a non-empty `instruction`,
  `env_description` (object), `scores_dict` (name → non-negative number), `timeout` (s), and an
  optional `image_path` (`images/x.jpg` under the robot's `/home/xr/diversity/`).
- Only `instruction` reaches the model. `env_description`, `scores_dict` and the image are for humans.
- Keys go in the WebUI's Edit Model Configuration, never in task JSON (the validator flags them).

## Design

Define objects, start positions, expected outcome and timeout first. Make success criteria
outcome-based ("cup is in the bowl and released"), not a single frame. Start with single-step tasks;
one Case per initial-layout variant.

## Workflow

1. Draft `tasks/local_<id>.json`.
2. Validate: `python3 .claude/skills/robot-task/scripts/validate_task.py tasks/local_<id>.json` → `[OK]`.
3. Deploy. The user runs this (SSH password), after confirming it's OK to change what the WebUI shows:
   `scp tasks/local_<id>.json xr@192.168.77.2:/home/xr/diversity/`
4. The user, in the WebUI: stop any current run → refresh the task list → re-select the task →
   Edit Model Configuration (address, `EndPose`, `sync`, params, auth) → Save.

## WebUI overrides

WebUI edits are saved per `collection_id` on the robot in
`/home/xr/diversity/local/state/model_overrides.json` and **override the task JSON's `model` block**.
When an address change doesn't take, inspect the file redacted (the user runs it):

```bash
ssh xr@192.168.77.2 cat /home/xr/diversity/local/state/model_overrides.json \
  | python3 .claude/skills/robot-task/scripts/show_overrides.py
```

"Restore Default" clears the override **and the API key**. New tasks don't inherit keys.
