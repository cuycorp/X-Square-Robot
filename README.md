# X-Square Robot: Prompt-Driven Autonomous Claude Control

Autonomous natural-language control for the `quanta_x1_pro` dual-arm desktop robot using Anthropic Claude Vision (`claude-sonnet-5-5`), without needing GPU inference hardware or Wall-X cloud access.

## Quickstart on the Robot PC

1. Clone this repository onto the robot PC and enter it:
   ```bash
   git clone https://github.com/cuycorp/X-Square-Robot.git
   cd X-Square-Robot
   ```
2. Launch the autonomous policy server:
   ```bash
   ./start.sh
   ```
   Or pass your initial prompt directly:
   ```bash
   python3 launch.py --prompt "Pick up the spoon and place it in the bowl"
   ```
3. When prompted, paste your **Claude (Anthropic) API key** (hidden input, optionally stored with `0600` permissions in `~/.config/anthropic/api_key` outside the repository).
4. In the robot WebUI (`http://<LAN_IP>/`):
   - Open **Task Management**, refresh, and select `claude_autonomous_prompt` (automatically copied to `/home/xr/diversity/local_claude_prompt_benchmark.json`).
   - Confirm `model_address` is `ws://<LAN_IP>:8001` (`EndPose`, `sync`, `public_v2`).
   - Click **Start Evaluation Inference Mode**.
5. Type new task prompts anytime in the terminal (`prompt> ...`) or in the browser Prompt & Camera Console (`http://<LAN_IP>:8002/`).

## Safety

- Never start another ROS2 control node (`start_desktop.sh` or `infer_with_dispatcher`) while the robot stack is already running.
- Stopping, homing, and scoring always happen in the robot WebUI. Click **End Evaluation Inference Mode** in the WebUI first before exiting the launcher.

## Offline Testing (No Robot Required)

Run the offline unit and WebSocket loopback tests:
```bash
python3 -m unittest discover -s harness -p "test_*.py" -v
```
Run the CPU tabletop simulator with offline heuristic or live Claude:
```bash
python3 launch.py --sim --mode offline
python3 launch.py --sim --mode claude --prompt "Pick up the red object and place it into the blue bowl."
```
