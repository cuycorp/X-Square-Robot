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

## Live cameras

In the Prompt & Camera Console, enter the robot's rosbridge URL, for example `ws://192.168.77.2:9090/` on the wired hackathon setup, and click **Connect**. The front camera and both numbered camera feeds appear side by side.

The camera preview works before inference starts. Each compressed image subscription requests up to 2.5 frames per second. **Disconnect** closes the telemetry connection. **Inference details** shows the separate frames used by the planner once it receives observations.

## Claude zero-shot planning

In Claude mode (`launch.py --mode claude`), enter a task in **Claude Zero-Shot Task** and click **Run Zero-Shot Prompt**. The planner begins processing it when evaluation inference is active in the robot WebUI. The policy receives three camera observations and measured arm poses from the robot client. Claude proposes up to three short Cartesian waypoints for the current stage. The policy generates a bounded pose trajectory toward one waypoint at a time, checks the measured pose before advancing, and replans when the sequence ends or after four observation chunks by default. The console shows the active stage and waypoint. Use `--plan-every` to change the replanning interval.

End the manual SDK session before starting the Claude policy. Claude mode requires an Anthropic API key on the machine running the launcher. The robot WebUI task must point to the policy WebSocket (port `8001` by default), and the browser prompt belongs in the policy console (port `8002` by default).

For a laptop-only key setup page on port 8004, use the following command for the wired station. The policy listens on the laptop's Ethernet address and accepts WebSocket sessions only from the assigned robot.

```bash
python harness/server.py --mode claude --host 192.168.77.10 --port 8001 --robot-ip 192.168.77.2 --console-host 127.0.0.1 --console-port 8004 --local-key-setup
```

Open `http://127.0.0.1:8004/`, verify an Anthropic API key, and enter a prompt. The key is kept in the policy server's memory. Click **Release Manual Control** in the manual console, wait for the robot to show idle, then select the task pointing to `ws://192.168.77.10:8001` in the robot WebUI and start evaluation. The policy refuses evaluation sessions until the key and a laptop prompt are set.

Workspace and step limits bound the generated trajectories. Collision and reachability checks depend on the downstream robot controller and its configuration. Claude mode sends inference camera observations to the configured Anthropic API endpoint.

### Direct Claude control over the local SDK

On the wired Desktop G station, start the SDK console from the laptop's Linux environment with the SDK installed and a matching station URDF:

```bash
python harness/jog_console.py --port 8003 --sdk-address 192.168.77.2:50051 \
  --urdf STATION.urdf --direct-rosbridge ws://192.168.77.2:9090/
```

Use this console as the station's only motion client. Open `http://127.0.0.1:8003/`, verify the Anthropic key, then click **Run Zero-Shot Prompt**. The server reads the three cameras from rosbridge and both arm poses from SDK joint feedback and the station URDF. It sends the images and poses to Anthropic for planning, then sends bounded left-arm and gripper targets to the robot over the local SDK. This mode runs independently of the robot WebUI and task list.

Manual motion is locked while a direct run is active. **Stop Claude Control** cancels further targets. A target already sent to the controller can still finish. Each observation cycle smoothly interpolates up to 30 mm and 12 degrees (in 0.6 degree joint substeps), within a 35 cm travel radius and 150 second run limit. An unclear plan, failed IK, or lost feedback stops the run. The station operator remains responsible for checking clearance and using the physical stop if needed.

For cube prompts, **Use depth for cube position** adds the locally measured top-face center and nearby tabletop height in `base_link`. Depth is registered to the front RGB image using timestamped frames, CameraInfo, and static transforms. RGB images and measured coordinates reach Claude, while the depth image stays on the local network. Clear the checkbox to use RGB alone. The status line shows SDK command acceptance and displacement from the next measured observation separately from Claude's plan.

## Joint angles and kinematics

Connecting to rosbridge also displays the measured angles for six joints on each arm. **Advanced IK** opens the pose preview, which uses the robot model from `/robot_description` and joint angles from `/joint_states`. Choose an arm and click **Use Current Pose** to start from its measured gripper-base pose, or enter a target in `base_link`. **Calculate IK** displays a joint solution and its pose residual when the solver converges.

The preview enforces the URDF joint limits and sends no robot commands. It performs a kinematic calculation without collision checks.

### Incremental jogging

**Incremental Jog** has +/- buttons for X, Y, Z, Rx, Ry, Rz and J1–J6. Cartesian buttons use the same URDF solver to move the gripper base in the robot's `base_link` frame. Joint buttons change one joint target. Each click uses the latest measurements and the selected step size.

Run `python harness/jog_console.py` for a local preview. Connect to rosbridge as above to calculate targets without writing to the robot.

For physical jogging on an assigned, idle Desktop station, install the [X Square Desktop SDK](https://github.com/X-Square-Robot/sdk_robot/tree/main/samples/desktop) on a supported Linux host. Stop evaluation inference and other motion controllers before starting the manual session. Run `python harness/jog_console.py --sdk-address HOST:PORT --urdf STATION.urdf` with the station's matching URDF. This starts a separate localhost console and selects SDK joint-position mode. Each accepted click uses fresh SDK feedback and sends one joint target. The jog solver limits increments and joint angles. It does not check the scene for collisions.

In the live manual console, use the left or right arm panel to home that arm or open and close its gripper. Homing uses the SDK reset. The gripper buttons use that gripper's limits from the station URDF. The joint table shows both measured gripper angles.

## Rubik's Cube Live Detection and 3D Model

Once connected to rosbridge (or while port `8001` inference frames are streaming), the console runs a local NumPy and Pillow detector (`harness/cube_vision.py`) on the selected camera stream and overlays the bounding quad, 3x3 sticker classifications, and an uncalibrated optical-frame distance estimate directly on the live camera tile.

- **Interactive 3D Cube Viewer**: Drag the 3D cube in the console to orbit (`Iso`, `Front`, `Back`, `Top`, `Bottom`), rendered offline with HTML5 Canvas and zero external CDN scripts.
- **Multi-Face Accumulation (`U R F D L B`)**: Click **Capture Face** to store the live detected 3x3 grid into `Auto face (center)` or an explicit face (`F`, `R`, `U`, `L`, `B`, `D`), click **Rotate Selected 90°** to align tilted scans, or click any sticker in the unfolded net to cycle its color (`W`, `Y`, `R`, `O`, `B`, `G`, `?`).
- **Claude Multi-Face Scan**: When an Anthropic key is configured, click **Scan with Claude** to inspect all visible live camera views at once and populate oblique faces.
- **Invariant Validation**: The console tracks scanned faces, per-color counts (expecting 9 per color and 6 unique centers), and outputs a canonical 54-character `U R F D L B` facelet string once all 6 faces are complete.

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
