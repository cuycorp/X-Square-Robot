"""Run the local console for increment previews or SDK-backed manual jogging."""

from __future__ import annotations

import argparse
import math
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from claude_policy import ClaudeEndPosePlanner, PlannerDecision, load_anthropic_api_key
from cube_depth import camera_mount
from direct_claude import DirectClaudeControl
from kinematics import arm_chain, jog_target, measured_pose, pose_target
from protocol import ArmState, ValidatedRequest
from server import PromptController, start_console_http_server
from trajectory import WorkspaceLimits, _min_jerk, clamp_pose, matrix_to_rpy, rotation_distance


DIRECT_MAX_STEP_POS_M = 0.03
DIRECT_MAX_STEP_ROT_RAD = math.radians(12)
DIRECT_MAX_TRAVEL_RADIUS_M = 0.35
DIRECT_SUBSTEP_MAX_JOINT_RAD = math.radians(0.6)
DIRECT_INTERP_PERIOD_S = 0.022
DIRECT_SETTLE_TIMEOUT_S = 0.6


class DesktopJogger:
    """Serialize manual SDK requests and check feedback before sending them."""

    def __init__(
        self, robot: Any, joint_mode: Any, command_type: Any, urdf: str, gripper_type: Any = None
    ) -> None:
        self.robot = robot
        self.joint_mode = joint_mode
        self.command_type = command_type
        self.gripper_type = gripper_type
        self.chains = {arm: arm_chain(urdf, arm) for arm in ("left", "right")}
        root = ET.fromstring(urdf)
        self.gripper_limits = {}
        for arm in self.chains:
            limit = root.find(f"./joint[@name='{arm}_arm_gripper']/limit")
            if limit is not None:
                lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                if math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper:
                    self.gripper_limits[arm] = (lower, upper)
        self._lock = threading.Lock()
        self._busy_until = -math.inf
        self._released = False
        self._direct_stop: threading.Event | None = None

    def reserve_direct(self, stop: threading.Event) -> None:
        with self._lock:
            if self._released or self._direct_stop is not None or time.monotonic() < self._busy_until:
                raise RuntimeError("Wait for manual motion to settle before starting Claude.")
            self._direct_stop = stop

    def finish_direct(self, stop: threading.Event) -> None:
        with self._lock:
            if self._direct_stop is stop:
                self._direct_stop = None

    @contextmanager
    def _stationary_arm(self, arm: str, *, direct_stop: threading.Event | None = None) -> Iterator[tuple[Any, Any]]:
        if arm not in self.chains:
            raise ValueError("Select the left or right arm")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A manual request is still in progress")
        try:
            if self._direct_stop is not direct_stop:
                raise RuntimeError("Manual controls are locked during direct Claude control.")
            if direct_stop is not None and direct_stop.is_set():
                raise RuntimeError("Direct Claude control stopped.")
            if self._released:
                raise RuntimeError("Manual SDK control has been released.")
            if time.monotonic() < self._busy_until:
                raise RuntimeError("Wait for the previous motion to settle")
            if self.robot.robot_control.get_manipulator_control_mode(timeout=1).mode != self.joint_mode:
                raise RuntimeError("The SDK controller is no longer in joint-position mode")
            controller = self.robot.left_arm if arm == "left" else self.robot.right_arm
            before = controller.get_joint_states(timeout=1).header.stamp
            time.sleep(0.04)
            current = controller.get_joint_states(timeout=1)
            stamp = current.header.stamp
            if (stamp.sec, stamp.nanosec) <= (before.sec, before.nanosec):
                raise RuntimeError("The SDK did not provide advancing joint feedback")
            velocity = dict(zip(current.name, current.velocity))
            expected = [f"{arm}_arm_joint{index}" for index in range(1, 7)]
            if any(not abs(velocity.get(name, math.inf)) <= 0.1 for name in expected):
                raise RuntimeError("Wait for the arm to be stationary with valid velocity feedback")
            yield controller, current
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            detail = exc.details() if callable(getattr(exc, "details", None)) else type(exc).__name__
            raise RuntimeError(f"Manual SDK request failed: {detail}") from exc
        finally:
            self._lock.release()

    def _send_joint_target(self, controller: Any, joints: dict[str, float]) -> None:
        command = self.command_type(positions=list(joints.values()))
        self._busy_until = time.monotonic() + 0.4
        result = controller.set_joint_positions(command, timeout=1)
        if not result.is_success:
            raise RuntimeError(f"SDK rejected the jog: {result.error_message}")

    def __call__(self, arm: str, axis: str, direction: int, step: float) -> dict[str, float]:
        with self._stationary_arm(arm) as (controller, current):
            joints = jog_target(self.chains[arm], dict(zip(current.name, current.position)), axis, direction, step)
            self._send_joint_target(controller, joints)
            return joints

    def _measured_arm(self, arm: str, state: Any) -> dict[str, Any]:
        measured = dict(zip(state.name, state.position))
        position, rpy = measured_pose(self.chains[arm], measured)
        gripper = self.robot.left_gripper if arm == "left" else self.robot.right_gripper
        opening = float(gripper.get_position(timeout=1).position)
        if not math.isfinite(opening):
            raise RuntimeError("The SDK did not provide a valid gripper position.")
        low, high = self.gripper_limits[arm]
        if not low - 0.01 <= opening <= high + 0.01:
            raise RuntimeError("The measured gripper position is outside the station limits.")
        return {"position": position, "rotation_rpy": rpy, "gripper": min(high, max(low, opening))}

    def observe_direct(self, stop: threading.Event) -> dict[str, Any]:
        states = {}
        for arm in ("left", "right"):
            with self._stationary_arm(arm, direct_stop=stop) as (_, current):
                states[f"{arm}_ee"] = self._measured_arm(arm, current)
        return states

    def _wait_arm_settled(self, controller: Any, arm: str, stop: threading.Event) -> None:
        expected = [f"{arm}_arm_joint{index}" for index in range(1, 7)]
        deadline = time.monotonic() + DIRECT_SETTLE_TIMEOUT_S
        while True:
            current = controller.get_joint_states(timeout=1)
            velocity = dict(zip(current.name, current.velocity))
            if all(abs(velocity.get(name, math.inf)) <= 0.1 for name in expected):
                return
            if time.monotonic() >= deadline or stop.wait(0.04):
                return

    def apply_direct(
        self, observation: ValidatedRequest, decision: PlannerDecision,
        start_position: list[float], stop: threading.Event, expires_at: float,
    ) -> str:
        with self._stationary_arm("left", direct_stop=stop) as (controller, current):
            actual = ArmState(**self._measured_arm("left", current))
            planned_from = observation.left_ee
            if (
                math.dist(actual.position, planned_from.position) > 0.004
                or rotation_distance(actual.rotation_rpy, planned_from.rotation_rpy) > math.radians(1.5)
                or abs(actual.gripper - planned_from.gripper) > 0.05
            ):
                raise RuntimeError("The left arm moved while Claude was planning. Start a fresh run.")
            if (
                math.dist(observation.right_ee.position, decision.right_target_pos) > 0.001
                or rotation_distance(observation.right_ee.rotation_rpy, decision.right_target_rpy) > 0.01
                or abs(observation.right_ee.gripper - decision.right_gripper) > 0.02
            ):
                raise RuntimeError("Claude requested right-arm movement during a left-arm run.")
            position, rotation, opening = clamp_pose(
                decision.left_target_pos, decision.left_target_rpy, decision.left_gripper, actual,
                self.gripper_limits["left"],
                WorkspaceLimits(max_step_pos_m=DIRECT_MAX_STEP_POS_M, max_step_rot_rad=DIRECT_MAX_STEP_ROT_RAD),
            )
            if math.dist(position, start_position) > DIRECT_MAX_TRAVEL_RADIUS_M:
                raise RuntimeError("The target exceeds the 35 cm travel radius for this run.")
            measured = dict(zip(current.name, current.position))
            moving = (
                math.dist(position, actual.position) > 0.0005
                or rotation_distance(actual.rotation_rpy, matrix_to_rpy(rotation)) > 0.004
            )
            joints = pose_target(self.chains["left"], measured, position, rotation) if moving else None
            if stop.is_set() or time.monotonic() >= expires_at:
                raise RuntimeError("The run stopped or the observation expired before dispatch.")
            if joints is not None:
                names = list(joints.keys())
                q_start = np.array([measured[name] for name in names], dtype=float)
                q_goal = np.array([joints[name] for name in names], dtype=float)
                max_dq = float(np.max(np.abs(q_goal - q_start)))
                substeps = max(1, math.ceil(1.9 * max_dq / DIRECT_SUBSTEP_MAX_JOINT_RAD))
                for step_idx in range(1, substeps + 1):
                    if stop.is_set() or time.monotonic() >= expires_at:
                        raise RuntimeError("The run stopped or the observation expired before dispatch.")
                    blend = _min_jerk(step_idx / substeps)
                    q_cmd = q_start + blend * (q_goal - q_start)
                    self._busy_until = time.monotonic() + 0.25
                    result = controller.set_joint_positions(
                        self.command_type(positions=q_cmd.tolist()), timeout=1
                    )
                    if not result.is_success:
                        raise RuntimeError(f"SDK rejected the jog: {result.error_message}")
                    if step_idx < substeps and stop.wait(DIRECT_INTERP_PERIOD_S):
                        raise RuntimeError("Direct Claude control stopped.")
                self._wait_arm_settled(controller, "left", stop)
                self._busy_until = time.monotonic() + 0.08
                return "left arm target accepted by SDK"
            if abs(opening - actual.gripper) > 0.025:
                delta_g = opening - actual.gripper
                g_steps = max(1, math.ceil(abs(delta_g) / 0.35))
                for g_idx in range(1, g_steps + 1):
                    if stop.is_set() or time.monotonic() >= expires_at:
                        raise RuntimeError("The run stopped or the observation expired before dispatch.")
                    g_cmd = actual.gripper + _min_jerk(g_idx / g_steps) * delta_g
                    self._busy_until = time.monotonic() + 0.35
                    result = self.robot.left_gripper.set_position(
                        self.gripper_type(position=float(g_cmd)), timeout=2
                    )
                    if not result.is_success:
                        raise RuntimeError(f"SDK rejected the gripper target: {result.error_message}")
                    if g_idx < g_steps and stop.wait(0.05):
                        raise RuntimeError("Direct Claude control stopped.")
                stop.wait(0.2)
                self._busy_until = time.monotonic() + 0.08
                return "left gripper target accepted by SDK"
            return "hold"

    def action(self, arm: str, action: str) -> str:
        if action not in ("home", "open", "close"):
            raise ValueError("Choose home, open or close")
        if action != "home" and (self.gripper_type is None or arm not in self.gripper_limits):
            raise ValueError("The selected gripper needs valid limits in the station URDF")
        with self._stationary_arm(arm) as (controller, _):
            if action == "home":
                self._busy_until = time.monotonic() + 3
                result = controller.reset(timeout=15)
                detail = f"SDK accepted homing for the {arm} arm. Watch the arm and measured angles."
            else:
                gripper = self.robot.left_gripper if arm == "left" else self.robot.right_gripper
                lower, upper = self.gripper_limits[arm]
                target = upper if action == "open" else lower
                self._busy_until = time.monotonic() + 0.4
                result = gripper.set_position(self.gripper_type(position=target), timeout=2)
                detail = f"SDK accepted the {arm} gripper target of {target:g} rad."
            if not result.is_success:
                raise RuntimeError(f"SDK rejected {action}: {result.error_message}")
            return detail

    def release(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Wait for the current manual request to finish.")
        try:
            if self._direct_stop is not None:
                raise RuntimeError("Stop direct Claude control before releasing the SDK.")
            if time.monotonic() < self._busy_until:
                raise RuntimeError("Wait for the previous motion to settle.")
            self._released = True
        finally:
            self._lock.release()


def connect_desktop(address: str, model: Path) -> DesktopJogger:
    from x2robot import connect
    from x2robot.sdk import (
        GripperPosition, JointPositions, ManipulatorControlMode, ManipulatorControlModeParam,
        RobotModeParam, RobotModelType, RobotWorkMode,
    )

    robot = connect(f"x2://{address}")
    if robot.system.get_model_type(timeout=2).model_type != RobotModelType.DESKTOP:
        raise RuntimeError("The SDK endpoint did not identify itself as an X Square Desktop robot")
    mode = ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS
    jogger = DesktopJogger(robot, mode, JointPositions, model.read_text(encoding="utf-8"), GripperPosition)
    try:
        result = robot.system.set_work_mode(RobotModeParam(mode=RobotWorkMode.SDK), timeout=8)
    except Exception as exc:
        raise RuntimeError("The Desktop SDK did not confirm its work-mode switch") from exc
    if not result.is_success:
        raise RuntimeError(f"The robot declined SDK mode: {result.error_message}")
    try:
        result = robot.robot_control.set_manipulator_control_mode(ManipulatorControlModeParam(mode=mode), timeout=8)
    except Exception as exc:
        raise RuntimeError("The Desktop SDK did not confirm joint-position mode") from exc
    if not result.is_success:
        raise RuntimeError(f"The robot declined joint-position mode: {result.error_message}")
    return jogger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--sdk-address", help="Exclusive Desktop SDK session in host:port form")
    parser.add_argument("--urdf", type=Path, help="Trusted URDF matching the station, required for SDK jogging")
    parser.add_argument("--direct-rosbridge", help="Enable direct Claude control with this robot's rosbridge URL")
    args = parser.parse_args()
    if bool(args.sdk_address) != bool(args.urdf):
        parser.error("--sdk-address and --urdf must be supplied together")
    if args.direct_rosbridge and not args.sdk_address:
        parser.error("--direct-rosbridge requires a live SDK connection")
    front_mount = camera_mount(args.urdf.read_text(encoding="utf-8")) if args.direct_rosbridge else None
    jogger = connect_desktop(args.sdk_address, args.urdf) if args.sdk_address else None
    stop = threading.Event()

    def release_manual() -> str:
        jogger.release()
        # The SDK's close() doesn't stop its heartbeat thread. Exit this process
        # once the browser has had time to receive the release response.
        timer = threading.Timer(0.5, stop.set)
        timer.daemon = True
        timer.start()
        return "Manual SDK server is stopping. Wait for the robot to return to idle before starting evaluation."

    key_planner = ClaudeEndPosePlanner(api_key=load_anthropic_api_key())
    prompt_ctrl = PromptController()
    direct = DirectClaudeControl(
        jogger, key_planner, prompt_ctrl, args.direct_rosbridge,
        front_mount=front_mount,
    ) if args.direct_rosbridge else None
    server = start_console_http_server(
        "127.0.0.1", args.port, prompt_ctrl,
        live_jog=jogger, live_action=jogger.action if jogger else None,
        release_manual=release_manual if jogger else None, prompts_enabled=False,
        key_planner=key_planner, direct_control=direct,
    )
    print(f"Console at http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if direct is not None:
            direct.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
