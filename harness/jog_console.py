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

from claude_policy import ClaudeEndPosePlanner, load_anthropic_api_key
from kinematics import arm_chain, jog_target
from server import PromptController, start_console_http_server


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

    @contextmanager
    def _stationary_arm(self, arm: str) -> Iterator[tuple[Any, Any]]:
        if arm not in self.chains:
            raise ValueError("Select the left or right arm")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A manual request is still in progress")
        try:
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

    def __call__(self, arm: str, axis: str, direction: int, step: float) -> dict[str, float]:
        with self._stationary_arm(arm) as (controller, current):
            joints = jog_target(self.chains[arm], dict(zip(current.name, current.position)), axis, direction, step)
            command = self.command_type(positions=list(joints.values()))
            self._busy_until = time.monotonic() + 0.4
            result = controller.set_joint_positions(command, timeout=1)
            if not result.is_success:
                raise RuntimeError(f"SDK rejected the jog: {result.error_message}")
            return joints

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
    args = parser.parse_args()
    if bool(args.sdk_address) != bool(args.urdf):
        parser.error("--sdk-address and --urdf must be supplied together")
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
    server = start_console_http_server(
        "127.0.0.1", args.port, PromptController(),
        live_jog=jogger, live_action=jogger.action if jogger else None,
        release_manual=release_manual if jogger else None, prompts_enabled=False,
        key_planner=key_planner,
    )
    print(f"Console at http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
