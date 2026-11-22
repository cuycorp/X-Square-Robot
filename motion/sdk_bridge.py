#!/usr/bin/env python3
"""Bridge to the official X2Robot SDK (v1.1.2). Runs inside .venv-sdk; talks JSON lines on stdin/stdout.

Started and owned by motion/sender.py. Each request is one JSON object per line; each reply is one JSON line:
  {"cmd": "status"}                                   -> operation_mode, e-stop, arm states, control mode
  {"cmd": "state", "arm": "left"}                     -> joint positions + gripper position
  {"cmd": "work_mode", "mode": "sdk" | "idle"}       -> system.set_work_mode (the SDK's own mode switch)
  {"cmd": "set_mode", "mode": "joint_positions"}      -> set_manipulator_control_mode
  {"cmd": "joints", "arm": "left", "q": [6 floats]}   -> <arm>.set_joint_positions
  {"cmd": "gripper", "arm": "left", "value": 0.5}     -> <arm>_gripper.set_position
  {"cmd": "estop"} / {"cmd": "recover_estop"}         -> robot_control emergency stop / recovery
Never decides anything itself: every target it receives was checked by motion/safety.py.
"""
import json
import sys

from x2robot import connect
from x2robot.sdk import (GripperPosition, JointPositions, ManipulatorControlMode, ManipulatorControlModeParam,
                         RobotModeParam, RobotWorkMode)

WORK_MODES = {"idle": RobotWorkMode.IDLE, "sdk": RobotWorkMode.SDK}

MODES = {"end_pose": ManipulatorControlMode.MANIPULATOR_END_POSE,
         "joint_positions": ManipulatorControlMode.MANIPULATOR_JOINT_POSITIONS,
         "gravity_compensation": ManipulatorControlMode.MANIPULATOR_GRAVITY_COMPENSATION}
ARMS = ("left", "right")


def result(res) -> dict:
    return {"ok": bool(getattr(res, "is_success", False)), "error": getattr(res, "error_message", "") or "",
            "code": getattr(res, "error_code", None)}


def handle(robot, req: dict) -> dict:
    cmd = req.get("cmd")
    if cmd == "status":
        data = robot.get_robot_status().data
        ex, safety = data.get("execution", {}), data.get("safety", {})
        try:
            mode = robot.robot_control.get_manipulator_control_mode().mode
            mode = next((k for k, v in MODES.items() if int(v) == int(mode)), str(mode))
        except Exception:  # noqa: BLE001 - "Control mode is not set"
            mode = None
        return {"ok": True, "operation_mode": ex.get("operation_mode"), "estop": safety.get("emergency_stop_active"),
                "arm_state": {"left": ex.get("arm_state_left"), "right": ex.get("arm_state_right")}, "control_mode": mode}
    arm = req.get("arm")
    if cmd in ("state", "joints", "gripper") and arm not in ARMS:
        return {"ok": False, "error": f"unknown arm {arm!r}"}
    if cmd == "state":
        js = getattr(robot, f"{arm}_arm").get_joint_states()
        g = getattr(robot, f"{arm}_gripper").get_position()
        return {"ok": True, "names": list(js.name), "q": [float(v) for v in js.position], "gripper": float(g.position)}
    if cmd == "set_mode":
        if req.get("mode") not in MODES:
            return {"ok": False, "error": f"unknown mode {req.get('mode')!r}"}
        return result(robot.robot_control.set_manipulator_control_mode(ManipulatorControlModeParam(mode=MODES[req["mode"]])))
    if cmd == "joints":
        q = [float(v) for v in req.get("q", [])]
        if len(q) != 6:
            return {"ok": False, "error": "need 6 joint positions"}
        return result(getattr(robot, f"{arm}_arm").set_joint_positions(JointPositions(positions=q)))
    if cmd == "gripper":
        return result(getattr(robot, f"{arm}_gripper").set_position(GripperPosition(position=float(req["value"]))))
    if cmd == "work_mode":
        if req.get("mode") not in WORK_MODES:
            return {"ok": False, "error": f"unknown work mode {req.get('mode')!r}"}
        return result(robot.system.set_work_mode(RobotModeParam(mode=WORK_MODES[req["mode"]])))
    if cmd == "estop":
        return result(robot.robot_control.emergency_stop())
    if cmd == "recover_estop":
        return result(robot.robot_control.recover_emergency_stop())
    return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def main() -> int:
    address = sys.argv[1] if len(sys.argv) > 1 else "x2://192.168.77.2:50051"
    robot = connect(address)
    print(json.dumps({"ok": True, "ready": True, "model": str(getattr(robot, "robot_model", ""))}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = handle(robot, json.loads(line))
        except Exception as e:  # noqa: BLE001 - report, never crash silently
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
        print(json.dumps(reply), flush=True)
    robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
