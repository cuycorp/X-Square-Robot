#!/usr/bin/env python3
"""Single-command launcher for prompt-driven autonomous dual-arm control.

Usage on the robot PC:
    python3 launch.py
    python3 launch.py --prompt "Pick up the spoon and place it in the bowl"
    python3 launch.py --sim --mode offline   # dry-run in CPU tabletop simulator
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
HARNESS_DIR = REPO_ROOT / "harness"
VENV_DIR = REPO_ROOT / ".venv"
TASK_FILE_REPO = REPO_ROOT / "tasks" / "local_claude_prompt_benchmark.json"
DIVERSITY_DIR = Path("/home/xr/diversity")
OVERRIDES_FILE = DIVERSITY_DIR / "local" / "state" / "model_overrides.json"


def _has_runtime_deps() -> bool:
    for mod in ("websockets", "msgpack", "msgpack_numpy", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def ensure_venv_and_reexec() -> None:
    if _has_runtime_deps():
        return
    venv_py = VENV_DIR / "bin" / "python"
    if Path(sys.executable).resolve() == venv_py.resolve():
        raise SystemExit("Missing dependencies inside .venv. Run: python3 -m pip install -r harness/requirements.txt")
    if not venv_py.exists():
        print("[setup] Creating .venv without ensurepip (robot-PC compatible)...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV_DIR)], check=True)
    print("[setup] Installing harness requirements into .venv...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "--python", str(venv_py), "install", "-r", str(HARNESS_DIR / "requirements.txt")],
        check=True,
    )
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


def detect_lan_ip() -> str:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()
        for ip in out:
            if ":" not in ip and not ip.startswith("127.") and not ip.startswith("172."):
                return ip
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return "127.0.0.1"


def build_task_dict(ws_url: str, instruction: str, gripper: str = "g") -> dict[str, Any]:
    embodiment = "quanta_x1_pro/gripper-h" if gripper.lower() == "h" else "quanta_x1_pro/gripper-g"
    return {
        "mode": "benchmark",
        "task_name": "claude_autonomous_prompt",
        "model": {
            "config_name": "claude_autonomous_endpose",
            "deploy": "lan",
            "model_address": ws_url,
            "robot_type": "DESKTOP",
            "mode_type": "EndPose",
            "infer_type": "sync",
            "infer_params": {
                "action_horizon": 32,
                "control_dt": 0.005,
                "interpolate_multiplier": 8,
                "async_params": {
                    "api_schema": "public_v2",
                    "embodiment": embodiment,
                    "normalization": "provided",
                    "control_context": "masked",
                },
            },
        },
        "data": [
            {
                "case_id": 1,
                "instruction": instruction,
                "image_path": "",
                "env_description": {
                    "mode": "Prompt-driven autonomous dual-arm EndPose control via Claude Vision",
                },
                "scores_dict": {
                    "Task completed safely": 10,
                },
                "timeout": 180,
            }
        ],
    }


def deploy_task_json(ws_url: str, instruction: str, gripper: str, deploy: bool = True) -> None:
    TASK_FILE_REPO.parent.mkdir(parents=True, exist_ok=True)
    payload = build_task_dict(ws_url, instruction, gripper=gripper)
    TASK_FILE_REPO.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if deploy and DIVERSITY_DIR.exists() and os.access(DIVERSITY_DIR, os.W_OK):
        dst = DIVERSITY_DIR / TASK_FILE_REPO.name
        shutil.copy2(TASK_FILE_REPO, dst)
        print(f"[task] Deployed {dst} (model_address={ws_url})", flush=True)
        if OVERRIDES_FILE.exists():
            try:
                ov = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
                entry = ov.get(TASK_FILE_REPO.stem)
                if isinstance(entry, dict) and entry.get("model_address") and entry.get("model_address") != ws_url:
                    print(
                        f"[warn] WebUI override in {OVERRIDES_FILE} sets model_address={entry.get('model_address')}.\n"
                        "       If the robot does not connect to the harness, click 'Restore Default' or update Edit Model Configuration in WebUI.",
                        flush=True,
                    )
            except Exception:
                pass


def prompt_for_api_key(load_fn: Any, save_fn: Any) -> str:
    existing = load_fn()
    if existing:
        return existing
    if not sys.stdin.isatty():
        raise SystemExit("ANTHROPIC_API_KEY is not set and stdin is not interactive.")
    key = getpass.getpass("Enter Claude (Anthropic) API key (hidden, then Enter): ").strip()
    for junk in ("\x1b[200~", "\x1b[201~"):
        key = key.replace(junk, "")
    if len(key) < 16 or any(c.isspace() or ord(c) < 32 for c in key):
        raise SystemExit("Invalid API key format. Nothing saved.")
    ans = input("Save key securely to ~/.config/anthropic/api_key (mode 0600) for future launches? [Y/n]: ").strip().lower()
    if ans in ("", "y", "yes"):
        saved = save_fn(key)
        print(f"[auth] Saved Claude key to {saved} (mode 0600).", flush=True)
    os.environ["ANTHROPIC_API_KEY"] = key
    return key


def prompt_loop(ctrl: Any, stop_evt: threading.Event, lan_ip: str) -> None:
    print(
        "\n--- Interactive Prompt Control ---\n"
        "Type a task prompt and press Enter at any time (e.g. 'Pick up the spoon and put it in the bowl').\n"
        "Commands: /hold (hold pose), /webui (use WebUI case prompt), /status, /quit\n",
        flush=True,
    )
    while not stop_evt.is_set():
        try:
            line = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            stop_evt.set()
            break
        cmd = line.strip()
        if not cmd:
            continue
        if cmd in ("/quit", "/exit"):
            print(
                f"Shutting down launcher. Confirm the robot session is stopped in WebUI (http://{lan_ip}/).",
                flush=True,
            )
            stop_evt.set()
            break
        if cmd == "/status":
            snap = ctrl.snapshot()
            print(
                json.dumps(
                    {
                        "connected": snap["connected"],
                        "status": snap["status"],
                        "stage": snap["stage"],
                        "chunk_index": snap["chunk_index"],
                        "effective_instruction": snap["effective_instruction"],
                        "reason": snap["reason"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            continue
        if cmd == "/hold":
            ctrl.set_prompt("Hold current pose safely.")
            print("[prompt] Holding current pose safely.", flush=True)
            continue
        if cmd == "/webui":
            ctrl.set_prompt(None)
            print("[prompt] Reverted to WebUI task Case instruction.", flush=True)
            continue
        ver = ctrl.set_prompt(cmd)
        print(f"[prompt v{ver}] Active goal set to: {cmd}", flush=True)


def run_sim_mode(args: argparse.Namespace, initial_prompt: str, api_key: str) -> int:
    from claude_policy import ClaudeEndPosePlanner, HeuristicOfflinePlanner
    from server import DualArmPolicyEngine, PromptController, serve_connection
    from sim_cpu_env import run_episode
    from trajectory import WorkspaceLimits
    import websockets

    ctrl = PromptController(initial_prompt=initial_prompt)
    planner = (
        HeuristicOfflinePlanner()
        if args.mode == "offline"
        else ClaudeEndPosePlanner(api_key=api_key, model=args.model or None, plan_every=args.plan_every)
    )
    engine = DualArmPolicyEngine(
        planner=planner,
        limits=WorkspaceLimits(max_step_pos_m=args.max_step_m, max_step_rot_rad=args.max_step_rad),
        log_path=REPO_ROOT / "logs" / "sim_chunks.jsonl",
        prompt_ctrl=ctrl,
    )

    port_holder: dict[str, int] = {}
    ready = threading.Event()
    stop_holder: dict[str, Any] = {}

    def _srv() -> None:
        async def _run() -> None:
            stop_evt = asyncio.Event()
            stop_holder["loop"] = asyncio.get_running_loop()
            stop_holder["evt"] = stop_evt
            async with websockets.serve(lambda c: serve_connection(c, engine), "127.0.0.1", 0, max_size=64 * 1024 * 1024) as srv:
                port_holder["port"] = srv.sockets[0].getsockname()[1]
                ready.set()
                await stop_evt.wait()

        asyncio.run(_run())

    t = threading.Thread(target=_srv, daemon=True)
    t.start()
    ready.wait(timeout=5)
    try:
        run_episode(
            ws_url=f"ws://127.0.0.1:{port_holder['port']}",
            instruction=initial_prompt,
            cycles=args.sim_cycles,
            use_rtc=False,
        )
    finally:
        stop_holder["loop"].call_soon_threadsafe(stop_holder["evt"].set)
        t.join(timeout=3)
    return 0


def main() -> int:
    ensure_venv_and_reexec()
    sys.path.insert(0, str(HARNESS_DIR))

    from claude_policy import (
        DEFAULT_CLAUDE_MODEL,
        ClaudeEndPosePlanner,
        HeuristicOfflinePlanner,
        load_anthropic_api_key,
        save_anthropic_api_key,
    )
    from server import DualArmPolicyEngine, PromptController, serve_connection, start_console_http_server
    from trajectory import WorkspaceLimits
    from websockets.asyncio.server import serve

    parser = argparse.ArgumentParser(description="Launch prompt-driven autonomous Claude control for Quanta X1 Pro")
    parser.add_argument("--prompt", default="", help="Initial natural-language task prompt")
    parser.add_argument("--mode", choices=("claude", "offline"), default="claude")
    parser.add_argument("--model", default="", help=f"Claude model ID (default: {DEFAULT_CLAUDE_MODEL})")
    parser.add_argument("--host", default="", help="Bind/advertised LAN IP (auto-detected if omitted)")
    parser.add_argument("--port", type=int, default=8001, help="WebSocket policy server port")
    parser.add_argument("--console-port", type=int, default=8002, help="Browser Prompt & Camera Console port (0 to disable)")
    parser.add_argument("--gripper", choices=("g", "h"), default="g", help="Physical gripper type (g or h)")
    parser.add_argument("--plan-every", type=int, default=4, help="Maximum observation chunks between Claude plans")
    parser.add_argument("--max-step-m", type=float, default=0.08, help="Max EndPose translation step per chunk (meters)")
    parser.add_argument("--max-step-rad", type=float, default=0.45, help="Max EndPose rotation step per chunk (radians)")
    parser.add_argument("--no-deploy", action="store_true", help="Do not copy task JSON to /home/xr/diversity")
    parser.add_argument("--sim", action="store_true", help="Run against the built-in CPU tabletop simulator instead of waiting for hardware")
    parser.add_argument("--sim-cycles", type=int, default=9)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(REPO_ROOT / "logs" / "launcher.log") if (REPO_ROOT / "logs").exists() else logging.NullHandler()],
    )
    (REPO_ROOT / "logs").mkdir(exist_ok=True)

    api_key = ""
    if args.mode == "claude":
        api_key = prompt_for_api_key(load_anthropic_api_key, save_anthropic_api_key)

    initial_prompt = args.prompt.strip()
    if not initial_prompt and sys.stdin.isatty():
        entered = input("Enter initial task prompt (or press Enter to use WebUI task instruction): ").strip()
        initial_prompt = entered

    if args.sim:
        sim_prompt = initial_prompt or "Pick up the red object and place it into the blue bowl."
        print(f"[sim] Running CPU simulator with mode={args.mode}, prompt={sim_prompt!r}", flush=True)
        return run_sim_mode(args, sim_prompt, api_key)

    lan_ip = args.host or detect_lan_ip()
    ws_url = f"ws://{lan_ip}:{args.port}"
    task_instruction = initial_prompt or "Execute the task instruction from the Claude Prompt Console."
    deploy_task_json(ws_url, task_instruction, gripper=args.gripper, deploy=not args.no_deploy)

    ctrl = PromptController(initial_prompt=initial_prompt or None)
    if args.console_port > 0:
        start_console_http_server("0.0.0.0", args.console_port, ctrl)

    planner: ClaudeEndPosePlanner | HeuristicOfflinePlanner
    if args.mode == "offline":
        planner = HeuristicOfflinePlanner()
    else:
        planner = ClaudeEndPosePlanner(api_key=api_key, model=args.model or None, plan_every=args.plan_every)

    engine = DualArmPolicyEngine(
        planner=planner,
        limits=WorkspaceLimits(max_step_pos_m=args.max_step_m, max_step_rot_rad=args.max_step_rad),
        log_path=REPO_ROOT / "logs" / "chunks.jsonl",
        prompt_ctrl=ctrl,
    )

    print(
        f"\n============================================================\n"
        f"Quanta X1 Pro Claude Policy Server READY ({args.mode})\n"
        f"  WebSocket Policy Endpoint : {ws_url}\n"
        f"  Live Prompt & Cam Console : http://{lan_ip}:{args.console_port}/\n"
        f"  Robot WebUI               : http://{lan_ip}/\n"
        f"\nNext steps in Robot WebUI:\n"
        f"  1. Open Task Management -> refresh -> select 'claude_autonomous_prompt'\n"
        f"  2. Confirm model_address is '{ws_url}' (EndPose, sync, public_v2)\n"
        f"  3. Click 'Start Evaluation Inference Mode'\n"
        f"  4. To stop & home hardware: click 'End Evaluation Inference Mode' in WebUI first!\n"
        f"============================================================\n",
        flush=True,
    )

    stop_evt = threading.Event()

    async def _serve_ws() -> None:
        async with serve(
            lambda conn: serve_connection(conn, engine),
            "0.0.0.0",
            args.port,
            ping_interval=None,
            compression=None,
            max_size=64 * 1024 * 1024,
        ):
            while not stop_evt.is_set():
                await asyncio.sleep(0.25)

    if sys.stdin.isatty():
        t_prompt = threading.Thread(target=prompt_loop, args=(ctrl, stop_evt, lan_ip), daemon=True)
        t_prompt.start()

    try:
        asyncio.run(_serve_ws())
    except KeyboardInterrupt:
        stop_evt.set()
        print(f"\nLauncher stopped. Confirm the robot session is stopped in WebUI (http://{lan_ip}/).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
