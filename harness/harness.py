#!/usr/bin/env python3
"""Sync Desktop client -> vision agent -> WALLX WebSocket relay example.

Supports both Anthropic Claude Messages API (ANTHROPIC_API_KEY) and any
OpenAI-compatible Vision Chat Completions endpoint (AGENT_CHAT_URL).
WebUI owns the robot session and stopping.
"""

import argparse
import asyncio
import base64
from collections import deque
import json
import os
from pathlib import Path
import time
import urllib.request

import msgpack
import msgpack_numpy
import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from claude_policy import DEFAULT_CLAUDE_MODEL, extract_json_object, load_anthropic_api_key


WALLX_URL = os.environ.get("WALLX_ENDPOINT", "wss://maas-api.x2robot.com/v1/vla/wallx")
MAX_FRAME_BYTES = 32 * 1024 * 1024

AGENT_SYSTEM_PROMPT = (
    "You choose the next short manipulation instruction for a robot VLA. "
    "Use the goal, current images, state, and instruction history. "
    "Ground the instruction in visible objects and avoid repeating completed steps. "
    "An operator controls stopping in WebUI. Do not claim that an instruction stops "
    "the hardware. Return only JSON with non-empty string fields instruction and reason. "
    "If you cannot choose a confident next step or the goal is already complete, "
    "return an empty instruction; the relay will stop requesting new actions and "
    "ask the operator to end the session."
)


def unpack(frame):
    if not isinstance(frame, bytes):
        raise ValueError("WALLX protocol requires a binary MessagePack frame")
    return msgpack.unpackb(frame, raw=False, object_hook=msgpack_numpy.decode)


def pack(value):
    return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)


def instruction_text(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("Expected a non-empty string instruction")


def validate_request(request):
    if not isinstance(request, dict) or request.get("normalization") != "provided":
        raise ValueError("Expected WALLX observation with normalization=provided")
    if request.get("embodiment") not in ("quanta_x1_pro", "quanta_x1_pro/gripper-g", "quanta_x1_pro/gripper-h"):
        raise ValueError("Unsupported embodiment")
    if any(k in request for k in ("use_rtc", "prev_action_chunk", "inference_delay")):
        raise ValueError("This harness uses synchronous requests without RTC")
    instruction_text(request.get("instruction"))
    obs = request["observation"]
    for name in ("front", "left_wrist", "right_wrist"):
        if not isinstance(obs["images"][name], str) or not obs["images"][name]:
            raise ValueError("Three JPEG Base64 images required")
    for arm in ("left_ee", "right_ee"):
        state = obs["state"][arm]
        for field in ("position", "rotation_rpy"):
            a = np.asarray(state[field], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError("Finite 3D position/RPY required")
        g = np.asarray(state["gripper"], dtype=float)
        if g.shape != () or not np.isfinite(g):
            raise ValueError("Finite scalar gripper required")


def json_state(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported state value: {type(value).__name__}")


def _b64_payload(image):
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    if not isinstance(image, str) or not image:
        raise ValueError("Missing JPEG observation")
    if "," in image and image.lstrip().startswith("data:image/"):
        return image.split(",", 1)[1]
    return image


class VisionAgent:
    """Sub-goal planner supporting Anthropic Claude and OpenAI-compatible endpoints."""

    def __init__(self, url="", model="", key="", timeout=15, provider="auto"):
        self.url = url
        self.model = model
        self.key = key
        self.timeout = timeout
        self.provider = provider
        self.reason = ""

    def _is_anthropic(self):
        if self.provider == "anthropic":
            return True
        if self.provider == "openai":
            return False
        if self.url and "anthropic.com" not in self.url:
            return False
        return bool(self.key or load_anthropic_api_key())

    def decide(self, observation, goal, previous, history):
        context = {
            "goal": goal,
            "previous_instruction": previous,
            "recent_instructions": list(history),
            "state": observation["observation"]["state"],
        }
        views = observation["observation"]["images"]

        if self._is_anthropic():
            api_key = self.key or load_anthropic_api_key()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic VisionAgent")
            endpoint = (self.url or "https://api.anthropic.com/v1/messages").rstrip("/")
            if endpoint.endswith("api.anthropic.com"):
                endpoint += "/v1/messages"
            content = [{"type": "text", "text": json.dumps(context, default=json_state)}]
            for name in ("front", "left_wrist", "right_wrist"):
                b64 = _b64_payload(views.get(name))
                content.extend([
                    {"type": "text", "text": name},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                ])
            body = {
                "model": self.model or DEFAULT_CLAUDE_MODEL,
                "max_tokens": 256,
                "system": AGENT_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            request = urllib.request.Request(endpoint, json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            text_parts = [blk.get("text", "") for blk in result.get("content", []) if blk.get("type") == "text"]
            decision = extract_json_object("\n".join(text_parts))
        else:
            content = [{"type": "text", "text": json.dumps(context, default=json_state)}]
            for name in ("front", "left_wrist", "right_wrist"):
                image = views.get(name)
                if isinstance(image, bytes):
                    image = base64.b64encode(image).decode("ascii")
                if not isinstance(image, str) or not image:
                    raise ValueError(f"Missing JPEG observation: {name}")
                content.extend([
                    {"type": "text", "text": name},
                    {"type": "image_url", "image_url": {"url": image if image.startswith("data:image/") else "data:image/jpeg;base64," + image}},
                ])
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            request = urllib.request.Request(self.url, json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
            decision = json.loads(result["choices"][0]["message"]["content"])

        if not isinstance(decision, dict):
            raise ValueError("Agent must return a JSON object")
        instruction = decision.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Agent requested operator intervention; stop the session in WebUI")
        if len(instruction) > 2000:
            raise ValueError("Agent instruction exceeds 2000 characters")
        self.reason = str(decision.get("reason", ""))
        return instruction.strip()


class Harness:
    def __init__(self, upstream_url, headers=None, agent=None, agent_every=4, timeout=60, prompt_ctrl=None):
        if agent_every < 1 or timeout <= 0:
            raise ValueError("agent_every and timeout must be positive")
        self.upstream_url = upstream_url
        self.headers = headers or {}
        self.agent = agent
        self.agent_every = agent_every
        self.timeout = timeout
        self.prompt_ctrl = prompt_ctrl

    async def handle(self, downstream):
        goal = previous = None
        history = deque(maxlen=8)
        request_index = 0
        try:
            async with connect(
                self.upstream_url,
                additional_headers=self.headers,
                proxy=None,
                ping_interval=None,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                max_size=MAX_FRAME_BYTES,
            ) as upstream:
                metadata = await asyncio.wait_for(upstream.recv(), self.timeout)
                if not isinstance(unpack(metadata), dict):
                    raise ValueError("Expected WALLX metadata object")
                await downstream.send(metadata)
                async for frame in downstream:
                    started = time.monotonic()
                    observation = unpack(frame)
                    validate_request(observation)
                    incoming_goal = instruction_text(observation.get("instruction"))
                    if self.prompt_ctrl is not None:
                        incoming_goal, _ = self.prompt_ctrl.get_effective_instruction(incoming_goal)
                    if incoming_goal != goal:
                        goal, previous = incoming_goal, incoming_goal
                        history.clear()
                        request_index = 0
                    if self.agent is not None:
                        if request_index % self.agent_every == 0:
                            previous = await asyncio.to_thread(
                                self.agent.decide, observation, goal, previous, history
                            )
                            history.append(previous)
                        observation["instruction"] = previous
                        frame = pack(observation)
                    elif self.prompt_ctrl is not None and previous != observation.get("instruction"):
                        observation["instruction"] = previous
                        frame = pack(observation)
                    await asyncio.wait_for(upstream.send(frame), self.timeout)
                    response = await asyncio.wait_for(upstream.recv(), self.timeout)
                    await asyncio.wait_for(downstream.send(response), self.timeout)
                    print(json.dumps({
                        "event": "chunk", "index": request_index,
                        "instruction": previous,
                        "reason": getattr(self.agent, "reason", ""),
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                    }), flush=True)
                    request_index += 1
        except ConnectionClosed:
            print("Connection closed; end/confirm the session in WebUI.", flush=True)
        except Exception as error:
            print(f"Harness failed: {type(error).__name__}. Stop the session in WebUI.", flush=True)
            await downstream.close(code=1011, reason="Harness failure; stop in WebUI")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--passthrough", action="store_true", help="Relay without calling the vision agent")
    parser.add_argument("--agent-every", type=int, default=4, help="Replan every N model requests")
    args = parser.parse_args()
    headers_file = os.environ.get("WALLX_HEADERS_FILE")
    headers = json.loads(Path(headers_file).read_text()) if headers_file else {}
    if not isinstance(headers, dict):
        parser.error("WALLX_HEADERS_FILE must contain a JSON object")
    agent = None
    if not args.passthrough:
        url = os.environ.get("AGENT_CHAT_URL", "")
        model = os.environ.get("AGENT_MODEL", "")
        key = os.environ.get("AGENT_API_KEY", "") or load_anthropic_api_key()
        if not url and not key:
            parser.error("Set ANTHROPIC_API_KEY or AGENT_CHAT_URL + AGENT_MODEL, or use --passthrough")
        agent = VisionAgent(url, model, key)
    relay = Harness(WALLX_URL, headers, agent=agent, agent_every=args.agent_every)
    async with serve(
        relay.handle, args.host, args.port,
        ping_interval=None, compression=None, max_size=MAX_FRAME_BYTES,
    ):
        print(f"Harness listening at ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Harness exited. Confirm the robot session is stopped in WebUI.")
