"""Ask a Nebius Token Factory vision model about camera images (OpenAI-compatible Chat Completions)."""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
# Chosen 2026-11-21 on a head-camera frame: Qwen3.8-27B with thinking off answered in ~1.2 s and was the
# most grounded. Runner-up: nvidia/Nemotron-3.5-Super-VL (~1.8 s). These are reasoning models: with
# thinking on, they spend the output budget on hidden reasoning first (~4-5 s).
DEFAULT_MODEL = os.environ.get("NEBIUS_VISION_MODEL", "Qwen/Qwen3.8-27B")
KEY_FILE = Path(os.environ.get("NEBIUS_KEY_FILE", "~/.config/nebius/api_key")).expanduser()


def load_key() -> str:
    """Key from $NEBIUS_API_KEY, else the private key file. Never printed."""
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(f"no Nebius key: run perception/set_nebius_key.py or set NEBIUS_API_KEY")
    return key


def build_messages(images: dict, prompt: str, system: str = "") -> list:
    """images: {label: jpeg_bytes}. Each image is preceded by its label so the model can tell them apart."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not images:
        raise ValueError("at least one image is required")
    content = []
    for label, jpeg in images.items():
        content.append({"type": "text", "text": f"Image from: {label}"})
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": content}]


def _request(path: str, key: str, payload=None, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Keep only the API's error message; never echo request headers.
        detail = e.read()[:500].decode("utf-8", "replace")
        raise RuntimeError(f"Nebius API HTTP {e.code}: {detail}") from None


def build_payload(images: dict, prompt: str, model: str, system: str = "",
                  think: bool = False, max_tokens: int | None = None) -> dict:
    payload = {"model": model, "messages": build_messages(images, prompt, system),
               "max_tokens": max_tokens or (2000 if think else 800), "temperature": 0.2}
    if not think:
        # Honoured by Qwen and Nemotron on Token Factory; Kimi ignores it.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def ask(images: dict, prompt: str, model: str = DEFAULT_MODEL, system: str = "",
        think: bool = False, max_tokens: int | None = None, key: str | None = None) -> dict:
    """Return {"text", "model", "usage"} for one question about the images."""
    payload = build_payload(images, prompt, model, system, think, max_tokens)
    data = _request("/chat/completions", key or load_key(), payload)
    try:
        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError(f"unexpected response shape: {str(data)[:300]}") from None
    if not text:
        raise RuntimeError(f"empty answer (finish_reason={choice.get('finish_reason')!r}): the model likely "
                           f"spent max_tokens={payload['max_tokens']} on reasoning; raise it or turn thinking off")
    return {"text": text, "model": data.get("model", model), "usage": data.get("usage", {})}


def complete_text(prompt: str, model: str = DEFAULT_MODEL, system: str = "", think: bool = False,
                  max_tokens: int | None = None, key: str | None = None) -> dict:
    """Text-only chat call (no images). Returns {"text", "model", "usage"}."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens or (2000 if think else 500),
               "temperature": 0.0}
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = _request("/chat/completions", key or load_key(), payload)
    try:
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError(f"unexpected response shape: {str(data)[:300]}") from None
    if not text:
        raise RuntimeError("empty answer from the model")
    return {"text": text, "model": data.get("model", model), "usage": data.get("usage", {})}


def list_models(key: str | None = None) -> list:
    """All model ids available to this account."""
    return sorted(m["id"] for m in _request("/models", key or load_key()).get("data", []))
