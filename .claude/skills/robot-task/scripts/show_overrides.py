#!/usr/bin/env python3
"""Print WebUI model overrides with secrets redacted."""
import json
from pathlib import Path

PATH = Path("/home/xr/diversity/local/state/model_overrides.json")
SECRET = ("key", "token", "auth", "secret", "header", "password")


def redact(o):
    if isinstance(o, dict):
        return {k: ("<set>" if any(s in k.lower() for s in SECRET) and v else redact(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [redact(x) for x in o]
    return o


if __name__ == "__main__":
    if not PATH.exists():
        print(f"{PATH} not found (no overrides yet)")
    else:
        print(json.dumps(redact(json.loads(PATH.read_text())), indent=2, ensure_ascii=False))
