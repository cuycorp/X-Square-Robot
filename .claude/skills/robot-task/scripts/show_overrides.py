#!/usr/bin/env python3
"""Print the robot WebUI's model overrides with secrets redacted.

The file lives on the robot, so pipe it in over SSH (the user types the SSH password):
  ssh xr@192.168.77.2 cat /home/xr/diversity/local/state/model_overrides.json \
    | python3 .claude/skills/robot-task/scripts/show_overrides.py
A local copy can be passed as an argument instead.
"""
import json
import sys
from pathlib import Path

SECRET = ("key", "token", "auth", "secret", "header", "password")


def redact(o):
    if isinstance(o, dict):
        return {k: ("<set>" if any(s in k.lower() for s in SECRET) and v else redact(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [redact(x) for x in o]
    return o


if __name__ == "__main__":
    raw = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()
    if not raw.strip():
        print("empty input (no overrides yet, or the ssh/cat failed)")
        sys.exit(1)
    print(json.dumps(redact(json.loads(raw)), indent=2, ensure_ascii=False))
