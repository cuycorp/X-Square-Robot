#!/usr/bin/env python3
"""Prompt for the WALLX key (hidden) and write ~/.config/wallx/headers.json (mode 600).

Run interactively, on its own line. The key is never printed.
"""
import getpass
import json
import os
import sys
from pathlib import Path

path = Path.home() / ".config" / "wallx" / "headers.json"

key = getpass.getpass("WALLX key (hidden, paste then Enter): ").strip()
# Drop accidental bracketed-paste markers and a pasted "Bearer " prefix.
for junk in ("\x1b[200~", "\x1b[201~"):
    key = key.replace(junk, "")
if key.lower().startswith("bearer "):
    key = key[7:].strip()

if len(key) < 16 or any(c.isspace() or c in '"\\' or ord(c) < 32 for c in key):
    print(f"Rejected: got {len(key)} chars that don't look like an API key. Nothing written.")
    sys.exit(1)

path.parent.mkdir(parents=True, exist_ok=True)
os.chmod(path.parent, 0o700)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump({"Authorization": f"Bearer {key}"}, f)
os.chmod(path, 0o600)
print(f"Wrote {path} (token length {len(key)}, mode 600).")
