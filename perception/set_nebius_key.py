#!/usr/bin/env python3
"""Prompt for the Nebius Token Factory API key (hidden) and save it to ~/.config/nebius/api_key (mode 600).

Run it yourself in a terminal: python3 perception/set_nebius_key.py
The key is never printed.
"""
import getpass
import os
import sys
from pathlib import Path

path = Path.home() / ".config" / "nebius" / "api_key"
key = getpass.getpass("Nebius Token Factory API key (hidden, paste then Enter): ").strip()
for junk in ("\x1b[200~", "\x1b[201~"):
    key = key.replace(junk, "")
if key.lower().startswith("bearer "):
    key = key[7:].strip()
if len(key) < 16 or any(c.isspace() or ord(c) < 32 for c in key):
    print(f"Rejected: {len(key)} characters doesn't look like an API key. Nothing written.")
    sys.exit(1)

path.parent.mkdir(parents=True, exist_ok=True)
os.chmod(path.parent, 0o700)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(key + "\n")
os.chmod(path, 0o600)
print(f"Saved to {path} ({len(key)} characters, mode 600).")
