#!/usr/bin/env python3
"""Ask a Nebius vision model what the robot's cameras see. Read-only: never commands the robot.

Examples:
  .venv/bin/python perception/see.py                         # describe the table (head camera)
  .venv/bin/python perception/see.py "Is there a pen?"       # any question
  .venv/bin/python perception/see.py --camera head cam3 "Where is the pen?"
  .venv/bin/python perception/see.py --json                  # structured object list (prompts.scene_json_prompt)
  .venv/bin/python perception/see.py --list-models           # vision-capable model ids
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import camera  # noqa: E402
import nebius_vision  # noqa: E402
import prompts  # noqa: E402

VISION_HINTS = ("vl", "vision", "gemma-3", "llama-4", "pixtral", "llava", "kimi", "qwen3-vl", "omni")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="question about the scene (default: describe the table)")
    ap.add_argument("--camera", nargs="+", default=["head"], choices=sorted(camera.CAMERAS),
                    help="cameras to send (default: head; wrist cameras may capture people)")
    ap.add_argument("--json", action="store_true", help="ask for the structured scene list")
    ap.add_argument("--model", default=nebius_vision.DEFAULT_MODEL)
    ap.add_argument("--think", action="store_true", help="let the model reason first (slower, ~4-5 s; for hard questions)")
    ap.add_argument("--save", metavar="DIR", help="also save the frames sent, as JPEGs")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    if a.list_models:
        ids = nebius_vision.list_models()
        likely = [m for m in ids if any(h in m.lower() for h in VISION_HINTS)]
        print("likely vision models:\n  " + "\n  ".join(likely or ["(none matched; see full list)"]))
        print(f"\nall {len(ids)} models:\n  " + "\n  ".join(ids))
        return 0

    prompt = prompts.scene_json_prompt() if a.json else (a.question or prompts.DESCRIBE)

    t0 = time.monotonic()
    frames = camera.grab_frames(a.camera)
    t1 = time.monotonic()
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        for name, jpeg in frames.items():
            (Path(a.save) / f"{name}.jpg").write_bytes(jpeg)
    labelled = {camera.CAMERAS[n][1]: jpeg for n, jpeg in frames.items()}
    result = nebius_vision.ask(labelled, prompt, model=a.model, system=prompts.SYSTEM, think=a.think)
    t2 = time.monotonic()

    print(result["text"])
    u = result["usage"]
    print(f"\n[{result['model']} | cameras: {', '.join(a.camera)} | grab {1000*(t1-t0):.0f} ms, "
          f"model {1000*(t2-t1):.0f} ms | tokens in/out: {u.get('prompt_tokens', '?')}/{u.get('completion_tokens', '?')}]",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
