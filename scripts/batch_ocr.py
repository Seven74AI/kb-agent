#!/usr/bin/env python3
"""Batch image reading via Modal — Qwen2.5-VL-7B, one cold-start per batch.

Usage:
    batch_ocr.py <input_dir> [--out-dir DIR]

<input_dir> may contain .png/.jpg/.jpeg/.webp images. Runs ONE Modal batch
(loads the VLM once, reads every image), writing <name>.ocr.json per input.
"""

import os
import subprocess
import sys
from pathlib import Path

# Isolate from the Hermes runtime's py3.14 PYTHONPATH (cross-venv mismatch).
os.environ.pop("PYTHONPATH", None)

_bootstrap_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bootstrap_root))
sys.path.insert(0, "/root/kb-agent")
from agent.notify import sanitize_visible_message  # noqa: E402

MODAL = "/root/modal"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".avif")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _print_scrubbed_child_tail(
    result: subprocess.CompletedProcess[str],
    *,
    tail_chars: int = 1500,
) -> None:
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    raw = stdout or stderr
    if not raw:
        return
    print(sanitize_visible_message(raw[-tail_chars:]))


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("usage: batch_ocr.py <input_dir> [--out-dir DIR]")
        return 1
    in_dir = args[0]
    out_dir = "."
    i = 1
    while i < len(args):
        if args[i] == "--out-dir" and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        else:
            i += 1
    os.makedirs(out_dir, exist_ok=True)

    names = [
        f for f in sorted(os.listdir(in_dir))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if not names:
        print("no images found in", sanitize_visible_message(in_dir))
        return 1
    print(f"batch: {len(names)} image(s) -> one Modal run")

    cmd = [
        os.path.join(MODAL, "venv", "bin", "modal"), "run", "kb_ocr.py",
        "--image-dir", in_dir, "--out-dir", out_dir,
    ]
    r = sh(cmd, cwd=MODAL, timeout=3600)
    _print_scrubbed_child_tail(r)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
