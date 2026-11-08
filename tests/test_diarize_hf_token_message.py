"""Public-repo hygiene: diarize.py must not print host paths or token-extraction recipes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_diarize_missing_hf_token_stderr_has_no_host_paths():
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "diarize.py"
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    result = subprocess.run(
        [sys.executable, str(script), "/tmp/audio.wav", "/tmp/out.json"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "/root/" not in result.stderr
    assert ".env" not in result.stderr
    assert "grep" not in result.stderr.lower()
    assert "HF_TOKEN" in result.stderr
