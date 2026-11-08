"""Invariant: listed ingest modules have no unused imports or dead locals (pyflakes)."""

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules touched for LOW import/local hygiene — keep in sync with pyflakes gate.
_HYGIENE_MODULES = (
    "agent/streaming.py",
    "agent/detection.py",
    "agent/pipelines/threads.py",
    "agent/pipelines/web.py",
    "agent/core/note.py",
    "agent/pipelines/youtube.py",
)


def test_low_hygiene_modules_pass_pyflakes():
    """Regression: no F401 unused imports or assigned-but-unused locals on gate set."""
    paths = [str(_REPO_ROOT / rel) for rel in _HYGIENE_MODULES]
    proc = subprocess.run(
        ["pyflakes", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_write_vault_note_not_reexported_from_core_note():
    """write_vault_note readers use agent.redact after note.py dropped dead re-export."""
    note_src = (_REPO_ROOT / "agent/core/note.py").read_text(encoding="utf-8")
    assert "write_vault_note" not in note_src
    for rel in (
        "agent/pipelines/threads.py",
        "agent/pipelines/web.py",
        "agent/pipelines/youtube.py",
    ):
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "write_vault_note" in text
        assert "from agent.redact import" in text
