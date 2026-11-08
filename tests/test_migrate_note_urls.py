"""Regression: vault URL migration must persist through the vault scrub choke."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "migrate_note_urls.py"


def _load_migrate_module():
    spec = importlib.util.spec_from_file_location("migrate_note_urls", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(_REPO))
    spec.loader.exec_module(mod)
    return mod


def test_migrate_note_urls_apply_scrubs_query_tokens(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    kb = vault / "Knowledge base" / "Article"
    kb.mkdir(parents=True)
    tokenized = "https://cdn.example.com/v.mp4?token=SECRET"
    prefix = f"http://127.0.0.1:9000/knowledge-base/videos/x.mp4\nsource: {tokenized}\n"
    note = kb / "note.md"
    note.write_text(prefix, encoding="utf-8")

    monkeypatch.setenv("KB_VAULT_PATH", str(vault))
    mod = _load_migrate_module()
    monkeypatch.setattr(mod, "APPLY", True, raising=False)

    assert mod.main() == 0
    written = note.read_text(encoding="utf-8")
    assert "token=SECRET" not in written
    assert "http://127.0.0.1:9000/knowledge-base/" not in written
    assert "videos/x.mp4" in written


def test_migrate_note_urls_cli_apply_scrubs(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    kb = vault / "Knowledge base"
    kb.mkdir(parents=True)
    note = kb / "clip.md"
    note.write_text(
        "http://127.0.0.1:9000/knowledge-base/videos/clip.mp4\n"
        "ref: https://cdn.example.com/v.mp4?token=SECRET\n",
        encoding="utf-8",
    )
    env = {**os.environ, "KB_VAULT_PATH": str(vault)}
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--apply"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    written = note.read_text(encoding="utf-8")
    assert "token=SECRET" not in written
    assert "videos/clip.mp4" in written
