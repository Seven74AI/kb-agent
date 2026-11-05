"""Tests for batch-run lock acquire/release."""

import os
from pathlib import Path

from agent.config import Config
from agent.core.batch_lock import acquire, is_locked, lock_path, release


def test_acquire_steals_corrupt_lock(tmp_path):
    config = Config(kb_agent_dir=tmp_path)
    path = lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-pid", encoding="utf-8")

    assert is_locked(config) is True
    assert acquire(config) is True
    assert path.exists()
    assert int(path.read_text().strip()) == os.getpid()
    release(config)


def test_acquire_yields_when_live_holder(tmp_path, monkeypatch):
    config = Config(kb_agent_dir=tmp_path)
    path = lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")

    assert acquire(config) is False
