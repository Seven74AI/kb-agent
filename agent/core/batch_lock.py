"""Batch-run lock — prevent concurrent batch collector runs (cron + MCP tool).

A stale lock (holder PID dead) is stolen so a crash never wedges the batch
permanently.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.config import Config


def lock_path(config: Config) -> Path:
    return Path(config.kb_agent_dir) / "data" / "batch.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def is_locked(config: Config) -> bool:
    """True if a live batch run currently holds the lock."""
    path = lock_path(config)
    if not path.exists():
        return False
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return True  # unreadable lock — treat as held; acquire() steals it
    return _pid_alive(pid)


def acquire(config: Config) -> bool:
    """Atomically acquire the lock. False if a live run already holds it."""
    path = lock_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            pass
        try:
            pid = int(path.read_text().strip())
        except (ValueError, OSError):
            # Corrupt/unreadable lock — remove and retry create
            try:
                path.unlink(missing_ok=True)
            except OSError:
                return False
            continue
        if _pid_alive(pid):
            return False  # live holder — give up
        # stale — steal it and retry
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False


def release(config: Config) -> None:
    lock_path(config).unlink(missing_ok=True)
