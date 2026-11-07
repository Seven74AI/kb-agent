"""Tests for Telegram notifications and shared visible-error sanitization."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.notify import notify_task, sanitize_visible_message


def test_sanitize_visible_message_strips_traceback_frames():
    raw = (
        "yt-dlp stderr: cookie at /root/kb-agent/cookies/yt.txt\n"
        "Traceback (most recent call last):\n"
        '  File "/root/kb-agent/scripts/foo.py", line 9, in run\n'
        "    raise ValueError('nope')\n"
        "ValueError: nope"
    )
    out = sanitize_visible_message(raw)
    assert "Traceback" not in out
    assert 'File "/root' not in out
    assert "/root/kb-agent" not in out
    assert "[path]" in out
    assert "yt-dlp stderr" in out
    assert "ValueError: nope" in out


def test_sanitize_visible_message_scrubs_paths_without_traceback():
    raw = "megadl failed: cookiefile /home/ubuntu/secret/cookies.txt"
    out = sanitize_visible_message(raw)
    assert "/home/ubuntu" not in out
    assert "[path]" in out
    assert "megadl failed" in out


def test_sanitize_visible_message_truncates_long_subprocess_stderr_line():
    long_stderr = "yt-dlp: " + ("x" * 400)
    out = sanitize_visible_message(long_stderr)
    assert len(out) <= 203
    assert out.endswith("...")
    assert out.startswith("yt-dlp:")


def test_sanitize_visible_message_caps_length():
    out = sanitize_visible_message("a" * 5000)
    assert len(out) == 2000


def test_notify_task_sanitizes_reason():
    cfg = Config(
        kb_agent_dir=Path("/tmp/k"),
        vault_path=Path("/tmp/v"),
        temp_dir=Path("/tmp/t"),
        cookie_dir=Path("/tmp/c"),
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    reason = (
        "failed\nTraceback (most recent call last):\n"
        '  File "/secret/path.py", line 1\n'
        "RuntimeError: x"
    )
    with patch("agent.notify.send_message") as send:
        notify_task(cfg, "slug-x", "web", "failed", reason)
        text = send.call_args[0][1]
        assert "Traceback" not in text
        assert "RuntimeError: x" in text
