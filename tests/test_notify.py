"""Tests for Telegram notifications and shared visible-error sanitization."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.notify import notify_batch, notify_task, sanitize_visible_message


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


def test_sanitize_visible_message_redacts_credential_shapes():
    raw = "upstream failed: api_key=abc123 and Authorization: Bearer xyz.token-99"
    out = sanitize_visible_message(raw)
    assert "abc123" not in out
    assert "xyz.token" not in out
    assert "[credential redacted]" in out
    assert "Bearer [redacted]" in out
    assert "upstream failed" in out


def test_sanitize_visible_message_redacts_cookie_session_pairs():
    raw = (
        "stderr: sessionid=supersecret; csrftoken=abc; PHPSESSID=xyz; "
        "SID=tail; _csrf=2; xsrf=3; auth-token=4"
    )
    out = sanitize_visible_message(raw)
    assert "supersecret" not in out
    assert "sessionid=[REDACTED]" in out
    assert "csrftoken=[REDACTED]" in out
    assert "PHPSESSID=[REDACTED]" in out
    assert "SID=tail" not in out
    assert "SID=[REDACTED]" in out


def test_notify_batch_sanitizes_causes():
    cfg = Config(
        kb_agent_dir=Path("/tmp/k"),
        vault_path=Path("/tmp/v"),
        temp_dir=Path("/tmp/t"),
        cookie_dir=Path("/tmp/c"),
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    causes = [
        "download failed: cookie at /root/kb-agent/cookies/yt.txt",
        "download failed: cookie at /root/kb-agent/cookies/yt.txt",
    ]
    with patch("agent.notify.send_message") as send:
        notify_batch(cfg, failed=1, total=2, dispatched=0, causes=causes)
        text = send.call_args[0][1]
        assert "/root/kb-agent" not in text
        assert "[path]" in text
        assert "cause:" in text


def test_notify_task_redacts_url_query_in_reason():
    cfg = Config(
        kb_agent_dir=Path("/tmp/k"),
        vault_path=Path("/tmp/v"),
        temp_dir=Path("/tmp/t"),
        cookie_dir=Path("/tmp/c"),
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    reason = "download failed: https://cdn.example.com/v/1?token=SECRET&expires=99"
    with patch("agent.notify.send_message") as send:
        notify_task(cfg, "slug-y", "youtube", "failed", reason)
        text = send.call_args[0][1]
        assert "SECRET" not in text
        assert "expires=99" not in text
        assert "?" not in text.split("reason:")[-1]
        assert "download failed" in text


def test_notify_batch_includes_all_unique_causes():
    cfg = Config(
        kb_agent_dir=Path("/tmp/k"),
        vault_path=Path("/tmp/v"),
        temp_dir=Path("/tmp/t"),
        cookie_dir=Path("/tmp/c"),
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    causes = [f"cause-{i}" for i in range(5)]
    with patch("agent.notify.send_message") as send:
        notify_batch(cfg, failed=5, total=5, dispatched=0, causes=causes)
        text = send.call_args[0][1]
        for c in causes:
            assert c in text


def test_notify_batch_redacts_url_query_in_causes():
    cfg = Config(
        kb_agent_dir=Path("/tmp/k"),
        vault_path=Path("/tmp/v"),
        temp_dir=Path("/tmp/t"),
        cookie_dir=Path("/tmp/c"),
        telegram_bot_token="tok",
        telegram_chat_id="123",
    )
    causes = ["https://signed.cdn.test/file?token=LEAKED"]
    with patch("agent.notify.send_message") as send:
        notify_batch(cfg, failed=1, total=1, dispatched=0, causes=causes)
        text = send.call_args[0][1]
        assert "LEAKED" not in text
        assert "token=" not in text


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
