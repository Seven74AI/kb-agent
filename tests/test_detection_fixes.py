"""Tests for detection helpers (dedup, threads cookies)."""

import subprocess
from pathlib import Path
import pytest

from agent.config import Config
from agent.detection import (
    DedupCheckError,
    _resolve_threads_cookie_path,
    dedup_check,
    preflight_health,
)


@pytest.fixture
def core_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault)


def test_dedup_check_uses_grep_fixed_string(core_config, monkeypatch):
    """URLs with regex metacharacters must not produce ambiguous grep patterns."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    url = "https://example.com/watch?v=abc.def"
    dedup_check(url, core_config.vault_path)
    assert captured
    assert "-F" in captured[0]
    assert f"source_url: {url}" in captured[0]


def test_dedup_check_raises_on_grep_failure(core_config, monkeypatch):
    """Grep errors must not be treated as 'not in vault'."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="permission denied")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    with pytest.raises(DedupCheckError, match="exit 2"):
        dedup_check("https://example.com/x", core_config.vault_path)


def test_preflight_youtube_uses_config_cookie_dir(tmp_path, monkeypatch):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    yt = custom_dir / "yt_cookies.txt"
    yt.write_text(
        "LOGIN_INFO=x\nSAPISID=x\n__Secure-3PSID=x\n",
        encoding="utf-8",
    )
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    ok, reason = preflight_health("youtube")
    assert ok is True
    assert reason == ""


def test_preflight_rumble_checks_flaresolverr(tmp_path, monkeypatch):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    (custom_dir / "rumble_cookies.txt").write_text("# Netscape\n", encoding="utf-8")
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout='{"status":"ok"}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is True
    assert reason == ""


def test_preflight_rumble_unhealthy_when_flaresolverr_down(tmp_path, monkeypatch):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    (custom_dir / "rumble_cookies.txt").write_text("# Netscape\n", encoding="utf-8")
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is False
    assert "Flaresolverr" in reason


def test_dedup_check_no_false_positive_from_regex_dot(core_config):
    """Literal URL must not match a note whose host differs only at dot positions."""
    kb = core_config.vault_path / "Knowledge base" / "articles"
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "note.md").write_text(
        "---\nsource_url: https://example.com/item\n---\n",
        encoding="utf-8",
    )
    assert not dedup_check("https://exaXple.com/item", core_config.vault_path)


def test_resolve_threads_cookie_path_prefers_config_cookie_dir(tmp_path, monkeypatch):
    custom_dir = tmp_path / "my-cookies"
    custom_dir.mkdir()
    cookie = custom_dir / "threads_cookies.txt"
    cookie.write_text("session=1\n", encoding="utf-8")

    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    assert _resolve_threads_cookie_path() == cookie


def test_fetch_content_threads_uses_resolved_cookie_path(monkeypatch, tmp_path):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    cookie = custom_dir / "threads_cookies.txt"
    cookie.write_text("session=1\n", encoding="utf-8")

    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)

    from agent.detection import _fetch_content

    _fetch_content("https://threads.net/@u/post/1", "threads")
    assert captured
    cmd = captured[0]
    assert "-b" in cmd
    assert str(cookie) in cmd
