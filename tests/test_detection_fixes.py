"""Tests for detection helpers (dedup, threads cookies)."""

import logging
import subprocess
from pathlib import Path
import pytest

from agent.config import Config
from agent.detection import (
    DedupCheckError,
    _fetch_content,
    _resolve_threads_cookie_path,
    dedup_check,
    detect_content_subtype,
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


def test_dedup_check_returncode_one_means_not_in_vault(core_config, monkeypatch):
    """Grep exit 1 (no matches) must return False, not raise."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    assert dedup_check("https://example.com/missing", core_config.vault_path) is False


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
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout='{"status":"ok"}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is True
    assert reason == ""


def test_preflight_rumble_unhealthy_when_flaresolverr_non_ok_json(tmp_path, monkeypatch):
    """Curl success with non-ok JSON must not count as healthy."""
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(
                cmd, 0, stdout='{"status":"error","message":"solver busy"}', stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is False
    assert "Flaresolverr error" in reason


def test_preflight_rumble_unhealthy_when_flaresolverr_down(tmp_path, monkeypatch):
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is False
    assert "Flaresolverr" in reason


def test_preflight_rumble_ok_without_cookie_file(tmp_path, monkeypatch):
    """Rumble preflight must not require rumble_cookies.txt."""
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault")
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        if any("8191" in str(part) for part in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout='{"status":"ok"}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    ok, reason = preflight_health("rumble")
    assert ok is True
    assert reason == ""


METADATA_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.mark.parametrize("platform", ["threads", "substack", "mega"])
def test_fetch_content_rejects_metadata_ip(platform, monkeypatch):
    """Subtype detection fetches must block cloud metadata SSRF targets."""

    def boom(*args, **kwargs):
        raise AssertionError("subprocess must not run for blocked URL")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    with pytest.raises(ValueError, match="not allowed"):
        _fetch_content(METADATA_URL, platform)


def test_comment_id_on_non_instagram_does_not_route_to_comment(monkeypatch):
    url = "https://example.com/page?comment_id=123"
    monkeypatch.setattr("agent.detection._fetch_content", lambda u, p: None)
    monkeypatch.setattr("agent.detection.route_jev", lambda u, c: "web")
    assert detect_content_subtype(url, "web") == "web"


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
    assert "--config" in cmd
    assert "-b" not in cmd
    assert str(cookie) not in cmd
    conf_path = custom_dir / ".curl-threads_cookies.txt.conf"
    assert conf_path.is_file()
    assert oct(conf_path.stat().st_mode)[-3:] == "600"
    assert "--max-redirs" in cmd
    assert cmd[cmd.index("--max-redirs") + 1] == "0"


def test_fetch_content_substack_curl_disables_redirects(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="<html></html>", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    _fetch_content("https://blog.substack.com/p/post", "substack")
    cmd = captured[0]
    assert "--max-redirs" in cmd
    assert cmd[cmd.index("--max-redirs") + 1] == "0"


def test_detect_threads_legacy_curl_disables_redirects_and_cookie_config(
    monkeypatch, tmp_path,
):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    cookie = custom_dir / "threads_cookies.txt"
    cookie.write_text("session=1\n", encoding="utf-8")
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='"video_versions":[]', stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)

    from agent.detection import _detect_threads_subtype

    _detect_threads_subtype("https://threads.net/@u/post/1")
    cmd = captured[0]
    assert "--max-redirs" in cmd
    assert cmd[cmd.index("--max-redirs") + 1] == "0"
    assert "--config" in cmd
    assert "-b" not in cmd
    assert str(cookie) not in cmd
    conf_path = custom_dir / ".curl-threads_cookies.txt.conf"
    assert conf_path.is_file()
    assert oct(conf_path.stat().st_mode)[-3:] == "600"


def test_fetch_content_threads_logs_on_failure(monkeypatch, caplog):
    def boom(cmd, **kwargs):
        raise OSError("curl vanished")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    caplog.set_level(logging.WARNING)
    assert _fetch_content("https://threads.net/@u/post/1", "threads") is None
    assert any("Threads content fetch failed" in rec.message for rec in caplog.records)


def test_fetch_content_substack_logs_on_failure(monkeypatch, caplog):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["curl"], timeout=20)

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    caplog.set_level(logging.WARNING)
    assert _fetch_content("https://blog.substack.com/p/post", "substack") is None
    assert any("Substack content fetch failed" in rec.message for rec in caplog.records)


def test_detect_content_subtype_jev_failure_logs_warning(monkeypatch, caplog):
    def boom(url, content):
        raise RuntimeError("jev down")

    monkeypatch.setattr("agent.detection._fetch_content", lambda u, p: "<html/>")
    monkeypatch.setattr("agent.detection.route_jev", boom)
    monkeypatch.setattr(
        "agent.detection._detect_substack_subtype", lambda url: "substack_text",
    )
    caplog.set_level(logging.WARNING)
    assert (
        detect_content_subtype("https://example.substack.com/p/x", "substack")
        == "substack_text"
    )
    assert any("Jev routing failed" in rec.message for rec in caplog.records)
