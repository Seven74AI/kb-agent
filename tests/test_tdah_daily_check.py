"""Regression tests for TDAH daily check SSRF guard on outbound fetches."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tdah_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "tdah_daily_check",
        _REPO_ROOT / "scripts" / "tdah_daily_check.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_tdah_fetch_rejects_private_url():
    mod = _load_tdah_module()
    with pytest.raises(ValueError, match="not allowed"):
        mod.fetch("http://127.0.0.1/evil")


def test_tdah_fetch_calls_guard_before_curl(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    seen: list[str] = []
    curl_cmds: list[list] = []

    monkeypatch.setattr(
        mod,
        "_assert_fetch_url_allowed",
        lambda url: seen.append(url),
    )

    def fake_sh(cmd, **kw):
        curl_cmds.append(list(cmd))
        return MagicMock(returncode=0, stdout="<html></html>", stderr="")

    monkeypatch.setattr(mod, "sh", fake_sh)

    out = mod.fetch("https://example.com/page")
    assert seen == ["https://example.com/page"]
    assert "<html>" in out
    assert curl_cmds
    curl = curl_cmds[0]
    assert "--max-redirs" in curl
    assert curl[curl.index("--max-redirs") + 1] == "0"


def test_tdah_fetch_redirect_target_never_guarded(monkeypatch):
    from unittest.mock import MagicMock

    from agent.fetch_guard import _assert_fetch_url_allowed as real_guard

    mod = _load_tdah_module()
    guarded: list[str] = []

    def track_guard(url: str) -> None:
        guarded.append(url)
        real_guard(url)

    monkeypatch.setattr(mod, "_assert_fetch_url_allowed", track_guard)
    monkeypatch.setattr(
        mod,
        "sh",
        lambda cmd, **kw: MagicMock(
            returncode=0, stdout="<html>no redirect follow</html>", stderr=""
        ),
    )
    mod.fetch("https://www.terragrandir.com/page")
    assert guarded == ["https://www.terragrandir.com/page"]
    with pytest.raises(ValueError, match="not allowed"):
        real_guard("http://127.0.0.1/stolen")


def test_tdah_bunny_cdn_url_passes_fetch_guard():
    mod = _load_tdah_module()
    url = f"{mod.CDN}/00000000-0000-0000-0000-000000000000/play_720p.mp4"
    mod._assert_fetch_url_allowed(url)


def test_tdah_storage_paths_use_config_bucket(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    config = MagicMock()
    config.storage_bucket = "kb-knowledge-base"
    config.storage_alias = "tigris"
    config.kb_agent_dir = MagicMock()
    ls_targets: list[str] = []

    def fake_sh(cmd, **kw):
        if cmd[:2] == ["mc", "ls"]:
            ls_targets.append(cmd[2])
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "sh", fake_sh)
    monkeypatch.setattr(mod, "load_config", lambda: config)
    monkeypatch.setattr(mod, "init_db", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        mod,
        "fetch",
        lambda url: (
            '<a href="/tdah-adulte-j1-newslug/">'
            if url == mod.MAIN
            else ""
        ),
    )
    monkeypatch.setattr(mod, "bunny_uuid", lambda slug: None)

    mod.main()

    expected_vb = "tigris/kb-knowledge-base/videos"
    expected_tb = "tigris/kb-knowledge-base/transcripts"
    assert expected_vb in ls_targets
    assert expected_tb in ls_targets
    assert "kb-agent-archive" not in ls_targets


def test_tdah_fully_processed_requires_video_and_transcript(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    vb = "tigris/bucket/videos"
    tb = "tigris/bucket/transcripts"
    slug = "tdah-adulte-2026-j2-only-mp4"

    def fake_sh(cmd, **kw):
        if cmd[2] == vb:
            return MagicMock(returncode=0, stdout=f"{slug}.mp4", stderr="")
        if cmd[2] == tb:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "sh", fake_sh)
    assert slug not in mod.fully_processed_slugs(vb, tb)


def test_tdah_skips_dispatch_when_mc_upload_fails(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    dispatched: list[str] = []
    monkeypatch.setattr(
        mod,
        "_finalize_video_dispatch",
        lambda *a, **k: dispatched.append("called") or "task-x",
    )

    def fake_sh(cmd, **kw):
        if cmd[:2] == ["mc", "cp"] and cmd[-1].endswith(".mp4"):
            return MagicMock(returncode=1, stderr="video fail", stdout="")
        if cmd[:2] == ["mc", "cp"] and cmd[-1].endswith(".json"):
            return MagicMock(returncode=0, stderr="", stdout="")
        return MagicMock(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(mod, "sh", fake_sh)

    mp4 = tmp_path / "v.mp4"
    mp4.write_bytes(b"x" * 200)
    mpath = tmp_path / "v.json"
    mpath.write_text("{}", encoding="utf-8")
    ns = "tdah-adulte-2026-j9-test"
    vb = "tigris/kb-knowledge-base/videos"
    tb = "tigris/kb-knowledge-base/transcripts"

    task_id = mod.upload_transcript_and_dispatch(
        ns,
        str(mp4),
        mpath,
        vb,
        tb,
        "https://example.com/",
        MagicMock(),
        MagicMock(),
    )

    assert task_id is None
    assert dispatched == []


def test_tdah_finalize_dispatch_returns_none_on_error(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    ns = "tdah-adulte-2026-j1-test"
    page_url = "https://www.terragrandir.com/tdah-adulte-j1-test/"

    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "error", "error": "url required"},
    )
    assert mod._finalize_video_dispatch(page_url, MagicMock(), MagicMock(), ns) is None


def test_tdah_finalize_dispatch_returns_task_id_on_success(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_tdah_module()
    ns = "tdah-adulte-2026-j1-test"
    page_url = "https://www.terragrandir.com/tdah-adulte-j1-test/"

    monkeypatch.setattr(
        mod,
        "dispatch_synthesize",
        lambda *a, **k: {"status": "pending", "task_id": "task-abc"},
    )
    assert mod._finalize_video_dispatch(page_url, MagicMock(), MagicMock(), ns) == "task-abc"


def test_tdah_bunny_download_guards_cdn_and_disables_redirects(monkeypatch):
    from agent.fetch_guard import _assert_fetch_url_allowed as real_guard

    mod = _load_tdah_module()
    guard_calls: list[str] = []
    sh_calls: list[list] = []

    def track_guard(url: str) -> None:
        guard_calls.append(url)
        real_guard(url)

    monkeypatch.setattr(mod, "_assert_fetch_url_allowed", track_guard)
    monkeypatch.setattr(mod, "sh", lambda cmd, **kw: sh_calls.append(cmd))

    uu = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    expected = f"{mod.CDN}/{uu}/play_720p.mp4"
    mod._assert_fetch_url_allowed(expected)
    mod.sh(
        ["curl", "-s", "--max-redirs", "0", "-H", mod.REF, "-o", "/tmp/x.mp4", expected],
    )

    assert guard_calls == [expected]
    assert sh_calls
    curl = sh_calls[0]
    assert "--max-redirs" in curl
    assert curl[curl.index("--max-redirs") + 1] == "0"
    with pytest.raises(ValueError, match="not allowed"):
        real_guard("http://127.0.0.1/stolen.mp4")
