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


def test_tdah_fetch_calls_guard_before_urlopen(monkeypatch):
    mod = _load_tdah_module()
    seen: list[str] = []

    monkeypatch.setattr(
        mod,
        "_assert_fetch_url_allowed",
        lambda url: seen.append(url),
    )

    def fake_urlopen(req, timeout=40):
        return type("R", (), {"read": lambda self: b"<html></html>"})()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    out = mod.fetch("https://example.com/page")
    assert seen == ["https://example.com/page"]
    assert "<html>" in out


def test_tdah_bunny_cdn_url_passes_fetch_guard():
    mod = _load_tdah_module()
    url = f"{mod.CDN}/00000000-0000-0000-0000-000000000000/play_720p.mp4"
    mod._assert_fetch_url_allowed(url)


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
