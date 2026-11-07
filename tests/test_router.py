"""Tests for Jev routing (agent.router) + the detection cascade."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import detection, router  # noqa: E402


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def test_route_jev_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        router.route_jev("https://example.com", api_key=None)


def test_route_jev_returns_choice(monkeypatch):
    payload = {"answers": {"content_type": {"choice": "threads_video"}}}
    monkeypatch.setattr(router.httpx, "post", lambda *a, **kw: _FakeResp(payload))
    got = router.route_jev("https://threads.net/@x/post/1", "video_versions present", api_key="k")
    assert got == "threads_video"


def test_route_jev_scrubs_credentials_in_openrouter_payload(monkeypatch):
    payload = {"answers": {"content_type": {"choice": "substack_text"}}}
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _FakeResp(payload)

    monkeypatch.setattr(router.httpx, "post", fake_post)
    secret = "api_key=supersecret123"
    router.route_jev(
        "https://example.substack.com/p/post",
        f"<title>Article</title><p>{secret}</p>",
        api_key="k",
    )
    state = captured["state"]
    assert "supersecret123" not in state
    assert "[credential redacted]" in state
    assert "Article" in state


def test_route_jev_error_snippet_scrubs_credentials(monkeypatch):
    body = '{"error":"invalid api_key=leaked-token-here"}'

    def fake_post(url, **kwargs):
        return _FakeResp(body, status_code=502)

    monkeypatch.setattr(router.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="OpenRouter 502") as excinfo:
        router.route_jev("https://example.com", api_key="k")
    assert "leaked-token-here" not in str(excinfo.value)
    assert "[credential redacted]" in str(excinfo.value)


def test_route_jev_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        router.httpx, "post", lambda *a, **kw: _FakeResp({"error": "x"}, status_code=400)
    )
    with pytest.raises(RuntimeError, match="OpenRouter 400"):
        router.route_jev("https://example.com", api_key="k")


def test_detect_content_subtype_simple_returns_platform():
    # regex-clean platforms return as-is, no Jev call
    assert detection.detect_content_subtype("https://youtube.com/watch?v=x", "youtube") == "youtube"
    assert detection.detect_content_subtype("https://instagram.com/reel/x/", "instagram_reel") == "instagram_reel"
    assert detection.detect_content_subtype("https://terragrandir.com/talk/", "bunny_stream") == "bunny_stream"


def test_detect_content_subtype_threads_deterministic(monkeypatch):
    # threads is fully deterministic: a non-null video_versions array == video,
    # its absence == text. Jev is never called for threads.
    monkeypatch.setattr(detection, "route_jev", lambda url, content: "UNREACHABLE")
    monkeypatch.setattr(
        detection, "_fetch_content",
        lambda url, p: '<html>"video_versions":[{"url":"x"}]</html>',
    )
    assert detection.detect_content_subtype("https://threads.net/@x/post/1", "threads") == "threads_video"
    # null video_versions (no array) is NOT a video
    monkeypatch.setattr(
        detection, "_fetch_content",
        lambda url, p: '<html>"video_versions":null</html>',
    )
    assert detection.detect_content_subtype("https://threads.net/@x/post/1", "threads") == "threads_text"


def test_detect_content_subtype_jev_failure_falls_back(monkeypatch):
    monkeypatch.setattr(detection, "_fetch_content", lambda url, p: None)

    def boom(url, content):
        raise RuntimeError("down")

    monkeypatch.setattr(detection, "route_jev", boom)
    monkeypatch.setattr(detection, "_detect_threads_subtype", lambda url: "threads_text")
    assert detection.detect_content_subtype("https://threads.net/@x/post/1", "threads") == "threads_text"


def test_detect_content_subtype_web_uses_jev(monkeypatch):
    # regex missed (web) → Jev can recover the real platform
    monkeypatch.setattr(detection, "_fetch_content", lambda url, p: None)
    monkeypatch.setattr(detection, "route_jev", lambda url, content: "instagram_reel")
    assert detection.detect_content_subtype("https://www.instagram.com/x/", "web") == "instagram_reel"
