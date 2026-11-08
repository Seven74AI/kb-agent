"""Invariant: temp/at-rest JSON sidecars must not store raw URL query, fragment, or signed CDN params."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.pipelines.bunny_stream import step_extract_metadata
from agent.pipelines.instagram import (
    step_ig_carousel_extract,
    step_ig_carousel_metadata,
    step_ig_comment_fetch,
)
from agent.pipelines.rumble import step_rumble_extract_metadata
from agent.pipelines.threads import step_threads_detect, step_threads_extract_text
from agent.redact import url_for_display
import scripts.batch_ingest as batch_ingest
from agent.db import AGENT_SCHEMA, ensure_parent_media_deferral, init_db

_HTTP_URL = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)


def _assert_sidecar_has_no_sensitive_url_strings(payload: object) -> None:
    """Fail when any embedded http(s) URL still carries ? or # (signed params / fragments)."""

    def walk(obj: object) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str):
            for match in _HTTP_URL.finditer(obj):
                url = match.group(0)
                assert "?" not in url, url
                assert "#" not in url, url

    walk(payload)


@pytest.fixture
def sidecar_config(tmp_path):
    temp = tmp_path / "temp"
    temp.mkdir(parents=True)
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=temp,
        cookie_dir=tmp_path / "cookies",
    )


@patch("agent.pipelines.rumble.resolve_rumble")
def test_rumble_metadata_sidecar_omits_signed_cdn_urls(mock_resolve, sidecar_config):
    signed = "https://hugh.cdn.rumble.cloud/video/x.mp4?token=SECRET&exp=999"
    mock_resolve.return_value = (
        "Title",
        [{"url": signed, "height": 720}],
    )
    task = {
        "id": "rumble-1",
        "slug": "rumble_slug",
        "source_url": "https://rumble.com/v123?utm=track",
    }
    step_rumble_extract_metadata(task, sidecar_config)
    meta_path = sidecar_config.temp_dir / "rumble_slug_metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(data)
    assert data["formats"] == [{"height": 720}]
    assert "url" not in data["formats"][0]
    assert data["source_url"] == url_for_display(task["source_url"])


@patch("agent.pipelines.bunny_stream._curl_page", return_value="<html><h1>Talk</h1></html>")
def test_bunny_metadata_sidecar_redacts_source_url(mock_curl, sidecar_config):
    task = {
        "id": "bunny-1",
        "slug": "bunny_slug",
        "source_url": "https://terragrandir.com/talk?access=secret#part",
    }
    step_extract_metadata(task, sidecar_config)
    meta_path = sidecar_config.temp_dir / "bunny_slug_metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(data)
    assert data["source_url"] == url_for_display(task["source_url"])


@patch("agent.pipelines.instagram._validate_ig_cookies")
@patch("agent.pipelines.instagram._assert_fetch_url_allowed")
@patch("subprocess.run")
def test_instagram_carousel_sidecars_omit_tracking_urls(
    mock_run, _mock_assert, _mock_cookies, sidecar_config,
):
    post_url = "https://www.instagram.com/p/ABC/?igsh=deadbeef"
    task = {
        "id": "ig-1",
        "slug": "instagram.com_p_ABC",
        "source_url": post_url,
        "content_type": "instagram_post",
    }

    call_n = {"n": 0}

    def fake_run(cmd, **kwargs):
        call_n["n"] += 1
        result = MagicMock(returncode=0, stderr="")
        if call_n["n"] == 1:
            result.stdout = "Title||user||1||2"
        else:
            result.stdout = "caption"
        return result

    mock_run.side_effect = fake_run
    meta_path = step_ig_carousel_metadata(task, sidecar_config)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(meta)
    assert "url" not in meta

    slides_json = json.dumps([
        {
            "slide": 1,
            "alt_text": "chart",
            "src": "https://cdn.instagram.com/x.jpg?ig_cache_key=SECRET",
        },
    ])
    mock_run.side_effect = lambda *a, **k: MagicMock(
        returncode=0, stdout=slides_json, stderr="",
    )
    slides_path = step_ig_carousel_extract(task, sidecar_config)
    slides = json.loads(Path(slides_path).read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(slides)
    assert "url" not in slides
    assert "src" not in slides["slides"][0]


@patch("agent.pipelines.instagram._fetch_ig_comment")
def test_instagram_ig_comment_sidecar_redacts_urls(mock_fetch, sidecar_config):
    mock_fetch.return_value = {
        "comment_id": "42",
        "comment_text": "See https://track.example/x?jwt=SECRET for more",
        "comment_author": "commenter",
        "comment_created_at": 1,
        "parent_shortcode": "ABC123",
        "parent_url": "https://www.instagram.com/reel/ABC123/?igsh=deadbeef",
        "parent_caption": "Caption with https://cdn.example/img.jpg?sig=TOKEN",
        "parent_author": "poster",
    }
    task = {
        "id": "ig-comment-1",
        "slug": "instagram.com_reel_ABC123",
        "source_url": "https://www.instagram.com/reel/ABC123/?comment_id=42&igsh=deadbeef",
    }
    step_ig_comment_fetch(task, sidecar_config)
    raw_path = sidecar_config.temp_dir / "instagram.com_reel_ABC123_ig_comment.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(data)
    assert data["parent_url"] == url_for_display(mock_fetch.return_value["parent_url"])
    assert "jwt=SECRET" not in data["comment_text"]
    assert "sig=TOKEN" not in data["parent_caption"]


@patch(
    "agent.pipelines.web._fetch_substack_html",
    return_value=(
        '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/img1_1200x800.webp&quot;}"></div>'
    ),
)
@patch("agent.pipelines.web._enqueue_embedded_media")
def test_substack_media_manifest_omits_parent_url(mock_enqueue, _mock_html, sidecar_config):
    from agent.exceptions import StepError
    from agent.pipelines.web import EmbeddedMediaEnqueueStats, step_substack_embedded_media

    mock_enqueue.return_value = EmbeddedMediaEnqueueStats(1, 1)
    task = {
        "id": "sub-1",
        "slug": "parent_slug",
        "source_url": "https://author.substack.com/p/post?token=abc",
        "content_type": "substack_text",
    }
    sidecar_config.kb_agent_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
    init_db(sidecar_config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
    with pytest.raises(StepError):
        step_substack_embedded_media(task, sidecar_config)
    manifest = json.loads(
        (sidecar_config.temp_dir / "parent_slug_media.json").read_text(encoding="utf-8"),
    )
    _assert_sidecar_has_no_sensitive_url_strings(manifest)
    assert "url" not in manifest
    assert manifest["total"] == 1


@patch("agent.pipelines.threads._fetch_parent_post")
@patch("agent.pipelines.threads._curl_threads")
def test_threads_detect_raw_sidecar_sanitizes_urls(mock_curl, mock_parent, sidecar_config):
    task = {
        "id": "thr-detect-1",
        "slug": "threads_detect_slug",
        "source_url": "https://www.threads.net/@u/post/ABC?igsh=track#frag",
    }
    html = (
        '{"caption":{"text":"See https://evil.com/x?secret=1 for more"},'
        '"username":"u","is_reply":true,"parent_post":{"code":"PARENT1"}}'
        '<div class="_ab6c">x</div>'
    )
    mock_curl.return_value = html
    mock_parent.return_value = (
        "Parent body with https://parent.example.com/p?jwt=SECRET embedded"
    )
    step_threads_detect(task, sidecar_config)
    raw_path = sidecar_config.temp_dir / "threads_detect_slug_threads_raw.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(data)
    assert data["url"] == url_for_display(task["source_url"])
    assert data["parent_url"] == "https://www.threads.net/t/PARENT1"
    assert "jwt=SECRET" not in (data.get("parent_context") or "")
    assert "secret=1" not in "".join(data.get("captions") or [])


@patch("agent.pipelines.threads._fetch_parent_post")
@patch("agent.pipelines.threads._curl_threads")
def test_threads_extract_text_raw_sidecar_sanitizes_urls(
    mock_curl, mock_parent, sidecar_config,
):
    task = {
        "id": "thr-extract-1",
        "slug": "threads_extract_slug",
        "source_url": "https://www.threads.net/@u/post/XYZ?utm=camp#top",
    }
    html = (
        '{"caption":{"text":"Link https://track.example/a?key=1"},'
        '"username":"u","is_reply":true,"parent_post":{"code":"PAR99"}}'
    )
    mock_curl.return_value = html
    mock_parent.return_value = "Replying to https://threads.net/t/PAR99?ref=old"
    raw_path = step_threads_extract_text(task, sidecar_config)
    data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(data)
    assert data["url"] == url_for_display(task["source_url"])
    assert data["parent_url"] == "https://www.threads.net/t/PAR99"
    assert "ref=old" not in (data.get("parent_context") or "")
    assert "key=1" not in "".join(data.get("captions") or [])


def test_refresh_parent_media_manifest_omits_deferral_source_url(tmp_path):
    conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    parent = "manifest_parent"
    secret_url = "https://blog.substack.com/p/x?jwt=SECRET"
    config = Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "temp",
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent_media_deferral(conn, parent, secret_url, "substack_text", 2)
    batch_ingest.refresh_parent_media_manifest(conn, config, parent)
    manifest = json.loads((config.temp_dir / f"{parent}_media.json").read_text(encoding="utf-8"))
    _assert_sidecar_has_no_sensitive_url_strings(manifest)
    assert "url" not in manifest
    assert manifest["total"] == 2
