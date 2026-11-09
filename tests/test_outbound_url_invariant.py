"""Invariant: URLs leaving the operator trust boundary use display-safe form (no userinfo/query/fragment)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.pipelines.threads import step_threads_extract_video
from agent.pipelines.web import step_firecrawl_extract
from agent.redact import url_for_display

_SENSITIVE_SOURCE = (
    "https://sekret:pass@www.threads.net/@u/post/ABC?igsh=track&utm=x#frag"
)


@pytest.fixture
def tracked_config(tmp_path):
    temp = tmp_path / "temp"
    temp.mkdir(parents=True)
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=tmp_path / "vault",
        temp_dir=temp,
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )


def _assert_no_credentials_in_payload(payload: str) -> None:
    assert "sekret" not in payload
    assert "pass@" not in payload
    assert "pass:" not in payload
    assert "igsh=track" not in payload
    assert "utm=x" not in payload
    assert "#frag" not in payload


@patch("agent.pipelines.threads._curl_threads")
@patch("subprocess.run")
@patch("agent.pipelines.threads.ensure_h264", side_effect=lambda p: p)
def test_threads_video_download_referer_is_display_safe(
    mock_ensure_h264, mock_run, mock_curl, tracked_config,
):
    html = (
        '{"video_versions":[{"url":"https://scontent.cdninstagram.com/v/test.mp4",'
        '"height":1080}],"caption":{"text":"x"},"username":"x"}'
        '<div class="_ab6c">x</div>'
    )
    mock_curl.return_value = (html, "https://www.threads.net/@u/post/XYZ")

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0, stdout="", stderr="")
        out = Path(tracked_config.temp_dir) / "tracked_slug_threads_video.mp4"
        out.write_bytes(b"x" * 2048)
        return result

    mock_run.side_effect = fake_run
    task = {
        "id": "thr-vid-1",
        "slug": "tracked_slug",
        "source_url": _SENSITIVE_SOURCE,
    }
    step_threads_extract_video(task, tracked_config)
    cmd = mock_run.call_args[0][0]
    referer = next(cmd[i + 1] for i, part in enumerate(cmd) if part == "-H" and cmd[i + 1].startswith("Referer:"))
    expected = url_for_display(_SENSITIVE_SOURCE)
    assert referer == f"Referer: {expected}"
    _assert_no_credentials_in_payload(referer)
    assert referer == f"Referer: {expected}"


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_scrape_json_url_is_display_safe(mock_run, tracked_config):
    tracked_config.temp_dir.mkdir(parents=True, exist_ok=True)
    long_markdown = "# Title\n\n" + ("Content paragraph. " * 80)
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"data": {"markdown": long_markdown}}),
        stderr="",
    )
    task = {
        "id": "fc-1",
        "slug": "fc_slug",
        "source_url": _SENSITIVE_SOURCE.replace("threads.net", "example.com"),
    }
    step_firecrawl_extract(task, tracked_config)
    cmd = mock_run.call_args[0][0]
    payload_json = cmd[cmd.index("-d") + 1]
    _assert_no_credentials_in_payload(payload_json)
    curl_payload = json.loads(payload_json)
    assert curl_payload["url"] == url_for_display(task["source_url"])
    assert "?" not in curl_payload["url"]
    assert "#" not in curl_payload["url"]
