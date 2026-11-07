"""Tests for the Bunny Stream / TDAH summit pipeline — detection and UUID extraction."""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.detection import detect_platform
from agent.exceptions import StepError
from agent.pipelines.bunny_stream import (
    BUNNY_UUID_RE,
    _assert_bunny_cdn_url_allowed,
    _curl_page,
    _resolve_cdn_base,
    extract_bunny_video_uuid,
)


METADATA_URL = "http://169.254.169.254/latest/meta-data/"


class TestBunnyStreamSsrf:

    @patch("agent.pipelines.bunny_stream.subprocess.run")
    def test_curl_page_rejects_metadata_ip(self, mock_run):
        with pytest.raises(StepError, match="not allowed"):
            _curl_page(METADATA_URL)
        mock_run.assert_not_called()


class TestBunnyStreamDetection:

    def test_detect_terragrandir(self):
        url = "https://www.terragrandir.com/tdah-adulte-j1-intro/"
        assert detect_platform(url) == "bunny_stream"

    def test_detect_player_mediadelivery(self):
        url = "https://player.mediadelivery.net/embed/12345/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert detect_platform(url) == "bunny_stream"

    def test_detect_youtube_unchanged(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert detect_platform(url) == "youtube"


class TestBunnyUuidExtraction:

    SAMPLE_HTML = """
    <html><body>
    <iframe src="https://player.mediadelivery.net/embed/384567/12345678-abcd-ef01-2345-6789abcdef01"
            allowfullscreen></iframe>
    </body></html>
    """

    def test_uuid_regex_on_sample_html(self):
        m = BUNNY_UUID_RE.search(self.SAMPLE_HTML)
        assert m is not None
        assert m.group(1) == "12345678-abcd-ef01-2345-6789abcdef01"

    def test_extract_bunny_video_uuid_from_html(self):
        uuid = extract_bunny_video_uuid(self.SAMPLE_HTML)
        assert uuid == "12345678-abcd-ef01-2345-6789abcdef01"

    def test_extract_from_player_url(self):
        url = (
            "https://player.mediadelivery.net/embed/99/"
            "abcdef12-3456-7890-abcd-ef1234567890"
        )
        uuid = extract_bunny_video_uuid(url)
        assert uuid == "abcdef12-3456-7890-abcd-ef1234567890"


class TestBunnyCdnAllowlist:

    def test_rejects_non_cdn_host(self):
        with pytest.raises(StepError, match="not allowlisted"):
            _assert_bunny_cdn_url_allowed("https://evil.example.com/play_720p.mp4")

    @patch("agent.pipelines.bunny_stream._curl_page")
    def test_resolve_cdn_base_validates_regex_match(self, mock_curl):
        uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        mock_curl.return_value = f"https://vz-testzone.b-cdn.net/{uuid}/playlist.m3u8"
        base = _resolve_cdn_base(
            f"https://player.mediadelivery.net/embed/1/{uuid}",
            uuid,
        )
        assert base == "https://vz-testzone.b-cdn.net"


class TestBunnyDownloadVideo:

    @pytest.fixture
    def config(self, tmp_path):
        from agent.config import Config

        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        return Config(
            kb_agent_dir=WORKSPACE,
            vault_path=tmp_path / "vault",
            temp_dir=temp_dir,
            cookie_dir=tmp_path / "cookies",
        )

    @pytest.fixture
    def task_dict(self):
        return {
            "id": "test-bunny-1",
            "source_url": "https://www.terragrandir.com/tdah-adulte-j1-test/",
            "content_type": "bunny_stream",
            "slug": "tdah-adulte-2026-j1-test",
        }

    @patch("agent.pipelines.bunny_stream.subprocess.run")
    def test_download_fetches_page_then_mp4(self, mock_run, config, task_dict):
        from agent.pipelines.bunny_stream import step_download_video

        page_html = (
            '<iframe src="https://player.mediadelivery.net/embed/1/'
            '12345678-abcd-ef01-2345-6789abcdef01"></iframe>'
        )
        embed_html = (
            'https://vz-testzone.b-cdn.net/12345678-abcd-ef01-2345-6789abcdef01/playlist.m3u8'
        )
        slug = task_dict["slug"]
        mp4_path = config.temp_dir / f"{slug}.mp4"

        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "curl" and task_dict["source_url"] in cmd:
                return MagicMock(returncode=0, stdout=page_html, stderr="")
            if cmd[0] == "curl" and "play_720p.mp4" in " ".join(cmd):
                mp4_path.write_bytes(b"x" * 200_000)
                return MagicMock(returncode=0, stdout="", stderr="")
            if cmd[0] == "curl" and "embed" in " ".join(cmd):
                return MagicMock(returncode=0, stdout=embed_html, stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        result = step_download_video(task_dict, config)
        assert result == str(mp4_path)
        assert mp4_path.exists()

        mp4_curls = [
            c for c in (mock_run.call_args_list or [])
            if (args := c[0][0])[0] == "curl" and "play_720p.mp4" in " ".join(args)
        ]
        assert mp4_curls, "expected curl mp4 download"
        mp4_cmd = mp4_curls[0][0][0]
        assert "--max-redirs" in mp4_cmd
        assert mp4_cmd[mp4_cmd.index("--max-redirs") + 1] == "0"
