"""Rumble pipeline tests — detection, routing, Flaresolverr resolution + CDN download."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.consumer import Consumer
from agent.detection import detect_content_subtype, detect_platform
from agent.exceptions import ErrorType, StepError
from agent.pipelines import get_pipeline
from agent.pipelines.rumble import _flaresolverr_get, resolve_rumble, step_rumble_download_video

RUMBLE_URL = (
    "https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy"
)


class TestRumbleDetection:

    def test_detect_platform_rumble_url(self):
        assert detect_platform(RUMBLE_URL) == "rumble"

    def test_detect_content_subtype_rumble(self):
        assert detect_content_subtype(RUMBLE_URL, "rumble") == "rumble"


class TestRumblePipelineRouting:

    def test_rumble_pipeline_registered(self):
        pipeline = get_pipeline("rumble")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        assert "download_video" in step_names
        assert "firecrawl_extract" not in step_names

    def test_unknown_content_type_does_not_fall_back_to_web(self):
        pipeline = get_pipeline("nonexistent_content_type_xyz")
        web_pipeline = get_pipeline("web")
        assert pipeline is None
        assert web_pipeline is not None


@pytest.fixture
def rumble_consumer(tmp_path):
    from agent.db import LOGS_SCHEMA, AGENT_SCHEMA, init_db

    config = Config(
        kb_agent_dir=WORKSPACE,
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        max_retries=3,
        retry_delays=[0.01, 0.02, 0.05],
        text_parallel=2,
        video_semaphore=1,
    )
    c = Consumer(config)
    c.agent_conn = init_db(tmp_path / "agent.db", AGENT_SCHEMA)
    c.logs_conn = init_db(tmp_path / "logs.db", LOGS_SCHEMA)
    return c


@pytest.mark.asyncio
async def test_consumer_fails_loudly_when_pipeline_missing(rumble_consumer):
    """Mis-routed tasks must not silently run web steps from STEP_REGISTRY."""
    with pytest.raises(StepError) as exc:
        await rumble_consumer._run_step(
            "firecrawl_extract",
            {"id": "t-rumble-regression", "content_type": "nonexistent_content_type_xyz"},
        )
    assert exc.value.error_type == ErrorType.FAIL
    assert "nonexistent_content_type_xyz" in str(exc.value)


class TestFlaresolverrSsrf:

    @patch("agent.pipelines.rumble.subprocess.run")
    def test_flaresolverr_rejects_metadata_ip(self, mock_run):
        with pytest.raises(ValueError, match="not allowed"):
            _flaresolverr_get("http://169.254.169.254/latest/meta-data/")
        mock_run.assert_not_called()

    @patch("agent.pipelines.rumble.subprocess.run")
    def test_flaresolverr_curl_disables_redirects(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "status": "ok",
                "solution": {"response": "<html></html>"},
            }),
            stderr="",
        )
        _flaresolverr_get("https://rumble.com/v12345-example")
        cmd = mock_run.call_args[0][0]
        assert "--max-redirs" in cmd
        assert cmd[cmd.index("--max-redirs") + 1] == "0"
        assert any("localhost:8191" in str(part) for part in cmd)


class TestResolveRumble:

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_resolves_slug_to_real_id_and_formats(self, mock_fs):
        # slug v2y5lfu-… resolves to the REAL id v2vk68c (redirect alias)
        mock_fs.side_effect = [
            '<html><script type="application/ld+json">{"@type":"VideoObject",'
            '"embedUrl":"https://rumble.com/embed/v2vk68c/"}</script></html>',
            '<pre>{"title":"Evidence of Crimes","ua":{"mp4":{'
            '"360":{"url":"https://hugh.cdn.rumble.cloud/video/...kaa.mp4","meta":{"h":360}},'
            '"720":{"url":"https://hugh.cdn.rumble.cloud/video/...baa.mp4","meta":{"h":720}}'
            '}}}</pre>',
        ]
        title, formats = resolve_rumble("https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy")
        assert title == "Evidence of Crimes"
        assert [f["height"] for f in formats] == [360, 720]
        assert formats[1]["url"].endswith("baa.mp4")

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_no_video_id_raises(self, mock_fs):
        mock_fs.return_value = "<html><body>no embed here</body></html>"
        try:
            resolve_rumble("https://rumble.com/vnosuchvideo")
        except Exception as exc:  # StepError(RETRY)
            assert "video id not found" in str(exc)
        else:
            raise AssertionError("expected StepError")

    @patch("agent.pipelines.rumble._flaresolverr_get")
    def test_no_mp4_formats_raises(self, mock_fs):
        mock_fs.side_effect = [
            '<a href="https://rumble.com/embed/v2vk68c/">embed</a>',
            '<pre>{"title":"T","ua":{}}</pre>',
        ]
        try:
            resolve_rumble("https://rumble.com/v2vk68c-foo")
        except Exception as exc:
            assert "No mp4 formats" in str(exc)
        else:
            raise AssertionError("expected StepError")


class TestRumbleDownloadUsesCachedMetadata:

    @pytest.fixture
    def rumble_config(self, tmp_path):
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return Config(
            kb_agent_dir=WORKSPACE,
            vault_path=tmp_path / "vault",
            temp_dir=temp_dir,
            cookie_dir=tmp_path / "cookies",
        )

    @patch("agent.pipelines.rumble.ensure_h264", side_effect=lambda p: p)
    @patch("agent.pipelines.rumble.subprocess.run")
    @patch("agent.pipelines.rumble.resolve_rumble")
    def test_download_reads_metadata_json_instead_of_resolve(
        self, mock_resolve, mock_run, _mock_h264, rumble_config,
    ):
        """download_video merges CDN URLs from resolve when the sidecar stores height only."""
        slug = "rumble_test_slug"
        cdn_url = "https://hugh.cdn.rumble.cloud/video/test-baa.mp4"
        meta_path = rumble_config.temp_dir / f"{slug}_metadata.json"
        meta_path.write_text(
            json.dumps({
                "title": "Cached Title",
                "formats": [{"height": 720}],
                "source_url": "https://rumble.com/v2vk68c-test",
            }),
            encoding="utf-8",
        )
        mock_resolve.return_value = (
            "Cached Title",
            [{"url": cdn_url, "height": 720}],
        )
        video_path = rumble_config.temp_dir / f"{slug}.mp4"
        video_path.write_bytes(b"\x00" * 2000)

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        task = {
            "id": "task-rumble-1",
            "slug": slug,
            "source_url": "https://rumble.com/v2vk68c-test",
        }
        result = step_rumble_download_video(task, rumble_config)

        mock_resolve.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cdn_url in cmd
        assert "--max-redirs" in cmd
        assert cmd[cmd.index("--max-redirs") + 1] == "0"
        with pytest.raises(ValueError, match="not allowed"):
            from agent.fetch_guard import _assert_fetch_url_allowed

            _assert_fetch_url_allowed("http://127.0.0.1/redirected.mp4")
        assert result == str(video_path)

    @patch("agent.pipelines.rumble.ensure_h264", side_effect=lambda p: p)
    @patch("agent.pipelines.rumble.subprocess.run")
    def test_download_rejects_non_allowlisted_cdn_host(
        self, mock_run, _mock_h264, rumble_config,
    ):
        slug = "rumble_evil_cdn"
        meta_path = rumble_config.temp_dir / f"{slug}_metadata.json"
        meta_path.write_text(
            json.dumps({
                "title": "Evil CDN",
                "formats": [{"url": "https://evil.example.com/video.mp4", "height": 720}],
                "source_url": "https://rumble.com/v2vk68c-test",
            }),
            encoding="utf-8",
        )
        task = {
            "id": "task-rumble-evil",
            "slug": slug,
            "source_url": "https://rumble.com/v2vk68c-test",
        }
        with pytest.raises(StepError, match="allowlisted"):
            step_rumble_download_video(task, rumble_config)
        mock_run.assert_not_called()
