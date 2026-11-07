"""Rumble pipeline tests — detection, routing, Flaresolverr resolution + CDN download."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.consumer import Consumer
from agent.detection import detect_content_subtype, detect_platform
from agent.exceptions import ErrorType, StepError
from agent.pipelines import get_pipeline
from agent.pipelines.rumble import resolve_rumble

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
