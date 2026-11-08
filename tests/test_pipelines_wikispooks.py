"""Tests for Wikispooks platform detection and pipeline bypass URL handling."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.core.detect import detect_type
from agent.detection import detect_platform
from agent.pipelines import PIPELINES
from agent.pipelines.wikispooks import (
    step_wikispooks_firecrawl_extract,
    wikispooks_bypass_url,
)

# Register the wikispooks pipeline for registry tests
import agent.pipelines.wikispooks  # noqa: F401


@pytest.fixture
def config(tmp_path):
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        article_min_quotes=2,
        article_min_size=100,
        firecrawl_url="http://localhost:3002/v2/scrape",
    )


class TestWikispooksDetection:
    def test_detect_platform_wikispooks_org(self):
        url = "https://wikispooks.org/wiki/COVID-19/Perpetrators/Bilderberg"
        assert detect_platform(url) == "wikispooks"

    def test_detect_platform_wikispooks_org_http(self):
        url = "http://wikispooks.org/wiki/Some_Page"
        assert detect_platform(url) == "wikispooks"

    def test_detect_platform_wikispooks_com(self):
        url = "https://wikispooks.com/wiki/Foo"
        assert detect_platform(url) == "wikispooks"

    def test_detect_type_wikispooks_hosts(self):
        assert detect_type("https://wikispooks.com/wiki/Foo") == {
            "platform": "wikispooks",
            "content_type": "wikispooks",
        }
        assert detect_type("https://wikispooks.org/wiki/Foo") == {
            "platform": "wikispooks",
            "content_type": "wikispooks",
        }

    def test_detect_platform_unrelated_url_unchanged(self):
        assert detect_platform("https://example.com/wiki/Foo") == "web"


class TestWikispooksBypassUrl:
    def test_bypass_url_transformation(self):
        direct = "https://wikispooks.org/wiki/COVID-19/Perpetrators/Bilderberg"
        expected = (
            "https://wikispooks.org/w/welcome_back.php?"
            "goto=%2Fwiki%2FCOVID-19%2FPerpetrators%2FBilderberg"
        )
        assert wikispooks_bypass_url(direct) == expected

    def test_bypass_preserves_scheme_and_host(self):
        direct = "https://wikispooks.com/wiki/Document/911"
        result = wikispooks_bypass_url(direct)
        assert result.startswith("https://wikispooks.com/w/welcome_back.php?goto=")
        assert "%2Fwiki%2FDocument%2F911" in result

    def test_bypass_strips_netloc_userinfo(self):
        direct = "https://user:pass@wikispooks.org/wiki/Secret_Page"
        result = wikispooks_bypass_url(direct)
        assert result.startswith("https://wikispooks.org/w/welcome_back.php?goto=")
        assert "user" not in result
        assert "pass" not in result


class TestWikispooksFirecrawlExtract:
    @patch("agent.pipelines.wikispooks.step_firecrawl_extract")
    def test_uses_bypass_url_leaves_task_source_direct(self, mock_extract, config):
        direct = "https://wikispooks.org/wiki/COVID-19/Perpetrators/Bilderberg"
        task = {
            "id": "task-abc",
            "source_url": direct,
            "slug": "wikispooks_covid_bilderberg",
        }
        mock_extract.return_value = "/tmp/wikispooks_covid_bilderberg_raw.md"

        step_wikispooks_firecrawl_extract(task, config)

        assert task["source_url"] == direct
        call_task = mock_extract.call_args[0][0]
        assert call_task["source_url"] == wikispooks_bypass_url(direct)
        assert call_task is not task

    @patch("agent.pipelines.web.subprocess.run")
    def test_firecrawl_payload_uses_bypass_url(self, mock_run, config, tmp_path):
        config.temp_dir = tmp_path / "tmp"
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        direct = "https://wikispooks.org/wiki/Test/Article"
        bypass = wikispooks_bypass_url(direct)
        long_markdown = "# Title\n\n" + ("Content paragraph. " * 80)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"data": {"markdown": long_markdown}}),
            stderr="",
        )
        task = {
            "id": "task-xyz",
            "source_url": direct,
            "slug": "wikispooks_test_article",
        }

        step_wikispooks_firecrawl_extract(task, config)

        assert task["source_url"] == direct
        cmd = mock_run.call_args[0][0]
        payload_json = cmd[cmd.index("-d") + 1]
        curl_payload = json.loads(payload_json)
        assert curl_payload["url"] == bypass
        assert curl_payload["url"] != direct


class TestWikispooksPipelineRegistration:
    def test_pipeline_registered(self):
        assert "wikispooks" in PIPELINES

    def test_pipeline_steps_match_web_pattern(self):
        steps = PIPELINES["wikispooks"]
        assert [s.name for s in steps] == [
            "firecrawl_extract",
            "synthesize_note",
            "quality_gate",
            "minio_upload",
            "see_also",
            "git_push",
        ]
