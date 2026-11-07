"""Tests for the Mega.nz video pipeline."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.exceptions import StepError
from agent.pipelines.mega_video import (
    step_mega_download_video,
    step_mega_extract_metadata,
)

METADATA_URL = "http://169.254.169.254/latest/meta-data/"


@pytest.fixture
def config(tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    return Config(
        kb_agent_dir=WORKSPACE,
        vault_path=tmp_path / "vault",
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
    )


@pytest.fixture
def task_dict():
    return {
        "id": "test-mega-video-1",
        "source_url": "https://mega.nz/file/abc#key",
        "content_type": "mega_video",
        "slug": "mega_video_test",
    }


class TestMegaVideoSsrf:

    @patch("agent.pipelines.mega_video.subprocess.run")
    def test_extract_metadata_rejects_metadata_ip(self, mock_run, config, task_dict):
        task_dict["source_url"] = METADATA_URL
        with pytest.raises(StepError, match="not allowed"):
            step_mega_extract_metadata(task_dict, config)
        mock_run.assert_not_called()

    @patch("agent.pipelines.mega_video.subprocess.run")
    def test_download_video_rejects_metadata_ip(self, mock_run, config, task_dict):
        task_dict["source_url"] = METADATA_URL
        with pytest.raises(StepError, match="not allowed"):
            step_mega_download_video(task_dict, config)
        mock_run.assert_not_called()
