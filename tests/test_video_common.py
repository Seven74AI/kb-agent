"""Tests for shared video pipeline helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.pipelines.video_common import step_fetch_transcript


@pytest.fixture
def config(tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
    )


def test_step_fetch_transcript_writes_metadata_from_segments(config):
    """Synthesize-only fetch should write minimal metadata for density gate."""
    slug = "channel_video"
    task = {"id": "t1", "slug": slug}
    merged = {
        "segments": [
            {"start": 0.0, "end": 120.5, "text": "hello"},
            {"start": 120.5, "end": 300.0, "text": "world"},
        ],
    }
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text(json.dumps(merged))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = step_fetch_transcript(task, config)

    assert result == str(merged_path)
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["duration"] == 300.0


def test_step_fetch_transcript_metadata_uses_top_level_duration(config):
    slug = "channel_video"
    task = {"id": "t1", "slug": slug}
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text(json.dumps({"duration": 90.0, "segments": []}))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        step_fetch_transcript(task, config)

    meta = json.loads((Path(config.temp_dir) / f"{slug}_metadata.json").read_text())
    assert meta["duration"] == 90.0
