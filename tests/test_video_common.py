"""Tests for shared video pipeline helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.pipelines.video_common import (
    step_fetch_transcript,
    upload_note_archive,
    upload_video_artifacts,
)

TOKENIZED = "https://cdn.example.com/spoken.mp3?token=secret&expires=999"


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


@patch("agent.archival.upload_file_to_minio")
def test_upload_video_artifacts_redacts_merged_segment_text(mock_upload, config):
    slug = "vid_slug"
    merged = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": f"visit {TOKENIZED} now"},
        ],
    }
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text(json.dumps(merged), encoding="utf-8")
    video_path = Path(config.temp_dir) / f"{slug}.mp4"
    video_path.write_bytes(b"")

    upload_video_artifacts(slug, config, video_path)

    on_disk = json.loads(merged_path.read_text(encoding="utf-8"))
    assert "token=secret" not in on_disk["segments"][0]["text"]
    assert "https://cdn.example.com/spoken.mp3" in on_disk["segments"][0]["text"]
    transcript_uploads = [
        c for c in mock_upload.call_args_list if "transcripts/" in str(c)
    ]
    assert transcript_uploads


@patch("agent.archival.upload_file_to_minio", return_value="articles/slug.md")
def test_upload_note_archive_redacts_note_body(mock_upload, config):
    slug = "ig_post"
    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        f"""---
source_files:
  raw: <filled by archival step>
---

Mention {TOKENIZED} in the body.
""",
        encoding="utf-8",
    )
    upload_note_archive(slug, config, note_path, "instagram")
    written = note_path.read_text(encoding="utf-8")
    assert "token=secret" not in written
    assert "https://cdn.example.com/spoken.mp3" in written
    assert "raw: articles/slug.md" in written


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
