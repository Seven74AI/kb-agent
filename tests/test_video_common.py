"""Tests for shared video pipeline helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.pipelines.video_common import (
    _synthesize_archived_keys,
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
        storage_bucket="kb-agent-archive",    )


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


@patch("agent.pipelines.video_common.subprocess.run")
@patch("agent.archival.upload_object")
def test_upload_video_artifacts_redacts_merged_segment_text(
    mock_upload, mock_run, config
):
    mock_run.return_value = MagicMock(returncode=1)
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

    upload_video_artifacts(slug, slug, config, video_path)

    on_disk = json.loads(merged_path.read_text(encoding="utf-8"))
    assert "token=secret" not in on_disk["segments"][0]["text"]
    assert "https://cdn.example.com/spoken.mp3" in on_disk["segments"][0]["text"]
    transcript_uploads = [
        c for c in mock_upload.call_args_list if "transcripts/" in str(c)
    ]
    assert transcript_uploads


@patch("agent.pipelines.video_common.subprocess.run")
@patch("agent.archival.upload_object")
def test_upload_video_artifacts_non_batch_keys_match_slug(mock_upload, mock_run, config):
    """Non-batch mode: object storage keys use the slug (same as legacy single-arg behavior)."""

    def _fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"mp3")
        return MagicMock(returncode=0)

    mock_run.side_effect = _fake_ffmpeg
    slug = "channel_video"
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    video_path = Path(config.temp_dir) / f"{slug}.mp4"
    video_path.write_bytes(b"mp4")

    keys = upload_video_artifacts(slug, slug, config, video_path)

    assert keys == [
        "videos/channel_video.mp4",
        "videos/channel_video.mp3",
        "transcripts/channel_video.json",
    ]
    storage_paths = [str(c.args[1]) for c in mock_upload.call_args_list]
    assert all(f"/videos/{slug}" in p or f"/transcripts/{slug}" in p for p in storage_paths)
    assert (config.temp_dir / f"{slug}.mp3").exists()


@patch("agent.pipelines.video_common.subprocess.run")
@patch("agent.archival.upload_object")
def test_upload_video_artifacts_batch_stem_vs_storage_slug(mock_upload, mock_run, config):
    """Batch ingest: temp files use task id; object storage keys use human slug."""

    def _fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"mp3")
        return MagicMock(returncode=0)

    mock_run.side_effect = _fake_ffmpeg
    task_id = "batch-ingest-uuid-9999"
    slug = "human_display_slug"
    merged_path = Path(config.temp_dir) / f"{task_id}_merged.json"
    merged_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
    video_path = Path(config.temp_dir) / f"{task_id}.mp4"
    video_path.write_bytes(b"mp4")

    keys = upload_video_artifacts(task_id, slug, config, video_path)

    assert keys == [
        f"videos/{slug}.mp4",
        f"videos/{slug}.mp3",
        f"transcripts/{slug}.json",
    ]
    for call in mock_upload.call_args_list:
        dest = str(call.args[1])
        assert task_id not in dest
        assert slug in dest
    assert (config.temp_dir / f"{task_id}.mp3").exists()
    assert not (config.temp_dir / f"{slug}.mp3").exists()


@patch("agent.archival.upload_object", return_value="articles/slug.md")
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


def test_step_fetch_transcript_falls_back_to_legacy_reels_prefix(config):
    slug = "legacy_reel"
    task = {"id": "t1", "slug": slug}
    merged = {"segments": [{"start": 0.0, "end": 1.0, "text": "ok"}]}
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "/reels/" in cmd[2]:
            merged_path = Path(cmd[3])
            merged_path.write_text(json.dumps(merged), encoding="utf-8")
            return MagicMock(returncode=0, stderr="")
        return MagicMock(returncode=1, stderr="not in transcripts")

    config.storage_alias = "tigris"
    with patch("subprocess.run", side_effect=fake_run):
        result = step_fetch_transcript(task, config)

    assert len(calls) == 2
    assert f"tigris/{config.storage_bucket}/transcripts/{slug}.json" in calls[0][2]
    assert f"tigris/{config.storage_bucket}/reels/{slug}.json" in calls[1][2]
    assert result == str(Path(config.temp_dir) / f"{slug}_merged.json")


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


@pytest.mark.parametrize(
    "content_type",
    ["youtube_synthesize", "rumble_synthesize", "substack_video_synthesize", "bunny_stream_synthesize"],
)
def test_synthesize_archived_keys_omit_mp3(content_type):
    slug = "my_video_slug"
    task = {"id": "t1", "slug": slug, "content_type": content_type}
    keys = _synthesize_archived_keys(task)
    assert keys == [f"videos/{slug}.mp4", f"transcripts/{slug}.json"]
    assert not any(k.endswith(".mp3") for k in keys)
