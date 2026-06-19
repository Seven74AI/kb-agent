"""Tests for the Instagram pipeline — reel video and carousel post extraction."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/root/kb-agent")

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMResponse
from agent.pipelines.instagram import (
    _validate_ig_cookies,
    _is_music_only,
    _format_transcript_for_llm,
    _slug_path,
    step_ig_reel_validate,
    step_ig_reel_download,
    step_ig_reel_audio,
    step_ig_reel_diarize,
    step_ig_reel_transcribe,
    step_ig_reel_merge,
    step_ig_reel_synthesize,
    step_ig_carousel_metadata,
    step_ig_carousel_extract,
    step_ig_carousel_synthesize,
    step_ig_quality_gate,
    step_ig_cleanup,
    step_ig_minio_upload,
    step_ig_git_push,
    IG_COOKIES_FILE,
    CAROUSEL_SLIDE_CAP,
    MUSIC_ONLY_CHAR_THRESHOLD,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Config with temp paths for testing."""
    vault_path = tmp_path / "vault"
    temp_dir = tmp_path / "tmp"
    vault_path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    (vault_path / "Knowledge base" / "instagram").mkdir(parents=True, exist_ok=True)

    return Config(
        kb_agent_dir=Path("/root/kb-agent"),
        vault_path=vault_path,
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        ig_carousel_min_size=100,  # Low for testing
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
    )


@pytest.fixture
def reel_task():
    return {
        "id": "test-ig-reel-12345",
        "source_url": "https://www.instagram.com/reel/ABC123/",
        "content_type": "instagram_reel",
        "slug": "instagram.com_reel_ABC123",
    }


@pytest.fixture
def post_task():
    return {
        "id": "test-ig-post-12345",
        "source_url": "https://www.instagram.com/p/DEF456/",
        "content_type": "instagram_post",
        "slug": "instagram.com_p_DEF456",
    }


# ── Helper Tests ─────────────────────────────────────────────────────────────

class TestHelpers:
    """Tests for internal helper functions."""

    def test_is_music_only_below_threshold(self):
        trans = {"segments": [
            {"text": "🎵"},
            {"text": "🎶"},
        ]}
        assert _is_music_only(trans) is True

    def test_is_music_only_above_threshold(self):
        trans = {"segments": [
            {"text": "Hello everyone, welcome to my Instagram reel where I talk about AI."},
        ]}
        assert _is_music_only(trans) is False

    def test_is_music_only_empty(self):
        trans = {"segments": []}
        assert _is_music_only(trans) is True

    def test_slug_path(self):
        assert "instagram.com_reel_ABC123" in _slug_path("instagram.com/reel/ABC123")
        assert "http://" not in _slug_path("http://example.com/p?q=1")

    def test_format_transcript_for_llm(self):
        merged = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello"},
                {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "World"},
            ]
        }
        text = _format_transcript_for_llm(merged)
        assert "SPEAKER_00: Hello" in text
        assert "SPEAKER_01: World" in text


# ── Cookie Validation ────────────────────────────────────────────────────────

class TestCookieValidation:
    """Tests for Instagram cookie validation."""

    @patch("agent.pipelines.instagram.Path.exists")
    @patch("agent.pipelines.instagram.Path.read_text")
    def test_validate_cookies_valid(self, mock_read, mock_exists, config, reel_task):
        mock_exists.return_value = True
        mock_read.return_value = "sessionid=abc123; domain=.instagram.com"

        result = step_ig_reel_validate(reel_task, config)
        assert result == "cookies_valid"

    @patch("agent.pipelines.instagram.Path.exists")
    def test_validate_cookies_missing_file(self, mock_exists, config, reel_task):
        mock_exists.return_value = False

        with pytest.raises(StepError) as exc:
            step_ig_reel_validate(reel_task, config)
        assert exc.value.error_type == ErrorType.PAUSE

    @patch("agent.pipelines.instagram.Path.exists")
    @patch("agent.pipelines.instagram.Path.read_text")
    def test_validate_cookies_no_sessionid(self, mock_read, mock_exists, config, reel_task):
        mock_exists.return_value = True
        mock_read.return_value = "ds_user_id=123"

        with pytest.raises(StepError) as exc:
            step_ig_reel_validate(reel_task, config)
        assert exc.value.error_type == ErrorType.PAUSE


# ── Reel Steps ───────────────────────────────────────────────────────────────

class TestReelDownload:
    """Tests for step_ig_reel_download."""

    @patch("agent.pipelines.instagram._validate_ig_cookies")
    @patch("subprocess.run")
    @patch("agent.pipelines.instagram.Path.exists")
    def test_download_success(self, mock_exists, mock_run, mock_validate, config, reel_task):
        mock_validate.return_value = None  # No exception = cookies valid

        # Create fake downloaded file
        def fake_run(*args, **kwargs):
            output_path = Path(config.temp_dir) / "instagram.com_reel_ABC123_reel.mp4"
            output_path.write_text("fake video data" * 100)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run
        mock_exists.return_value = True

        result = step_ig_reel_download(reel_task, config)
        assert "_reel.mp4" in result or "_reel" in result


class TestReelAudio:
    """Tests for step_ig_reel_audio."""

    @patch("subprocess.run")
    def test_audio_extraction_success(self, mock_run, config, reel_task):
        # Create fake reel video
        video_path = Path(config.temp_dir) / "instagram.com_reel_ABC123_reel.mp4"
        video_path.write_text("fake video")

        def fake_run(*args, **kwargs):
            cmd_str = str(args[0])
            if "16000" in cmd_str:
                Path(config.temp_dir / "instagram.com_reel_ABC123_16k.wav").write_text("fake 16k")
            elif "8000" in cmd_str:
                Path(config.temp_dir / "instagram.com_reel_ABC123_8k.wav").write_text("fake 8k")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_ig_reel_audio(reel_task, config)
        data = json.loads(result)
        assert "16k.wav" in data["wav_16k"]
        assert "8k.wav" in data["wav_8k"]


class TestReelMerge:
    """Tests for step_ig_reel_merge with music-only detection."""

    def test_merge_normal(self, config, reel_task):
        diar = {"segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]}
        trans = {"segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello world, this is a real reel.", "speaker": "SPEAKER_00"},
        ]}

        Path(config.temp_dir / "instagram.com_reel_ABC123_diarization.json").write_text(json.dumps(diar))
        Path(config.temp_dir / "instagram.com_reel_ABC123_transcript.json").write_text(json.dumps(trans))

        result = step_ig_reel_merge(reel_task, config)
        data = json.loads(result)
        assert data["music_only"] is False
        assert data["speakers"] == 1

        # Verify merged file has music_only flag
        merged = json.loads(Path(config.temp_dir / "instagram.com_reel_ABC123_merged.json").read_text())
        assert merged["music_only"] is False

    def test_merge_music_only(self, config, reel_task):
        diar = {"segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]}
        trans = {"segments": [
            {"start": 0.0, "end": 5.0, "text": "🎵", "speaker": "SPEAKER_00"},
        ]}

        Path(config.temp_dir / "instagram.com_reel_ABC123_diarization.json").write_text(json.dumps(diar))
        Path(config.temp_dir / "instagram.com_reel_ABC123_transcript.json").write_text(json.dumps(trans))

        result = step_ig_reel_merge(reel_task, config)
        data = json.loads(result)
        assert data["music_only"] is True


class TestReelSynthesis:
    """Tests for step_ig_reel_synthesize."""

    @patch("agent.pipelines.instagram.get_llm_client")
    def test_synthesize_normal(self, mock_get_llm, config, reel_task):
        merged = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello everyone!"},
            ],
            "music_only": False,
        }
        Path(config.temp_dir / "instagram.com_reel_ABC123_merged.json").write_text(json.dumps(merged))

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="""---
date: 2026-01-01
source: instagram.com
confidence: plausible
tags: [instagram, reel]
---

# Test Reel
## Summary
Good content.""")
        mock_get_llm.return_value = mock_llm

        result = step_ig_reel_synthesize(reel_task, config)
        assert result.endswith(".md")
        assert Path(result).exists()

    @patch("agent.pipelines.instagram.get_llm_client")
    def test_synthesize_music_only(self, mock_get_llm, config, reel_task):
        merged = {
            "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "🎵"}],
            "music_only": True,
        }
        Path(config.temp_dir / "instagram.com_reel_ABC123_merged.json").write_text(json.dumps(merged))

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="""---
date: 2026-01-01
source: instagram.com
confidence: low
tags: [instagram, reel, music]
---

# Music Reel Note
## Summary
Music-only reel.""")
        mock_get_llm.return_value = mock_llm

        result = step_ig_reel_synthesize(reel_task, config)
        assert result.endswith(".md")


# ── Carousel Steps ───────────────────────────────────────────────────────────

class TestCarouselMetadata:
    """Tests for step_ig_carousel_metadata."""

    @patch("agent.pipelines.instagram._validate_ig_cookies")
    @patch("subprocess.run")
    def test_metadata_success(self, mock_run, mock_validate, config, post_task):
        mock_validate.return_value = None

        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.stdout = "Test Post Title||testuser||42||5"
            else:
                result.stdout = "This is a carousel caption."
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_ig_carousel_metadata(post_task, config)
        data = json.loads(Path(result).read_text())
        assert data["uploader"] == "testuser"
        assert data["playlist_count"] == 5
        assert "carousel caption" in data.get("caption", "")


class TestCarouselExtract:
    """Tests for step_ig_carousel_extract."""

    @patch("subprocess.run")
    def test_extract_success(self, mock_run, config, post_task):
        slides = json.dumps([
            {"slide": 1, "alt_text": "An AI diagram showing neural networks", "src": "https://cdn.ig/img1.jpg"},
            {"slide": 2, "alt_text": "A chart of AI growth statistics", "src": "https://cdn.ig/img2.jpg"},
        ])

        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = slides
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_ig_carousel_extract(post_task, config)
        data = json.loads(Path(result).read_text())
        assert data["slide_count"] == 2
        assert len(data["slides"]) == 2
        assert "Hard cap" in data["cap_note"]


class TestCarouselSynthesis:
    """Tests for step_ig_carousel_synthesize."""

    @patch("agent.pipelines.instagram.get_llm_client")
    def test_synthesize_success(self, mock_get_llm, config, post_task):
        # Create carousel slides
        slides = {
            "slides": [
                {"slide": 1, "alt_text": "AI diagram", "src": "https://cdn/img1.jpg"},
                {"slide": 2, "alt_text": "Statistics chart", "src": "https://cdn/img2.jpg"},
            ],
            "slide_count": 2,
            "url": post_task["source_url"],
        }
        Path(config.temp_dir / "instagram.com_p_DEF456_carousel_slides.json").write_text(json.dumps(slides))

        # Optional metadata
        meta = {"caption": "Check out these AI stats!", "uploader": "testuser", "like_count": 100}
        Path(config.temp_dir / "instagram.com_p_DEF456_carousel_meta.json").write_text(json.dumps(meta))

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="""---
date: 2026-01-01
source: instagram.com (@testuser)
confidence: plausible
tags: [instagram, carousel]
---

# Test Carousel
## Summary
Great slides.""")
        mock_get_llm.return_value = mock_llm

        result = step_ig_carousel_synthesize(post_task, config)
        assert result.endswith(".md")
        assert Path(result).exists()


# ── Quality Gate ─────────────────────────────────────────────────────────────

class TestQualityGate:
    """Tests for step_ig_quality_gate."""

    def test_quality_gate_carousel_pass(self, config, post_task):
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{post_task['slug']}.md"
        content = "## Slide-by-Slide Analysis\n" * 50  # 2000+ chars
        note_path.write_text(content)

        result = step_ig_quality_gate(post_task, config)
        data = json.loads(result)
        assert data["passed"] is True

    def test_quality_gate_carousel_missing_section(self, config, post_task):
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{post_task['slug']}.md"
        note_path.write_text("## Summary\nContent\n" * 50)

        result = step_ig_quality_gate(post_task, config)
        data = json.loads(result)
        assert data["passed"] is False
        assert any("Slide-by-Slide" in f for f in data.get("failures", []))


# ── Cleanup ─────────────────────────────────────────────────────────────────

class TestCleanup:
    """Tests for step_ig_cleanup."""

    def test_cleanup_removes_files(self, config, reel_task):
        files = [
            "instagram.com_reel_ABC123_reel.mp4",
            "instagram.com_reel_ABC123_16k.wav",
            "instagram.com_reel_ABC123_8k.wav",
        ]
        for f in files:
            Path(config.temp_dir / f).write_text("x")

        result = step_ig_cleanup(reel_task, config)
        assert "Cleaned up:" in result

        for f in files:
            assert not Path(config.temp_dir / f).exists()


# ── MinIO ────────────────────────────────────────────────────────────────────

class TestMinIO:
    """Tests for step_ig_minio_upload."""

    @patch("subprocess.run")
    def test_minio_upload_success(self, mock_run, config, reel_task):
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{reel_task['slug']}.md"
        note_path.write_text("""---
date: 2026-01-01
source: instagram.com
minio: <filled by archival step>
---
# Content""")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = step_ig_minio_upload(reel_task, config)
        assert "minio://" in result

        content = note_path.read_text()
        assert "minio: minio://" in content


# ── Git Push ─────────────────────────────────────────────────────────────────

class TestGitPush:
    """Tests for step_ig_git_push."""

    @patch("subprocess.run")
    def test_git_push_success(self, mock_run, config, reel_task):
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{reel_task['slug']}.md"
        note_path.write_text("# Content")

        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "commit def456abc"
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_ig_git_push(reel_task, config)
        assert result == "def456abc"


# ── Pipeline Registration ───────────────────────────────────────────────────

class TestPipelineRegistration:
    """Verify pipeline registration."""

    def test_instagram_pipelines_registered(self):
        from agent.pipelines import PIPELINES
        assert "instagram_reel" in PIPELINES
        assert "instagram_post" in PIPELINES

    def test_instagram_reel_pipeline_steps(self):
        from agent.pipelines import PIPELINES
        steps = PIPELINES["instagram_reel"]
        step_names = [s.name for s in steps]
        assert "ig_reel_validate" in step_names
        assert "ig_reel_download" in step_names
        assert "ig_reel_audio" in step_names
        assert "ig_reel_diarize" in step_names
        assert "ig_reel_transcribe" in step_names
        assert "ig_reel_merge" in step_names
        assert "ig_reel_synthesize" in step_names
        assert "ig_git_push" in step_names

    def test_instagram_post_pipeline_steps(self):
        from agent.pipelines import PIPELINES
        steps = PIPELINES["instagram_post"]
        step_names = [s.name for s in steps]
        assert "ig_reel_validate" in step_names
        assert "ig_carousel_metadata" in step_names
        assert "ig_carousel_extract" in step_names
        assert "ig_carousel_synthesize" in step_names
        assert "ig_git_push" in step_names


class TestMusicOnlyDetection:
    """Verify music-only detection edge cases."""

    def test_music_only_emoji_only(self):
        trans = {"segments": [{"text": "🎵🎶🎵"}]}
        assert _is_music_only(trans) is True

    def test_music_only_empty_segments(self):
        trans = {"segments": []}
        assert _is_music_only(trans) is True

    def test_not_music_only_substantial_text(self):
        trans = {"segments": [
            {"text": "In this video I will explain the three key principles of effective leadership."},
        ]}
        assert _is_music_only(trans) is False

    def test_carousel_slide_cap_constant(self):
        assert CAROUSEL_SLIDE_CAP == 2, "Carousel slide cap must be 2 per spec"
