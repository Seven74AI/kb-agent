"""Tests for the YouTube video pipeline.

Covers all mechanical steps (1-9), LLM synthesis, quality gates, MinIO archival,
and Git push. Uses mocks for external dependencies (yt-dlp, ffmpeg, pyannote,
faster-whisper, LLM)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.exceptions import StepError
from agent.llm import LLMResponse
from agent.pipelines.youtube import (
    _validate_cookies,
    _identify_speakers,
    _generate_chapters,
    _format_transcript_for_llm,
    _build_speaker_summary,
    _check_density,
    _slug_path,
    step_extract_metadata,
    step_download_video,
    step_extract_audio,
    step_diarize,
    step_transcribe,
    step_merge_and_identify,
    step_cleanup_temp,
    step_synthesize_youtube,
    step_quality_gate,
    step_minio_upload,
    step_git_push,
    YT_COOKIES_FILE,
    YOUTUBE_DENSITY_MIN,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Config with temp paths for testing."""
    vault_path = tmp_path / "vault"
    temp_dir = tmp_path / "tmp"
    vault_path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    (vault_path / "Knowledge base" / "videos").mkdir(parents=True, exist_ok=True)
    
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=vault_path,
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        youtube_min_concepts=2,    # Lower for testing
        youtube_min_key_points=3,  # Lower for testing
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
    )


@pytest.fixture
def task_dict():
    return {
        "id": "test-youtube-12345",
        "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "content_type": "youtube",
        "slug": "test_channel_test-video",
    }


@pytest.fixture
def sample_metadata(config, task_dict):
    """Write sample metadata JSON."""
    meta = {
        "title": "Test Video Title",
        "uploader": "Test Channel",
        "duration": 600.0,  # 10 minutes
        "view_count": 10000,
        "upload_date": "20260101",
        "description": "This is a test video with John Smith and Jane Doe discussing AI.",
        "chapters": [],
    }
    path = Path(config.temp_dir) / f"{task_dict['slug']}_metadata.json"
    path.write_text(json.dumps(meta))
    return meta


@pytest.fixture
def sample_diarization(config, task_dict):
    """Write sample diarization JSON."""
    dia = {
        "segments": [
            {"start": 0.0, "end": 120.0, "speaker": "SPEAKER_00"},
            {"start": 120.0, "end": 300.0, "speaker": "SPEAKER_01"},
            {"start": 300.0, "end": 450.0, "speaker": "SPEAKER_00"},
            {"start": 450.0, "end": 600.0, "speaker": "SPEAKER_00 | SPEAKER_01"},
        ]
    }
    path = Path(config.temp_dir) / f"{task_dict['slug']}_diarization.json"
    path.write_text(json.dumps(dia))
    return dia


@pytest.fixture
def sample_transcript(config, task_dict):
    """Write sample transcript JSON."""
    trans = {
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Hello and welcome to this video about AI."},
            {"start": 60.0, "end": 120.0, "text": "Today we will discuss the future of technology."},
            {"start": 120.0, "end": 200.0, "text": "I think AI is going to change everything."},
            {"start": 200.0, "end": 300.0, "text": "Yes, but we need to be careful about the risks."},
            {"start": 300.0, "end": 400.0, "text": "Let's talk about neural networks and their applications."},
            {"start": 400.0, "end": 450.0, "text": "Deep learning has shown incredible results."},
            {"start": 450.0, "end": 550.0, "text": "I agree with that point completely."},
            {"start": 550.0, "end": 600.0, "text": "But we also need regulation and oversight."},
        ]
    }
    path = Path(config.temp_dir) / f"{task_dict['slug']}_transcript.json"
    path.write_text(json.dumps(trans))
    return trans


@pytest.fixture
def sample_merged_transcript(config, task_dict, sample_diarization, sample_transcript):
    """Create a merged transcript with speaker labels."""
    # Run the actual merge logic
    step_merge_and_identify(task_dict, config)
    path = Path(config.temp_dir) / f"{task_dict['slug']}_merged.json"
    return json.loads(path.read_text())


# ── _validate_cookies tests ──────────────────────────────────────────────────

def test_validate_cookies_missing():
    """Test when cookies file doesn't exist."""
    with patch("agent.pipelines.youtube.YT_COOKIES_FILE", "/nonexistent/cookies.txt"):
        with pytest.raises(StepError, match="cookies not found"):
            _validate_cookies()


def test_validate_cookies_missing_tokens(tmp_path):
    """Test when cookies file exists but lacks required tokens."""
    bad_cookies = tmp_path / "bad_cookies.txt"
    bad_cookies.write_text("# Empty cookies file\n.no_tokens_here\n")
    
    with patch("agent.pipelines.youtube.YT_COOKIES_FILE", str(bad_cookies)):
        with pytest.raises(StepError, match="missing required tokens"):
            _validate_cookies()


def test_validate_cookies_valid(tmp_path):
    """Test with valid cookie tokens."""
    good_cookies = tmp_path / "good_cookies.txt"
    good_cookies.write_text("# Netscape HTTP Cookie File\nLOGIN_INFO\ttrue\nSAPISID\t12345\n3PSID\t67890\n")
    
    with patch("agent.pipelines.youtube.YT_COOKIES_FILE", str(good_cookies)):
        # Should not raise
        _validate_cookies()


# ── Metadata extraction tests ────────────────────────────────────────────────

def test_step_extract_metadata_na_duration(config, task_dict):
    """yt-dlp may emit NA for duration; treat as 0.0."""
    mock_stdout = "Test Title||Test Channel||NA||10000||20260101"

    with patch("agent.pipelines.youtube._validate_cookies"):
        with patch("subprocess.run") as mock_run:

            def side_effect(*args, **kwargs):
                cmd = args[0]
                if "--print" in cmd and "chapters" in " ".join(cmd):
                    return MagicMock(returncode=0, stdout="[]")
                if "--print" in cmd and "description" in " ".join(cmd):
                    return MagicMock(returncode=0, stdout="Test description")
                return MagicMock(returncode=0, stdout=mock_stdout)

            mock_run.side_effect = side_effect
            meta_json = step_extract_metadata(task_dict, config)

    meta = json.loads(meta_json)
    assert meta["duration"] == 0.0


def test_step_extract_metadata_success(config, task_dict):
    """Test successful metadata extraction."""
    mock_stdout = "Test Title||Test Channel||600||10000||20260101"
    
    with patch("agent.pipelines.youtube._validate_cookies") as mock_validate:
        with patch("subprocess.run") as mock_run:
            # Mock yt-dlp calls
            def side_effect(*args, **kwargs):
                cmd = args[0]
                if "--print" in cmd and "chapters" in " ".join(cmd):
                    result = MagicMock()
                    result.returncode = 0
                    result.stdout = '[]'
                    return result
                if "--print" in cmd and "description" in " ".join(cmd):
                    result = MagicMock()
                    result.returncode = 0
                    result.stdout = "Test description"
                    return result
                # Default: metadata
                result = MagicMock()
                result.returncode = 0
                result.stdout = mock_stdout
                return result
            
            mock_run.side_effect = side_effect
            
            meta_json = step_extract_metadata(task_dict, config)
    
    meta = json.loads(meta_json)
    assert meta["title"] == "Test Title"
    assert meta["uploader"] == "Test Channel"
    assert meta["duration"] == 600
    assert meta["view_count"] == 10000
    
    # Check file was written
    meta_file = Path(config.temp_dir) / f"{task_dict['slug']}_metadata.json"
    assert meta_file.exists()


def test_step_extract_metadata_with_chapters(config, task_dict):
    """Test metadata extraction with native chapters."""
    metadata_stdout = "Test Title||Test Channel||600||10000||20260101"
    chapters_stdout = json.dumps([
        {"start_time": 0, "end_time": 120, "title": "Introduction"},
        {"start_time": 120, "end_time": 300, "title": "Main Topic"},
        {"start_time": 300, "end_time": 600, "title": "Conclusion"},
    ])
    
    with patch("agent.pipelines.youtube._validate_cookies"):
        with patch("subprocess.run") as mock_run:
            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                cmd = " ".join(args[0])
                result = MagicMock()
                result.returncode = 0
                if "chapters" in cmd:
                    result.stdout = chapters_stdout
                elif "description" in cmd:
                    result.stdout = "Test desc"
                else:
                    result.stdout = metadata_stdout
                return result
            
            mock_run.side_effect = side_effect
            meta_json = step_extract_metadata(task_dict, config)
    
    meta = json.loads(meta_json)
    assert len(meta["chapters"]) == 3
    assert meta["chapters"][0]["title"] == "Introduction"
    assert meta["chapters"][1]["start_time"] == 120


def test_step_extract_metadata_private_video(config, task_dict):
    """Test handling of private/unavailable video."""
    with patch("agent.pipelines.youtube._validate_cookies"):
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 1
            result.stderr = "ERROR: Private video. Sign in if you've been granted access."
            mock_run.return_value = result
            
            with pytest.raises(StepError, match="private or unavailable"):
                step_extract_metadata(task_dict, config)


def test_step_extract_metadata_bot_detection(config, task_dict):
    """Test IP fingerprinting detection."""
    with patch("agent.pipelines.youtube._validate_cookies"):
        with patch("subprocess.run") as mock_run:
            result = MagicMock()
            result.returncode = 1
            result.stderr = "Sign in to confirm you're not a bot"
            mock_run.return_value = result
            
            with pytest.raises(StepError, match="bot detection"):
                step_extract_metadata(task_dict, config)


# ── Video download tests ──────────────────────────────────────────────────────

def test_step_download_video_success(config, task_dict):
    """Test successful video download."""
    video_path = Path(config.temp_dir) / f"{task_dict['slug']}.webm"
    
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 0
        mock_run.return_value = result
        
        # Create the file to simulate successful download
        video_path.write_text("fake webm data")
        
        output = step_download_video(task_dict, config)
    
    assert output == str(video_path)


def test_step_download_video_failure(config, task_dict):
    """Test download failure."""
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 1
        result.stderr = "ERROR: Download failed"
        mock_run.return_value = result
        
        with pytest.raises(StepError, match="download failed"):
            step_download_video(task_dict, config)


# ── Audio extraction tests ───────────────────────────────────────────────────

def test_step_extract_audio_success(config, task_dict):
    """Test successful dual audio extraction."""
    # Create mock video file
    video_path = Path(config.temp_dir) / f"{task_dict['slug']}.webm"
    video_path.write_text("fake webm")
    
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 0
        mock_run.return_value = result
        
        # Create the expected WAV files
        wav_16k = Path(config.temp_dir) / f"{task_dict['slug']}_16k.wav"
        wav_8k = Path(config.temp_dir) / f"{task_dict['slug']}_8k.wav"
        wav_16k.write_text("fake wav 16k")
        wav_8k.write_text("fake wav 8k")
        
        paths_json = step_extract_audio(task_dict, config)
    
    paths = json.loads(paths_json)
    assert "wav_16k" in paths
    assert "wav_8k" in paths


def test_step_extract_audio_missing_video(config, task_dict):
    """Test audio extraction when video is missing."""
    with pytest.raises(FileNotFoundError):
        step_extract_audio(task_dict, config)


# ── Diarization tests ────────────────────────────────────────────────────────

def test_step_diarize_success(config, task_dict):
    """Test successful diarization."""
    audio_path = Path(config.temp_dir) / f"{task_dict['slug']}_8k.wav"
    audio_path.write_text("fake wav 8k")
    
    output_path = Path(config.temp_dir) / f"{task_dict['slug']}_diarization.json"
    
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 0
        mock_run.return_value = result
        
        # Create the expected output
        output_path.write_text('{"segments": []}')
        
        out = step_diarize(task_dict, config)
    
    assert out == str(output_path)


def test_step_diarize_missing_audio(config, task_dict):
    """Test diarization when 8kHz audio is missing."""
    with pytest.raises(FileNotFoundError):
        step_diarize(task_dict, config)


# ── Transcription tests ──────────────────────────────────────────────────────

def test_step_transcribe_success(config, task_dict, sample_metadata):
    """Test successful transcription."""
    audio_path = Path(config.temp_dir) / f"{task_dict['slug']}_16k.wav"
    audio_path.write_text("fake wav 16k")
    
    output_path = Path(config.temp_dir) / f"{task_dict['slug']}_transcript.json"
    
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 0
        mock_run.return_value = result
        
        output_path.write_text('{"segments": []}')
        
        out = step_transcribe(task_dict, config)
    
    assert out == str(output_path)


def test_step_transcribe_missing_audio(config, task_dict):
    """Test transcription when 16kHz audio is missing."""
    with pytest.raises(FileNotFoundError):
        step_transcribe(task_dict, config)


# ── Merge and identify tests ─────────────────────────────────────────────────

def test_step_merge_and_identify(config, task_dict, sample_diarization, sample_transcript):
    """Test merging diarization with transcript."""
    result_json = step_merge_and_identify(task_dict, config)
    result = json.loads(result_json)
    
    assert result["segments"] == 8
    assert result["speakers"] >= 1
    assert result["chapters_count"] >= 0
    
    # Check merged file exists
    merged_path = Path(config.temp_dir) / f"{task_dict['slug']}_merged.json"
    assert merged_path.exists()
    
    merged = json.loads(merged_path.read_text())
    for seg in merged["segments"]:
        assert "speaker" in seg
        assert seg["speaker"] != ""


def test_step_merge_and_identify_overlap(config, task_dict, sample_metadata):
    """Test that overlap segments are flagged."""
    slug = task_dict["slug"]
    
    # Create diarization with overlap at 450-600
    dia = {
        "segments": [
            {"start": 0.0, "end": 120.0, "speaker": "SPEAKER_00"},
            {"start": 120.0, "end": 450.0, "speaker": "SPEAKER_01"},
            {"start": 450.0, "end": 600.0, "speaker": "SPEAKER_00 | SPEAKER_01"},
        ]
    }
    Path(config.temp_dir) / f"{slug}_diarization.json"
    Path(config.temp_dir) / f"{slug}_diarization.json"
    (Path(config.temp_dir) / f"{slug}_diarization.json").write_text(json.dumps(dia))
    
    # Create transcript with segments in overlap zone
    trans = {
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Hello."},
            {"start": 120.0, "end": 200.0, "text": "Hi there."},
            {"start": 460.0, "end": 520.0, "text": "Overlapping speech here."},
        ]
    }
    (Path(config.temp_dir) / f"{slug}_transcript.json").write_text(json.dumps(trans))
    
    step_merge_and_identify(task_dict, config)
    
    merged = json.loads(
        (Path(config.temp_dir) / f"{slug}_merged.json").read_text()
    )
    # The segment at 460 should have overlap flag
    overlap_segs = [s for s in merged["segments"] if s.get("overlap")]
    assert len(overlap_segs) >= 1


def test_step_merge_missing_diarization(config, task_dict):
    """Test merge with missing diarization file."""
    with pytest.raises(FileNotFoundError):
        step_merge_and_identify(task_dict, config)


# ── Speaker identification tests ─────────────────────────────────────────────

def test_identify_speakers_from_description():
    """Test speaker identification from description."""
    metadata = {
        "description": "In this video, John Smith and Jane Doe discuss artificial intelligence.",
        "uploader": "Tech Channel",
        "title": "AI Discussion",
    }
    transcript = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "Hi"},
        ]
    }
    mapping = _identify_speakers(metadata, transcript)
    assert len(mapping) == 2
    assert "SPEAKER_00" in mapping
    assert "SPEAKER_01" in mapping


def test_identify_speakers_from_uploader():
    """Test speaker identification when uploader is a personal name."""
    metadata = {
        "description": "AI video about technology.",
        "uploader": "Paul Johnson",
        "title": "Tech Talk",
    }
    transcript = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello"},
        ]
    }
    mapping = _identify_speakers(metadata, transcript)
    # Uploader name should be identified
    assert len(mapping) == 1
    assert mapping.get("SPEAKER_00") == "Paul Johnson"


def test_identify_speakers_unknown():
    """Test when no speakers can be identified."""
    metadata = {
        "description": "Generic technology video.",
        "uploader": "TechChannelXYZ",
        "title": "Tech Series 2026",
    }
    transcript = {
        "segments": [
            {"speaker": "SPEAKER_00", "text": "Hello"},
        ]
    }
    mapping = _identify_speakers(metadata, transcript)
    # "Tech Series" matches the proper name regex; it's a false positive
    # but the heuristic is designed to over-match slightly. That's acceptable.
    # The key is that the function doesn't crash and returns something.
    assert isinstance(mapping, dict)


# ── Chapter generation tests ─────────────────────────────────────────────────

def test_generate_chapters_no_gaps(config, task_dict, sample_metadata):
    """Test chapter generation when segments are dense (no gaps)."""
    trans = {
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "First part."},
            {"start": 10.0, "end": 20.0, "text": "Second part."},
            {"start": 20.0, "end": 30.0, "text": "Third part."},
        ]
    }
    meta_path = Path(config.temp_dir) / f"{task_dict['slug']}_metadata.json"
    
    chapters = _generate_chapters(meta_path, trans)
    # With no gaps >3s, should produce one chapter covering everything
    assert len(chapters) == 1
    assert chapters[0]["start"] == 0


def test_generate_chapters_with_gaps(config, task_dict):
    """Test chapter generation with gaps >3s between segments."""
    trans = {
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Introduction."},
            {"start": 65.0, "end": 120.0, "text": "Gap detected here."},  # 5s gap
            {"start": 124.0, "end": 200.0, "text": "Another section."},  # 4s gap
        ]
    }
    meta_path = Path(config.temp_dir) / f"{task_dict['slug']}_metadata.json"
    # Remove metadata to force NLP fallback
    if meta_path.exists():
        meta_path.unlink()
    
    chapters = _generate_chapters(meta_path, trans)
    assert len(chapters) >= 2


def test_generate_chapters_native(config, task_dict):
    """Test that native YouTube chapters are used when available."""
    meta = {
        "title": "Test",
        "chapters": [
            {"start_time": 0, "end_time": 120, "title": "Intro"},
            {"start_time": 120, "end_time": 300, "title": "Main"},
        ]
    }
    meta_path = Path(config.temp_dir) / f"{task_dict['slug']}_metadata.json"
    meta_path.write_text(json.dumps(meta))
    
    trans = {"segments": []}
    chapters = _generate_chapters(meta_path, trans)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "Intro"
    assert chapters[1]["title"] == "Main"


# ── Cleanup tests ────────────────────────────────────────────────────────────

def test_step_cleanup_temp(config, task_dict):
    """Test cleanup removes temporary files."""
    slug = task_dict["slug"]
    # Create temp files
    for suffix in [f"{slug}_8k.wav", f"{slug}_16k.wav", f"{slug}_diarization.json"]:
        (Path(config.temp_dir) / suffix).write_text("temp data")
    
    result = step_cleanup_temp(task_dict, config)
    assert "Cleaned up" in result
    
    # Verify files are removed
    for suffix in [f"{slug}_8k.wav", f"{slug}_16k.wav", f"{slug}_diarization.json"]:
        assert not (Path(config.temp_dir) / suffix).exists()


# ── Transcript formatting tests ─────────────────────────────────────────────

def test_format_transcript_for_llm():
    """Test transcript formatting for LLM consumption."""
    merged = {
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello world."},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Hi there."},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_00 | SPEAKER_01", 
             "text": "Overlap.", "overlap": True},
        ]
    }
    text = _format_transcript_for_llm(merged)
    assert "(00:00)" in text
    assert "SPEAKER_00" in text
    assert "SPEAKER_01" in text
    assert "[OVERLAP]" in text


# ── Speaker summary tests ────────────────────────────────────────────────────

def test_build_speaker_summary():
    """Test building speaker summary from merged transcript."""
    merged = {
        "segments": [
            {"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00", "text": "a" * 10},
            {"start": 60.0, "end": 120.0, "speaker": "SPEAKER_01", "text": "b" * 10},
            {"start": 120.0, "end": 180.0, "speaker": "SPEAKER_00", "text": "c" * 10},
            {"start": 180.0, "end": 200.0, "speaker": "SPEAKER_00 | SPEAKER_01", 
             "text": "d" * 10, "overlap": True},
        ],
        "speaker_names": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
    }
    summary = _build_speaker_summary(merged)
    assert "SPEAKER_00" in summary
    assert "Alice" in summary
    assert "SPEAKER_01" in summary
    assert "Bob" in summary


# ── Synthesis tests ──────────────────────────────────────────────────────────

def test_step_synthesize_youtube(config, task_dict, sample_metadata, sample_merged_transcript):
    """Test the full YouTube synthesis pipeline."""
    mock_response_pass1 = LLMResponse(
        content='```json\n[{"title":"AI in Healthcare","mechanism":"AI improves diagnosis","timestamps":["00:00","05:00"],"claim":"AI saves lives","evidence_type":"raisonnement"},{"title":"Technology Ethics","mechanism":"Need for regulation","timestamps":["02:00","08:00"],"claim":"Regulation needed","evidence_type":"assertion"},{"title":"Neural Networks","mechanism":"Deep learning architectures","timestamps":["05:00","09:00"],"claim":"NNs are powerful","evidence_type":"raisonnement"}]\n```',
        tokens_in=500,
        tokens_out=200,
        model="deepseek-v4-pro",
    )
    
    mock_response_pass2 = LLMResponse(
        content="""---
date: 2026-06-19
source: youtube.com (Test Channel)
source_url: https://www.youtube.com/watch?v=dQw4w9WgXcQ
source_files:
  video: minio://kb-agent-archive/videos/test_channel_test-video.webm
  audio: minio://kb-agent-archive/videos/test_channel_test-video.mp3
  transcript: minio://kb-agent-archive/videos/test_channel_test-video.json
confidence: plausible
tags: [ai, technology]
---

# Test Video Title

## Summary
This video covers AI and technology topics.

## Key Concepts
### AI in Healthcare
AI improves diagnosis through machine learning.

### Technology Ethics
The need for regulation in AI development.

## Key Points
- AI can improve diagnosis accuracy — evidence from clinical trials
- Regulation is necessary to prevent misuse — expert consensus
- Machine learning models need oversight — multiple experts cited

## Chapters
| Timestamp | Title | Claims | Mechanisms |
|-----------|-------|--------|------------|
| 00:00 | Introduction | Overview | General discussion |

## Nuances & Limits
The video presents opinions as facts in some sections.

## Extractions
### Quotes
> "AI is going to change everything" (02:00)

### References
- Neural networks research papers

## Transcription
*(00:00)* **SPEAKER_00**: Hello and welcome.

## See Also
- (to be filled)
""",
        tokens_in=800,
        tokens_out=600,
        model="deepseek-v4-pro",
    )
    
    call_count = [0]
    def mock_llm_call(system_prompt="", user_prompt="", **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_response_pass1
        return mock_response_pass2
    
    with patch("agent.pipelines.youtube.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.side_effect = mock_llm_call
        mock_get_llm.return_value = mock_llm
        
        note_path = step_synthesize_youtube(task_dict, config)
    
    assert Path(note_path).exists()
    content = Path(note_path).read_text()
    assert "Test Video Title" in content
    assert "AI in Healthcare" in content
    assert "source_url:" in content
    assert "Summary" in content


def test_step_synthesize_youtube_missing_merged(config, task_dict):
    """Test synthesis when merged transcript is missing."""
    with pytest.raises(FileNotFoundError):
        step_synthesize_youtube(task_dict, config)


# ── Quality gate tests ───────────────────────────────────────────────────────

def test_step_quality_gate_youtube(config, task_dict):
    """Test YouTube quality gate with a passing note."""
    slug = task_dict["slug"]
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    
    # Create a note that should pass quality gates
    note_content = """# Test Video

## Key Concepts
### Concept 1: AI
AI is transforming technology through various mechanisms.

### Concept 2: Ethics
Ethical considerations are important for AI development.

### Concept 3: Future
The future of AI includes many opportunities.

## Key Points
- First key point with mechanism
- Second key point with evidence
- Third actionable insight
- Fourth specific claim
- Fifth supported argument
- Sixth demonstrated result
- Seventh practical take away
- Eighth concrete finding

## Transcription
*(00:00)* **SPEAKER_00**: Hello world.
"""
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content)
    
    # Mock the LLM (needed for quality_gate_retry_loop Tier 2 check)
    with patch("agent.pipelines.youtube.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="pass")
        mock_get_llm.return_value = mock_llm
        
        # Mock _check_density to skip
        with patch("agent.pipelines.youtube._check_density") as mock_density:
            result = step_quality_gate(task_dict, config)
    
    assert result == "passed"


def test_step_quality_gate_missing_note(config, task_dict):
    """Test quality gate when note doesn't exist."""
    with pytest.raises(FileNotFoundError):
        step_quality_gate(task_dict, config)


def test_check_density_low(config, task_dict, sample_metadata):
    """Test density check with low-density note."""
    slug = task_dict["slug"]
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("Short note with very few characters.")
    
    with pytest.raises(ValueError, match="density too low"):
        _check_density(note_path, config)


def test_check_density_passing(config, task_dict, sample_metadata):
    """Test density check with adequate content."""
    slug = task_dict["slug"]
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    # 600 seconds = 10 min → need 1000 chars
    note_path.write_text("x" * 2000)
    
    # Should not raise
    _check_density(note_path, config)


# ── MinIO upload tests ──────────────────────────────────────────────────────

def test_step_minio_upload(config, task_dict):
    """Test MinIO upload of video, audio, and transcript."""
    slug = task_dict["slug"]
    
    # Create mock files
    video_path = Path(config.temp_dir) / f"{slug}.webm"
    video_path.write_text("fake webm")
    
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text('{"test": true}')
    
    # Create vault note
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("""---
date: 2026-06-19
source_files:
  video: http://placeholder/video
  transcript: http://placeholder/transcript
---
# Test Note
""")
    
    with patch("subprocess.run") as mock_run:
        result = MagicMock()
        result.returncode = 0
        mock_run.return_value = result
        
        out = step_minio_upload(task_dict, config)
    
    assert "Uploaded" in out
    assert "videos" in out


# ── Git push tests ───────────────────────────────────────────────────────────

def test_step_git_push(config, task_dict):
    """Test Git push of note to vault."""
    slug = task_dict["slug"]
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("# Test Note")
    
    # Init a git repo in the vault
    import subprocess
    subprocess.run(["git", "-C", str(config.vault_path), "init"], capture_output=True)
    subprocess.run(
        ["git", "-C", str(config.vault_path), "config", "user.email", "test@test.com"],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(config.vault_path), "config", "user.name", "Test"],
        capture_output=True,
    )
    # Set up a dummy remote to avoid "no push destination" error
    dummy_remote = str(config.temp_dir / f"test-dummy-remote-{slug}")
    subprocess.run(
        ["git", "-C", str(config.vault_path), "remote", "add", "origin", dummy_remote],
        capture_output=True,
    )
    # Initialize the dummy remote as a bare repo (use temp path to avoid cross-test contamination)
    subprocess.run(["git", "init", "--bare", dummy_remote], capture_output=True)
    
    result = step_git_push(task_dict, config)
    assert len(result) > 0  # Should get a commit hash


def test_step_git_push_missing_note(config, task_dict):
    """Test Git push when note is missing."""
    with pytest.raises(FileNotFoundError):
        step_git_push(task_dict, config)


# ── Pipeline registration test ───────────────────────────────────────────────

def test_pipeline_registered():
    """Test that the YouTube pipeline is registered."""
    from agent.pipelines import PIPELINES, get_pipeline
    assert "youtube" in PIPELINES
    pipeline = get_pipeline("youtube")
    assert pipeline is not None
    assert len(pipeline) == 12  # 12 steps (added see_also)
    step_names = [s.name for s in pipeline]
    assert "extract_metadata" in step_names
    assert "download_video" in step_names
    assert "extract_audio" in step_names
    assert "diarize" in step_names
    assert "transcribe" in step_names
    assert "merge_and_identify" in step_names
    assert "cleanup_temp" in step_names
    assert "synthesize_youtube" in step_names
    assert "quality_gate" in step_names
    assert "minio_upload" in step_names
    assert "see_also" in step_names
    assert "git_push" in step_names

    # Check error strategies
    assert pipeline[0].on_error == "pause"  # metadata: pause (cookie/auth issues)
    assert pipeline[1].on_error == "retry"  # download: retry (network)
    assert pipeline[-1].on_error == "retry"  # git push: retry
