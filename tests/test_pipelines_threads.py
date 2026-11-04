"""Tests for the Threads pipeline — text and video extraction."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMResponse
from agent.pipelines.threads import (
    _has_video_versions,
    _extract_video_url,
    _extract_all_captions,
    _extract_author,
    _is_reply_post,
    _is_login_wall,
    _format_transcript_for_llm,
    _slug_path,
    step_threads_detect,
    step_threads_extract_text,
    step_threads_extract_video,
    step_threads_extract_audio,
    step_threads_diarize,
    step_threads_transcribe,
    step_threads_merge,
    step_threads_video_synthesize,
    step_threads_text_synthesize,
    step_threads_quality_gate,
    step_threads_cleanup,
    step_threads_minio_upload,
    step_threads_git_push,
    GOOGLEBOT_UA,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Config with temp paths for testing."""
    vault_path = tmp_path / "vault"
    temp_dir = tmp_path / "tmp"
    vault_path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    (vault_path / "Knowledge base" / "threads").mkdir(parents=True, exist_ok=True)

    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=vault_path,
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        threads_text_min_quotes=2,
        threads_text_min_size=100,  # Low for testing
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
    )


@pytest.fixture
def task_dict():
    return {
        "id": "test-threads-12345",
        "source_url": "https://www.threads.net/@testuser/post/ABC123",
        "content_type": "threads_text",
        "slug": "testuser_abc123",
    }


@pytest.fixture
def mock_threads_html_text():
    """Simulated Threads page HTML for a text post."""
    return '''<!DOCTYPE html>
<html><head><meta property="og:description" content="Just sharing my thoughts on AI..."/></head>
<body>
<script>window.__INITIAL_STATE__={"caption":{"text":"Just sharing my thoughts on AI and the future of work. What do you think?"},"username":"testuser","is_reply":false}</script>
<div class="_ab6c">Real content</div>
</body></html>'''


@pytest.fixture
def mock_threads_html_video():
    """Simulated Threads page HTML for a video post."""
    return '''<!DOCTYPE html>
<html><head></head>
<body>
<script>window.__INITIAL_STATE__={"caption":{"text":"Check out this video!"},"username":"testuser","is_reply":false,"video_versions":[{"url":"https://video.cdn.threads.net/test.mp4","height":1080}]}</script>
<div class="_ab6c">Video content</div>
</body></html>'''


@pytest.fixture
def mock_threads_html_reply():
    """Simulated Threads page HTML for a reply post."""
    return '''<!DOCTYPE html>
<html><head></head>
<body>
<script>window.__INITIAL_STATE__={"caption":{"text":"I completely agree with this take."},"username":"replier","is_reply":true,"parent_post":{"code":"XYZ789"}}</script>
<div class="_ab6c">Reply content</div>
</body></html>'''


@pytest.fixture
def mock_threads_html_login_wall():
    """Simulated Threads login wall page."""
    return '''<!DOCTYPE html>
<html><head><meta property="og:description" content="Follow @testuser on Threads to see their posts."/></head>
<body><div class="login-wall">Log in to continue</div></body></html>'''


# ── Helper Tests ─────────────────────────────────────────────────────────────

class TestHelpers:
    """Tests for internal helper functions."""

    def test_has_video_versions_true(self):
        html = '{"video_versions":[{"url":"x","height":720}]}'
        assert _has_video_versions(html) is True

    def test_has_video_versions_false(self):
        html = '{"caption":{"text":"hello"}}'
        assert _has_video_versions(html) is False

    def test_extract_video_url(self):
        html = '{"video_versions":[{"url":"https://v.cdn/v.mp4","height":1080},{"url":"https://v.cdn/v_720.mp4","height":720}]}'
        url = _extract_video_url(html)
        assert url == "https://v.cdn/v.mp4"  # Highest quality

    def test_extract_video_url_nested_brackets_in_url(self):
        html = (
            '{"video_versions":[{"url":"https://v.cdn/v.mp4?token=[nested]","height":720},'
            '{"url":"https://v.cdn/v_hd.mp4","height":1080}]}'
        )
        url = _extract_video_url(html)
        assert url == "https://v.cdn/v_hd.mp4"

    def test_extract_video_url_none(self):
        html = '{"caption":{"text":"hello"}}'
        assert _extract_video_url(html) is None

    def test_extract_all_captions(self):
        html = '{"caption":{"text":"First caption text."}} and {"text":"Second caption text."}'
        captions = _extract_all_captions(html)
        assert captions == ["First caption text."]

    def test_extract_all_captions_multiple_caption_objects(self):
        html = (
            '{"caption":{"text":"First caption."}}'
            ',"thread_items":[{"post":{"caption":{"text":"Second caption."}}}]'
        )
        captions = _extract_all_captions(html)
        assert "First caption." in captions
        assert "Second caption." in captions

    def test_extract_all_captions_empty(self):
        html = '<html><body>Nothing here</body></html>'
        captions = _extract_all_captions(html)
        assert captions == []

    def test_extract_author(self):
        html = '{"username":"testuser"}'
        assert _extract_author(html) == "testuser"

    def test_extract_author_unknown(self):
        html = '<html></html>'
        assert _extract_author(html) == "unknown"

    def test_is_reply_post_true(self):
        html = '{"is_reply":true}'
        assert _is_reply_post(html) is True

    def test_is_reply_post_false(self):
        html = '{"is_reply":false}'
        assert _is_reply_post(html) is False

    def test_is_login_wall_login_redirect(self):
        html = '<html>/login Threads</html>'
        assert _is_login_wall(html) is True

    def test_is_login_wall_follow_pattern(self):
        html = '<meta property="og:description" content="Follow @user on Threads"/>'
        assert _is_login_wall(html) is True

    def test_is_login_wall_false(self):
        html = '<meta property="og:description" content="Actual post content"/>'
        assert _is_login_wall(html) is False

    def test_slug_path(self):
        assert _slug_path("hello world") == "hello world"
        assert _slug_path("test?query=1") == "test_query_1"

    def test_format_transcript_for_llm(self):
        merged = {
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "Hi there"},
            ]
        }
        text = _format_transcript_for_llm(merged)
        assert "SPEAKER_00: Hello world" in text
        assert "SPEAKER_01: Hi there" in text


# ── Step Tests: Detection ────────────────────────────────────────────────────

class TestDetection:
    """Tests for step_threads_detect."""

    @patch("agent.pipelines.threads._curl_threads")
    def test_detect_text_post(self, mock_curl, config, task_dict):
        html = '{"caption":{"text":"hello"},"username":"testuser"}<div class="_ab6c">x</div>'
        mock_curl.return_value = html

        result = step_threads_detect(task_dict, config)
        data = json.loads(result)
        assert data["content_type"] == "threads_text"
        assert data["has_video"] is False

    @patch("agent.pipelines.threads._curl_threads")
    def test_detect_video_post(self, mock_curl, config, task_dict):
        html = '{"video_versions":[{"url":"x","height":720}],"caption":{"text":"video!"}}<div class="_ab6c">x</div>'
        mock_curl.return_value = html

        result = step_threads_detect(task_dict, config)
        data = json.loads(result)
        assert data["content_type"] == "threads_video"
        assert data["has_video"] is True

    @patch("agent.pipelines.threads._curl_threads")
    def test_detect_login_wall_pauses(self, mock_curl, config, task_dict):
        html = '<meta property="og:description" content="Follow @user on Threads"/>'
        mock_curl.return_value = html

        with pytest.raises(StepError) as exc:
            step_threads_detect(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE


# ── Step Tests: Text Extraction ──────────────────────────────────────────────

class TestTextExtraction:
    """Tests for step_threads_extract_text."""

    @patch("agent.pipelines.threads._curl_threads")
    def test_extract_text_basic(self, mock_curl, config, task_dict):
        html = '{"caption":{"text":"Hello world from Threads!"},"username":"testuser","is_reply":false}'
        mock_curl.return_value = html

        result = step_threads_extract_text(task_dict, config)
        data = json.loads(Path(result).read_text())
        assert data["author"] == "testuser"
        assert any("Hello world" in c for c in data["captions"])
        assert data["is_reply"] is False
        assert data["confidence"] == "plausible"

    @patch("agent.pipelines.threads._curl_threads")
    def test_extract_text_login_wall_pauses(self, mock_curl, config, task_dict):
        html = '<meta property="og:description" content="Follow @user on Threads"/>'
        mock_curl.return_value = html

        with pytest.raises(StepError) as exc:
            step_threads_extract_text(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE

    @patch("agent.pipelines.threads._curl_threads")
    def test_extract_text_reply_post(self, mock_curl, config, task_dict):
        html = '{"caption":{"text":"Great point!"},"username":"replier","is_reply":true,"parent_post":{"code":"XYZ789"}}'
        mock_curl.return_value = html

        result = step_threads_extract_text(task_dict, config)
        data = json.loads(Path(result).read_text())
        assert data["is_reply"] is True
        assert data["parent_url"] == "https://www.threads.net/t/XYZ789"


# ── Step Tests: Video Extraction ─────────────────────────────────────────────

class TestVideoExtraction:
    """Tests for step_threads_extract_video."""

    @patch("agent.pipelines.threads._curl_threads")
    @patch("subprocess.run")
    def test_extract_video_success(self, mock_run, mock_curl, config, task_dict):
        html = '{"video_versions":[{"url":"https://v.cdn/v.mp4","height":1080}],"caption":{"text":"x"},"username":"x"}<div class="_ab6c">x</div>'
        mock_curl.return_value = html

        # Simulate successful curl download — actually create the output file
        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            # Create output file to satisfy existence check
            safe_slug = "testuser_abc123"
            output_path = Path(config.temp_dir) / f"{safe_slug}_threads_video.mp4"
            output_path.write_text("fake video data" * 100)
            return result
        mock_run.side_effect = fake_run

        result = step_threads_extract_video(task_dict, config)
        assert "_threads_video.mp4" in result
        assert Path(result).exists()

    @patch("agent.pipelines.threads._curl_threads")
    def test_extract_video_no_video_in_page(self, mock_curl, config, task_dict):
        html = '{"caption":{"text":"text only"},"username":"x"}<div class="_ab6c">x</div>'
        mock_curl.return_value = html

        with pytest.raises(StepError) as exc:
            step_threads_extract_video(task_dict, config)
        assert exc.value.error_type == ErrorType.FAIL


# ── Step Tests: Audio Extraction ─────────────────────────────────────────────

class TestAudioExtraction:
    """Tests for step_threads_extract_audio."""

    @patch("subprocess.run")
    def test_extract_audio_success(self, mock_run, config, task_dict, tmp_path):
        # Create fake video file
        video_path = Path(config.temp_dir) / "testuser_abc123_threads_video.mp4"
        video_path.write_text("fake video data")

        def fake_run(*args, **kwargs):
            # Create fake WAV files
            cmd_str = str(args[0])
            if "-ar" in cmd_str and "16000" in cmd_str:
                Path(config.temp_dir / "testuser_abc123_16k.wav").write_text("fake 16k wav")
            elif "-ar" in cmd_str and "8000" in cmd_str:
                Path(config.temp_dir / "testuser_abc123_8k.wav").write_text("fake 8k wav")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_threads_extract_audio(task_dict, config)
        data = json.loads(result)
        assert "16k.wav" in data["wav_16k"]
        assert "8k.wav" in data["wav_8k"]


# ── Step Tests: Diarization ──────────────────────────────────────────────────

class TestDiarization:
    """Tests for step_threads_diarize."""

    @patch("subprocess.run")
    def test_diarize_success(self, mock_run, config, task_dict):
        # Create fake 8k WAV
        Path(config.temp_dir / "testuser_abc123_8k.wav").write_text("fake audio")

        def fake_run(*args, **kwargs):
            # Create fake diarization output
            output_path = Path(args[0][3])  # 4th arg is output JSON path
            output_path.write_text(json.dumps({"segments": [{"speaker": "SPEAKER_00"}]}))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_threads_diarize(task_dict, config)
        assert "diarization.json" in result

    def test_diarize_missing_audio(self, config, task_dict):
        with pytest.raises(FileNotFoundError):
            step_threads_diarize(task_dict, config)


# ── Step Tests: Transcription ────────────────────────────────────────────────

class TestTranscription:
    """Tests for step_threads_transcribe."""

    @patch("subprocess.run")
    def test_transcribe_success(self, mock_run, config, task_dict):
        # Create fake 16k WAV
        Path(config.temp_dir / "testuser_abc123_16k.wav").write_text("fake audio")

        def fake_run(*args, **kwargs):
            output_path = Path(args[0][3])
            output_path.write_text(json.dumps({"segments": [{"text": "hello"}]}))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_threads_transcribe(task_dict, config)
        assert "transcript.json" in result

    def test_transcribe_missing_audio(self, config, task_dict):
        with pytest.raises(FileNotFoundError):
            step_threads_transcribe(task_dict, config)


# ── Step Tests: Merge ────────────────────────────────────────────────────────

class TestMerge:
    """Tests for step_threads_merge."""

    def test_merge_success(self, config, task_dict):
        # Create fake input files
        diar = {"segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]}
        trans = {"segments": [
            {"start": 0.0, "end": 4.0, "text": "Hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 9.0, "text": "World", "speaker": "SPEAKER_01"},
        ]}

        Path(config.temp_dir / "testuser_abc123_diarization.json").write_text(json.dumps(diar))
        Path(config.temp_dir / "testuser_abc123_transcript.json").write_text(json.dumps(trans))

        result = step_threads_merge(task_dict, config)
        data = json.loads(result)
        assert data["segments"] == 2
        assert data["speakers"] == 2

        # Verify merged file exists
        merged_path = Path(config.temp_dir) / "testuser_abc123_merged.json"
        assert merged_path.exists()


# ── Step Tests: LLM Synthesis ────────────────────────────────────────────────

class TestSynthesis:
    """Tests for LLM synthesis steps."""

    @patch("agent.pipelines.threads.get_llm_client")
    def test_text_synthesize(self, mock_get_llm, config, task_dict):
        # Create fake raw extraction
        raw_path = Path(config.temp_dir) / f"{task_dict['slug']}_threads_raw.json"
        raw_path.write_text(json.dumps({
            "captions": ["Just sharing thoughts on AI."],
            "author": "testuser",
            "is_reply": False,
            "confidence": "plausible",
        }))

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="""---
date: 2026-01-01
source: threads.net (@testuser)
source_url: https://test
confidence: plausible
tags: [threads]
---

# Test Note
## Summary
Great content.""")
        mock_get_llm.return_value = mock_llm

        result = step_threads_text_synthesize(task_dict, config)
        assert result.endswith(".md")
        assert Path(result).exists()

    @patch("agent.pipelines.threads.get_llm_client")
    def test_video_synthesize(self, mock_get_llm, config, task_dict):
        # Create fake merged transcript
        merged = {"segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hello world"},
        ]}
        merged_path = Path(config.temp_dir) / f"{task_dict['slug']}_merged.json"
        merged_path.write_text(json.dumps(merged))

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content="""---
date: 2026-01-01
source: threads.net (@testuser)
source_url: https://test
confidence: plausible
tags: [threads, video]
---

# Test Video Note
## Summary
Video content.""")
        mock_get_llm.return_value = mock_llm

        result = step_threads_video_synthesize(task_dict, config)
        assert result.endswith(".md")
        assert Path(result).exists()


# ── Step Tests: Quality Gate ─────────────────────────────────────────────────

class TestQualityGate:
    """Tests for step_threads_quality_gate."""

    def test_quality_gate_text_pass(self, config, task_dict):
        # Create a passing note
        note_path = Path(config.vault_path) / "Knowledge base" / "threads" / f"{task_dict['slug']}.md"
        note_content = ("## Summary\n> Quote 1\n> Quote 2\nSome analysis text.\n" * 80)  # 2000+ chars with 2+ quotes
        note_path.write_text(note_content)

        result = step_threads_quality_gate(task_dict, config)
        data = json.loads(result)
        assert data["passed"] is True

    def test_quality_gate_text_fail_too_short(self, config, task_dict):
        note_path = Path(config.vault_path) / "Knowledge base" / "threads" / f"{task_dict['slug']}.md"
        note_path.write_text("Too short")

        with pytest.raises(StepError, match="Threads text quality gate failed"):
            step_threads_quality_gate(task_dict, config)


# ── Step Tests: Cleanup ──────────────────────────────────────────────────────

class TestCleanup:
    """Tests for step_threads_cleanup."""

    def test_cleanup_removes_files(self, config, task_dict):
        # Create temp files
        for ext in ["_threads_video.mp4", "_16k.wav", "_8k.wav"]:
            Path(config.temp_dir / f"testuser_abc123{ext}").write_text("x")

        result = step_threads_cleanup(task_dict, config)
        assert "Cleaned up:" in result
        # Verify files removed
        for ext in ["_threads_video.mp4", "_16k.wav", "_8k.wav"]:
            assert not Path(config.temp_dir / f"testuser_abc123{ext}").exists()


# ── Step Tests: MinIO ────────────────────────────────────────────────────────

class TestMinIO:
    """Tests for step_threads_minio_upload."""

    @patch("subprocess.run")
    def test_minio_upload_success(self, mock_run, config, task_dict):
        # Create note
        note_path = Path(config.vault_path) / "Knowledge base" / "threads" / f"{task_dict['slug']}.md"
        note_path.write_text("""---
date: 2026-01-01
source: threads.net
minio: <filled by archival step>
---
# Content""")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = step_threads_minio_upload(task_dict, config)
        assert "http://vmi3304846.tail5c02a1.ts.net:9000/" in result

        # Verify frontmatter updated
        content = note_path.read_text()
        assert "minio: http://vmi3304846.tail5c02a1.ts.net:9000/" in content
        assert "<filled by archival step>" not in content


# ── Step Tests: Git Push ─────────────────────────────────────────────────────

class TestGitPush:
    """Tests for step_threads_git_push."""

    @patch("subprocess.run")
    def test_git_push_success(self, mock_run, config, task_dict):
        # Create note
        note_path = Path(config.vault_path) / "Knowledge base" / "threads" / f"{task_dict['slug']}.md"
        note_path.write_text("# Content")

        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "commit abc123def"
            result.stderr = ""
            return result
        mock_run.side_effect = fake_run

        result = step_threads_git_push(task_dict, config)
        assert result == "abc123def"


class TestPipelineRegistration:
    """Verify pipeline registration."""

    def test_threads_pipelines_registered(self):
        from agent.pipelines import PIPELINES
        assert "threads_text" in PIPELINES
        assert "threads_video" in PIPELINES

    def test_threads_text_pipeline_steps(self):
        from agent.pipelines import PIPELINES
        steps = PIPELINES["threads_text"]
        step_names = [s.name for s in steps]
        assert "threads_detect" in step_names
        assert "threads_extract_text" in step_names
        assert "threads_text_synthesize" in step_names
        assert "threads_git_push" in step_names

    def test_threads_video_pipeline_steps(self):
        from agent.pipelines import PIPELINES
        steps = PIPELINES["threads_video"]
        step_names = [s.name for s in steps]
        assert "threads_detect" in step_names
        assert "threads_extract_video" in step_names
        assert "threads_diarize" in step_names
        assert "threads_transcribe" in step_names
        assert "threads_video_synthesize" in step_names
