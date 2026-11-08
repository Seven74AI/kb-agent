"""Tests for the Instagram pipeline — reel video and carousel post extraction."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMResponse
from agent.pipelines.instagram import (
    _validate_ig_cookies,
    _ytdlp_subprocess_env,
    _fetch_ig_caption,
    _fetch_ig_comment,
    _is_music_only,
    _format_transcript_for_llm,
    _slug_path,
    step_ig_reel_validate,
    step_ig_reel_download,
    step_ig_reel_merge,
    step_ig_reel_synthesize,
    step_ig_carousel_metadata,
    step_ig_carousel_extract,
    step_ig_carousel_synthesize,
    step_ig_quality_gate,
    step_ig_cleanup,
    step_ig_minio_upload,
    step_ig_git_push,
    CAROUSEL_SLIDE_CAP,
    MUSIC_ONLY_CHAR_THRESHOLD,
)
from agent.pipelines.video_common import step_extract_audio

METADATA_URL = "http://169.254.169.254/latest/meta-data/"

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
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=vault_path,
        temp_dir=temp_dir,
        cookie_dir=tmp_path / "cookies",
        ig_carousel_min_size=100,  # Low for testing
        minio_bucket="kb-agent-archive",
        minio_endpoint="http://localhost:9000",
        minio_tailscale_base="http://minio.test/base",
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
    def test_download_rejects_metadata_ip(self, mock_run, mock_validate, config, reel_task):
        reel_task["source_url"] = METADATA_URL
        with pytest.raises(StepError, match="not allowed") as exc:
            step_ig_reel_download(reel_task, config)
        assert exc.value.error_type == ErrorType.FAIL
        mock_run.assert_not_called()

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
    """Tests for step_extract_audio (shared)."""

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

        result = step_extract_audio(reel_task, config)
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
            {"start": 0.0, "end": 5.0, "text": "Hello world, this is a real reel with actual speech content that exceeds the music-only threshold of fifty characters.", "speaker": "SPEAKER_00"},
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
    @patch("agent.pipelines.instagram._fetch_ig_caption", return_value="")
    def test_synthesize_normal(self, mock_fetch_caption, mock_get_llm, config, reel_task):
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
    @patch("agent.pipelines.instagram._fetch_ig_caption", return_value="This caption carries the actual claim about the herbal remedy.")
    def test_synthesize_music_only(self, mock_fetch_caption, mock_get_llm, config, reel_task):
        merged = {
            "segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "🎵"}],
            "music_only": True,
        }
        Path(config.temp_dir / "instagram.com_reel_ABC123_merged.json").write_text(json.dumps(merged))

        tokenized = "https://www.instagram.com/reel/ABC123/?igsh=tracking-secret"
        reel_task["source_url"] = tokenized

        mock_llm = MagicMock()
        mock_llm.call.return_value = LLMResponse(content=f"""---
date: 2026-01-01
source: instagram.com
source_url: {tokenized}
confidence: low
tags: [instagram, reel, music]
---

# Music Reel Note
## Summary
Music-only reel.""")
        mock_get_llm.return_value = mock_llm

        result = step_ig_reel_synthesize(reel_task, config)
        assert result.endswith(".md")

        system_prompt = mock_llm.call.call_args.kwargs.get("system_prompt", "")
        assert "Instagram reel (https://www.instagram.com/reel/ABC123/)" in system_prompt
        assert "igsh=" not in system_prompt

        # Caption must reach the LLM prompt (primary content for music-only reels)
        user_prompt = mock_llm.call.call_args.kwargs.get("user_prompt", "")
        assert "the actual claim about the herbal remedy" in user_prompt

        written = Path(result).read_text(encoding="utf-8")
        assert "igsh=" not in written.split("---")[1]


# ── Carousel Steps ───────────────────────────────────────────────────────────

class TestFetchIgCaptionSsrf:
    """SSRF guard on caption fetch helper."""

    @patch("subprocess.run")
    def test_fetch_caption_rejects_metadata_ip(self, mock_run, config):
        with pytest.raises(ValueError, match="not allowed"):
            _fetch_ig_caption(METADATA_URL, config)
        mock_run.assert_not_called()

    @patch("agent.pipelines.instagram.logger")
    @patch("subprocess.run")
    def test_fetch_caption_logs_scrubbed_warning_on_failure(self, mock_run, mock_log, config):
        mock_run.side_effect = RuntimeError(
            "yt-dlp failed for https://www.instagram.com/reel/X/?igsh=secret"
        )
        assert _fetch_ig_caption("https://www.instagram.com/reel/X/?igsh=secret", config) == ""
        mock_log.warning.assert_called_once()
        args = mock_log.warning.call_args[0]
        assert "igsh=secret" not in args[1]
        assert "https://www.instagram.com/reel/X/" in args[1]


class TestCarouselMetadata:
    """Tests for step_ig_carousel_metadata."""

    @patch("agent.pipelines.instagram._validate_ig_cookies")
    @patch("subprocess.run")
    def test_carousel_metadata_rejects_metadata_ip(self, mock_run, mock_validate, config, post_task):
        post_task["source_url"] = METADATA_URL
        with pytest.raises(StepError, match="not allowed") as exc:
            step_ig_carousel_metadata(post_task, config)
        assert exc.value.error_type == ErrorType.FAIL
        mock_run.assert_not_called()

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

    @patch("agent.pipelines.instagram._validate_ig_cookies")
    @patch("subprocess.run")
    def test_metadata_redacts_urls_in_caption(self, mock_run, mock_validate, config, post_task):
        mock_validate.return_value = None
        tracking = "https://link.example/promo?igsh=deadbeef&ref=track"
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.stdout = "Title||testuser||1||2"
            else:
                result.stdout = f"See more at {tracking}"
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = fake_run

        result = step_ig_carousel_metadata(post_task, config)
        data = json.loads(Path(result).read_text(encoding="utf-8"))
        caption = data.get("caption", "")
        assert "igsh=" not in caption
        assert "?" not in caption
        assert "https://link.example/promo" in caption

    @patch("agent.pipelines.instagram._validate_ig_cookies")
    @patch("subprocess.run")
    def test_metadata_redacts_urls_in_title_and_uploader(
        self, mock_run, mock_validate, config, post_task,
    ):
        mock_validate.return_value = None
        title_url = "https://bio.example/name?igsh=deadbeef"
        uploader_url = "https://profile.example/user?token=SECRET"
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.stdout = f"{title_url}||{uploader_url}||1||2"
            else:
                result.stdout = "caption"
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = fake_run

        result = step_ig_carousel_metadata(post_task, config)
        data = json.loads(Path(result).read_text(encoding="utf-8"))
        assert "?" not in data["title"]
        assert "igsh=" not in data["title"]
        assert "?" not in data["uploader"]
        assert "token=" not in data["uploader"]


class TestCarouselExtract:
    """Tests for step_ig_carousel_extract."""

    @patch("subprocess.run")
    def test_extract_rejects_metadata_ip(self, mock_run, config, post_task):
        post_task["source_url"] = METADATA_URL
        with pytest.raises(StepError, match="not allowed") as exc:
            step_ig_carousel_extract(post_task, config)
        assert exc.value.error_type == ErrorType.FAIL
        mock_run.assert_not_called()

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

    @patch("subprocess.run")
    def test_extract_redacts_tokenized_urls_in_alt_text(self, mock_run, config, post_task):
        tokenized = "https://promo.example/offers?jwt=SECRET&igsh=track"
        slides = json.dumps([
            {
                "slide": 1,
                "alt_text": f"Shop at {tokenized}",
                "src": "https://cdn.ig/img1.jpg?sig=HIDDEN",
            },
        ])

        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = slides
            result.stderr = ""
            return result

        mock_run.side_effect = fake_run
        result = step_ig_carousel_extract(post_task, config)
        data = json.loads(Path(result).read_text(encoding="utf-8"))
        alt = data["slides"][0]["alt_text"]
        assert "jwt=SECRET" not in alt
        assert "igsh=track" not in alt
        assert "https://promo.example/offers" in alt
        assert "src" not in data["slides"][0]

    @patch("subprocess.run")
    def test_extract_escapes_url_in_playwright_script(self, mock_run, config):
        task = {
            "id": "test-ig-post-12345",
            "source_url": 'https://www.instagram.com/p/DEF456/"); import os; os.system("touch /tmp/pwned") #',
            "content_type": "instagram_post",
            "slug": "instagram.com_p_DEF456",
        }
        captured = {}

        def fake_run(*args, **kwargs):
            script_path = Path(args[0][1])  # argv = ["python3", script_path]
            captured["script"] = script_path.read_text(encoding="utf-8")
            r = MagicMock()
            r.returncode = 0
            r.stdout = "[]"
            r.stderr = ""
            return r
        mock_run.side_effect = fake_run

        step_ig_carousel_extract(task, config)

        script = captured["script"]
        # the injected URL's closing quote is backslash-escaped — the payload
        # stays inside the string literal, no code breakout
        assert '\\"); import os' in script
        assert 'page.goto("https://www.instagram.com/p/DEF456/");' not in script

    @patch("subprocess.run")
    def test_extract_playwright_script_omits_tracking_query(self, mock_run, config, post_task):
        post_task["source_url"] = "https://www.instagram.com/p/ABC/?igsh=deadbeef"
        captured = {}

        def fake_run(*args, **kwargs):
            script_path = Path(args[0][1])
            captured["script"] = script_path.read_text(encoding="utf-8")
            r = MagicMock()
            r.returncode = 0
            r.stdout = "[]"
            r.stderr = ""
            return r

        mock_run.side_effect = fake_run
        step_ig_carousel_extract(post_task, config)
        script = captured["script"]
        assert "igsh" not in script
        assert "deadbeef" not in script
        assert "www.instagram.com/p/ABC/" in script

    @patch("subprocess.run")
    def test_extract_unlinks_playwright_script_on_failure(self, mock_run, config, post_task):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="timeout")
        script_path = Path(config.temp_dir) / f"{post_task['slug']}_playwright.py"
        with pytest.raises(StepError, match="Playwright carousel extraction failed"):
            step_ig_carousel_extract(post_task, config)
        assert not script_path.exists()

    @patch("subprocess.run")
    def test_extract_goto_failure_raises_no_carousel_artifact(self, mock_run, config, post_task):
        """page.goto failure must fail-closed: StepError, no thin carousel JSON on disk."""
        slides_path = Path(config.temp_dir) / f"{post_task['slug']}_carousel_slides.json"

        def fake_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = "[]"
            result.stderr = "Page.goto: Timeout 30000ms exceeded."
            return result

        mock_run.side_effect = fake_run

        with pytest.raises(StepError, match="Playwright carousel extraction failed") as exc:
            step_ig_carousel_extract(post_task, config)
        assert exc.value.error_type == ErrorType.RETRY
        assert not slides_path.exists()

    @patch("subprocess.run")
    def test_extract_playwright_script_does_not_swallow_goto_errors(self, mock_run, config, post_task):
        """Embedded script must not bare-except page.goto (navigation errors must propagate)."""
        captured = {}

        def fake_run(*args, **kwargs):
            script_path = Path(args[0][1])
            captured["script"] = script_path.read_text(encoding="utf-8")
            r = MagicMock()
            r.returncode = 0
            r.stdout = json.dumps([
                {"slide": 1, "alt_text": "alt", "src": "https://cdn.ig/x.jpg"},
            ])
            r.stderr = ""
            return r

        mock_run.side_effect = fake_run
        step_ig_carousel_extract(post_task, config)
        script = captured["script"]
        goto_pos = script.index("await page.goto(")
        chunk = script[goto_pos : goto_pos + 200]
        assert "except Exception:" not in chunk
        assert "pass" not in chunk.split("page.goto(")[-1].split("wait_for_timeout")[0]


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
        content = ("## Slide-by-Slide Analysis\nSlide content goes here with analysis.\n" * 80)  # 2000+ chars
        note_path.write_text(content)

        result = step_ig_quality_gate(post_task, config)
        data = json.loads(result)
        assert data["passed"] is True

    def test_quality_gate_carousel_missing_section(self, config, post_task):
        from agent.exceptions import StepError

        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{post_task['slug']}.md"
        note_path.write_text("## Summary\nContent\n" * 50)

        with pytest.raises(StepError, match="Slide-by-Slide"):
            step_ig_quality_gate(post_task, config)

    @patch("agent.pipelines.instagram.run_threads_video_quality_gate", return_value=(True, 1))
    def test_quality_gate_reel_synthesize_uses_key_point_gate(self, mock_gate, config, reel_task):
        # Re-synthesized reels carry content_type "instagram_reel_synthesize",
        # which must route to the key-point gate, NOT the carousel slide check.
        reel_task["content_type"] = "instagram_reel_synthesize"
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{reel_task['slug']}.md"
        note_path.write_text("## Post Description\n> caption\n\n## Key Points\n- a\n- b\n- c\n")

        result = step_ig_quality_gate(reel_task, config)
        data = json.loads(result)
        assert data["passed"] is True
        mock_gate.assert_called_once()


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

    def test_cleanup_removes_ig_comment_artifacts(self, config, reel_task):
        slug = reel_task["slug"]
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        comment_json = Path(config.temp_dir) / f"{slug}_ig_comment.json"
        comment_json.write_text("{}")
        conf = cookie_dir / ".curl-ig-comment-42.conf"
        conf.write_text('header = "Cookie: sessionid=x"\n')
        task = {
            **reel_task,
            "source_url": f"https://www.instagram.com/reel/X/?comment_id=42",
        }
        result = step_ig_cleanup(task, config)
        assert "ig_comment.json" in result
        assert conf.name in result
        assert not comment_json.exists()
        assert not conf.exists()

    @pytest.mark.parametrize("task_fixture", ["reel_task", "post_task"])
    def test_cleanup_removes_ytdlp_xdg_after_ytdlp_env(self, config, request, task_fixture):
        """Success and failure paths both leave ytdlp_xdg until ig_cleanup runs."""
        task = request.getfixturevalue(task_fixture)
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text("sessionid=1\n", encoding="utf-8")
        _ytdlp_subprocess_env(config)
        xdg_root = Path(config.temp_dir) / "ytdlp_xdg"
        assert xdg_root.is_dir()
        conf = xdg_root / "yt-dlp" / "config"
        assert conf.is_file()
        assert "ig_cookies.txt" in conf.read_text(encoding="utf-8")

        result = step_ig_cleanup(task, config)
        assert "ytdlp_xdg" in result
        assert not xdg_root.exists()

    def test_cleanup_removes_carousel_artifacts_success_path(self, config, post_task):
        from agent.pipelines.video_common import _artifact_stem

        stem = _artifact_stem(post_task)
        temp = config.temp_dir
        for name in (
            f"{stem}_playwright.py",
            f"{stem}_carousel_slides.json",
            f"{stem}_carousel_meta.json",
        ):
            Path(temp / name).write_text("{}", encoding="utf-8")
        _ytdlp_subprocess_env(config)

        result = step_ig_cleanup(post_task, config)
        assert "ytdlp_xdg" in result
        for name in (
            f"{stem}_playwright.py",
            f"{stem}_carousel_slides.json",
            f"{stem}_carousel_meta.json",
        ):
            assert not Path(temp / name).exists()

    def test_cleanup_removes_batch_id_keyed_carousel_sidecars(self, config, post_task):
        """Batch ingest writes id-keyed sidecars; cleanup must remove those, not slug-only decoys."""
        from agent.pipelines.video_common import _artifact_stem

        batch_task = {**post_task, "_batch_ingest": True}
        stem = _artifact_stem(batch_task)
        assert stem == batch_task["id"]
        temp = config.temp_dir
        Path(temp / f"{stem}_carousel_meta.json").write_text("{}", encoding="utf-8")
        slug_decoy = Path(temp / f"{post_task['slug']}_carousel_slides.json")
        slug_decoy.write_text("{}", encoding="utf-8")

        step_ig_cleanup(batch_task, config)
        assert not Path(temp / f"{stem}_carousel_meta.json").exists()
        assert slug_decoy.exists()

    def test_cleanup_removes_carousel_artifacts_failure_residual(self, config, post_task):
        """Failed carousel extract can leave playwright; ig_cleanup must remove it."""
        from agent.pipelines.video_common import _artifact_stem

        stem = _artifact_stem(post_task)
        script = Path(config.temp_dir) / f"{stem}_playwright.py"
        script.write_text("# failed extract leftover\n", encoding="utf-8")
        _ytdlp_subprocess_env(config)

        step_ig_cleanup(post_task, config)
        assert not script.exists()
        assert not (Path(config.temp_dir) / "ytdlp_xdg").exists()


# ── MinIO ────────────────────────────────────────────────────────────────────

class TestMinIO:
    """Tests for step_ig_minio_upload."""

    @patch("subprocess.run")
    def test_minio_upload_reel_source_archival(self, mock_run, config, reel_task):
        slug = reel_task["slug"]
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{slug}.md"
        note_path.write_text("""---
date: 2026-01-01
source: instagram.com
source_files:
  video: <filled by archival step>
---
# Content""")

        Path(config.temp_dir / f"{slug}_reel.mp4").write_bytes(b"video")
        Path(config.temp_dir / f"{slug}_merged.json").write_text("{}")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = step_ig_minio_upload(reel_task, config)
        assert result.startswith("Uploaded:")
        assert f"videos/{slug}.mp4" in result
        assert f"transcripts/{slug}.json" in result

        content = note_path.read_text()
        assert "source_files:" in content
        assert f"videos/{slug}.mp4" in content

    @patch("subprocess.run")
    def test_minio_upload_post_note_archival(self, mock_run, config, post_task):
        slug = post_task["slug"]
        note_path = Path(config.vault_path) / "Knowledge base" / "instagram" / f"{slug}.md"
        note_path.write_text("""---
date: 2026-01-01
source: instagram.com
raw: <filled by archival step>
---
# Content""")

        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = step_ig_minio_upload(post_task, config)
        assert result.startswith("instagram/")

        content = note_path.read_text()
        assert f"raw: {result}" in content
        assert "<filled by archival step>" not in content


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
        assert "extract_audio" in step_names
        assert "diarize" in step_names
        assert "transcribe" in step_names
        assert "merge_and_identify" in step_names
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
        assert "ig_cleanup" in step_names
        minio_idx = step_names.index("ig_minio_upload")
        cleanup_idx = step_names.index("ig_cleanup")
        assert cleanup_idx == minio_idx + 1
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


class TestIgCommentFetchSecurity:

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_uses_cookie_file_not_argv_header(self, mock_run, config, tmp_path):
        """Session cookies must not appear in curl argv (visible in ps)."""
        secret_session = "SUPER_SECRET_SESSIONID_VALUE_XYZ"
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        cookie_file = cookie_dir / "ig_cookies.txt"
        cookie_file.write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", secret_session,
            ]) + "\n"
            + "\t".join([
                ".instagram.com", "TRUE", "/", "FALSE", "0",
                "csrftoken", "csrf_test_token",
            ]) + "\n",
            encoding="utf-8",
        )

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "status": "ok",
                "comments": [{
                    "pk": 999888777,
                    "text": "Test comment body",
                    "created_at": 1700000000,
                    "user": {"username": "commenter"},
                }],
                "caption": {
                    "text": "Parent reel caption",
                    "user": {"username": "parent_user"},
                },
            }),
        )

        url = "https://www.instagram.com/reel/ShortCode1/?comment_id=999888777"
        _fetch_ig_comment(url, config)

        cmd = mock_run.call_args[0][0]
        argv_str = " ".join(cmd)
        assert "--max-redirs" in cmd
        assert cmd[cmd.index("--max-redirs") + 1] == "0"
        assert secret_session not in argv_str
        assert "csrf_test_token" not in argv_str
        assert "csrftoken" not in argv_str
        assert "-K" not in cmd
        assert "--config" in cmd
        assert mock_run.call_args.kwargs.get("input") is None
        assert "-b" not in cmd
        assert str(cookie_file) not in argv_str
        conf_path = config.cookie_dir / ".curl-ig-comment-999888777.conf"
        assert not conf_path.exists(), "credential sidecar must not persist after fetch"

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_rejects_path_traversal_comment_id(
        self, mock_run, config, tmp_path,
    ):
        """comment_id must not embed path separators (curl sidecar stays under cookie_dir)."""
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        escape_target = tmp_path / "tmp" / "x.conf"
        escape_target.parent.mkdir(parents=True, exist_ok=True)

        url = "https://www.instagram.com/reel/ShortCode1/?comment_id=../../../tmp/x"
        with pytest.raises(StepError, match="Invalid comment_id"):
            _fetch_ig_comment(url, config)

        mock_run.assert_not_called()
        assert not escape_target.exists()
        assert list(cookie_dir.glob(".curl-ig-comment-*.conf")) == []

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_curl_conf_stays_under_cookie_dir(
        self, mock_run, config,
    ):
        """A normal numeric comment_id writes the sidecar only inside cookie_dir."""
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "status": "ok",
                "comments": [{"pk": 42, "text": "hi", "user": {"username": "u"}}],
                "caption": {"text": "cap", "user": {"username": "p"}},
            }),
        )
        _fetch_ig_comment(
            "https://www.instagram.com/reel/ShortCode1/?comment_id=42",
            config,
        )
        conf_path = cookie_dir / ".curl-ig-comment-42.conf"
        assert not conf_path.exists(), "credential sidecar must not persist after fetch"

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_paginates_past_first_page(self, mock_run, config):
        """Deep comment_id values must not hard-fail when absent from page one."""
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        target_pk = 555444333
        page_one = {
            "status": "ok",
            "comments": [{"pk": 1, "text": "shallow", "user": {"username": "a"}}],
            "next_min_id": "cursor-page-2",
            "caption": {"text": "cap", "user": {"username": "p"}},
        }
        page_two = {
            "status": "ok",
            "comments": [{
                "pk": target_pk,
                "text": "deep comment",
                "user": {"username": "deep_user"},
            }],
            "caption": {"text": "cap", "user": {"username": "p"}},
        }
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=json.dumps(page_one)),
            MagicMock(returncode=0, stdout=json.dumps(page_two)),
        ]
        url = f"https://www.instagram.com/reel/ShortCode1/?comment_id={target_pk}"
        info = _fetch_ig_comment(url, config)
        assert info["comment_text"] == "deep comment"
        assert mock_run.call_count == 2
        conf_path = cookie_dir / f".curl-ig-comment-{target_pk}.conf"
        assert not conf_path.exists(), "credential sidecar must not persist after fetch"
        second_cmd = mock_run.call_args_list[1][0][0]
        assert "--config" in second_cmd
        assert str(conf_path) == second_cmd[second_cmd.index("--config") + 1]

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_missing_after_pages_pauses(self, mock_run, config):
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "status": "ok",
                "comments": [],
                "caption": {"text": "cap", "user": {"username": "p"}},
            }),
        )
        url = "https://www.instagram.com/reel/ShortCode1/?comment_id=missing"
        with pytest.raises(StepError) as exc:
            _fetch_ig_comment(url, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "not found in comment pages" in str(exc.value)

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_non_json_step_error_scrubs_stdout(
        self, mock_run, config,
    ):
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "not json path=/home/ubuntu/.ig/session api_key=deadbeefdeadbeefdeadbeefdeadbeef"
            ),
        )
        url = "https://www.instagram.com/reel/ShortCode1/?comment_id=1"
        with pytest.raises(StepError) as exc:
            _fetch_ig_comment(url, config)
        msg = str(exc.value)
        assert exc.value.error_type == ErrorType.RETRY
        assert "/home/ubuntu" not in msg
        assert "deadbeef" not in msg
        assert "[path]" in msg

    @patch("agent.pipelines.instagram.subprocess.run")
    def test_fetch_ig_comment_fail_status_step_error_scrubs_message(
        self, mock_run, config,
    ):
        cookie_dir = config.cookie_dir
        cookie_dir.mkdir(parents=True, exist_ok=True)
        (cookie_dir / "ig_cookies.txt").write_text(
            "\t".join([
                ".instagram.com", "TRUE", "/", "TRUE", "0",
                "sessionid", "sess",
            ]) + "\n",
            encoding="utf-8",
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "status": "fail",
                "message": (
                    "blocked path=/var/secret "
                    "api_key=deadbeefdeadbeefdeadbeefdeadbeef"
                ),
            }),
        )
        url = "https://www.instagram.com/reel/ShortCode1/?comment_id=1"
        with pytest.raises(StepError) as exc:
            _fetch_ig_comment(url, config)
        msg = str(exc.value)
        assert exc.value.error_type == ErrorType.RETRY
        assert "/var/secret" not in msg
        assert "deadbeef" not in msg
        assert "[path]" in msg
        assert "[credential redacted]" in msg
