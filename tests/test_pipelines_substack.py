"""Tests for the Substack pipeline — detection, cleanup, video pause, image OCR."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Point to workspace, not installed version
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.detection import _detect_substack_subtype
from agent.pipelines.web import (
    step_substack_cleanup,
    step_substack_image_extract,
    SUBSTACK_BOILERPLATE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        firecrawl_url="http://localhost:3002/v2/scrape",
    )


@pytest.fixture
def task_dict(config):
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "id": "test-substack-12345",
        "source_url": "https://testblog.substack.com/p/test-article",
        "content_type": "substack_text",
        "slug": "testblog.substack.com_p_test-article",
    }


# ── Detection Tests ───────────────────────────────────────────────────────────

class TestSubstackDetection:

    @patch("agent.detection.subprocess.run")
    def test_detect_text_no_markers(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="<html><body><article>This is a text post</article></body></html>",
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/text-only")
        assert result == "substack_text"

    @patch("agent.detection.subprocess.run")
    def test_detect_video_preloads(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='<html><body><script>window.__preloads = {"attachments":[{"type":"video","id":"abc123"}]};</script></body></html>',
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/video-post")
        assert result == "substack_video"

    @patch("agent.detection.subprocess.run")
    def test_detect_video_fallback(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='<html><body><div data-video-id="abc123"><a href="/api/v1/video/upload/abc123/src">Download</a></div></body></html>',
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/video-post")
        assert result == "substack_video"

    @patch("agent.detection.subprocess.run")
    def test_detect_video_aria_label(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='<html><body><div aria-label="Video player"></div></body></html>',
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/video-post")
        assert result == "substack_video"

    @patch("agent.detection.subprocess.run")
    def test_detect_image_preloads(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='<html><body><script>window.__preloads = {"attachments":[{"type":"image","url":"https://substackimg.com/img.jpg"}]};</script></body></html>',
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/image-post")
        assert result == "substack_image"

    @patch("agent.detection.subprocess.run")
    def test_detect_image_body_class(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='<html><body class="image-post"><img src="photo.jpg"></body></html>',
        )
        result = _detect_substack_subtype("https://blog.substack.com/p/image-post")
        assert result == "substack_image"

    @patch("agent.detection.subprocess.run")
    def test_detect_curl_failure(self, mock_run):
        mock_run.side_effect = Exception("Connection refused")
        result = _detect_substack_subtype("https://blog.substack.com/p/something")
        assert result == "substack_text"


# ── Cleanup Tests ─────────────────────────────────────────────────────────────

class TestSubstackCleanup:

    def test_removes_subscribe_cta(self, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("# Article Title\n\nGood content here.\n\n---\n\nSubscribe\n\nGet full access to this blog.\n\nClick here to subscribe now.\n")
        result = step_substack_cleanup(task_dict, config)
        assert result == "cleaned"
        cleaned = raw_path.read_text()
        assert "Article Title" in cleaned
        assert "Good content" in cleaned
        # "Subscribe" (standalone line) should be gone
        assert "Subscribe\n" not in cleaned

    def test_removes_thanks_cta(self, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("# Article Title\n\nKey content.\n\nThanks for reading Test Blog! Subscribe for free to receive new posts.\n")
        step_substack_cleanup(task_dict, config)
        cleaned = raw_path.read_text()
        assert "Key content" in cleaned
        assert "Thanks for reading" not in cleaned

    def test_removes_substack_footer(self, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("# Article Title\n\nMain content.\n\nSubstack\n\xa9 2024 Author Name\n")
        step_substack_cleanup(task_dict, config)
        cleaned = raw_path.read_text()
        assert "Main content" in cleaned
        # Footer with copyright symbol should be stripped
        assert "Substack\n" not in cleaned

    def test_collapses_multiple_blank_lines(self, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("Para one.\n\n\n\n\n\nPara two.")
        step_substack_cleanup(task_dict, config)
        cleaned = raw_path.read_text()
        assert "\n\n\n" not in cleaned

    def test_no_raw_file(self, config, task_dict):
        result = step_substack_cleanup(task_dict, config)
        assert result == "no raw file to clean"


# ── Image Extraction Tests ────────────────────────────────────────────────────

class TestSubstackImageExtract:

    @patch("agent.pipelines.web.subprocess.run")
    def test_extract_from_preloads(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("# Article Title\n\nContent before image.\n")

        # Pre-create a fake image file so the stat checks pass
        img_path = config.temp_dir / f"{slug}_image"
        img_path.write_bytes(b"x" * 200)

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='<script>window.__preloads = {"attachments":[{"type":"image","url":"https://img.example.com/photo.jpg"}]};</script>'),
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=0, stdout="OCR text from image"),
        ]

        result = step_substack_image_extract(task_dict, config)
        assert "image_extracted" in result
        cleaned = raw_path.read_text()
        assert "## Image" in cleaned
        assert "OCR text from image" in cleaned

    @patch("agent.pipelines.web.subprocess.run")
    def test_extract_from_og_image(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        raw_path = config.temp_dir / f"{slug}_raw.md"
        raw_path.write_text("# Article Title\n")

        img_path = config.temp_dir / f"{slug}_image"
        img_path.write_bytes(b"y" * 200)

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='<html><head><meta property="og:image" content="https://cdn.example.com/og-photo.jpg"></head><body></body></html>'),
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=0, stdout="Fallback OCR text"),
        ]

        result = step_substack_image_extract(task_dict, config)
        assert "image_extracted" in result

    @patch("agent.pipelines.web.subprocess.run")
    def test_no_image_found(self, mock_run, config, task_dict):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="<html><body>No images here</body></html>",
        )
        result = step_substack_image_extract(task_dict, config)
        assert result == "no image found"

    @patch("agent.pipelines.web.subprocess.run")
    def test_image_download_fails(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='<meta property="og:image" content="https://broken.example.com/img.jpg">'),
            MagicMock(returncode=7, stderr="Could not resolve host"),
        ]
        result = step_substack_image_extract(task_dict, config)
        assert "image download failed" in result


# ── Boilerplate Pattern Tests ───────────────────────────────────────────────

def test_boilerplate_patterns_non_empty():
    import re
    for pattern in SUBSTACK_BOILERPLATE:
        try:
            re.compile(pattern, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        except re.error as e:
            pytest.fail(f"Invalid regex pattern '{pattern}': {e}")


def test_boilerplate_matches_subscribe():
    import re
    pattern = SUBSTACK_BOILERPLATE[0]
    text = "Subscribe\n\nGet full access to this blog."
    assert re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)


def test_boilerplate_matches_substack_footer():
    import re
    pattern = SUBSTACK_BOILERPLATE[9]
    text = "\nSubstack\n\xa9 2024 Author"
    assert re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
