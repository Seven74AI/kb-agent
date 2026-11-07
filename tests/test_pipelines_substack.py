"""Tests for the Substack pipeline — detection, cleanup, embedded media, native video."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Point to workspace, not installed version
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent.config import Config
from agent.detection import _detect_substack_subtype
from agent.exceptions import ErrorType, StepError
from agent.pipelines.web import (
    step_substack_cleanup,
    step_substack_embedded_media,
    step_substack_video_extract_metadata,
    _extract_body_images,
    _extract_rumble_urls,
    _extract_youtube_ids,
    _fetch_substack_html,
    _is_content_image,
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


# ── Embedded Media Tests (body images + embedded videos) ─────────────────────

class TestSubstackEmbeddedMedia:

    def test_extract_body_images_multiple(self):
        html = (
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/aaa-1_1458x704.png&quot;}"></div>'
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/bbb-1_1664x1184.png&quot;}"></div>'
        )
        imgs = _extract_body_images(html)
        assert len(imgs) == 2
        assert imgs[0].endswith("aaa-1_1458x704.png")

    def test_extract_body_images_filters_avatars(self):
        html = (
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/avatar_144x144.png&quot;}"></div>'
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/content_1664x1184.png&quot;}"></div>'
        )
        imgs = _extract_body_images(html)
        assert imgs == [
            "https://substack-post-media.s3.amazonaws.com/public/images/content_1664x1184.png"
        ]

    def test_extract_body_images_encoded_cdn_form(self):
        html = (
            '<img src="https://substackcdn.com/image/fetch/p,g_auto/'
            'https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fchart_1086x518.png">'
        )
        imgs = _extract_body_images(html)
        assert imgs == [
            "https://substack-post-media.s3.amazonaws.com/public/images/chart_1086x518.png"
        ]

    def test_extract_body_images_drops_square_and_banner(self):
        html = (
            '<img src="https://substackcdn.com/image/fetch/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fheadshot_553x554.jpeg">'
            '<img src="https://substackcdn.com/image/fetch/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbanner_1100x210.png">'
            '<img src="https://substackcdn.com/image/fetch/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fchart_1664x1184.png">'
        )
        imgs = _extract_body_images(html)
        assert imgs == [
            "https://substack-post-media.s3.amazonaws.com/public/images/chart_1664x1184.png"
        ]

    def test_extract_youtube_ids_iframe_and_videoid(self):
        html = (
            '<iframe src="https://www.youtube-nocookie.com/embed/xwmL7Qr6ft4?rel=0"></iframe>'
            '<div data-attrs="{&quot;videoId&quot;:&quot;xwmL7Qr6ft4&quot;}"></div>'
            '<div data-attrs="{&quot;videoId&quot;:&quot;abcDEF12345&quot;}"></div>'
        )
        ids = _extract_youtube_ids(html)
        assert ids == ["xwmL7Qr6ft4", "abcDEF12345"]

    def test_extract_rumble_urls(self):
        html = (
            '<a href="https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy">a</a>'
            '<a href="https://rumble.com/vabc123-short">b</a>'
            '<a href="https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy">dup</a>'
        )
        assert _extract_rumble_urls(html) == [
            "https://rumble.com/v2y5lfu-evidence-of-crimes-against-humanity-darkfield-blood-microscopy",
            "https://rumble.com/vabc123-short",
        ]

    def test_drops_non_image_bin_asset(self):
        # Substack sometimes serves a .bin under the images/ prefix — not an image.
        html = (
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/b24204eb_496x220.bin&quot;}"></div>'
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/chart_1664x1184.png&quot;}"></div>'
        )
        assert _extract_body_images(html) == [
            "https://substack-post-media.s3.amazonaws.com/public/images/chart_1664x1184.png"
        ]

    def test_keeps_heic_and_avif(self):
        assert _is_content_image("de5ede07_1690x2240.heic") is True
        assert _is_content_image("335e0379_1536x1024.avif") is True


class TestSubstackVideoMetadata:

    @patch("agent.pipelines.web.subprocess.run")
    def test_finds_video_upload_id(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                '<title>I MADE A VLOG</title>'
                '<script>window._preloads = {"post":{"type":"video",'
                '"videoUpload":{"id":"a7315fa8-8027-4432-a7e3-652479bc5275"}}};</script>'
            ),
        )
        result = step_substack_video_extract_metadata(task_dict, config)
        assert "a7315fa8" in result
        meta = json.loads(
            (config.temp_dir / f"{task_dict['slug']}_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["video_id"] == "a7315fa8-8027-4432-a7e3-652479bc5275"
        assert meta["title"] == "I MADE A VLOG"
        assert meta["domain"] == "testblog.substack.com"

    @patch("agent.pipelines.web.subprocess.run")
    def test_no_video_id_raises_pause(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mock_run.return_value = MagicMock(returncode=0, stdout="<html><body>text post</body></html>")
        with pytest.raises(Exception) as exc:
            step_substack_video_extract_metadata(task_dict, config)
        assert "videoUpload.id" in str(exc.value) or "video id" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    @patch("agent.pipelines.web._enqueue_embedded_media")
    def test_defers_synthesis_when_media_found(self, mock_enqueue, mock_fetch, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mock_fetch.return_value = (
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/img1_1200x800.webp&quot;}"></div>'
            '<iframe src="https://www.youtube-nocookie.com/embed/xyzXYZ12345"></iframe>'
        )
        with pytest.raises(Exception) as exc:
            step_substack_embedded_media(task_dict, config)
        assert "awaiting media" in str(exc.value)
        manifest = json.loads(
            (config.temp_dir / f"{task_dict['slug']}_media.json").read_text(encoding="utf-8")
        )
        assert manifest["total"] == 2
        assert manifest["url"] == task_dict["source_url"]
        mock_enqueue.assert_called_once_with(
            task_dict["slug"],
            ["https://substack-post-media.s3.amazonaws.com/public/images/img1_1200x800.webp"],
            ["xyzXYZ12345"], [], config, None,
        )

    @patch("agent.pipelines.web.subprocess.run")
    def test_fetch_substack_html_passes_cookie_when_present(self, mock_run, config):
        config.cookie_dir.mkdir(parents=True, exist_ok=True)
        cookie_path = config.cookie_dir / "substack_cookies.txt"
        cookie_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        mock_run.return_value = MagicMock(stdout="<html></html>", returncode=0)
        _fetch_substack_html("https://author.substack.com/p/post", config)
        cmd = mock_run.call_args[0][0]
        assert "-b" in cmd
        assert str(cookie_path) in cmd

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_no_media_synthesizes_inline(self, mock_fetch, config, task_dict):
        mock_fetch.return_value = "<html><body>plain text article</body></html>"
        result = step_substack_embedded_media(task_dict, config)
        assert result == "no embedded media"

    @patch("agent.pipelines.web._fetch_substack_html")
    @patch("agent.pipelines.web._enqueue_embedded_media")
    def test_empty_html_raises_retry(self, mock_enqueue, mock_fetch, config, task_dict):
        mock_fetch.return_value = ""
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.RETRY
        mock_enqueue.assert_not_called()


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
