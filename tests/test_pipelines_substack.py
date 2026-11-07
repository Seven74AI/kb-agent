"""Tests for the Substack pipeline — detection, cleanup, embedded media, native video."""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse
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
    step_synthesize_web,
    step_substack_video_extract_metadata,
    _extract_body_images,
    _extract_native_video_ids,
    step_substack_video_download,
    _extract_rumble_urls,
    _extract_youtube_ids,
    _fetch_substack_html,
    _is_content_image,
    _is_substack_unauth_paywall_shell,
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

    def test_extract_youtube_ids_prose_links(self):
        html = (
            '<p>See https://www.youtube.com/watch?v=dQw4w9WgXcQ and '
            'https://youtu.be/abcDEF12345 for more.</p>'
            '<p>Short: https://www.youtube.com/shorts/shortVid1234</p>'
        )
        ids = _extract_youtube_ids(html)
        assert ids == ["dQw4w9WgXcQ", "shortVid1234", "abcDEF12345"]

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

    def test_extract_rumble_embed_urls_normalized(self):
        html = '<iframe src="https://rumble.com/embed/v2vk68c/"></iframe>'
        assert _extract_rumble_urls(html) == ["https://rumble.com/v2vk68c"]

    def test_fetch_substack_html_rejects_metadata_ip(self, monkeypatch):
        monkeypatch.setattr(
            "agent.pipelines.web.subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )
        with pytest.raises(ValueError, match="not allowed"):
            _fetch_substack_html("http://169.254.169.254/latest/meta-data/")

    @patch("agent.pipelines.web.subprocess.run")
    def test_substack_video_metadata_rejects_metadata_ip(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        task_dict["source_url"] = "http://169.254.169.254/latest/meta-data/"
        with pytest.raises(ValueError, match="not allowed"):
            step_substack_video_extract_metadata(task_dict, config)
        mock_run.assert_not_called()

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
            (config.temp_dir / f"{task_dict['id']}_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["video_id"] == "a7315fa8-8027-4432-a7e3-652479bc5275"
        assert meta["title"] == "I MADE A VLOG"
        assert meta["domain"] == "testblog.substack.com"

    @patch("agent.pipelines.web.ensure_h264", side_effect=lambda p: p)
    @patch("agent.pipelines.web.subprocess.run")
    def test_two_sibling_native_videos_distinct_download_artifacts(
        self, mock_run, _h264, config, tmp_path,
    ):
        """Shared slug siblings must not clobber metadata or mp4 in temp_dir."""
        from agent.core.slugify import slugify

        config.temp_dir.mkdir(parents=True, exist_ok=True)
        post = "https://author.substack.com/p/multi-video"
        slug = slugify(post)
        vid_a = "aaaaaaaa-1111-4111-8111-111111111111"
        vid_b = "bbbbbbbb-2222-4222-8222-222222222222"
        assert slugify(f"{post}#{vid_a}") == slugify(f"{post}#{vid_b}")

        def curl_side_effect(cmd, *args, **kwargs):
            url = next((c for c in cmd if c.startswith("http")), "")
            if "extract" in str(kwargs) or "-o" not in cmd:
                frag = urlparse(url).fragment
                body = (
                    f'<title>Post</title>'
                    f'"videoUpload":{{"id":"{frag or vid_a}"}}'
                )
                return MagicMock(returncode=0, stdout=body)
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_bytes(b"\x00" * 2000)
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_run.side_effect = curl_side_effect

        task_a = {
            "id": "ingest-row-aaa",
            "slug": slug,
            "source_url": f"{post}#{vid_a}",
            "content_type": "substack_video",
        }
        task_b = {
            "id": "ingest-row-bbb",
            "slug": slug,
            "source_url": f"{post}#{vid_b}",
            "content_type": "substack_video",
        }
        step_substack_video_extract_metadata(task_a, config)
        step_substack_video_extract_metadata(task_b, config)
        meta_a = json.loads((config.temp_dir / f"{task_a['id']}_metadata.json").read_text())
        meta_b = json.loads((config.temp_dir / f"{task_b['id']}_metadata.json").read_text())
        assert meta_a["video_id"] == vid_a
        assert meta_b["video_id"] == vid_b

        step_substack_video_download(task_a, config)
        step_substack_video_download(task_b, config)
        mp4_a = config.temp_dir / f"{task_a['id']}.mp4"
        mp4_b = config.temp_dir / f"{task_b['id']}.mp4"
        assert mp4_a.is_file() and mp4_b.is_file()
        assert mp4_a != mp4_b
        assert not (config.temp_dir / f"{slug}_metadata.json").exists()
        assert not (config.temp_dir / f"{slug}.mp4").exists()

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
        mock_enqueue.return_value = 2
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
            ["xyzXYZ12345"], [], [], task_dict["source_url"], config, None,
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
    def test_zero_embeds_with_pending_siblings_pauses_not_inline(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, init_db, queue_ingest

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        mock_fetch.return_value = "<html><body>transient empty shell</body></html>"

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        queue_ingest(
            conn, f"{post}#vid", "substack_video", slug,
            parent_slug=slug, media_kind="video", status="queued",
        )
        conn.close()

        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "re-fetch found no embeds but siblings pending" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_zero_embeds_manifest_and_queue_gone_but_deferral_marker_pauses(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        mock_fetch.return_value = "<html><body>empty re-fetch</body></html>"

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        task_id = create_task(
            conn, post, "substack_text", slug,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        )
        conn.execute(
            """
            UPDATE task_steps SET output = ?, status = 'done'
            WHERE task_id = ? AND name = 'substack_cleanup'
            """,
            (f"cleaned\nmedia_deferred:{slug}", task_id),
        )
        conn.commit()
        conn.close()

        task_dict["id"] = task_id
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "prior deferral" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_zero_embeds_deferral_marker_allows_inline_when_synthesize_task_exists(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        mock_fetch.return_value = "<html><body>plain text</body></html>"

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        task_id = create_task(
            conn, post, "substack_text", slug,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        )
        conn.execute(
            "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'substack_cleanup'",
            (f"cleaned\nmedia_deferred:{slug}", task_id),
        )
        create_task(
            conn, post, "substack_text_synthesize", slug,
            ["synthesize_note"], status="completed",
        )
        conn.commit()
        conn.close()

        task_dict["id"] = task_id
        result = step_substack_embedded_media(task_dict, config)
        assert result == "no embedded media"

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_zero_embeds_raw_media_sections_alone_do_not_unblock_deferral(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        (config.temp_dir / f"{slug}_raw.md").write_text(
            "# Article\n\nIntro.\n\n## Image\n\nChart OCR.\n",
            encoding="utf-8",
        )
        mock_fetch.return_value = "<html><body>plain text</body></html>"

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        task_id = create_task(
            conn, post, "substack_text", slug,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        )
        conn.execute(
            "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'substack_cleanup'",
            (f"media_deferred:{slug}", task_id),
        )
        conn.commit()
        conn.close()

        task_dict["id"] = task_id
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "prior deferral" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_zero_embeds_manifest_parity_without_synthesize_pauses(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest, update_ingest_status

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "content_type": "substack_text", "total": 1}),
            encoding="utf-8",
        )
        mock_fetch.return_value = "<html><body>plain text</body></html>"

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        task_id = create_task(
            conn, post, "substack_text", slug,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        )
        conn.execute(
            "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'substack_cleanup'",
            (f"media_deferred:{slug}", task_id),
        )
        ingest_id, _ = queue_ingest(
            conn, f"{post}#vid", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, ingest_id, "dispatched")
        conn.commit()
        conn.close()

        task_dict["id"] = task_id
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "re-fetch found no embeds but siblings pending" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_deferring_media_persists_marker_on_parent_task(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        mock_fetch.return_value = (
            '<div data-attrs="{&quot;src&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/img1_1200x800.webp&quot;}"></div>'
        )

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        task_id = create_task(
            conn, post, "substack_text", slug,
            ["substack_cleanup", "substack_embedded_media", "synthesize_note"],
        )
        conn.execute(
            "UPDATE task_steps SET output = 'cleaned' WHERE task_id = ? AND name = 'substack_cleanup'",
            (task_id,),
        )
        conn.commit()
        conn.close()

        task_dict["id"] = task_id
        with pytest.raises(StepError):
            step_substack_embedded_media(task_dict, config)

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        row = conn.execute(
            "SELECT output FROM task_steps WHERE task_id = ? AND name = 'substack_cleanup'",
            (task_id,),
        ).fetchone()
        conn.close()
        assert row[0] and f"media_deferred:{slug}" in row[0]

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_zero_embeds_with_deferral_manifest_pauses_not_inline(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "content_type": "substack_text", "total": 2}),
            encoding="utf-8",
        )
        mock_fetch.return_value = "<html><body>cookie loss — no embeds in html</body></html>"

        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "re-fetch found no embeds but siblings pending" in str(exc.value)

    @patch("agent.pipelines.web._fetch_substack_html")
    @patch("agent.pipelines.web._enqueue_embedded_media")
    def test_empty_html_raises_retry(self, mock_enqueue, mock_fetch, config, task_dict):
        mock_fetch.return_value = ""
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.RETRY
        mock_enqueue.assert_not_called()

    def test_extract_native_video_ids_from_preloads(self):
        html = (
            '{"videoUpload":{"id":"a7315fa8-8027-4432-a7e3-652479bc5275"}}'
        )
        assert _extract_native_video_ids(html) == ["a7315fa8-8027-4432-a7e3-652479bc5275"]

    def test_extract_native_video_ids_from_api_path(self):
        html = '<a href="/api/v1/video/upload/deadbeef-1111-2222-3333-444455556666/src">'
        assert _extract_native_video_ids(html) == [
            "deadbeef-1111-2222-3333-444455556666"
        ]

    def test_extract_native_video_ids_multiple_uploads(self):
        html = (
            '{"videoUpload":{"id":"aaaa1111-1111-1111-1111-111111111111"}}'
            'later in page '
            '"videoUpload":{"id":"bbbb2222-2222-2222-2222-222222222222"}'
        )
        assert _extract_native_video_ids(html) == [
            "aaaa1111-1111-1111-1111-111111111111",
            "bbbb2222-2222-2222-2222-222222222222",
        ]

    def test_paywall_shell_detected(self):
        html = (
            "<html><body><div data-testid='paywall'>Subscribe to continue</div></body></html>"
        )
        assert _is_substack_unauth_paywall_shell(html) is True

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_unauth_paywall_pauses_not_inline_synthesize(self, mock_fetch, config, task_dict):
        config.cookie_dir.mkdir(parents=True, exist_ok=True)
        mock_fetch.return_value = (
            "<html><body>Subscribe to continue reading</body></html>"
        )
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "substack cookie" in str(exc.value).lower()

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_native_videos_enqueue_distinct_urls(self, mock_fetch, config, task_dict, tmp_path):
        from agent.db import AGENT_SCHEMA, init_db

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        post = task_dict["source_url"]
        mock_fetch.return_value = (
            "<html><body>"
            '{"videoUpload":{"id":"aaaa1111-1111-1111-1111-111111111111"}}'
            '{"videoUpload":{"id":"bbbb2222-2222-2222-2222-222222222222"}}'
            "</body></html>"
        )
        with pytest.raises(StepError):
            step_substack_embedded_media(task_dict, config)
        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        try:
            rows = conn.execute(
                "SELECT url FROM ingest_queue WHERE parent_slug=? ORDER BY url",
                (task_dict["slug"],),
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == [
            f"{post}#aaaa1111-1111-1111-1111-111111111111",
            f"{post}#bbbb2222-2222-2222-2222-222222222222",
        ]

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_retry_skips_pause_when_siblings_dispatched(self, mock_fetch, config, task_dict, tmp_path):
        from agent.db import AGENT_SCHEMA, init_db, queue_ingest, update_ingest_status

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        vid = "a7315fa8-8027-4432-a7e3-652479bc5275"
        manifest = config.temp_dir / f"{slug}_media.json"
        manifest.write_text(json.dumps({"url": post, "total": 1}), encoding="utf-8")

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        ingest_id, _ = queue_ingest(
            conn, f"{post}#{vid}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, ingest_id, "dispatched")
        conn.close()

        mock_fetch.return_value = (
            f'<html><body><script>"videoUpload":{{"id":"{vid}"}}</script></body></html>'
        )
        result = step_substack_embedded_media(task_dict, config)
        assert result == "media ready — deferred to batch"

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_media_ready_defers_when_synthesize_sibling_active(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest, update_ingest_status

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        vid = "a7315fa8-8027-4432-a7e3-652479bc5275"
        manifest = config.temp_dir / f"{slug}_media.json"
        manifest.write_text(json.dumps({"url": post, "total": 1}), encoding="utf-8")

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        ingest_id, _ = queue_ingest(
            conn, f"{post}#{vid}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, ingest_id, "dispatched")
        create_task(
            conn, post, "substack_text_synthesize", slug,
            ["synthesize_note"], status="pending",
        )
        conn.commit()
        conn.close()

        mock_fetch.return_value = (
            f'<html><body><script>"videoUpload":{{"id":"{vid}"}}</script></body></html>'
        )
        result = step_substack_embedded_media(task_dict, config)
        assert result == "media ready — deferred to batch"

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_media_ready_when_manifest_total_matches_queue_count(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, init_db, queue_ingest, update_ingest_status

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        vid_a = "aaaa1111-1111-1111-1111-111111111111"
        vid_b = "bbbb2222-2222-2222-2222-222222222222"
        manifest = config.temp_dir / f"{slug}_media.json"
        manifest.write_text(json.dumps({"url": post, "total": 2}), encoding="utf-8")

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        id_a, _ = queue_ingest(
            conn, f"{post}#{vid_a}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        id_b, _ = queue_ingest(
            conn, f"{post}#{vid_b}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, id_a, "dispatched")
        update_ingest_status(conn, id_b, "dispatched")
        conn.close()

        mock_fetch.return_value = (
            "<html><body>"
            f'"videoUpload":{{"id":"{vid_a}"}}'
            f'"videoUpload":{{"id":"{vid_b}"}}'
            "</body></html>"
        )
        result = step_substack_embedded_media(task_dict, config)
        assert result == "media ready — deferred to batch"
        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        row = conn.execute(
            "SELECT COUNT(*) FROM ingest_queue WHERE parent_slug = ?", (slug,),
        ).fetchone()
        conn.close()
        updated = json.loads(manifest.read_text(encoding="utf-8"))
        assert updated["total"] == row[0] == 2

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_media_ready_with_prior_manifest_never_inline_even_without_synthesize_task(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import AGENT_SCHEMA, init_db, queue_ingest, update_ingest_status

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        vid = "a7315fa8-8027-4432-a7e3-652479bc5275"
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "total": 1}), encoding="utf-8",
        )

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        ingest_id, _ = queue_ingest(
            conn, f"{post}#{vid}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, ingest_id, "dispatched")
        conn.close()

        mock_fetch.return_value = (
            f'<html><body><script>"videoUpload":{{"id":"{vid}"}}</script></body></html>'
        )
        result = step_substack_embedded_media(task_dict, config)
        assert result == "media ready — deferred to batch"

    @patch("agent.pipelines.web.get_llm_client")
    @patch("agent.pipelines.web._synthesize_single_pass", return_value="# Note\n")
    def test_synthesize_pipeline_not_blocked_by_deferred_manifest(
        self, _mock_syn, _mock_llm, config, task_dict, tmp_path,
    ):
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "content_type": "substack_text", "total": 1}),
            encoding="utf-8",
        )
        (config.temp_dir / f"{slug}_raw.md").write_text("# Post\n\nBody.\n", encoding="utf-8")
        synth_task = {**task_dict, "content_type": "substack_text_synthesize"}
        step_synthesize_web(synth_task, config)
        assert (config.vault_path / "Knowledge base" / "articles" / f"{slug}.md").is_file()

    def test_deferred_manifest_parent_does_not_inline_synthesize(self, config, task_dict, tmp_path):
        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "content_type": "substack_text", "total": 1}),
            encoding="utf-8",
        )
        (config.temp_dir / f"{slug}_raw.md").write_text("# Post\n\nBody.\n", encoding="utf-8")

        with pytest.raises(StepError) as exc:
            step_synthesize_web(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "deferred to batch synthesis" in str(exc.value).lower()
        assert not (config.vault_path / "Knowledge base" / "articles" / f"{slug}.md").exists()

    def test_deferred_media_batch_finished_requires_synthesize_signal(self, config, tmp_path):
        from agent.db import AGENT_SCHEMA, create_task, init_db, queue_ingest, update_ingest_status
        from agent.pipelines.web import _deferred_media_batch_finished

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = "defer-batch-finish"
        post = "https://author.substack.com/p/defer-batch-finish"
        (config.temp_dir / f"{slug}_media.json").write_text(
            json.dumps({"url": post, "total": 1}), encoding="utf-8",
        )

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        ingest_id, _ = queue_ingest(
            conn, f"{post}#vid", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, ingest_id, "dispatched")
        conn.commit()
        assert _deferred_media_batch_finished(conn, config, slug) is False

        create_task(
            conn, post, "substack_text_synthesize", slug,
            ["synthesize_note"], status="pending",
        )
        conn.commit()
        assert _deferred_media_batch_finished(conn, config, slug) is True
        conn.close()

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_media_gate_blocks_when_siblings_removed_below_manifest_total(
        self, mock_fetch, config, task_dict, tmp_path,
    ):
        from agent.db import (
            AGENT_SCHEMA,
            init_db,
            queue_ingest,
            remove_ingest,
            update_ingest_status,
        )

        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = task_dict["slug"]
        post = task_dict["source_url"]
        vid_a = "aaaa1111-1111-1111-1111-111111111111"
        vid_b = "bbbb2222-2222-2222-2222-222222222222"
        manifest = config.temp_dir / f"{slug}_media.json"
        manifest.write_text(json.dumps({"url": post, "total": 2}), encoding="utf-8")

        conn = init_db(config.kb_agent_dir / "data" / "agent.db", AGENT_SCHEMA)
        id_a, _ = queue_ingest(
            conn, f"{post}#{vid_a}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        id_b, _ = queue_ingest(
            conn, f"{post}#{vid_b}", "substack_video", slug,
            parent_slug=slug, media_kind="video",
        )
        update_ingest_status(conn, id_a, "dispatched")
        update_ingest_status(conn, id_b, "failed", error="transcription error")
        remove_ingest(conn, id_b)
        conn.close()

        # HTML only shows the surviving sibling — re-enqueue does not restore the removed row.
        mock_fetch.return_value = (
            f'<html><body><script>"videoUpload":{{"id":"{vid_a}"}}</script></body></html>'
        )
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert exc.value.error_type == ErrorType.PAUSE
        assert "media gate" in str(exc.value)
        assert "ingest rows 1 != manifest total 2" in str(exc.value)

    @patch("agent.pipelines.web.subprocess.run")
    def test_substack_video_download_rejects_blocked_endpoint(self, mock_run, config, task_dict):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        meta_path = config.temp_dir / f"{task_dict['id']}_metadata.json"
        meta_path.write_text(
            json.dumps({
                "video_id": "vid-1",
                "domain": "127.0.0.1",
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not allowed"):
            step_substack_video_download(task_dict, config)
        mock_run.assert_not_called()

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_native_video_enqueued_with_text_post(self, mock_fetch, config, task_dict, tmp_path):
        config.kb_agent_dir = tmp_path / "kb"
        (config.kb_agent_dir / "data").mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mock_fetch.return_value = (
            "<html><body><p>Intro text</p>"
            '<script>window._preloads = {"post":{"videoUpload":{"id":'
            '"a7315fa8-8027-4432-a7e3-652479bc5275"}}};</script></body></html>'
        )
        with pytest.raises(StepError) as exc:
            step_substack_embedded_media(task_dict, config)
        assert "awaiting media" in str(exc.value)
        manifest = json.loads(
            (config.temp_dir / f"{task_dict['slug']}_media.json").read_text(encoding="utf-8")
        )
        assert manifest["total"] == 1

    @patch("agent.pipelines.web._fetch_substack_html")
    def test_auth_plain_text_no_media(self, mock_fetch, config, task_dict):
        config.cookie_dir.mkdir(parents=True, exist_ok=True)
        (config.cookie_dir / "substack_cookies.txt").write_text(
            "# Netscape HTTP Cookie File\n", encoding="utf-8",
        )
        mock_fetch.return_value = "<html><body>plain text article</body></html>"
        result = step_substack_embedded_media(task_dict, config)
        assert result == "no embedded media"


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
