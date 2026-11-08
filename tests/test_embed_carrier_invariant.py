"""Structural and property invariants for :mod:`agent.embed_carriers`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from agent import embed_carriers
from agent.embed_carriers import (
    EMBED_HOST_HANDLERS,
    ExtractedMediaIndex,
    classify_carrier_url_bucket,
    classify_page_embeds,
    collect_embed_carrier_urls,
    extract_embed_media,
    handled_embed_carrier_re,
    handled_embed_host_re,
)
from agent.pipelines.web import (
    _detect_unenqueued_handled_embed,
    _detect_unsupported_foreign_embed,
    _extract_native_video_ids,
)


# Regression corpus: prior drift modes (a)–(e) plus matrix fixtures from substack tests.
_EMBED_HTML_FIXTURES: list[tuple[str, str]] = [
    ("plain_citation_instagram", (
        "<html><body><p>As noted in "
        '<a href="https://instagram.com/p/abc">this post</a>'
        "</p></body></html>"
    )),
    ("prose_instagram_media_class", (
        "<html><body><p>Do not use instagram-media class</p></body></html>"
    )),
    ("dom_instagram_class", (
        '<html><body><blockquote class="instagram-media" '
        'data-instgrm-permalink="https://www.instagram.com/p/abc/">'
        "</blockquote></body></html>"
    )),
    ("truncated_bunny", (
        '<iframe src="https://player.mediadelivery.net/embed/99/"></iframe>'
    )),
    ("native_video_no_slash", (
        '<iframe src="https://author.substack.com/api/v1/video/upload/'
        "cccc3333-3333-4333-8333-333333333333"
        '"></iframe>'
    )),
    ("spotify_unknown", (
        '<iframe src="https://open.spotify.com/embed/track/abc123"></iframe>'
    )),
    ("youtube_covered", (
        '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>'
    )),
    ("handled_rumble_missing", (
        '<iframe src="https://rumble.com/embed/unknown"></iframe>'
    )),
]


class TestEmbedCarrierStructural:
    def test_no_parallel_handled_regex_in_web_module(self):
        import agent.pipelines.web as web

        assert not hasattr(web, "_HANDLED_EMBED_URL_RE")
        assert not hasattr(web, "_HANDLED_EMBED_CARRIER_URL_RE")
        assert not hasattr(web, "_FOREIGN_EMBED_URL_RULES")

    def test_handlers_imply_canonical_matcher(self):
        for handler in EMBED_HOST_HANDLERS:
            assert handler.canonical_matcher is not None
            assert handler.host_re.pattern
            assert handler.carrier_re.pattern

    def test_derived_handled_regex_matches_every_handler_host(self):
        host_re = handled_embed_host_re()
        carrier_re = handled_embed_carrier_re()
        samples = {
            "youtube": (
                "https://www.youtube.com/embed/abcdef",
                "https://www.youtube.com/embed/abcdef",
            ),
            "rumble": (
                "https://rumble.com/embed/v1abc",
                "https://rumble.com/embed/v1abc",
            ),
            "bunny": (
                "https://player.mediadelivery.net/embed/1/uuid",
                "https://player.mediadelivery.net/embed/1/uuid",
            ),
            "native": (
                "https://x.substack.com/api/v1/video/upload/u",
                "https://x.substack.com/api/v1/video/upload/u",
            ),
        }
        for handler in EMBED_HOST_HANDLERS:
            host_url, carrier_url = samples[handler.extractor_key]
            assert handler.host_re.search(host_url)
            assert handler.carrier_re.search(carrier_url)
            assert host_re.search(host_url)
            assert carrier_re.search(carrier_url)


class TestEmbedCarrierProperty:
    @pytest.mark.parametrize("name,html", _EMBED_HTML_FIXTURES)
    def test_each_carrier_url_in_exactly_one_primary_bucket(self, name: str, html: str):
        extracted = extract_embed_media(html)
        report = classify_page_embeds(html, extracted)
        buckets: dict[str, list[str]] = {
            "covered": [],
            "handled_unenqueued": [],
            "foreign": [],
        }
        for url in collect_embed_carrier_urls(html):
            bucket = classify_carrier_url_bucket(url, extracted)
            if bucket == "covered":
                buckets["covered"].append(url)
            elif bucket == "handled_unenqueued":
                buckets["handled_unenqueued"].append(url)
            elif bucket == "foreign":
                buckets["foreign"].append(url)
        for url in buckets["handled_unenqueued"]:
            assert url not in buckets["foreign"]
        if report.handled_unenqueued:
            assert report.handled_unenqueued in buckets["handled_unenqueued"]
        dom_foreign = _detect_unsupported_foreign_embed(html)
        if dom_foreign and not collect_embed_carrier_urls(html):
            assert buckets["foreign"] == []
        overlap = set(buckets["handled_unenqueued"]) & set(buckets["foreign"])
        assert not overlap, f"{name}: URL in both unenqueued and foreign: {overlap}"


class TestEmbedCarrierRegression:
    def test_bare_instagram_citation_no_pause(self):
        html = _EMBED_HTML_FIXTURES[0][1]
        assert _detect_unsupported_foreign_embed(html) is None

    def test_prose_instagram_media_mention_no_pause(self):
        html = _EMBED_HTML_FIXTURES[1][1]
        assert _detect_unsupported_foreign_embed(html) is None

    def test_instagram_class_pauses(self):
        html = _EMBED_HTML_FIXTURES[2][1]
        assert _detect_unsupported_foreign_embed(html) == "instagram"

    def test_truncated_bunny_handled_unenqueued_not_foreign(self):
        html = _EMBED_HTML_FIXTURES[3][1]
        assert _detect_unenqueued_handled_embed(html, [], [], [], []) is not None
        assert _detect_unsupported_foreign_embed(html) is None

    def test_native_video_iframe_without_trailing_slash(self):
        vid = "cccc3333-3333-4333-8333-333333333333"
        html = _EMBED_HTML_FIXTURES[4][1]
        assert _extract_native_video_ids(html) == [vid]
        assert _detect_unenqueued_handled_embed(html, [], [], [], [vid]) is None

    def test_unknown_iframe_host_pauses_hostname(self):
        html = _EMBED_HTML_FIXTURES[5][1]
        assert _detect_unsupported_foreign_embed(html) == "open.spotify.com"


def test_fresh_subprocess_import_embed_carriers():
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", "import agent.embed_carriers"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
