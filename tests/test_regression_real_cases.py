"""Regression tests from real ingestion failures (2026-11-05 bulk import).

Each case is a real URL or real payload shape that misclassified during the
import, plus the correct outcome. Run on every kb workflow change — these are
the concrete failures the detection/routing/normalization/extraction fixes
must never reintroduce.
"""

import pytest
import subprocess

from agent.config import Config
from agent import detection
from agent.pipelines.instagram import _shortcode_to_media_id
from agent.pipelines.threads import _extract_json_array_after_key, _extract_video_url
from agent.pipelines.video_common import strip_code_fence


# ── Ground truth from the failed import ─────────────────────────────────────

# Real Threads video posts (non-null video_versions array in the payload).
THREADS_VIDEO = [
    "https://www.threads.com/@tayjmcm/post/DZQkGBvFDB5",
    # null video_versions FIRST, real array LATER — the extractor must skip null
    "https://www.threads.com/@bourne_ti/post/Da3I_d0ls4E",
    "https://www.threads.com/share/BBJahcWRBo/",
]
# Real Threads text posts (every video_versions is null).
THREADS_TEXT = [
    "https://www.threads.com/share/BABsuaYfBY/",
    "https://www.threads.com/share/BAV8aVwjWb/",
    "https://www.threads.com/share/BAE5AoLC_g/",
]
# Real Instagram comment deep-link.
COMMENT_URL = "https://www.instagram.com/reel/DdnESNvo13o/?comment_id=18132350785588878"

# Minimal payload shapes — the structural signal, not the whole ~900KB page.
REAL_VIDEO_JSON = '"video_versions":[{"type":101,"url":"https://scontent.cdninstagram.com/v.mp4","height":1080}]'
NULL_VIDEO_JSON = '"video_versions":null,"canonical_url":null,"usertags":null'
NULL_THEN_REAL = NULL_VIDEO_JSON + ',"is_dash_eligible":null,' + REAL_VIDEO_JSON + ',"x":"y"'


# ── Platform detection ──────────────────────────────────────────────────────

def test_share_link_is_threads_not_web():
    # threads.com/share/ short links must route to threads, not the web fallback.
    assert detection.detect_platform("https://www.threads.com/share/BBJahcWRBo/") == "threads"


# ── Subtype detection (deterministic, null-aware) ───────────────────────────

@pytest.mark.parametrize("url", THREADS_VIDEO)
def test_threads_video_detected(monkeypatch, url):
    monkeypatch.setattr(detection, "_fetch_content", lambda u, p: (REAL_VIDEO_JSON, u))
    monkeypatch.setattr(detection, "_resolve_threads_share_redirect", lambda u, ca, timeout=0: u)
    assert detection.detect_content_subtype(url, "threads") == "threads_video"


@pytest.mark.parametrize("url", THREADS_TEXT)
def test_threads_text_not_misdetected_as_video(monkeypatch, url):
    # a bare video_versions substring (null) must NOT classify as video
    monkeypatch.setattr(detection, "_fetch_content", lambda u, p: (NULL_VIDEO_JSON, u))
    monkeypatch.setattr(detection, "_resolve_threads_share_redirect", lambda u, ca, timeout=0: u)
    assert detection.detect_content_subtype(url, "threads") == "threads_text"


def test_null_then_real_still_video(monkeypatch):
    # the original extraction bug: first occurrence null, real array later
    monkeypatch.setattr(detection, "_fetch_content", lambda u, p: (NULL_THEN_REAL, u))
    monkeypatch.setattr(detection, "_resolve_threads_share_redirect", lambda u, ca, timeout=0: u)
    assert detection.detect_content_subtype(THREADS_VIDEO[1], "threads") == "threads_video"


# ── Share-link redirect resolution (real 2026-11-09 failure) ────────────────
# threads.com/share/<code> 302-redirects to the canonical @handle/post URL. The
# SSRF-safe fetch resolves that hop manually; without it the fetch hits
# --max-redirs 0 and returns an empty body, misclassifying a video share link
# as threads_text.

SHARE_URL = "https://www.threads.com/share/BAYD9ij7_f"
SHARE_CANONICAL = "https://www.threads.com/@cejimages/post/DfEAyOoCd_p"
SHARE_VIDEO_BODY = '"video_versions":[{"type":101,"url":"https://scontent.cdninstagram.com/v.mp4","height":1080}]'


def _share_redirect_run(monkeypatch, *, location=SHARE_CANONICAL, body=SHARE_VIDEO_BODY):
    def fake_run(cmd, **kwargs):
        if "-I" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"HTTP/2 302\nlocation: {location}\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout=body, stderr="")
    monkeypatch.setattr(detection.subprocess, "run", fake_run)


def test_share_link_resolves_to_canonical_url(monkeypatch):
    _share_redirect_run(monkeypatch)
    resolved = detection._resolve_threads_share_redirect(SHARE_URL, [], timeout=15)
    assert resolved == SHARE_CANONICAL


def test_share_link_video_detected_as_threads_video(monkeypatch, tmp_path):
    _share_redirect_run(monkeypatch)
    # Keep _resolve_threads_cookie_path off the real /root/kb-agent/cookies —
    # on CI the runner is not root and os.stat there raises PermissionError.
    monkeypatch.setattr(detection, "_resolve_threads_cookie_path", lambda: tmp_path / "threads_cookies.txt")
    monkeypatch.setattr(detection, "_curl_netscape_cookie_config", lambda p: ([], None))
    subtype, fetch_failed = detection._detect_threads_subtype(SHARE_URL)
    assert subtype == "threads_video"
    assert fetch_failed is False


def test_share_link_redirect_rejects_off_origin_host(monkeypatch):
    _share_redirect_run(monkeypatch, location="https://evil.example.com/steal")
    with pytest.raises(ValueError, match="not allowlisted"):
        detection._resolve_threads_share_redirect(SHARE_URL, [], timeout=15)


def test_canonical_threads_url_not_probed(monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
    monkeypatch.setattr(detection.subprocess, "run", fake_run)
    resolved = detection._resolve_threads_share_redirect(SHARE_CANONICAL, [], timeout=15)
    assert resolved == SHARE_CANONICAL
    assert calls == []


def test_share_link_subtype_scopes_on_resolved_url(monkeypatch):
    # _fetch_content returns the URL it actually fetched (the resolved canonical
    # post); the submit-time path must scope on that post code, not the share
    # short code (which is absent from the canonical document).
    captured = {}
    monkeypatch.setattr(detection, "_fetch_content", lambda u, p: ("html", SHARE_CANONICAL))

    def fake_subtype(html, source_url):
        captured["source_url"] = source_url
        return "threads_text"

    monkeypatch.setattr(detection, "threads_content_subtype_from_html", fake_subtype)
    detection.detect_content_subtype(SHARE_URL, "threads")
    assert captured["source_url"] == SHARE_CANONICAL


def test_comment_id_routes_to_comment():
    assert detection.detect_content_subtype(COMMENT_URL, "instagram_reel") == "instagram_comment"


# ── URL normalization ───────────────────────────────────────────────────────

def test_comment_id_preserved():
    # comment_id is semantic — must survive normalization
    assert detection.normalize_url(COMMENT_URL) == COMMENT_URL


def test_tracking_params_stripped_comment_id_kept():
    got = detection.normalize_url("https://www.instagram.com/reel/ID/?igsh=abc&comment_id=42&igsi=z")
    assert "comment_id=42" in got
    assert "igsh" not in got and "igsi" not in got


def test_youtube_v_param_untouched():
    assert (
        detection.normalize_url("https://www.youtube.com/watch?v=abc123DEF45&list=PLx")
        == "https://www.youtube.com/watch?v=abc123DEF45&list=PLx"
    )


@pytest.mark.parametrize(
    "bad_url",
    [
        "javascript:alert(1)",
        "JavaScript:alert(document.cookie)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
    ],
)
def test_normalize_url_rejects_non_http_schemes(bad_url):
    with pytest.raises(ValueError, match="http or https"):
        detection.normalize_url(bad_url)


# ── Video extraction (null-skip) ────────────────────────────────────────────

def test_extract_array_skips_null_occurrence():
    arr = _extract_json_array_after_key(NULL_THEN_REAL, "video_versions")
    assert arr and '"url"' in arr


def test_extract_video_url_null_then_real():
    assert _extract_video_url(NULL_THEN_REAL) is not None


def test_extract_video_url_all_null_is_none():
    assert _extract_video_url(NULL_VIDEO_JSON) is None


def test_extract_video_url_scoped_to_task_post_not_first_unrelated_array():
    """Two non-null video_versions arrays: pick the one nearest the task post id."""
    unrelated = (
        '"video_versions":[{"url":"https://scontent.cdninstagram.com/v/UNRELATED.mp4",'
        '"height":720}]'
    )
    task_marker = (
        '"code":"Da3I_d0ls4E",'
        '"permalink":"https:\\/\\/www.threads.com\\/\\u0040bourne_ti\\/post\\/Da3I_d0ls4E"'
    )
    task_video = (
        '"video_versions":[{"url":"https://scontent.cdninstagram.com/v/TASK.mp4",'
        '"height":1080}]'
    )
    html = unrelated + ',"suggested":true,' + task_marker + "," + task_video
    task_url = "https://www.threads.com/@bourne_ti/post/Da3I_d0ls4E"
    url = _extract_video_url(html, task_url)
    assert url is not None
    assert url.endswith("TASK.mp4")
    assert "UNRELATED" not in url


# ── Note code-fence strip ───────────────────────────────────────────────────

def test_strip_code_fence_yaml():
    out = strip_code_fence("```yaml\ndate: 2026-11-05\n---\n# T\nbody\n```")
    assert out.startswith("---\ndate: 2026-11-05")
    assert "```" not in out


def test_strip_code_fence_restores_missing_delimiter():
    # the LLM emits the fence in place of the opening --- (no opening delimiter)
    out = strip_code_fence("```yaml\ndate: 2026-11-05\nsource: x\n---\nbody")
    assert out.startswith("---\ndate: 2026-11-05")


def test_strip_code_fence_adds_closing_delimiter():
    # frontmatter with opening --- but missing the closing --- before the heading
    out = strip_code_fence("---\ndate: 2026-11-05\ntags: [x]\n\n# Title\nbody")
    assert "---\ndate: 2026-11-05\ntags: [x]\n---" in out


def test_strip_code_fence_noop():
    assert strip_code_fence("---\ndate: x\n---\nbody") == "---\ndate: x\n---\nbody\n"


# ── Shortcode decode ────────────────────────────────────────────────────────

def test_shortcode_to_media_id():
    # base-64 decode over A-Za-z0-9-_ ; the comment-API media id
    assert _shortcode_to_media_id("DdnESNvo13o") == 3992178438581738984


# ── Author extraction ───────────────────────────────────────────────────────

def test_extract_author_uses_permalink_not_session_user():
    # the first "username" is the logged-in session user, not the post author
    from agent.pipelines.threads import _extract_author

    html = '"username":"lieutarchitect","permalink":"https:\\/\\/www.threads.com\\/\\u0040tayjmcm\\/post\\/ABC"'
    assert _extract_author(html) == "tayjmcm"


# ── Quality gates (real note shapes) ────────────────────────────────────────

# A Threads video note: Key Points with indented quotes, no Concepts section.
THREADS_VIDEO_NOTE = """---
date: 2026-11-05
source: threads.net (@tayjmcm)
---

# A Threads Video Note

## Summary
Core message in a couple of sentences.

## Key Points
- First key point
  > "First direct quote from the transcript"
- Second key point
  > "Second direct quote"
- Third key point
  > "Third direct quote"

## Speaker Analysis
Who is speaking and their perspective.

## Full Transcript
The complete transcript with speaker labels.

## Context
Platform context.

## Critical Analysis
What is new versus restated.

## Reliability
Confidence level with justification.
"""

# A Threads text note: bullet-wrapped quotes.
THREADS_TEXT_NOTE = """---
date: 2026-11-05
source: threads.net (@exit.notes)
---

# A Threads Text Note

## Summary
Core message.

## Key Claims
- > first direct quote
- > second direct quote
- > third direct quote

## Context
Platform context.

## Critical Analysis
What is new versus restated.
"""


def test_threads_video_gate_accepts_valid_note(tmp_path):
    from agent.config import Config
    from agent.quality import quality_gate_threads_video

    p = tmp_path / "note.md"
    p.write_text(THREADS_VIDEO_NOTE)
    passed, failures = quality_gate_threads_video(p, Config())
    assert passed, failures


# A real music-only Instagram reel note: the audio was empty, so the caption is
# the content, carried verbatim under "## Post Description".
REEL_CAPTION_NOTE = """---
date: 2026-11-06
source: instagram.com
source_url: https://www.instagram.com/reel/Da-vuxHN19S/
confidence: plausible
tags: [instagram, reel]
---

# Natural Deworming Routine Promoted with Neem and Senna

## Summary
A music-only reel whose post description promotes a natural deworming routine.

## Post Description
> Vu tout ce qu'on mange, se déparasiter régulièrement me semble nécessaire...
>
> Je vous partage ma routine régulière de déparasitage au naturel.

## Key Points
- Recommends a one-week plant-based vermifuge
- Lists neem leaves, Artemisia, and soursop as the active plants
- Directs viewers to a commercial site to buy the herbs

## Speaker Analysis
Music-only reel, no speech.

## Reliability
Low — caption is the only source.
"""


def test_music_only_reel_caption_note_passes_gate(tmp_path):
    # The real failure: music-only reels produced empty notes. With the caption
    # captured verbatim, the note must pass the short-form (key-point) gate.
    from agent.config import Config
    from agent.quality import quality_gate_threads_video

    p = tmp_path / "note.md"
    p.write_text(REEL_CAPTION_NOTE)
    passed, failures = quality_gate_threads_video(p, Config())
    assert passed, failures


def test_reel_note_has_verbatim_post_description_section():
    # The caption must be a dedicated verbatim section, not paraphrased away.
    assert "## Post Description" in REEL_CAPTION_NOTE
    assert "> Vu tout ce qu'on mange" in REEL_CAPTION_NOTE


# ── Video codec normalization (browser compatibility) ────────────────────────

def test_ensure_h264_transcodes_vp9(tmp_path):
    # Real failure: an Instagram reel archived as VP9-in-.mp4 wouldn't play in
    # the browser. ensure_h264 must normalize it to H.264.
    import shutil
    import subprocess as sp
    from agent.pipelines.video_common import ensure_h264

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        import pytest as _pytest
        _pytest.skip("ffmpeg/ffprobe not available")

    vp9 = tmp_path / "reel.mp4"
    sp.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=size=64x64:rate=1:color=black",
         "-t", "1", "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", str(vp9)],
        capture_output=True, check=True,
    )
    out = ensure_h264(vp9)
    assert out != vp9
    codec = sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    ).stdout.strip()
    assert codec == "h264"


def test_ensure_h264_passthrough_h264(tmp_path):
    import shutil
    import subprocess as sp
    from agent.pipelines.video_common import ensure_h264

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        import pytest as _pytest
        _pytest.skip("ffmpeg/ffprobe not available")

    h264 = tmp_path / "reel.mp4"
    sp.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=size=64x64:rate=1:color=black",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264)],
        capture_output=True, check=True,
    )
    assert ensure_h264(h264) == h264


def test_threads_text_note_quotes_are_counted():
    from agent.quality import count_quotes

    assert count_quotes(THREADS_TEXT_NOTE.split("\n")) >= 2


def test_count_quotes_tolerates_bullet_marker():
    from agent.quality import count_quotes

    assert count_quotes(["> own line", "  > indented", "- > bullet"]) == 3


# ── Storage seam: notes store keys, never absolute URLs (Tigris migration) ───

def test_write_source_files_block_emits_keys(tmp_path):
    """write_source_files_block must write object keys, not host-prefixed URLs."""
    from agent.archival import write_source_files_block

    note = tmp_path / "note.md"
    note.write_text(
        "---\ndate: 2026-01-01\nsource_files:\n"
        "  video: http://old-host/knowledge-base/videos/x.mp4\n---\n# Note\n",
        encoding="utf-8",
    )

    class _Cfg:
        storage_bucket = "knowledge-base"

    write_source_files_block(note, ["videos/x.mp4", "transcripts/x.json"], _Cfg())
    content = note.read_text(encoding="utf-8")
    assert "video: videos/x.mp4" in content
    assert "transcript: transcripts/x.json" in content
    assert "http://old-host" not in content


def test_upload_object_returns_key(tmp_path):
    """upload_object returns the bucket-stripped object key."""
    from unittest.mock import MagicMock, patch

    from agent.archival import upload_object

    src = tmp_path / "raw.md"
    src.write_text("# hi", encoding="utf-8")
    with patch("agent.archival.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        cfg = Config(storage_bucket="knowledge-base", storage_alias="tigris")
        key = upload_object(src, "knowledge-base/articles/slug.md", cfg)
    assert key == "articles/slug.md"
