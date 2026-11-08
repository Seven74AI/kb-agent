"""Instagram pipeline — reel video and carousel post extraction.

Routing:
- /reel/ → video pipeline (yt-dlp download → dual audio → diarize →
  transcribe → merge → synthesize → quality → MinIO → Git)
- /p/ → carousel pipeline (yt-dlp metadata → Playwright alt text extraction →
  LLM synthesis → quality → MinIO → Git)

Pitfalls handled:
- Cookie validation: grep -c sessionid >= 1
- Rate limiting: yt-dlp --sleep-requests 3 --sleep-interval 5 --max-sleep-interval 15 --limit-rate 2M
- Music-only reels: post-transcription segments < 50 chars → use caption as primary content
- Carousel hard cap: 2 slides max (Playwright alt text extraction), slides 3+ not extractable
"""

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent.archival import append_note_tags
from agent.config import Config
from agent.curl_config import (
    curl_config_header_line,
    curl_config_quoted_assignment,
)
from agent.detection import _safe_url_for_log
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.db import init_db, LOGS_SCHEMA
from agent.exceptions import ErrorType, StepError
from agent.redact import redact_http_urls_in_text, url_for_display, write_merged_transcript_json, write_vault_note
from agent.notify import sanitize_visible_message
from agent.llm import get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also

logger = logging.getLogger(__name__)

from agent.pipelines.video_common import (
    _artifact_stem,
    _slug_path,
    ensure_h264,
    format_transcript_for_llm,
    merge_speaker_labels,
    run_threads_video_quality_gate,
    step_diarize,
    step_extract_audio,
    step_fetch_transcript,
    step_transcribe,
    strip_code_fence,
    upload_note_archive,
    upload_video_source,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# yt-dlp binary — resolved explicitly so it works regardless of PATH (the MCP
# server runs in a separate venv with system PATH, no yt-dlp).
YTDLP_BIN = shutil.which("yt-dlp") or "/usr/local/lib/hermes-agent/venv/bin/yt-dlp"

# yt-dlp rate limiting for Instagram (per spec)
# yt-dlp is a delegated fetcher: only the user-supplied input URL is validated via
# fetch_guard. Segment / CDN URLs yt-dlp opens internally are outside fetch_guard
# (accepted residual, same class as Flaresolverr).


def _ytdlp_subprocess_env(config: Config) -> dict[str, str]:
    """Load IG cookies via XDG yt-dlp config so paths stay off subprocess argv."""
    cookie_path = config.cookie_dir / "ig_cookies.txt"
    xdg_root = Path(config.temp_dir) / "ytdlp_xdg"
    conf_dir = xdg_root / "yt-dlp"
    conf_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(xdg_root, 0o700)
    os.chmod(conf_dir, 0o700)
    conf_path = conf_dir / "config"
    conf_path.write_text(f"--cookies\n{cookie_path}\n", encoding="utf-8")
    os.chmod(conf_path, 0o600)
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(xdg_root)
    return env


IGDL_RATE_FLAGS = [
    "--sleep-requests", "3",
    "--sleep-interval", "5",
    "--max-sleep-interval", "15",
    "--limit-rate", "2M",
]

# Music-only detection threshold (total chars across all segments)
MUSIC_ONLY_CHAR_THRESHOLD = 50

# Hard cap for carousel slides (Playwright alt text extraction)
CAROUSEL_SLIDE_CAP = 2


def _download_artifact_stem(task: dict) -> str:
    """Temp-dir stem for download-phase artifacts (id-keyed in nightly batch)."""
    from agent.pipelines.video_common import _artifact_stem

    return _artifact_stem(task)


def _carousel_meta_path(config: Config, task: dict) -> Path:
    """Canonical temp path for carousel metadata JSON (writers/readers/cleanup)."""
    return Path(config.temp_dir) / f"{_artifact_stem(task)}_carousel_meta.json"


def _carousel_slides_path(config: Config, task: dict) -> Path:
    """Canonical temp path for carousel slides JSON (writers/readers/cleanup)."""
    return Path(config.temp_dir) / f"{_artifact_stem(task)}_carousel_slides.json"


def _carousel_playwright_script_path(config: Config, task: dict) -> Path:
    """Ephemeral Playwright script path (carousel extract; cleaned in step)."""
    return Path(config.temp_dir) / f"{_artifact_stem(task)}_playwright.py"


def _ig_comment_sidecar_path(config: Config, task: dict) -> Path:
    """Canonical temp path for ig_comment JSON (writers/readers/cleanup)."""
    return Path(config.temp_dir) / f"{_artifact_stem(task)}_ig_comment.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_ig_cookies(cookie_dir: Path | None = None) -> None:
    """Validate Instagram cookies before any operation.

    Checks: sessionid present in cookie file (grep -c sessionid >= 1).

    Args:
        cookie_dir: Directory containing ig_cookies.txt. Defaults to
            /root/kb-agent/cookies (matching config.cookie_dir default).

    Raises:
        StepError(PAUSE): If cookies invalid or missing.
    """
    if cookie_dir is None:
        cookie_dir = Path("/root/kb-agent/cookies")
    cookie_path = cookie_dir / "ig_cookies.txt"
    if not cookie_path.exists():
        raise StepError(
            f"Instagram cookies not found: {cookie_path}",
            error_type=ErrorType.PAUSE,
            step_name="cookie_validate",
            details=f"Export cookies from Chrome to {cookie_path}",
        )

    cookies_text = cookie_path.read_text()
    sessionid_count = cookies_text.count("sessionid")
    if sessionid_count < 1:
        raise StepError(
            "Instagram cookies missing sessionid",
            error_type=ErrorType.PAUSE,
            step_name="cookie_validate",
            details="Re-export cookies from Chrome — sessionid is required",
        )


def _is_music_only(transcript: dict) -> bool:
    """Detect if a reel is music-only (no real speech).

    Post-transcription: if total chars across all segments < threshold,
    the reel is likely music/BGM-only.

    Args:
        transcript: Transcript dict with segments.

    Returns:
        True if the reel appears to be music-only.
    """
    total_chars = sum(
        len(seg.get("text", "").strip())
        for seg in transcript.get("segments", [])
    )
    return total_chars < MUSIC_ONLY_CHAR_THRESHOLD


def _fetch_ig_caption(url: str, config: Config) -> str:
    """Fetch an Instagram post caption via yt-dlp %(description)s.

    The caption (post description) often carries the substantive content for
    music-only reels and adds context for speech reels alike. Degrades to ""
    on any failure (stale cookies, private post, timeout).
    """
    _assert_fetch_url_allowed(url)
    try:
        result = subprocess.run(
            [
                YTDLP_BIN,
                "--js-runtimes", "node",
                *IGDL_RATE_FLAGS,
                "--print", "%(description)s",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=_ytdlp_subprocess_env(config),
        )
        if result.returncode == 0:
            caption = result.stdout.strip()
            if caption and caption.lower() not in ("none", "null"):
                return caption
        detail = sanitize_visible_message(
            result.stderr.strip()[:500] or f"yt-dlp exit {result.returncode}",
        )
        logger.warning(
            "IG caption fetch failed for %s: %s",
            _safe_url_for_log(url),
            detail,
        )
    except Exception as exc:
        logger.warning(
            "IG caption fetch failed for %s: %s",
            _safe_url_for_log(url),
            sanitize_visible_message(str(exc)),
        )
    return ""


# ── Reel Steps ────────────────────────────────────────────────────────────────

def step_ig_reel_validate(task: dict, config: Config) -> str:
    """Validate Instagram cookies for reel processing.

    Args:
        task: Task dict.
        config: Config dataclass.

    Returns:
        "cookies_valid"
    """
    _validate_ig_cookies(cookie_dir=config.cookie_dir)
    return "cookies_valid"


def step_ig_reel_download(task: dict, config: Config) -> str:
    """Download Instagram reel video via yt-dlp.

    Uses cookies + rate limiting. Saves to temp dir.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the downloaded video file.
    """
    _validate_ig_cookies(cookie_dir=config.cookie_dir)

    from agent.pipelines.video_common import instagram_reel_ytdlp_output

    url = task["source_url"]
    stem = _download_artifact_stem(task)
    batch = task.get("_batch_ingest")
    output_path, ytdlp_out, glob_pattern = instagram_reel_ytdlp_output(task, config)

    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="reel_download",
        ) from exc
    result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--output", ytdlp_out,
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=_ytdlp_subprocess_env(config),
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Sign in" in stderr or "login" in stderr.lower():
            raise StepError(
                "Instagram auth failed — sessionid may be expired",
                error_type=ErrorType.PAUSE,
                step_name="reel_download",
                details="Re-export cookies from Chrome",
            )
        if "Private" in stderr or "unavailable" in stderr.lower():
            raise StepError(
                "Reel is private or unavailable",
                error_type=ErrorType.FAIL,
                step_name="reel_download",
            )
        raise StepError(
            f"yt-dlp reel download failed: {stderr[:200]}",
            error_type=ErrorType.RETRY,
            step_name="reel_download",
        )

    # yt-dlp may add extension automatically; find the actual file
    actual_files = [
        p for p in Path(config.temp_dir).glob(glob_pattern)
        if p.is_file()
        and (not batch or p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov"))
    ]
    if not actual_files:
        raise StepError(
            f"No output file found for stem {stem}",
            error_type=ErrorType.RETRY,
            step_name="reel_download",
        )

    output_path = actual_files[0]

    if output_path.stat().st_size < 1024:
        raise StepError(
            f"Downloaded reel too small ({output_path.stat().st_size} bytes)",
            error_type=ErrorType.FAIL,
            step_name="reel_download",
        )

    # Instagram sometimes serves VP9/HEVC in an .mp4 container — browsers
    # can't play it. Normalize to H.264 so the archived video always renders.
    output_path = ensure_h264(output_path)

    return str(output_path)


def step_ig_reel_merge(task: dict, config: Config) -> str:
    """Merge diarization speaker labels with transcript segments.

    Merges at segment midpoint. Performs music-only detection.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string summarizing merge result (includes music_only flag).
    """
    stem = _artifact_stem(task)
    diar_path = Path(config.temp_dir) / f"{stem}_diarization.json"
    trans_path = Path(config.temp_dir) / f"{stem}_transcript.json"

    if not diar_path.exists():
        raise FileNotFoundError(f"Diarization not found: {diar_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transcript not found: {trans_path}")

    diar = json.loads(diar_path.read_text())
    trans = json.loads(trans_path.read_text())

    # Music-only detection BEFORE merge (avoid unnecessary work)
    music_only = _is_music_only(trans)

    merge_speaker_labels(diar, trans)

    # Count distinct speakers
    speakers = set(
        s.get("speaker", "Unknown") for s in trans.get("segments", [])
    )

    trans["music_only"] = music_only

    # Write merged
    merged_path = Path(config.temp_dir) / f"{stem}_merged.json"
    write_merged_transcript_json(merged_path, trans)

    return json.dumps({
        "segments": len(trans.get("segments", [])),
        "speakers": len(speakers),
        "music_only": music_only,
        "merged_path": str(merged_path),
    })


def step_ig_reel_synthesize(task: dict, config: Config) -> str:
    """LLM synthesis for Instagram reel videos.

    Music-only detection: if transcript < 50 chars total, uses caption
    as primary content with warning annotation.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    stem = _artifact_stem(task)
    source_url = task.get("source_url", "")
    display_source_url = url_for_display(source_url)

    merged_path = Path(config.temp_dir) / f"{stem}_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged transcript not found: {merged_path}")

    merged = json.loads(merged_path.read_text())
    music_only = merged.get("music_only", False)

    # Fetch the post caption/description — carries substantive content for
    # music-only reels and adds context for speech reels alike.
    caption = _fetch_ig_caption(source_url, config)

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    # Build compact transcript
    transcript_text = _format_transcript_for_llm(merged)
    date_str = datetime.now().strftime("%Y-%m-%d")
    if music_only:
        # Music-only reel: use caption as primary content
        system_prompt = f"""You are a knowledge base curator processing an Instagram reel.

WARNING: This reel is music-only — the audio contains no meaningful speech.
The transcript is provided for completeness but the primary content
comes from the reel's caption/description, which is provided in the
user prompt under "POST CAPTION / DESCRIPTION".

SOURCE: Instagram reel ({display_source_url})
DATE: {date_str}

TASK: Write a note that:
1. Notes this is a music/BGM reel with minimal speech
2. Includes the provided caption VERBATIM in a blockquote under a `## Post Description` heading
3. Analyzes the caption as the primary content
4. Describes what can be observed from the transcript (if any words were detected)

OUTPUT FORMAT:
- YAML frontmatter: date, source: instagram.com, source_url: {display_source_url}, minio placeholder,
  confidence: low (music-only reel), tags: [instagram, reel, music]
- Note sections: Title, Summary, Post Description (verbatim caption), Caption Analysis,
  Transcript Notes, Reliability (note music-only limitation)"""
    else:
        system_prompt = f"""You are a knowledge base curator specialized in Instagram reel content.

SOURCE: An Instagram reel video transcript plus its post caption/description.

INSTAGRAM CONTEXT:
- This is short-form social media content — typically 15-90 seconds
- Format: vertical video, often trending audio + visual content
- The post caption/description is provided separately and may carry the primary message (hashtags, claims, links)

TASK: Write a concise but thorough analysis note.

SECTIONS:
1. YAML frontmatter
2. Title — descriptive
3. Summary — 2-3 sentences
4. Post Description — the caption text VERBATIM in a blockquote (>), under a `## Post Description` heading; if no caption, write "No post description."
5. Key Points — ≥2, with direct quotes from transcript
6. Speaker Analysis — who is speaking, perspective
7. Context — platform context, trends if identifiable
8. Critical Analysis — what's being communicated, what's omitted
9. Reliability — confidence assessment

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: instagram.com
  source_url: {display_source_url}
  source_files:
    video: videos/{safe_slug}.mp4
    transcript: transcripts/{safe_slug}.json
  confidence: plausible
  tags: [instagram, reel]
- Then sections as above."""

    caption_block = (
        f"POST CAPTION / DESCRIPTION:\n{caption}\n=== END CAPTION ==="
        if caption
        else "POST CAPTION / DESCRIPTION: (none detected)"
    )
    user_prompt = f"""TRANSCRIPT:
{transcript_text}

=== END TRANSCRIPT ===

{caption_block}

Write a note following the template above.
{'This is a music-only reel — prioritize any detected caption text over the transcript.' if music_only else 'Analyze the full transcript for key points.'}"""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = strip_code_fence(response.content)

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    write_vault_note(note_path, note_content)
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


# ── Carousel Steps ────────────────────────────────────────────────────────────

def step_ig_carousel_metadata(task: dict, config: Config) -> str:
    """Extract Instagram carousel metadata via yt-dlp.

    Gets: caption, likes, author, playlist_count (slide count).

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to metadata JSON file.
    """
    _validate_ig_cookies(cookie_dir=config.cookie_dir)

    url = task["source_url"]

    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="ig_carousel_metadata",
        ) from exc

    # Extract metadata via yt-dlp print
    result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--print", "%(title)s||%(uploader)s||%(like_count)s||%(playlist_count)s",
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_ytdlp_subprocess_env(config),
    )

    metadata = {
        "title": "",
        "uploader": "",
        "like_count": 0,
        "playlist_count": 0,
    }

    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("||")
        if len(parts) >= 1:
            metadata["title"] = redact_http_urls_in_text(parts[0])
        if len(parts) >= 2:
            metadata["uploader"] = redact_http_urls_in_text(parts[1])
        if len(parts) >= 3:
            try:
                metadata["like_count"] = int(parts[2])
            except (ValueError, TypeError):
                pass
        if len(parts) >= 4:
            try:
                metadata["playlist_count"] = int(parts[3])
            except (ValueError, TypeError):
                pass

    # Also try to get caption via description
    caption_result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--print", "%(description)s",
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_ytdlp_subprocess_env(config),
    )

    if caption_result.returncode == 0:
        metadata["caption"] = redact_http_urls_in_text(caption_result.stdout.strip())

    # Write metadata
    meta_path = _carousel_meta_path(config, task)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return str(meta_path)


def step_ig_carousel_extract(task: dict, config: Config) -> str:
    """Extract alt text from carousel slides via Playwright.

    Hard cap: 2 slides. Slides 3+ are NOT extractable (anti-bot measures).
    Saves alt texts to JSON. No vision_analyze needed (alt text is content).

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to extracted slides JSON file.
    """
    url = task["source_url"]

    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="ig_carousel_extract",
        ) from exc

    # Persisted script must not embed credential-bearing query strings from the task URL.
    display_nav_url = url_for_display(url)

    # Build Playwright script for alt text extraction
    playwright_script = f"""import asyncio, json, sys
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        )
        page = await context.new_page()

        await page.goto({json.dumps(display_nav_url)}, wait_until="networkidle", timeout=30000)

        await page.wait_for_timeout(3000)
        
        slides = []
        # Find all img elements — Instagram carousel slides
        images = await page.query_selector_all('article img')
        
        slide_num = 0
        for img in images:
            if slide_num >= {CAROUSEL_SLIDE_CAP}:
                break
            
            alt = await img.get_attribute('alt') or ''
            src = await img.get_attribute('src') or ''
            
            if src and (alt or slide_num < {CAROUSEL_SLIDE_CAP}):
                slides.append({{
                    "slide": slide_num + 1,
                    "alt_text": alt,
                    "src": src,
                }})
                slide_num += 1
        
        await browser.close()
        # Output slides JSON to stdout — captured by the pipeline via subprocess.run().stdout
        print(json.dumps(slides))
        
        if slide_num < {CAROUSEL_SLIDE_CAP}:
            sys.stderr.write(f"WARNING: Only found {{slide_num}}/{CAROUSEL_SLIDE_CAP} slides\\n")

asyncio.run(main())
"""

    script_path = _carousel_playwright_script_path(config, task)
    script_path.write_text(playwright_script)

    try:
        result = subprocess.run(
            ["python3", str(script_path)],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            raise StepError(
                "Playwright carousel extraction failed",
                error_type=ErrorType.RETRY,
                step_name="ig_carousel_extract",
                details=sanitize_visible_message(result.stderr.strip()[:500]),
            )

        try:
            slides = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            raise StepError(
                "Failed to parse Playwright output for carousel",
                error_type=ErrorType.RETRY,
                step_name="ig_carousel_extract",
                details=sanitize_visible_message(result.stdout.strip()[:500]),
            )

        # Write extracted slides
        slides_path = _carousel_slides_path(config, task)
        slides_persisted = [
            {
                "slide": s["slide"],
                "alt_text": redact_http_urls_in_text(s.get("alt_text", "")),
            }
            for s in slides
        ]
        slides_data = {
            "slides": slides_persisted,
            "slide_count": len(slides_persisted),
            "cap_note": f"Hard cap: {CAROUSEL_SLIDE_CAP} slides. Slides beyond {CAROUSEL_SLIDE_CAP} are not extractable.",
        }
        slides_path.write_text(json.dumps(slides_data, indent=2, ensure_ascii=False))

        return str(slides_path)
    finally:
        try:
            if script_path.is_file():
                script_path.unlink()
        except OSError:
            pass


def step_ig_carousel_synthesize(task: dict, config: Config) -> str:
    """LLM synthesis for Instagram carousel posts.

    Uses alt text from extracted slides + metadata caption.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    source_url = task.get("source_url", "")
    display_source_url = url_for_display(source_url)

    # Load carousel data
    slides_path = _carousel_slides_path(config, task)
    if not slides_path.exists():
        raise FileNotFoundError(f"Carousel slides not found: {slides_path}")

    slides_data = json.loads(slides_path.read_text())

    # Load metadata
    meta_path = _carousel_meta_path(config, task)
    caption = ""
    uploader = ""
    like_count = 0
    playlist_count = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        caption = meta.get("caption", "")
        uploader = meta.get("uploader", "")
        like_count = meta.get("like_count", 0)
        playlist_count = meta.get("playlist_count", 0)

    # Build content for LLM
    slides_text = "\n\n".join(
        f"Slide {s['slide']}: {s.get('alt_text', '(no alt text)')}"
        for s in slides_data.get("slides", [])
    )

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    date_str = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""You are a knowledge base curator specialized in Instagram carousel posts.

SOURCE: An Instagram carousel post by @{uploader}.

CONTEXT:
- {playlist_count} slides in carousel (extracted {len(slides_data.get('slides', []))})
- {like_count} likes
- Caption: {caption[:500] if caption else '(no caption)'}

TASK: Write a comprehensive note analyzing this carousel post.
Each slide's alt text is provided — this IS the slide content.

SECTIONS:
1. YAML frontmatter
2. Title — descriptive title
3. Summary — 2-3 sentences capturing the carousel's message
4. Slide-by-Slide Analysis — for each slide, describe its content from alt text
5. Key Themes — patterns across slides
6. Context — who posted, why, platform context
7. Critical Analysis — strengths, weaknesses, what's communicated
8. Reliability — confidence level

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: instagram.com (@{uploader})
  source_url: {display_source_url}
  source_files:
    raw: <filled by archival step>
  confidence: plausible
  tags: [instagram, carousel]
- Then sections as above."""

    user_prompt = f"""CAPTION:
{caption if caption else '(no caption)'}

SLIDES ({len(slides_data.get('slides', []))} of {playlist_count} extracted):
{slides_text}

=== END SLIDES ===

Write a note analyzing this carousel post.
Note: only {len(slides_data.get('slides', []))} of {playlist_count} slides could be extracted (anti-bot limitation on slides 3+)."""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = strip_code_fence(response.content)

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    write_vault_note(note_path, note_content)
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


# ── Quality Gate ──────────────────────────────────────────────────────────────

def step_ig_quality_gate(task: dict, config: Config) -> str:
    """Run quality gates on the generated Instagram note.

    Uses YouTube quality gate for reel videos.
    Uses simpler size check for carousel posts.

    Args:
        task: Task dict with slug, content_type.
        config: Config dataclass.

    Returns:
        Status string: "passed" or description of failures.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    content_type = task.get("content_type", "instagram_post")

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    if not note_path.exists():
        raise FileNotFoundError(f"Note not found: {note_path}")

    if content_type in ("instagram_reel", "instagram_reel_synthesize"):
        # Reels are short-form — key points only, not the YouTube Concepts gate.
        passed, rounds = run_threads_video_quality_gate(note_path, config, task["id"])
        if not passed:
            content = note_path.read_text(encoding="utf-8")
            raise StepError(
                f"Instagram reel quality gate failed after {rounds} rounds. "
                f"First line: {content.split(chr(10))[0]}",
                error_type=ErrorType.FAIL,
                step_name="ig_quality_gate",
            )
        return json.dumps({"passed": True, "rounds": rounds})
    elif content_type == "instagram_comment":
        # Comment note: must capture the comment + parent context.
        content = note_path.read_text(encoding="utf-8")
        size = len(content)
        failures = []
        if size < 1500:
            failures.append(f"Note too short: {size} chars < 1500")
        if "## The Comment" not in content:
            failures.append("Missing The Comment section")
        if "## Parent Context" not in content:
            failures.append("Missing Parent Context section")
        if failures:
            raise StepError(
                f"Instagram comment quality gate failed: {'; '.join(failures)}",
                error_type=ErrorType.FAIL,
                step_name="ig_quality_gate",
            )
        return json.dumps({"passed": True, "failures": [], "rounds": 1})
    else:
        # Carousel: simple size + section check
        content = note_path.read_text(encoding="utf-8")
        size = len(content)

        # Must have Slide-by-Slide Analysis section
        has_slide_analysis = "## Slide-by-Slide Analysis" in content or "## Slide-by-Slide" in content

        failures = []
        if size < 2000:
            failures.append(f"Note too short: {size} chars < 2000")
        if not has_slide_analysis:
            failures.append("Missing Slide-by-Slide Analysis section")

        if failures:
            raise StepError(
                f"Instagram quality gate failed: {'; '.join(failures)}",
                error_type=ErrorType.FAIL,
                step_name="ig_quality_gate",
            )

        return json.dumps({
            "passed": True,
            "failures": [],
            "rounds": 1,
        })


# ── Cleanup ───────────────────────────────────────────────────────────────────

def step_ig_cleanup(task: dict, config: Config) -> str:
    """Clean up temporary files after successful pipeline execution.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Status message listing removed files.
    """
    from agent.pipelines.video_common import (
        _artifact_stem,
        instagram_reel_glob_pattern,
        resolve_instagram_reel_video,
    )

    stem = _artifact_stem(task)
    content_type = task.get("content_type", "instagram_post")

    paths_to_unlink: list[Path] = []
    if content_type == "instagram_reel":
        batch = task.get("_batch_ingest")
        glob_pattern = instagram_reel_glob_pattern(task)
        for p in Path(config.temp_dir).glob(glob_pattern):
            if p.is_file() and (
                not batch or p.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")
            ):
                paths_to_unlink.append(p)
        reel_default = resolve_instagram_reel_video(config, task)
        if reel_default not in paths_to_unlink and reel_default.exists():
            paths_to_unlink.append(reel_default)
        for ext in ["_16k.wav", "_8k.wav"]:
            path = Path(config.temp_dir) / f"{stem}{ext}"
            if path.exists():
                paths_to_unlink.append(path)
    for ext in ["_playwright.py", "_carousel_slides.json", "_carousel_meta.json",
                "_ig_comment.json"]:
        path = Path(config.temp_dir) / f"{stem}{ext}"
        if path.exists():
            paths_to_unlink.append(path)

    source_url = task.get("source_url", "")
    parsed = urlparse(source_url)
    qs = parse_qs(parsed.query)
    comment_id = (qs.get("comment_id") or [""])[0]
    if comment_id:
        try:
            path_token = _safe_comment_id_path_token(comment_id)
            conf_path = Path(config.cookie_dir) / f".curl-ig-comment-{path_token}.conf"
            if conf_path.is_file():
                paths_to_unlink.append(conf_path)
        except StepError:
            pass

    removed: list[str] = []
    for path in paths_to_unlink:
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass

    ytdlp_xdg = Path(config.temp_dir) / "ytdlp_xdg"
    if ytdlp_xdg.exists():
        shutil.rmtree(ytdlp_xdg, ignore_errors=True)
        removed.append("ytdlp_xdg")

    return f"Cleaned up: {', '.join(removed)}" if removed else "Nothing to clean up"


# ── MinIO Upload ──────────────────────────────────────────────────────────────

def step_ig_minio_upload(task: dict, config: Config) -> str:
    """Upload the generated reel/post artifacts to MinIO.

    Reel: source-archival (video + transcript). Post/carousel: note-archival.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    content_type = task.get("content_type", "instagram_post")
    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"

    if content_type == "instagram_reel":
        from agent.pipelines.video_common import resolve_instagram_reel_video

        # Source-archival: video + transcript (matches the batch collector).
        from agent.pipelines.video_common import _resolve_slug

        storage_slug = _resolve_slug(task)
        video_path = resolve_instagram_reel_video(config, task)
        dest_ext = video_path.suffix.lower().lstrip(".") or "mp4"
        uploaded = upload_video_source(storage_slug, config, video_path, dest_ext, note_path)
        return f"Uploaded: {', '.join(uploaded)}"

    # Note-archival: post/carousel.
    return upload_note_archive(safe_slug, config, note_path, "instagram")


# ── Git Push ──────────────────────────────────────────────────────────────────

def step_ig_git_push(task: dict, config: Config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Delegates to archival.step_git_push with instagram-specific params.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Commit hash.
    """
    from agent.archival import step_git_push as archival_git_push

    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    # Pass safe_slug so archival finds the right file
    modified_task = {**task, "slug": safe_slug}
    return archival_git_push(
        modified_task, config,
        vault_subfolder="instagram",
        commit_prefix="add instagram:",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_transcript_for_llm(merged: dict) -> str:
    """Format merged transcript as compact text for LLM consumption."""
    return format_transcript_for_llm(merged, style="seconds")


# ── Comment Steps (comment_id deep-links) ────────────────────────────────────

IG_WEB_APP_ID = "936619743392459"
IG_APP_UA = (
    "Instagram 275.0.0.27.98 Android (30/11; 420dpi; 1080x2340; samsung; "
    "SM-G991B; o1s; exynos2100; en_US; 458229258)"
)
_IG_SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_COMMENT_ID_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_IG_COMMENT_PAGE_CAP = 25


def _safe_comment_id_path_token(comment_id: str) -> str:
    """Reject comment_id values unsafe for use in a cookie_dir filename (path traversal)."""
    if (
        not comment_id
        or ".." in comment_id
        or "/" in comment_id
        or "\\" in comment_id
    ):
        raise StepError(
            "Invalid comment_id in URL",
            error_type=ErrorType.FAIL,
            step_name="ig_comment_fetch",
        )
    if not _COMMENT_ID_PATH_TOKEN.fullmatch(comment_id):
        raise StepError(
            "Invalid comment_id in URL",
            error_type=ErrorType.FAIL,
            step_name="ig_comment_fetch",
        )
    return comment_id


def _shortcode_to_media_id(shortcode: str) -> int:
    """Decode an Instagram shortcode to its numeric media id."""
    media_id = 0
    for ch in shortcode:
        media_id = media_id * 64 + _IG_SHORTCODE_ALPHABET.index(ch)
    return media_id


def _read_ig_cookies(cookie_dir: Path) -> dict[str, str]:
    """Parse the Netscape-format ig_cookies.txt into {name: value}."""
    cookies: dict[str, str] = {}
    for line in (cookie_dir / "ig_cookies.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            cookies[parts[5]] = parts[6]
    return cookies


def _fetch_ig_comment(url: str, config: Config) -> dict:
    """Fetch a specific Instagram comment + its parent reel caption.

    Hits the web API (www.instagram.com/api/v1/media/{id}/comments/) with the
    sessionid cookie + x-ig-app-id header. The parent caption rides in the same
    response, so one call yields both the comment and its parent context.
    """
    parsed = urlparse(url)
    shortcode = parsed.path.rstrip("/").split("/")[-1]
    qs = parse_qs(parsed.query)
    comment_id = (qs.get("comment_id") or [""])[0]
    if not comment_id:
        raise StepError(
            "No comment_id in URL", error_type=ErrorType.FAIL, step_name="ig_comment_fetch"
        )
    path_token = _safe_comment_id_path_token(comment_id)

    cookies = _read_ig_cookies(config.cookie_dir)
    if "sessionid" not in cookies:
        raise StepError(
            "Instagram sessionid missing — re-export cookies",
            error_type=ErrorType.PAUSE, step_name="ig_comment_fetch",
        )
    media_id = _shortcode_to_media_id(shortcode)
    conf_path = config.cookie_dir / f".curl-ig-comment-{path_token}.conf"
    resolved_conf = conf_path.resolve()
    cookie_dir_resolved = config.cookie_dir.resolve()
    if not resolved_conf.is_relative_to(cookie_dir_resolved):
        raise StepError(
            "Invalid comment_id in URL",
            error_type=ErrorType.FAIL,
            step_name="ig_comment_fetch",
        )
    csrftoken = cookies.get("csrftoken", "")

    def _comments_api_url(min_id: str | None) -> str:
        base = (
            f"https://www.instagram.com/api/v1/media/{media_id}/comments/"
            "?can_support_threading=true"
        )
        if min_id:
            return f"{base}&min_id={min_id}"
        return base

    comment = None
    data: dict = {}
    min_id: str | None = None
    try:
        for _ in range(_IG_COMMENT_PAGE_CAP):
            api_url = _comments_api_url(min_id)
            _assert_fetch_url_allowed(api_url)
            curl_cfg = "\n".join(
                [
                    "silent",
                    "max-time = 25",
                    curl_config_quoted_assignment("user-agent", IG_APP_UA),
                    curl_config_header_line(f"x-ig-app-id: {IG_WEB_APP_ID}"),
                    curl_config_header_line(f"x-csrftoken: {csrftoken}"),
                    curl_config_header_line(
                        f'Cookie: sessionid={cookies["sessionid"]}'
                    ),
                    curl_config_quoted_assignment("url", api_url),
                ]
            ) + "\n"
            conf_path.write_text(curl_cfg, encoding="utf-8")
            os.chmod(conf_path, 0o600)

            result = subprocess.run(
                ["curl", "-s", "--max-redirs", "0", "--config", str(conf_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                snippet = sanitize_visible_message(result.stdout[:200])
                raise StepError(
                    f"Comment API returned non-JSON: {snippet}",
                    error_type=ErrorType.RETRY, step_name="ig_comment_fetch",
                )
            if data.get("status") == "fail":
                raise StepError(
                    f"Comment API failed: {sanitize_visible_message(data.get('message', ''))}",
                    error_type=ErrorType.RETRY, step_name="ig_comment_fetch",
                )

            comment = next(
                (c for c in data.get("comments", []) if str(c.get("pk")) == comment_id),
                None,
            )
            if comment is not None:
                break
            min_id = data.get("next_min_id")
            if not min_id:
                break

        if comment is None:
            raise StepError(
                f"comment_id {comment_id} not found in comment pages",
                error_type=ErrorType.PAUSE, step_name="ig_comment_fetch",
            )

        caption = data.get("caption") or {}
        return {
            "comment_id": comment_id,
            "comment_text": comment.get("text", ""),
            "comment_author": (comment.get("user") or {}).get("username", "unknown"),
            "comment_created_at": comment.get("created_at"),
            "parent_shortcode": shortcode,
            "parent_url": f"https://www.instagram.com/reel/{shortcode}/",
            "parent_caption": caption.get("text", ""),
            "parent_author": (caption.get("user") or {}).get("username", "unknown"),
        }
    finally:
        try:
            if conf_path.is_file():
                conf_path.unlink()
        except OSError:
            pass


_IG_COMMENT_TEXT_FIELDS = ("comment_text", "parent_caption")


def _ig_comment_sidecar_for_disk(info: dict) -> dict:
    """Return ig_comment payload safe for at-rest JSON (display URLs, redacted free text)."""
    out = dict(info)
    parent_url = out.get("parent_url")
    if isinstance(parent_url, str):
        out["parent_url"] = url_for_display(parent_url)
    for key in _IG_COMMENT_TEXT_FIELDS:
        val = out.get(key)
        if isinstance(val, str):
            out[key] = redact_http_urls_in_text(val)
    return out


def step_ig_comment_fetch(task: dict, config: Config) -> str:
    """Fetch the comment + parent reel context and write it to a raw JSON file."""
    url = task["source_url"]
    info = _fetch_ig_comment(url, config)
    raw_path = _ig_comment_sidecar_path(config, task)
    raw_path.write_text(
        json.dumps(_ig_comment_sidecar_for_disk(info), indent=2, ensure_ascii=False),
    )
    return str(raw_path)


def step_ig_comment_synthesize(task: dict, config: Config) -> str:
    """Synthesize a note from the comment + its parent reel context."""
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    source_url = task.get("source_url", "")
    display_source_url = url_for_display(source_url)

    raw_path = _ig_comment_sidecar_path(config, task)
    if not raw_path.exists():
        raise FileNotFoundError(f"Comment raw not found: {raw_path}")
    info = json.loads(raw_path.read_text())

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    date_str = datetime.now().strftime("%Y-%m-%d")
    comment_author = info["comment_author"]
    parent_author = info["parent_author"]
    parent_caption = info["parent_caption"]
    comment_text = info["comment_text"]

    system_prompt = f"""You are a knowledge base curator. Capture an Instagram comment in the context of its parent reel.

SOURCE: a comment by @{comment_author} on a reel by @{parent_author}.

SECTIONS:
1. YAML frontmatter:
   date: {date_str}
   source: instagram comment (@{comment_author})
   source_url: {display_source_url}
   source_files:
     raw: <filled by archival step>
   confidence: plausible
   tags: [instagram, comment]
2. # Title — the comment's claim or topic
3. ## The Comment — the exact comment text as a blockquote (>), with author and date
4. ## Parent Context — the reel's caption, uploader, and link
5. ## Analysis — what the comment adds (claim / advice / reaction), and reliability

Keep it proportionate — one comment, not a thesis."""

    user_prompt = f"""COMMENT (by @{comment_author}):
{comment_text}

PARENT REEL (by @{parent_author}):
Caption: {parent_caption}
Link: {info['parent_url']}

=== END ===

Write the note following the template."""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = strip_code_fence(response.content)

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    write_vault_note(note_path, note_content)
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


# ── Pipeline Registration ────────────────────────────────────────────────────

# Instagram reel pipeline: validate → download → audio → diarize → transcribe → merge → synthesize → quality → cleanup → minio → git
register_pipeline("instagram_reel", [
    Step(name="ig_reel_validate", fn=step_ig_reel_validate, on_error="pause"),
    Step(name="ig_reel_download", fn=step_ig_reel_download, on_error="retry"),
    Step(name="extract_audio", fn=step_extract_audio, on_error="retry"),
    Step(name="diarize", fn=step_diarize, on_error="retry"),
    Step(name="transcribe", fn=step_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=step_ig_reel_merge, on_error="retry"),
    Step(name="ig_reel_synthesize", fn=step_ig_reel_synthesize, on_error="retry"),
    Step(name="ig_quality_gate", fn=step_ig_quality_gate, on_error="retry"),
    Step(name="ig_minio_upload", fn=step_ig_minio_upload, on_error="retry"),
    Step(name="ig_cleanup", fn=step_ig_cleanup, on_error="fail"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="ig_git_push", fn=step_ig_git_push, on_error="retry"),
])

register_pipeline("instagram_reel_synthesize", [
    Step(name="fetch_transcript", fn=step_fetch_transcript, on_error="retry"),
    Step(name="ig_reel_synthesize", fn=step_ig_reel_synthesize, on_error="retry"),
    Step(name="ig_quality_gate", fn=step_ig_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="ig_git_push", fn=step_ig_git_push, on_error="retry"),
])

# Instagram carousel pipeline: validate → metadata → extract → synthesize → quality → minio → git
register_pipeline("instagram_post", [
    Step(name="ig_reel_validate", fn=step_ig_reel_validate, on_error="pause"),
    Step(name="ig_carousel_metadata", fn=step_ig_carousel_metadata, on_error="retry"),
    Step(name="ig_carousel_extract", fn=step_ig_carousel_extract, on_error="retry"),
    Step(name="ig_carousel_synthesize", fn=step_ig_carousel_synthesize, on_error="retry"),
    Step(name="ig_quality_gate", fn=step_ig_quality_gate, on_error="retry"),
    Step(name="ig_minio_upload", fn=step_ig_minio_upload, on_error="retry"),
    Step(name="ig_cleanup", fn=step_ig_cleanup, on_error="fail"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="ig_git_push", fn=step_ig_git_push, on_error="retry"),
])

# Instagram comment pipeline: fetch comment + parent caption → synthesize → quality → minio → git
register_pipeline("instagram_comment", [
    Step(name="ig_comment_fetch", fn=step_ig_comment_fetch, on_error="retry"),
    Step(name="ig_comment_synthesize", fn=step_ig_comment_synthesize, on_error="retry"),
    Step(name="ig_quality_gate", fn=step_ig_quality_gate, on_error="retry"),
    Step(name="ig_minio_upload", fn=step_ig_minio_upload, on_error="retry"),
    Step(name="ig_cleanup", fn=step_ig_cleanup, on_error="fail"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="ig_git_push", fn=step_ig_git_push, on_error="retry"),
])
