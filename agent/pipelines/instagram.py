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
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from agent.archival import append_note_tags
from agent.config import Config
from agent.db import init_db, LOGS_SCHEMA
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMClient, get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import (
    _slug_path,
    format_transcript_for_llm,
    merge_speaker_labels,
    run_video_quality_gate,
    step_diarize,
    step_extract_audio,
    step_fetch_transcript,
    step_transcribe,
    upload_note_archive,
    upload_video_source,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# yt-dlp rate limiting for Instagram (per spec)
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

    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

    output_path = Path(config.temp_dir) / f"{safe_slug}_reel.mp4"

    result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", str(config.cookie_dir / "ig_cookies.txt"),
            "--js-runtimes", "node",
            "--output", str(output_path.with_suffix("")),
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=300,
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
    actual_files = list(Path(config.temp_dir).glob(f"{safe_slug}_reel.*"))
    if not actual_files:
        raise StepError(
            f"No output file found for slug {safe_slug}",
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
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    diar_path = Path(config.temp_dir) / f"{safe_slug}_diarization.json"
    trans_path = Path(config.temp_dir) / f"{safe_slug}_transcript.json"

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
    merged_path = Path(config.temp_dir) / f"{safe_slug}_merged.json"
    merged_path.write_text(json.dumps(trans, indent=2, ensure_ascii=False))

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
    source_url = task.get("source_url", "")

    merged_path = Path(config.temp_dir) / f"{safe_slug}_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged transcript not found: {merged_path}")

    merged = json.loads(merged_path.read_text())
    music_only = merged.get("music_only", False)

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    # Build compact transcript
    transcript_text = _format_transcript_for_llm(merged)
    source_domain = urlparse(source_url).netloc or "instagram.com"
    date_str = datetime.now().strftime("%Y-%m-%d")
    if music_only:
        # Music-only reel: use caption as primary content
        system_prompt = f"""You are a knowledge base curator processing an Instagram reel.

WARNING: This reel is music-only — the audio contains no meaningful speech.
The transcript is provided for completeness but the primary content
comes from the reel's caption/description.

SOURCE: Instagram reel ({source_url})
DATE: {date_str}

TASK: Write a note that:
1. Notes this is a music/BGM reel with minimal speech
2. If a caption was detected, analyze it as the primary content
3. Describe what can be observed from the transcript (if any words were detected)

OUTPUT FORMAT:
- YAML frontmatter: date, source: instagram.com, source_url, minio placeholder,
  confidence: low (music-only reel), tags: [instagram, reel, music]
- Note sections: Title, Summary, Caption Analysis (if any), Transcript Notes,
  Reliability (note music-only limitation)"""
    else:
        system_prompt = f"""You are a knowledge base curator specialized in Instagram reel content.

SOURCE: An Instagram reel video transcript.

INSTAGRAM CONTEXT:
- This is short-form social media content — typically 15-90 seconds
- Format: vertical video, often trending audio + visual content

TASK: Write a concise but thorough analysis note.

SECTIONS:
1. YAML frontmatter
2. Title — descriptive
3. Summary — 2-3 sentences
4. Key Points — ≥2, with direct quotes from transcript
5. Speaker Analysis — who is speaking, perspective
6. Context — platform context, trends if identifiable
7. Critical Analysis — what's being communicated, what's omitted
8. Reliability — confidence assessment

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: instagram.com
  source_url: {source_url}
  source_files:
    video: {config.minio_tailscale_base}/{config.minio_bucket}/videos/{safe_slug}.mp4
    transcript: {config.minio_tailscale_base}/{config.minio_bucket}/transcripts/{safe_slug}.json
  confidence: plausible
  tags: [instagram, reel]
- Then sections as above."""

    user_prompt = f"""TRANSCRIPT:
{transcript_text}

=== END TRANSCRIPT ===

Write a note following the template above.
{'This is a music-only reel — prioritize any detected caption text over the transcript.' if music_only else 'Analyze the full transcript for key points.'}"""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = response.content

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content, encoding="utf-8")
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
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

    # Extract metadata via yt-dlp print
    result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", str(config.cookie_dir / "ig_cookies.txt"),
            "--js-runtimes", "node",
            "--print", "%(title)s||%(uploader)s||%(like_count)s||%(playlist_count)s",
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=120,
    )

    metadata = {
        "url": url,
        "title": "",
        "uploader": "",
        "like_count": 0,
        "playlist_count": 0,
    }

    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("||")
        if len(parts) >= 1:
            metadata["title"] = parts[0]
        if len(parts) >= 2:
            metadata["uploader"] = parts[1]
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
            "yt-dlp",
            "--cookies", str(config.cookie_dir / "ig_cookies.txt"),
            "--js-runtimes", "node",
            "--print", "%(description)s",
            *IGDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=120,
    )

    if caption_result.returncode == 0:
        metadata["caption"] = caption_result.stdout.strip()

    # Write metadata
    meta_path = Path(config.temp_dir) / f"{safe_slug}_carousel_meta.json"
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
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

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
        
        try:
            await page.goto("{url}", wait_until="networkidle", timeout=30000)
        except Exception:
            pass
        
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

    script_path = Path(config.temp_dir) / f"{safe_slug}_playwright.py"
    script_path.write_text(playwright_script)

    result = subprocess.run(
        ["python3", str(script_path)],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        raise StepError(
            f"Playwright carousel extraction failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="carousel_extract",
        )

    try:
        slides = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        raise StepError(
            f"Failed to parse Playwright output for carousel: {result.stdout[:200]}",
            error_type=ErrorType.RETRY,
            step_name="carousel_extract",
        )

    # Write extracted slides
    slides_path = Path(config.temp_dir) / f"{safe_slug}_carousel_slides.json"
    slides_data = {
        "url": url,
        "slides": slides,
        "slide_count": len(slides),
        "cap_note": f"Hard cap: {CAROUSEL_SLIDE_CAP} slides. Slides beyond {CAROUSEL_SLIDE_CAP} are not extractable.",
    }
    slides_path.write_text(json.dumps(slides_data, indent=2, ensure_ascii=False))

    # Clean up the temp script
    try:
        script_path.unlink()
    except OSError:
        pass

    return str(slides_path)


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

    # Load carousel data
    slides_path = Path(config.temp_dir) / f"{safe_slug}_carousel_slides.json"
    if not slides_path.exists():
        raise FileNotFoundError(f"Carousel slides not found: {slides_path}")

    slides_data = json.loads(slides_path.read_text())

    # Load metadata
    meta_path = Path(config.temp_dir) / f"{safe_slug}_carousel_meta.json"
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

    source_domain = urlparse(source_url).netloc or "instagram.com"
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
  source_url: {source_url}
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
    note_content = response.content

    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content, encoding="utf-8")
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

    if content_type == "instagram_reel":
        passed, rounds = run_video_quality_gate(note_path, config, task["id"])
        return json.dumps({"passed": passed, "rounds": rounds})
    else:
        # Carousel: simple size + section check
        content = note_path.read_text(encoding="utf-8")
        size = len(content)

        # Must have Slide-by-Slide Analysis section
        has_slide_analysis = "## Slide-by-Slide Analysis" in content or "## Slide-by-Slide" in content

        passed = size >= 2000 and has_slide_analysis
        failures = []
        if size < 2000:
            failures.append(f"Note too short: {size} chars < 2000")
        if not has_slide_analysis:
            failures.append("Missing Slide-by-Slide Analysis section")

        return json.dumps({
            "passed": passed,
            "failures": failures,
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
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

    to_remove = []
    for ext in ["_reel.mp4", "_reel.webm", "_16k.wav", "_8k.wav",
                "_playwright.py", "_carousel_slides.json", "_carousel_meta.json"]:
        path = Path(config.temp_dir) / f"{safe_slug}{ext}"
        if path.exists():
            to_remove.append(str(path.name))

    removed = []
    for filename in to_remove:
        path = Path(config.temp_dir) / filename
        try:
            path.unlink()
            removed.append(filename)
        except OSError:
            pass

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
        # Source-archival: video + transcript (matches the batch collector).
        reel_files = sorted(Path(config.temp_dir).glob(f"{safe_slug}_reel.*"))
        video_path = reel_files[0] if reel_files else Path(config.temp_dir) / f"{safe_slug}_reel.mp4"
        dest_ext = video_path.suffix.lower().lstrip(".") or "mp4"
        uploaded = upload_video_source(safe_slug, config, video_path, dest_ext, note_path)
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
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="ig_git_push", fn=step_ig_git_push, on_error="retry"),
])
