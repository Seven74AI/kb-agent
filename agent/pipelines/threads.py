"""Threads pipeline — text and video extraction from Threads posts.

Detection (Phase 0): curl + cookies + Googlebot UA → grep video_versions.
If present → VIDEO → full video pipeline (download → dual audio → diarize →
transcribe → merge → synthesize → quality → MinIO → Git).
If absent → TEXT → text extraction pipeline (curl → regex captions →
instant LLM synthesis → quality → MinIO → Git).

Pitfalls handled:
- media_type=19 (Instagram video cross-post): check video_versions regardless
- Reply posts: check is_reply field, fetch parent for context
- Caption truncation: regex matches ALL caption texts, not just first
- Login-wall: quality gate detects og:description = login-wall text → SKIP
"""

import json
import re
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
    strip_code_fence,
    upload_note_archive,
    upload_video_source,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Googlebot user-agent for Threads scraping
GOOGLEBOT_UA = (
    "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.204 Mobile Safari/537.36 "
    "(compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
)

# Rate limiting for Threads
THREADS_SLEEP_MIN = 5
THREADS_SLEEP_MAX = 10
THREADS_MAX_POSTS_PER_SESSION = 3

# Density target for video notes
THREADS_VIDEO_DENSITY_MIN = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def _curl_threads(url: str, timeout: int = 30, cookie_dir: Path | None = None) -> str:
    """Fetch a Threads URL with cookies + Googlebot UA.

    Args:
        url: The Threads post URL.
        timeout: Curl max-time in seconds.
        cookie_dir: Directory containing threads_cookies.txt. Defaults to
            /root/kb-agent/cookies (matching config.cookie_dir default).

    Returns:
        Raw HTTP response body.

    Raises:
        StepError: If curl fails.
    """
    if cookie_dir is None:
        cookie_dir = Path("/root/kb-agent/cookies")
    cookie_path = cookie_dir / "threads_cookies.txt"
    if not cookie_path.exists():
        raise StepError(
            f"Threads cookies not found: {cookie_path}",
            error_type=ErrorType.PAUSE,
            step_name="curl_threads",
            details=f"Export cookies from Chrome to {cookie_path}",
        )

    result = subprocess.run(
        [
            "curl", "-s", "--max-time", str(timeout),
            "-b", str(cookie_path),
            "-H", f"User-Agent: {GOOGLEBOT_UA}",
            "-H", "Accept: text/html,application/xhtml+xml",
            "-H", "Accept-Language: en-US,en;q=0.9",
            "-L",
            url,
        ],
        capture_output=True, text=True, timeout=timeout + 10,
    )

    if result.returncode != 0:
        raise StepError(
            f"Threads curl failed (exit {result.returncode}): {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="curl_threads",
        )

    return result.stdout


def _has_video_versions(html: str) -> bool:
    """True if the payload has a NON-NULL video_versions array.

    A bare ``video_versions`` substring matches ``"video_versions": null`` (a
    post with no video) — a false positive. The real signal is the key followed
    by an array bracket.
    """
    return re.search(r'"video_versions"\s*:\s*\[', html) is not None


def _extract_json_array_after_key(html: str, key: str) -> str | None:
    """Return the first NON-NULL balanced JSON array that follows ``"key":``.

    Threads payloads repeat the key many times — earlier occurrences are often
    ``"video_versions": null`` (related/suggested posts), with the post's own
    video appearing later. Skip ``null`` and return the first real array.
    """
    marker = f'"{key}"'
    search_from = 0
    while True:
        pos = html.find(marker, search_from)
        if pos < 0:
            return None
        i = pos + len(marker)
        while i < len(html) and html[i] in " \t\n":
            i += 1
        if i < len(html) and html[i] == ":":
            i += 1
            while i < len(html) and html[i] in " \t\n":
                i += 1
        if i < len(html) and html[i] == "[":
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < len(html):
                ch = html[j]
                if escape:
                    escape = False
                elif ch == "\\" and in_string:
                    escape = True
                elif ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            return html[i : j + 1]
                j += 1
            return None
        # null or other value — skip to the next occurrence
        search_from = i


def _extract_video_url(html: str) -> str | None:
    """Extract the highest quality video URL from video_versions JSON.

    Searches for video_versions array in the page JSON and picks the
    entry with the largest 'height' value.

    Args:
        html: Raw Threads page HTML/JSON.

    Returns:
        Video URL string or None if not found.
    """
    array_text = _extract_json_array_after_key(html, "video_versions")
    if not array_text:
        return None

    try:
        versions = json.loads(array_text)
        if not versions or not isinstance(versions, list):
            return None

        # Pick highest quality (largest height)
        best = max(
            versions,
            key=lambda v: v.get("height", 0) if isinstance(v, dict) else 0,
        )
        return best.get("url") if isinstance(best, dict) else None
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def _extract_all_captions(html: str) -> list[str]:
    """Extract ALL caption texts from the Threads page JSON.

    Uses regex to find all occurrences of 'text' fields in caption objects.
    Handles caption truncation by matching ALL instances, not just the first.

    Args:
        html: Raw Threads page HTML/JSON.

    Returns:
        List of caption text strings (may be empty).
    """
    captions = []
    # Caption objects on Threads payloads: "caption": { ... "text": "..." ... }
    caption_block = r'"caption"\s*:\s*\{[^{}]*?"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
    for match in re.finditer(caption_block, html):
        try:
            text = json.loads(f'"{match.group(1)}"')
        except (json.JSONDecodeError, ValueError):
            continue
        if text and text.strip() and text.strip() not in captions:
            captions.append(text.strip())

    return captions


def _extract_author(html: str) -> str:
    """Extract the POST author's username (not the logged-in session user).

    The first ``"username"`` in the payload is the session user, so that field
    is wrong. The post's own author rides in its ``permalink`` field
    (``\\u0040<username>`` — ``@`` JSON-escaped). Parse that instead.

    Args:
        html: Raw Threads page HTML/JSON.

    Returns:
        Username string or 'unknown'.
    """
    match = re.search(r'"permalink"\s*:\s*"[^"]*?(?:\\u0040|@)([A-Za-z0-9._]+)', html)
    return match.group(1) if match else "unknown"


def _is_reply_post(html: str) -> bool:
    """Check if this is a reply post (has parent context).

    Args:
        html: Raw Threads page HTML/JSON.

    Returns:
        True if is_reply field is true.
    """
    match = re.search(r'"is_reply"\s*:\s*(true|false)', html)
    return match.group(1) == "true" if match else False


def _fetch_parent_post(parent_url: str, cookie_dir: Path | None = None) -> str | None:
    """Fetch parent post content for reply context.

    Args:
        parent_url: The parent post URL.
        cookie_dir: Directory containing threads_cookies.txt.

    Returns:
        Caption text of the parent post, or None if fetch fails.
    """
    try:
        html = _curl_threads(parent_url, timeout=20, cookie_dir=cookie_dir)
        captions = _extract_all_captions(html)
        return captions[0] if captions else None
    except Exception:
        return None


def _extract_og_description(html: str) -> str | None:
    """Extract og:description meta tag content.

    Args:
        html: Raw HTML.

    Returns:
        og:description content or None.
    """
    match = re.search(
        r'<meta\s+property="og:description"\s+content="([^"]*)"',
        html, re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _is_login_wall(html: str) -> bool:
    """Detect if the page returned a login wall instead of real content.

    Typical login-wall responses: redirect to /login, or og:description
    that says 'Follow @user on Threads' or similar generic text.

    Args:
        html: Raw Threads page HTML.

    Returns:
        True if the page appears to be a login wall.
    """
    # Check for explicit login redirect
    if '/login' in html[:500] and 'Threads' in html:
        return True

    # Check og:description for generic login-wall patterns
    og = _extract_og_description(html) or ""
    if og:
        # If og:description contains login-wall patterns
        if "Follow @" in og and "Threads" in og:
            return True
        if ("Log in" in og or "Sign up" in og) and "Threads" in og:
            return True
    elif "class=\"_ab6c\"" not in html and "caption" not in html.lower():
        # No og:description AND no real content markers AND no caption data
        return True

    return False


# ── Step 1: Detection ─────────────────────────────────────────────────────────

def step_threads_detect(task: dict, config: Config) -> str:
    """Detect whether a Threads post is text or video.

    Fetches the page, checks for video_versions in the JSON. Also extracts
    author, captions, and reply/parent context into the raw metadata file so
    the VIDEO pipeline has it — it has no extract_text step, and without this a
    video reply loses both its author and its parent.

    Args:
        task: Task dict with source_url.
        config: Config dataclass.

    Returns:
        JSON string: {"content_type": "threads_text" | "threads_video", "raw_page": "..."}
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])

    html = _curl_threads(url, cookie_dir=config.cookie_dir)

    # Check for login wall before doing deeper analysis
    if _is_login_wall(html):
        raise StepError(
            "Threads returned login wall — cookies may be invalid",
            error_type=ErrorType.PAUSE,
            step_name="threads_detect",
            details="Re-export cookies from Chrome to threads_cookies.txt",
        )

    has_video = _has_video_versions(html)
    content_type = "threads_video" if has_video else "threads_text"

    # Reply/parent context: a comment is the reply + its parent. Extract both so
    # the video pipeline (and synthesis) keeps them instead of dropping the
    # parent on video replies.
    author = _extract_author(html)
    is_reply = _is_reply_post(html)
    parent_url = None
    parent_context = None
    if is_reply:
        parent_match = re.search(
            r'"parent_post"\s*:\s*\{[^}]*"code"\s*:\s*"([^"]+)"',
            html,
        )
        if parent_match:
            parent_url = f"https://www.threads.net/t/{parent_match.group(1)}"
            parent_context = _fetch_parent_post(parent_url, cookie_dir=config.cookie_dir)

    raw_path = Path(config.temp_dir) / f"{_slug_path(slug)}_threads_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "url": url,
                "author": author,
                "captions": _extract_all_captions(html),
                "is_reply": is_reply,
                "parent_url": parent_url,
                "parent_context": parent_context,
                "extracted_at": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return json.dumps({
        "content_type": content_type,
        "has_video": has_video,
    })


# ── Step 2a: Text Extraction ──────────────────────────────────────────────────

def step_threads_extract_text(task: dict, config: Config) -> str:
    """Extract text content from a Threads post.

    Fetches the page, extracts all captions via regex, validates quality,
    extracts author, checks reply context.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        JSON string with captions, author, is_reply, parent_context, og_description,
        confidence flag.

    Raises:
        StepError(SKIP): If captions empty AND og:description is login-wall text.
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])

    html = _curl_threads(url, cookie_dir=config.cookie_dir)

    # Check for login wall
    if _is_login_wall(html):
        raise StepError(
            "Threads returned login wall — cannot extract content",
            error_type=ErrorType.PAUSE,
            step_name="extract_text",
            details="Re-export cookies from Chrome to threads_cookies.txt",
        )

    # Extract all captions (handles truncation by matching ALL)
    captions = _extract_all_captions(html)

    # Extract og:description as fallback
    og_description = _extract_og_description(html) or ""

    # Quality gate: empty captions AND login-wall og → SKIP
    if not captions and _is_login_wall(html):
        raise StepError(
            "Threads post has no accessible content (login wall)",
            error_type=ErrorType.FAIL,
            step_name="extract_text",
            details="Post may require authentication beyond cookies",
        )

    # If only og:description is available, flag confidence=untested
    confidence = "plausible"
    if not captions and og_description:
        captions = [og_description]
        confidence = "untested"

    # Extract author
    author = _extract_author(html)

    # Check if reply post — fetch parent for context
    is_reply = _is_reply_post(html)
    parent_context = None
    parent_url = None

    if is_reply:
        # Try to find parent URL from the page
        parent_match = re.search(
            r'"parent_post"\s*:\s*\{[^}]*"code"\s*:\s*"([^"]+)"',
            html,
        )
        if parent_match:
            parent_url = f"https://www.threads.net/t/{parent_match.group(1)}"
            parent_context = _fetch_parent_post(parent_url, cookie_dir=config.cookie_dir)

    # Write raw extraction for archival
    raw_path = Path(config.temp_dir) / f"{_slug_path(slug)}_threads_raw.json"
    extracted = {
        "url": url,
        "author": author,
        "captions": captions,
        "og_description": og_description,
        "is_reply": is_reply,
        "parent_url": parent_url,
        "parent_context": parent_context,
        "confidence": confidence,
        "extracted_at": datetime.now().isoformat(),
    }
    raw_path.write_text(json.dumps(extracted, indent=2, ensure_ascii=False))

    return str(raw_path)


# ── Step 2b: Video Extraction ─────────────────────────────────────────────────

def step_threads_extract_video(task: dict, config: Config) -> str:
    """Extract video URL from a Threads video post and download it.

    Fetches the page, extracts video URL from video_versions JSON,
    downloads with curl + Referer header.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the downloaded video file.
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

    html = _curl_threads(url, cookie_dir=config.cookie_dir)

    if _is_login_wall(html):
        raise StepError(
            "Threads returned login wall — cannot extract video",
            error_type=ErrorType.PAUSE,
            step_name="extract_video",
            details="Re-export cookies from Chrome to threads_cookies.txt",
        )

    # Extract video URL from video_versions JSON
    video_url = _extract_video_url(html)
    if not video_url:
        raise StepError(
            "No video URL found in Threads post — may not contain video",
            error_type=ErrorType.FAIL,
            step_name="extract_video",
        )

    # Download video with Referer header
    output_path = Path(config.temp_dir) / f"{safe_slug}_threads_video.mp4"

    result = subprocess.run(
        [
            "curl", "-s", "-L", "--max-time", "300",
            "-H", f"Referer: {url}",
            "-H", f"User-Agent: {GOOGLEBOT_UA}",
            "-o", str(output_path),
            video_url,
        ],
        capture_output=True, text=True, timeout=360,
    )

    if result.returncode != 0 or not output_path.exists():
        raise StepError(
            f"Video download failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="extract_video",
        )

    # Verify minimum size (> 1KB)
    if output_path.stat().st_size < 1024:
        raise StepError(
            f"Downloaded video too small ({output_path.stat().st_size} bytes)",
            error_type=ErrorType.FAIL,
            step_name="extract_video",
        )

    return str(output_path)


# ── Step 6: Merge Diarization + Transcript ────────────────────────────────────

def step_threads_merge(task: dict, config: Config) -> str:
    """Merge diarization speaker labels with transcript segments.

    Merges at segment midpoint. Adds speaker labels and overlap flags.
    Generates chapters.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string summarizing merge result.
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

    merge_speaker_labels(diar, trans)

    # Count distinct speakers
    speakers = set(
        s.get("speaker", "Unknown") for s in trans.get("segments", [])
    )

    # Write merged
    merged_path = Path(config.temp_dir) / f"{safe_slug}_merged.json"
    merged_path.write_text(json.dumps(trans, indent=2, ensure_ascii=False))

    return json.dumps({
        "segments": len(trans.get("segments", [])),
        "speakers": len(speakers),
        "merged_path": str(merged_path),
    })


# ── Step 7: LLM Synthesis ─────────────────────────────────────────────────────

def step_threads_video_synthesize(task: dict, config: Config) -> str:
    """LLM synthesis for Threads video posts.

    Uses the YouTube two-pass synthesis pattern adapted for Threads
    (typically shorter content, comment thread context).

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

    # Load raw extraction for author context
    raw_path = Path(config.temp_dir) / f"{safe_slug}_threads_raw.json"
    author = "unknown"
    parent_context = None
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        author = raw.get("author", "unknown")
        parent_context = raw.get("parent_context")

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    # Build compact transcript
    transcript_text = _format_transcript_for_llm(merged)

    # Build prompt
    source_domain = urlparse(source_url).netloc or "threads.net"
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Pre-compute reply context line
    if parent_context:
        reply_line = f"- REPLY CONTEXT: {parent_context[:500]}"
    else:
        reply_line = "- This is an original post (not a reply)"

    system_prompt = f"""You are a knowledge base curator specialized in Threads video content.

SOURCE: A Threads video post transcript.

THREADS CONTEXT:
- Author: @{author}
- This is short-form social media content — typically 1-5 minutes
{reply_line}

STRUCTURE (follow this template):
# [Descriptive Title]

## Summary
2-3 sentences capturing the core message.

## Key Points
≥3 key points, each with a direct quote from the transcript (> prefix).

## Speaker Analysis
Brief analysis of who is speaking and their perspective.

## Full Transcript
The complete transcript with speaker labels.

## Context
Who posted this, why it matters, platform context.

## Critical Analysis
What's new vs restating? What's omitted?

## Reliability
Confidence level with justification.
- Is this an original take or a reaction?
- Does the transcript provide enough context?

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: threads.net (@{author})
  source_url: {source_url}
  source_files:
    video: {config.minio_tailscale_base}/{config.minio_bucket}/videos/{safe_slug}.mp4
    transcript: {config.minio_tailscale_base}/{config.minio_bucket}/transcripts/{safe_slug}.json
  confidence: plausible
  tags: [threads, video]
- Then the note sections as shown above.
- Leave See Also section empty."""

    user_prompt = f"""TRANSCRIPT:
{transcript_text}

=== END TRANSCRIPT ===

Write a comprehensive note following the template above.
Extract 3-5 key points with direct quotes.
Analyze the speaker's perspective and credibility.
Note whether this appears to be an original post or a reaction."""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = strip_code_fence(response.content)

    note_path = config.vault_path / "Knowledge base" / "threads" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content, encoding="utf-8")
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


def step_threads_text_synthesize(task: dict, config: Config) -> str:
    """LLM synthesis for Threads text posts.

    Instant synthesis (single pass) for text-only Threads posts.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    source_url = task.get("source_url", "")

    raw_path = Path(config.temp_dir) / f"{safe_slug}_threads_raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw extraction not found: {raw_path}")

    raw = json.loads(raw_path.read_text())
    captions = raw.get("captions", [])
    author = raw.get("author", "unknown")
    parent_context = raw.get("parent_context")
    confidence = raw.get("confidence", "plausible")

    if not captions:
        raise StepError(
            "No caption text found for synthesis",
            error_type=ErrorType.FAIL,
            step_name="text_synthesize",
        )

    content = "\n\n".join(captions)

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    source_domain = urlparse(source_url).netloc or "threads.net"
    date_str = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""You are a knowledge base curator specialized in social media content.

SOURCE: A Threads post by @{author}.

CONTEXT:
{f"Reply context (parent post): {parent_context[:500]}" if parent_context else "Original post (not a reply)"}

TASK: Write a concise but thorough note analyzing this Threads post.

SECTIONS:
1. YAML frontmatter with date, source, source_url, minio placeholder, confidence, tags
2. # Title — descriptive title for the post
3. ## Summary — 2-3 sentences capturing the post's message
4. ## Key Claims — ≥2 specific claims, each with the original text quoted (> )
5. ## Context — who posted this, why it matters, reply context if applicable
6. ## Critical Analysis — what's new, what's omitted, strengths/weaknesses
7. ## Reliability — confidence ({confidence}), justification

If this is a reply, include the parent context as background.
Keep the note proportionate — a single Threads post doesn't need 5000 words.

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: threads.net (@{author})
  source_url: {source_url}
  source_files:
    raw: <filled by archival step>
  confidence: {confidence}
  tags: [threads, social-media]
- Then the note sections."""

    user_prompt = f"""THREADS POST:
{content}

=== END POST ===

Write a note following the template above.
The confidence level is '{confidence}' — be appropriately cautious if it's 'untested'."""

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    note_content = strip_code_fence(response.content)

    note_path = config.vault_path / "Knowledge base" / "threads" / f"{safe_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content, encoding="utf-8")
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


# ── Step 8: Cleanup ───────────────────────────────────────────────────────────

def step_threads_cleanup(task: dict, config: Config) -> str:
    """Clean up temporary files after successful pipeline execution.

    Removes: video file, WAV files, intermediate JSON files.
    Keeps: merged transcript, final note.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Status message listing removed files.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)

    to_remove = []
    for ext in ["_threads_video.mp4", "_16k.wav", "_8k.wav"]:
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


# ── Step 9: Quality Gate ──────────────────────────────────────────────────────

def step_threads_quality_gate(task: dict, config: Config) -> str:
    """Run quality gates on the generated Threads note.

    For video posts: checks concept count, key points, density.
    For text posts: checks quote count, size.

    Args:
        task: Task dict with slug, content_type.
        config: Config dataclass.

    Returns:
        Status string: "passed" or raises StepError on failure.

    Raises:
        StepError: If quality gates fail.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    content_type = task.get("content_type", "threads_text")

    folder = "threads"
    note_path = config.vault_path / "Knowledge base" / folder / f"{safe_slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    if "video" in content_type:
        passed, rounds = run_video_quality_gate(note_path, config, task["id"])
        if not passed:
            content = note_path.read_text(encoding="utf-8")
            raise StepError(
                f"Threads video quality gate failed after {rounds} rounds. "
                f"First line: {content.split(chr(10))[0]}",
                error_type=ErrorType.FAIL,
                step_name="threads_quality_gate",
            )
        return json.dumps({"passed": True, "rounds": rounds})
    else:
        # Text posts: simple size + quote check
        content = note_path.read_text(encoding="utf-8")
        quote_count = sum(1 for line in content.split("\n") if line.strip().startswith("> "))
        size = len(content)

        passed = quote_count >= 2 and size >= 2000
        failures = []
        if quote_count < 2:
            failures.append(f"Too few quotes: {quote_count} < 2")
        if size < 2000:
            failures.append(f"Note too short: {size} chars < 2000")

        if not passed:
            raise StepError(
                f"Threads text quality gate failed: {'; '.join(failures)}",
                error_type=ErrorType.FAIL,
                step_name="threads_quality_gate",
            )

        return json.dumps({
            "passed": True,
            "failures": [],
            "rounds": 1,
        })


# ── MinIO Upload ──────────────────────────────────────────────────────────────

def step_threads_minio_upload(task: dict, config: Config) -> str:
    """Upload the generated video/text artifacts to MinIO.

    Video: source-archival (video + transcript). Text: note-archival.
    """
    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    content_type = task.get("content_type", "threads_text")
    note_path = config.vault_path / "Knowledge base" / "threads" / f"{safe_slug}.md"

    if content_type == "threads_video":
        # Source-archival: video + transcript (matches the batch collector).
        video_path = Path(config.temp_dir) / f"{safe_slug}_threads_video.mp4"
        uploaded = upload_video_source(safe_slug, config, video_path, "mp4", note_path)
        return f"Uploaded: {', '.join(uploaded)}"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found: {note_path}")

    # Note-archival: text post.
    return upload_note_archive(safe_slug, config, note_path, "threads")


# ── Step 11: Git Push ─────────────────────────────────────────────────────────

def step_threads_git_push(task: dict, config: Config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Delegates to archival.step_git_push with threads-specific params.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Commit hash.
    """
    from agent.archival import step_git_push as archival_git_push

    slug = task.get("slug", task["id"][:8])
    safe_slug = _slug_path(slug)
    # Pass safe_slug as the slug so archival finds the right file
    modified_task = {**task, "slug": safe_slug}
    return archival_git_push(
        modified_task, config,
        vault_subfolder="threads",
        commit_prefix="add threads:",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_transcript_for_llm(merged: dict) -> str:
    """Format merged transcript as compact text for LLM consumption."""
    return format_transcript_for_llm(merged, style="seconds")


# ── Pipeline Registration ────────────────────────────────────────────────────

# Threads video pipeline: detection → download → audio → diarize → transcribe → merge → synthesize → quality → cleanup → minio → git
register_pipeline("threads_video", [
    Step(name="threads_detect", fn=step_threads_detect, on_error="pause"),
    Step(name="threads_extract_video", fn=step_threads_extract_video, on_error="retry"),
    Step(name="extract_audio", fn=step_extract_audio, on_error="retry"),
    Step(name="diarize", fn=step_diarize, on_error="retry"),
    Step(name="transcribe", fn=step_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=step_threads_merge, on_error="retry"),
    Step(name="threads_video_synthesize", fn=step_threads_video_synthesize, on_error="retry"),
    Step(name="threads_quality_gate", fn=step_threads_quality_gate, on_error="retry"),
    Step(name="threads_minio_upload", fn=step_threads_minio_upload, on_error="retry"),
    Step(name="threads_cleanup", fn=step_threads_cleanup, on_error="fail"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="threads_git_push", fn=step_threads_git_push, on_error="retry"),
])

register_pipeline("threads_video_synthesize", [
    Step(name="fetch_transcript", fn=step_fetch_transcript, on_error="retry"),
    Step(name="threads_video_synthesize", fn=step_threads_video_synthesize, on_error="retry"),
    Step(name="threads_quality_gate", fn=step_threads_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="threads_git_push", fn=step_threads_git_push, on_error="retry"),
])

# Threads text pipeline: detection → extract_text → synthesize → quality → minio → git
register_pipeline("threads_text", [
    Step(name="threads_detect", fn=step_threads_detect, on_error="pause"),
    Step(name="threads_extract_text", fn=step_threads_extract_text, on_error="retry"),
    Step(name="threads_text_synthesize", fn=step_threads_text_synthesize, on_error="retry"),
    Step(name="threads_quality_gate", fn=step_threads_quality_gate, on_error="retry"),
    Step(name="threads_minio_upload", fn=step_threads_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="threads_git_push", fn=step_threads_git_push, on_error="retry"),
])
