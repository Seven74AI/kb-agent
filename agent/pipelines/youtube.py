"""YouTube video pipeline — download, diarize, transcribe, two-pass synthesis.

Full pipeline: yt-dlp metadata + download → ffmpeg dual WAV extraction →
pyannote diarization → faster-whisper transcription → merge + speaker ID +
chapter generation → LLM two-pass synthesis → quality gates → MinIO archival
→ Git push.

Uses canonical scripts at /root/kb-agent/scripts/ (symlinked to
knowledge-base skill). Diarization mandatory for all video content.
"""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from agent.archival import append_note_tags
from agent.config import Config
from agent.fetch_guard import _assert_fetch_url_allowed
from agent.db import init_db, LOGS_SCHEMA
from agent.exceptions import ErrorType, StepError
from agent.redact import redact_http_urls_in_text, url_for_display, write_merged_transcript_json, write_vault_note
from agent.llm import LLMClient, get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.video_common import (
    VIDEO_DENSITY_MIN,
    _artifact_stem,
    _find_video_file,
    check_density as _check_density,
    ensure_h264,
    format_transcript_for_llm as _format_transcript_for_llm,
    merge_speaker_labels,
    run_video_quality_gate,
    step_diarize,
    step_extract_audio,
    step_fetch_transcript,
    step_transcribe,
    upload_video_artifacts,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# yt-dlp binary — resolved explicitly so it works regardless of PATH (the cron
# and MCP run with a system PATH that lacks the py3.11 venv's bin).
YTDLP_BIN = shutil.which("yt-dlp") or "/usr/local/lib/hermes-agent/venv/bin/yt-dlp"

# yt-dlp is a delegated fetcher: only the user-supplied input URL is validated via
# fetch_guard. Media segment URLs yt-dlp opens internally are outside fetch_guard
# (accepted residual, same class as Flaresolverr).

# yt-dlp rate limiting (per video-pipeline-global.md)
YTDL_RATE_FLAGS = [
    "--sleep-requests", "1",
    "--sleep-interval", "3",
    "--max-sleep-interval", "10",
    "--limit-rate", "4M",
]

# Max retry rounds for quality gate (re-exported from video_common)
# MAX_QUALITY_RETRIES is imported above.

# Density target for YouTube notes (chars/min) — re-exported for bunny_stream.
YOUTUBE_DENSITY_MIN = VIDEO_DENSITY_MIN

# Approximate token ratio
CHARS_PER_TOKEN = 4


def _youtube_cookies_path(config: Config) -> Path:
    """Netscape cookie file for yt-dlp (uploaded via kba.sevend.io dashboard)."""
    return Path(config.cookie_dir) / "yt_cookies.txt"


def _ytdlp_subprocess_env(config: Config) -> dict[str, str]:
    """Load YT cookies via XDG yt-dlp config so paths stay off subprocess argv."""
    cookie_path = _youtube_cookies_path(config)
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


def _download_artifact_stem(task: dict) -> str:
    """Temp-dir stem for download-phase artifacts (id-keyed in nightly batch)."""
    return _artifact_stem(task)


# ── Cookie Pre-flight ─────────────────────────────────────────────────────────

def _validate_cookies(config: Config) -> None:
    """Validate YouTube cookies before any yt-dlp operation.

    Tier 1: Check cookie file for required auth tokens.
    Tier 2: Test with oembed API.

    Raises:
        StepError(PAUSE): If cookies are invalid or missing.
    """
    cookie_path = _youtube_cookies_path(config)
    if not cookie_path.exists():
        raise StepError(
            f"YouTube cookies not found: {cookie_path}",
            error_type=ErrorType.PAUSE,
            step_name="cookie_check",
            details=(
                "Upload fresh cookies via the kba.sevend.io dashboard "
                f"(saves to {cookie_path})"
            ),
        )

    cookies_text = cookie_path.read_text()

    # Tier 1: Check for required auth tokens
    required = ["LOGIN_INFO", "SAPISID"]
    missing = [t for t in required if t not in cookies_text]
    # __Secure-3PSID or 3PSID (one of them must be present)
    has_3psid = "3PSID" in cookies_text

    if missing or not has_3psid:
        msg = f"Cookies missing required tokens: {missing}"
        if not has_3psid:
            msg += " and 3PSID"
        raise StepError(
            msg,
            error_type=ErrorType.PAUSE,
            step_name="cookie_check",
            details="Cookies expired — re-export from Chrome Profile 5",
        )

    # Tier 2: Test with oembed API
    # (Note: full Tier 2 test requires a video URL; defer to metadata extraction)
    # Quick structural check passed — cookies are well-formed


# ── Metadata Extraction ──────────────────────────────────────────────────────

def step_extract_metadata(task: dict, config: Config) -> str:
    """Extract YouTube video metadata via yt-dlp.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        JSON string with title, uploader, duration, view_count, upload_date,
        description, chapters.

    Raises:
        StepError: On cookie validation failure, PAUSE on fingerprinting,
                   FAIL on 404/private video.
    """
    _validate_cookies(config)

    url = task["source_url"]
    stem = _download_artifact_stem(task)
    _assert_fetch_url_allowed(url)
    ytdlp_env = _ytdlp_subprocess_env(config)

    # --- Metadata ---
    result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "--no-update",
            "--print", "%(title)s||%(uploader)s||%(duration)s||%(view_count)s||%(upload_date)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=ytdlp_env,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Sign in to confirm" in stderr:
            raise StepError(
                "YouTube bot detection — IP fingerprinting",
                error_type=ErrorType.PAUSE,
                step_name="extract_metadata",
                details="Run yt-dlp on Mac via --cookies-from-browser",
            )
        if "Private video" in stderr or "Video unavailable" in stderr:
            raise StepError(
                "Video is private or unavailable",
                error_type=ErrorType.FAIL,
                step_name="extract_metadata",
            )
        raise StepError(
            f"yt-dlp metadata failed: {stderr[:200]}",
            error_type=ErrorType.RETRY,
            step_name="extract_metadata",
        )

    parts = result.stdout.strip().split("||")
    if len(parts) < 5:
        raise StepError(
            f"Metadata parse error: expected 5 fields, got {len(parts)}",
            error_type=ErrorType.RETRY,
            step_name="extract_metadata",
        )

    metadata = {
        "title": redact_http_urls_in_text(parts[0]),
        "uploader": redact_http_urls_in_text(parts[1]),
        "duration": float(parts[2]) if parts[2] not in ("NA", "None", "") else 0.0,
        "view_count": int(parts[3]) if parts[3] not in ("NA", "None", "") else 0,
        "upload_date": parts[4],
        "description": "",
        "chapters": [],
    }

    # --- Chapters ---
    chapter_result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "--no-update",
            "--print", "%(chapters)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=ytdlp_env,
    )
    if chapter_result.returncode == 0 and chapter_result.stdout.strip():
        try:
            chapters_raw = json.loads(chapter_result.stdout)
            if isinstance(chapters_raw, list):
                metadata["chapters"] = [
                    {
                        "start_time": c.get("start_time", 0),
                        "end_time": c.get("end_time", 0),
                        "title": redact_http_urls_in_text(c.get("title", "")),
                    }
                    for c in chapters_raw
                    if c.get("title")  # Skip empty-title chapters
                ]
        except (json.JSONDecodeError, TypeError):
            pass

    # --- Description ---
    desc_result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "--no-update",
            "--print", "%(description)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=ytdlp_env,
    )
    if desc_result.returncode == 0:
        metadata["description"] = desc_result.stdout.strip()

    # Save metadata to temp
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    return json.dumps(metadata)


# ── Video Download ───────────────────────────────────────────────────────────

def step_download_video(task: dict, config: Config) -> str:
    """Download YouTube video in WebM VP9 720p.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the downloaded .mp4 file.

    Raises:
        StepError(RETRY): On download failure.
    """
    url = task["source_url"]
    stem = _download_artifact_stem(task)
    output_path = Path(config.temp_dir) / f"{stem}.mp4"
    try:
        _assert_fetch_url_allowed(url)
    except ValueError as exc:
        raise StepError(
            str(exc),
            error_type=ErrorType.FAIL,
            step_name="download_video",
        ) from exc

    ytdlp_env = _ytdlp_subprocess_env(config)
    result = subprocess.run(
        [
            YTDLP_BIN,
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "--no-update",
            "-f", "bestvideo[height<=720][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<=720]+bestaudio/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=ytdlp_env,
    )

    if result.returncode != 0:
        # Try fallback format
        result = subprocess.run(
            [
                YTDLP_BIN,
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                "--no-update",
                "-f", "best[height<=720]",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                *YTDL_RATE_FLAGS,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=ytdlp_env,
        )

    if result.returncode != 0:
        raise StepError(
            f"Video download failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="download_video",
        )

    if not output_path.exists():
        raise StepError(
            f"Download claimed success but file missing: {output_path}",
            error_type=ErrorType.RETRY,
            step_name="download_video",
        )

    # Some YouTube videos only offer VP9/AV1 at the chosen height — normalize
    # to H.264 so the archived video plays in every browser (incl. Safari).
    output_path = Path(ensure_h264(output_path))

    return str(output_path)


# ── Audio Extraction ─────────────────────────────────────────────────────────
# (step_extract_audio is imported from video_common)

# ── Diarization ──────────────────────────────────────────────────────────────
# (step_diarize is imported from video_common)

# ── Transcription ────────────────────────────────────────────────────────────
# (step_transcribe is imported from video_common)

# ── Merge Diarization + Transcript ───────────────────────────────────────────

def step_merge_and_identify(task: dict, config: Config) -> str:
    """Merge diarization speaker labels with transcript segments.

    Merges at segment midpoint. Adds speaker labels and overlap flags.
    Attempts speaker identification from metadata heuristics.
    Generates chapters (native or NLP fallback).

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string summarizing merge result.
    """
    stem = _download_artifact_stem(task)
    diar_path = Path(config.temp_dir) / f"{stem}_diarization.json"
    trans_path = Path(config.temp_dir) / f"{stem}_transcript.json"
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"

    if not diar_path.exists():
        raise FileNotFoundError(f"Diarization not found: {diar_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transcript not found: {trans_path}")

    diar = json.loads(diar_path.read_text())
    trans = json.loads(trans_path.read_text())

    merge_speaker_labels(diar, trans)

    # --- Speaker Identification (heuristic) ---
    speaker_names = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        speaker_names = _identify_speakers(meta, trans)

    if speaker_names:
        trans["speaker_names"] = speaker_names

    # --- Chapter Generation ---
    chapters = _generate_chapters(meta_path, trans)

    trans["chapters"] = chapters

    # Write merged transcript
    merged_path = Path(config.temp_dir) / f"{stem}_merged.json"
    write_merged_transcript_json(merged_path, trans)

    # Count stats
    num_speakers = len(set(
        s.get("speaker", "Unknown")
        for s in trans.get("segments", [])
    ))
    num_overlap = sum(
        1 for s in trans.get("segments", [])
        if s.get("overlap")
    )

    result = {
        "segments": len(trans.get("segments", [])),
        "speakers": num_speakers,
        "overlap_segments": num_overlap,
        "speaker_names": speaker_names,
        "chapters_count": len(chapters),
        "merged_path": str(merged_path),
    }
    return json.dumps(result)


def _identify_speakers(metadata: dict, transcript: dict) -> dict:
    """Heuristic speaker identification from video metadata.

    Returns:
        Dict mapping SPEAKER_XX → real name.
    """
    desc = metadata.get("description", "")
    uploader = metadata.get("uploader", "")
    title = metadata.get("title", "")

    # Collect distinct speaker labels
    speakers = set(
        s.get("speaker", "Unknown").split(" | ")[0]
        for s in transcript.get("segments", [])
        if s.get("speaker", "Unknown") != "Unknown"
    )
    speakers.discard("Unknown")

    if not speakers:
        return {}

    # Find proper names in description (capitalized multi-word)
    names = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', desc))
    # Filter out common false positives
    skip_words = {"The", "This", "That", "These", "Those", "About", "Watch",
                  "More", "Follow", "Like", "Share", "Subscribe", "Comment"}
    names = {n for n in names if n.split()[0] not in skip_words}

    # If channel name looks like a personal name, include it
    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', uploader):
        names.add(uploader)

    # Check video title for "X interview Y" / "X & Y" patterns
    title_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', title)
    names.update(title_names)

    mapping = {}
    sorted_speakers = sorted(speakers)
    sorted_names = sorted(names)

    if len(sorted_speakers) == len(sorted_names) and len(sorted_speakers) <= 4:
        mapping = dict(zip(sorted_speakers, sorted_names))
    elif len(sorted_speakers) <= len(sorted_names):
        # Assign the first N names
        for i, spk in enumerate(sorted_speakers):
            if i < len(sorted_names):
                mapping[spk] = sorted_names[i]

    return mapping


def _generate_chapters(meta_path: Path, transcript: dict) -> list[dict]:
    """Generate chapters from native YouTube data or NLP gap detection.

    Args:
        meta_path: Path to metadata JSON file.
        transcript: Transcript dict with segments.

    Returns:
        List of chapter dicts with start, title, and optional summary.
    """
    # Try native chapters first
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        native_chapters = meta.get("chapters", [])
        # Filter out empty-title chapters and validate
        native_chapters = [
            c for c in native_chapters
            if c.get("title") and c.get("start_time", 0) >= 0
        ]
        if native_chapters:
            return [
                {
                    "start": c["start_time"],
                    "end": c.get("end_time", c["start_time"]),
                    "title": c["title"],
                }
                for c in native_chapters
            ]

    # NLP fallback: gap detection (>3s between segments)
    segments = transcript.get("segments", [])
    if len(segments) < 2:
        return []

    chapters = []
    current_start = segments[0].get("start", 0)
    current_text = []

    for i, seg in enumerate(segments):
        if i > 0:
            gap = seg.get("start", 0) - segments[i - 1].get("end", 0)
            if gap > 3.0 and current_text:
                chapters.append({
                    "start": current_start,
                    "end": segments[i - 1].get("end", 0),
                    "title": f"Section {len(chapters) + 1}",
                    "summary": " ".join(current_text)[:500],
                })
                current_start = seg.get("start", 0)
                current_text = []
        current_text.append(seg.get("text", ""))

    # Final chapter
    if current_text and segments:
        chapters.append({
            "start": current_start,
            "end": segments[-1].get("end", 0),
            "title": f"Section {len(chapters) + 1}",
            "summary": " ".join(current_text)[:500],
        })

    return chapters


# ── Cleanup ──────────────────────────────────────────────────────────────────

def step_cleanup_temp(task: dict, config: Config) -> str:
    """Remove temporary files: 8kHz/16kHz WAV + diarization JSON.

    Keeps video (.mp4) and merged transcript JSON.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Status message.
    """
    stem = _artifact_stem(task)
    removed = []

    for suffix in [f"{stem}_8k.wav", f"{stem}_16k.wav",
                    f"{stem}_diarization.json"]:
        path = Path(config.temp_dir) / suffix
        if path.exists():
            path.unlink()
            removed.append(suffix)

    ytdlp_xdg = Path(config.temp_dir) / "ytdlp_xdg"
    if ytdlp_xdg.exists():
        shutil.rmtree(ytdlp_xdg, ignore_errors=True)
        removed.append("ytdlp_xdg")

    return f"Cleaned up: {', '.join(removed)}"


# ── LLM Two-Pass Synthesis ────────────────────────────────────────────────────

def step_synthesize_youtube(task: dict, config: Config) -> str:
    """Two-pass LLM synthesis for YouTube videos.

    Pass 1: Extract concepts from transcript.
    Pass 2: Build full note from resume-prompt.md template.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.

    Raises:
        StepError: If synthesis fails.
    """
    stem = _artifact_stem(task)
    display_slug = task.get("slug", task["id"][:8])
    source_url = task.get("source_url", "")

    # Load merged transcript
    merged_path = Path(config.temp_dir) / f"{stem}_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged transcript not found: {merged_path}")

    merged = json.loads(merged_path.read_text())

    # Load metadata
    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = {}

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    # Build transcript text for the LLM (compact but complete)
    transcript_text = _format_transcript_for_llm(merged)

    # Determine concept count target
    duration_min = metadata.get("duration", 0) / 60
    if duration_min <= 45:
        target_concepts = (3, 6)
    elif duration_min <= 90:
        target_concepts = (5, 8)
    else:
        target_concepts = (8, 10)

    # Pass 1: Extract concepts
    concepts = _pass1_extract_concepts(
        transcript_text, target_concepts, llm, display_slug, config,
    )

    # Pass 2: Build full note
    note_content = _pass2_build_note(
        transcript_text, concepts, metadata, merged, source_url, display_slug, config, llm,
    )

    # Write note to vault
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{display_slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    write_vault_note(note_path, note_content)
    append_note_tags(note_path, task.get("tags"))

    return str(note_path)


def _pass1_extract_concepts(
    transcript: str,
    target_range: tuple[int, int],
    llm: LLMClient,
    slug: str,
    config: Config,
) -> str:
    """Pass 1: Extract key concepts from transcript.

    Returns:
        JSON string with concepts list.
    """
    lo, hi = target_range
    system_prompt = f"""You are a concept extraction specialist for video transcripts.
Read the FULL transcript and identify {lo}-{hi} key transversal concepts.

For each concept, produce EXACTLY:
- title: short descriptive name
- mechanism: 2-4 sentences explaining the "how" behind the concept, anchored in specific transcript segments
- timestamps: 2-3 moments where this concept appears
- claim: the central claim associated with this concept
- evidence_type: one of [étude_citée, raisonnement, témoignage, assertion]

DO NOT write a summary or note. Only output the concepts as a JSON array.

Respond with ONLY valid JSON:
```json
[
  {{
    "title": "...",
    "mechanism": "...",
    "timestamps": ["MM:SS", "MM:SS"],
    "claim": "...",
    "evidence_type": "..."
  }},
  ...
]
```"""

    user_prompt = (
        f"Extract {lo}-{hi} concepts from this transcript.\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    content = response.content.strip()

    # Extract JSON from response (may have markdown code fences)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        concepts_json = json_match.group(1).strip()
    elif content.startswith("["):
        concepts_json = content
    else:
        # Try to find JSON array anywhere
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            concepts_json = json_match.group()
        else:
            # Fallback: wrap the whole thing
            concepts_json = content

    # Validate JSON
    try:
        concepts = json.loads(concepts_json)
    except json.JSONDecodeError:
        debug_root = Path(config.temp_dir) / "pass1_debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        os.chmod(debug_root, 0o700)
        debug_path = debug_root / f"{slug}_pass1_debug.txt"
        debug_path.write_text(redact_http_urls_in_text(content), encoding="utf-8")
        os.chmod(debug_path, 0o600)
        raise StepError(
            f"Pass 1 concepts JSON parse failed. Raw output saved to {debug_path.name}",
            error_type=ErrorType.RETRY,
            step_name="synthesize_youtube_pass1",
        )

    if not isinstance(concepts, list) or len(concepts) < target_range[0]:
        raise StepError(
            f"Too few concepts extracted: {len(concepts) if isinstance(concepts, list) else 0} < {target_range[0]}",
            error_type=ErrorType.RETRY,
            step_name="synthesize_youtube_pass1",
        )

    return json.dumps(concepts)


def _pass2_build_note(
    transcript: str,
    concepts_json: str,
    metadata: dict,
    merged: dict,
    source_url: str,
    slug: str,
    config: Config,
    llm: LLMClient,
) -> str:
    """Pass 2: Build full knowledge base note from concepts + transcript."""
    
    # Load templates
    resume_prompt = _load_resume_prompt(config)
    note_template = _load_youtube_template(config)

    # Build speaker summary
    speaker_summary = _build_speaker_summary(merged)

    # Build chapter list
    chapters = merged.get("chapters", [])
    chapters_text = "\n".join(
        f"- {c['start']:.0f}s: {c.get('title', 'Untitled')}"
        for c in chapters
    ) if chapters else "No chapters available"

    # Build metadata text
    duration_secs = metadata.get("duration", 0)
    duration_min = int(duration_secs / 60)
    speaker_names = merged.get("speaker_names", {})
    speaker_labels = list(speaker_names.keys()) if speaker_names else ["SPEAKER_00"]

    metadata_text = f"""- **Channel:** {metadata.get('uploader', 'Unknown')}
- **Duration:** {duration_min} min
- **Language:** auto-detected
- **Published:** {metadata.get('upload_date', 'Unknown')}
- **Views:** {metadata.get('view_count', 0)}
- **Speakers:** {', '.join(f'{s} ({speaker_names.get(s, "Unknown")})' for s in speaker_labels)}"""

    date_str = datetime.now().strftime("%Y-%m-%d")
    source_domain = urlparse(source_url).netloc or "youtube.com"
    display_source_url = url_for_display(source_url)

    system_prompt = f"""You are a knowledge base curator specializing in video content synthesis.
Follow the deep treatment standard for YouTube videos.

{resume_prompt}

TEMPLATE (use this exact structure):
{note_template}

QUALITY RULES (non-negotiable):
- Key Concepts: {3}-10 detailed concepts, each with clear mechanism explanation
- Key Points: 8-15 actionable, specific points (no vague statements)
- No hallucinated content: if information is not in the transcript, say so
- Content in the SOURCE LANGUAGE of the video, section labels always in English
- Anchor every claim to the transcript
- Never translate the content — keep original language

METADATA:
{metadata_text}

SPEAKER SUMMARY:
{speaker_summary}

CHAPTERS:
{chapters_text}

CONCEPTS EXTRACTED (from Pass 1):
{concepts_json}

OUTPUT FORMAT:
- Start with YAML frontmatter:
  date: {date_str}
  source: {source_domain} ({metadata.get('uploader', 'Unknown')})
  source_url: {display_source_url}
  source_files:
    video: videos/{slug}.mp4
    transcript: transcripts/{slug}.json
  confidence: plausible
  tags: [tag1, tag2, tag3]
- Then the note sections as shown in the template.
- Leave See Also section empty."""

    # Truncate transcript if excessively long (keep ~50K chars for context window)
    max_transcript = 60000
    if len(transcript) > max_transcript:
        half = max_transcript // 2
        transcript = transcript[:half] + f"\n\n... [{len(transcript) - max_transcript} chars omitted] ...\n\n" + transcript[-half:]

    user_prompt = (
        f"Using the concepts extracted above, the speaker summary, AND the full "
        f"transcript below, write a comprehensive knowledge base note following "
        f"the template and quality rules.\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END ==="
    )

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt, role="synthesis")
    return response.content


def _load_resume_prompt(config: Config) -> str:
    """Load the resume prompt template."""
    path = config.kb_agent_dir / "templates" / "resume-prompt.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_youtube_template(config: Config) -> str:
    """Load the YouTube note template."""
    path = config.kb_agent_dir / "templates" / "youtube-note-template.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _build_speaker_summary(merged: dict) -> str:
    """Build speaker statistics from merged transcript."""
    speakers = {}
    for seg in merged.get("segments", []):
        spk = seg.get("speaker", "Unknown")
        duration = seg.get("end", 0) - seg.get("start", 0)
        is_overlap = seg.get("overlap", False)

        if spk not in speakers:
            speakers[spk] = {"duration": 0, "segments": 0, "overlap": 0}
        speakers[spk]["segments"] += 1
        if not is_overlap:
            speakers[spk]["duration"] += duration
        else:
            speakers[spk]["overlap"] += 1

    lines = []
    for spk, stats in sorted(speakers.items(), key=lambda x: -x[1]["duration"]):
        minutes = int(stats["duration"] / 60)
        name_hint = ""
        if merged.get("speaker_names", {}).get(spk):
            name_hint = f" ({merged['speaker_names'][spk]})"
        lines.append(
            f"- **{spk}{name_hint}** ({minutes} min, {stats['segments']} segments"
            + (f", {stats['overlap']} overlap" if stats["overlap"] else "")
            + ")"
        )

    return "\n".join(lines) if lines else "No speaker data"


# ── Quality Gate ─────────────────────────────────────────────────────────────

def step_quality_gate(task: dict, config: Config) -> str:
    """Run YouTube-specific quality gates on the synthesized note.

    Gates: concept_count >= 3, key_points >= 8, density >= 100 chars/min.
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    passed, rounds = run_video_quality_gate(note_path, config, task["id"])

    if not passed:
        content = note_path.read_text(encoding="utf-8")
        raise StepError(
            f"Quality gate failed after {rounds} rounds. "
            f"First line: {content.split(chr(10))[0]}",
            error_type=ErrorType.FAIL,
            step_name="quality_gate",
        )

    # Additional check: density (chars/min)
    _check_density(note_path, config, task)

    return "passed"


# ── MinIO Upload ─────────────────────────────────────────────────────────────

def step_minio_upload(task: dict, config: Config) -> str:
    """Upload video, audio, and transcript to MinIO.

    Uploads .mp4, extracted .mp3, and transcript JSON. Delegates upload to
    archival.upload_file_to_minio and updates note frontmatter with MinIO URLs.
    """
    stem = _artifact_stem(task)
    display_slug = task.get("slug", task["id"][:8])
    video_path = _find_video_file(config, stem)

    uploaded = upload_video_artifacts(stem, config, video_path)

    # Update note frontmatter with MinIO URLs
    _update_note_minio_urls(display_slug, config, uploaded)

    return f"Uploaded: {', '.join(uploaded)}"


def _update_note_minio_urls(slug: str, config: Config, uploaded: list) -> None:
    """Update note frontmatter with actual MinIO URLs."""
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    if not note_path.exists():
        return
    if not uploaded:
        return

    content = note_path.read_text(encoding="utf-8")

    # Build source_files block
    source_files_lines = ["source_files:"]
    for u in uploaded:
        suffix = Path(u).suffix.lower().lstrip(".")
        kind = {
            "json": "transcript",
            "webm": "video",
            "mp4": "video",
            "mp3": "audio",
            "m4a": "audio",
        }.get(suffix, suffix)
        source_files_lines.append(f"  {kind}: {u}")
    source_files_block = "\n".join(source_files_lines)

    # Replace placeholder source_files with real URLs
    if "source_files:" in content:
        content = re.sub(
            r'source_files:.*?(?=\n\w|\Z)',
            source_files_block,
            content,
            flags=re.DOTALL,
        )

    write_vault_note(note_path, content)


# ── Git Push ─────────────────────────────────────────────────────────────────

def step_git_push(task: dict, config: Config) -> str:
    """Push the generated note to the Obsidian vault Git repo.

    Delegates to archival.step_git_push.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Commit hash.
    """
    from agent.archival import step_git_push as archival_git_push

    return archival_git_push(task, config, vault_subfolder="videos")


# ── Pipeline Registration ────────────────────────────────────────────────────

register_pipeline("youtube", [
    Step(name="extract_metadata", fn=step_extract_metadata, on_error="pause"),
    Step(name="download_video", fn=step_download_video, on_error="retry"),
    Step(name="extract_audio", fn=step_extract_audio, on_error="retry"),
    Step(name="diarize", fn=step_diarize, on_error="retry"),
    Step(name="transcribe", fn=step_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=step_merge_and_identify, on_error="retry"),
    Step(name="cleanup_temp", fn=step_cleanup_temp, on_error="fail"),
    Step(name="synthesize_youtube", fn=step_synthesize_youtube, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("youtube_synthesize", [
    Step(name="fetch_transcript", fn=step_fetch_transcript, on_error="retry"),
    Step(name="synthesize_youtube", fn=step_synthesize_youtube, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
