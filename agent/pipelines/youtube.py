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
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMClient, get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.quality import quality_gate_retry_loop, quality_gate_youtube

# ── Constants ─────────────────────────────────────────────────────────────────

# Fallback yt-dlp config path (system-wide preferred)
YT_COOKIES_FILE = "/root/.hermes/cookies/yt_cookies.txt"

# yt-dlp rate limiting (per video-pipeline-global.md)
YTDL_RATE_FLAGS = [
    "--sleep-requests", "1",
    "--sleep-interval", "3",
    "--max-sleep-interval", "10",
    "--limit-rate", "4M",
]

# Max retry rounds for quality gate
MAX_QUALITY_RETRIES = 3

# Density target for YouTube notes (chars/min)
YOUTUBE_DENSITY_MIN = 100

# Approximate token ratio
CHARS_PER_TOKEN = 4

# ── Cookie Pre-flight ─────────────────────────────────────────────────────────

def _validate_cookies() -> None:
    """Validate YouTube cookies before any yt-dlp operation.

    Tier 1: Check cookie file for required auth tokens.
    Tier 2: Test with oembed API.

    Raises:
        StepError(PAUSE): If cookies are invalid or missing.
    """
    cookie_path = Path(YT_COOKIES_FILE)
    if not cookie_path.exists():
        raise StepError(
            f"YouTube cookies not found: {YT_COOKIES_FILE}",
            error_type=ErrorType.PAUSE,
            step_name="cookie_check",
            details="Export cookies from Chrome Profile 5 to /root/.hermes/cookies/yt_cookies.txt",
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
    _validate_cookies()

    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])

    # --- Metadata ---
    result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", YT_COOKIES_FILE,
            "--js-runtimes", "node",
            "--print", "%(title)s||%(uploader)s||%(duration)s||%(view_count)s||%(upload_date)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=120,
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
        "title": parts[0],
        "uploader": parts[1],
        "duration": float(parts[2]),
        "view_count": int(parts[3]) if parts[3] not in ("NA", "None", "") else 0,
        "upload_date": parts[4],
        "description": "",
        "chapters": [],
    }

    # --- Chapters ---
    chapter_result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", YT_COOKIES_FILE,
            "--js-runtimes", "node",
            "--print", "%(chapters)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if chapter_result.returncode == 0 and chapter_result.stdout.strip():
        try:
            chapters_raw = json.loads(chapter_result.stdout)
            if isinstance(chapters_raw, list):
                metadata["chapters"] = [
                    {
                        "start_time": c.get("start_time", 0),
                        "end_time": c.get("end_time", 0),
                        "title": c.get("title", ""),
                    }
                    for c in chapters_raw
                    if c.get("title")  # Skip empty-title chapters
                ]
        except (json.JSONDecodeError, TypeError):
            pass

    # --- Description ---
    desc_result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", YT_COOKIES_FILE,
            "--js-runtimes", "node",
            "--print", "%(description)s",
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if desc_result.returncode == 0:
        metadata["description"] = desc_result.stdout.strip()

    # Save metadata to temp
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Save slug for later steps
    _slug_path(config, slug).write_text(slug)

    return json.dumps(metadata)


def _slug_path(config: Config, slug: str) -> Path:
    """Path to the slug marker file."""
    return Path(config.temp_dir) / f"{slug}_slug.txt"


# ── Video Download ───────────────────────────────────────────────────────────

def step_download_video(task: dict, config: Config) -> str:
    """Download YouTube video in WebM VP9 720p.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the downloaded .webm file.

    Raises:
        StepError(RETRY): On download failure.
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])
    output_path = Path(config.temp_dir) / f"{slug}.webm"

    result = subprocess.run(
        [
            "yt-dlp",
            "--cookies", YT_COOKIES_FILE,
            "--js-runtimes", "node",
            "-f", "bestvideo[height<=720][vcodec^=vp9]+bestaudio[acodec^=opus]/bestvideo[height<=720]+bestaudio/best[height<=720]",
            "--merge-output-format", "webm",
            "-o", str(output_path),
            *YTDL_RATE_FLAGS,
            url,
        ],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        # Try fallback format
        result = subprocess.run(
            [
                "yt-dlp",
                "--cookies", YT_COOKIES_FILE,
                "--js-runtimes", "node",
                "-f", "best[height<=720]",
                "--merge-output-format", "webm",
                "-o", str(output_path),
                *YTDL_RATE_FLAGS,
                url,
            ],
            capture_output=True, text=True, timeout=600,
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

    return str(output_path)


# ── Audio Extraction ─────────────────────────────────────────────────────────

def step_extract_audio(task: dict, config: Config) -> str:
    """Extract dual audio: 16kHz WAV (whisper) + 8kHz WAV (pyannote).

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        JSON string with paths to both WAV files.
    """
    slug = task.get("slug", task["id"][:8])
    video_path = Path(config.temp_dir) / f"{slug}.webm"
    wav_16k_path = Path(config.temp_dir) / f"{slug}_16k.wav"
    wav_8k_path = Path(config.temp_dir) / f"{slug}_8k.wav"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract 16kHz WAV for whisper
    result_16k = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_16k_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result_16k.returncode != 0:
        raise StepError(
            f"16kHz audio extraction failed: {result_16k.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="extract_audio",
        )

    # Extract 8kHz WAV for pyannote
    result_8k = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_8k_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result_8k.returncode != 0:
        raise StepError(
            f"8kHz audio extraction failed: {result_8k.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="extract_audio",
        )

    paths = {
        "wav_16k": str(wav_16k_path),
        "wav_8k": str(wav_8k_path),
    }
    return json.dumps(paths)


# ── Diarization ──────────────────────────────────────────────────────────────

def step_diarize(task: dict, config: Config) -> str:
    """Diarize audio via pyannote using the canonical script.

    MUST run before transcription (sequential — RAM safety).

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Path to the diarization JSON file.
    """
    slug = task.get("slug", task["id"][:8])
    audio_8k = Path(config.temp_dir) / f"{slug}_8k.wav"
    output_json = Path(config.temp_dir) / f"{slug}_diarization.json"

    if not audio_8k.exists():
        raise FileNotFoundError(f"8kHz audio not found: {audio_8k}")

    # Use canonical script from kb-agent scripts dir
    script_path = config.kb_agent_dir / "scripts" / "diarize.py"
    if not script_path.exists():
        raise StepError(
            f"Diarize script not found: {script_path}",
            error_type=ErrorType.FAIL,
            step_name="diarize",
        )

    result = subprocess.run(
        ["python3", str(script_path), str(audio_8k), str(output_json)],
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "401" in stderr or "403" in stderr:
            raise StepError(
                "Pyannote model access denied — check HF_TOKEN",
                error_type=ErrorType.PAUSE,
                step_name="diarize",
                details=stderr[:500],
            )
        raise StepError(
            f"Diarization failed: {stderr[:200]}",
            error_type=ErrorType.RETRY,
            step_name="diarize",
        )

    if not output_json.exists():
        raise StepError(
            f"Diarization claimed success but output missing: {output_json}",
            error_type=ErrorType.RETRY,
            step_name="diarize",
        )

    return str(output_json)


# ── Transcription ────────────────────────────────────────────────────────────

def step_transcribe(task: dict, config: Config) -> str:
    """Transcribe audio via faster-whisper using the canonical script.

    MUST run after diarization (sequential — RAM safety).

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Path to the transcript JSON file.
    """
    slug = task.get("slug", task["id"][:8])
    audio_16k = Path(config.temp_dir) / f"{slug}_16k.wav"
    output_json = Path(config.temp_dir) / f"{slug}_transcript.json"

    # Get duration from metadata
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    duration_secs = "0"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        duration_secs = str(meta.get("duration", 0))

    if not audio_16k.exists():
        raise FileNotFoundError(f"16kHz audio not found: {audio_16k}")

    # Use canonical script
    script_path = config.kb_agent_dir / "scripts" / "transcribe.py"
    if not script_path.exists():
        raise StepError(
            f"Transcribe script not found: {script_path}",
            error_type=ErrorType.FAIL,
            step_name="transcribe",
        )

    result = subprocess.run(
        ["python3", str(script_path), str(audio_16k), str(output_json), duration_secs],
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode != 0:
        raise StepError(
            f"Transcription failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="transcribe",
        )

    if not output_json.exists():
        raise StepError(
            f"Transcription claimed success but output missing: {output_json}",
            error_type=ErrorType.RETRY,
            step_name="transcribe",
        )

    return str(output_json)


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
    slug = task.get("slug", task["id"][:8])
    diar_path = Path(config.temp_dir) / f"{slug}_diarization.json"
    trans_path = Path(config.temp_dir) / f"{slug}_transcript.json"
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"

    if not diar_path.exists():
        raise FileNotFoundError(f"Diarization not found: {diar_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transcript not found: {trans_path}")

    diar = json.loads(diar_path.read_text())
    trans = json.loads(trans_path.read_text())

    dia_segments = diar.get("segments", [])
    if not dia_segments:
        raise StepError(
            "Diarization produced zero segments",
            error_type=ErrorType.FAIL,
            step_name="merge",
        )

    # Merge: assign speaker to each transcript segment at midpoint
    for seg in trans.get("segments", []):
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2

        best_speaker = None
        best_overlap = 0
        for dia in dia_segments:
            if dia.get("start", 0) <= seg_mid <= dia.get("end", 0):
                speaker = dia.get("speaker", "Unknown")
                # Check for overlap composite labels (SPEAKER_00 | SPEAKER_01)
                if "|" in speaker:
                    best_overlap = max(best_overlap, 1)
                    if best_speaker is None:
                        best_speaker = speaker
                else:
                    if best_speaker is None or "|" in best_speaker:
                        best_speaker = speaker

        seg["speaker"] = best_speaker if best_speaker else "Unknown"
        if "|" in str(seg.get("speaker", "")):
            seg["overlap"] = True

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
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    merged_path.write_text(json.dumps(trans, indent=2, ensure_ascii=False))

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

    Keeps video (.webm) and merged transcript JSON.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Status message.
    """
    slug = task.get("slug", task["id"][:8])
    removed = []

    for suffix in [f"{slug}_8k.wav", f"{slug}_16k.wav",
                    f"{slug}_diarization.json"]:
        path = Path(config.temp_dir) / suffix
        if path.exists():
            path.unlink()
            removed.append(suffix)

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
    slug = task.get("slug", task["id"][:8])
    source_url = task.get("source_url", "")

    # Load merged transcript
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged transcript not found: {merged_path}")

    merged = json.loads(merged_path.read_text())

    # Load metadata
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = {}

    llm = get_llm_client(config)

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
        transcript_text, target_concepts, llm, slug,
    )

    # Pass 2: Build full note
    note_content = _pass2_build_note(
        transcript_text, concepts, metadata, merged, source_url, slug, config, llm,
    )

    # Write note to vault
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note_content, encoding="utf-8")

    return str(note_path)


def _format_transcript_for_llm(merged: dict) -> str:
    """Format merged transcript as compact text for LLM consumption.

    Format: (start) SPEAKER: text
    """
    lines = []
    for seg in merged.get("segments", []):
        start = seg.get("start", 0)
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        overlap = " [OVERLAP]" if seg.get("overlap") else ""

        h = int(start // 3600)
        m = int((start % 3600) // 60)
        s = int(start % 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        lines.append(f"({ts}) {speaker}{overlap}: {text}")

    return "\n".join(lines)


def _pass1_extract_concepts(
    transcript: str,
    target_range: tuple[int, int],
    llm: LLMClient,
    slug: str,
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

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
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
        # Save raw response for debugging
        debug_path = Path(f"/tmp/{slug}_pass1_debug.txt")
        debug_path.write_text(content)
        raise StepError(
            f"Pass 1 concepts JSON parse failed. Raw output saved to {debug_path}",
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
  source_url: {source_url}
  source_files:
    video: minio://kb-agent-archive/videos/{slug}.webm
    audio: minio://kb-agent-archive/videos/{slug}.mp3
    transcript: minio://kb-agent-archive/videos/{slug}.json
  confidence: plausible
  tags: [tag1, tag2, tag3]
- Then the note sections as shown in the template.
- Leave minio: field for later filling.
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

    response = llm.call(system_prompt=system_prompt, user_prompt=user_prompt)
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

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        'passed' if all gates pass.

    Raises:
        StepError: If quality gates fail after max retries.
    """
    slug = task.get("slug", task["id"][:8])
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    # Run standard YouTube quality gates (concepts + key points)
    llm = get_llm_client(config)

    passed, rounds = quality_gate_retry_loop(
        note_path, config, quality_gate_youtube, llm, max_rounds=MAX_QUALITY_RETRIES,
    )

    if not passed:
        content = note_path.read_text(encoding="utf-8")
        raise StepError(
            f"Quality gate failed after {rounds} rounds. "
            f"First line: {content.split(chr(10))[0]}",
            error_type=ErrorType.FAIL,
            step_name="quality_gate",
        )

    # Additional check: density (chars/min)
    _check_density(note_path, config)

    return "passed"


def _check_density(note_path: Path, config: Config) -> None:
    """Verify note density meets minimum chars/min threshold."""
    slug = note_path.stem
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"

    if not meta_path.exists():
        return  # Can't check density without metadata

    meta = json.loads(meta_path.read_text())
    duration_secs = meta.get("duration", 0)
    if duration_secs < 60:
        return  # Skip for very short videos

    duration_min = duration_secs / 60
    note_size = len(note_path.read_text(encoding="utf-8"))
    density = note_size / duration_min
    min_density = getattr(config, "youtube_min_density", YOUTUBE_DENSITY_MIN)

    if density < min_density:
        raise ValueError(
            f"Note density too low: {density:.0f} chars/min < "
            f"{min_density} required "
            f"({note_size} chars / {duration_min:.1f} min)"
        )


# ── MinIO Upload ─────────────────────────────────────────────────────────────

def step_minio_upload(task: dict, config: Config) -> str:
    """Upload video, audio, and transcript to MinIO.
    
    Uploads .webm, extracted .mp3, and transcript JSON.
    Delegates upload to archival.upload_file_to_minio.
    Updates note frontmatter with MinIO URLs.

    Args:
        task: Task dict with slug.
        config: Config dataclass.

    Returns:
        Status message listing uploaded files.
    """
    from agent.archival import upload_file_to_minio

    slug = task.get("slug", task["id"][:8])
    bucket = config.minio_bucket
    uploaded = []

    # 1. Upload video (.webm)
    video_path = Path(config.temp_dir) / f"{slug}.webm"
    if video_path.exists():
        dest = f"{bucket}/videos/{slug}.webm"
        upload_file_to_minio(video_path, dest, timeout=120)
        uploaded.append(f"videos/{slug}.webm")

    # 2. Extract and upload MP3
    mp3_path = Path(config.temp_dir) / f"{slug}.mp3"
    if video_path.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and mp3_path.exists():
            dest = f"{bucket}/videos/{slug}.mp3"
            upload_file_to_minio(mp3_path, dest, timeout=120)
            uploaded.append(f"videos/{slug}.mp3")

    # 3. Upload merged transcript JSON
    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    if merged_path.exists():
        dest = f"{bucket}/videos/{slug}.json"
        upload_file_to_minio(merged_path, dest, timeout=120)
        uploaded.append(f"videos/{slug}.json")

    # 4. Update note frontmatter with MinIO URLs
    _update_note_minio_urls(slug, config, uploaded)

    return f"Uploaded: {', '.join(uploaded)}"


def _update_note_minio_urls(slug: str, config: Config, uploaded: list) -> None:
    """Update note frontmatter with actual MinIO URLs."""
    note_path = config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"
    if not note_path.exists():
        return

    tailscale_base = "http://vmi3304846.tail5c02a1.ts.net:9000"
    content = note_path.read_text(encoding="utf-8")

    # Build source_files block
    source_files_lines = ["source_files:"]
    for u in uploaded:
        source_files_lines.append(f"  {Path(u).stem}: {tailscale_base}/{config.minio_bucket}/{u}")
    source_files_block = "\n".join(source_files_lines)

    # Replace placeholder source_files with real URLs
    if "source_files:" in content:
        content = re.sub(
            r'source_files:.*?(?=\n\w|\Z)',
            source_files_block,
            content,
            flags=re.DOTALL,
        )

    note_path.write_text(content, encoding="utf-8")


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
