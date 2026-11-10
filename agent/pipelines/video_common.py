"""Shared helpers for video pipelines.

Holds the duplicated video-handler steps once: audio extraction, diarization,
transcription, speaker-merge, transcript formatting, quality gates, and object storage
archival. Each content-type pipeline (youtube, bunny_stream, instagram, threads,
mega_video) imports the generic implementations and keeps only its
platform-specific download / synthesis / note-path logic.

Also includes the synthesize-only (batch) path: fetch a pre-uploaded merged
transcript from object storage and normalize it for downstream density checks.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from agent.config import Config
from agent.db import LOGS_SCHEMA, init_db
from agent.exceptions import ErrorType, StepError
from agent.notify import sanitize_visible_message
from agent.redact import (
    ensure_merged_transcript_file_redacted,
    redact_http_urls_in_text,
    url_for_display,
    write_merged_transcript_json,
    write_transcription_sidecar_json,
    write_vault_note,
)
from agent.llm import get_llm_client
from agent.quality import quality_gate_retry_loop, quality_gate_threads_video, quality_gate_youtube
from agent.core.slugify import slug_path
from agent.transcription import install_modal_outputs, is_modal_backend, transcribe_modal

# Density target for video notes (chars/min)
VIDEO_DENSITY_MIN = 100

# Max retry rounds for the video-style quality gate
MAX_QUALITY_RETRIES = 3


# ── Slugs ────────────────────────────────────────────────────────────────────

_slug_path = slug_path  # backward-compatible alias for pipeline imports


def _resolve_slug(task: dict) -> str:
    """Resolve the filesystem-safe slug used to name temp artifacts.

    Slugs produced by slugify() (youtube / bunny_stream / mega_video) are
    already safe; instagram / threads slugs may embed URL fragments and are
    sanitized. The sanitizer is a no-op on already-safe slugs, so it is applied
    uniformly.
    """
    return _slug_path(task.get("slug", task["id"][:8]))


def resolve_artifact_stem(task: dict) -> str:
    """Temp-dir artifact stem: ingest id in batch runs, else filesystem-safe slug."""
    if task.get("_batch_ingest"):
        return task["id"]
    return _resolve_slug(task)


_artifact_stem = resolve_artifact_stem


def ensure_h264(video_path: Path) -> Path:
    """Transcode a downloaded video to H.264/MP4 if it isn't already.

    Instagram (and some other sources) can deliver VP9/HEVC video in an .mp4
    container, which browsers — Safari especially — cannot play. Probe the
    video codec and transcode to libx264 when needed. Returns the (possibly
    replaced) video path; falls back to the original on any probe/transcode
    failure so a codec mismatch never blocks ingestion.
    """
    codec = ""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0:
            codec = probe.stdout.strip().lower()
    except Exception:
        codec = ""

    if codec in ("h264", "avc1"):
        return video_path

    out_path = video_path.with_name(video_path.stem + "_h264" + video_path.suffix)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-c:v", "libx264", "-preset", "medium", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(out_path)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            return out_path
    except Exception:
        pass
    return video_path



# ── Audio Extraction ─────────────────────────────────────────────────────────

_VIDEO_SUFFIXES_BATCH = (".mp4", ".webm", ".mkv", ".mov")


def instagram_reel_glob_pattern(task: dict) -> str:
    """Glob pattern for yt-dlp Instagram reel output under the artifact stem."""
    stem = _artifact_stem(task)
    if task.get("_batch_ingest"):
        return f"{stem}.*"
    return f"{stem}_reel.*"


def instagram_reel_ytdlp_output(task: dict, config: Config) -> tuple[Path, str, str]:
    """Return (preferred_path, yt-dlp --output base, post-download glob) for reels."""
    stem = _artifact_stem(task)
    batch = task.get("_batch_ingest")
    glob_pattern = instagram_reel_glob_pattern(task)
    if batch:
        output_path = Path(config.temp_dir) / f"{stem}.mp4"
    else:
        output_path = Path(config.temp_dir) / f"{stem}_reel.mp4"
    return output_path, str(output_path.with_suffix("")), glob_pattern


def resolve_instagram_reel_video(config: Config, task: dict) -> Path:
    """Resolve the on-disk Instagram reel video using download naming rules."""
    stem = _artifact_stem(task)
    batch = task.get("_batch_ingest")
    glob_pattern = instagram_reel_glob_pattern(task)
    actual_files = [
        p for p in Path(config.temp_dir).glob(glob_pattern)
        if p.is_file()
        and (not batch or p.suffix.lower() in _VIDEO_SUFFIXES_BATCH)
    ]
    if actual_files:
        return sorted(actual_files)[0]
    if batch:
        return Path(config.temp_dir) / f"{stem}.mp4"
    return Path(config.temp_dir) / f"{stem}_reel.mp4"


def threads_video_filename(task: dict) -> str:
    """Basename (under temp_dir) for a downloaded Threads video."""
    stem = _artifact_stem(task)
    if task.get("_batch_ingest"):
        return f"{stem}.mp4"
    return f"{stem}_threads_video.mp4"


def resolve_threads_video_path(config: Config, task: dict) -> Path:
    """Resolve the on-disk Threads video path using download naming rules."""
    return Path(config.temp_dir) / threads_video_filename(task)


def threads_raw_path(config: Config, task: dict) -> Path:
    """Canonical temp path for threads raw JSON (detect / extract writers)."""
    stem = _artifact_stem(task)
    return Path(config.temp_dir) / f"{stem}_threads_raw.json"


def resolve_threads_raw_path(config: Config, task: dict) -> Path:
    """Resolve threads raw JSON for synthesis reads (consumer slug, batch id fallback)."""
    slug_path = threads_raw_path(config, {**task, "_batch_ingest": False})
    if slug_path.is_file():
        return slug_path
    task_id = (task.get("id") or "").strip()
    if task_id:
        batch_sidecar = Path(config.temp_dir) / f"{task_id}_threads_raw.json"
        if batch_sidecar.is_file():
            return batch_sidecar
    return slug_path


def sanitize_threads_raw_sidecar(record: dict) -> dict:
    """Return a copy of threads raw JSON safe for at-rest temp storage (display URLs, redacted text)."""
    out = dict(record)
    if out.get("url"):
        out["url"] = url_for_display(str(out["url"]))
    if out.get("parent_url"):
        out["parent_url"] = url_for_display(str(out["parent_url"]))
    if out.get("parent_context") is not None:
        out["parent_context"] = redact_http_urls_in_text(str(out["parent_context"]))
    if out.get("og_description"):
        out["og_description"] = redact_http_urls_in_text(str(out["og_description"]))
    captions = out.get("captions")
    if isinstance(captions, list):
        out["captions"] = [redact_http_urls_in_text(str(c)) for c in captions]
    return out


def write_threads_raw_sidecar(config: Config, task: dict, record: dict) -> Path:
    """Write ``{stem}_threads_raw.json`` with display-safe URL fields and redacted free text."""
    raw_path = threads_raw_path(config, task)
    safe = sanitize_threads_raw_sidecar(record)
    from agent.persist import persist_durable_json

    persist_durable_json(raw_path, safe, policy="boundary4")
    return raw_path


def resolve_batch_ingest_video(
    config: Config,
    task: dict,
    *,
    hint: Path | str | None = None,
) -> Path | None:
    """Locate the staged video for a batch-ingest row (ingest-id keyed artifacts)."""
    stem = _artifact_stem(task)
    tmp = Path(config.temp_dir)

    if hint is not None:
        hint_path = Path(hint)
        if hint_path.is_file() and hint_path.suffix.lower() in _VIDEO_SUFFIXES_BATCH:
            return hint_path

    candidates: list[Path] = []
    for pattern in (f"{stem}*.mp4", f"{stem}.*"):
        for path in tmp.glob(pattern):
            if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES_BATCH:
                if path not in candidates:
                    candidates.append(path)

    if not candidates:
        return None

    def sort_key(p: Path) -> tuple:
        h264_rank = 0 if "_h264" in p.stem else 1
        try:
            size = -p.stat().st_size
        except OSError:
            size = 0
        return (h264_rank, size, p.name)

    return sorted(candidates, key=sort_key)[0]


def _find_video_file(config: Config, slug: str) -> Path:
    """Locate the downloaded video for a slug across the known naming schemes."""
    tmp = Path(config.temp_dir)
    for name in (f"{slug}.webm", f"{slug}.mp4", f"{slug}_threads_video.mp4"):
        path = tmp / name
        if path.exists():
            return path
    # ensure_h264 may leave only {slug}_h264.mp4 on disk.
    for path in sorted(tmp.glob(f"{slug}*.mp4")):
        if path.is_file():
            return path
    # Instagram reels — yt-dlp may append its own extension.
    for path in sorted(tmp.glob(f"{slug}_reel.*")):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Video file not found for slug: {slug}")


def extract_dual_wav(video_path: Path, slug: str, config: Config,
                     step_name: str = "extract_audio") -> dict:
    """Extract 16kHz (whisper) + 8kHz (pyannote) mono WAVs from a video.

    Returns a dict with the ``wav_16k`` / ``wav_8k`` paths.
    """
    wav_16k_path = Path(config.temp_dir) / f"{slug}_16k.wav"
    wav_8k_path = Path(config.temp_dir) / f"{slug}_8k.wav"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result_16k = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_16k_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result_16k.returncode != 0:
        raise StepError(
            f"16kHz audio extraction failed: {result_16k.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name=step_name,
        )

    result_8k = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(wav_8k_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result_8k.returncode != 0:
        raise StepError(
            f"8kHz audio extraction failed: {result_8k.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name=step_name,
        )

    return {"wav_16k": str(wav_16k_path), "wav_8k": str(wav_8k_path)}


def step_extract_audio(task: dict, config: Config) -> str:
    """Extract dual WAV audio (16kHz + 8kHz) from the downloaded video."""
    slug = _artifact_stem(task)
    video_path = _find_video_file(config, slug)
    return json.dumps(extract_dual_wav(video_path, slug, config))


# ── Diarization ──────────────────────────────────────────────────────────────

def step_diarize(task: dict, config: Config) -> str:
    """Diarize audio via pyannote using the canonical script.

    MUST run before transcription (sequential — RAM safety).
    Returns the path to the diarization JSON file.
    """
    slug = _artifact_stem(task)
    audio_8k = Path(config.temp_dir) / f"{slug}_8k.wav"
    output_json = Path(config.temp_dir) / f"{slug}_diarization.json"

    if is_modal_backend(config):
        audio_16k = Path(config.temp_dir) / f"{slug}_16k.wav"
        if not audio_16k.exists():
            raise FileNotFoundError(f"16kHz audio not found: {audio_16k}")

        logs_conn = task.get("_logs_conn")
        task_id = task.get("_task_id", "")
        step_name = task.get("_step_name", "diarize")
        if logs_conn:
            from agent.streaming import log_progress
            log_progress(logs_conn, task_id, step_name,
                         f"Starting Modal GPU diarization+transcription for {slug}")

        diar_modal, trans_modal = transcribe_modal(slug, audio_16k, config)
        install_modal_outputs(slug, config, diar_modal, trans_modal)

        if logs_conn:
            from agent.streaming import log_progress
            log_progress(logs_conn, task_id, step_name,
                         "Modal diarization complete")
        return str(output_json)

    if not audio_8k.exists():
        raise FileNotFoundError(f"8kHz audio not found: {audio_8k}")

    script_path = config.kb_agent_dir / "scripts" / "diarize.py"
    if not script_path.exists():
        raise StepError(
            f"Diarize script not found: {script_path}",
            error_type=ErrorType.FAIL,
            step_name="diarize",
        )

    logs_conn = task.get("_logs_conn")
    task_id = task.get("_task_id", "")
    step_name = task.get("_step_name", "diarize")
    if logs_conn:
        from agent.streaming import log_progress
        log_progress(logs_conn, task_id, step_name,
                     f"Starting pyannote diarization for {slug}")

    result = subprocess.run(
        ["python3", str(script_path), str(audio_8k), str(output_json)],
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode != 0:
        if logs_conn:
            log_progress(logs_conn, task_id, step_name,
                         f"Diarization FAILED (rc={result.returncode})",
                         stream="stderr")
        stderr = result.stderr.strip()
        if "401" in stderr or "403" in stderr:
            raise StepError(
                "Pyannote model access denied — check HF_TOKEN",
                error_type=ErrorType.PAUSE,
                step_name="diarize",
                details=sanitize_visible_message(stderr[:500]),
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

    data = json.loads(output_json.read_text(encoding="utf-8"))
    write_transcription_sidecar_json(output_json, data, indent=2)

    if logs_conn:
        segment_count = str(len(data.get("segments", [])))
        log_progress(logs_conn, task_id, step_name,
                     f"Diarization complete: {segment_count} segments")

    return str(output_json)


# ── Transcription ────────────────────────────────────────────────────────────

def step_transcribe(task: dict, config: Config) -> str:
    """Transcribe audio via faster-whisper using the canonical script.

    MUST run after diarization (sequential — RAM safety).
    Returns the path to the transcript JSON file.
    """
    slug = _artifact_stem(task)
    audio_16k = Path(config.temp_dir) / f"{slug}_16k.wav"
    output_json = Path(config.temp_dir) / f"{slug}_transcript.json"

    if is_modal_backend(config):
        if not output_json.exists():
            if not audio_16k.exists():
                raise FileNotFoundError(f"16kHz audio not found: {audio_16k}")
            _, trans_modal = transcribe_modal(slug, audio_16k, config)
            shutil.copy2(trans_modal, output_json)
        logs_conn = task.get("_logs_conn")
        task_id = task.get("_task_id", "")
        step_name = task.get("_step_name", "transcribe")
        if logs_conn:
            from agent.streaming import log_progress
            log_progress(logs_conn, task_id, step_name,
                         f"Modal transcription ready for {slug}")
        return str(output_json)

    # Get duration from metadata (falls back to "0" when absent)
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    duration_secs = "0"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        duration_secs = str(meta.get("duration", 0))

    if not audio_16k.exists():
        raise FileNotFoundError(f"16kHz audio not found: {audio_16k}")

    script_path = config.kb_agent_dir / "scripts" / "transcribe.py"
    if not script_path.exists():
        raise StepError(
            f"Transcribe script not found: {script_path}",
            error_type=ErrorType.FAIL,
            step_name="transcribe",
        )

    logs_conn = task.get("_logs_conn")
    task_id = task.get("_task_id", "")
    step_name = task.get("_step_name", "transcribe")
    if logs_conn:
        from agent.streaming import log_progress
        dur_info = f" (~{float(duration_secs)/60:.0f}min)" if duration_secs != "0" else ""
        log_progress(logs_conn, task_id, step_name,
                     f"Starting faster-whisper transcription for {slug}{dur_info}")

    result = subprocess.run(
        ["python3", str(script_path), str(audio_16k), str(output_json), duration_secs],
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode != 0:
        if logs_conn:
            log_progress(logs_conn, task_id, step_name,
                         f"Transcription FAILED (rc={result.returncode})",
                         stream="stderr")
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

    data = json.loads(output_json.read_text(encoding="utf-8"))
    write_transcription_sidecar_json(output_json, data, indent=2)

    if logs_conn:
        seg_count = str(len(data.get("segments", [])))
        log_progress(logs_conn, task_id, step_name,
                     f"Transcription complete: {seg_count} segments")

    return str(output_json)


# ── Merge Diarization + Transcript ───────────────────────────────────────────

def merge_speaker_labels(diar: dict, trans: dict) -> dict:
    """Assign speaker labels to transcript segments at each segment midpoint.

    Mutates and returns ``trans``. Flags overlap segments whose composite
    speaker label contains ``|``. Raises StepError(FAIL) if diarization
    produced zero segments.
    """
    dia_segments = diar.get("segments", [])
    if not dia_segments:
        raise StepError(
            "Diarization produced zero segments",
            error_type=ErrorType.FAIL,
            step_name="merge",
        )

    for seg in trans.get("segments", []):
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2

        best_speaker = None
        for dia in dia_segments:
            if dia.get("start", 0) <= seg_mid <= dia.get("end", 0):
                speaker = dia.get("speaker", "Unknown")
                if "|" in speaker:
                    if best_speaker is None:
                        best_speaker = speaker
                else:
                    if best_speaker is None or "|" in best_speaker:
                        best_speaker = speaker

        seg["speaker"] = best_speaker if best_speaker else "Unknown"
        if "|" in str(seg.get("speaker", "")):
            seg["overlap"] = True

    return trans


# ── Transcript formatting ────────────────────────────────────────────────────

def format_transcript_for_llm(merged: dict, style: str = "clock") -> str:
    """Format a merged transcript as compact text for LLM consumption.

    style="clock": ``(MM:SS) SPEAKER [OVERLAP]: text`` (youtube / bunny).
    style="seconds": ``(SS.Ss) SPEAKER: text``, skipping empty text
    (instagram / threads).
    """
    lines = []
    for seg in merged.get("segments", []):
        start = seg.get("start", 0)
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip() if style == "seconds" else seg.get("text", "")
        if style == "seconds" and not text:
            continue

        if style == "seconds":
            ts = f"{start:.1f}s"
            overlap = ""
        else:
            h = int(start // 3600)
            m = int((start % 3600) // 60)
            s = int(start % 60)
            ts = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            overlap = " [OVERLAP]" if seg.get("overlap") else ""

        lines.append(f"({ts}) {speaker}{overlap}: {text}")

    return "\n".join(lines)


# ── Quality Gate ─────────────────────────────────────────────────────────────

def run_video_quality_gate(note_path: Path, config: Config, task_id: str,
                           max_rounds: int = MAX_QUALITY_RETRIES) -> tuple[bool, int]:
    """Run the video-style quality gate (concepts + key points) with LLM retry.

    Initializes the LLM client + task context and delegates to
    quality_gate_retry_loop. Returns ``(passed, rounds_used)``; callers decide
    whether to raise and whether to add the density check.
    """
    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task_id, logs_conn)

    return quality_gate_retry_loop(
        note_path, config, quality_gate_youtube, llm, max_rounds=max_rounds,
    )


def run_threads_video_quality_gate(note_path: Path, config: Config, task_id: str,
                                   max_rounds: int = MAX_QUALITY_RETRIES) -> tuple[bool, int]:
    """Run the Threads video quality gate (key points only) with LLM retry."""
    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task_id, logs_conn)

    return quality_gate_retry_loop(
        note_path, config, quality_gate_threads_video, llm, max_rounds=max_rounds,
    )


def check_density(note_path: Path, config: Config, task: dict | None = None) -> None:
    """Verify note density meets the minimum chars/min threshold."""
    meta_stem = _artifact_stem(task) if task is not None else note_path.stem
    meta_path = Path(config.temp_dir) / f"{meta_stem}_metadata.json"

    if not meta_path.exists():
        return  # Can't check density without metadata

    meta = json.loads(meta_path.read_text())
    duration_secs = meta.get("duration", 0)
    if duration_secs < 60:
        return  # Skip for very short videos

    duration_min = duration_secs / 60
    note_size = len(note_path.read_text(encoding="utf-8"))
    density = note_size / duration_min
    min_density = getattr(config, "youtube_min_density", VIDEO_DENSITY_MIN)

    if density < min_density:
        raise ValueError(
            f"Note density too low: {density:.0f} chars/min < "
            f"{min_density} required "
            f"({note_size} chars / {duration_min:.1f} min)"
        )


def strip_code_fence(content: str) -> str:
    """Normalize LLM note output into a valid Obsidian note.

    The synthesis LLM sometimes wraps the YAML frontmatter in a ```yaml code
    fence — and uses the fence *in place of* the opening ``---`` delimiter —
    which breaks the frontmatter parser + quality gate. Strip the fence and
    ensure BOTH the opening and closing ``---`` delimiters are present.
    """
    content = (content or "").strip()
    for prefix in ("```yaml", "```markdown", "```"):
        if content.startswith(prefix):
            content = content[len(prefix):].lstrip()
            break
    if content.endswith("```"):
        content = content[:-3].rstrip()
    content = content.strip()
    # the LLM often emits the fence instead of the opening --- ; restore it
    if not content.startswith("---"):
        content = "---\n" + content
    # ensure a closing --- before the first heading (frontmatter terminator)
    lines = content.split("\n")
    heading_idx = next((i for i, l in enumerate(lines) if l.startswith("#")), None)
    if heading_idx is not None and not any(l.strip() == "---" for l in lines[1:heading_idx]):
        while heading_idx > 1 and lines[heading_idx - 1].strip() == "":
            heading_idx -= 1
        lines.insert(heading_idx, "---")
        lines.insert(heading_idx + 1, "")
        content = "\n".join(lines)
    return content + "\n"


# ── object storage Upload ─────────────────────────────────────────────────────────────

def upload_video_artifacts(
    artifact_stem: str,
    storage_slug: str,
    config: Config,
    video_path: Path,
) -> list[str]:
    """Upload the video, an extracted MP3, and the merged transcript to object storage.

    ``artifact_stem`` names local temp artifacts; ``storage_slug`` keys object storage objects.

    Returns the list of uploaded object keys (relative to the bucket).
    """
    from agent.archival import upload_object

    bucket = config.storage_bucket
    uploaded = []

    ext = video_path.suffix.lower()
    if video_path.exists():
        upload_object(
            video_path, f"{bucket}/videos/{storage_slug}{ext}", config, timeout=120,
        )
        uploaded.append(f"videos/{storage_slug}{ext}")

    mp3_path = Path(config.temp_dir) / f"{artifact_stem}.mp3"
    if video_path.exists():
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and mp3_path.exists():
            upload_object(
                mp3_path, f"{bucket}/videos/{storage_slug}.mp3", config, timeout=120,
            )
            uploaded.append(f"videos/{storage_slug}.mp3")

    merged_path = Path(config.temp_dir) / f"{artifact_stem}_merged.json"
    if merged_path.exists():
        ensure_merged_transcript_file_redacted(merged_path)
        upload_object(
            merged_path, f"{bucket}/transcripts/{storage_slug}.json", config, timeout=120,
        )
        uploaded.append(f"transcripts/{storage_slug}.json")

    return uploaded


def upload_video_source(storage_slug: str, config: Config, video_path: Path,
                        dest_ext: str, note_path: Path, *,
                        artifact_stem: str) -> list[str]:
    """Upload a video source + merged transcript, then rewrite note source_files.

    ``artifact_stem`` names local temp artifacts; ``storage_slug`` keys object storage objects
    (see ``step_fetch_transcript`` for the same split).
    """
    from agent.archival import upload_object, write_source_files_block

    bucket = config.storage_bucket
    uploaded = []

    if video_path.exists():
        upload_object(
            video_path, f"{bucket}/videos/{storage_slug}.{dest_ext}", config, timeout=120,
        )
        uploaded.append(f"videos/{storage_slug}.{dest_ext}")

    merged = Path(config.temp_dir) / f"{artifact_stem}_merged.json"
    if merged.exists():
        ensure_merged_transcript_file_redacted(merged)
        upload_object(
            merged, f"{bucket}/transcripts/{storage_slug}.json", config, timeout=120,
        )
        uploaded.append(f"transcripts/{storage_slug}.json")

    write_source_files_block(note_path, uploaded, config)
    return uploaded


def upload_note_archive(slug: str, config: Config, note_path: Path, folder: str) -> str:
    """Upload a synthesized note for archival and link its object storage URL in frontmatter."""
    from agent.archival import upload_object

    bucket = config.storage_bucket
    note_key = f"{folder}/{slug}.md"
    storage_url = upload_object(note_path, f"{bucket}/{note_key}", config)

    content = note_path.read_text(encoding="utf-8")
    updated = re.sub(
        r"raw:\s*<filled by archival step>",
        f"raw: {storage_url}",
        content,
    )
    write_vault_note(note_path, updated)

    return storage_url


# ── Synthesize-only (batch) path ─────────────────────────────────────────────

def _synthesize_note_path(task: dict, config: Config) -> Path:
    slug = task.get("slug", task["id"][:8])
    ct = (task.get("content_type") or "").lower()
    if ct == "bunny_stream_synthesize":
        return config.vault_path / "Knowledge base" / f"{slug}.md"
    return config.vault_path / "Knowledge base" / "videos" / f"{slug}.md"


def _synthesize_archived_keys(task: dict) -> list[str]:
    storage_slug = _resolve_slug(task)
    return [
        f"videos/{storage_slug}.mp4",
        f"transcripts/{storage_slug}.json",
    ]


def step_rewrite_synthesize_source_files(task: dict, config: Config) -> str:
    """Rewrite note ``source_files`` from known storage keys (synthesize-only pipelines)."""
    from agent.archival import write_source_files_block

    keys = _synthesize_archived_keys(task)
    note_path = _synthesize_note_path(task, config)
    write_source_files_block(note_path, keys, config)
    return f"source_files: {', '.join(keys)}"


def _duration_from_merged(merged: dict) -> float:
    """Infer media duration (seconds) from a merged transcript JSON payload."""
    duration = merged.get("duration")
    if isinstance(duration, (int, float)):
        return float(duration)

    nested = merged.get("transcript")
    if isinstance(nested, dict):
        inner_duration = nested.get("duration")
        if isinstance(inner_duration, (int, float)):
            return float(inner_duration)
        segments = nested.get("segments") or []
    else:
        segments = merged.get("segments") or []

    max_end = 0.0
    for seg in segments:
        if isinstance(seg, dict):
            end = seg.get("end", 0) or 0
            try:
                max_end = max(max_end, float(end))
            except (TypeError, ValueError):
                continue
    return max_end


def step_fetch_transcript(task: dict, config: Config) -> str:
    """Fetch the pre-uploaded merged transcript from object storage (synthesize-only path).

    Reads ``{storage_bucket}/transcripts/{slug}.json`` and writes it to
    ``{temp_dir}/{slug}_merged.json`` so the content-type synthesize step can
    consume it exactly as if it had just merged locally.
    """
    stem = _artifact_stem(task)
    storage_slug = _resolve_slug(task)
    merged_path = Path(config.temp_dir) / f"{stem}_merged.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    alias = config.storage_alias
    bucket = config.storage_bucket
    dest_transcripts = f"{bucket}/transcripts/{storage_slug}.json"
    dest_legacy = f"{bucket}/reels/{storage_slug}.json"
    for dest in (dest_transcripts, dest_legacy):
        result = subprocess.run(
            ["mc", "cp", f"{alias}/{dest}", str(merged_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            break
    else:
        raise StepError(
            f"Transcript fetch failed: {result.stderr.strip()[:200]}",
            error_type=ErrorType.RETRY,
            step_name="fetch_transcript",
        )

    # Synthesize-only path skips extract_metadata: normalize batch segments
    # (`time` -> `start`/`end`) and write minimal metadata so downstream density
    # checks (e.g. YouTube check_density) can run.
    try:
        data = json.loads(merged_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise StepError(
            f"Transcript fetch produced invalid JSON: {exc}",
            error_type=ErrorType.RETRY,
            step_name="fetch_transcript",
        )
    segs = data.get("segments", [])
    for i, seg in enumerate(segs):
        if isinstance(seg, dict) and "time" in seg and "start" not in seg:
            seg["start"] = seg.pop("time")
        if isinstance(seg, dict) and "end" not in seg:
            nxt = segs[i + 1].get("start", segs[i + 1].get("time")) if i + 1 < len(segs) else None
            seg["end"] = nxt if nxt is not None else seg.get("start", 0) + 5.0
    write_merged_transcript_json(merged_path, data, indent=None)

    meta_path = Path(config.temp_dir) / f"{stem}_metadata.json"
    if not meta_path.exists():
        from agent.persist import persist_durable_json

        persist_durable_json(
            meta_path,
            {"duration": _duration_from_merged(data)},
            policy="boundary4",
        )

    return str(merged_path)
