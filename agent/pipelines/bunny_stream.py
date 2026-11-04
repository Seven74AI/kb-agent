"""Bunny Stream pipeline — terragrandir / TDAH summit conference videos.

Download from Bunny CDN (Referer required) → dual WAV → diarize → transcribe →
merge → French TDAH note synthesis → quality gates → MinIO → Git push.
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from agent.archival import append_note_tags
from agent.config import Config
from agent.db import LOGS_SCHEMA, init_db
from agent.exceptions import ErrorType, StepError
from agent.llm import get_llm_client
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.youtube import (
    step_cleanup_temp,
    step_diarize,
    step_merge_and_identify,
    step_transcribe,
    _format_transcript_for_llm,
    MAX_QUALITY_RETRIES,
    YOUTUBE_DENSITY_MIN,
)
from agent.pipelines.video_common import step_fetch_transcript
from agent.quality import quality_gate_retry_loop, quality_gate_youtube

# ── Constants ─────────────────────────────────────────────────────────────────

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BUNNY_REFERER = "Referer: https://player.mediadelivery.net/"

BUNNY_UUID_RE = re.compile(
    r"player\.mediadelivery\.net/embed/\d+/([a-f0-9]{8}-[a-f0-9]{4}-"
    r"[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    re.IGNORECASE,
)

BUNNY_PLAYLIST_RE = re.compile(
    r"https://vz-[a-z0-9-]+\.b-cdn\.net/([a-f0-9-]+)/playlist\.m3u8",
    re.IGNORECASE,
)

BUNNY_CDN_BASE_RE = re.compile(
    r"(https://vz-[a-z0-9-]+\.b-cdn\.net)/[a-f0-9-]+/playlist\.m3u8",
    re.IGNORECASE,
)

# Vault notes for TDAH talks live at Knowledge base/<slug>.md (not videos/)
VAULT_NOTE_SUBFOLDER = ""


def extract_bunny_video_uuid(text: str) -> str | None:
    """Extract Bunny video UUID from page HTML or embed URL."""
    m = BUNNY_UUID_RE.search(text)
    return m.group(1).lower() if m else None


def _curl_page(url: str, timeout: int = 40) -> str:
    result = subprocess.run(
        [
            "curl", "-s", "--max-time", str(timeout),
            "-A", BROWSER_UA,
            url,
        ],
        capture_output=True, text=True, timeout=timeout + 10,
    )
    if result.returncode != 0:
        raise StepError(
            f"Failed to fetch page: {url} ({result.stderr.strip()[:200]})",
            error_type=ErrorType.RETRY,
            step_name="download_video",
        )
    return result.stdout


def _resolve_embed_url(source_url: str) -> tuple[str, str]:
    """Return (embed_page_url, video_uuid) for a terragrandir or player URL."""
    if "player.mediadelivery.net" in source_url:
        uuid = extract_bunny_video_uuid(source_url)
        if not uuid:
            raise StepError(
                "Could not parse Bunny video UUID from player URL",
                error_type=ErrorType.FAIL,
                step_name="download_video",
            )
        return source_url, uuid

    html = _curl_page(source_url)
    uuid = extract_bunny_video_uuid(html)
    if not uuid:
        raise StepError(
            "No Bunny Stream embed found on page",
            error_type=ErrorType.FAIL,
            step_name="download_video",
        )

    embed_match = re.search(
        r"https://player\.mediadelivery\.net/embed/\d+/[a-f0-9-]{36}",
        html,
        re.IGNORECASE,
    )
    embed_url = embed_match.group(0) if embed_match else (
        f"https://player.mediadelivery.net/embed/0/{uuid}"
    )
    return embed_url, uuid


def _resolve_cdn_base(embed_url: str, video_uuid: str) -> str:
    """Discover vz-*.b-cdn.net base URL from the embed page."""
    html = _curl_page(embed_url)
    m = BUNNY_CDN_BASE_RE.search(html)
    if m:
        return m.group(1)
    # Fallback CDN zone used by terragrandir TDAH summit (tdah_daily_check.py)
    return "https://vz-46d29c85-edb.b-cdn.net"


def step_extract_metadata(task: dict, config: Config) -> str:
    """Extract basic metadata from the terragrandir detail page."""
    slug = task.get("slug", task["id"][:8])
    source_url = task["source_url"]

    title = slug.replace("-", " ").title()
    if "terragrandir.com" in source_url:
        html = _curl_page(source_url)
        og = re.search(
            r'<meta\s+property="og:title"\s+content="([^"]+)"',
            html, re.IGNORECASE,
        )
        if og:
            title = og.group(1).strip()
        elif "<h1" in html:
            h1 = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
            if h1:
                title = h1.group(1).strip()

    metadata = {
        "title": title,
        "uploader": "Terr'à Grandir — Sommet TDAH Adulte",
        "duration": 0.0,
        "description": "",
        "chapters": [],
        "source_url": source_url,
    }
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return json.dumps(metadata)


def step_download_video(task: dict, config: Config) -> str:
    """Download Bunny Stream video (MP4 preferred, HLS fallback)."""
    slug = task.get("slug", task["id"][:8])
    source_url = task["source_url"]
    output_path = Path(config.temp_dir) / f"{slug}.mp4"

    embed_url, video_uuid = _resolve_embed_url(source_url)
    cdn_base = _resolve_cdn_base(embed_url, video_uuid)

    mp4_url = f"{cdn_base}/{video_uuid}/play_720p.mp4"
    dl = subprocess.run(
        [
            "curl", "-sL", "--max-time", "600",
            "-H", BUNNY_REFERER,
            "-o", str(output_path),
            mp4_url,
        ],
        capture_output=True, text=True, timeout=620,
    )

    if dl.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 100_000:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        hls_url = f"{cdn_base}/{video_uuid}/720p/video.m3u8"
        ff = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-headers", BUNNY_REFERER,
                "-i", hls_url,
                "-c", "copy", "-bsf:a", "aac_adtstoasc",
                str(output_path),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if ff.returncode != 0 or not output_path.exists():
            raise StepError(
                f"Bunny video download failed: {ff.stderr.strip()[:200]}",
                error_type=ErrorType.RETRY,
                step_name="download_video",
            )

    # Probe duration for transcription + density gate
    probe = subprocess.run(
        [
            "ffprobe", "-hide_banner", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(output_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    duration = 0.0
    if probe.returncode == 0 and probe.stdout.strip():
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            pass

    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"title": slug, "duration": duration}
    meta["duration"] = duration
    meta["bunny_video_uuid"] = video_uuid
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return str(output_path)


def step_extract_audio(task: dict, config: Config) -> str:
    """Extract dual audio: 16kHz WAV (whisper) + 8kHz WAV (pyannote)."""
    slug = task.get("slug", task["id"][:8])
    video_path = Path(config.temp_dir) / f"{slug}.mp4"
    wav_16k_path = Path(config.temp_dir) / f"{slug}_16k.wav"
    wav_8k_path = Path(config.temp_dir) / f"{slug}_8k.wav"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

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

    return json.dumps({"wav_16k": str(wav_16k_path), "wav_8k": str(wav_8k_path)})


def _note_path(config: Config, slug: str) -> Path:
    base = config.vault_path / "Knowledge base"
    if VAULT_NOTE_SUBFOLDER:
        return base / VAULT_NOTE_SUBFOLDER / f"{slug}.md"
    return base / f"{slug}.md"


def _build_synthesis_prompt(source_url: str, slug: str, config: Config) -> str:
    """Build system prompt for French TDAH summit notes."""
    template_path = config.kb_agent_dir / "templates" / "tdah-summit-note-template.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

    date_str = datetime.now().strftime("%Y-%m-%d")
    tailscale_base = "http://vmi3304846.tail5c02a1.ts.net:9000"

    return f"""Tu es un curateur de base de connaissances spécialisé dans les conférences TDAH.
La vidéo source est en FRANÇAIS. Tu DOIS rédiger TOUT le contenu ET TOUS les titres de sections en FRANÇAIS.

SOURCE URL: {source_url}

MODÈLE (structure exacte, ordre canonique des sections) :
{template}

RÈGLES DE QUALITÉ (non négociables) :
- Concepts clés : 3 à 10 concepts transversaux avec mécanismes expliqués (pas seulement conclusions)
- Points clés : 8 à 15 points actionnables au format « [Affirmation] — [Mécanisme/Preuve] »
- Ancrer chaque affirmation dans la transcription — pas d'hallucination
- Section Transcription : inclure la transcription dialogue complète en annexe (format du modèle)
- Tags : inclure sommet-tdah-2026 et tdah

FORMAT DE SORTIE :
- Commencer par le frontmatter YAML :
  date: 2026
  created: {date_str} 00:00:00
  modified: {date_str} 00:00:00
  source: "<Intervenant> — <Titre> (Sommet TDAH Adulte — Terr'à Grandir)"
  source_url: {source_url}
  source_files:
    video: {tailscale_base}/knowledge-base/videos/{slug}.mp4
    transcript: {tailscale_base}/knowledge-base/transcripts/{slug}.json
  confidence: plausible
  tags: [tdah, adhd, conference, sommet-tdah-2026]
- Puis les sections dans l'ordre canonique du modèle (Métadonnées → … → Voir aussi).
- Laisser la section Voir aussi vide (remplie par une étape ultérieure).

NE TRADUIS JAMAIS vers l'anglais. Contenu et libellés de sections en français uniquement."""


def step_synthesize_note(task: dict, config: Config) -> str:
    """LLM synthesis for Bunny Stream / TDAH summit talks (French note)."""
    slug = task.get("slug", task["id"][:8])
    source_url = task.get("source_url", "")

    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged transcript not found: {merged_path}")

    merged = json.loads(merged_path.read_text())
    transcript_text = _format_transcript_for_llm(merged)

    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

    system_prompt = _build_synthesis_prompt(source_url, slug, config)
    user_prompt = (
        "Voici la transcription complète (locuteurs et timestamps). "
        "Lis-la intégralement, puis rédige la note de base de connaissances "
        "en suivant le modèle et les règles du prompt système.\n\n"
        f"Titre suggéré : {metadata.get('title', slug)}\n\n"
        f"=== TRANSCRIPTION ===\n{transcript_text}\n=== FIN ==="
    )

    max_transcript = 60_000
    if len(user_prompt) > max_transcript + 500:
        half = max_transcript // 2
        transcript_text = (
            transcript_text[:half]
            + f"\n\n... [{len(transcript_text) - max_transcript} caractères omis] ...\n\n"
            + transcript_text[-half:]
        )
        user_prompt = (
            "Voici la transcription (tronquée au centre si très longue). "
            "Rédige la note complète en français.\n\n"
            f"=== TRANSCRIPTION ===\n{transcript_text}\n=== FIN ==="
        )

    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        role="synthesis",
    )

    note_path = _note_path(config, slug)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(response.content, encoding="utf-8")
    append_note_tags(note_path, task.get("tags"))
    return str(note_path)


def step_quality_gate(task: dict, config: Config) -> str:
    """Run video-style quality gates (concepts + key points) on the French note."""
    slug = task.get("slug", task["id"][:8])
    note_path = _note_path(config, slug)

    if not note_path.exists():
        raise FileNotFoundError(f"Note not found for quality gate: {note_path}")

    logs_conn = init_db(config.kb_agent_dir / "data" / "logs.db", LOGS_SCHEMA)
    llm = get_llm_client(config)
    llm.set_task_context(task["id"], logs_conn)

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

    _check_density(note_path, config)
    return "passed"


def _check_density(note_path: Path, config: Config) -> None:
    slug = note_path.stem
    meta_path = Path(config.temp_dir) / f"{slug}_metadata.json"
    if not meta_path.exists():
        return

    meta = json.loads(meta_path.read_text())
    duration_secs = meta.get("duration", 0)
    if duration_secs < 60:
        return

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


def step_minio_upload(task: dict, config: Config) -> str:
    """Upload MP4, MP3, and merged transcript JSON to MinIO."""
    from agent.archival import upload_file_to_minio

    slug = task.get("slug", task["id"][:8])
    bucket = config.minio_bucket
    uploaded = []

    video_path = Path(config.temp_dir) / f"{slug}.mp4"
    if video_path.exists():
        dest = f"{bucket}/videos/{slug}.mp4"
        upload_file_to_minio(video_path, dest, timeout=120)
        uploaded.append(f"videos/{slug}.mp4")

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

    merged_path = Path(config.temp_dir) / f"{slug}_merged.json"
    if merged_path.exists():
        dest = f"{bucket}/transcripts/{slug}.json"
        upload_file_to_minio(merged_path, dest, timeout=120)
        uploaded.append(f"transcripts/{slug}.json")

    _update_note_source_files(slug, config, uploaded)
    return f"Uploaded: {', '.join(uploaded)}"


def _update_note_source_files(slug: str, config: Config, uploaded: list) -> None:
    note_path = _note_path(config, slug)
    if not note_path.exists():
        return

    tailscale_base = "http://vmi3304846.tail5c02a1.ts.net:9000"
    content = note_path.read_text(encoding="utf-8")

    lines = ["source_files:"]
    for u in uploaded:
        key = Path(u).suffix.lstrip(".")
        if u.startswith("transcripts/"):
            lines.append(f"  transcript: {tailscale_base}/{config.minio_bucket}/{u}")
        elif u.endswith(".mp4"):
            lines.append(f"  video: {tailscale_base}/{config.minio_bucket}/{u}")
        elif u.endswith(".mp3"):
            lines.append(f"  audio: {tailscale_base}/{config.minio_bucket}/{u}")
        else:
            lines.append(f"  {key}: {tailscale_base}/{config.minio_bucket}/{u}")

    source_files_block = "\n".join(lines)
    if "source_files:" in content:
        content = re.sub(
            r"source_files:.*?(?=\n\w|\Z)",
            source_files_block,
            content,
            flags=re.DOTALL,
        )
        note_path.write_text(content, encoding="utf-8")


def step_git_push(task: dict, config: Config) -> str:
    """Push the TDAH note at Knowledge base/<slug>.md."""
    from agent.archival import step_git_push as archival_git_push

    return archival_git_push(task, config, vault_subfolder=VAULT_NOTE_SUBFOLDER)


# ── Pipeline Registration ────────────────────────────────────────────────────

register_pipeline("bunny_stream", [
    Step(name="extract_metadata", fn=step_extract_metadata, on_error="retry"),
    Step(name="download_video", fn=step_download_video, on_error="retry"),
    Step(name="extract_audio", fn=step_extract_audio, on_error="retry"),
    Step(name="diarize", fn=step_diarize, on_error="retry"),
    Step(name="transcribe", fn=step_transcribe, on_error="retry"),
    Step(name="merge_and_identify", fn=step_merge_and_identify, on_error="retry"),
    Step(name="cleanup_temp", fn=step_cleanup_temp, on_error="fail"),
    Step(name="synthesize_note", fn=step_synthesize_note, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])

register_pipeline("bunny_stream_synthesize", [
    Step(name="fetch_transcript", fn=step_fetch_transcript, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_note, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
