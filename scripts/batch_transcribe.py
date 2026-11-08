#!/usr/bin/env python3
"""General batch transcription via Modal — any language, one cold-start.

Usage:
    batch_transcribe.py <input_dir> [--out-dir DIR] [--language fr] [--hotwords "..."] [--initial-prompt "..."]

<input_dir> may contain videos (.mp4/.webm/.mkv/.mov) or already-extracted .wav files
(16 kHz mono). Extracts WAVs, runs ONE Modal batch (auto-detect language unless
--language is given), then merges diarization + transcription into a
speaker-attributed <name>.json per input.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Isolate from the Hermes runtime's py3.14 PYTHONPATH (cross-venv cffi mismatch
# in the modal/yt-dlp subprocesses).
os.environ.pop("PYTHONPATH", None)

_bootstrap_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bootstrap_root))
sys.path.insert(0, "/root/kb-agent")
from agent.notify import scrub_outbound_visible_message  # noqa: E402
from agent.redact import write_merged_transcript_json  # noqa: E402
from agent.transcription import merge_speakers_from_paths  # noqa: E402

MODAL = "/root/modal"
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")
# Per-input ceiling so one stuck ffmpeg does not rely on batch_ingest's 7200s parent timeout.
FFMPEG_TIMEOUT = 900


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _child_captured_output(result: subprocess.CompletedProcess[str]) -> str:
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    return stdout or stderr


def _print_scrubbed_child_tail(
    result: subprocess.CompletedProcess[str],
    *,
    tail_chars: int = 1500,
) -> None:
    raw = _child_captured_output(result)
    if not raw:
        return
    print(scrub_outbound_visible_message(raw[-tail_chars:]))


def extract_wav(video_path: str, wav_path: str) -> subprocess.CompletedProcess:
    """Extract 16 kHz mono WAV from a video; fails explicitly on timeout."""
    return sh(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-ac", "1", "-ar", "16000", wav_path,
        ],
        timeout=FFMPEG_TIMEOUT,
    )


def main():
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("usage: batch_transcribe.py <input_dir> [--out-dir DIR] [--language fr] [--hotwords ...] [--initial-prompt ...]")
        return 1
    in_dir = args[0]
    out_dir = "."
    lang = hot = prompt = None
    i = 1
    while i < len(args):
        if args[i] == "--out-dir" and i + 1 < len(args):
            out_dir = args[i + 1]; i += 2
        elif args[i] == "--language" and i + 1 < len(args):
            lang = args[i + 1]; i += 2
        elif args[i] == "--hotwords" and i + 1 < len(args):
            hot = args[i + 1]; i += 2
        elif args[i] == "--initial-prompt" and i + 1 < len(args):
            prompt = args[i + 1]; i += 2
        else:
            i += 1
    os.makedirs(out_dir, exist_ok=True)

    wav_dir = os.path.join(out_dir, ".wavs")
    os.makedirs(wav_dir, exist_ok=True)
    names = []
    for f in sorted(os.listdir(in_dir)):
        p = os.path.join(in_dir, f)
        base = os.path.splitext(f)[0]
        if f.lower().endswith(VIDEO_EXTS):
            wav = os.path.join(wav_dir, base + ".wav")
            try:
                r = extract_wav(p, wav)
            except subprocess.TimeoutExpired:
                print(f"[error] {f}: ffmpeg timed out after {FFMPEG_TIMEOUT}s")
                continue
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                detail = scrub_outbound_visible_message(err) if err else str(r.returncode)
                print(f"[error] {f}: ffmpeg failed ({detail})")
                continue
            names.append(base)
        elif f.lower().endswith(".wav"):
            shutil.copy(p, os.path.join(wav_dir, f))
            names.append(base)
    if not names:
        print("no videos or .wav files found in", scrub_outbound_visible_message(in_dir))
        return 1
    print(f"batch: {len(names)} input(s) -> one Modal run")

    cmd = [os.path.join(MODAL, "venv", "bin", "modal"), "run", "kb_transcribe.py",
           "--audio-dir", wav_dir, "--out-dir", out_dir]
    if lang:
        cmd += ["--language", lang]
    if hot:
        cmd += ["--hotwords", hot]
    if prompt:
        cmd += ["--initial-prompt", prompt]
    r = sh(cmd, cwd=MODAL, timeout=3600)
    _print_scrubbed_child_tail(r)
    if r.returncode != 0:
        print(f"[error] Modal run failed ({r.returncode}); aborting")
        return r.returncode

    made = []
    for n in names:
        diar = os.path.join(out_dir, f"{n}.diarization.json")
        trans = os.path.join(out_dir, f"{n}.transcript.json")
        if not (os.path.exists(diar) and os.path.exists(trans)):
            print(f"[warn] {n}: missing Modal output")
            continue
        mpath = Path(out_dir) / f"{n}.json"
        merged = merge_speakers_from_paths(diar, trans)
        write_merged_transcript_json(mpath, merged, indent=None)
        made.append(n)
    print("MERGED: " + ", ".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
