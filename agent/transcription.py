"""Shared transcription backends — CPU (local scripts) and Modal GPU."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Union

from agent.config import Config
from agent.exceptions import ErrorType, StepError

MODAL_RUN_TIMEOUT_SECS = 3600


def merge_speakers_from_paths(
    diar_path: Union[str, Path],
    trans_path: Union[str, Path],
) -> dict:
    """Merge diarization + transcript JSON into batch-style ``{segments: [...]}``."""
    with open(diar_path, encoding="utf-8") as df:
        d = json.load(df)["segments"]
    with open(trans_path, encoding="utf-8") as tf:
        t = json.load(tf)["segments"]
    out = []
    for seg in t:
        best, spk = -1, "SPEAKER_00"
        for s in d:
            ov = min(seg["end"], s["end"]) - max(seg["start"], s["start"])
            if ov > best:
                best, spk = ov, s["speaker"]
        out.append(
            {
                "time": round(seg["start"], 1),
                "speaker": spk,
                "text": seg["text"],
            },
        )
    return {"segments": out}


def is_modal_backend(config: Config) -> bool:
    return config.transcription_backend == "modal"


def transcribe_modal(slug: str, wav_16k_path: Path | str, config: Config) -> tuple[Path, Path]:
    """Run Modal kb_transcribe for one 16 kHz WAV.

    Places ``<slug>.wav`` in a per-task audio dir, invokes the Modal CLI, and
    returns paths to ``<slug>.diarization.json`` and ``<slug>.transcript.json``.

    Raises:
        StepError: On CLI failure or missing output files.
    """
    wav_16k_path = Path(wav_16k_path)
    if not wav_16k_path.exists():
        raise FileNotFoundError(f"16kHz audio not found: {wav_16k_path}")

    work_dir = Path(config.temp_dir) / f"modal_{slug}_{uuid.uuid4().hex[:8]}"
    wav_dir = work_dir / "wavs"
    out_dir = work_dir / "out"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_wav = wav_dir / f"{slug}.wav"
    shutil.copy2(wav_16k_path, dest_wav)

    modal_script = config.kb_agent_dir / "modal" / "kb_transcribe.py"
    cmd = [
        config.modal_cli,
        "run",
        str(modal_script),
        "--audio-dir",
        str(wav_dir),
        "--out-dir",
        str(out_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=MODAL_RUN_TIMEOUT_SECS,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise StepError(
            f"Modal transcription failed: {stderr[:500]}",
            error_type=ErrorType.RETRY,
            step_name="transcribe_modal",
        )

    diar_path = out_dir / f"{slug}.diarization.json"
    trans_path = out_dir / f"{slug}.transcript.json"
    if not diar_path.exists() or not trans_path.exists():
        raise StepError(
            f"Modal run succeeded but outputs missing under {out_dir}",
            error_type=ErrorType.RETRY,
            step_name="transcribe_modal",
        )

    return diar_path, trans_path


def install_modal_outputs(
    slug: str,
    config: Config,
    diar_modal: Path,
    trans_modal: Path,
) -> tuple[Path, Path]:
    """Copy Modal JSON outputs to the canonical temp filenames merge expects."""
    diar_out = Path(config.temp_dir) / f"{slug}_diarization.json"
    trans_out = Path(config.temp_dir) / f"{slug}_transcript.json"
    shutil.copy2(diar_modal, diar_out)
    shutil.copy2(trans_modal, trans_out)
    return diar_out, trans_out
