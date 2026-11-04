"""Tests for batch_transcribe ffmpeg timeout handling."""

import subprocess
from unittest.mock import patch

import scripts.batch_transcribe as bt


def test_ffmpeg_timeout_constant_defined():
    assert isinstance(bt.FFMPEG_TIMEOUT, int)
    assert bt.FFMPEG_TIMEOUT > 0


def test_extract_wav_passes_ffmpeg_timeout(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    wav = tmp_path / "clip.wav"

    with patch("scripts.batch_transcribe.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(["ffmpeg"], 0, "", "")
        bt.extract_wav(str(video), str(wav))
        assert run.call_args.kwargs.get("timeout") == bt.FFMPEG_TIMEOUT
