"""Tests for batch_transcribe ffmpeg timeout handling."""

import subprocess
import sys
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


def test_batch_transcribe_scrubs_ffmpeg_failure(capsys, tmp_path, monkeypatch):
    stage = tmp_path / "in"
    stage.mkdir()
    (stage / "clip.mp4").write_bytes(b"\x00")
    out = tmp_path / "out"

    def fake_extract(_video_path, _wav_path):
        return subprocess.CompletedProcess(
            ["ffmpeg"],
            1,
            "",
            "Error opening /root/kb-agent/cookies/yt.txt token=ffmpegbad",
        )

    monkeypatch.setattr(bt, "extract_wav", fake_extract)
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["batch_transcribe.py", str(stage), "--out-dir", str(out)])
        rc = bt.main()

    assert rc == 1
    logged = capsys.readouterr().out
    assert "/root/kb-agent" not in logged
    assert "ffmpegbad" not in logged
    assert "[path]" in logged
    assert "[credential redacted]" in logged


def test_batch_transcribe_scrubs_modal_child_tail(capsys, tmp_path, monkeypatch):
    stage = tmp_path / "in"
    stage.mkdir()
    (stage / "ready.wav").write_bytes(b"RIFF")
    out = tmp_path / "out"
    sensitive = "modal tail /home/ubuntu/modal token=modalbad benign"

    def fake_sh(cmd, **kw):
        if cmd[0].endswith("modal"):
            return subprocess.CompletedProcess(cmd, 0, sensitive, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(bt, "sh", fake_sh)
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["batch_transcribe.py", str(stage), "--out-dir", str(out)])
        rc = bt.main()

    assert rc == 0
    logged = capsys.readouterr().out
    assert "/home/ubuntu" not in logged
    assert "modalbad" not in logged
    assert "[path]" in logged
    assert "[credential redacted]" in logged
    assert "benign" in logged
