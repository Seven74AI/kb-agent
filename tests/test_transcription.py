"""Tests for transcription backend config and Modal dispatch."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config, load_config
from agent.pipelines.youtube import step_diarize, step_transcribe
from agent.transcription import MODAL_RUN_TIMEOUT_SECS, is_modal_backend, transcribe_modal


@pytest.fixture
def task_dict():
    return {
        "id": "test-youtube-12345",
        "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "content_type": "youtube",
        "slug": "test_channel_test-video",
    }


@pytest.fixture
def base_config(tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=temp_dir,
    )


def test_config_defaults_cpu_backend():
    cfg = Config()
    assert cfg.transcription_backend == "cpu"
    assert cfg.modal_cli == "/root/modal/venv/bin/modal"
    assert is_modal_backend(cfg) is False


def test_load_config_transcription_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("temp_dir: /tmp\n")
    cfg = load_config(config_path=config_path, env_path=tmp_path / "missing.env")
    assert cfg.transcription_backend == "cpu"
    assert cfg.modal_cli == "/root/modal/venv/bin/modal"


def test_load_config_transcription_modal(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "transcription_backend: modal\nmodal_cli: /usr/local/bin/modal\n"
    )
    cfg = load_config(config_path=config_path, env_path=tmp_path / "missing.env")
    assert cfg.transcription_backend == "modal"
    assert cfg.modal_cli == "/usr/local/bin/modal"


def test_transcribe_modal_invokes_cli(base_config, tmp_path):
    slug = "my-video"
    wav = base_config.temp_dir / f"{slug}_16k.wav"
    wav.write_bytes(b"fake-wav")

    diar = tmp_path / "out" / f"{slug}.diarization.json"
    trans = tmp_path / "out" / f"{slug}.transcript.json"
    diar.parent.mkdir(parents=True)
    diar.write_text('{"segments": []}')
    trans.write_text('{"segments": []}')

    base_config.transcription_backend = "modal"
    base_config.modal_cli = "/fake/modal"

    def fake_run(cmd, **kwargs):
        assert kwargs["timeout"] == MODAL_RUN_TIMEOUT_SECS
        assert cmd[0] == "/fake/modal"
        assert cmd[1] == "run"
        assert str(base_config.kb_agent_dir / "modal" / "kb_transcribe.py") == cmd[2]
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        audio_dir = Path(cmd[cmd.index("--audio-dir") + 1])
        assert (audio_dir / f"{slug}.wav").exists()
        (out_dir / f"{slug}.diarization.json").write_text('{"segments": []}')
        (out_dir / f"{slug}.transcript.json").write_text('{"segments": []}')
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("agent.transcription.subprocess.run", side_effect=fake_run):
        diar_path, trans_path = transcribe_modal(slug, wav, base_config)

    assert diar_path.name == f"{slug}.diarization.json"
    assert trans_path.name == f"{slug}.transcript.json"


def test_step_diarize_modal_backend(base_config, task_dict):
    slug = task_dict["slug"]
    wav_16k = base_config.temp_dir / f"{slug}_16k.wav"
    wav_16k.write_bytes(b"wav")
    base_config.transcription_backend = "modal"

    diar_out = base_config.temp_dir / f"{slug}_diarization.json"
    trans_out = base_config.temp_dir / f"{slug}_transcript.json"

    with patch("agent.pipelines.video_common.transcribe_modal") as mock_modal:
        mock_modal.return_value = (
            Path("/modal/out/diar.json"),
            Path("/modal/out/trans.json"),
        )
        with patch("agent.pipelines.video_common.install_modal_outputs") as mock_install:
            mock_install.return_value = (diar_out, trans_out)
            out = step_diarize(task_dict, base_config)

    mock_modal.assert_called_once_with(slug, wav_16k, base_config)
    mock_install.assert_called_once()
    assert out == str(diar_out)


def test_step_diarize_cpu_uses_pyannote(base_config, task_dict):
    slug = task_dict["slug"]
    (base_config.temp_dir / f"{slug}_8k.wav").write_bytes(b"8k")
    output_path = base_config.temp_dir / f"{slug}_diarization.json"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        output_path.write_text('{"segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}]}')
        out = step_diarize(task_dict, base_config)

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "diarize.py" in cmd[1]
    assert out == str(output_path)


TOKENIZED = "https://cdn.example.com/spoken.mp3?token=secret&expires=999"


def test_cpu_transcription_scripts_wired_to_sidecar_writer():
    for name in ("diarize.py", "transcribe.py"):
        text = (Path(__file__).resolve().parents[1] / "scripts" / name).read_text(
            encoding="utf-8",
        )
        assert "write_transcription_sidecar_json" in text
        assert "json.dump(" not in text


def test_step_diarize_cpu_rescrubs_sidecar_after_subprocess(base_config, task_dict):
    slug = task_dict["slug"]
    (base_config.temp_dir / f"{slug}_8k.wav").write_bytes(b"8k")
    output_path = base_config.temp_dir / f"{slug}_diarization.json"
    leaky = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "text": f"api_key=spokenleak see {TOKENIZED}",
            },
        ],
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        output_path.write_text(json.dumps(leaky))
        step_diarize(task_dict, base_config)

    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    text = parsed["segments"][0]["text"]
    assert "spokenleak" not in text
    assert "token=secret" not in text
    assert "[credential redacted]" in text


def test_step_transcribe_cpu_rescrubs_sidecar_after_subprocess(base_config, task_dict):
    slug = task_dict["slug"]
    (base_config.temp_dir / f"{slug}_16k.wav").write_bytes(b"16k")
    (base_config.temp_dir / f"{slug}_metadata.json").write_text(
        json.dumps({"duration": 60}),
    )
    output_path = base_config.temp_dir / f"{slug}_transcript.json"
    leaky = {
        "language": "en",
        "duration": 60.0,
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": f"sessionid=spokencookie {TOKENIZED}",
            },
        ],
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        output_path.write_text(json.dumps(leaky))
        step_transcribe(task_dict, base_config)

    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    text = parsed["segments"][0]["text"]
    assert "spokencookie" not in text
    assert "token=secret" not in text
    assert "sessionid=[REDACTED]" in text


def test_step_transcribe_modal_skips_cpu_script(base_config, task_dict):
    slug = task_dict["slug"]
    trans_out = base_config.temp_dir / f"{slug}_transcript.json"
    trans_out.write_text('{"segments": []}')
    base_config.transcription_backend = "modal"

    with patch("subprocess.run") as mock_run:
        out = step_transcribe(task_dict, base_config)

    mock_run.assert_not_called()
    assert out == str(trans_out)
