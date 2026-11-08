"""Regression tests for medium/low deferred-media ingest invariants."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.core.submit import plan_submission
from agent.detection import _detect_threads_subtype, _fetch_content
from agent.exceptions import ErrorType, StepError
from agent.pipelines.instagram import step_ig_carousel_extract
from agent.pipelines.video_common import _artifact_stem, step_fetch_transcript
from agent.pipelines.youtube import _ytdlp_subprocess_env, step_cleanup_temp


@pytest.fixture
def core_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Knowledge base" / "articles").mkdir(parents=True)
    return Config(kb_agent_dir=tmp_path, vault_path=vault, cookie_dir=tmp_path / "cookies")


@pytest.fixture
def config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    temp = tmp_path / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    cookies = tmp_path / "cookies"
    cookies.mkdir()
    return Config(
        kb_agent_dir=tmp_path,
        vault_path=vault,
        temp_dir=temp,
        cookie_dir=cookies,
        minio_bucket="kb-agent-archive",
    )


def test_detect_threads_subtype_defaults_text_on_fetch_failure(monkeypatch, caplog):
    def boom(cmd, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    caplog.set_level("WARNING")
    assert _detect_threads_subtype("https://threads.net/@u/post/1") == "threads_text"


def test_plan_submission_threads_fetch_failure_paused_not_dropped(core_config, monkeypatch, tmp_path):
    cookie_dir = core_config.cookie_dir
    cookie_dir.mkdir(parents=True, exist_ok=True)
    (cookie_dir / "threads_cookies.txt").write_text("x", encoding="utf-8")

    def boom(cmd, **kwargs):
        raise OSError("down")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    monkeypatch.setattr("agent.core.submit.preflight_health", lambda p: (True, ""))
    monkeypatch.setattr("agent.core.submit.dedup_check", lambda url, path: False)

    plan = plan_submission("https://threads.net/@user/post/abc", core_config)
    assert plan["status"] == "paused"
    assert plan["content_type"] == "threads_text"
    assert "Threads subtype fetch failed" in plan["reason"]


def test_fetch_content_unlinks_curl_sidecar(monkeypatch, tmp_path):
    custom_dir = tmp_path / "cookies"
    custom_dir.mkdir()
    cookie = custom_dir / "threads_cookies.txt"
    cookie.write_text("session=1\n", encoding="utf-8")
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=custom_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", fake_run)
    _fetch_content("https://threads.net/@u/post/1", "threads")
    conf_path = custom_dir / ".curl-threads_cookies.txt.conf"
    assert not conf_path.exists()


def test_step_fetch_transcript_uses_artifact_stem_for_batch(config):
    task_id = "batch-ingest-uuid"
    task = {"id": task_id, "slug": "chan/vid", "_batch_ingest": True}
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

    def fake_mc(cmd, **kwargs):
        dest = Path(cmd[-1])
        dest.write_text(json.dumps(payload), encoding="utf-8")
        return MagicMock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=fake_mc):
        step_fetch_transcript(task, config)
    assert (config.temp_dir / f"{task_id}_merged.json").is_file()
    assert (config.temp_dir / f"{task_id}_metadata.json").is_file()


def test_step_cleanup_temp_removes_ytdlp_xdg(config):
    task = {"id": "t1", "slug": "sluggy"}
    cookie_dir = config.cookie_dir
    cookie_dir.mkdir(parents=True, exist_ok=True)
    (cookie_dir / "yt_cookies.txt").write_text("# c\n", encoding="utf-8")
    _ytdlp_subprocess_env(config)
    xdg_root = Path(config.temp_dir) / "ytdlp_xdg"
    assert xdg_root.is_dir()
    (config.temp_dir / f"{_artifact_stem(task)}_8k.wav").write_text("x", encoding="utf-8")
    step_cleanup_temp(task, config)
    assert not xdg_root.exists()


def test_diarize_steperror_details_sanitized(config, monkeypatch, tmp_path):
    from agent.pipelines import video_common as vc

    slug = "vid_slug"
    task = {"id": "t1", "slug": slug}
    (config.temp_dir / f"{slug}_8k.wav").write_text("x", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "diarize.py").write_text("# stub\n", encoding="utf-8")
    config.kb_agent_dir = tmp_path

    monkeypatch.setattr(vc, "is_modal_backend", lambda cfg: False)
    secret = "HTTP Error 403: Forbidden /secret/sessionid=deadbeef"

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=secret)

    monkeypatch.setattr(vc.subprocess, "run", fake_run)

    with pytest.raises(StepError) as excinfo:
        vc.step_diarize(task, config)
    assert excinfo.value.error_type == ErrorType.PAUSE
    assert excinfo.value.details is not None
    assert "/secret/" not in (excinfo.value.details or "")
    assert len(excinfo.value.details) <= 500


def test_carousel_extract_sanitizes_subprocess_details(config, monkeypatch):
    task = {
        "id": "t1",
        "slug": "ig_carousel",
        "source_url": "https://www.instagram.com/p/ABC/",
    }
    secret = "sessionid=supersecret" + "z" * 3000

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=secret)

    monkeypatch.setattr("agent.pipelines.instagram.subprocess.run", fake_run)
    with pytest.raises(StepError) as excinfo:
        step_ig_carousel_extract(task, config)
    assert excinfo.value.error_type == ErrorType.RETRY
    assert excinfo.value.details is not None
    assert len(excinfo.value.details) <= 500


def test_kb_ocr_registers_heif_and_avif_openers():
    text = (Path(__file__).resolve().parents[1] / "modal" / "kb_ocr.py").read_text(
        encoding="utf-8"
    )
    assert "register_heif_opener" in text
    assert "pillow_avif" in text
    assert "_sanitize_modal_log" in text
