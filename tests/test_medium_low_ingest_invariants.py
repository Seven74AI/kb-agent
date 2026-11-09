"""Regression tests for medium/low deferred-media ingest invariants."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.config import Config
from agent.core.submit import plan_submission
from agent.detection import (
    _detect_threads_subtype,
    _fetch_content,
    detect_content_subtype_with_signals,
)
from agent.exceptions import ErrorType, StepError
from agent.pipelines.instagram import (
    _carousel_meta_path,
    _carousel_slides_path,
    _ig_comment_sidecar_path,
    step_ig_carousel_extract,
    step_ig_carousel_metadata,
)
from agent.pipelines.video_common import (
    _artifact_stem,
    check_density,
    resolve_batch_ingest_video,
    resolve_instagram_reel_video,
    resolve_threads_video_path,
    resolve_threads_raw_path,
    threads_raw_path,
    step_fetch_transcript,
)
from agent.pipelines.instagram import step_ig_cleanup, step_ig_storage_upload
from agent.pipelines.threads import step_threads_cleanup, step_threads_storage_upload
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
        storage_bucket="kb-agent-archive",
    )


def test_detect_threads_subtype_defaults_text_on_fetch_failure(monkeypatch, caplog):
    def boom(cmd, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("agent.detection.subprocess.run", boom)
    caplog.set_level("WARNING")
    subtype, fetch_failed = _detect_threads_subtype("https://threads.net/@u/post/1")
    assert subtype == "threads_text"
    assert fetch_failed is True


def test_threads_subtype_fetch_failed_isolated_per_detection_call(monkeypatch, tmp_path):
    """One request's Threads fetch failure must not affect another request's signal."""
    cookie_dir = tmp_path / "cookies"
    cookie_dir.mkdir(parents=True, exist_ok=True)
    (cookie_dir / "threads_cookies.txt").write_text("x", encoding="utf-8")
    cfg = Config(kb_agent_dir=tmp_path, vault_path=tmp_path / "vault", cookie_dir=cookie_dir)
    monkeypatch.setattr("agent.detection.load_config", lambda: cfg)
    monkeypatch.setattr("agent.detection._fetch_content", lambda url, p: (None, url))

    call_n = {"n": 0}

    def alternating_subprocess(cmd, **kwargs):
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise OSError("down")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.detection.subprocess.run", alternating_subprocess)
    url = "https://threads.net/@user/post/abc"
    failed = detect_content_subtype_with_signals(url, "threads")
    ok = detect_content_subtype_with_signals(url, "threads")
    assert failed.threads_subtype_fetch_failed is True
    assert ok.threads_subtype_fetch_failed is False


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
    assert excinfo.value.step_name == "ig_carousel_extract"
    assert excinfo.value.details is not None
    assert len(excinfo.value.details) <= 500


@patch("subprocess.run")
def test_ig_storage_upload_batch_uses_artifact_stem_video(mock_run, config):
    task_id = "batch-reel-uuid-1234"
    slug = "owner_reel_abc"
    task = {
        "id": task_id,
        "slug": slug,
        "content_type": "instagram_reel",
        "_batch_ingest": True,
    }
    note_path = config.vault_path / "Knowledge base" / "instagram" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "---\nsource_files:\n  video: <filled>\n---\n# Reel\n" + ("x" * 100),
        encoding="utf-8",
    )
    (config.temp_dir / f"{task_id}.mp4").write_bytes(b"video-bytes")
    (config.temp_dir / f"{task_id}_merged.json").write_text("{}", encoding="utf-8")
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    result = step_ig_storage_upload(task, config)
    assert result.startswith("Uploaded:")
    assert f"videos/{slug}.mp4" in result
    assert f"transcripts/{slug}.json" in result


def test_ig_cleanup_batch_removes_id_keyed_reel_mp4(config):
    task_id = "batch-reel-cleanup"
    task = {
        "id": task_id,
        "slug": "ig/slug",
        "content_type": "instagram_reel",
        "_batch_ingest": True,
    }
    video = config.temp_dir / f"{task_id}.mp4"
    video.write_bytes(b"v")
    (config.temp_dir / f"{task_id}_16k.wav").write_text("a", encoding="utf-8")
    step_ig_cleanup(task, config)
    assert not video.exists()
    assert not (config.temp_dir / f"{task_id}_16k.wav").exists()


@patch("subprocess.run")
def test_threads_storage_upload_batch_uses_artifact_stem_video(mock_run, config):
    task_id = "batch-threads-uuid"
    slug = "user_post"
    task = {
        "id": task_id,
        "slug": slug,
        "content_type": "threads_video",
        "_batch_ingest": True,
    }
    note_path = config.vault_path / "Knowledge base" / "threads" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "---\nsource_files:\n  video: <filled>\n---\n# Vid\n",
        encoding="utf-8",
    )
    (config.temp_dir / f"{task_id}.mp4").write_bytes(b"video")
    (config.temp_dir / f"{task_id}_merged.json").write_text("{}", encoding="utf-8")
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    result = step_threads_storage_upload(task, config)
    assert result.startswith("Uploaded:")
    assert f"videos/{slug}.mp4" in result
    assert f"transcripts/{slug}.json" in result


def test_threads_cleanup_batch_removes_plain_mp4(config):
    task_id = "batch-threads-clean"
    task = {
        "id": task_id,
        "slug": "thr_slug",
        "content_type": "threads_video",
        "_batch_ingest": True,
    }
    video = config.temp_dir / f"{task_id}.mp4"
    video.write_bytes(b"v")
    (config.temp_dir / f"{task_id}_8k.wav").write_text("x", encoding="utf-8")
    step_threads_cleanup(task, config)
    assert not video.exists()
    assert not (config.temp_dir / f"{task_id}_8k.wav").exists()


def test_check_density_uses_artifact_stem_metadata(config, tmp_path):
    vault = config.vault_path
    note_path = vault / "Knowledge base" / "videos" / "display-slug.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("x" * 2000, encoding="utf-8")
    task = {"id": "ingest-meta-id", "slug": "display-slug", "_batch_ingest": True}
    meta = {"duration": 600}
    (config.temp_dir / "ingest-meta-id_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    check_density(note_path, config, task)


def test_threads_raw_path_writer_uses_artifact_stem(config):
    ingest_id = "ingest-threads-raw-001"
    slug = "threads.net_user_post_X"
    batch_task = {
        "id": ingest_id,
        "slug": slug,
        "_batch_ingest": True,
    }
    assert threads_raw_path(config, batch_task).name == f"{ingest_id}_threads_raw.json"
    consumer = {"id": ingest_id, "slug": slug}
    assert threads_raw_path(config, consumer).name == f"{slug}_threads_raw.json"


@patch("agent.pipelines.instagram._validate_ig_cookies")
@patch("subprocess.run")
def test_ig_carousel_sidecars_use_artifact_stem_batch(mock_run, _mock_cookies, config):
    ingest_id = "ingest-ig-carousel-001"
    slug = "instagram.com_p_DEF456"
    task = {
        "id": ingest_id,
        "slug": slug,
        "source_url": "https://www.instagram.com/p/DEF456/",
        "content_type": "instagram_post",
        "_batch_ingest": True,
    }
    mock_run.return_value = MagicMock(returncode=0, stdout="t||u||0||2", stderr="")
    meta_written = step_ig_carousel_metadata(task, config)
    assert Path(meta_written).name == f"{ingest_id}_carousel_meta.json"
    assert _carousel_meta_path(config, task).is_file()
    assert not (config.temp_dir / f"{slug}_carousel_meta.json").exists()


def test_ig_carousel_path_helpers_match_consumer_slug_when_not_batch(config):
    slug = "instagram.com_p_DEF456"
    task = {"id": "solo-id", "slug": slug, "content_type": "instagram_post"}
    assert _carousel_slides_path(config, task).name == f"{slug}_carousel_slides.json"
    assert _ig_comment_sidecar_path(config, task).name == f"{slug}_ig_comment.json"


def test_ig_cleanup_batch_removes_carousel_and_comment_sidecars(config):
    ingest_id = "ingest-ig-cleanup-001"
    slug = "instagram.com_p_XYZ"
    task = {
        "id": ingest_id,
        "slug": slug,
        "content_type": "instagram_post",
        "_batch_ingest": True,
        "source_url": "https://www.instagram.com/p/XYZ/",
    }
    temp = config.temp_dir
    for name in (
        f"{ingest_id}_carousel_slides.json",
        f"{ingest_id}_carousel_meta.json",
        f"{ingest_id}_ig_comment.json",
        f"{ingest_id}_playwright.py",
    ):
        Path(temp / name).write_text("{}", encoding="utf-8")
    decoy = Path(temp / f"{slug}_carousel_meta.json")
    decoy.write_text("{}", encoding="utf-8")
    step_ig_cleanup(task, config)
    assert not Path(temp / f"{ingest_id}_carousel_slides.json").exists()
    assert not Path(temp / f"{ingest_id}_carousel_meta.json").exists()
    assert decoy.exists()


def test_resolve_threads_raw_path_finds_batch_sidecar_for_synthesize_slug(config):
    ingest_id = "ingest-threads-raw-002"
    slug = "threads.net_user_post_Y"
    payload = {
        "author": "threaduser",
        "captions": ["caption from batch detect"],
        "parent_context": "parent post text",
    }
    (config.temp_dir / f"{ingest_id}_threads_raw.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    synth_task = {"id": ingest_id, "slug": slug}
    resolved = resolve_threads_raw_path(config, synth_task)
    assert resolved.is_file()
    assert json.loads(resolved.read_text(encoding="utf-8"))["author"] == "threaduser"


def test_resolve_batch_ingest_video_prefers_h264_glob(config):
    batch_task = {"id": "tid-h264", "slug": "s", "_batch_ingest": True}
    h264 = config.temp_dir / "tid-h264_h264.mp4"
    h264.write_bytes(b"2" * 100)
    assert resolve_batch_ingest_video(config, batch_task) == h264
    assert not (config.temp_dir / "tid-h264.mp4").exists()


def test_find_video_file_resolves_slug_h264_only(config):
    from agent.pipelines.video_common import _find_video_file

    slug = "consumer-slug-only-h264"
    (config.temp_dir / f"{slug}_h264.mp4").write_bytes(b"vid")
    assert _find_video_file(config, slug).name == f"{slug}_h264.mp4"


def test_resolve_video_paths_match_download_stems(config):
    batch_task = {"id": "tid1", "slug": "s", "_batch_ingest": True}
    (config.temp_dir / "tid1.mp4").write_bytes(b"1")
    assert resolve_instagram_reel_video(config, batch_task).name == "tid1.mp4"
    assert resolve_threads_video_path(config, batch_task).name == "tid1.mp4"
    solo = {"id": "t2", "slug": "solo_slug", "_batch_ingest": False}
    (config.temp_dir / "solo_slug_reel.webm").write_bytes(b"1")
    assert resolve_instagram_reel_video(config, solo).suffix == ".webm"
    (config.temp_dir / "solo_slug_threads_video.mp4").write_bytes(b"1")
    assert resolve_threads_video_path(config, solo).name == "solo_slug_threads_video.mp4"


def test_kb_ocr_registers_heif_and_avif_openers():
    text = (Path(__file__).resolve().parents[1] / "modal" / "kb_ocr.py").read_text(
        encoding="utf-8"
    )
    assert "register_heif_opener" in text
    assert "pillow_avif" in text
    assert "_sanitize_emit" in text
    assert "sanitize_visible_message" in text


@patch("agent.pipelines.books.subprocess.run")
def test_megadl_download_temp_dir_uses_artifact_stem_under_batch(mock_run, config):
    """LOCAL mega_* dirs use ingest id when slug differs under _batch_ingest."""
    from agent.pipelines.books import step_megadl_download

    ingest_id = "batch-ingest-book-parent-001"
    task = {
        "id": ingest_id,
        "slug": "human_readable_slug",
        "_batch_ingest": True,
        "source_url": "https://mega.nz/file/abc123",
    }
    megadl_paths: list[Path] = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "megadl":
            dl_path = Path(cmd[cmd.index("--path") + 1])
            megadl_paths.append(dl_path)
            (dl_path / "sample.epub").write_bytes(b"epub")
            return MagicMock(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected command: {cmd}")

    mock_run.side_effect = fake_run
    step_megadl_download(task, config)
    assert megadl_paths == [config.temp_dir / f"mega_{ingest_id}"]
