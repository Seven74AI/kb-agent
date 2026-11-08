"""Regression tests for scripts/batch_ocr.py log scrubbing."""

import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

import scripts.batch_ocr as bo


def test_batch_ocr_scrubs_modal_child_tail(capsys, tmp_path, monkeypatch):
    stage = tmp_path / "imgs"
    stage.mkdir()
    (stage / "a.png").write_bytes(b"x" * 50)
    out = tmp_path / "out"
    sensitive = (
        "vlm ok prefix path=/home/ubuntu/secret/img.png api_key=ocrleak suffix"
    )

    def fake_sh(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, sensitive, "")

    monkeypatch.setattr(bo, "sh", fake_sh)
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["batch_ocr.py", str(stage), "--out-dir", str(out)])
        rc = bo.main()

    assert rc == 0
    logged = capsys.readouterr().out
    assert "/home/ubuntu" not in logged
    assert "ocrleak" not in logged
    assert "[path]" in logged
    assert "[credential redacted]" in logged
    assert "vlm ok prefix" in logged


def test_ocr_error_json_scrubs_paths_and_credentials(tmp_path):
    from agent.notify import ocr_error_json

    raw = (
        "boom path=/home/ubuntu/secret/img.png "
        "api_key=deadbeefdeadbeefdeadbeefdeadbeef suffix"
    )
    doc = ocr_error_json(raw)
    assert "error" in doc
    assert "/home/ubuntu/secret/img.png" in doc["error"]
    assert "deadbeef" not in doc["error"]
    assert "[path]" not in doc["error"]
    assert "[credential redacted]" in doc["error"]

    from agent.persist import persist_durable_json

    out = tmp_path / "fail.png.ocr.json"
    persist_durable_json(out, doc, policy="boundary4")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "/home/ubuntu/secret/img.png" in loaded["error"]


def test_kb_ocr_success_path_wires_prepare_ocr_reading_for_write():
    text = (WORKSPACE / "modal" / "kb_ocr.py").read_text(encoding="utf-8")
    assert "persist_durable_json" in text
    assert 'json.dump(' not in text
