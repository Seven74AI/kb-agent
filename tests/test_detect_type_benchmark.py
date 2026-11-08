"""395-URL detect_type benchmark (legacy-path snapshot; see fixture header)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.core.detect import detect_type

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "detect_type_benchmark.jsonl"


def _load_benchmark() -> list[dict]:
    rows = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@pytest.fixture
def legacy_detection(monkeypatch):
    """Disable Jev/network so CI matches the committed snapshot labels."""

    def _no_jev(*args, **kwargs):
        raise RuntimeError("jev disabled in benchmark")

    monkeypatch.setattr("agent.detection.route_jev", _no_jev)
    monkeypatch.setattr("agent.detection._fetch_content", lambda url, p: None)
    monkeypatch.setattr(
        "agent.detection._detect_threads_subtype",
        lambda url: ("threads_text", False),
    )


def test_benchmark_fixture_has_395_urls():
    rows = _load_benchmark()
    assert len(rows) == 395
    urls = [r["url"] for r in rows]
    assert len(urls) == len(set(urls)), "benchmark URLs must be unique"


def test_detect_type_matches_benchmark(legacy_detection):
    rows = _load_benchmark()
    mismatches = []
    for row in rows:
        got = detect_type(row["url"])
        if got["content_type"] != row["content_type"]:
            mismatches.append(
                (row["url"], row["content_type"], got["content_type"])
            )
    assert not mismatches, f"{len(mismatches)} mismatches; first: {mismatches[:5]}"
