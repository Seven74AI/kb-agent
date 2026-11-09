#!/usr/bin/env python3
"""Regenerate tests/fixtures/detect_type_benchmark.jsonl (395-row snapshot).

Uses the legacy detection path (Jev disabled, no fetch) so CI stays offline.
Re-run after intentional changes to agent.detection routing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import agent.detection as det  # noqa: E402
from agent.core.detect import detect_type  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "detect_type_benchmark.jsonl"


def main() -> None:
    det.route_jev = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("jev off"))
    det._fetch_content = lambda url, p: (None, url)

    rows = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != 395:
        raise SystemExit(f"expected 395 rows, found {len(rows)}")

    out = []
    for row in rows:
        got = detect_type(row["url"])
        out.append(
            {
                "url": row["url"],
                "platform": got["platform"],
                "content_type": got["content_type"],
            }
        )

    FIXTURE.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in out) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {FIXTURE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
