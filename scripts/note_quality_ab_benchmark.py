#!/usr/bin/env python3
"""Note-quality A/B benchmark — kb-agent vs Hermes knowledge-base pipeline.

Feeds the same URL set to both systems (optional ``run`` mode), then produces a
comparison report with quality-gate metrics and a side-by-side HTML review page.

Typical workflow on the VPS:

1. ``run`` — enqueue Hermes kanban tickets and kb-agent tasks for each URL.
2. After both pipelines finish, copy notes into staging dirs (or use dual vault paths).
3. ``compare`` — emit ``comparison.json`` + ``review.html`` for manual review.

Examples::

    python scripts/note_quality_ab_benchmark.py compare \\
        --urls tests/fixtures/note_quality_ab/urls.jsonl \\
        --kb-agent-dir tests/fixtures/note_quality_ab/kb_agent \\
        --hermes-dir tests/fixtures/note_quality_ab/hermes \\
        --output /tmp/note-quality-ab

    python scripts/note_quality_ab_benchmark.py run \\
        --urls urls.jsonl --output data/ab-run-001 \\
        --submit-hermes --submit-kb-agent --ignore-dedup
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.benchmark.hermes import hermes_cli_available, submit_hermes_kanban  # noqa: E402
from agent.benchmark.note_quality import (  # noqa: E402
    compare_from_staging,
    compare_from_vaults,
    default_hermes_vault,
    load_url_set,
    write_comparison_report,
)
from agent.config import load_config  # noqa: E402
from agent.core.detect import detect_type  # noqa: E402
from agent.core.submit import submit_url  # noqa: E402
from agent.db import AGENT_SCHEMA, init_db  # noqa: E402


def _enrich_content_types(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("content_type"):
            out.append(row)
            continue
        detected = detect_type(row["url"])
        out.append({**row, "content_type": detected["content_type"]})
    return out


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(
        config_path=Path(args.config) if args.config else REPO / "config.yaml",
    )
    rows = _enrich_content_types(load_url_set(Path(args.urls)))
    out_dir = Path(args.output)

    if args.kb_agent_dir and args.hermes_dir:
        entries = compare_from_staging(
            rows,
            kb_agent_dir=Path(args.kb_agent_dir),
            hermes_dir=Path(args.hermes_dir),
            config=config,
        )
    elif args.kb_vault or args.hermes_vault:
        kb_vault = Path(args.kb_vault) if args.kb_vault else config.vault_path
        hermes_vault = Path(args.hermes_vault) if args.hermes_vault else default_hermes_vault()
        entries = compare_from_vaults(
            rows,
            kb_vault=kb_vault,
            hermes_vault=hermes_vault,
            config=config,
        )
    else:
        print(
            "compare requires --kb-agent-dir + --hermes-dir, or --kb-vault / --hermes-vault",
            file=sys.stderr,
        )
        return 2

    paths = write_comparison_report(entries, out_dir)
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))["summary"]
    print(
        json.dumps(
            {
                "summary": summary,
                "report_json": str(paths["json"]),
                "review_html": str(paths["html"]),
            },
            indent=2,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(
        config_path=Path(args.config) if args.config else REPO / "config.yaml",
    )
    rows = _enrich_content_types(load_url_set(Path(args.urls)))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.jsonl"

    if not args.submit_hermes and not args.submit_kb_agent:
        print("run: enable at least one of --submit-hermes or --submit-kb-agent", file=sys.stderr)
        return 2

    if args.submit_hermes and not hermes_cli_available():
        print("run: hermes CLI not on PATH; cannot --submit-hermes", file=sys.stderr)
        return 2

    db_path = Path(args.agent_db) if args.agent_db else config.kb_agent_dir / "data" / "agent.db"
    conn = None
    if args.submit_kb_agent:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = init_db(db_path, AGENT_SCHEMA)
        conn.row_factory = sqlite3.Row

    results = []
    try:
        for row in rows:
            url = row["url"]
            record: dict = {"url": url, "content_type": row.get("content_type")}
            if args.submit_hermes:
                record["hermes"] = submit_hermes_kanban(
                    url,
                    assignee=args.hermes_assignee,
                    max_runtime_seconds=args.hermes_max_runtime,
                )
            if args.submit_kb_agent and conn is not None:
                record["kb_agent"] = submit_url(
                    url,
                    config,
                    conn,
                    ignore_dedup=args.ignore_dedup,
                )
            results.append(record)
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
    finally:
        if conn is not None:
            conn.close()

    readme = out_dir / "README.txt"
    readme.write_text(
        "After Hermes workers and the kb-agent consumer finish:\n"
        "1. Snapshot notes to kb_agent/ and hermes/ under this directory, OR\n"
        "2. Run compare with --kb-vault and --hermes-vault if vaults differ.\n"
        f"3. python scripts/note_quality_ab_benchmark.py compare --urls {args.urls} "
        f"--kb-agent-dir {out_dir}/kb_agent --hermes-dir {out_dir}/hermes "
        f"--output {out_dir}/report\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "url_count": len(results)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="Build metrics report + side-by-side HTML")
    p_compare.add_argument("--urls", required=True, help="JSONL URL list")
    p_compare.add_argument("--output", required=True, help="Report output directory")
    p_compare.add_argument("--config", help="config.yaml path (default: repo config.yaml)")
    p_compare.add_argument("--kb-agent-dir", help="Staged kb-agent notes directory")
    p_compare.add_argument("--hermes-dir", help="Staged Hermes notes directory")
    p_compare.add_argument("--kb-vault", help="kb-agent vault root (grep by source_url)")
    p_compare.add_argument("--hermes-vault", help="Hermes vault root (grep by source_url)")
    p_compare.set_defaults(func=cmd_compare)

    p_run = sub.add_parser("run", help="Enqueue URLs to Hermes and/or kb-agent")
    p_run.add_argument("--urls", required=True, help="JSONL URL list")
    p_run.add_argument("--output", required=True, help="Run output directory")
    p_run.add_argument("--config", help="config.yaml path")
    p_run.add_argument("--submit-hermes", action="store_true")
    p_run.add_argument("--submit-kb-agent", action="store_true")
    p_run.add_argument("--ignore-dedup", action="store_true", help="kb-agent: enqueue even if URL in vault")
    p_run.add_argument("--hermes-assignee", default="researcher")
    p_run.add_argument("--hermes-max-runtime", type=int, default=3600)
    p_run.add_argument("--agent-db", help="Path to agent.db for kb-agent enqueue")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
