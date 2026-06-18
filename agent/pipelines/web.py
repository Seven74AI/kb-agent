"""Web article pipeline — Firecrawl extraction and LLM synthesis.

The simplest pipeline. No cookies, no video. Verifies the entire architecture.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from agent.pipelines import Step, register_pipeline


def step_firecrawl_extract(task: dict, config) -> str:
    """Extract article content via Firecrawl, save raw markdown.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the raw markdown file.

    Raises:
        ValueError: If content is too short or extraction fails.
    """
    url = task["source_url"]
    slug = task.get("slug", task["id"][:8])

    result = subprocess.run(
        [
            "curl", "-s", "--max-time", "60",
            config.firecrawl_url,
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "url": url,
                "formats": ["markdown"],
                "waitFor": 3000,
            }),
        ],
        capture_output=True, text=True, timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Firecrawl curl failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
        markdown = data["data"]["markdown"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Firecrawl response parse error: {e}") from e

    if len(markdown) < 500:
        raise ValueError(f"Content too short ({len(markdown)} chars) — possible 404 or empty page")

    raw_path = Path(config.temp_dir) / f"{slug}_raw.md"
    raw_path.write_text(markdown, encoding="utf-8")

    return str(raw_path)


def step_synthesize_web(task: dict, config) -> str:
    """Placeholder: LLM synthesis for web articles.

    Full implementation in Phase 1 (Task 10).
    Currently writes a stub note to demonstrate the pipeline works.

    Args:
        task: Task dict with source_url, slug.
        config: Config dataclass.

    Returns:
        Path to the generated note file.
    """
    slug = task.get("slug", task["id"][:8])
    source_url = task.get("source_url", "")

    note_path = config.vault_path / "Knowledge base" / "articles" / f"{slug}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    note_path.write_text(f"""---
date: {datetime.now().strftime('%Y-%m-%d')}
source: web
source_url: {source_url}
confidence: plausible
tags: []
---

# Stub Note

## Summary
This is a placeholder note. Full LLM synthesis will be implemented in Phase 1.

## Key Claims
(To be filled by LLM synthesis step)

## Section-by-Section Analysis
(To be filled by LLM synthesis step)

## Critical Analysis
(To be filled by LLM synthesis step)
""", encoding="utf-8")

    return str(note_path)


# Register the web pipeline
register_pipeline("web", [
    Step(name="firecrawl_extract", fn=step_firecrawl_extract),
    Step(name="synthesize_note", fn=step_synthesize_web),
    Step(name="quality_gate", fn=lambda t, c: "stub"),
    Step(name="minio_upload", fn=lambda t, c: "stub"),
    Step(name="git_push", fn=lambda t, c: "stub"),
])
