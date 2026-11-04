"""Hermes knowledge-base pipeline triggers for benchmark runs."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def hermes_cli_available() -> bool:
    return shutil.which("hermes") is not None


def submit_hermes_kanban(
    url: str,
    *,
    assignee: str = "researcher",
    max_runtime_seconds: int = 3600,
) -> dict[str, Any]:
    """Create a Hermes knowledge-base kanban ticket for one benchmark URL."""
    if not hermes_cli_available():
        return {"ok": False, "error": "hermes CLI not found on PATH"}

    body = (
        "Note-quality A/B benchmark — process this URL through the Hermes "
        "knowledge-base pipeline.\n\n"
        f"URL: {url}\n\n"
        "Write the note to the vault with correct source_url frontmatter. "
        "After the worker finishes, snapshot the note for comparison with kb-agent."
    )
    title = f"KB A/B: {url}"
    cmd = [
        "hermes",
        "kanban",
        "--board",
        "knowledge-base",
        "create",
        "--assignee",
        assignee,
        "--skill",
        "knowledge-base",
        "--max-runtime",
        str(max_runtime_seconds),
        "--body",
        body,
        title,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}

    if result.returncode != 0:
        return {
            "ok": False,
            "error": (result.stderr or result.stdout or "hermes kanban create failed").strip(),
            "returncode": result.returncode,
        }
    return {
        "ok": True,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
