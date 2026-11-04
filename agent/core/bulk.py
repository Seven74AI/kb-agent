"""Bulk URL submission — playlist expansion and multi-URL import."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

import sqlite3

from agent.config import Config
from agent.core.submit import submit_url

# yt-dlp lives in the py3.11 hermes-agent venv; the MCP server runs in a separate
# mcp-venv with system PATH (no yt-dlp), so resolve it explicitly.
YTDLP_BIN = shutil.which("yt-dlp") or "/usr/local/lib/hermes-agent/venv/bin/yt-dlp"

ExpandPlaylistFn = Callable[[str], list[str]]
SubmitFn = Callable[[str, Config, sqlite3.Connection], dict]


def expand_youtube_playlist(playlist_url: str) -> list[str]:
    """Return watch URLs for entries in a YouTube playlist (flat extract, no download)."""
    playlist_url = (playlist_url or "").strip()
    if not playlist_url:
        return []
    try:
        proc = subprocess.run(
            [
                YTDLP_BIN,
                "--flat-playlist",
                "--print",
                "url",
                playlist_url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def bulk_import(
    urls: list[str],
    config: Config,
    agent_conn: sqlite3.Connection,
    *,
    submit_fn: SubmitFn | None = None,
) -> dict:
    """Submit each URL via :func:`submit_url` (or injected *submit_fn*)."""
    submit = submit_fn or submit_url
    cleaned = [(u or "").strip() for u in urls if (u or "").strip()]
    results = [submit(u, config, agent_conn) for u in cleaned]
    return {"count": len(results), "results": results}


def ingest_playlist(
    playlist_url: str,
    config: Config,
    agent_conn: sqlite3.Connection,
    *,
    expand_fn: ExpandPlaylistFn | None = None,
    submit_fn: SubmitFn | None = None,
) -> dict:
    """Expand a playlist to N video URLs and submit each one."""
    expand = expand_fn or expand_youtube_playlist
    playlist_url = (playlist_url or "").strip()
    if not playlist_url:
        return {
            "status": "error",
            "error": "playlist_url required",
            "count": 0,
            "results": [],
        }
    urls = expand(playlist_url)
    if not urls:
        return {
            "status": "error",
            "error": "no videos found in playlist",
            "playlist_url": playlist_url,
            "count": 0,
            "results": [],
        }
    batch = bulk_import(urls, config, agent_conn, submit_fn=submit_fn)
    return {
        "status": "ok",
        "playlist_url": playlist_url,
        "count": batch["count"],
        "results": batch["results"],
    }
