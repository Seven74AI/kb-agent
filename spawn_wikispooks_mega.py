#!/usr/bin/env python3
"""Spawn TWO Cursor cloud agents in parallel: Wikispooks pipeline + Mega-video routing fix.

Disjoint files: wikispooks.py (new) + detection.py vs books.py (mega_video registration).
Run AFTER the TDAH agent's PR is merged (so detection.py already carries bunny_stream).
"""
import os, threading, json

env = {}
for line in open('/root/.hermes/.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
key = env['CURSOR_API_KEY']
os.environ['CURSOR_API_KEY'] = key

from cursor_sdk import Agent, AgentOptions, CloudAgentOptions, CloudRepository

REPO = "https://github.com/Seven74AI/kb-agent.git"

WIKISPOOKS_PROMPT = """Implement a "Wikispooks" pipeline in this kb-agent codebase.

Read these first (do not modify):
- agent/detection.py — URL detection
- agent/pipelines/__init__.py — Step + register_pipeline
- agent/pipelines/web.py — the closest pattern (Firecrawl extract → deep-treatment synthesis → quality gate → minio → git push)
- references/pipeline-wikispooks.md — READ FULLY (DDOS bypass mechanics)

TASKS:
1. DETECTION — add (r"wikispooks\\.org/", "wikispooks") to PLATFORM_PATTERNS in agent/detection.py (before the generic fallthrough).
2. PIPELINE — create agent/pipelines/wikispooks.py, register_pipeline("wikispooks", [...]):
   a. firecrawl_extract: apply the DDOS bypass URL transformation FIRST (/wiki/X/Y → /w/welcome_back.php?goto=%2Fwiki%2FX%2FY), then reuse step_firecrawl_extract from agent.pipelines.web. The note's source_url stays the DIRECT URL (not the bypass).
   b. synthesize_note: deep treatment — reuse step_synthesize_web from agent.pipelines.web.
   c. quality_gate, minio_upload, see_also, git_push: reuse the same step functions from agent.pipelines.web / agent.archival / agent.pipelines.see_also.
3. TESTS — create tests/test_pipelines_wikispooks.py: (a) detect_platform returns "wikispooks" for wikispooks.org URLs; (b) the bypass URL transformation produces the correct welcome_back.php URL. Mock subprocess for network.
4. Run `pytest -q` — all existing tests plus new ones must pass.

CONSTRAINTS: reuse web.py steps by importing them (do not reimplement synthesis/quality/minio); do not modify other pipelines; keep source_url = direct URL in the note.
Commit with a clear message and open a PR.
"""

MEGA_PROMPT = """Fix the Mega VIDEO routing bug in this kb-agent codebase.

Read these first (do not modify):
- agent/detection.py — _detect_mega_subtype returns "mega_video" for .mp4/.mkv/.webm files
- agent/pipelines/__init__.py — get_pipeline falls back to "web" for unregistered types
- agent/pipelines/youtube.py — the canonical video pipeline steps (download, extract audio, diarize, transcribe, synthesize, quality gate, minio, git push)
- agent/pipelines/books.py — see the existing "mega" pipeline registration (book-batch download) and how _detect_mega_subtype classifies files

PROBLEM: _detect_mega_subtype returns "mega_video" for video files, but no "mega_video" pipeline is registered, so get_pipeline("mega_video") silently falls back to the "web" pipeline (Firecrawl text extraction) — wrong for a video file.

TASKS:
1. Register a "mega_video" pipeline (register_pipeline("mega_video", [...])) — either in agent/pipelines/books.py next to the existing "mega" registration, or in a new agent/pipelines/mega_video.py module. The steps must mirror youtube.py's video pipeline but download from Mega (megadl or mega.py) instead of yt-dlp: download_video → extract_audio (16k + 8k WAV) → diarize (mandatory) → transcribe → merge/identify → synthesize (youtube note template) → quality_gate → minio_upload → see_also → git_push. Reuse youtube.py's step functions by importing them where possible.
2. Do NOT modify agent/detection.py — it already returns "mega_video"; the only gap is the missing pipeline registration.
3. TESTS — in tests/test_pipelines_books.py (or a new file), assert get_pipeline("mega_video") returns the video pipeline (NOT the web fallback).
4. Run `pytest -q` — all existing tests plus new ones must pass.

CONSTRAINTS: reuse youtube.py step functions by import (do not reimplement diarize/transcribe/synthesize); do not modify detection.py; do not touch other pipelines.
Commit with a clear message and open a PR.
"""

TASKS = [
    ("wikispooks", WIKISPOOKS_PROMPT),
    ("mega_video", MEGA_PROMPT),
]
results = {}

def run_task(name, prompt):
    opts = AgentOptions(
        api_key=key, model="composer-2.5",
        cloud=CloudAgentOptions(
            repos=[CloudRepository(url=REPO)],
            auto_create_pr=True,
            open_as_cursor_github_app=True,
            skip_reviewer_request=True,
        ),
    )
    try:
        with Agent.create(opts) as agent:
            run = agent.send(prompt)
            result = run.wait()
            pr_url = None
            branch = None
            if result.git and result.git.branches:
                branch = result.git.branches[0].branch
                pr_url = result.git.branches[0].pr_url
            results[name] = {"status": result.status, "branch": branch, "pr_url": pr_url}
    except Exception as e:
        results[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

threads = [threading.Thread(target=run_task, args=(n, p)) for n, p in TASKS]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("=== CURSOR RESULTS ===")
print(json.dumps(results, indent=2, default=str))
