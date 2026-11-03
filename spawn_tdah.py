#!/usr/bin/env python3
"""Spawn ONE Cursor cloud agent to implement the Bunny Stream / TDAH pipeline."""
import os

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

PROMPT = """Implement a new "Bunny Stream / TDAH summit" pipeline in this kb-agent codebase, following the existing architecture exactly.

First read these files to learn the patterns (do not modify them):
- agent/detection.py — URL detection (PLATFORM_PATTERNS, detect_platform)
- agent/pipelines/__init__.py — Step dataclass + register_pipeline
- agent/pipelines/youtube.py — the video pipeline structure (download, extract audio, diarize, transcribe, synthesize, quality gate, minio, git push)
- agent/pipelines/web.py — synthesis prompt builder + quality gate wiring
- agent/exceptions.py — ErrorType, StepError
- agent/archival.py — step_minio_upload, step_git_push
- references/pipeline-bunny-stream.md — Bunny CDN download mechanics (READ THIS FULLY)
- templates/tdah-summit-note-template.md — the note structure (READ THIS FULLY)
- scripts/tdah_daily_check.py — Bunny UUID regex + download header details

TASKS:

1. DETECTION — in agent/detection.py, add two URL patterns to PLATFORM_PATTERNS (insert BEFORE the generic fallthrough, i.e. after the existing platform entries):
   (r"player\\.mediadelivery\\.net/", "bunny_stream")
   (r"terragrandir\\.com/", "bunny_stream")
   The content_type is "bunny_stream" (no subtype — these are all French conference videos).

2. PIPELINE — create agent/pipelines/bunny_stream.py. Register it with register_pipeline("bunny_stream", [...]). Steps (mirror youtube.py's step order and reuse the same helpers where possible):
   a. download_video: Given a terragrandir detail URL, extract the Bunny video UUID via regex r"player\\.mediadelivery\\.net/embed/\\d+/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})" from the page HTML (curl with a browser UA). Then download via ffmpeg HLS from the Bunny CDN host, sending header "Referer: https://player.mediadelivery.net/" (403 without it). If the URL is already a player.mediadelivery.net URL, use it directly. Save the video to the task temp dir. See references/pipeline-bunny-stream.md for the exact CDN host + HLS mechanics.
   b. extract_audio: ffmpeg → 16kHz mono WAV (whisper) + 8kHz mono WAV (pyannote), same as youtube.py.
   c. diarize + transcribe: reuse the SAME canonical scripts as youtube.py (scripts/diarize.py + scripts/transcribe.py, which are symlinked). Diarization is mandatory. language auto-detect is fine (whisper detects French).
   d. synthesize_note: write the note using templates/tdah-summit-note-template.md — content in FRENCH, section labels in FRENCH, in the exact canonical order stated in that template (Métadonnées → Résumé → Chapitres → Concepts clés → Points clés → Résumé par chapitre → Résumé par locuteur → Extractions utiles → Nuances & Limites → Transcription → Voir aussi). Build the synthesis prompt like web.py's _build_synthesis_prompt but loading templates/tdah-summit-note-template.md instead of the deep-treatment template, and instruct FRENCH content + FRENCH labels (this pipeline is the exception to the general English-label rule).
   e. quality_gate, minio_upload, see_also, git_push: reuse the same steps/helpers as the other pipelines (import from agent.archival, agent.pipelines.see_also, agent.quality as appropriate).

3. TESTS — create tests/test_pipelines_bunny_stream.py following the pattern of tests/test_pipelines_substack.py. Test at minimum: (a) detect_platform returns "bunny_stream" for terragrandir and player.mediadelivery.net URLs; (b) the UUID-extraction regex works on a sample detail-page HTML. Mock subprocess for any network calls.

4. VERIFY — run `pytest -q` and ensure ALL existing tests still pass PLUS your new tests.

CONSTRAINTS:
- Follow existing code style exactly (dataclasses, ErrorType, StepError, register_pipeline).
- Do NOT modify any other pipeline module or detection for other platforms.
- Reuse agent/archival.py and agent/pipelines/see_also.py — do not reimplement MinIO/git/see-also.
- The note MUST be FRENCH content + FRENCH section labels (unlike youtube/web which use English labels).
- Commit with a clear message and open a PR.
"""

opts = AgentOptions(
    api_key=key,
    model="composer-2.5",
    cloud=CloudAgentOptions(
        repos=[CloudRepository(url=REPO)],
        auto_create_pr=True,
        open_as_cursor_github_app=True,
        skip_reviewer_request=True,
    ),
)

with Agent.create(opts) as agent:
    run = agent.send(PROMPT)
    result = run.wait()
    print("STATUS:", result.status)
    if result.git and result.git.branches:
        b = result.git.branches[0]
        print("BRANCH:", b.branch)
        print("PR_URL:", b.pr_url)
    else:
        print("NO_GIT_INFO")
