"""Pure domain layer — submit, detect, status, resume, search, note.

No Flask or MCP imports in this package.
"""

from agent.core.bulk import bulk_import, ingest_playlist
from agent.core.detect import detect_type
from agent.core.note import get_note
from agent.core.pipelines import PIPELINE_STEPS, steps_for_content_type
from agent.core.resume import normalize_reason_pattern, resume_tasks, retry_task
from agent.core.search import search_notes
from agent.core.slugify import slugify
from agent.core.status import compute_staleness, format_elapsed, get_task, list_tasks
from agent.core.submit import plan_submission, submit_plan, submit_url
from agent.core.watch_source import (
    diff_items,
    extract_items,
    is_poll_due,
    items_to_urls,
    list_watch_sources,
    poll_all_due_sources,
    poll_watch_source,
    register_watch_source,
    run_watch_scheduler,
)

__all__ = [
    "PIPELINE_STEPS",
    "bulk_import",
    "compute_staleness",
    "diff_items",
    "detect_type",
    "extract_items",
    "format_elapsed",
    "get_note",
    "get_task",
    "ingest_playlist",
    "is_poll_due",
    "items_to_urls",
    "list_tasks",
    "list_watch_sources",
    "normalize_reason_pattern",
    "plan_submission",
    "poll_all_due_sources",
    "poll_watch_source",
    "register_watch_source",
    "resume_tasks",
    "run_watch_scheduler",
    "retry_task",
    "search_notes",
    "slugify",
    "steps_for_content_type",
    "submit_plan",
    "submit_url",
]
