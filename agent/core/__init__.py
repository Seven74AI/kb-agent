"""Pure domain layer — submit, detect, status, resume, search, note.

No Flask or MCP imports in this package.
"""

from agent.core.bulk import bulk_import, expand_youtube_playlist, ingest_playlist
from agent.core.detect import detect_type
from agent.core.diff import (
    ContentChangeResult,
    ContentChangeStatus,
    check_content_change,
    content_fingerprint,
    normalize_content,
    normalize_etag,
    normalize_source_content,
    sha256_fingerprint,
)
from agent.core.note import get_note
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
    "ContentChangeResult",
    "ContentChangeStatus",
    "check_content_change",
    "compute_staleness",
    "content_fingerprint",
    "diff_items",
    "detect_type",
    "extract_items",
    "format_elapsed",
    "get_note",
    "get_task",
    "is_poll_due",
    "items_to_urls",
    "list_tasks",
    "list_watch_sources",
    "normalize_content",
    "normalize_etag",
    "normalize_reason_pattern",
    "normalize_source_content",
    "sha256_fingerprint",
    "plan_submission",
    "poll_all_due_sources",
    "poll_watch_source",
    "register_watch_source",
    "resume_tasks",
    "run_watch_scheduler",
    "retry_task",
    "search_notes",
    "slugify",
    "submit_plan",
    "submit_url",
    "bulk_import",
    "expand_youtube_playlist",
    "ingest_playlist",
]
