"""Telegram notifications — failure alerts from the consumer and batch collector.

No-op when telegram_bot_token / telegram_chat_id are unset, so callers can
always call these without checking config first. Uses only stdlib (urllib) so
it works in every environment the app runs in (service, cron, MCP).
"""

import json
import re
import urllib.parse
import urllib.request

from agent.config import Config

_VISIBLE_ERROR_MAX_LEN = 2000
_STDERR_LINE_MAX = 200
_KNOWN_ROOT_PATH_RE = re.compile(r"/(?:root|home|tmp)(?:/[^\s\"')\]]+)?", re.IGNORECASE)
_MULTI_SEGMENT_ABS_PATH_RE = re.compile(r"/[\w.-]+(?:/[\w.-]+)+")
_ROOTED_REL_PATH_RE = re.compile(r"/[a-z_][a-z0-9_]*(?:/[^\s\"')\]]+)+")
_SUBPROCESS_STDERR_LINE_RE = re.compile(
    r"^(?:yt-dlp|curl|megadl|ffmpeg|mc:|ERROR:|\[download\])",
    re.IGNORECASE,
)
_CREDENTIAL_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*(?!Bearer\b)\S+"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)


def redact_credential_shapes(text: str) -> str:
    """Replace token-shaped assignment and Bearer headers with fixed placeholders."""
    if not text:
        return text
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _CREDENTIAL_SECRET_RE.sub("[credential redacted]", text)
    return text


def _scrub_paths(text: str) -> str:
    text = _KNOWN_ROOT_PATH_RE.sub("[path]", text)
    text = _MULTI_SEGMENT_ABS_PATH_RE.sub("[path]", text)
    text = _ROOTED_REL_PATH_RE.sub("[path]", text)
    return text


def _truncate_subprocess_stderr_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        head = stripped.lstrip()
        if (
            _SUBPROCESS_STDERR_LINE_RE.match(head)
            or (head.lower().startswith("stderr") and len(stripped) > _STDERR_LINE_MAX)
        ):
            if len(stripped) > _STDERR_LINE_MAX:
                stripped = stripped[: _STDERR_LINE_MAX - 3] + "..."
        lines.append(stripped)
    return "\n".join(lines)


def _finalize_visible_message(text: str) -> str:
    text = _scrub_paths(text)
    text = redact_credential_shapes(text)
    text = _truncate_subprocess_stderr_lines(text)
    return text[:_VISIBLE_ERROR_MAX_LEN]


def ocr_error_json(error: str) -> dict:
    """Artifact-safe OCR failure payload for ``*.ocr.json`` writers."""
    return {"error": sanitize_visible_message(error)}


def sanitize_visible_message(message: str) -> str:
    """Strip stack frames, paths, and bulky stderr from user-visible error text."""
    marker = "Traceback (most recent call last):"
    if marker not in message:
        return _finalize_visible_message(message)
    head, tail = message.split(marker, 1)
    exc_line = ""
    for line in reversed(tail.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("File "):
            exc_line = stripped
            break
    summary = head.rstrip()
    if exc_line and exc_line not in summary:
        summary = f"{summary}\n{exc_line}" if summary else exc_line
    return _finalize_visible_message(summary)


def send_message(config: Config, text: str) -> bool:
    """Send a Telegram message. Returns True on success, False otherwise.

    Silently returns False when unconfigured or when the API call fails —
    notification is best-effort and must never crash the pipeline/consumer.
    """
    if not (config.telegram_bot_token and config.telegram_chat_id):
        return False
    try:
        url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        body = urllib.parse.urlencode(
            {"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "HTML"}
        ).encode()
        req = urllib.request.Request(url, data=body)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return bool(json.loads(resp.read().decode()).get("ok"))
    except Exception:  # noqa: BLE001 — best-effort notification
        return False


def notify_task(config: Config, slug: str, content_type: str, status: str, reason: str = "") -> None:
    """Immediate alert for a single task failure/pause (consumer)."""
    if status == "failed":
        head = f"⛔ Task FAILED: <code>{slug}</code> ({content_type})"
    else:
        head = f"⏸ Task PAUSED: <code>{slug}</code> ({content_type})"
    lines = [head]
    if reason:
        lines.append(f"reason: {sanitize_visible_message(reason)}")
    send_message(config, "\n".join(lines))


def notify_batch(config: Config, failed: int, total: int, dispatched: int, causes: list[str]) -> None:
    """Digest for a batch collector run (crashed, or had failed items)."""
    if total > 0 and failed >= total:
        head = f"🚨 Batch run: ALL {failed}/{total} failed"
    else:
        head = f"⚠️ Batch run: {failed}/{total} failed · {dispatched} dispatched"
    lines = [head]
    if causes:
        scrubbed = [sanitize_visible_message(c) for c in dict.fromkeys(causes[:3])]
        lines.append("cause: " + "; ".join(scrubbed))
    send_message(config, "\n".join(lines))
