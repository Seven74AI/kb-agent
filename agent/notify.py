"""Telegram notifications — failure alerts from the consumer and batch collector.

No-op when telegram_bot_token / telegram_chat_id are unset, so callers can
always call these without checking config first. Uses only stdlib (urllib) so
it works in every environment the app runs in (service, cron, MCP).
"""

import json
import urllib.parse
import urllib.request

from agent.config import Config

_VISIBLE_ERROR_MAX_LEN = 2000


def sanitize_visible_message(message: str) -> str:
    """Strip stack frames from user-visible error text (dashboard, Telegram, tasks)."""
    marker = "Traceback (most recent call last):"
    if marker not in message:
        return message[:_VISIBLE_ERROR_MAX_LEN]
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
    return summary[:_VISIBLE_ERROR_MAX_LEN]


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
        lines.append("cause: " + "; ".join(dict.fromkeys(causes[:3])))
    send_message(config, "\n".join(lines))
