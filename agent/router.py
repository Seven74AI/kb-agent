"""Jev-based content routing via the OpenRouter decisions API.

Replaces the fragile scrape-and-parse subtype detection (threads/substack/mega)
with a dedicated decision model: URL + fetched content → content type.

The regex platform detector (detection.detect_platform) stays as the fast, free,
deterministic first pass; this module handles the ambiguous tail.
"""

import os

import httpx

from agent.core.redact import url_for_display
from agent.notify import redact_credential_shapes, sanitize_visible_message

MODEL = "typesafe/jev-1.14"
API_URL = "https://openrouter.ai/api/alpha/decisions"

# Platforms the regex first pass resolves directly — no Jev needed.
SIMPLE_PLATFORMS = frozenset(
    {
        "youtube",
        "instagram_reel",
        "instagram_post",
        "bunny_stream",
        "wikispooks",
        "book",
        "rumble",
    }
)

# Platforms whose subtype requires inspecting fetched content (Jev's job).
SUBTYPE_PLATFORMS = frozenset({"threads", "substack", "mega"})

# Routing taxonomy — descriptions must be unambiguous; Jev relies on them.
CRITERIA = {
    "youtube": "A YouTube video (youtube.com/watch, youtu.be, /shorts, /embed, music.youtube)",
    "threads_text": "A Threads (threads.net / threads.com) text post with no video attachment",
    "threads_video": "A Threads post that contains a video (video_versions present or an embedded video)",
    "instagram_reel": "An Instagram Reel (instagram.com/reel/) — short video",
    "instagram_post": "An Instagram post (instagram.com/p/) — image or carousel, not a Reel",
    "substack_text": "A Substack article that is plain text, no embedded media",
    "substack_video": "A Substack post containing an embedded video",
    "substack_image": "A Substack post that is primarily an image",
    "mega_video": "A Mega.nz link to a video file (.mp4/.mkv/.webm/.mov)",
    "book": "A single book file (.epub/.pdf/.mobi/.azw3)",
    "book_batch": "A book archive or multiple books (zip/rar/7z or many book files)",
    "bunny_stream": "A TDAH conference talk (terragrandir.com or player.mediadelivery.net)",
    "wikispooks": "A Wikispooks.org wiki page",
    "rumble": "A Rumble video (rumble.com/v… or rumble.com/embed/…)",
    "web": "A generic web article or page from any other domain",
}


def route_jev(url: str, content: str | None = None, api_key: str | None = None) -> str:
    """Classify a URL (+ optional fetched content) via Jev.

    Args:
        url: The source URL.
        content: Optional fetched page content (HTML / filename list). Improves
            subtype accuracy for threads/substack/mega; omit for URL-only.
        api_key: Optional OpenRouter key override (defaults to OPENROUTER_API_KEY).

    Returns:
        The content type string (one of CRITERIA's keys).

    Raises:
        RuntimeError on any failure — the caller is expected to fall back to
        the legacy scrape.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    safe_url = url_for_display(url)
    state = f"URL: {safe_url}"
    if content:
        state += f"\nContent: {redact_credential_shapes(content)}"

    try:
        r = httpx.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "state": state,
                "questions": {
                    "content_type": {
                        "type": "choice",
                        "instructions": "What content type is this? Pick the single best match.",
                        "criteria": CRITERIA,
                    }
                },
            },
            timeout=90,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"OpenRouter request failed: {sanitize_visible_message(str(e))}"
        ) from e

    if r.status_code != 200:
        snippet = sanitize_visible_message(r.text[:120])
        raise RuntimeError(f"OpenRouter {r.status_code}: {snippet}")

    try:
        return r.json()["answers"]["content_type"]["choice"].lower().strip()
    except (KeyError, TypeError, ValueError) as e:
        snippet = sanitize_visible_message(r.text[:200])
        raise RuntimeError(f"Unexpected OpenRouter response: {snippet}") from e
