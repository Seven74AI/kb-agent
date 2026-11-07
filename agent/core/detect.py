"""Content type detection — platform + subtype without enqueue."""

from agent.detection import detect_content_subtype, detect_platform, normalize_url


def detect_type(url: str) -> dict:
    """Detect platform and content subtype for a URL.

    Returns:
        Dict with keys platform, content_type (subtype used for pipeline routing).
    """
    url = normalize_url((url or "").strip())
    platform = detect_platform(url)
    content_type = detect_content_subtype(url, platform)
    return {
        "platform": platform,
        "content_type": content_type,
    }
