"""Pipeline step lists — content type → ordered mechanical + synthesis steps."""

from __future__ import annotations

# Canonical step lists (today duplicated from agent.webhook.PIPELINE_STEPS).
# Adapters and the async consumer will converge on this module in a follow-up migration.

PIPELINE_STEPS: dict[str, list[str]] = {
    "web": [
        "firecrawl_extract", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "youtube": [
        "extract_metadata", "download_video", "extract_audio",
        "diarize", "transcribe", "merge_and_identify",
        "cleanup_temp", "synthesize_youtube",
        "quality_gate", "minio_upload", "git_push",
    ],
    "instagram_reel": [
        "ig_reel_validate", "ig_reel_download", "ig_reel_audio",
        "ig_reel_diarize", "ig_reel_transcribe", "ig_reel_merge",
        "ig_reel_synthesize", "ig_quality_gate", "ig_cleanup",
        "ig_minio_upload", "ig_git_push",
    ],
    "instagram_post": [
        "ig_reel_validate", "ig_carousel_metadata", "ig_carousel_extract",
        "ig_carousel_synthesize", "ig_quality_gate",
        "ig_minio_upload", "ig_git_push",
    ],
    "threads_text": [
        "threads_detect", "threads_extract_text", "threads_text_synthesize",
        "threads_quality_gate", "threads_minio_upload", "threads_git_push",
    ],
    "threads_video": [
        "threads_detect", "threads_extract_video", "threads_extract_audio",
        "threads_diarize", "threads_transcribe", "threads_merge",
        "threads_video_synthesize", "threads_quality_gate", "threads_cleanup",
        "threads_minio_upload", "threads_git_push",
    ],
    "substack": [
        "firecrawl_extract", "substack_cleanup", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "substack_text": [
        "firecrawl_extract", "substack_cleanup", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "substack_video": [
        "substack_video_pause",
    ],
    "substack_image": [
        "firecrawl_extract", "substack_image_extract", "synthesize_note", "quality_gate",
        "minio_upload", "see_also", "git_push",
    ],
    "book": [
        "extract_text", "synthesize_note", "quality_gate",
        "minio_upload", "git_push",
    ],
    "book_epub": [
        "extract_text", "synthesize_note", "quality_gate",
        "minio_upload", "git_push",
    ],
    "book_pdf": [
        "extract_text", "synthesize_note", "quality_gate",
        "minio_upload", "git_push",
    ],
    "book_batch": [
        "megadl_download", "classify_content", "delegate",
    ],
    "mega": [
        "megadl_download", "classify_content", "delegate",
    ],
}


def resolve_pipeline_steps(content_type: str, platform: str) -> list[str]:
    """Return the ordered step names for a content type / platform pair."""
    return PIPELINE_STEPS.get(
        content_type,
        PIPELINE_STEPS.get(platform, PIPELINE_STEPS["web"]),
    )
