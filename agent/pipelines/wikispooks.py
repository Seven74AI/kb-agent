"""Wikispooks article pipeline — Firecrawl with DDOS bypass URL, then web deep treatment."""

from urllib.parse import quote, urlparse

from agent.config import Config
from agent.pipelines import Step, register_pipeline
from agent.pipelines.see_also import step_see_also
from agent.pipelines.web import (
    step_firecrawl_extract,
    step_git_push,
    step_minio_upload,
    step_quality_gate,
    step_synthesize_web,
)


def wikispooks_bypass_url(direct_url: str) -> str:
    """Transform a direct Wikispooks article URL into the DDOS bypass URL.

    Example:
        https://wikispooks.org/wiki/X/Y
        → https://wikispooks.org/w/welcome_back.php?goto=%2Fwiki%2FX%2FY
    """
    parsed = urlparse(direct_url)
    wiki_path = parsed.path
    if not wiki_path.startswith("/wiki/"):
        return direct_url
    encoded_path = quote(wiki_path, safe="")
    return f"{parsed.scheme}://{parsed.netloc}/w/welcome_back.php?goto={encoded_path}"


def step_wikispooks_firecrawl_extract(task: dict, config: Config) -> str:
    """Extract via Firecrawl using the bypass URL; note source_url stays direct."""
    bypass_url = wikispooks_bypass_url(task["source_url"])
    extract_task = {**task, "source_url": bypass_url}
    return step_firecrawl_extract(extract_task, config)


register_pipeline("wikispooks", [
    Step(name="firecrawl_extract", fn=step_wikispooks_firecrawl_extract, on_error="retry"),
    Step(name="synthesize_note", fn=step_synthesize_web, on_error="retry"),
    Step(name="quality_gate", fn=step_quality_gate, on_error="retry"),
    Step(name="minio_upload", fn=step_minio_upload, on_error="retry"),
    Step(name="see_also", fn=step_see_also, on_error="fail"),
    Step(name="git_push", fn=step_git_push, on_error="retry"),
])
