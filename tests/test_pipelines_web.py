"""Tests for the web article pipeline — Firecrawl → LLM → quality → MinIO → Git."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add kb-agent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.exceptions import ErrorType, StepError
from agent.llm import LLMResponse
from agent.pipelines.web import (
    step_firecrawl_extract,
    step_synthesize_web,
    step_quality_gate,
    step_minio_upload,
    step_git_push,
    _estimate_tokens,
    _build_synthesis_prompt,
    _equal_chunks,
    CHUNK_TOKEN_THRESHOLD,
    CHARS_PER_TOKEN,
)
from agent.pipelines.see_also import step_see_also, _extract_keywords

TOKENIZED_TRACKING = "https://cdn.example.com/article?token=secret&expires=999"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Create a Config with temp paths for testing."""
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        article_min_quotes=2,  # Lower for testing
        article_min_size=100,   # Lower for testing
        firecrawl_url="http://localhost:3002/v2/scrape",
        minio_tailscale_base="http://minio.test/base",
    )


@pytest.fixture
def task_dict():
    """Create a sample task dict for web pipeline tests."""
    return {
        "id": "test-task-12345",
        "source_url": "https://example.com/article-about-ai",
        "content_type": "web",
        "slug": "example.com_article-about-ai",
    }


@pytest.fixture
def sample_source(config, task_dict):
    """Write a sample source file for testing."""
    source_path = Path(config.temp_dir) / f"{task_dict['slug']}_raw.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    content = """# The Future of AI in Healthcare

## Introduction
Artificial intelligence is transforming healthcare at an unprecedented pace.

> "AI will save millions of lives through early diagnosis" — Dr. Smith

## Current Applications
Today, AI is used in radiology, pathology, and drug discovery.

> "We've seen a 40% reduction in diagnostic errors" — Stanford study

## Challenges and Risks
However, challenges remain around data privacy and algorithmic bias.

> "Bias in training data leads to disparities in care" — Prof. Johnson

## Future Outlook
The next decade will see AI integrated into every aspect of healthcare.

> "By 2035, AI-assisted diagnosis will be the standard of care" — WHO report

## Conclusion
AI's potential in healthcare is vast, but careful implementation is essential.
"""
    source_path.write_text(content, encoding="utf-8")
    return str(source_path)


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_extract_rejects_metadata_ip(mock_run, config, task_dict):
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    task_dict["source_url"] = "http://169.254.169.254/latest/meta-data/"
    with pytest.raises(ValueError, match="not allowed"):
        step_firecrawl_extract(task_dict, config)
    mock_run.assert_not_called()


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_extract_curl_disables_redirects(mock_run, config, task_dict, tmp_path):
    """Firecrawl client curl must not follow redirects (SSRF policy)."""
    config.temp_dir = tmp_path / "tmp"
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    long_markdown = "# Title\n\n" + ("Content paragraph. " * 80)
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"data": {"markdown": long_markdown}}),
        stderr="",
    )
    step_firecrawl_extract(task_dict, config)
    cmd = mock_run.call_args[0][0]
    assert "--max-redirs" in cmd
    assert cmd[cmd.index("--max-redirs") + 1] == "0"


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_extract_curl_failure_raises_step_error_retry(
    mock_run, config, task_dict,
):
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="connection reset")
    with pytest.raises(StepError) as exc:
        step_firecrawl_extract(task_dict, config)
    assert exc.value.error_type == ErrorType.RETRY
    assert exc.value.step_name == "firecrawl_extract"
    assert "Firecrawl curl failed" in str(exc.value)


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_extract_short_content_raises_step_error_fail(
    mock_run, config, task_dict, tmp_path,
):
    config.temp_dir = tmp_path / "tmp"
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"data": {"markdown": "# Hi\n\nshort"}}),
        stderr="",
    )
    with pytest.raises(StepError) as exc:
        step_firecrawl_extract(task_dict, config)
    assert exc.value.error_type == ErrorType.FAIL
    assert "Content too short" in str(exc.value)


@patch("agent.pipelines.web.subprocess.run")
def test_firecrawl_extract_redacts_tracking_urls_in_raw_md(
    mock_run, config, task_dict, tmp_path,
):
    config.temp_dir = tmp_path / "tmp"
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    tracked = "https://partner.example/out?click_id=SECRET&utm_source=mail"
    long_markdown = (
        "# Article\n\n"
        f"Read more at {tracked}.\n\n"
        + ("Paragraph with body text. " * 40)
    )
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"data": {"markdown": long_markdown}}),
        stderr="",
    )
    out_path = step_firecrawl_extract(task_dict, config)
    body = Path(out_path).read_text(encoding="utf-8")
    assert "click_id=SECRET" not in body
    assert "utm_source" not in body
    assert "https://partner.example/out" in body


def test_enqueue_embedded_media_logs_submit_failures(config, caplog):
    import logging

    from agent.pipelines.web import _enqueue_embedded_media

    caplog.set_level(logging.WARNING, logger="agent.pipelines.web")

    with patch("agent.core.submit.submit_ingest", side_effect=RuntimeError("db locked")):
        with patch("agent.db.init_db") as mock_init:
            conn = MagicMock()
            conn.execute.return_value.fetchone.return_value = {"n": 0}
            mock_init.return_value = conn
            stats = _enqueue_embedded_media(
                "parent_slug",
                ["https://example.com/img.png"],
                [],
                [],
                [],
                "https://example.com/post",
                config,
                None,
            )

    assert stats.success_count == 0
    warns = [r for r in caplog.records if "embedded media enqueue failed" in r.message]
    assert len(warns) == 1
    assert warns[0].exc_info is not None


def test_enqueue_embedded_media_logs_redact_image_url_query(config, caplog):
    import logging

    from agent.pipelines.web import _enqueue_embedded_media

    caplog.set_level(logging.WARNING, logger="agent.pipelines.web")
    sensitive_img = "https://cdn.substack.com/img.png?token=leak&X-Amz-Signature=abc"

    with patch("agent.core.submit.submit_ingest", side_effect=RuntimeError("db locked")):
        with patch("agent.db.init_db") as mock_init:
            conn = MagicMock()
            conn.execute.return_value.fetchone.return_value = {"n": 0}
            mock_init.return_value = conn
            _enqueue_embedded_media(
                "parent_slug",
                [sensitive_img],
                [],
                [],
                [],
                "https://example.com/post",
                config,
                None,
            )

    combined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "leak" not in combined
    assert "X-Amz-Signature" not in combined
    assert "cdn.substack.com/img.png" in combined


def test_enqueue_embedded_media_logs_redact_rumble_url_query(config, caplog):
    import logging

    from agent.pipelines.web import _enqueue_embedded_media

    caplog.set_level(logging.WARNING, logger="agent.pipelines.web")
    sensitive_rumble = "https://rumble.com/vabc-post?key=secret&ref=track"

    with patch("agent.core.submit.submit_ingest", side_effect=RuntimeError("db locked")):
        with patch("agent.db.init_db") as mock_init:
            conn = MagicMock()
            conn.execute.return_value.fetchone.return_value = {"n": 0}
            mock_init.return_value = conn
            _enqueue_embedded_media(
                "parent_slug",
                [],
                [],
                [sensitive_rumble],
                [],
                "https://example.com/post",
                config,
                None,
            )

    combined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "secret" not in combined
    assert "key=" not in combined
    assert "rumble.com/vabc-post" in combined


# ── _estimate_tokens tests ────────────────────────────────────────────────────

def test_estimate_tokens_empty():
    assert _estimate_tokens("") == 0


def test_estimate_tokens_english():
    # ~4 chars per token for English
    text = "Hello world, this is a test."  # 30 chars → ~7 tokens
    assert _estimate_tokens(text) == 7


def test_estimate_tokens_long():
    text = "a" * 10_000  # 10K chars → ~2.5K tokens
    assert _estimate_tokens(text) == 2500


# ── _equal_chunks tests ───────────────────────────────────────────────────────

def test_equal_chunks_basic():
    source = "para1\n\npara2\n\npara3\n\npara4\n\n"
    chunks = _equal_chunks(source, 2)
    assert len(chunks) == 2
    assert chunks[0][0] == 0
    assert chunks[1][1] == len(source)


def test_equal_chunks_coverage():
    source = "x\n\n" * 100
    chunks = _equal_chunks(source, 3)
    assert len(chunks) == 3
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(source)
    # All chunks should be non-empty
    for start, end in chunks:
        assert end > start


# ── _extract_keywords tests ───────────────────────────────────────────────────

def test_extract_keywords_from_headers():
    note = """# Test Note
## Summary
A summary.
## Current Applications
This section covers current uses.
## Challenges and Risks
This section covers risks.
### 1. AI Diagnostic Tools
Details about diagnostic tools.
### 2. Data Privacy
Privacy concerns in healthcare AI.
## Critical Analysis
Some analysis.
"""
    keywords = _extract_keywords(note)
    # Template headers should be excluded
    assert "Summary" not in keywords
    assert "Critical Analysis" not in keywords
    # Content headers should be included
    assert "Current Applications" in keywords
    assert "Challenges and Risks" in keywords
    # Key Claim sub-headers should be included
    assert "AI Diagnostic Tools" in keywords
    assert "Data Privacy" in keywords


def test_extract_keywords_max_limit():
    note = "\n".join(f"### {i}. Topic {i}" for i in range(20))
    note = "# Note\n" + note
    keywords = _extract_keywords(note, max_keywords=5)
    assert len(keywords) <= 5


# ── _build_synthesis_prompt tests ─────────────────────────────────────────────

def test_build_synthesis_prompt_structure(config):
    prompt = _build_synthesis_prompt(
        "https://example.com/article",
        "example_slug",
        config,
    )
    # Must include key sections
    assert "knowledge base curator" in prompt.lower()
    assert "QUALITY RULES" in prompt
    assert "Key Claims" in prompt
    assert "Section-by-Section" in prompt
    assert "Critical Analysis" in prompt
    assert "Nuances" in prompt
    # Must include frontmatter fields
    assert "date:" in prompt
    assert "source_url:" in prompt
    assert "confidence:" in prompt
    assert "tags:" in prompt
    # Date must be current
    from datetime import datetime
    assert datetime.now().strftime("%Y-%m-%d") in prompt


def test_build_synthesis_prompt_includes_source_domain(config):
    prompt = _build_synthesis_prompt(
        "https://stripe.com/blog/engineering-culture",
        "stripe_blog",
        config,
    )
    assert "stripe.com" in prompt


def test_build_synthesis_prompt_uses_display_safe_source_url(config):
    tokenized = "https://cdn.example.com/article?token=secret"
    prompt = _build_synthesis_prompt(tokenized, "cdn_article", config)
    assert "source_url: https://cdn.example.com/article" in prompt
    assert "token=secret" not in prompt


# ── step_synthesize_web tests ─────────────────────────────────────────────────

def test_step_synthesize_web_redacts_tracking_urls_in_body(
    config, task_dict, sample_source,
):
    """Synthesize vault writes must redact http(s) URLs in LLM body text, not only frontmatter."""
    mock_response = LLMResponse(
        content=f"""---
date: 2026-06-19
source: example.com
source_url: {TOKENIZED_TRACKING}
confidence: plausible
tags: [web]
---

# Article

## Summary
Full story at {TOKENIZED_TRACKING}.

## Key Claims
### 1. Claim
> "Quote" — source

## Section-by-Section Analysis
### Intro
Context.

## Context
Published 2026.

## Critical Analysis
N/A

## Nuances
N/A

## Reliability
plausible

## Sources
- {TOKENIZED_TRACKING}

## See Also
- (filled later)
""",
        tokens_in=100,
        tokens_out=80,
        model="test-model",
    )

    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        note_path = step_synthesize_web(task_dict, config)

    written = Path(note_path).read_text(encoding="utf-8")
    assert "token=secret" not in written
    assert "https://cdn.example.com/article" in written


def test_step_synthesize_web_single_pass(config, task_dict, sample_source):
    """Test synthesis with a small source (single pass)."""
    # Mock the LLM client
    mock_response = LLMResponse(
        content="""---
date: 2026-06-19
source: example.com
source_url: https://example.com/article-about-ai
confidence: plausible
tags: [ai, healthcare]
---

# AI in Healthcare

## Summary
AI is transforming healthcare through early diagnosis and improved accuracy.

## Key Claims
### 1. AI reduces diagnostic errors
> "We've seen a 40% reduction in diagnostic errors" — Stanford study

Analysis: This claim is supported by empirical evidence.

### 2. AI will be standard of care by 2035
> "By 2035, AI-assisted diagnosis will be the standard of care" — WHO report

Analysis: WHO projection based on current trends.

## Section-by-Section Analysis
### Introduction
Sets up the promise of AI in healthcare.
### Current Applications
Covers radiology, pathology, and drug discovery with specific examples.
### Challenges and Risks
Addresses data privacy and algorithmic bias concerns.
### Future Outlook
Projects AI integration across all healthcare by 2035.

## Context
Written by healthcare technology researchers. Published in 2026.

## Critical Analysis
The article focuses on benefits while downplaying implementation costs.
The bias discussion is surface-level.

## Nuances
Omits discussion of regulatory hurdles. Overestimates near-term adoption.

## Reliability
⚠️ plausible — well-sourced but optimistic bias.

## Sources
- Original source: example.com/article-about-ai

## See Also
- (filled by See Also step)
""",
        tokens_in=500,
        tokens_out=400,
        model="deepseek-v4-pro",
    )

    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        note_path = step_synthesize_web(task_dict, config)

    # Verify note was written
    assert Path(note_path).exists()
    content = Path(note_path).read_text()
    assert "AI in Healthcare" in content
    assert "> \"We've seen a 40%" in content
    assert "Key Claims" in content
    assert "Section-by-Section" in content


def test_step_synthesize_web_missing_source(config, task_dict):
    """Test synthesis fails when source file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        step_synthesize_web(task_dict, config)


def test_step_synthesize_web_large_content_triggers_chunking(config, task_dict):
    """Test that large content triggers chunking."""
    # Create a source that's over 50K tokens
    large_source = ("x" * (CHUNK_TOKEN_THRESHOLD + 1) * CHARS_PER_TOKEN)

    source_path = Path(config.temp_dir) / f"{task_dict['slug']}_raw.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(large_source, encoding="utf-8")

    mock_response = LLMResponse(
        content="# Test Note\n\n## Summary\nTest.",
        tokens_in=100,
        tokens_out=50,
        model="deepseek-v4-pro",
    )

    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Patch _find_semantic_boundaries to return chunks
        with patch("agent.pipelines.web._find_semantic_boundaries") as mock_boundaries:
            mock_boundaries.return_value = [
                (0, len(large_source) // 2),
                (len(large_source) // 2, len(large_source)),
            ]

            note_path = step_synthesize_web(task_dict, config)

    assert Path(note_path).exists()
    # Verify chunking was triggered (multiple LLM calls: 2 chunks + polish)
    assert mock_llm.call.call_count >= 2


# ── step_quality_gate tests ───────────────────────────────────────────────────

def test_step_quality_gate_passes(config, task_dict):
    """Test quality gate passes on a well-formed note."""
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{task_dict['slug']}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("""---
date: 2026-06-19
source: example.com
source_url: https://example.com/article
---

# Test Note

## Summary
A comprehensive analysis of the article.

## Key Claims
### 1. Claim One
> "Direct quote from source one"

Analysis of claim one with multiple sentences.

### 2. Claim Two
> "Direct quote from source two"

Analysis of claim two.

## Section-by-Section Analysis
### Section 1: Introduction
This section introduces the topic with deep analysis and multiple paragraphs
that demonstrate thorough understanding of the source material presented.
The author makes several key points worth examining in detail here.
The opening establishes the central thesis and provides necessary background
for readers unfamiliar with the subject matter. This thorough introduction
sets the stage for the more detailed analysis that follows in later sections.

### Section 2: Body
The body covers the main arguments with substantial treatment of each claim.
This section is longer than any other section in the note to ensure
the dominant section requirement is met with ample analysis and detail.
Each argument is examined from multiple angles, with supporting evidence
and counterarguments considered in turn. The analysis is comprehensive
and leaves no major point unaddressed.

### Section 3: Evidence Review
The evidence section examines the data supporting the main claims.
Multiple studies are cited and their methodologies are evaluated.
The strength of each piece of evidence is assessed independently,
giving readers a clear picture of what is well-supported and what
requires further investigation.

### Section 4: Conclusion
The conclusion wraps up with final thoughts and synthesizes the
key takeaways from the analysis. It identifies open questions
and suggests directions for future research or policy action.

## Context
Written by experts in the field. Published in 2026.

## Critical Analysis
Some critique but shorter than Section-by-Section.

### Methodological Issues
- The sample size could be larger.

### Factual Issues
- Claim X on page 2 contradicts research by Smith (2020).
- The timeline on page 5 is off by two years.

### What Remains Interesting / Valid
- The core argument stands despite these issues.

## Nuances
Some limitations noted.

## Reliability
⚠️ plausible with justification.
""")

    mock_response = LLMResponse(content="pass", tokens_in=50, tokens_out=5, model="test")

    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = step_quality_gate(task_dict, config)

    assert result == "passed"


def test_step_quality_gate_missing_note(config, task_dict):
    """Test quality gate fails when note doesn't exist."""
    with pytest.raises(FileNotFoundError):
        step_quality_gate(task_dict, config)


def test_step_quality_gate_too_few_quotes(config, task_dict):
    """Test quality gate fails on note with too few quotes."""
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{task_dict['slug']}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("""---
date: 2026-06-19
source: example.com
---

# Test Note

## Summary
Brief summary without quotes.

## Key Claims
Just one claim here.

## Section-by-Section Analysis
### Section 1
Content without any quotes at all.
""")

    mock_response = LLMResponse(content="fail: too few quotes", tokens_in=50, tokens_out=10, model="test")

    with patch("agent.pipelines.web.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        with pytest.raises(StepError, match="Quality gate failed"):
            step_quality_gate(task_dict, config)


# ── Pipeline registration tests ───────────────────────────────────────────────

def test_pipeline_registered():
    """Test that the web pipeline is registered with correct steps."""
    from agent.pipelines import get_pipeline

    pipeline = get_pipeline("web")
    assert pipeline is not None
    step_names = [s.name for s in pipeline]
    assert "firecrawl_extract" in step_names
    assert "synthesize_note" in step_names
    assert "quality_gate" in step_names
    assert "minio_upload" in step_names
    assert "see_also" in step_names
    assert "git_push" in step_names
    # Verify correct order
    assert step_names.index("synthesize_note") < step_names.index("quality_gate")
    assert step_names.index("quality_gate") < step_names.index("minio_upload")
    assert step_names.index("minio_upload") < step_names.index("git_push")


def test_pipeline_unknown_content_type_returns_none():
    """Unknown content types must not silently use the web pipeline."""
    from agent.pipelines import get_pipeline

    pipeline = get_pipeline("nonexistent")
    assert pipeline is None
    web_pipeline = get_pipeline("web")
    assert web_pipeline is not None
    assert len(web_pipeline) > 0


# ── step_see_also tests ───────────────────────────────────────────────────────

def test_step_see_also_no_vault(config, task_dict):
    """Test see_also when vault doesn't exist yet."""
    # Don't create vault directory
    result = step_see_also(task_dict, config)
    assert "No related notes found" in result or "No note to add" in result


def test_step_see_also_empty_vault(config, task_dict):
    """Test see_also with an empty vault."""
    # Create note but no other notes in vault
    note_path = config.vault_path / "Knowledge base" / "articles" / f"{task_dict['slug']}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("""---
date: 2026-06-19
source: example.com
source_url: https://example.com/article
---

# Test Note

## Summary
A note about AI and healthcare.

## Key Claims
### 1. AI improves diagnosis
> "AI improves diagnosis accuracy"

## See Also
""")

    mock_response = LLMResponse(
        content="## See Also\n_No related notes found._",
        tokens_in=50,
        tokens_out=20,
        model="test",
    )

    with patch("agent.pipelines.see_also.get_llm_client") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.call.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = step_see_also(task_dict, config)

    assert "0" in result or "no related" in result.lower()


# ── step_minio_upload tests ───────────────────────────────────────────────────

def test_step_minio_upload_missing_source(config, task_dict):
    """Test minio upload fails when source file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        step_minio_upload(task_dict, config)


def test_step_minio_upload_with_mock(config, task_dict):
    """Test minio upload with mocked subprocess — placeholder replaced with real URL."""
    source_path = Path(config.temp_dir) / f"{task_dict['slug']}_raw.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Test source content", encoding="utf-8")

    with patch("agent.archival.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create the note with a minio: placeholder (real-world LLM output)
        note_path = config.vault_path / "Knowledge base" / "articles" / f"{task_dict['slug']}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("""---
date: 2026-06-19
source: test
minio: http://minio.test/base/knowledge-base/placeholder/slug.md
---

# Test

## Sources
- Archived at minio: <placeholder>
""")

        result = step_minio_upload(task_dict, config)

    assert result.startswith("articles/")

    # Verify the placeholder was replaced with the real MinIO URL in frontmatter
    updated_content = note_path.read_text(encoding="utf-8")
    assert result in updated_content, (
        f"Real MinIO URL ({result}) not found in note frontmatter. "
        f"Placeholder was not replaced."
    )
    # The minio: field in frontmatter must point to the object key (not the placeholder path)
    assert "articles/" in updated_content, (
        "MinIO key should contain the real folder path (articles/...), not the template placeholder"
    )


# ── step_git_push tests ───────────────────────────────────────────────────────

def test_step_git_push_missing_note(config, task_dict):
    """Test git push fails when note doesn't exist."""
    with pytest.raises(FileNotFoundError):
        step_git_push(task_dict, config)


# ── Integration: end-to-end pipeline steps ────────────────────────────────────

def test_full_pipeline_order():
    """Test that pipeline steps are in the correct execution order."""
    from agent.pipelines import get_pipeline

    pipeline = get_pipeline("web")
    step_names = [s.name for s in pipeline]

    # Extraction must come first
    assert step_names[0] == "firecrawl_extract"
    # Synthesis before quality gate
    assert step_names.index("synthesize_note") < step_names.index("quality_gate")
    # Quality gate before archival
    assert step_names.index("quality_gate") < step_names.index("minio_upload")
    # MinIO before Git (so minio URL can be added to note)
    assert step_names.index("minio_upload") < step_names.index("git_push")
    # See Also before Git (so links are committed)
    assert step_names.index("see_also") < step_names.index("git_push")
    # Git push is the final step
    assert step_names[-1] == "git_push"
