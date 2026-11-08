"""Tests for the book pipeline — Mega.nz, ePub/PDF, 3-tier pre-flight,
chapter-level synthesis, quality gates, MinIO, Git."""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Add kb-agent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.detection import _classify_mega_filenames
from agent.db import AGENT_SCHEMA, create_task, init_db
from agent.exceptions import ErrorType, StepError
from agent.pipelines.books import (
    CLASSIFY_BOOK_PIPELINES,
    _book_child_slug,
    _chunked_book_synthesis,
    _detect_boundaries_llm,
    _resolve_local_book_path,
    _vault_frontmatter_source_url,
    _pdf_preflight,
    quality_score,
    step_extract_epub,
    step_extract_pdf,
    step_extract_mobi,
    step_extract_pdf_degraded,
    step_extract_pdf_scanned,
    step_classify_content,
    step_delegate_books,
    step_book_quality_gate,
    step_synthesize_book,
    step_megadl_download,
    step_book_minio_upload,
    step_book_git_push,
    _build_book_prompt,
    _today,
    BOOK_EXTENSIONS,
    PDF_MIN_CHARS_SCANNED,
    PDF_QUALITY_SCORE_THRESHOLD,
    CHARS_PER_TOKEN,
)
from agent.quality import quality_gate_book


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Create a Config with temp paths for testing."""
    return Config(
        kb_agent_dir=Path(__file__).resolve().parents[1],
        vault_path=tmp_path / "vault",
        temp_dir=tmp_path / "tmp",
        cookie_dir=tmp_path / "cookies",
        book_min_lines=10,
        book_min_quotes=2,
        book_min_size=100,
    )


@pytest.fixture
def book_task():
    """Create a sample book task dict."""
    return {
        "id": "test-book-abc123",
        "source_url": "/tmp/test_book.epub",
        "content_type": "book",
        "slug": "test_book_slug",
    }


@pytest.fixture
def mega_task():
    """Create a sample Mega task dict."""
    return {
        "id": "test-mega-abc123",
        "source_url": "https://mega.nz/file/abc123#key",
        "content_type": "mega",
        "slug": "mega_nz_file_abc123",
    }


METADATA_URL = "http://169.254.169.254/latest/meta-data/"


class TestBooksMegadlSsrf:

    @patch("agent.pipelines.books.subprocess.run")
    def test_megadl_download_rejects_metadata_ip(self, mock_run, config, mega_task):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        mega_task["source_url"] = METADATA_URL
        from agent.exceptions import StepError

        with pytest.raises(StepError, match="not allowed"):
            step_megadl_download(mega_task, config)
        mock_run.assert_not_called()


# ── Detection Tests ───────────────────────────────────────────────────────────

class TestMegaFilenamesClassification:
    """Tests for _classify_mega_filenames."""

    def test_single_epub_returns_book(self):
        """Single .epub file → 'book'."""
        result = _classify_mega_filenames(["my-book.epub"])
        assert result == "book"

    def test_single_pdf_returns_book(self):
        """Single .pdf file → 'book'."""
        result = _classify_mega_filenames(["document.pdf"])
        assert result == "book"

    def test_single_zip_returns_book_batch(self):
        """Single .zip archive → 'book_batch'."""
        result = _classify_mega_filenames(["books-pack.zip"])
        assert result == "book_batch"

    def test_single_rar_returns_book_batch(self):
        """Single .rar archive → 'book_batch'."""
        result = _classify_mega_filenames(["collection.rar"])
        assert result == "book_batch"

    def test_majority_epub_returns_book_batch(self):
        """Multiple books → 'book_batch'."""
        result = _classify_mega_filenames([
            "book1.epub", "book2.epub", "book3.pdf", "notes.txt"
        ])
        assert result == "book_batch"

    def test_majority_video_returns_mega_video(self):
        """Multiple videos → 'mega_video'."""
        result = _classify_mega_filenames([
            "movie.mp4", "trailer.mkv", "readme.txt"
        ])
        assert result == "mega_video"

    def test_single_video_returns_mega_video(self):
        """Single .mp4 → 'mega_video'."""
        result = _classify_mega_filenames(["lecture.mp4"])
        assert result == "mega_video"

    def test_unknown_extension_returns_unknown(self):
        """Unknown extension → 'unknown'."""
        result = _classify_mega_filenames(["data.bin", "stuff.dat"])
        assert result == "unknown"

    def test_empty_list_returns_unknown(self):
        """Empty list → 'unknown'."""
        result = _classify_mega_filenames([])
        assert result == "unknown"

    def test_archive_priority_over_books(self):
        """Archive + books → 'book_batch' (archive takes priority)."""
        result = _classify_mega_filenames([
            "archive.zip", "book1.epub", "book2.pdf"
        ])
        assert result == "book_batch"

    def test_mixed_books_and_videos_returns_unknown(self):
        """Mixed books and videos with no majority → 'unknown'."""
        result = _classify_mega_filenames([
            "book.epub", "movie.mp4"
        ])
        assert result == "unknown"

    def test_case_insensitive_extensions(self):
        """Extensions are case-insensitive."""
        result = _classify_mega_filenames(["Book.EPUB"])
        assert result == "book"


# ── Quality Score Tests ───────────────────────────────────────────────────────

class TestQualityScore:
    """Tests for quality_score and _pdf_preflight functions."""

    def test_quality_score_returns_int(self):
        """quality_score returns an integer 0-100."""
        score = quality_score("/nonexistent/path.pdf")
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_quality_score_handles_missing_file(self):
        """Unreadable/missing PDF must not score as perfect quality."""
        score = quality_score("/tmp/definitely_not_a_real_file_12345.pdf")
        assert isinstance(score, int)
        assert 0 <= score <= 100
        assert score != 100
        assert score < PDF_QUALITY_SCORE_THRESHOLD

    def test_quality_score_handles_non_pdf(self):
        """Non-PDF paths must not score as perfect quality."""
        score = quality_score("/tmp/not_a_pdf.txt")
        assert isinstance(score, int)
        assert 0 <= score <= 100
        assert score != 100
        assert score < PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_quality_score_open_error_not_perfect(self, mock_pymupdf):
        """Corrupt/unopenable PDFs must not score as perfect quality."""
        mock_pymupdf.open.side_effect = RuntimeError("cannot open document")
        score = quality_score("/fake/bad.pdf")
        assert score != 100
        assert score < PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_quality_score_clean_pdf_still_scores_high(self, mock_pymupdf):
        """Valid PDFs still get normal quality scores."""
        mock_doc = MagicMock()
        mock_doc.page_count = 20
        clean_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal sentence with standard "
            "English text and common punctuation marks. "
            * 5
        )
        mock_doc.__getitem__.return_value.get_text.return_value = clean_text
        mock_pymupdf.open.return_value = mock_doc

        score = quality_score("/fake/clean.pdf")
        assert score >= PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_scanned_detection(self, mock_pymupdf):
        """PDF with < 500 chars → 'scanned'."""
        # Mock a document with very little text
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.__getitem__.return_value.get_text.return_value = ""  # 0 chars per page
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/scanned.pdf"))
        assert classification == "scanned"
        assert score == 0

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_clean_text(self, mock_pymupdf):
        """PDF with high-quality text → 'text' with high score."""
        mock_doc = MagicMock()
        mock_doc.page_count = 20
        # Return clean, normal text for each page
        clean_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal sentence with standard "
            "English text and common punctuation marks. "
            * 5  # Enough chars to pass tier 1
        )
        mock_doc.__getitem__.return_value.get_text.return_value = clean_text
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/clean.pdf"))
        assert classification == "text"
        assert score >= PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_degraded_ocr(self, mock_pymupdf):
        """PDF with degraded OCR → 'degraded' with low score."""
        mock_doc = MagicMock()
        mock_doc.page_count = 16
        # Simulate degraded OCR with long-s patterns
        degraded_text = (
            "The fafe and fubject of this fection is the firft "
            "attempt at defcribing the fubject. %J. C  A  "
            * 20  # Enough chars but garbled
        )
        mock_doc.__getitem__.return_value.get_text.return_value = degraded_text
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/degraded.pdf"))
        assert classification == "degraded"
        assert score < PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_open_error_returns_degraded(self, mock_pymupdf):
        """Corrupt/password PDFs must not be classified as clean text."""
        mock_pymupdf.open.side_effect = RuntimeError("cannot open encrypted pdf")
        classification, score = _pdf_preflight(Path("/fake/bad.pdf"))
        assert classification == "degraded"
        assert score == 0

    @patch("agent.pipelines.books.HAS_PYMUPDF", False)
    def test_pdf_preflight_without_pymupdf_fail_closed(self):
        """Missing pymupdf must not assume clean text."""
        classification, score = _pdf_preflight(Path("/fake/any.pdf"))
        assert classification == "degraded"
        assert score == 0

    @patch("agent.pipelines.books.HAS_PYMUPDF", False)
    def test_quality_score_without_pymupdf_returns_zero(self):
        """Missing pymupdf must not score as perfect quality."""
        assert quality_score("/fake/any.pdf") == 0

    @patch("agent.pipelines.books.pymupdf")
    def test_quality_score_all_page_sampling_failures_returns_zero(
        self, mock_pymupdf,
    ):
        """When every sampled page fails text extraction, score is 0."""
        mock_doc = MagicMock()
        mock_doc.page_count = 16

        def page_at(idx):
            page = MagicMock()
            page.get_text.side_effect = RuntimeError("corrupt page")
            return page

        mock_doc.__getitem__.side_effect = page_at
        mock_pymupdf.open.return_value = mock_doc

        assert quality_score("/fake/all-bad-pages.pdf") == 0

    @patch("agent.pipelines.books.pymupdf")
    def test_quality_score_zero_pages_after_open_returns_zero(self, mock_pymupdf):
        """PDF with no pages must not score as perfect quality."""
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_pymupdf.open.return_value = mock_doc

        assert quality_score("/fake/empty.pdf") == 0

    @patch("agent.pipelines.books.pymupdf")
    def test_quality_score_partial_page_failures_not_high(self, mock_pymupdf):
        """Partially corrupt PDFs must not score as good text."""
        mock_doc = MagicMock()
        mock_doc.page_count = 16
        clean_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal sentence with standard "
            "English text and common punctuation marks. "
            * 5
        )

        def page_at(idx):
            page = MagicMock()
            if idx % 2 == 0:
                page.get_text.side_effect = RuntimeError("corrupt page")
            else:
                page.get_text.return_value = clean_text
            return page

        mock_doc.__getitem__.side_effect = page_at
        mock_pymupdf.open.return_value = mock_doc

        score = quality_score("/fake/partially-corrupt.pdf")
        assert score < PDF_QUALITY_SCORE_THRESHOLD

    @patch("agent.pipelines.books.pymupdf")
    def test_pdf_preflight_partial_page_failures_not_text_tier(
        self, mock_pymupdf,
    ):
        """Partially corrupt PDFs must not be classified as clean text."""
        mock_doc = MagicMock()
        mock_doc.page_count = 16
        clean_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal sentence with standard "
            "English text and common punctuation marks. "
            * 5
        )

        def page_at(idx):
            page = MagicMock()
            if idx % 2 == 0:
                page.get_text.side_effect = RuntimeError("corrupt page")
            else:
                page.get_text.return_value = clean_text
            return page

        mock_doc.__getitem__.side_effect = page_at
        mock_pymupdf.open.return_value = mock_doc

        classification, score = _pdf_preflight(Path("/fake/partially-corrupt.pdf"))
        assert classification == "degraded"
        assert score < PDF_QUALITY_SCORE_THRESHOLD


# ── ePub Extraction Tests ─────────────────────────────────────────────────────

class TestEpubExtraction:
    """Tests for step_extract_epub."""

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_basic(self, mock_bs, mock_epub, config, book_task):
        """Basic ePub extraction with mocked ebooklib."""
        # Create a dummy epub file so the file-existence check passes
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        # Mock the book
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        # Create mock document items with unique soups
        soups = []
        items = []
        chapters = [
            "Chapter 1: Introduction\nThis is the first chapter with enough text to pass the threshold.",
            "Chapter 2: Methods\nThis is the second chapter with detailed methodology description.",
            "Chapter 3: Results\nThis is the third chapter presenting the key findings and analysis.",
        ]
        for body_text in chapters:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items

        # Mock metadata
        mock_book.get_metadata.return_value = [("Test Book",)]

        # Ensure temp dir exists
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        data = json.loads(result)

        assert data["chapter_count"] == 3
        assert data["total_chars"] > 0
        assert "full_text_path" in data

        # Verify file was written
        full_text_path = Path(data["full_text_path"])
        assert full_text_path.exists()
        content = full_text_path.read_text()
        assert "CHAPTER 0" in content
        assert "CHAPTER 1" in content

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_preserves_order(self, mock_bs, mock_epub, config, book_task):
        """ePub extraction preserves chapter order (not size-sorted)."""
        # Create dummy epub file
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        # Chapter 1 is long, Chapter 2 is shorter but >50 chars
        soups = []
        items = []
        for body_text in ["A" * 500, "B" * 60]:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items
        mock_book.get_metadata.return_value = [("Test",)]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        full_text = Path(json.loads(result)["full_text_path"]).read_text()

        # Chapter 0 (A) should come before Chapter 1 (B*60)
        idx_a = full_text.index("CHAPTER 0")
        idx_b = full_text.index("CHAPTER 1")
        assert idx_a < idx_b, "Chapter order must be preserved"

    @patch("agent.pipelines.books.ITEM_DOCUMENT", 9)
    @patch("agent.pipelines.books.epub")
    @patch("agent.pipelines.books.BeautifulSoup")
    def test_extract_epub_skips_tiny_fragments(self, mock_bs, mock_epub, config, book_task):
        """Tiny fragments (< 50 chars) are skipped."""
        # Create dummy epub file
        epub_path = Path(config.temp_dir) / "test_book.epub"
        epub_path.parent.mkdir(parents=True, exist_ok=True)
        epub_path.write_text("dummy epub content")
        book_task["source_url"] = str(epub_path)
        mock_book = MagicMock()
        mock_epub.read_epub.return_value = mock_book
        mock_epub.ITEM_DOCUMENT = 9

        soups = []
        items = []
        for body_text in ["Tiny", "A" * 100]:
            soup = MagicMock()
            soup.get_text.return_value = body_text
            soups.append(soup)
            item = MagicMock()
            item.get_body_content.return_value = b"<p>test</p>"
            items.append(item)
        mock_bs.side_effect = soups
        mock_book.get_items_of_type.return_value = items
        mock_book.get_metadata.return_value = [("Test",)]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_epub(book_task, config)
        data = json.loads(result)
        assert data["chapter_count"] == 1  # Only the 100-char one


# ── PDF Extraction Tests ──────────────────────────────────────────────────────

class TestPdfExtraction:
    """Tests for step_extract_pdf."""

    @patch("agent.pipelines.books.pymupdf")
    def test_extract_pdf_basic(self, mock_pymupdf, config, book_task):
        """Basic PDF extraction with mocked pymupdf."""
        # Create dummy pdf file
        pdf_path = Path(config.temp_dir) / "test_book.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("dummy pdf content")
        book_task["source_url"] = str(pdf_path)

        # pymupdf.open is called twice (preflight + extraction), so return new doc each time
        text_per_page = (
            "Page content with enough characters to pass "
            "the scanned detection threshold of 500 chars. "
            * 5
        )

        def make_doc():
            mock_doc = MagicMock()
            mock_doc.page_count = 3
            # Make doc iterable over pages
            pages = []
            for _ in range(3):
                page = MagicMock()
                page.get_text.return_value = text_per_page
                pages.append(page)
            mock_doc.__iter__.return_value = iter(pages)
            mock_doc.__getitem__.return_value = pages[0]
            return mock_doc

        mock_pymupdf.open.side_effect = [make_doc(), make_doc()]

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        result = step_extract_pdf(book_task, config)
        data = json.loads(result)

        assert data["page_count"] == 3
        assert data["total_chars"] > 0
        assert "quality_score" in data

    @patch("agent.pipelines.books.pymupdf")
    def test_extract_pdf_redacts_urls_in_full_text_file(self, mock_pymupdf, config, book_task):
        tokenized = "https://cdn.example.com/ref?token=secret&expires=999"
        pdf_path = Path(config.temp_dir) / "test_book.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("dummy pdf content")
        book_task["source_url"] = str(pdf_path)
        base = (
            "Page content with enough characters to pass "
            "the scanned detection threshold of 500 chars. "
            * 5
        )
        text_per_page = f"{base} See {tokenized}"

        def make_doc():
            mock_doc = MagicMock()
            mock_doc.page_count = 3
            pages = []
            for _ in range(3):
                page = MagicMock()
                page.get_text.return_value = text_per_page
                pages.append(page)
            mock_doc.__iter__.return_value = iter(pages)
            mock_doc.__getitem__.side_effect = lambda i: pages[i]
            return mock_doc

        mock_pymupdf.open.side_effect = [make_doc(), make_doc(), make_doc()]
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
        step_extract_pdf(book_task, config)
        slug = book_task["slug"]
        full_txt = Path(config.temp_dir) / f"book_{slug}_full.txt"
        body = full_txt.read_text(encoding="utf-8")
        assert "token=secret" not in body
        assert "https://cdn.example.com/ref" in body

    @patch("agent.pipelines.books.pymupdf")
    def test_extract_pdf_scanned_raises(self, mock_pymupdf, config, book_task):
        """Scanned PDF raises StepError."""
        # Create dummy pdf file
        pdf_path = Path(config.temp_dir) / "test_book.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text("dummy pdf content")
        book_task["source_url"] = str(pdf_path)

        # Return a doc with 0 text → scanned
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value.get_text.return_value = ""  # 0 chars
        mock_pymupdf.open.return_value = mock_doc

        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

        from agent.exceptions import StepError
        with pytest.raises(StepError) as excinfo:
            step_extract_pdf(book_task, config)
        assert "needs OCR" in str(excinfo.value)


# ── Book Quality Gate Tests ───────────────────────────────────────────────────

class TestBookQualityGate:
    """Tests for step_book_quality_gate."""

    def test_quality_gate_passes_for_valid_note(self, config, book_task):
        """Quality gate passes for a valid book note."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"

        # Build a long enough book note without repetition
        # (repetition with *3 breaks section indexing)
        chapters = []
        for i in range(1, 11):
            chapters.append(f"### Chapter {i}")
            chapters.append(f"This is a detailed summary of chapter {i}. " * 10)

        note_content = "\n".join([
            "## Summary",
            "This is a comprehensive book summary that captures the central thesis.",
            "",
            "## Key Claims",
            '> "Direct quote one from the book" — Chapter 1',
            '> "Direct quote two proving a key point" — Chapter 2',
            '> "Direct quote three with evidence" — Chapter 5',
            '> "Direct quote four with analysis" — Chapter 8',
            "",
            "## Chapter Summaries",
            *chapters,
            "",
            "## Who Is the Author",
            "A detailed author biography with credentials and background information. " * 5,
            "",
            "## Critical Analysis",
            "### Methodological Issues",
            "- The study design lacks a control group.",
            "",
            "### Factual Issues",
            "- Claim X on page 42 contradicts research by Smith (2020).",
            "- The historical timeline on page 88 is off by two years.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands despite these issues.",
            "",
            "## Nuances",
            "Some nuanced points about where the book falls short.",
            "",
            "## Reliability",
            "plausible",
            "",
            "## Sources",
            "Author. *Title* (2024)",
        ])
        note_path.write_text(note_content)

        result = step_book_quality_gate(book_task, config)
        data = json.loads(result)
        assert data["passed"] is True

    def test_quality_gate_fails_too_few_quotes(self, config, book_task):
        """Quality gate fails when quote count < required."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"

        note_content = "\n".join([
            "No quotes here.",
            "Just text.",
        ] * 50)  # Enough lines but no quotes
        note_path.write_text(note_content)

        from agent.exceptions import StepError
        with pytest.raises(StepError) as excinfo:
            step_book_quality_gate(book_task, config)
        assert "Too few quotes" in str(excinfo.value)

    def test_quality_gate_fails_too_short(self, config, book_task):
        """Quality gate fails when note is too short."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"
        note_path.write_text("Too short.")

        from agent.exceptions import StepError
        with pytest.raises(StepError):
            step_book_quality_gate(book_task, config)

    def test_quality_gate_note_not_found(self, config, book_task):
        """Quality gate fails when note doesn't exist."""
        from agent.exceptions import StepError
        with pytest.raises(StepError):
            step_book_quality_gate(book_task, config)

    def test_quality_gate_fails_too_few_fact_checks(self, config, book_task):
        """Quality gate fails when fact_check_count < 2."""
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{book_task['slug']}.md"

        # Build a valid note that passes everything EXCEPT fact_checks
        chapters = []
        for i in range(1, 11):
            chapters.append(f"### Chapter {i}")
            chapters.append(f"This is a detailed summary of chapter {i}. " * 10)

        note_content = "\n".join([
            "## Summary",
            "This is a comprehensive book summary that captures the central thesis.",
            "",
            "## Key Claims",
            '> "Direct quote one from the book" — Chapter 1',
            '> "Direct quote two proving a key point" — Chapter 2',
            '> "Direct quote three with evidence" — Chapter 5',
            '> "Direct quote four with analysis" — Chapter 8',
            "",
            "## Chapter Summaries",
            *chapters,
            "",
            "## Who Is the Author",
            "A detailed author biography with credentials and background information. " * 5,
            "",
            "## Critical Analysis",
            "### Methodological Issues",
            "- The study design lacks a control group.",
            "",
            "### Factual Issues",
            "Only one fact check — the author got the date wrong.",
            "",
            "### What Remains Interesting / Valid",
            "- The core argument stands despite these issues.",
            "",
            "## Nuances",
            "Some nuanced points about where the book falls short.",
            "",
            "## Reliability",
            "plausible",
            "",
            "## Sources",
            "Author. *Title* (2024)",
        ])
        note_path.write_text(note_content)

        from agent.exceptions import StepError
        with pytest.raises(StepError) as excinfo:
            step_book_quality_gate(book_task, config)
        assert "Too few fact checks" in str(excinfo.value)

    def _minimal_passing_book_note_lines(self, chapter_headers: list[str]) -> list[str]:
        """Book note body that passes size/quote/line/fact-check gates in tests."""
        chapters = []
        for hdr in chapter_headers:
            chapters.extend([hdr, "Summary " * 30])
        return [
            "## Summary",
            "Central thesis " * 20,
            "",
            "## Key Claims",
            '> "Quote one" — Ch 1',
            '> "Quote two" — Ch 2',
            "",
            "## Chapter Summaries",
            *chapters,
            "",
            "## Critical Analysis",
            "### Factual Issues",
            "- Issue one.",
            "- Issue two.",
            "",
            "## Sources",
            "Author. *Title* (2024)",
        ]

    def test_quality_gate_chapter_coverage_uses_temp_dir_not_tmp(
        self, config, book_task, tmp_path,
    ):
        """Chapter coverage must read full text from config.temp_dir, not /tmp."""
        slug = book_task["slug"]
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{slug}.md"
        note_path.write_text("\n".join(self._minimal_passing_book_note_lines(["### Chapter 1"])))

        source_chapters = "\n".join(
            f"\n=== CHAPTER {i} ===\n" + ("x" * 200) for i in range(1, 9)
        )
        full_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(source_chapters, encoding="utf-8")
        # Stale path under /tmp must not drive coverage when temp_dir differs
        stale = tmp_path / "book_stale_full.txt"
        assert not stale.exists()

        passed, failures = quality_gate_book(
            note_path, config, vault_path=config.vault_path,
        )
        assert not passed
        assert any("Chapter coverage" in f for f in failures)

    def test_step_quality_gate_enforces_chapter_coverage(self, config, book_task):
        """step_book_quality_gate passes vault_path so coverage is enforced."""
        slug = book_task["slug"]
        note_dir = config.vault_path / "Knowledge base" / "books"
        note_dir.mkdir(parents=True, exist_ok=True)
        note_path = note_dir / f"{slug}.md"
        note_path.write_text("\n".join(self._minimal_passing_book_note_lines(["### Chapter 1"])))

        source = "\n".join(f"=== CHAPTER {i} ===\n" + ("y" * 100) for i in range(1, 7))
        full_path = Path(config.temp_dir) / f"book_{slug}_full.txt"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(source, encoding="utf-8")

        with pytest.raises(StepError) as excinfo:
            step_book_quality_gate(book_task, config)
        assert "Chapter coverage" in str(excinfo.value)


# ── Synthesis Prompt Tests ────────────────────────────────────────────────────

class TestBookSynthesis:
    """Tests for book synthesis functions."""

    def test_build_book_prompt_includes_title_author(self):
        """_build_book_prompt includes title and author."""
        prompt = _build_book_prompt("Some text.", "My Book", "Jane Doe")
        assert "My Book" in prompt
        assert "Jane Doe" in prompt
        assert "Some text." in prompt

    def test_build_book_prompt_truncates_long_text(self):
        """_build_book_prompt truncates text > 180K chars."""
        long_text = "A" * 250_000
        prompt = _build_book_prompt(long_text, "Title", "Author")
        assert len(prompt) < 210_000  # Should be truncated
        assert "[... content truncated" in prompt

    def test_today_returns_date_string(self):
        """_today returns a YYYY-MM-DD date string."""
        result = _today()
        import re
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', result)

    @patch("agent.pipelines.books._detect_boundaries_llm")
    def test_chunked_synthesis_surfaces_llm_failure(self, mock_boundaries, config):
        """A failed progressive chunk must RETRY, not emit placeholder text."""
        mock_boundaries.return_value = [0, 50_000, 100_000]
        full_text = "word " * 120_000
        llm = MagicMock()
        llm.call.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(StepError) as excinfo:
            _chunked_book_synthesis(
                full_text,
                Path(config.temp_dir) / "book_x_full.txt",
                "Title",
                "Author",
                llm,
                config,
            )
        assert excinfo.value.error_type == ErrorType.RETRY
        assert "Progressive book synthesis failed" in str(excinfo.value)

    def test_detect_boundaries_llm_logs_json_parse_error(self, caplog):
        """Boundary JSON parse errors are logged, not silently discarded."""
        llm = MagicMock()
        llm.call.return_value = MagicMock(content="[1000, 50000]")

        with caplog.at_level(logging.WARNING, logger="agent.pipelines.books"):
            with patch(
                "agent.pipelines.books.json.loads",
                side_effect=json.JSONDecodeError("err", "doc", 0),
            ):
                _detect_boundaries_llm("A" * 250_000, llm)

        assert any("boundary JSON parse failed" in rec.message for rec in caplog.records)


# ── Content Classification Tests ──────────────────────────────────────────────

class TestContentClassification:
    """Tests for step_classify_content."""

    def test_classify_content_with_epubs(self, config, mega_task):
        """Classify content with ePub files."""
        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy book files
        (dl_dir / "book1.epub").write_text("dummy epub content")
        (dl_dir / "book2.epub").write_text("dummy epub content")

        result = step_classify_content(mega_task, config)
        data = json.loads(result)

        assert data["total_files"] == 2
        assert data["clean_text"] == 2
        assert data["scanned"] == 0
        for book in data["books"]:
            assert book["classification"] == "text"
            assert book["pipeline"] == "book_epub"

    @patch("agent.pipelines.books.pymupdf")
    def test_classify_content_with_pdfs(self, mock_pymupdf, config, mega_task):
        """Classify content with PDF files."""
        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)

        (dl_dir / "clean.pdf").write_text("dummy pdf content")

        # Mock because classify_content calls _pdf_preflight internally
        mock_doc = MagicMock()
        mock_doc.page_count = 20
        mock_doc.__getitem__.return_value.get_text.return_value = (
            "The quick brown fox jumps over the lazy dog. "
            * 10
        )
        mock_pymupdf.open.return_value = mock_doc

        result = step_classify_content(mega_task, config)
        data = json.loads(result)

        assert data["total_files"] == 1
        for book in data["books"]:
            assert book["extension"] == ".pdf"
            assert "classification" in book
            assert "quality_score" in book


class TestDelegateBooks:
    """Regression: delegate must fan out into per-book child tasks."""

    def test_delegate_creates_child_tasks_per_book(self, config, mega_task, tmp_path):
        """delegate creates one pending child task per classified epub/pdf book."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)
        epub_a = dl_dir / "alpha.epub"
        epub_b = dl_dir / "beta.epub"
        epub_a.write_text("epub a")
        epub_b.write_text("epub b")

        classify_json = step_classify_content(mega_task, config)
        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_json, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        result = step_delegate_books(mega_task, config)
        summary = json.loads(result)

        assert summary["child_tasks_created"] == 2
        child_ids = {c["task_id"] for c in summary["created"]}
        assert len(child_ids) == 2

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        rows = conn.execute(
            "SELECT id, content_type, status, slug FROM tasks "
            "WHERE id != ? ORDER BY slug",
            (parent_id,),
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        for row in rows:
            assert row[1] == "book_epub"
            assert row[2] == "pending"
        slugs = {row[3] for row in rows}
        assert slugs == {
            f"{mega_task['slug']}_alpha",
            f"{mega_task['slug']}_beta",
        }

    @patch("agent.pipelines.books.pymupdf")
    def test_delegate_creates_child_for_mobi_and_scanned(
        self, mock_pymupdf, config, mega_task, tmp_path,
    ):
        """delegate registers book_mobi / book_scanned — must not skip."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)
        mobi = dl_dir / "kindle.mobi"
        pdf = dl_dir / "scan.pdf"
        mobi.write_bytes(b"mobi")
        pdf.write_bytes(b"%PDF")

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value.get_text.return_value = ""
        mock_pymupdf.open.return_value = mock_doc

        classify_json = step_classify_content(mega_task, config)
        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_json, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        result = step_delegate_books(mega_task, config)
        summary = json.loads(result)

        assert summary["child_tasks_created"] == 2
        assert summary["skipped"] == 0
        pipelines = {c["pipeline"] for c in summary["created"]}
        assert pipelines == {"book_mobi", "book_scanned"}

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        rows = conn.execute(
            "SELECT content_type FROM tasks WHERE id != ?",
            (parent_id,),
        ).fetchall()
        conn.close()
        assert {r[0] for r in rows} == {"book_mobi", "book_scanned"}

    def test_delegate_retry_does_not_duplicate_child_tasks(self, config, mega_task, tmp_path):
        """Re-running delegate after partial completion must not create duplicate child owners."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)
        (dl_dir / "alpha.epub").write_text("epub a")
        (dl_dir / "beta.epub").write_text("epub b")

        classify_json = step_classify_content(mega_task, config)
        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_json, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        step_delegate_books(mega_task, config)
        step_delegate_books(mega_task, config)

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        rows = conn.execute(
            "SELECT id, slug FROM tasks WHERE id != ? ORDER BY slug",
            (parent_id,),
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        slugs = [row[1] for row in rows]
        assert slugs == sorted(slugs)
        assert slugs == [
            f"{mega_task['slug']}_alpha",
            f"{mega_task['slug']}_beta",
        ]

    def test_delegate_reuses_failed_child_instead_of_duplicate(
        self, config, mega_task, tmp_path,
    ):
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)
        epub_path = dl_dir / "alpha.epub"
        epub_path.write_text("epub a")

        classify_json = json.dumps({
            "books": [{
                "path": str(epub_path),
                "filename": "alpha.epub",
                "pipeline": "book_epub",
            }],
        })

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        child_slug = f"{mega_task['slug']}_alpha"
        failed_id = create_task(
            conn,
            str(epub_path),
            "book_epub",
            child_slug,
            ["extract_text", "synthesize_note"],
            status="failed",
        )
        conn.execute(
            "UPDATE task_steps SET status = 'failed', output = 'boom' WHERE task_id = ?",
            (failed_id,),
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_json, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        result = json.loads(step_delegate_books(mega_task, config))
        assert result["child_tasks_created"] == 1
        assert result["created"][0]["task_id"] == failed_id

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        rows = conn.execute(
            "SELECT id, status FROM tasks WHERE slug = ? AND content_type = 'book_epub'",
            (child_slug,),
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == failed_id
        assert rows[0][1] == "pending"

    def test_delegate_redelgates_completed_child_only_when_source_changes(
        self, config, mega_task, tmp_path,
    ):
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        dl_dir = Path(config.temp_dir) / f"mega_{mega_task['slug']}"
        dl_dir.mkdir(parents=True, exist_ok=True)
        sub_a = dl_dir / "pack_a"
        sub_b = dl_dir / "pack_b"
        sub_a.mkdir(parents=True)
        sub_b.mkdir(parents=True)
        epub_v1 = sub_a / "alpha.epub"
        epub_v2 = sub_b / "alpha.epub"
        epub_v1.write_text("v1")
        epub_v2.write_text("v2")

        child_slug = f"{mega_task['slug']}_alpha"
        classify_v1 = json.dumps({
            "books": [{
                "path": str(epub_v1),
                "filename": "alpha.epub",
                "pipeline": "book_epub",
            }],
        })
        classify_v2 = json.dumps({
            "books": [{
                "path": str(epub_v2),
                "filename": "alpha.epub",
                "pipeline": "book_epub",
            }],
        })

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        done_id = create_task(
            conn,
            str(epub_v1),
            "book_epub",
            child_slug,
            ["extract_text"],
            status="completed",
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_v1, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        json.loads(step_delegate_books(mega_task, config))

        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        row = conn.execute(
            "SELECT status, source_url FROM tasks WHERE id = ?",
            (done_id,),
        ).fetchone()
        assert row[0] == "completed"
        assert Path(row[1]).resolve() == epub_v1.resolve()

        conn.execute(
            "UPDATE task_steps SET output = ? WHERE task_id = ? AND name = 'classify_content'",
            (classify_v2, parent_id),
        )
        conn.commit()
        conn.close()

        json.loads(step_delegate_books(mega_task, config))
        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        row = conn.execute(
            "SELECT status, source_url FROM tasks WHERE id = ?",
            (done_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "pending"
        assert Path(row[1]).resolve() == epub_v2.resolve()


class TestBookChildSlug:
    def test_truncated_child_slugs_differ_by_path_hash(self):
        from agent.core.slugify import SLUG_MAX_LEN

        parent = "p" * (SLUG_MAX_LEN - 5)
        path_a = f"/tmp/{'a' * 80}.epub"
        path_b = f"/tmp/{'b' * 80}.epub"
        slug_a = _book_child_slug(parent, path_a)
        slug_b = _book_child_slug(parent, path_b)
        assert len(slug_a) <= SLUG_MAX_LEN
        assert len(slug_b) <= SLUG_MAX_LEN
        assert slug_a != slug_b


class TestBookVaultSourceUrl:
    def test_vault_redacts_local_path_and_mega_fragment(self):
        local = "/secret/downloads/mybook.epub"
        assert _vault_frontmatter_source_url(local) == "mybook.epub"
        mega = "https://mega.nz/file/abc123#super-secret-key"
        assert "#" not in _vault_frontmatter_source_url(mega)
        assert "super-secret-key" not in _vault_frontmatter_source_url(mega)


class TestMegadlZeroArtifacts:
    @patch("agent.pipelines.books.subprocess.run")
    def test_megadl_fails_when_no_book_files_after_download(self, mock_run, config, mega_task):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "megadl":
                dl_path = Path(cmd[cmd.index("--path") + 1])
                (dl_path / "readme.txt").write_text("not a book", encoding="utf-8")
                return MagicMock(returncode=0, stderr="", stdout="")
            raise AssertionError(f"unexpected command: {cmd}")

        mock_run.side_effect = fake_run
        from agent.exceptions import ErrorType, StepError

        with pytest.raises(StepError) as excinfo:
            step_megadl_download(mega_task, config)
        assert excinfo.value.step_name == "megadl_download"
        assert excinfo.value.error_type == ErrorType.FAIL

    def test_delegate_fails_when_no_child_tasks_created(self, config, mega_task, tmp_path):
        from agent.exceptions import ErrorType, StepError

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        data_dir = kb_dir / "data"
        data_dir.mkdir()
        config.kb_agent_dir = kb_dir

        classify_json = json.dumps({"books": [{"pipeline": "not_registered", "filename": "x.pdf"}]})
        conn = init_db(data_dir / "agent.db", AGENT_SCHEMA)
        parent_id = create_task(
            conn,
            mega_task["source_url"],
            "book_batch",
            mega_task["slug"],
            ["megadl_download", "classify_content", "delegate"],
        )
        conn.execute(
            "UPDATE task_steps SET status = 'done', output = ? "
            "WHERE task_id = ? AND name = 'classify_content'",
            (classify_json, parent_id),
        )
        conn.commit()
        conn.close()

        mega_task["id"] = parent_id
        with pytest.raises(StepError) as excinfo:
            step_delegate_books(mega_task, config)
        assert excinfo.value.step_name == "delegate"
        assert excinfo.value.error_type == ErrorType.FAIL


class TestMegadlArchiveExtraction:
    @patch("agent.pipelines.books.subprocess.run")
    def test_megadl_fails_closed_when_unar_fails(self, mock_run, config, mega_task):
        def fake_run(cmd, **kwargs):
            if cmd[0] == "megadl":
                dl_path = Path(cmd[cmd.index("--path") + 1])
                (dl_path / "bundle.rar").write_bytes(b"fake")
                return MagicMock(returncode=0, stderr="", stdout="")
            return MagicMock(returncode=1, stderr="unar failed", stdout="")

        mock_run.side_effect = fake_run
        from agent.exceptions import ErrorType, StepError

        with pytest.raises(StepError) as excinfo:
            step_megadl_download(mega_task, config)
        assert excinfo.value.step_name == "megadl_download"
        assert excinfo.value.error_type == ErrorType.FAIL
        assert any(call.args[0][0] == "unar" for call in mock_run.call_args_list)


def test_resolve_local_book_path_prefers_slug_match_in_mega_dir(config):
    child_slug = "mega_pack_child_alpha"
    dl_dir = Path(config.temp_dir) / f"mega_{child_slug}"
    dl_dir.mkdir(parents=True)
    (dl_dir / "zzz_other.epub").write_text("wrong", encoding="utf-8")
    slug_match = dl_dir / f"{child_slug}.epub"
    slug_match.write_text("right", encoding="utf-8")
    task = {"id": "child", "slug": child_slug, "source_url": ""}
    resolved = _resolve_local_book_path(task, config, {".epub"}, "extract_epub")
    assert resolved == slug_match


def test_resolve_local_book_path_exact_stem_not_lexicographic_substring(config):
    shared_token = "mega_batch_shared"
    child_slug = f"{shared_token}_vol_one"
    dl_dir = Path(config.temp_dir) / f"mega_{child_slug}"
    dl_dir.mkdir(parents=True)
    (dl_dir / f"{shared_token}_vol_two.epub").write_text("two", encoding="utf-8")
    expected = dl_dir / f"{child_slug}.epub"
    expected.write_text("one", encoding="utf-8")
    task = {"id": "child", "slug": child_slug, "source_url": ""}
    resolved = _resolve_local_book_path(task, config, {".epub"}, "extract_epub")
    assert resolved == expected


def test_resolve_local_book_path_hash_suffix_requires_exact_stem(config):
    from agent.exceptions import StepError

    parent = "mega_parent"
    stem = "a" * 200
    child_slug = _book_child_slug(parent, f"/pack/{stem}.epub")
    dl_dir = Path(config.temp_dir) / f"mega_{child_slug}"
    dl_dir.mkdir(parents=True)
    (dl_dir / f"{parent}_{stem[:20]}.epub").write_text("partial", encoding="utf-8")
    task = {"id": "child", "slug": child_slug, "source_url": ""}
    with pytest.raises(StepError, match="exact stem matching hash-suffixed slug"):
        _resolve_local_book_path(task, config, {".epub"}, "extract_epub")


TOKENIZED_BOOK_URL = "https://cdn.example.com/book.mp3?token=secret&expires=999"


class TestBookVaultWriteSanitization:
    """Book vault writes must scrub URLs at rest (synthesis + MinIO frontmatter patch)."""

    @patch("agent.pipelines.books.get_llm_client")
    def test_synthesize_book_scrubs_urls_in_vault_note(
        self, mock_llm_factory, config, book_task,
    ):
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        slug = book_task["slug"]
        full_txt = Path(config.temp_dir) / f"book_{slug}_full.txt"
        full_txt.write_text("short book text " * 30, encoding="utf-8")
        book_task["source_url"] = TOKENIZED_BOOK_URL
        book_task["_book_title"] = "Test Title"
        book_task["_book_author"] = "Test Author"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            f"See also {TOKENIZED_BOOK_URL} for the audio edition.\n"
            "## Summary\n\nContent here.\n"
        )
        mock_llm.call.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        step_synthesize_book(book_task, config)
        note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"
        written = note_path.read_text(encoding="utf-8")
        assert "token=secret" not in written
        assert "https://cdn.example.com/book.mp3" in written

    @patch("agent.archival.upload_file_to_minio")
    def test_minio_upload_scrubs_note_after_frontmatter_patch(
        self, mock_upload, config, book_task,
    ):
        slug = book_task["slug"]
        full_txt = Path(config.temp_dir) / f"book_{slug}_full.txt"
        full_txt.parent.mkdir(parents=True, exist_ok=True)
        full_txt.write_text("extracted book text " * 20)
        note_path = config.vault_path / "Knowledge base" / "books" / f"{slug}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            f"""---
source_url: {TOKENIZED_BOOK_URL}
source_files:
  text: <filled by archival step>
---

Body links {TOKENIZED_BOOK_URL} in prose before patch.
""",
            encoding="utf-8",
        )
        book_task["source_url"] = str(full_txt)
        step_book_minio_upload(book_task, config)
        written = note_path.read_text(encoding="utf-8")
        assert "token=secret" not in written
        assert "https://cdn.example.com/book.mp3" in written
        assert "source_files:" in written


# ── MinIO upload source resolution ─────────────────────────────────────────────

class TestBookMinioUpload:
    """MinIO upload must resolve delegated child local paths."""

    @patch("agent.archival.upload_file_to_minio")
    def test_minio_upload_uses_child_source_url_not_mega_slug_dir(
        self, mock_upload, config, tmp_path,
    ):
        parent_slug = "mega_parent_batch"
        child_slug = f"{parent_slug}_nested_book"
        epub = tmp_path / "nested_book.epub"
        epub.write_text("epub bytes")
        full_txt = Path(config.temp_dir) / f"book_{child_slug}_full.txt"
        full_txt.parent.mkdir(parents=True, exist_ok=True)
        full_txt.write_text("extracted book text " * 20)

        mega_child_dir = Path(config.temp_dir) / f"mega_{child_slug}"
        mega_child_dir.mkdir(parents=True)
        assert not list(mega_child_dir.glob("*.epub"))

        task = {
            "id": "child-task",
            "slug": child_slug,
            "source_url": str(epub),
        }
        result = step_book_minio_upload(task, config)
        data = json.loads(result)
        assert f"books/{child_slug}.txt" in data["uploaded"]
        assert f"books/{child_slug}.epub" in data["uploaded"]
        uploaded_paths = [Path(c[0][0]) for c in mock_upload.call_args_list]
        assert epub.resolve() in {p.resolve() for p in uploaded_paths}


class TestBookSourceUrlSurfaces:
    """Regression: local paths stay in DB for pipelines; status/MCP surfaces redact them."""

    def test_status_redacts_local_source_url(self, tmp_path):
        from agent.core.status import get_task

        db_path = tmp_path / "agent.db"
        conn = init_db(db_path, AGENT_SCHEMA)
        local = "/secret/mega_downloads/hidden_dir/alpha.epub"
        task_id = create_task(conn, local, "book_epub", "child_alpha", ["extract_epub"])
        conn.commit()
        conn.close()

        conn = init_db(db_path, AGENT_SCHEMA)
        enriched = get_task(conn, task_id)
        conn.close()

        assert enriched is not None
        assert enriched["source_url"] == "alpha.epub"
        assert local not in enriched["source_url"]
        assert enriched.get("source_url_href") is None

        conn = init_db(db_path, AGENT_SCHEMA)
        row = conn.execute(
            "SELECT source_url FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        conn.close()
        assert row[0] == local

    def test_book_extract_still_resolves_local_source(self, config, book_task, tmp_path):
        """Child pipeline steps read raw source_url from the task dict (not status API)."""
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
        epub_path = tmp_path / "resolved.epub"
        epub_path.write_text("not a real epub")
        book_task["source_url"] = str(epub_path)
        book_task["slug"] = "resolved_slug"

        mock_book = MagicMock()
        mock_book.get_items_of_type.return_value = []
        mock_book.get_metadata.return_value = []

        with patch("agent.pipelines.books.HAS_EBOOKLIB", True), patch(
            "agent.pipelines.books.epub.read_epub", return_value=mock_book
        ):
            result = json.loads(step_extract_epub(book_task, config))

        assert result["chapter_count"] == 0
        assert Path(result["full_text_path"]).is_file()


# ── Pipeline Registration Tests ───────────────────────────────────────────────

class TestPipelineRegistration:
    """Tests that pipelines are correctly registered."""

    def test_book_pipeline_registered(self):
        """The 'book' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        assert "extract_text" in step_names
        assert "synthesize_note" in step_names
        assert "quality_gate" in step_names
        assert "minio_upload" in step_names
        assert "git_push" in step_names

    def test_book_pdf_pipeline_registered(self):
        """The 'book_pdf' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book_pdf")
        assert pipeline is not None

    def test_mega_pipeline_registered(self):
        """The 'mega' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("mega")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        assert "megadl_download" in step_names
        assert "classify_content" in step_names

    def test_book_batch_pipeline_registered(self):
        """The 'book_batch' pipeline is registered."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book_batch")
        assert pipeline is not None

    def test_book_epub_pipeline_resolves_to_book(self):
        """get_pipeline('book_epub') returns the book pipeline, not web fallback."""
        from agent.pipelines import get_pipeline
        pipeline = get_pipeline("book_epub")
        assert pipeline is not None
        step_names = [s.name for s in pipeline]
        # Must have book pipeline steps, not web pipeline steps
        assert "extract_text" in step_names
        assert "synthesize_note" in step_names
        assert "quality_gate" in step_names
        assert "minio_upload" in step_names
        assert "git_push" in step_names
        # Web pipeline has firecrawl_extract — book pipeline should not
        assert "firecrawl_extract" not in step_names

    @pytest.mark.parametrize("pipeline_name", sorted(CLASSIFY_BOOK_PIPELINES))
    def test_classify_book_pipelines_registered(self, pipeline_name):
        """Invariant: every pipeline from classify_content is registered."""
        from agent.pipelines import get_pipeline

        pipeline = get_pipeline(pipeline_name)
        assert pipeline is not None
        assert "extract_text" in [s.name for s in pipeline]

    def test_mega_video_pipeline_registered(self):
        """get_pipeline('mega_video') returns the video pipeline, not web fallback."""
        from agent.pipelines import get_pipeline

        pipeline = get_pipeline("mega_video")
        web_pipeline = get_pipeline("web")
        assert pipeline is not None
        assert pipeline is not web_pipeline

        step_names = [s.name for s in pipeline]
        assert "firecrawl_extract" not in step_names
        assert "download_video" in step_names
        assert "extract_audio" in step_names
        assert "diarize" in step_names
        assert "transcribe" in step_names
        assert "merge_and_identify" in step_names
        assert "synthesize_youtube" in step_names
        assert "quality_gate" in step_names
        assert "minio_upload" in step_names
        assert "see_also" in step_names
        assert "git_push" in step_names


# ── Book Constants Tests ──────────────────────────────────────────────────────

class TestConstants:
    """Tests for books module constants."""

    def test_book_extensions_includes_common(self):
        """BOOK_EXTENSIONS includes epub, pdf, mobi, azw3."""
        assert ".epub" in BOOK_EXTENSIONS
        assert ".pdf" in BOOK_EXTENSIONS
        assert ".mobi" in BOOK_EXTENSIONS
        assert ".azw3" in BOOK_EXTENSIONS

    def test_scanned_threshold_reasonable(self):
        """PDF_MIN_CHARS_SCANNED is a reasonable value."""
        assert PDF_MIN_CHARS_SCANNED > 0
        assert PDF_MIN_CHARS_SCANNED < 10000

    def test_quality_threshold_reasonable(self):
        """PDF_QUALITY_SCORE_THRESHOLD is 0-100."""
        assert 0 <= PDF_QUALITY_SCORE_THRESHOLD <= 100
