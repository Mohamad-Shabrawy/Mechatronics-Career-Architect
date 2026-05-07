"""
test_error_handling.py — Unit Tests for FR-008

FR-008: The system must handle malformed, empty, or image-only PDF inputs
        gracefully — returning a descriptive error dict, never raising.

The three error codes we test here:
  FILE_NOT_FOUND  — the path doesn't exist
  INVALID_PDF     — the file exists but isn't a valid PDF
  NO_TEXT_CONTENT — valid PDF but no machine-readable text (scanned/image-only)

A key principle: parse_cv() NEVER raises an exception.
Any error, no matter how broken the input, must come back as a dict.
"""

import os
import pytest
import fitz  # We use fitz to build edge-case test PDFs

from src.parser import parse_cv


# ──────────────────────────────────────────────────────────────────────────────
# FILE_NOT_FOUND — the path doesn't point to anything
# ──────────────────────────────────────────────────────────────────────────────

def test_nonexistent_path_returns_file_not_found():
    """
    Passing a path that doesn't exist must return the FILE_NOT_FOUND error code.
    """
    result = parse_cv("/this/path/cannot/possibly/exist/cv.pdf")
    assert result.get("error") == "FILE_NOT_FOUND", (
        f"Expected FILE_NOT_FOUND, got: {result}"
    )


def test_nonexistent_path_includes_helpful_message():
    """
    The error message should be descriptive enough for the user to understand
    what went wrong — not just "error" with no context.
    """
    result = parse_cv("/fake/path/cv.pdf")
    assert "message" in result
    assert len(result["message"]) > 5, "Error message is too short to be helpful"


def test_nonexistent_path_never_raises(tmp_path):
    """
    Even the FILE_NOT_FOUND path must not bubble an exception.
    We wrap in try/except to be absolutely sure.
    """
    try:
        result = parse_cv(str(tmp_path / "does_not_exist.pdf"))
    except Exception as exc:
        pytest.fail(
            f"parse_cv() raised {type(exc).__name__} on missing file: {exc}"
        )
    assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# INVALID_PDF — the file exists but is not a valid PDF
# ──────────────────────────────────────────────────────────────────────────────

def test_non_pdf_file_returns_invalid_pdf(tmp_path):
    """
    A plain text file (not a PDF) should trigger INVALID_PDF.
    We create a .pdf file that contains plain text, which fitz will reject.
    """
    # Write a file that looks like a PDF by name but has garbage content
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"This is definitely not a PDF file, just plain text bytes.")

    result = parse_cv(str(fake_pdf))
    assert result.get("error") == "INVALID_PDF", (
        f"Expected INVALID_PDF for a non-PDF file, got: {result}"
    )


def test_empty_file_returns_invalid_pdf(tmp_path):
    """
    A completely empty file (0 bytes) is also not a valid PDF.
    """
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    result = parse_cv(str(empty_file))
    assert result.get("error") == "INVALID_PDF", (
        f"Expected INVALID_PDF for empty file, got: {result}"
    )


def test_invalid_pdf_never_raises(tmp_path):
    """
    Even corrupted/invalid files must not cause exceptions.
    """
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4\nThis is fake PDF content with broken structure")

    try:
        result = parse_cv(str(corrupt_file))
    except Exception as exc:
        pytest.fail(
            f"parse_cv() raised {type(exc).__name__} on invalid PDF: {exc}"
        )
    # It might succeed partially or return an error — either is acceptable,
    # as long as it doesn't raise
    assert isinstance(result, dict)


def test_invalid_pdf_message_is_human_readable(tmp_path):
    """
    The INVALID_PDF message should help the user understand the problem,
    not just say "error".
    """
    fake = tmp_path / "not_real.pdf"
    fake.write_bytes(b"not a pdf at all")

    result = parse_cv(str(fake))
    if "error" in result and result["error"] == "INVALID_PDF":
        assert len(result["message"]) > 10, "INVALID_PDF message is too short"


# ──────────────────────────────────────────────────────────────────────────────
# NO_TEXT_CONTENT — valid PDF but no extractable text (image-only/scanned)
# ──────────────────────────────────────────────────────────────────────────────

def test_image_only_pdf_returns_no_text_content(tmp_path):
    """
    A PDF that is valid but contains only images (no text layer) should
    return the NO_TEXT_CONTENT error code.

    We simulate this by creating a PDF with only a rectangle drawing
    and no text insertion — fitz will see no text blocks at all.
    """
    # Build a valid PDF page with only a drawing (no text)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Draw a filled rectangle to simulate an "image" — no text
    page.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    pdf_path = tmp_path / "image_only.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = parse_cv(str(pdf_path))
    assert result.get("error") == "NO_TEXT_CONTENT", (
        f"Expected NO_TEXT_CONTENT for image-only PDF, got: {result}"
    )


def test_no_text_content_message_mentions_scanned(tmp_path):
    """
    The NO_TEXT_CONTENT message should guide the user to understand that
    scanned PDFs are not supported in Phase 1.
    """
    doc = fitz.open()
    doc.new_page()  # A completely blank page with zero text
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()

    result = parse_cv(str(pdf_path))
    assert result.get("error") == "NO_TEXT_CONTENT"
    message = result.get("message", "").lower()
    # The message should mention "scanned" or "image" so users understand why it failed
    assert "scanned" in message or "image" in message, (
        f"Expected 'scanned' or 'image' in NO_TEXT_CONTENT message, got: '{message}'"
    )


def test_image_only_pdf_never_raises(tmp_path):
    """
    Even the image-only path must not throw any exception.
    """
    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / "blank_no_raise.pdf"
    doc.save(str(pdf_path))
    doc.close()

    try:
        result = parse_cv(str(pdf_path))
    except Exception as exc:
        pytest.fail(
            f"parse_cv() raised {type(exc).__name__} on image-only PDF: {exc}"
        )
    assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────────────────────
# General behavioral guarantee — parse_cv() NEVER raises
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_input", [
    "",                          # empty string path
    "   ",                       # whitespace-only path
    "C:\\nonexistent\\path.pdf", # Windows-style path that doesn't exist
    "/dev/null",                 # special file (Unix) — may exist but isn't a PDF
    "a" * 300 + ".pdf",          # absurdly long filename
])
def test_parse_cv_never_raises_for_any_bad_path(bad_input):
    """
    parse_cv() is used as a library function from the Flask server.
    If it ever raises, the web server crashes. We test a variety of
    bad inputs to make sure NO exception propagates to the caller.
    """
    try:
        result = parse_cv(bad_input)
    except Exception as exc:
        pytest.fail(
            f"parse_cv() raised {type(exc).__name__} for input '{bad_input[:50]}...': {exc}"
        )
    # Must always return a dict — either a success dict or an error dict
    assert isinstance(result, dict), (
        f"parse_cv() must return a dict, got {type(result)}"
    )
