"""
test_form_validation.py — Unit tests for validate_cv_upload()

These tests verify that the CV file validator correctly rejects:
  - missing files (None)
  - files with wrong content type
  - files with wrong extension
  - files exceeding 10 MB

And accepts a valid small PDF.

Written BEFORE implementation (TDD — must FAIL until routes.py exists).
"""

import io
import pytest
from werkzeug.datastructures import FileStorage

# Import the function we're about to write.
from src.api.v1.routes import validate_cv_upload


def _make_filestorage(
    content: bytes,
    filename: str = "test.pdf",
    content_type: str = "application/pdf",
) -> FileStorage:
    """Helper to build a werkzeug FileStorage object from raw bytes."""
    return FileStorage(
        stream=io.BytesIO(content),
        filename=filename,
        content_type=content_type,
    )


class TestValidateCvUploadMissingFile:
    """None (missing cv_file field) must produce a 400 error tuple."""

    def test_none_returns_error_tuple(self):
        result = validate_cv_upload(None)
        assert result is not None, "Expected an error tuple, got None (None means valid)"
        message, status_code = result
        assert status_code == 400
        assert isinstance(message, str) and len(message) > 0


class TestValidateCvUploadWrongType:
    """A file with content_type != application/pdf must be rejected."""

    def test_wrong_content_type_returns_error(self):
        f = _make_filestorage(
            content=b"PK\x03\x04fake docx",
            filename="resume.docx",
            content_type="application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document",
        )
        result = validate_cv_upload(f)
        assert result is not None
        message, status_code = result
        assert status_code == 400
        assert "PDF" in message

    def test_pdf_content_type_but_wrong_extension(self):
        # content_type says PDF but filename extension says otherwise
        f = _make_filestorage(
            content=b"%PDF-1.4\n",
            filename="resume.txt",
            content_type="application/pdf",
        )
        result = validate_cv_upload(f)
        assert result is not None
        _, status_code = result
        assert status_code == 400


class TestValidateCvUploadOversized:
    """Files larger than 10 MB must be rejected with a size error."""

    def test_eleven_mb_returns_size_error(self):
        eleven_mb = b"%PDF-1.4\n" + b"x" * (11 * 1024 * 1024)
        f = _make_filestorage(content=eleven_mb)
        result = validate_cv_upload(f)
        assert result is not None
        message, status_code = result
        assert status_code == 400
        assert "10MB" in message or "10 MB" in message


class TestValidateCvUploadValidFile:
    """A small, properly-typed PDF must return None (no error)."""

    def test_valid_small_pdf_returns_none(self):
        tiny_pdf = b"%PDF-1.4\n" + b"x" * 512
        f = _make_filestorage(
            content=tiny_pdf,
            filename="test.pdf",
            content_type="application/pdf",
        )
        result = validate_cv_upload(f)
        assert result is None, (
            f"Expected None for a valid file, got {result!r}"
        )
