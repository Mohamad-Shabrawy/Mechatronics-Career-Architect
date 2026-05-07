"""
test_extraction.py — Unit Tests for FR-001 & FR-002

FR-001: The system must extract all machine-readable text from a PDF.
FR-002: The system must detect multi-column layouts and read them in
        the correct order (left column fully before right column).

These tests focus on the raw text extraction step and the column-ordering step.
We use synthetic PDFs built by the fixtures in conftest.py so the tests
run fast and don't need real CV files.
"""

import pytest
from src.parser import parse_cv


# ──────────────────────────────────────────────────────────────────────────────
# FR-001 — Basic text extraction (single-column CVs)
# ──────────────────────────────────────────────────────────────────────────────

def test_single_column_pdf_returns_nonempty_output(single_column_pdf):
    """
    The most basic check: a valid single-column CV should produce at least
    some content in at least one section — not all empty lists.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result, f"Unexpected error: {result}"

    # Count total blocks across all sections
    total_blocks = sum(len(blocks) for blocks in result.values())
    assert total_blocks > 0, "Expected at least one text block, got nothing"


def test_single_column_pdf_extracts_known_text(single_column_pdf):
    """
    We know exactly what text the fixture puts in the PDF, so we can verify
    that key phrases actually appear somewhere in the extracted output.
    This catches character-dropping bugs (e.g. encoding issues in fitz).
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result

    # Gather ALL text from all sections into one big string for easy searching
    all_text = " ".join(
        block["text"]
        for blocks in result.values()
        for block in blocks
    ).lower()

    # These phrases were written into the fixture PDF — they must appear in output
    assert "python" in all_text,    "Expected 'python' in extracted text"
    assert "matlab" in all_text,    "Expected 'matlab' in extracted text"
    assert "solidworks" in all_text,"Expected 'solidworks' in extracted text"
    assert "stm32" in all_text,     "Expected 'stm32' in extracted text"


def test_nonexistent_file_returns_file_not_found_error():
    """
    Passing a path that doesn't exist should return an error dict —
    NOT raise a FileNotFoundError exception to the caller.
    """
    result = parse_cv("/totally/fake/path/cv.pdf")
    assert "error" in result
    assert result["error"] == "FILE_NOT_FOUND"
    assert "message" in result
    assert len(result["message"]) > 0


def test_file_not_found_never_raises_exception():
    """
    parse_cv() must NEVER raise. All error paths return dicts.
    This test is extra paranoid — it wraps the call in a try/except
    to prove no exception bubbles up.
    """
    try:
        result = parse_cv("does_not_exist.pdf")
    except Exception as exc:
        pytest.fail(
            f"parse_cv() raised {type(exc).__name__} instead of returning error dict: {exc}"
        )
    # If we get here without an exception, the behavioral guarantee is met
    assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# FR-002 — Column detection and reading order
# ──────────────────────────────────────────────────────────────────────────────

def test_two_column_pdf_left_column_appears_before_right(two_column_pdf):
    """
    In a two-column CV, the left column content must appear before the right
    column content in the combined extracted text.

    Our fixture puts "MATLAB" in the LEFT column and "ROS2" in the RIGHT column.
    After correct column ordering: MATLAB comes first, ROS2 comes second.
    If the columns are interleaved incorrectly, this order would break.
    """
    result = parse_cv(two_column_pdf)
    assert "error" not in result

    # Collect all text blocks across all sections, in the order they appear
    all_text_blocks = [
        block["text"]
        for blocks in result.values()
        for block in blocks
    ]
    combined = "\n".join(all_text_blocks).lower()

    # Verify both keywords are present at all
    assert "matlab" in combined, "Expected 'matlab' from left column"
    assert "ros2"   in combined, "Expected 'ros2' from right column"

    # The critical check: MATLAB (left col) must appear BEFORE ROS2 (right col)
    matlab_pos = combined.index("matlab")
    ros2_pos   = combined.index("ros2")
    assert matlab_pos < ros2_pos, (
        f"Left-column content (matlab at {matlab_pos}) should appear before "
        f"right-column content (ros2 at {ros2_pos}). "
        "Columns may be interleaved or reversed."
    )


def test_two_column_pdf_no_content_lost(two_column_pdf):
    """
    When we reorder the columns, we must not DROP any text.
    Both columns' content should be present in the output.
    """
    result = parse_cv(two_column_pdf)
    assert "error" not in result

    all_text = " ".join(
        block["text"]
        for blocks in result.values()
        for block in blocks
    ).lower()

    # Left column keywords (written into the fixture)
    assert "solidworks" in all_text
    assert "autocad"    in all_text

    # Right column keywords (written into the fixture)
    assert "moveit"     in all_text     # "MoveIt!" → lowercased
    assert "pid"        in all_text


def test_single_column_pdf_unchanged_by_column_logic(single_column_pdf):
    """
    The column-detection logic must not break single-column CVs.
    A single-column PDF should still produce correct, complete output
    even after the column classification step runs.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result

    all_text = " ".join(
        block["text"]
        for blocks in result.values()
        for block in blocks
    ).lower()

    # Spot check a few things from the fixture to prove nothing was lost
    assert "python"     in all_text
    assert "stm32"      in all_text
    assert "solidworks" in all_text
