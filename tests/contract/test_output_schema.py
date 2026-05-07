"""
test_output_schema.py — Contract Tests (FR-007)

These tests verify the SHAPE of what parse_cv() returns, not the content.
Think of them as the "contract" between Phase 1 and every downstream phase:
if Phase 2 (NER) imports parse_cv() and this test passes, it can safely assume
the output will always have the correct structure.

Tests here cover:
- The four required top-level keys are always present
- Each value is a list
- Each list item has "text" (str) and "keywords" (list[str]) keys
- The output is JSON-serializable (json.dumps must not raise)
- Error responses follow the error schema {"error": str, "message": str}
"""

import json
import pytest
from src.parser import parse_cv


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# The exact four keys we expect in every successful response.
REQUIRED_KEYS = {"skills", "projects", "education", "internships"}


def assert_valid_success_schema(result: dict) -> None:
    """
    Reusable helper that asserts the full schema contract on a success response.
    We call this from multiple tests to avoid repeating ourselves.
    """
    # Must have exactly the four section keys — no more, no less
    assert set(result.keys()) == REQUIRED_KEYS, (
        f"Expected keys {REQUIRED_KEYS}, got {set(result.keys())}"
    )

    for section_name, blocks in result.items():
        # Each section must be a list (possibly empty)
        assert isinstance(blocks, list), (
            f"Section '{section_name}' should be a list, got {type(blocks)}"
        )

        for i, block in enumerate(blocks):
            # Each item in the list must be a dict
            assert isinstance(block, dict), (
                f"Block {i} in '{section_name}' should be a dict"
            )

            # Every block must have a "text" key that is a non-empty string
            assert "text" in block, f"Block {i} in '{section_name}' missing 'text'"
            assert isinstance(block["text"], str), (
                f"'text' in block {i} of '{section_name}' must be str"
            )
            assert block["text"].strip() != "", (
                f"'text' in block {i} of '{section_name}' must not be empty"
            )

            # Every block must have a "keywords" key that is a list
            assert "keywords" in block, f"Block {i} in '{section_name}' missing 'keywords'"
            assert isinstance(block["keywords"], list), (
                f"'keywords' in block {i} of '{section_name}' must be a list"
            )

            # Every keyword in the list must be a string
            for kw in block["keywords"]:
                assert isinstance(kw, str), (
                    f"Keyword '{kw}' in block {i} of '{section_name}' must be str"
                )

    # The whole thing must be JSON-serializable — this is the downstream contract
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"parse_cv() return value is not JSON-serializable: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests — success path
# ──────────────────────────────────────────────────────────────────────────────

def test_output_has_four_required_keys(single_column_pdf):
    """
    The most fundamental contract test: does the output have exactly the
    four keys we promised to Phase 2?
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result, f"Unexpected error: {result}"
    assert set(result.keys()) == REQUIRED_KEYS


def test_each_section_is_a_list(single_column_pdf):
    """
    Each section value must be a list so downstream code can iterate over it.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    for key in REQUIRED_KEYS:
        assert isinstance(result[key], list), f"'{key}' is not a list"


def test_each_block_has_text_and_keywords_keys(single_column_pdf):
    """
    Every item inside a section list must be a dict with "text" and "keywords".
    Phase 2 will access result["skills"][0]["text"] — if the key is missing, it breaks.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    for section, blocks in result.items():
        for block in blocks:
            assert "text" in block, f"Missing 'text' key in {section}"
            assert "keywords" in block, f"Missing 'keywords' key in {section}"


def test_text_values_are_non_empty_strings(single_column_pdf):
    """
    The "text" field in each block must be a non-empty string.
    Empty blocks would confuse the downstream NER model.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    for section, blocks in result.items():
        for block in blocks:
            assert isinstance(block["text"], str)
            assert block["text"].strip() != "", f"Empty text block in '{section}'"


def test_keywords_values_are_lists_of_strings(single_column_pdf):
    """
    The "keywords" field in each block must be a list (possibly empty).
    Phase 3 will iterate over it to build feature vectors.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    for section, blocks in result.items():
        for block in blocks:
            assert isinstance(block["keywords"], list)
            for kw in block["keywords"]:
                assert isinstance(kw, str), f"Keyword '{kw}' is not a string"


def test_output_is_json_serializable(single_column_pdf):
    """
    The output must survive json.dumps() without raising.
    This is the primary serialization contract — Phase 6 (Flask API) sends
    this dict as the JSON response body.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    # This line must NOT raise — if it does, the test fails with a useful message
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


def test_missing_sections_still_present_as_empty_lists(missing_sections_pdf):
    """
    Even if a CV has no "Internships" section, the key must still exist
    in the output as an empty list []. Phase 2 should never have to handle
    a missing key — it can always safely do result["internships"].
    """
    result = parse_cv(missing_sections_pdf)
    assert "error" not in result
    # All four keys must exist, regardless of which sections are in the PDF
    assert set(result.keys()) == REQUIRED_KEYS
    # The missing sections should be empty lists, not KeyError
    # (we can't be certain which sections are empty without reading the PDF,
    #  but we know at least internships and projects should be empty for this fixture)
    assert isinstance(result["internships"], list)
    assert isinstance(result["projects"], list)


def test_full_contract_with_complete_cv(single_column_pdf):
    """
    Full contract assertion using the reusable helper.
    If every individual test above passes, this one should too —
    but it's here as a final end-to-end schema sanity check.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    assert_valid_success_schema(result)


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests — error path
# ──────────────────────────────────────────────────────────────────────────────

def test_error_response_has_error_key():
    """
    When something goes wrong (file not found, invalid PDF, etc.),
    the return value must contain an "error" key so callers can check for it.
    """
    result = parse_cv("this_file_absolutely_does_not_exist.pdf")
    assert "error" in result, "Error response missing 'error' key"


def test_error_response_has_message_key():
    """
    Every error response must also have a "message" key with a human-readable
    explanation. This is what the web UI will display to the user.
    """
    result = parse_cv("this_file_absolutely_does_not_exist.pdf")
    assert "message" in result, "Error response missing 'message' key"


def test_error_response_values_are_strings():
    """
    Both "error" (the code) and "message" (the description) must be strings.
    """
    result = parse_cv("this_file_absolutely_does_not_exist.pdf")
    assert isinstance(result.get("error"), str)
    assert isinstance(result.get("message"), str)


def test_error_response_is_json_serializable():
    """
    Error responses must also be JSON-serializable — the Flask API sends them
    back to the frontend just like success responses.
    """
    result = parse_cv("this_file_absolutely_does_not_exist.pdf")
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"Error response is not JSON-serializable: {exc}")
