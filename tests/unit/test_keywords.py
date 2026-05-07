"""
test_keywords.py — Unit Tests for FR-005 & FR-006

FR-005: The system must recognize technical Mechatronics keywords from
        the 240+ term lexicon within segmented sections.
FR-006: Keywords must be recorded per section context, deduplicated
        (each keyword appears at most once per block's keywords list).

We test these properties directly via parse_cv() on synthetic PDFs,
and also by calling the internal _recognize_keywords() helper directly
so we can test its logic in isolation.
"""

import pytest
from src.parser import parse_cv, _recognize_keywords


# ──────────────────────────────────────────────────────────────────────────────
# Direct _recognize_keywords() unit tests
# These don't need PDF fixtures — they test the keyword scanner in isolation.
# ──────────────────────────────────────────────────────────────────────────────

def test_known_keyword_is_recognized():
    """
    The simplest possible test: "PID controller was used" contains "pid",
    which is in our lexicon. The scanner must return ["pid"].
    This exact example comes from the spec (tasks.md US4 test).
    """
    result = _recognize_keywords("PID controller was used in the project.")
    assert "pid" in result, f"Expected 'pid' in keywords, got: {result}"


def test_keyword_recognition_is_case_insensitive():
    """
    A CV might write MATLAB, matlab, or Matlab — they should all match.
    """
    for variant in ["MATLAB", "matlab", "Matlab", "mAtLaB"]:
        result = _recognize_keywords(f"Used {variant} for simulation.")
        assert "matlab" in result, (
            f"Expected 'matlab' recognized for variant '{variant}', got: {result}"
        )


def test_duplicate_keyword_in_same_text_appears_only_once():
    """
    "PID pid PID" in one block should yield ["pid"], not ["pid", "pid", "pid"].
    This is the deduplication requirement from FR-006.
    (Per spec tasks.md: "PID pid PID" in one block yields ["pid"])
    """
    result = _recognize_keywords("PID pid PID controller is very important.")
    pid_count = result.count("pid")
    assert pid_count == 1, (
        f"'pid' appeared {pid_count} times in keywords list — expected exactly 1 (dedup)"
    )


def test_block_with_no_keywords_returns_empty_list():
    """
    If the text has no Mechatronics keywords at all, the keywords list
    should be an empty list [], not None or some other value.
    """
    result = _recognize_keywords("I enjoy reading books and going to the gym.")
    assert result == [], f"Expected empty list, got: {result}"


def test_whole_word_matching_micropython_does_not_match_python():
    """
    "micropython" contains "python" as a substring, but it's a different thing.
    Our word-boundary matching must NOT flag "python" inside "micropython".
    This verifies FR-005's whole-word-only requirement.
    """
    result = _recognize_keywords("I have experience with micropython on ESP boards.")
    # "micropython" is not "python" — "python" should NOT be matched here
    # because the \b (or (?<!\w)) boundary before "python" fails when
    # it's preceded by "micro" (a word character)
    assert "python" not in result, (
        "Incorrectly matched 'python' inside 'micropython'. "
        "Whole-word matching is broken."
    )


def test_whole_word_matching_catia_does_not_match_c_language():
    """
    "CATIA" starts with "C" but "C" (the programming language) should NOT
    be matched here. This is the most common false-positive trap in the lexicon.
    """
    result = _recognize_keywords("Used CATIA V5 for 3D modeling of the bracket.")
    # "catia" might or might not be in the lexicon as a full keyword,
    # but "c" (the C language) should NOT be extracted from within "catia"
    assert "c" not in result, (
        "Incorrectly matched 'c' language keyword inside 'catia'. "
        "Whole-word matching failed."
    )


def test_keywords_in_different_text_blocks_are_independent():
    """
    Keywords from two different blocks must be collected independently.
    Calling _recognize_keywords on block A and block B separately
    should give separate results (no shared state between calls).
    """
    block_a = "Python and MATLAB are used for simulation."
    block_b = "SolidWorks and AutoCAD for mechanical design."

    result_a = _recognize_keywords(block_a)
    result_b = _recognize_keywords(block_b)

    # A's keywords should NOT appear in B and vice versa
    assert "python" in result_a
    assert "matlab" in result_a
    assert "python" not in result_b
    assert "matlab" not in result_b

    assert "solidworks" in result_b
    assert "autocad"    in result_b
    assert "solidworks" not in result_a
    assert "autocad"    not in result_a


def test_keywords_list_is_sorted():
    """
    The keywords list must always be in sorted order for deterministic output.
    Two runs on the same text must produce the same ordered list — this matters
    for reproducibility in Phase 3 feature vectorization.
    """
    result = _recognize_keywords(
        "Used Python, MATLAB, Simulink, and SolidWorks extensively."
    )
    assert result == sorted(result), (
        f"Keywords are not sorted: {result}"
    )


def test_multi_word_keyword_recognized():
    """
    Not all keywords are single words — "pid controller tuning",
    "can bus", "freertos" are multi-word or compound terms.
    These must be recognized correctly.
    """
    result = _recognize_keywords(
        "Implemented FreeRTOS on the STM32 for task scheduling."
    )
    assert "freertos" in result, f"Expected 'freertos', got: {result}"
    assert "stm32"    in result, f"Expected 'stm32', got: {result}"


# ──────────────────────────────────────────────────────────────────────────────
# FR-006 — Keywords recorded per section (via full parse_cv pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def test_keywords_recorded_in_correct_section(single_column_pdf):
    """
    "Python, MATLAB, Simulink" appears in the SKILLS section of the fixture.
    "ROS2, Gazebo Simulator, OpenCV" appear in the PROJECTS section.
    Each keyword must be in its own section's blocks, not cross-contaminated.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result

    # Collect all keywords found in skills blocks
    skills_keywords = {
        kw
        for block in result["skills"]
        for kw in block["keywords"]
    }

    # Collect all keywords found in projects blocks
    project_keywords = {
        kw
        for block in result["projects"]
        for kw in block["keywords"]
    }

    # Skills section should have "python" or "matlab"
    assert "python" in skills_keywords or "matlab" in skills_keywords, (
        f"Expected Python or MATLAB in skills keywords, got: {skills_keywords}"
    )

    # Projects section should have "ros2" or "gazebo"
    assert "ros2" in project_keywords or "gazebo simulator" in project_keywords, (
        f"Expected ROS2 or Gazebo in project keywords, got: {project_keywords}"
    )


def test_no_keywords_pdf_produces_empty_keyword_lists(no_keywords_pdf):
    """
    A CV with zero Mechatronics keywords must produce blocks where every
    "keywords" list is empty. The parser must not crash or return None.
    """
    result = parse_cv(no_keywords_pdf)
    assert "error" not in result

    for section, blocks in result.items():
        for block in blocks:
            assert block["keywords"] == [], (
                f"Expected empty keywords in '{section}', got: {block['keywords']}"
            )


def test_keywords_are_lowercase_in_output(single_column_pdf):
    """
    The spec says keywords in the output must be lowercase strings.
    Even if the CV writes "MATLAB" in all caps, the keyword list must
    contain "matlab" (lowercase).
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result

    for section, blocks in result.items():
        for block in blocks:
            for kw in block["keywords"]:
                assert kw == kw.lower(), (
                    f"Keyword '{kw}' is not lowercase in section '{section}'"
                )
