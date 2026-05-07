"""
test_segmentation.py — Unit Tests for FR-003 & FR-004

FR-003: The system must segment extracted text into the four labeled sections
        using a fixed synonym dictionary with case-insensitive exact matching.
FR-004: The system must handle CVs where one or more sections are absent —
        missing sections appear as empty lists [], not errors.

These tests use synthetic PDFs so we know EXACTLY what's in them and can
verify that the parser puts each block in the right bucket.
"""

import pytest
from src.parser import parse_cv


# ──────────────────────────────────────────────────────────────────────────────
# FR-003 — Correct section assignment
# ──────────────────────────────────────────────────────────────────────────────

def test_skills_section_is_populated(single_column_pdf):
    """
    The fixture CV has a "SKILLS" header with Python, MATLAB, etc. below it.
    After segmentation, result["skills"] must contain those blocks.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    assert len(result["skills"]) > 0, "Expected skills section to have content"

    skills_text = " ".join(b["text"] for b in result["skills"]).lower()
    assert "python" in skills_text, "Expected 'python' in skills section"


def test_education_section_is_populated(single_column_pdf):
    """
    The fixture has an "EDUCATION" header followed by degree text.
    That degree text should end up in result["education"], not somewhere else.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    assert len(result["education"]) > 0, "Expected education section to have content"

    edu_text = " ".join(b["text"] for b in result["education"]).lower()
    # "cairo uni" is part of the fixture's education content
    assert "mechatronics" in edu_text or "engineering" in edu_text


def test_projects_section_is_populated(single_column_pdf):
    """
    The fixture has a "PROJECTS" header with robot project descriptions.
    The ROS2 reference should land in result["projects"].
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    assert len(result["projects"]) > 0, "Expected projects section to have content"

    proj_text = " ".join(b["text"] for b in result["projects"]).lower()
    assert "ros2" in proj_text or "conveyor" in proj_text


def test_internships_section_is_populated(single_column_pdf):
    """
    The fixture has an "INTERNSHIPS" header with work experience details.
    The Schneider Electric reference should land in result["internships"].
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result
    assert len(result["internships"]) > 0, "Expected internships section to have content"

    intern_text = " ".join(b["text"] for b in result["internships"]).lower()
    assert "schneider" in intern_text or "scada" in intern_text


def test_content_does_not_bleed_across_sections(single_column_pdf):
    """
    Content that belongs to "Skills" must NOT also appear in "Education".
    This tests that a section switch is clean — no double-counting.
    """
    result = parse_cv(single_column_pdf)
    assert "error" not in result

    skills_text = " ".join(b["text"] for b in result["skills"]).lower()
    edu_text    = " ".join(b["text"] for b in result["education"]).lower()

    # "cairo uni" is only in the education fixture block — not in skills
    assert "cairo uni" not in skills_text, (
        "Education content bled into the skills section"
    )


# ──────────────────────────────────────────────────────────────────────────────
# FR-003 — Non-standard header synonym mapping
# ──────────────────────────────────────────────────────────────────────────────

def test_technical_skills_maps_to_skills_section(nonstandard_headers_pdf):
    """
    "Technical Skills" (not "Skills") should still correctly map to the skills
    canonical section. Our SECTION_SYNONYMS dict covers this variant.
    """
    result = parse_cv(nonstandard_headers_pdf)
    assert "error" not in result
    assert len(result["skills"]) > 0, (
        "Expected 'Technical Skills' header to map to skills section"
    )


def test_work_experience_maps_to_internships_section(nonstandard_headers_pdf):
    """
    "Work Experience" is the most common alternative header for internships.
    It must be recognized and its content must land in result["internships"].
    """
    result = parse_cv(nonstandard_headers_pdf)
    assert "error" not in result
    assert len(result["internships"]) > 0, (
        "Expected 'Work Experience' header to map to internships section"
    )


def test_academic_background_maps_to_education_section(nonstandard_headers_pdf):
    """
    "Academic Background" instead of "Education" — should still work.
    """
    result = parse_cv(nonstandard_headers_pdf)
    assert "error" not in result
    assert len(result["education"]) > 0, (
        "Expected 'Academic Background' header to map to education section"
    )


def test_engineering_projects_maps_to_projects_section(nonstandard_headers_pdf):
    """
    "Engineering Projects" instead of "Projects" — should still work.
    """
    result = parse_cv(nonstandard_headers_pdf)
    assert "error" not in result
    assert len(result["projects"]) > 0, (
        "Expected 'Engineering Projects' header to map to projects section"
    )


# ──────────────────────────────────────────────────────────────────────────────
# FR-004 — Missing sections handled gracefully
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_sections_are_empty_lists_not_errors(missing_sections_pdf):
    """
    The fixture only has Skills and Education.
    Projects and Internships sections don't exist in the PDF at all.
    The output must still have all four keys — just with empty lists.
    """
    result = parse_cv(missing_sections_pdf)
    assert "error" not in result, f"Unexpected error: {result}"

    # All four keys must be present
    assert "skills"      in result
    assert "projects"    in result
    assert "education"   in result
    assert "internships" in result

    # The missing sections should be empty lists
    assert result["projects"]    == [], (
        "Expected empty list for missing 'projects' section"
    )
    assert result["internships"] == [], (
        "Expected empty list for missing 'internships' section"
    )


def test_present_sections_still_correct_when_others_missing(missing_sections_pdf):
    """
    Just because some sections are missing doesn't mean the present ones break.
    Skills and Education should still be correctly populated.
    """
    result = parse_cv(missing_sections_pdf)
    assert "error" not in result

    # Skills should have content from "SolidWorks, AutoCAD, Python, MATLAB"
    assert len(result["skills"]) > 0, "Skills should still be populated"

    # Education should have content from the degree line
    assert len(result["education"]) > 0, "Education should still be populated"


def test_no_keyerror_on_any_of_the_four_keys(missing_sections_pdf):
    """
    Defensive test: accessing all four keys on a "missing sections" CV
    must never raise a KeyError. This would break Phase 2 and beyond.
    """
    result = parse_cv(missing_sections_pdf)
    # If any of these raise KeyError, the test fails
    _ = result["skills"]
    _ = result["projects"]
    _ = result["education"]
    _ = result["internships"]
