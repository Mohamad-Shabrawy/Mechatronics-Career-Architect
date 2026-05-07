"""
conftest.py — Shared Test Fixtures

pytest automatically loads this file before any test runs.
Fixtures defined here are available to ALL test files without importing them.

We use fitz (PyMuPDF) to build tiny synthetic PDFs on-the-fly for unit tests.
This means unit tests never need real CV files — they're fully self-contained.
The integration tests, however, DO use the real sample PDFs from the data/samples folder.
"""

import os
import pytest
import fitz  # PyMuPDF — same library the parser uses


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

# Find the root of our project regardless of where pytest is launched from.
# __file__ is this conftest.py → go up two levels to reach the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The sample CVs live in the sibling "Claude Training" folder (not inside this project).
# We resolve that path here once so every test can import SAMPLES_DIR.
SAMPLES_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT),   # one level up = "Python Engineering Projects"
    "Claude Training",
    "data",
    "samples",
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic PDF factories
# These build minimal in-memory PDFs using fitz.
# They're fast (no disk I/O during the fixture) and deterministic.
# ──────────────────────────────────────────────────────────────────────────────

def _make_pdf_at(path: str, blocks: list[tuple]) -> str:
    """
    Internal helper — writes a PDF to `path` with the given text blocks.
    Each tuple in `blocks` is (x, y, text, fontsize).
    Returns the path string so callers can pass it directly to parse_cv().
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 dimensions in points

    for x, y, text, fontsize in blocks:
        page.insert_text((x, y), text, fontsize=fontsize)

    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def single_column_pdf(tmp_path) -> str:
    """
    A simple, clean single-column CV.
    All text starts near the left margin (x ≈ 50), so there's no right column.
    Contains all four sections: Skills, Education, Projects, Internships.
    Several keywords from the lexicon are embedded so keyword tests can run.
    """
    pdf_path = str(tmp_path / "single_column.pdf")
    blocks = [
        # Header / name area
        (50, 50,  "John Doe — Mechatronics Engineer",         12),

        # SKILLS section
        (50, 90,  "SKILLS",                                    11),
        (50, 110, "Python, MATLAB, Simulink, SolidWorks",      10),
        (50, 125, "STM32, FreeRTOS, Embedded C, PID control",  10),
        (50, 140, "Arduino, Proteus, KiCad, PCB Design",       10),

        # EDUCATION section
        (50, 175, "EDUCATION",                                  11),
        (50, 195, "B.Sc. Mechatronics Engineering, Cairo Uni", 10),
        (50, 210, "GPA: 3.7 / 4.0  —  Graduated 2024",        10),

        # PROJECTS section
        (50, 245, "PROJECTS",                                   11),
        (50, 265, "Autonomous Line-Following Robot",            10),
        (50, 280, "Used ROS2, Gazebo Simulator, and OpenCV.",   10),
        (50, 310, "PLC-Controlled Conveyor System",             10),
        (50, 325, "Programmed Siemens S7-1200 via TIA Portal.", 10),

        # INTERNSHIPS section
        (50, 360, "INTERNSHIPS",                                11),
        (50, 380, "Summer Internship — Schneider Electric",     10),
        (50, 395, "Worked on SCADA HMI screens with WinCC.",    10),
    ]
    return _make_pdf_at(pdf_path, blocks)


@pytest.fixture
def two_column_pdf(tmp_path) -> str:
    """
    A two-column CV layout.
    Left column (x ≈ 50) holds Skills and Education.
    Right column (x ≈ 320) holds Projects and Internships.

    The test that uses this fixture verifies that left-column content
    appears BEFORE right-column content in the extracted output —
    i.e., we don't interleave the two columns.
    """
    pdf_path = str(tmp_path / "two_column.pdf")
    blocks = [
        # ── LEFT COLUMN ──────────────────────────────────────
        (50, 80,  "SKILLS",                                    11),
        (50, 100, "MATLAB, Python, STM32",                     10),
        (50, 115, "SolidWorks, AutoCAD, ANSYS",                10),

        (50, 150, "EDUCATION",                                  11),
        (50, 170, "B.Sc. Mechatronics, AUC 2023",              10),

        # ── RIGHT COLUMN ─────────────────────────────────────
        # x=320 puts these firmly in the right half of an A4 page (width=595)
        (320, 80,  "PROJECTS",                                  11),
        (320, 100, "Delta Robot Arm — used ROS2 and MoveIt!",   10),
        (320, 115, "PID controller tuned in Simulink",          10),

        (320, 150, "INTERNSHIPS",                               11),
        (320, 170, "ABB PLC intern — programmed FBD logic",     10),
    ]
    return _make_pdf_at(pdf_path, blocks)


@pytest.fixture
def missing_sections_pdf(tmp_path) -> str:
    """
    A student CV that only has Skills and Education.
    No Projects section, no Internships section.
    The parser should return empty lists for the missing sections, not crash.
    """
    pdf_path = str(tmp_path / "missing_sections.pdf")
    blocks = [
        (50, 50,  "Jane Smith — Final Year Student",    12),
        (50, 90,  "SKILLS",                             11),
        (50, 110, "SolidWorks, AutoCAD, Python, MATLAB",10),
        (50, 145, "EDUCATION",                          11),
        (50, 165, "B.Sc. Mechanical Engineering, 2024", 10),
    ]
    return _make_pdf_at(pdf_path, blocks)


@pytest.fixture
def nonstandard_headers_pdf(tmp_path) -> str:
    """
    A CV that uses non-standard section header names.
    These should still map to the four canonical sections via SECTION_SYNONYMS:
      "Technical Skills"  → skills
      "Work Experience"   → internships
      "Academic Background" → education
      "Engineering Projects" → projects
    """
    pdf_path = str(tmp_path / "nonstandard_headers.pdf")
    blocks = [
        (50, 50,  "Ahmed Hassan — Embedded Systems Engineer",  12),

        (50, 90,  "Technical Skills",                          11),
        (50, 110, "STM32, FreeRTOS, CAN Bus, I2C, SPI, UART", 10),

        (50, 145, "Academic Background",                        11),
        (50, 165, "B.Sc. Electronics Engineering, Cairo Uni",  10),

        (50, 200, "Engineering Projects",                       11),
        (50, 220, "STM32-based Motor Controller with PID loop", 10),

        (50, 255, "Work Experience",                            11),
        (50, 275, "Internship at Valeo — AUTOSAR, CAN Bus",     10),
    ]
    return _make_pdf_at(pdf_path, blocks)


@pytest.fixture
def no_keywords_pdf(tmp_path) -> str:
    """
    A CV with valid sections but zero Mechatronics keywords.
    The keyword lists should all be empty — the parser shouldn't crash.
    """
    pdf_path = str(tmp_path / "no_keywords.pdf")
    blocks = [
        (50, 50,  "Generic Resume",                  12),
        (50, 90,  "SKILLS",                          11),
        (50, 110, "Communication and teamwork skills",10),
        (50, 125, "Fast learner, hardworking",        10),
        (50, 160, "EDUCATION",                        11),
        (50, 180, "B.A. Business Administration",     10),
    ]
    return _make_pdf_at(pdf_path, blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Sample PDF path helpers (used in integration tests)
# ──────────────────────────────────────────────────────────────────────────────

def sample_pdf(filename: str) -> str:
    """
    Returns the absolute path to one of the real sample CVs.
    Used by integration tests that need actual PDF content.
    """
    return os.path.join(SAMPLES_DIR, filename)
