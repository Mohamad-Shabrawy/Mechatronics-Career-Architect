"""
parser.py — Phase 1: Advanced NLP Text Extraction & Layout Analysis
===================================================================

This is the brain of Phase 1. It's the only file that other phases need to
import — specifically the single public function: parse_cv().

What it does, from start to finish:
  1. Opens a PDF file using PyMuPDF (fitz)
  2. Extracts every text block from every page, capturing their exact positions
  3. Detects whether the page layout is single-column or double-column
  4. Sorts the blocks into the correct human reading order
     (left column top-to-bottom, then right column top-to-bottom)
  5. Walks through the sorted blocks and assigns each one to a CV section
     (Skills, Projects, Education, or Internships)
  6. Scans each block's text for recognized Mechatronics keywords
  7. Returns a clean, JSON-serializable dict that the downstream phases consume

Contract reference: specs/001-nlp-text-extraction/contracts/parser_contract.md
Data model:        specs/001-nlp-text-extraction/data-model.md

Author: Phase 1 implementation
"""

import os
import re
import json

import fitz  # PyMuPDF — the library that reads PDF files

from src.lexicon import MECHATRONICS_KEYWORDS
from src.synonyms import SECTION_SYNONYMS

# Only expose parse_cv to callers — internal helpers stay private
__all__ = ["parse_cv"]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def parse_cv(file_path: str) -> dict:
    """
    The single public entry point for Phase 1.

    Accepts a path to a PDF file and returns a structured dict with four keys:
        {
            "skills":      [{"text": str, "keywords": list[str]}, ...],
            "projects":    [{"text": str, "keywords": list[str]}, ...],
            "education":   [{"text": str, "keywords": list[str]}, ...],
            "internships": [{"text": str, "keywords": list[str]}, ...],
        }

    On any failure, returns an error dict instead of raising:
        {
            "error":   "FILE_NOT_FOUND" | "INVALID_PDF" | "NO_TEXT_CONTENT",
            "message": "Human-readable explanation of what went wrong",
        }

    Behavioral guarantee: this function NEVER raises an exception to the caller.
    """

    # ── Step 1: Open the PDF ─────────────────────────────────────────────────
    # On Windows, fitz.open() doesn't always raise FileNotFoundError for a
    # missing path — it can throw a different internal exception.
    # So we do an explicit existence check first: cleaner and cross-platform.
    if not file_path or not os.path.isfile(file_path):
        return {
            "error": "FILE_NOT_FOUND",
            "message": f"File not found: {file_path}",
        }

    try:
        doc = fitz.open(file_path)
    except Exception:
        # fitz raises different exceptions for different corruption scenarios —
        # we catch them all with a generic handler and report INVALID_PDF
        return {
            "error": "INVALID_PDF",
            "message": "The file is not a valid PDF or is corrupted.",
        }

    # ── Step 2: Extract all text blocks from every page ──────────────────────
    # We ask fitz for structured output ("dict" mode) which gives us each text
    # block with its bounding box coordinates (x0, y0, x1, y1).
    # Those coordinates are crucial for detecting column layout in Step 3.
    all_blocks = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width  # how wide is this page in points?

            # get_text("dict") returns a dict with a "blocks" list.
            # Each block is a rectangular region of text on the page.
            raw_page_data = page.get_text("dict")

            for block in raw_page_data.get("blocks", []):
                # Blocks can be type 0 (text) or type 1 (image).
                # We only care about text blocks — skip images silently.
                if block.get("type") != 0:
                    continue

                # Each block contains "lines", each line contains "spans",
                # each span contains a chunk of text with font info.
                # We just want the raw text, so we join everything together.
                block_lines = []
                for line in block.get("lines", []):
                    # A line can have multiple spans (e.g. bold word + regular word)
                    # We join the spans with a space to form a single readable line
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if line_text:
                        block_lines.append(line_text)

                block_text = "\n".join(block_lines).strip()

                # Skip blocks that turned out to have no actual text
                if not block_text:
                    continue

                # The bbox is a 4-tuple: (x0, y0, x1, y1) in page points.
                # We'll use x0 (left edge) to decide which column this block is in.
                bbox = block["bbox"]

                all_blocks.append({
                    "text":       block_text,
                    "x0":         bbox[0],
                    "y0":         bbox[1],
                    "x1":         bbox[2],
                    "y1":         bbox[3],
                    "page_num":   page_num,
                    "page_width": page_width,
                })
    finally:
        # Always close the document to free memory, even if something went wrong above
        doc.close()

    # ── Step 3: Check that we actually got some text ─────────────────────────
    # If all_blocks is empty at this point, the PDF is either completely blank
    # or it's a scanned/image-only PDF with no embedded text layer.
    if not all_blocks:
        return {
            "error": "NO_TEXT_CONTENT",
            "message": (
                "No extractable text found. "
                "Scanned or image-only PDFs are not supported in Phase 1."
            ),
        }

    # ── Step 4: Detect column layout and sort blocks in reading order ─────────
    # For a two-column CV, we want to read the entire left column before
    # moving to the right column — not interleave them.
    ordered_blocks = _sort_blocks_by_reading_order(all_blocks)

    # ── Step 5: Segment blocks into the four CV sections ─────────────────────
    # This walks through the ordered blocks and assigns each one to Skills,
    # Projects, Education, or Internships based on section headers.
    structured_output = _segment_blocks(ordered_blocks)

    return structured_output


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# (These are internal implementation details — callers should only use parse_cv)
# ══════════════════════════════════════════════════════════════════════════════

def _sort_blocks_by_reading_order(blocks: list) -> list:
    """
    Takes a flat list of text blocks and returns them in the order a human
    would read them.

    For a single-column page: just sort by (page number, vertical position y0).

    For a two-column page: all left-column blocks first (sorted by y0),
    then all right-column blocks (sorted by y0). This way we don't
    interleave content from both columns.

    We detect two-column layout per page: if any block on a page starts
    beyond 40% of the page width, we treat the whole page as two-column.
    The split point between left and right columns is the page midpoint (50%).
    """
    # Group blocks by page number first — column detection is per-page
    pages: dict = {}
    for block in blocks:
        pn = block["page_num"]
        if pn not in pages:
            pages[pn] = []
        pages[pn].append(block)

    ordered = []

    for page_num in sorted(pages.keys()):
        page_blocks = pages[page_num]

        # Get the page width from the first block on this page
        page_width = page_blocks[0]["page_width"] if page_blocks else 595

        # Two-column detection: if any block's left edge (x0) starts more than
        # 40% across the page width, we assume a two-column layout.
        # On a 595pt wide A4 page, 40% = ~238pt. A right column in a two-column
        # CV typically starts around 300pt (halfway). Single-column CVs put ALL
        # text near the left margin (x0 < 100pt or so), well under 40%.
        has_right_column = any(b["x0"] > page_width * 0.40 for b in page_blocks)

        if has_right_column:
            # Two-column layout — process left column first, then right column.
            # The split is at the exact midpoint of the page.
            midpoint = page_width / 2

            left_blocks  = [b for b in page_blocks if b["x0"] < midpoint]
            right_blocks = [b for b in page_blocks if b["x0"] >= midpoint]

            # Within each column, sort top-to-bottom by vertical position (y0)
            left_blocks.sort(key=lambda b: b["y0"])
            right_blocks.sort(key=lambda b: b["y0"])

            # Left column content always comes before right column content
            ordered.extend(left_blocks)
            ordered.extend(right_blocks)
        else:
            # Single-column layout — simple top-to-bottom sort
            page_blocks.sort(key=lambda b: b["y0"])
            ordered.extend(page_blocks)

    return ordered


def _detect_section_header(line: str) -> str | None:
    """
    Checks whether a single line of text is a CV section header.

    Returns the canonical section name ("skills", "projects", "education",
    "internships") if the line matches, or None if it's just body text.

    How we avoid false positives:
    1. We lowercase and strip the line, then remove trailing colons
       (so "SKILLS:" becomes "skills" which matches the synonym "skills")
    2. We reject anything longer than 60 characters — section headers are
       always short. "Experience working with PLC for 3 years" won't match
       even though it starts with "experience".
    3. We only match EXACT strings against the synonym list — no substring
       matching — so "Work Experience Summary" won't match "work experience".
    """
    # Normalize the line: strip whitespace, lowercase, remove trailing colon
    normalized = line.strip().lower().rstrip(":").strip()

    if not normalized:
        return None

    # Headers are short by nature — if it's longer than 60 chars it's body text
    if len(normalized) > 60:
        return None

    # Check if the normalized line exactly matches any known synonym
    for section_name, synonyms in SECTION_SYNONYMS.items():
        if normalized in synonyms:
            return section_name

    return None


def _recognize_keywords(text: str) -> list[str]:
    """
    Scans a block of text and returns all recognized Mechatronics keywords
    found in it. The result is sorted alphabetically and deduplicated.

    Matching is:
    - Case-insensitive: "MATLAB" and "matlab" are the same
    - Whole-word only: we use (?<!\\w) and (?!\\w) word boundary assertions
      so "micropython" does NOT match "python", and "catia" does NOT match "c"

    Why (?<!\\w) instead of \\b?
    Some keywords end with special characters like ")" — for example
    "adc (analog-to-digital converter)". The \\b assertion only works between
    a word character and a non-word character, so it fails right before "(".
    Using (?<!\\w) / (?!\\w) lookbehind/lookahead handles these correctly.

    Returns a sorted list so the output is deterministic (same text always
    produces the same ordered list — important for Phase 3 feature vectors).
    """
    found: set = set()
    text_lower = text.lower()

    for keyword in MECHATRONICS_KEYWORDS:
        # Build a pattern that matches the keyword as a complete unit,
        # not as a substring embedded inside a larger word.
        # re.escape() handles special chars like "+", "(", ".", "*" safely.
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"

        if re.search(pattern, text_lower):
            # We always store the lowercased canonical keyword string
            found.add(keyword)

    # Sort for deterministic output — sets are unordered, lists must be consistent
    return sorted(found)


def _segment_blocks(ordered_blocks: list) -> dict:
    """
    The core segmentation logic. Takes a list of text blocks in reading order
    and groups them under the correct CV section keys.

    How it works:
    - We walk through the blocks one by one.
    - If a block's text looks like a section header (e.g. "SKILLS" or
      "Work Experience"), we switch our "current_section" tracker.
    - All subsequent blocks go into that section until the next header appears.
    - Blocks that come before the first recognized header are discarded
      (they're usually the candidate's name, contact info, etc.).

    For blocks that contain BOTH a header line AND content on subsequent lines
    (e.g. a block whose first line is "SKILLS" and second line is "Python, MATLAB"),
    we split them: the header switches the section, the remaining lines become content.

    Returns a dict with all four keys always present. Sections not found in the
    PDF are empty lists [].
    """
    # Initialize all four sections as empty lists.
    # They stay empty if the PDF never has a matching header.
    result = {
        "skills":      [],
        "projects":    [],
        "education":   [],
        "internships": [],
    }

    # We start with no active section. Blocks before the first recognized header
    # are discarded (they're typically the name, contact info, summary text).
    current_section: str | None = None

    for block in ordered_blocks:
        block_text = block["text"]
        lines = block_text.splitlines()

        # --- Check if this block contains a section header ---
        # We check the first few lines because headers are usually at the top
        # of their block. We stop at the first header we find.
        header_line_index: int | None = None
        detected_section: str | None = None

        for i, line in enumerate(lines):
            section = _detect_section_header(line)
            if section is not None:
                header_line_index = i
                detected_section = section
                break  # Only the first header in a block counts

        if detected_section is not None:
            # This block starts with (or is entirely) a section header.
            # Switch the active section tracker.
            current_section = detected_section

            # Now check if there are content lines BELOW the header in this same block.
            # Example: a block that says "SKILLS\nPython, MATLAB" — the "Python, MATLAB"
            # part should go into the skills section as content.
            content_lines = [
                line for i, line in enumerate(lines)
                if i != header_line_index and line.strip()
            ]

            if content_lines and current_section is not None:
                content_text = "\n".join(content_lines).strip()
                keywords = _recognize_keywords(content_text)
                result[current_section].append({
                    "text":     content_text,
                    "keywords": keywords,
                })

        elif current_section is not None:
            # No header found in this block — it's a content block.
            # Add it to whichever section we're currently inside.
            keywords = _recognize_keywords(block_text)
            result[current_section].append({
                "text":     block_text,
                "keywords": keywords,
            })

        # If current_section is None and no header was found:
        # this block is pre-header content (name, contact info, etc.) — discard it.

    return result
