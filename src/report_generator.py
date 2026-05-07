"""
report_generator.py — Career Roadmap PDF Export

This module assembles a career roadmap PDF using reportlab's PLATYPUS framework.
The PDF includes:
  1. Top-3 niche ranking table (rank, niche name, fit score)
  2. Skills gap section for the #1 niche (present / missing skills)
  3. Embedded radar chart PNG image

Everything is generated in-memory (io.BytesIO) — no files are written to disk.
This keeps the module stateless and safe to call from concurrent web handlers.

Contract reference: specs/005-skills-gap-analytics/contracts/
Data model:         specs/005-skills-gap-analytics/data-model.md
"""

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def generate_career_roadmap_pdf(
    fit_score_result: dict,
    skills_gap_result: dict,
    chart_image_bytes: bytes,
    skills_gap_results: dict = None,
) -> dict:
    """
    Assembles a career roadmap PDF from the scoring and gap analysis results.

    The report is structured in three sections:
      1. Top Niche Recommendations — a ranked table of the user's top 3 niches
         with their composite fit scores, making the primary recommendation clear.
      2. Skills Gap Analysis — lists which top-10 benchmark skills for the #1 niche
         the user already has (present) and which they need to develop (missing).
      3. Radar Chart — the 6-spoke proficiency chart embedded as a PNG image,
         giving a visual overview of strength across all 6 niches.

    All content is built in-memory using reportlab PLATYPUS. The resulting PDF
    bytes are returned directly — never written to disk.

    The function never raises — all failures are returned as error dicts so the
    calling pipeline can handle them without try/except.

    Parameters
    ----------
    fit_score_result : dict
        A FitScoreResult dict from compute_fit_score().
    skills_gap_result : dict
        A SkillsGapResult dict from analyze_skills_gap().
    chart_image_bytes : bytes
        PNG bytes of the radar chart from generate_radar_chart().

    Returns
    -------
    dict
        On success — CareerRoadmapReport:
          {
            "pdf_bytes":     bytes,  # the full PDF file content
            "filename":      str,    # suggested filename, e.g. "career_roadmap_20260418_120000.pdf"
            "generated_at":  str,    # ISO 8601 UTC timestamp
          }
        On failure — error dict:
          {"error": "<ERROR_CODE>", "message": "<explanation>"}
    """
    try:
        # ── Input validation ──────────────────────────────────────────────────
        if not isinstance(fit_score_result, dict) or "error" in fit_score_result:
            return {
                "error": "INVALID_INPUT",
                "message": "fit_score_result is an error dict or invalid",
            }
        if not isinstance(skills_gap_result, dict) or "error" in skills_gap_result:
            return {
                "error": "INVALID_INPUT",
                "message": "skills_gap_result is an error dict or invalid",
            }
        if not chart_image_bytes:
            return {
                "error": "INVALID_INPUT",
                "message": "chart_image_bytes is empty",
            }

        # ── Build PDF in memory ───────────────────────────────────────────────
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Mechatronics Career Roadmap", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        # ── Section 1: Top Niche Recommendations ─────────────────────────────
        # Show the top 3 niches in a styled table. The rank column makes
        # priority immediately visible; the fit score gives a numeric basis.
        story.append(Paragraph("Top Niche Recommendations", styles["Heading2"]))
        ranked = fit_score_result.get("ranked_niches", [])
        top3_rows = [["Rank", "Niche", "Fit Score"]]
        for entry in ranked[:3]:
            top3_rows.append([
                str(entry.get("rank", "")),
                entry.get("niche", "").replace("_", " ").title(),
                f"{entry.get('composite_score', 0):.2f}",
            ])

        tbl = Table(top3_rows, colWidths=[2 * cm, 8 * cm, 4 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

        # ── Section 2: Skills Gap Analysis ───────────────────────────────────
        # When skills_gap_results (a dict of niche → gap result) is provided
        # we render a sub-section for each of the top 3 niches so the user can
        # compare their readiness across all realistic career options.
        # If the dict is absent we fall back to the single top-niche result for
        # backward compatibility with older callers.
        story.append(Paragraph("Skills Gap Analysis — Top 3 Niches", styles["Heading2"]))

        # Build the ordered list of (niche_key, gap_result) pairs to render.
        # Preserve the ranking order by walking the ranked_niches list instead
        # of iterating the dict, which has no guaranteed order.
        ranked = fit_score_result.get("ranked_niches", [])
        if skills_gap_results:
            # Pull the top 3 in rank order; skip any niche that didn't produce
            # a valid gap result (e.g. if the artifact is missing benchmarks).
            gap_pairs = [
                (entry["niche"], skills_gap_results[entry["niche"]])
                for entry in ranked[:3]
                if entry["niche"] in skills_gap_results
            ]
        else:
            # Fallback: single result from the legacy parameter
            gap_pairs = [(skills_gap_result.get("target_niche", "top"), skills_gap_result)]

        for rank_idx, (niche_key, gap) in enumerate(gap_pairs, start=1):
            niche_label = niche_key.replace("_", " ").title()
            story.append(
                Paragraph(f"#{rank_idx} — {niche_label}", styles["Heading3"])
            )
            coverage = gap.get("coverage_percentage", 0.0)
            story.append(
                Paragraph(
                    f"Coverage: {coverage}% of benchmark skills present",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 0.2 * cm))

            present = gap.get("present_skills", [])
            missing = gap.get("missing_skills", [])

            if present:
                story.append(Paragraph("Present skills:", styles["Normal"]))
                for skill in present:
                    story.append(Paragraph(f"\u2022 {skill}", styles["Normal"]))
                story.append(Spacer(1, 0.15 * cm))

            if missing:
                story.append(Paragraph("Missing skills to develop:", styles["Normal"]))
                for skill in missing:
                    story.append(Paragraph(f"\u2022 {skill}", styles["Normal"]))
            else:
                story.append(
                    Paragraph(
                        "No missing skills identified for this niche.",
                        styles["Normal"],
                    )
                )
            story.append(Spacer(1, 0.4 * cm))

        story.append(Spacer(1, 0.1 * cm))

        # ── Section 3: Radar Chart ────────────────────────────────────────────
        # Embed the PNG as an Image flowable. We wrap chart_image_bytes in a
        # BytesIO so reportlab can read it without any disk I/O.
        story.append(Paragraph("Niche Proficiency Radar Chart", styles["Heading2"]))
        chart_buf = io.BytesIO(chart_image_bytes)
        img = Image(chart_buf, width=300, height=300)
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))

        # Footer: timestamp so the user knows when the report was generated.
        generated_at = datetime.now(timezone.utc).isoformat()
        story.append(Paragraph(f"Generated: {generated_at}", styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        pdf_bytes = buf.read()

        # Build a timestamped filename for the suggested download name.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"career_roadmap_{ts}.pdf"

        return {
            "pdf_bytes":    pdf_bytes,
            "filename":     filename,
            "generated_at": generated_at,
        }

    except Exception as e:
        # Safety net — PDF generation must never crash the calling pipeline.
        return {"error": "RENDER_ERROR", "message": str(e)}
