"""
test_phase5_pipeline.py — Integration Tests for Phase 5 Pipeline

These tests exercise the full Phase 5 chain end-to-end using a fixture model
artifact (with niche_top_skills). They verify that:
  1. analyze_skills_gap() produces a valid SkillsGapResult
  2. generate_radar_chart() with the scorer result produces a valid RadarChartResult
  3. generate_career_roadmap_pdf() with all three produces a valid CareerRoadmapReport

We do NOT load the real on-disk .joblib model here — the existing artifact predates
Phase 5 and would fail load_model() validation. Integration tests use a fixture
artifact built in-memory so they are self-contained and fast.

Spec reference: specs/005-skills-gap-analytics/tasks.md (T023)
"""

import pytest

from src.feature_vector import VECTOR_INDEX_ORDER
from src.gap_analyzer import analyze_skills_gap
from src.model_io import CANONICAL_NICHES
from src.report_generator import generate_career_roadmap_pdf
from src.visualizer import generate_radar_chart


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fixture_artifact():
    """
    A fully in-memory model artifact with both niche_centroids (Phase 4) and
    niche_top_skills (Phase 5). Designed to be self-contained — no disk reads,
    no trained classifier needed for gap analysis.

    Each niche gets unique top-10 dimensions based on its position in
    CANONICAL_NICHES so tests can distinguish per-niche behavior.
    """
    niche_top_skills = {}
    niche_centroids = {}
    for i, niche in enumerate(CANONICAL_NICHES):
        # Build a synthetic centroid: niche_idx * 4 is the "dominant" dimension,
        # matching the training data pattern used by the unit tests.
        centroid = [0.0] * 40
        centroid[i * 4] = 10.0  # Strong signal at the niche's unique dimension
        niche_centroids[niche] = centroid

        # Top-10 = ranked by centroid value descending. With only one nonzero
        # dimension, the rest are all 0.0 ties — we break ties by index.
        ranked = sorted(enumerate(centroid), key=lambda x: x[1], reverse=True)
        niche_top_skills[niche] = [
            (idx, f"{VECTOR_INDEX_ORDER[idx][0]}_{VECTOR_INDEX_ORDER[idx][1]}")
            for idx, _ in ranked[:10]
        ]

    return {
        "niche_top_skills": niche_top_skills,
        "niche_centroids":  niche_centroids,
    }


@pytest.fixture
def fit_score_result_fixture():
    """
    A valid FitScoreResult for use in visualizer and report generator steps.
    Scores are spread across the range 0.3–0.8 so the radar chart has visible variation.
    """
    return {
        "ranked_niches": [
            {
                "niche":            niche,
                "composite_score":  0.3 + i * 0.1,
                "cv_component":     0.4 + i * 0.05,
                "intent_component": 0.5,
                "penalty":          0.0,
                "rank":             i + 1,
            }
            for i, niche in enumerate(CANONICAL_NICHES)
        ],
        "pivot_applied": False,
        "fallback_used": False,
    }


@pytest.fixture
def known_feature_vector():
    """
    A feature vector with signal in specific dimensions, designed to produce
    predictable present/missing results when analyzed against any niche.

    Dimensions 0, 4, 8, 12, 16, 20 are active (value = 1).
    All others are 0. This matches the unique-dimension pattern of fixture_artifact.
    """
    v = [0] * 40
    for d in [0, 4, 8, 12, 16, 20]:
        v[d] = 1
    return v


# ── Integration tests ─────────────────────────────────────────────────────────

class TestPhase5Pipeline:
    """Full Phase 5 pipeline: gap analysis → radar chart → PDF report."""

    def test_gap_analysis_step_succeeds(
        self, known_feature_vector, fixture_artifact
    ):
        """
        Step 1: analyze_skills_gap() with the fixture artifact and a known vector
        must return a valid SkillsGapResult with no error.
        """
        result = analyze_skills_gap(
            known_feature_vector, CANONICAL_NICHES[0], fixture_artifact
        )
        assert "error" not in result, (
            f"Gap analysis step failed: {result}"
        )
        assert result["target_niche"] == CANONICAL_NICHES[0]
        assert result["benchmark_size"] == 10
        total = len(result["present_skills"]) + len(result["missing_skills"])
        assert total == 10, (
            f"present + missing must == 10, got {total}"
        )
        assert 0.0 <= result["coverage_percentage"] <= 100.0

    def test_radar_chart_step_succeeds(self, fit_score_result_fixture):
        """
        Step 2: generate_radar_chart() with the fit score result must return
        a valid RadarChartResult containing PNG bytes.
        """
        result = generate_radar_chart(fit_score_result_fixture)
        assert "error" not in result, (
            f"Radar chart step failed: {result}"
        )
        assert isinstance(result["chart_image_bytes"], bytes)
        assert result["chart_image_bytes"][:4] == b"\x89PNG", (
            "Radar chart must be a valid PNG"
        )
        assert result["chart_format"] == "png"
        assert len(result["niche_labels"]) == 6
        assert len(result["scores"]) == 6

    def test_pdf_report_step_succeeds(
        self, known_feature_vector, fixture_artifact, fit_score_result_fixture
    ):
        """
        Step 3: generate_career_roadmap_pdf() with all three pipeline outputs
        must return a valid CareerRoadmapReport containing a real PDF.
        """
        # Run the full chain
        gap_result = analyze_skills_gap(
            known_feature_vector, CANONICAL_NICHES[0], fixture_artifact
        )
        chart_result = generate_radar_chart(fit_score_result_fixture)
        pdf_result = generate_career_roadmap_pdf(
            fit_score_result_fixture,
            gap_result,
            chart_result["chart_image_bytes"],
        )

        assert "error" not in pdf_result, (
            f"PDF generation step failed: {pdf_result}"
        )
        assert isinstance(pdf_result["pdf_bytes"], bytes)
        assert pdf_result["pdf_bytes"][:4] == b"%PDF", (
            "PDF must start with '%PDF' header"
        )
        assert pdf_result["filename"].startswith("career_roadmap_")
        assert pdf_result["filename"].endswith(".pdf")

    def test_full_pipeline_is_deterministic_for_same_input(
        self, known_feature_vector, fixture_artifact
    ):
        """
        Running the full pipeline twice with identical inputs must produce
        gap results with the same present_skills and missing_skills.
        The PDF bytes may differ (timestamps), but the analysis content must not.
        """
        target_niche = CANONICAL_NICHES[0]

        gap1 = analyze_skills_gap(known_feature_vector, target_niche, fixture_artifact)
        gap2 = analyze_skills_gap(known_feature_vector, target_niche, fixture_artifact)

        assert "error" not in gap1 and "error" not in gap2
        assert gap1["present_skills"] == gap2["present_skills"], (
            "present_skills must be identical across repeated calls with same input"
        )
        assert gap1["missing_skills"] == gap2["missing_skills"], (
            "missing_skills must be identical across repeated calls with same input"
        )
        assert gap1["coverage_percentage"] == gap2["coverage_percentage"]

    def test_all_6_niches_work_end_to_end(
        self, known_feature_vector, fixture_artifact
    ):
        """
        The pipeline must work for all 6 canonical niches — each call to
        analyze_skills_gap() with a different target_niche must succeed.
        """
        for niche in CANONICAL_NICHES:
            gap_result = analyze_skills_gap(
                known_feature_vector, niche, fixture_artifact
            )
            assert "error" not in gap_result, (
                f"Gap analysis failed for niche '{niche}': {gap_result}"
            )
            assert gap_result["target_niche"] == niche
            total = (
                len(gap_result["present_skills"])
                + len(gap_result["missing_skills"])
            )
            assert total == 10, (
                f"present + missing must == 10 for niche '{niche}', got {total}"
            )

    def test_pipeline_with_zero_vector_produces_valid_pdf(
        self, fixture_artifact, fit_score_result_fixture
    ):
        """
        A user with no CV signal (all-zero vector) must still produce a complete
        PDF with 100% missing skills — the pipeline handles this edge case.
        """
        zero_vector = [0] * 40
        gap_result = analyze_skills_gap(zero_vector, CANONICAL_NICHES[0], fixture_artifact)
        assert "error" not in gap_result
        assert len(gap_result["missing_skills"]) == 10
        assert gap_result["coverage_percentage"] == 0.0

        chart_result = generate_radar_chart(fit_score_result_fixture)
        assert "error" not in chart_result

        pdf_result = generate_career_roadmap_pdf(
            fit_score_result_fixture, gap_result, chart_result["chart_image_bytes"]
        )
        assert "error" not in pdf_result, (
            f"PDF generation failed for zero-vector user: {pdf_result}"
        )
        assert pdf_result["pdf_bytes"][:4] == b"%PDF"
