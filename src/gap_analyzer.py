"""
gap_analyzer.py — Skills Gap Analysis for Target Niche

This module answers the question: "What skills do I need to get from where I am
to where I want to be?" It compares a user's feature vector against the top-10
most important skill dimensions for a target niche and tells the user which skills
they already have (present) and which they're missing.

The core insight: a feature vector entry > 0 means that skill type appeared in
the user's CV. The niche_top_skills from the model artifact tells us which
dimensions matter most for a given niche. Mapping those dimensions to human-readable
benchmark names from niche_benchmarks.json gives the user actionable labels.

Positional mapping contract: rank-0 in niche_top_skills (highest centroid value)
maps to rank-0 in the benchmark JSON's "top_skills" list (rank: 1). Both lists are
ordered by importance descending, so index i in niche_top_skills aligns with index i
in the JSON benchmark. This means the user sees "PLC Programming" as their top gap,
not the raw internal label "plc_hardware_skills".

Contract reference: specs/005-skills-gap-analytics/contracts/
Data model:         specs/005-skills-gap-analytics/data-model.md
"""

import json
import pathlib

from src.model_io import CANONICAL_NICHES

# ── Niche name mapping ────────────────────────────────────────────────────────
# niche_benchmarks.json uses Title Case keys; CANONICAL_NICHES uses snake_case.
# This mapping bridges the two naming conventions so we can look up benchmark
# skill names by canonical niche name.
_JSON_TO_CANONICAL = {
    "Industrial Automation": "industrial_automation",
    "Robotics": "robotics",
    "Embedded Systems": "embedded_systems",
    "Automotive": "automotive",
    "Mechanical Design": "mechanical_design",
    "Technical Management": "technical_management",
}

# ── Benchmark data loading ────────────────────────────────────────────────────
# Load the benchmark JSON at module import time. Any I/O failure falls back to
# an empty dict — gap analysis still works, using the raw internal labels
# instead of human-readable names. Better a degraded result than a crash.

_BENCHMARKS_PATH = pathlib.Path(__file__).parent.parent / "data" / "niche_benchmarks.json"


def _load_benchmark_labels() -> dict[str, list[str]]:
    """
    Reads niche_benchmarks.json and returns a dict mapping each canonical niche
    to its ordered list of 10 human-readable skill names.

    The order matches the "rank" field in the JSON (rank 1 = index 0). This
    aligns with niche_top_skills where index 0 = highest centroid dimension,
    so index i in both lists refers to the same importance rank.

    Returns an empty dict on any I/O or parsing failure — callers fall back
    to raw internal labels rather than crashing.
    """
    try:
        with open(_BENCHMARKS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        result = {}
        for json_niche, canonical in _JSON_TO_CANONICAL.items():
            skills = raw.get(json_niche, {}).get("top_skills", [])
            # Sort by "rank" field just in case the JSON order drifts, then extract name
            skills_sorted = sorted(skills, key=lambda s: s.get("rank", 999))
            result[canonical] = [s["skill"] for s in skills_sorted[:10]]
        return result
    except Exception:
        # Graceful fallback — gap analyzer will still work with raw internal labels
        return {}


_BENCHMARK_LABELS: dict[str, list[str]] = _load_benchmark_labels()


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def analyze_skills_gap(
    feature_vector: list,
    target_niche: str,
    model_artifact: dict,
) -> dict:
    """
    Compares a user's CV feature vector against the top-10 skill dimensions for
    the target niche and returns which skills are present and which are missing.

    The function never raises — all failures are returned as error dicts so that
    the calling pipeline can handle them gracefully without try/except blocks.

    Parameters
    ----------
    feature_vector : list
        A 40-element list of numeric values from the user's enriched CV.
        Each element corresponds to one (entity_type, section) dimension.
        Values > 0 mean the user's CV contains that skill type in that section.
    target_niche : str
        One of the 6 canonical niche names (snake_case, e.g. "robotics").
    model_artifact : dict
        A loaded model artifact dict containing "niche_top_skills" — a dict
        mapping each niche to its top-10 centroid dimensions.

    Returns
    -------
    dict
        On success — SkillsGapResult:
          {
            "target_niche":        str,         # e.g. "robotics"
            "present_skills":      list[str],   # benchmark skills the user has
            "missing_skills":      list[str],   # benchmark skills the user lacks
            "coverage_percentage": float,       # 0.0–100.0
            "benchmark_size":      int,         # always 10
          }
        On failure — error dict:
          {"error": "<ERROR_CODE>", "message": "<human-readable explanation>"}
    """
    try:
        # ── Validation: feature vector length ────────────────────────────────
        if len(feature_vector) != 40:
            return {
                "error": "INVALID_VECTOR",
                "message": (
                    f"feature_vector must have exactly 40 elements, "
                    f"got {len(feature_vector)}"
                ),
            }

        # ── Validation: target niche ──────────────────────────────────────────
        if target_niche not in CANONICAL_NICHES:
            return {
                "error": "INVALID_NICHE",
                "message": (
                    f"'{target_niche}' is not a valid canonical niche. "
                    f"Must be one of: {CANONICAL_NICHES}"
                ),
            }

        # ── Validation: artifact has niche_top_skills ─────────────────────────
        if "niche_top_skills" not in model_artifact:
            return {
                "error": "MISSING_BENCHMARK",
                "message": (
                    "model_artifact is missing 'niche_top_skills'. "
                    "Retrain the model with Phase 5-compatible trainer.py."
                ),
            }

        # ── Core analysis ─────────────────────────────────────────────────────
        # Get the top-10 (dim_index, raw_label) tuples for this niche.
        # Each tuple identifies a feature vector dimension that is most diagnostic
        # for this niche, ordered by centroid value descending.
        benchmark = model_artifact["niche_top_skills"][target_niche]

        # Get human-readable labels from the loaded benchmark JSON.
        # Falls back to raw internal labels (e.g. "plc_hardware_skills") if the
        # benchmark file was unavailable at import time.
        hr_labels = _BENCHMARK_LABELS.get(target_niche, [])

        present_skills: list[str] = []
        missing_skills: list[str] = []

        for i, (idx, raw_label) in enumerate(benchmark):
            # Use the human-readable name if available at this rank position;
            # otherwise fall back to the raw "<type>_<section>" label.
            label = hr_labels[i] if i < len(hr_labels) else raw_label

            # A feature vector value > 0 means the user has some CV signal for
            # this skill type in this section — they have the skill.
            if feature_vector[idx] > 0:
                present_skills.append(label)
            else:
                missing_skills.append(label)

        # Coverage = fraction of the top-10 benchmark skills the user already has.
        # Expressed as a percentage with one decimal place for readability.
        coverage_percentage = round(len(present_skills) / 10 * 100, 1)

        return {
            "target_niche":        target_niche,
            "present_skills":      present_skills,
            "missing_skills":      missing_skills,
            "coverage_percentage": coverage_percentage,
            "benchmark_size":      10,
        }

    except Exception as e:
        # Catch-all safety net — gap analysis must never propagate exceptions
        # to the calling pipeline. The error code lets the caller distinguish
        # this unexpected failure from the expected validation errors above.
        return {"error": "GAP_ERROR", "message": str(e)}
