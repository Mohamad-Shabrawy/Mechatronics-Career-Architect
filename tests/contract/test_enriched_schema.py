"""
test_enriched_schema.py — Phase 2 Contract Tests

These tests verify the SHAPE of what enrich_cv() returns, not the specific
content. They are the contract between Phase 2 and Phase 3 (the ML classifier):
if these tests pass, Phase 3 can safely assume the enriched output always has
the correct structure.

Tests here cover:
- The 7 required top-level keys are always present
- feature_vector is always exactly 40 elements long
- cluster_profile always has exactly 7 keys
- All 7 cluster keys are present in cluster_profile
- entities is a list of dicts with "term", "type", "section" keys
- Output is JSON-serializable
- Phase 1 error dicts pass through unchanged (no processing occurs)
- Empty Phase 1 input produces valid zero-valued enriched output with correct schema

Contract reference: specs/002-semantic-ner-features/contracts/enricher_contract.md
"""

import json
import pytest
from src.enricher import enrich_cv


# ──────────────────────────────────────────────────────────────────────────────
# Constants and helpers
# ──────────────────────────────────────────────────────────────────────────────

# The exact 7 keys we expect in every successful enriched response.
# Four are Phase 1 passthrough; three are new Phase 2 additions.
REQUIRED_ENRICHED_KEYS = {
    "skills", "projects", "education", "internships",
    "entities", "cluster_profile", "feature_vector",
}

# The exact 7 cluster keys that must always be present in cluster_profile.
# These are the six Mechatronics domain clusters plus the "unclassified" fallback.
REQUIRED_CLUSTER_KEYS = {
    "industrial_automation",
    "robotics",
    "embedded_systems",
    "automotive",
    "mechanical_design",
    "technical_management",
    "unclassified",
}

# An empty Phase 1 output — the minimal valid input to enrich_cv().
# All four section lists are empty, but all four keys are present.
EMPTY_PHASE1_OUTPUT = {
    "skills":      [],
    "projects":    [],
    "education":   [],
    "internships": [],
}

# A minimal Phase 1 output with one known entity to exercise the full pipeline.
# "s7-1200" is in ENTITY_TAXONOMY as plc_hardware and in SEMANTIC_MAP → industrial_automation.
MINIMAL_PHASE1_OUTPUT = {
    "skills": [{"text": "Programmed using S7-1200 and TIA Portal", "keywords": ["s7-1200"]}],
    "projects":    [],
    "education":   [],
    "internships": [],
}


def assert_valid_enriched_schema(result: dict) -> None:
    """
    Reusable assertion helper — checks the full schema of a successful
    enrich_cv() response. Called from multiple tests to avoid repetition.
    """
    # Must have exactly the 7 required keys
    assert set(result.keys()) == REQUIRED_ENRICHED_KEYS, (
        f"Expected keys {REQUIRED_ENRICHED_KEYS}, got {set(result.keys())}"
    )

    # Phase 1 sections must still be lists (content unchanged)
    for section in ["skills", "projects", "education", "internships"]:
        assert isinstance(result[section], list), (
            f"Section '{section}' must be a list, got {type(result[section])}"
        )

    # entities must be a list
    assert isinstance(result["entities"], list), (
        f"'entities' must be a list, got {type(result['entities'])}"
    )

    # Every entity record must have the three required keys with correct types
    for i, entity in enumerate(result["entities"]):
        assert isinstance(entity, dict), f"Entity {i} must be a dict"
        assert "term" in entity, f"Entity {i} missing 'term'"
        assert "type" in entity, f"Entity {i} missing 'type'"
        assert "section" in entity, f"Entity {i} missing 'section'"
        assert isinstance(entity["term"], str), f"Entity {i} 'term' must be str"
        assert isinstance(entity["type"], str), f"Entity {i} 'type' must be str"
        assert isinstance(entity["section"], str), f"Entity {i} 'section' must be str"

    # cluster_profile must have exactly 7 keys
    assert isinstance(result["cluster_profile"], dict), "'cluster_profile' must be a dict"
    assert len(result["cluster_profile"]) == 7, (
        f"'cluster_profile' must have exactly 7 keys, got {len(result['cluster_profile'])}"
    )
    assert set(result["cluster_profile"].keys()) == REQUIRED_CLUSTER_KEYS, (
        f"cluster_profile keys mismatch: {set(result['cluster_profile'].keys())}"
    )
    for cluster_key, count in result["cluster_profile"].items():
        assert isinstance(count, int), f"cluster_profile['{cluster_key}'] must be int"
        assert count >= 0, f"cluster_profile['{cluster_key}'] must be >= 0"

    # feature_vector must be a list of exactly 40 integers
    assert isinstance(result["feature_vector"], list), "'feature_vector' must be a list"
    assert len(result["feature_vector"]) == 40, (
        f"'feature_vector' must have exactly 40 elements, got {len(result['feature_vector'])}"
    )
    for i, val in enumerate(result["feature_vector"]):
        assert isinstance(val, int), f"feature_vector[{i}] must be int, got {type(val)}"
        assert val >= 0, f"feature_vector[{i}] must be >= 0"

    # The whole thing must survive json.dumps() — Phase 3 and the Flask API will serialize it
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"enrich_cv() return value is not JSON-serializable: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests — success path
# ──────────────────────────────────────────────────────────────────────────────

def test_enriched_output_has_seven_required_keys():
    """
    The most fundamental contract test: does enrich_cv() always return exactly
    the 7 keys Phase 3 depends on?
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result, f"Unexpected error: {result}"
    assert set(result.keys()) == REQUIRED_ENRICHED_KEYS


def test_feature_vector_is_exactly_40_elements():
    """
    The feature vector must always have exactly 40 elements (10 types × 4 sections).
    Phase 3 hardcodes this dimensionality in its model architecture — a different
    length would silently corrupt the classifier.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    assert len(result["feature_vector"]) == 40


def test_cluster_profile_has_exactly_seven_keys():
    """
    The cluster_profile must always have exactly 7 keys — the 6 domain clusters
    plus "unclassified". Phase 3 iterates over these keys to build its feature table.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    assert len(result["cluster_profile"]) == 7


def test_cluster_profile_has_all_required_cluster_keys():
    """
    All 7 specific cluster keys must be present, not just 7 arbitrary keys.
    Phase 3 accesses these by name: result["cluster_profile"]["robotics"], etc.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    assert set(result["cluster_profile"].keys()) == REQUIRED_CLUSTER_KEYS


def test_cluster_profile_values_are_non_negative_integers():
    """
    All cluster counts must be non-negative integers.
    A negative count or float would be a bug that could corrupt Phase 3's training data.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    for key, val in result["cluster_profile"].items():
        assert isinstance(val, int), f"cluster_profile['{key}'] must be int, got {type(val)}"
        assert val >= 0, f"cluster_profile['{key}'] must be >= 0, got {val}"


def test_entities_is_a_list():
    """
    The entities field must always be a list — even when the CV has no recognized
    entities (in which case it should be an empty list, not None or absent).
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    assert isinstance(result["entities"], list)


def test_entity_records_have_required_keys():
    """
    Every record in the entities list must have "term", "type", and "section" keys.
    Phase 3 accesses all three when building its input features.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    for entity in result["entities"]:
        assert "term" in entity, f"Entity missing 'term': {entity}"
        assert "type" in entity, f"Entity missing 'type': {entity}"
        assert "section" in entity, f"Entity missing 'section': {entity}"


def test_output_is_json_serializable():
    """
    The enriched output must survive json.dumps() without raising.
    This is the downstream contract — Phase 5 (Flask API) sends this as a JSON body.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    serialized = json.dumps(result)   # Must not raise
    assert isinstance(serialized, str)


def test_phase1_sections_preserved_unchanged():
    """
    The four Phase 1 section lists must appear in the enriched output exactly
    as they were in the input — enrich_cv() must not modify them.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    # The skills section from input must appear verbatim in output
    assert result["skills"] == MINIMAL_PHASE1_OUTPUT["skills"]
    assert result["projects"] == MINIMAL_PHASE1_OUTPUT["projects"]
    assert result["education"] == MINIMAL_PHASE1_OUTPUT["education"]
    assert result["internships"] == MINIMAL_PHASE1_OUTPUT["internships"]


def test_full_schema_with_complete_helper():
    """
    Full schema assertion using the reusable helper — a one-shot end-to-end
    schema sanity check covering all individual assertions at once.
    """
    result = enrich_cv(MINIMAL_PHASE1_OUTPUT)
    assert "error" not in result
    assert_valid_enriched_schema(result)


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests — empty input path (FR-012)
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_input_produces_valid_schema():
    """
    When all four Phase 1 section lists are empty (no text was extracted),
    enrich_cv() must still return a valid enriched output with the correct schema.
    It must NOT raise an error just because there's nothing to enrich.
    """
    result = enrich_cv(EMPTY_PHASE1_OUTPUT)
    assert "error" not in result, f"Empty input should not produce an error: {result}"
    assert_valid_enriched_schema(result)


def test_empty_input_produces_empty_entities():
    """
    An all-empty CV should produce an empty entities list — no entities to find.
    """
    result = enrich_cv(EMPTY_PHASE1_OUTPUT)
    assert "error" not in result
    assert result["entities"] == []


def test_empty_input_produces_all_zero_cluster_profile():
    """
    With no entities, all cluster counts must be zero.
    The schema is still complete — just with zero values everywhere.
    """
    result = enrich_cv(EMPTY_PHASE1_OUTPUT)
    assert "error" not in result
    for key, val in result["cluster_profile"].items():
        assert val == 0, f"Expected 0 for '{key}' with empty input, got {val}"


def test_empty_input_produces_zero_feature_vector():
    """
    With no entities, the feature vector must be a 40-element list of all zeros.
    Phase 3 treats this as a "blank CV" — it must not crash on it.
    """
    result = enrich_cv(EMPTY_PHASE1_OUTPUT)
    assert "error" not in result
    assert len(result["feature_vector"]) == 40
    assert all(v == 0 for v in result["feature_vector"]), (
        "All feature vector values should be 0 for empty input"
    )


def test_empty_input_json_roundtrip():
    """
    Empty input → enriched output → json.dumps → json.loads must preserve all values.
    This tests the JSON round-trip guarantee (SC-006) for the edge case.
    """
    result = enrich_cv(EMPTY_PHASE1_OUTPUT)
    assert "error" not in result
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized["entities"] == []
    assert len(deserialized["feature_vector"]) == 40
    assert len(deserialized["cluster_profile"]) == 7


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests — Phase 1 error pass-through
# ──────────────────────────────────────────────────────────────────────────────

def test_phase1_error_dict_passes_through_unchanged():
    """
    If Phase 1 returned an error dict (with "error" and "message" keys),
    enrich_cv() must return it unchanged — not process it, not wrap it,
    not modify it in any way.
    """
    phase1_error = {"error": "INVALID_PDF", "message": "The file is corrupted."}
    result = enrich_cv(phase1_error)
    # The output must be exactly the same object/content as the input
    assert result == phase1_error


def test_phase1_file_not_found_error_passes_through():
    """
    FILE_NOT_FOUND errors from Phase 1 must pass through just as cleanly
    as any other error type.
    """
    phase1_error = {
        "error": "FILE_NOT_FOUND",
        "message": "File not found: /path/to/cv.pdf",
    }
    result = enrich_cv(phase1_error)
    assert result["error"] == "FILE_NOT_FOUND"
    assert result["message"] == phase1_error["message"]


def test_phase1_no_text_content_error_passes_through():
    """
    NO_TEXT_CONTENT errors must also pass through unchanged.
    We verify that enrich_cv() does not accidentally add extra keys.
    """
    phase1_error = {
        "error": "NO_TEXT_CONTENT",
        "message": "Scanned PDF with no embedded text.",
    }
    result = enrich_cv(phase1_error)
    # Must be the exact same dict — no extra Phase 2 keys added
    assert set(result.keys()) == {"error", "message"}
    assert result["error"] == "NO_TEXT_CONTENT"


def test_enrich_cv_never_raises():
    """
    enrich_cv() must never raise an exception regardless of what is passed in.
    The contract guarantees this — callers should never need a try/except.
    """
    # Test with various pathological inputs — none should raise
    for bad_input in [None, 42, "not a dict", [], {}]:
        try:
            result = enrich_cv(bad_input)
            # Result should be a dict (either error or valid output)
            assert isinstance(result, dict), f"Expected dict for input {bad_input!r}"
        except Exception as exc:
            pytest.fail(
                f"enrich_cv({bad_input!r}) raised an exception when it should not: {exc}"
            )
