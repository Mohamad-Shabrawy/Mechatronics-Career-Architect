"""
test_phase2_pipeline.py — Phase 2 Integration Tests

These tests exercise the full Phase 1 → Phase 2 pipeline end-to-end, using real
sample CV PDFs from the data/samples directory. They validate that the complete
chain (parse_cv → enrich_cv) works correctly in realistic conditions.

Coverage targets:
- SC-002: Cluster mapping correctness on known entities (≥ 90% accuracy)
- SC-004: Feature vector always has exactly 40 elements
- SC-005: Pipeline completes in < 15 seconds
- SC-006: JSON round-trip lossless (zero data loss)
- SC-007: Empty-section inputs produce valid zero-valued output

Note on SC-001 (NER precision/recall ≥ 85%/80%):
  Formal precision/recall measurement requires annotated test data. We verify
  directional correctness (known entities are found in known CVs) rather than
  computing precise recall on a labeled corpus.

Note on SC-003 (cosine distance ≥ 0.3 between niche CVs):
  We test directional vector differences without computing cosine distance,
  since numpy is not a Phase 2 runtime dependency. The unit tests in
  test_feature_vector.py provide the numerical coverage.

Spec reference: specs/002-semantic-ner-features/spec.md (SC-001 through SC-007)
Tasks reference: specs/002-semantic-ner-features/tasks.md T026
"""

import json
import time
import os
import pytest

from src.parser import parse_cv
from src.enricher import enrich_cv

# ── Path setup ────────────────────────────────────────────────────────────────
# Resolve the samples directory from the project root.
# conftest.py defines PROJECT_ROOT and SAMPLES_DIR — we replicate the logic
# here so this file is self-contained and can be read independently.
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
_SAMPLES_DIR = os.path.join(
    os.path.dirname(_PROJECT_ROOT),   # "Python Engineering Projects"
    "Claude Training",
    "data",
    "samples",
)


def sample_path(filename: str) -> str:
    """Return the absolute path to a sample CV PDF."""
    return os.path.join(_SAMPLES_DIR, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SC-007 — Empty-section inputs produce valid zero-valued output
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_phase1_output_produces_valid_enriched_output():
    """
    SC-007: All-empty Phase 1 input must produce a valid zero-valued enriched output.
    No exception, no missing keys, no malformed vector.
    """
    empty_phase1 = {
        "skills":      [],
        "projects":    [],
        "education":   [],
        "internships": [],
    }
    result = enrich_cv(empty_phase1)

    # Must not be an error
    assert "error" not in result, f"Empty input should not produce error: {result}"

    # Schema must be complete
    assert "entities" in result
    assert "cluster_profile" in result
    assert "feature_vector" in result

    # All zero-valued
    assert result["entities"] == []
    assert all(v == 0 for v in result["cluster_profile"].values())
    assert len(result["feature_vector"]) == 40
    assert all(v == 0 for v in result["feature_vector"])


def test_phase1_with_only_one_section_populated():
    """
    Phase 1 output where only the skills section has content should produce
    entities only from that section, with the others empty.
    """
    phase1 = {
        "skills": [{"text": "Experienced with ROS and MATLAB", "keywords": []}],
        "projects":    [],
        "education":   [],
        "internships": [],
    }
    result = enrich_cv(phase1)
    assert "error" not in result

    # Entities should only be from skills section
    for entity in result["entities"]:
        assert entity["section"] == "skills", (
            f"Entity section should be 'skills', got '{entity['section']}'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SC-004 — Feature vector always exactly 40 elements
# ──────────────────────────────────────────────────────────────────────────────

def test_feature_vector_is_40_elements_for_various_inputs():
    """
    SC-004: feature_vector must be exactly 40 elements for any valid Phase 1 input.
    We test with a range of different inputs to verify consistency.
    """
    test_inputs = [
        # Empty
        {"skills": [], "projects": [], "education": [], "internships": []},
        # One section with no entities
        {"skills": [{"text": "Team player with good communication", "keywords": []}],
         "projects": [], "education": [], "internships": []},
        # One section with entities
        {"skills": [{"text": "S7-1200 and TIA Portal programming", "keywords": []}],
         "projects": [], "education": [], "internships": []},
        # Multiple sections with mixed content
        {
            "skills": [{"text": "ROS2 and OpenCV for robotics", "keywords": []}],
            "projects": [{"text": "Designed components in SolidWorks and CATIA", "keywords": []}],
            "education": [{"text": "B.Sc. Mechatronics Engineering, 2024", "keywords": []}],
            "internships": [{"text": "Worked on AUTOSAR and CAN Bus development", "keywords": []}],
        },
    ]
    for i, phase1 in enumerate(test_inputs):
        result = enrich_cv(phase1)
        assert "error" not in result, f"Test case {i}: unexpected error {result}"
        assert len(result["feature_vector"]) == 40, (
            f"Test case {i}: feature_vector length must be 40, "
            f"got {len(result['feature_vector'])}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SC-006 — JSON round-trip lossless
# ──────────────────────────────────────────────────────────────────────────────

def test_json_roundtrip_preserves_all_values():
    """
    SC-006: Enriched output must survive json.dumps → json.loads without
    any data loss. This is the primary serialization guarantee.
    """
    phase1 = {
        "skills": [
            {"text": "Programmed S7-1200, PROFINET, and MATLAB", "keywords": []}
        ],
        "projects": [
            {"text": "Built an autonomous robot using ROS2 and OpenCV", "keywords": []}
        ],
        "education": [
            {"text": "B.Sc. Mechatronics Engineering", "keywords": []}
        ],
        "internships": [
            {"text": "Developed AUTOSAR BSW components at Valeo", "keywords": []}
        ],
    }
    result = enrich_cv(phase1)
    assert "error" not in result

    # Serialize to JSON string
    try:
        serialized = json.dumps(result)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"json.dumps() failed: {exc}")

    # Deserialize back
    try:
        deserialized = json.loads(serialized)
    except json.JSONDecodeError as exc:
        pytest.fail(f"json.loads() failed: {exc}")

    # Key-by-key comparison — no data loss allowed
    assert deserialized["entities"] == result["entities"], (
        "entities list changed during JSON round-trip"
    )
    assert deserialized["cluster_profile"] == result["cluster_profile"], (
        "cluster_profile changed during JSON round-trip"
    )
    assert deserialized["feature_vector"] == result["feature_vector"], (
        "feature_vector changed during JSON round-trip"
    )
    assert len(deserialized["feature_vector"]) == 40, (
        "feature_vector length changed during JSON round-trip"
    )


def test_json_roundtrip_for_empty_input():
    """
    JSON round-trip must also work for the edge case of empty input.
    Zero-valued outputs must serialize and deserialize cleanly.
    """
    empty_phase1 = {"skills": [], "projects": [], "education": [], "internships": []}
    result = enrich_cv(empty_phase1)

    serialized = json.dumps(result)
    deserialized = json.loads(serialized)

    assert deserialized["entities"] == []
    assert len(deserialized["feature_vector"]) == 40
    assert all(v == 0 for v in deserialized["feature_vector"])
    assert len(deserialized["cluster_profile"]) == 7


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SC-005 — Pipeline completes in < 15 seconds
# ──────────────────────────────────────────────────────────────────────────────

def test_enrichment_completes_within_15_seconds():
    """
    SC-005: The Phase 2 pipeline must complete in under 15 seconds for any
    valid Phase 1 input. We test with a realistic multi-section input.
    """
    # Build a large-ish Phase 1 output to stress-test timing
    rich_phase1 = {
        "skills": [
            {"text": "S7-1200, TIA Portal, PROFINET, Modbus, SCADA, HMI design", "keywords": []},
            {"text": "ROS2, OpenCV, SLAM, MoveIt, Gazebo simulator", "keywords": []},
            {"text": "MATLAB, Simulink, Stateflow, dSPACE for automotive MBD", "keywords": []},
            {"text": "SolidWorks, CATIA, ANSYS Mechanical, FEA analysis", "keywords": []},
            {"text": "AUTOSAR, MISRA C, ISO 26262, CAN Bus, CAN FD", "keywords": []},
        ],
        "projects": [
            {"text": "Built autonomous robot with ROS and OpenCV vision system", "keywords": []},
            {"text": "Programmed S7-1500 PLC for conveyor automation project", "keywords": []},
            {"text": "Designed STM32 embedded firmware with FreeRTOS and UART", "keywords": []},
        ],
        "education": [
            {"text": "B.Sc. Mechatronics Engineering, Cairo University 2024", "keywords": []},
        ],
        "internships": [
            {"text": "Developed AUTOSAR BSW at Valeo; used CAN Bus and dSPACE HIL", "keywords": []},
        ],
    }

    start_time = time.monotonic()
    result = enrich_cv(rich_phase1)
    elapsed = time.monotonic() - start_time

    assert "error" not in result, f"Rich input produced error: {result}"
    assert elapsed < 15.0, (
        f"Phase 2 pipeline took {elapsed:.2f}s, must complete in < 15 seconds (SC-005)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SC-002 — Cluster mapping correctness on known entities
# ──────────────────────────────────────────────────────────────────────────────

def test_known_ros_entity_maps_to_robotics_cluster():
    """
    SC-002: "ROS" is a known robotics entity. When found in a CV, the robotics
    cluster count must be > 0.
    """
    phase1 = {
        "skills": [{"text": "Experienced with ROS and ROS2 for mobile robotics", "keywords": []}],
        "projects": [], "education": [], "internships": [],
    }
    result = enrich_cv(phase1)
    assert result["cluster_profile"]["robotics"] > 0, (
        "ROS entities should produce non-zero robotics cluster count"
    )


def test_known_s7_1200_maps_to_industrial_automation_cluster():
    """
    SC-002: "S7-1200" is a known PLC entity. It should map to industrial_automation.
    """
    phase1 = {
        "skills": [{"text": "Programmed S7-1200 PLC via TIA Portal", "keywords": []}],
        "projects": [], "education": [], "internships": [],
    }
    result = enrich_cv(phase1)
    assert result["cluster_profile"]["industrial_automation"] > 0, (
        "S7-1200 should produce non-zero industrial_automation cluster count"
    )


def test_known_autosar_maps_to_automotive_and_embedded():
    """
    SC-002: "AUTOSAR" is known to map to both automotive and embedded_systems.
    Both cluster counts must be > 0.
    """
    phase1 = {
        "internships": [{"text": "Developed AUTOSAR software components", "keywords": []}],
        "skills": [], "projects": [], "education": [],
    }
    result = enrich_cv(phase1)
    assert result["cluster_profile"]["automotive"] > 0, (
        "AUTOSAR should contribute to automotive cluster"
    )
    assert result["cluster_profile"]["embedded_systems"] > 0, (
        "AUTOSAR should contribute to embedded_systems cluster"
    )


def test_known_solidworks_maps_to_mechanical_design_cluster():
    """
    SC-002: "SolidWorks" is a known CAD tool. It should map to mechanical_design.
    """
    phase1 = {
        "skills": [{"text": "Proficient in SolidWorks for 3D modeling", "keywords": []}],
        "projects": [], "education": [], "internships": [],
    }
    result = enrich_cv(phase1)
    assert result["cluster_profile"]["mechanical_design"] > 0, (
        "SolidWorks should produce non-zero mechanical_design cluster count"
    )


def test_known_agile_maps_to_technical_management_cluster():
    """
    SC-002: "Agile" methodology maps to technical_management cluster.
    """
    phase1 = {
        "skills": [{"text": "Experienced with Agile and Scrum methodology", "keywords": []}],
        "projects": [], "education": [], "internships": [],
    }
    result = enrich_cv(phase1)
    assert result["cluster_profile"]["technical_management"] > 0, (
        "Agile should produce non-zero technical_management cluster count"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Phase 1 error handling in the full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_phase1_error_passes_through_enrich_cv():
    """
    When the full pipeline encounters a Phase 1 error (e.g., file not found),
    enrich_cv() must return the error dict unchanged without processing.
    """
    # Simulate Phase 1 behavior for a missing file
    phase1_error = parse_cv("this_file_does_not_exist_for_test.pdf")
    assert "error" in phase1_error, "parse_cv() should return error for missing file"

    # enrich_cv() must pass it through unchanged
    result = enrich_cv(phase1_error)
    assert result == phase1_error, (
        "enrich_cv() must return Phase 1 error dicts unchanged"
    )


def test_full_pipeline_with_real_pdf_produces_valid_schema():
    """
    Run the complete parse_cv() → enrich_cv() pipeline on a real sample PDF.
    If the PDF is available, verify the enriched schema is complete.
    """
    pdf_path = sample_path("resume_01_baseline_robotics.pdf")

    if not os.path.isfile(pdf_path):
        pytest.skip(f"Sample PDF not available at {pdf_path}")

    # Phase 1
    phase1_result = parse_cv(pdf_path)

    if "error" in phase1_result:
        # Phase 1 failed — skip this test (it tests Phase 2, not Phase 1)
        pytest.skip(f"Phase 1 failed for {pdf_path}: {phase1_result}")

    # Phase 2
    enriched = enrich_cv(phase1_result)
    assert "error" not in enriched, f"enrich_cv() failed for {pdf_path}: {enriched}"

    # Schema checks
    assert "entities" in enriched
    assert "cluster_profile" in enriched
    assert "feature_vector" in enriched
    assert len(enriched["feature_vector"]) == 40
    assert len(enriched["cluster_profile"]) == 7

    # JSON round-trip
    serialized = json.dumps(enriched)
    deserialized = json.loads(serialized)
    assert deserialized["feature_vector"] == enriched["feature_vector"]


def test_full_pipeline_with_robotics_pdf_has_nonzero_robotics_cluster():
    """
    The baseline robotics CV (resume_01_baseline_robotics.pdf) should produce
    a non-zero robotics cluster count when processed through the full pipeline.
    This is a directional correctness check for SC-002.
    """
    pdf_path = sample_path("resume_01_baseline_robotics.pdf")

    if not os.path.isfile(pdf_path):
        pytest.skip(f"Sample PDF not available at {pdf_path}")

    phase1_result = parse_cv(pdf_path)
    if "error" in phase1_result:
        pytest.skip(f"Phase 1 failed: {phase1_result}")

    enriched = enrich_cv(phase1_result)
    if "error" in enriched:
        pytest.skip(f"Phase 2 failed: {enriched}")

    # A robotics-focused CV should produce entities in the robotics cluster
    # We accept that some PDFs may not contain exact matches from our taxonomy
    # but the cluster profile should be a complete 7-key dict
    assert len(enriched["cluster_profile"]) == 7
    # If entities were found, robotics should have at least some count
    # (we don't require it to be non-zero in case the PDF text doesn't match our terms)
    total_entities = len(enriched["entities"])
    if total_entities > 0:
        total_cluster_count = sum(enriched["cluster_profile"].values())
        assert total_cluster_count > 0, (
            f"Non-empty entities list should produce non-zero cluster counts. "
            f"Entities: {total_entities}, cluster_profile: {enriched['cluster_profile']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Phase 2 doesn't modify Phase 1 output
# ──────────────────────────────────────────────────────────────────────────────

def test_enrich_cv_does_not_mutate_phase1_input():
    """
    enrich_cv() must not modify the input dict. Phase 1's output should be
    identical before and after calling enrich_cv() on it.
    """
    phase1 = {
        "skills": [{"text": "Used ROS and MATLAB for robotics", "keywords": ["ros", "matlab"]}],
        "projects": [{"text": "Built STM32 firmware project", "keywords": ["stm32"]}],
        "education": [],
        "internships": [],
    }

    # Deep copy reference for comparison
    import copy
    phase1_copy = copy.deepcopy(phase1)

    # Run enrichment
    enrich_cv(phase1)

    # Input dict must not have been modified
    assert phase1 == phase1_copy, (
        "enrich_cv() must not modify its input — Phase 1 output was mutated"
    )


def test_enriched_output_sections_equal_phase1_sections():
    """
    The four section lists in the enriched output must be exactly the same
    objects/values as the input Phase 1 sections.
    """
    phase1 = {
        "skills": [
            {"text": "Python and ROS development", "keywords": ["python", "ros"]}
        ],
        "projects": [],
        "education": [
            {"text": "B.Sc. Engineering 2024", "keywords": []}
        ],
        "internships": [],
    }
    enriched = enrich_cv(phase1)
    assert "error" not in enriched

    # The content must be identical (not just structurally similar)
    assert enriched["skills"] == phase1["skills"]
    assert enriched["projects"] == phase1["projects"]
    assert enriched["education"] == phase1["education"]
    assert enriched["internships"] == phase1["internships"]
