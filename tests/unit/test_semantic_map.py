"""
test_semantic_map.py — Unit Tests for Semantic Cluster Mapping (User Story 2)

These tests verify that the cluster mapping logic correctly assigns entities to
Mechatronics domain clusters, handles multi-cluster assignments, and gracefully
handles entities not in the SEMANTIC_MAP.

What we're verifying:
- Known entities map to the correct cluster(s)
- Multi-cluster entities (e.g., "autosar") increment multiple counters correctly
- Entities not in SEMANTIC_MAP are assigned to "unclassified" without error
- cluster_profile always has all 7 keys, even when some counts are 0
- The cluster_profile counts are correctly derived from the entity list

Spec reference: specs/002-semantic-ner-features/spec.md User Story 2
Tasks reference: specs/002-semantic-ner-features/tasks.md T014
"""

import pytest
from src.enricher import enrich_cv
from src.semantic_map import SEMANTIC_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_phase1_with_text(section: str, text: str) -> dict:
    """Build a minimal Phase 1 output with one text block in one section."""
    return {
        "skills":      [{"text": text, "keywords": []}] if section == "skills" else [],
        "projects":    [{"text": text, "keywords": []}] if section == "projects" else [],
        "education":   [{"text": text, "keywords": []}] if section == "education" else [],
        "internships": [{"text": text, "keywords": []}] if section == "internships" else [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SEMANTIC_MAP structure
# ──────────────────────────────────────────────────────────────────────────────

def test_semantic_map_is_a_dict():
    """SEMANTIC_MAP must be a dict — it's imported directly by the enricher."""
    assert isinstance(SEMANTIC_MAP, dict), "SEMANTIC_MAP must be a dict"


def test_semantic_map_keys_are_lowercase():
    """
    All keys in SEMANTIC_MAP must be lowercase strings.
    The NER scanner always lowercases entity terms before lookup,
    so uppercase keys would never be found.
    """
    for key in SEMANTIC_MAP:
        assert isinstance(key, str), f"SEMANTIC_MAP key '{key}' must be a str"
        assert key == key.lower(), (
            f"SEMANTIC_MAP key '{key}' must be lowercase, got mixed case"
        )


def test_semantic_map_values_are_lists_of_strings():
    """
    Every value in SEMANTIC_MAP must be a non-empty list of cluster name strings.
    An empty list [] would cause an entity to contribute nothing to the profile.
    """
    valid_clusters = {
        "industrial_automation", "robotics", "embedded_systems",
        "automotive", "mechanical_design", "technical_management", "unclassified",
    }
    for entity, clusters in SEMANTIC_MAP.items():
        assert isinstance(clusters, list), (
            f"SEMANTIC_MAP['{entity}'] must be a list, got {type(clusters)}"
        )
        assert len(clusters) >= 1, (
            f"SEMANTIC_MAP['{entity}'] must not be empty"
        )
        for cluster in clusters:
            assert isinstance(cluster, str), (
                f"SEMANTIC_MAP['{entity}'] contains non-string cluster: {cluster!r}"
            )
            assert cluster in valid_clusters, (
                f"SEMANTIC_MAP['{entity}'] contains unknown cluster '{cluster}'. "
                f"Valid clusters: {valid_clusters}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Specific entity cluster assignments
# ──────────────────────────────────────────────────────────────────────────────

def test_ros_maps_to_robotics():
    """
    "ROS" (Robot Operating System) must map to the "robotics" cluster.
    This is one of the clearest 1:1 mappings in the taxonomy.

    Spec acceptance scenario 1 analogue: FreeRTOS → embedded_systems;
    here we test the equivalent for ROS → robotics.
    """
    assert "ros" in SEMANTIC_MAP, "'ros' must be in SEMANTIC_MAP"
    assert "robotics" in SEMANTIC_MAP["ros"], (
        f"'ros' should map to 'robotics', got {SEMANTIC_MAP['ros']}"
    )


def test_s7_1200_maps_to_industrial_automation():
    """
    The Siemens S7-1200 PLC maps to industrial_automation.
    This tests one of the foundational cluster assignments.
    """
    assert "s7-1200" in SEMANTIC_MAP, "'s7-1200' must be in SEMANTIC_MAP"
    assert "industrial_automation" in SEMANTIC_MAP["s7-1200"], (
        f"'s7-1200' should map to 'industrial_automation', got {SEMANTIC_MAP['s7-1200']}"
    )


def test_autosar_maps_to_embedded_systems_and_automotive():
    """
    AUTOSAR is both an automotive standard (Automotive cluster) AND a firmware
    architecture framework (Embedded Systems cluster). It should map to both.

    This is the multi-cluster entity example from the spec:
    "AUTOSAR → both Embedded Systems and Automotive clusters"
    """
    assert "autosar" in SEMANTIC_MAP, "'autosar' must be in SEMANTIC_MAP"
    clusters = SEMANTIC_MAP["autosar"]
    assert "embedded_systems" in clusters, (
        f"'autosar' should include 'embedded_systems', got {clusters}"
    )
    assert "automotive" in clusters, (
        f"'autosar' should include 'automotive', got {clusters}"
    )


def test_pmp_maps_to_technical_management():
    """
    PMP (Project Management Professional) is a management certification.
    It must map to technical_management.
    """
    assert "pmp" in SEMANTIC_MAP, "'pmp' must be in SEMANTIC_MAP"
    assert "technical_management" in SEMANTIC_MAP["pmp"], (
        f"'pmp' should map to 'technical_management', got {SEMANTIC_MAP['pmp']}"
    )


def test_solidworks_maps_to_mechanical_design():
    """SolidWorks is a CAD tool — it belongs to mechanical_design."""
    assert "solidworks" in SEMANTIC_MAP, "'solidworks' must be in SEMANTIC_MAP"
    assert "mechanical_design" in SEMANTIC_MAP["solidworks"], (
        f"'solidworks' should map to 'mechanical_design', got {SEMANTIC_MAP['solidworks']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: cluster_profile schema — always 7 keys, always present
# ──────────────────────────────────────────────────────────────────────────────

def test_cluster_profile_always_has_all_seven_keys():
    """
    No matter what entities are found, cluster_profile must have exactly 7 keys.
    Phase 3 accesses each cluster by name — a missing key would raise KeyError.
    """
    # Use a text with known entities from a specific cluster
    phase1 = make_phase1_with_text("skills", "Programmed S7-1200 via TIA Portal")
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    required_keys = {
        "industrial_automation", "robotics", "embedded_systems",
        "automotive", "mechanical_design", "technical_management", "unclassified",
    }
    assert set(profile.keys()) == required_keys, (
        f"cluster_profile missing keys: {required_keys - set(profile.keys())}"
    )


def test_cluster_profile_has_seven_keys_even_with_zero_entities():
    """
    When enrich_cv() finds no entities at all (empty CV), cluster_profile must
    still have all 7 keys — all set to zero. Not an empty dict, not fewer keys.
    """
    phase1 = {"skills": [], "projects": [], "education": [], "internships": []}
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    assert len(profile) == 7, (
        f"cluster_profile must have 7 keys even for empty input, got {len(profile)}"
    )
    # All values should be zero since no entities were found
    for key, val in profile.items():
        assert val == 0, f"Expected 0 for '{key}' with empty input, got {val}"


def test_cluster_profile_values_are_non_negative_integers():
    """
    All cluster counts must be non-negative integers.
    A float or negative value would indicate a calculation error.
    """
    phase1 = make_phase1_with_text("skills", "Used ROS and AUTOSAR and SolidWorks")
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    for key, val in profile.items():
        assert isinstance(val, int), (
            f"cluster_profile['{key}'] must be int, got {type(val)}"
        )
        assert val >= 0, (
            f"cluster_profile['{key}'] must be >= 0, got {val}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Multi-cluster entities increment multiple counters
# ──────────────────────────────────────────────────────────────────────────────

def test_autosar_entity_increments_both_embedded_and_automotive_counters():
    """
    "AUTOSAR" maps to both embedded_systems and automotive.
    Both counters must be incremented when AUTOSAR is found in the CV.
    """
    phase1 = make_phase1_with_text("skills", "Developed AUTOSAR components for ECU software")
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    # AUTOSAR is in ENTITY_TAXONOMY as automotive_standard, and maps to both clusters
    # We verify that both target clusters were incremented
    assert profile["embedded_systems"] >= 1, (
        "AUTOSAR should increment embedded_systems counter"
    )
    assert profile["automotive"] >= 1, (
        "AUTOSAR should increment automotive counter"
    )


def test_can_bus_increments_automotive_and_embedded():
    """
    "CAN Bus" maps to automotive AND embedded_systems.
    Both cluster counters should be incremented.
    """
    phase1 = make_phase1_with_text(
        "internships",
        "Implemented CAN bus communication between ECUs"
    )
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    # CAN Bus is in communication_protocol and maps to automotive + embedded_systems
    assert profile["automotive"] >= 1, "CAN bus should increment automotive counter"
    assert profile["embedded_systems"] >= 1, "CAN bus should increment embedded_systems counter"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Unclassified fallback for entities not in SEMANTIC_MAP
# ──────────────────────────────────────────────────────────────────────────────

def test_unclassified_cluster_exists_in_profile():
    """
    The "unclassified" key must always be in cluster_profile, even if its value is 0.
    This ensures callers can always safely access profile["unclassified"].
    """
    phase1 = make_phase1_with_text("skills", "Used ROS and MATLAB")
    result = enrich_cv(phase1)
    assert "unclassified" in result["cluster_profile"], (
        "cluster_profile must always contain 'unclassified' key"
    )


def test_cluster_profile_reflects_entity_distribution():
    """
    A CV with clear robotics content should have a higher robotics cluster count
    than industrial_automation. This is a smoke test that the mapping is working
    directionally correctly.
    """
    # Pure robotics CV: ROS, ROS2, MoveIt, Gazebo, OpenCV
    phase1 = make_phase1_with_text(
        "projects",
        "Built a mobile robot navigation system using ROS, ROS2, MoveIt, and Gazebo simulator"
    )
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    # Robotics cluster should have the highest count
    assert profile["robotics"] > profile["industrial_automation"], (
        f"Robotics-heavy CV should have more robotics than industrial_automation. "
        f"Profile: {profile}"
    )
    assert profile["robotics"] > 0, "Robotics cluster count should be > 0 for this CV"


def test_industrial_automation_cluster_for_plc_heavy_cv():
    """
    A CV dominated by PLC content should have the highest count in industrial_automation.
    """
    phase1 = make_phase1_with_text(
        "skills",
        "Programmed S7-1200, TIA Portal, PROFINET, and PROFIBUS for factory automation"
    )
    result = enrich_cv(phase1)
    profile = result["cluster_profile"]

    # Industrial automation should dominate
    assert profile["industrial_automation"] > 0, (
        "industrial_automation count should be > 0 for PLC-heavy CV"
    )
    assert profile["industrial_automation"] > profile["robotics"], (
        f"PLC-heavy CV should have more industrial_automation than robotics. "
        f"Profile: {profile}"
    )


def test_cluster_counts_match_entity_assignments():
    """
    The cluster_profile counts must be consistent with the entities list.
    For each entity, we can look up its clusters and verify the totals add up.
    This tests the internal consistency of the enricher output.
    """
    phase1 = {
        "skills": [
            {"text": "Experienced with ROS and AUTOSAR development", "keywords": []}
        ],
        "projects": [],
        "education": [],
        "internships": [],
    }
    result = enrich_cv(phase1)
    entities = result["entities"]
    profile = result["cluster_profile"]

    # Manually recompute the cluster counts from the entity list
    expected_profile = {
        "industrial_automation": 0,
        "robotics":              0,
        "embedded_systems":      0,
        "automotive":            0,
        "mechanical_design":     0,
        "technical_management":  0,
        "unclassified":          0,
    }
    for entity in entities:
        clusters = SEMANTIC_MAP.get(entity["term"], ["unclassified"])
        for cluster in clusters:
            if cluster in expected_profile:
                expected_profile[cluster] += 1

    assert profile == expected_profile, (
        f"cluster_profile doesn't match recomputed counts.\n"
        f"Got:      {profile}\n"
        f"Expected: {expected_profile}"
    )
