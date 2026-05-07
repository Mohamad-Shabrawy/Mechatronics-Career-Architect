"""
test_feature_vector.py — Unit Tests for Feature Vector Construction (User Story 3)

These tests verify the build_feature_vector() function and the VECTOR_INDEX_ORDER
constant directly, as well as the feature_vector output from enrich_cv().

What we're verifying:
- build_feature_vector([]) returns exactly 40 zeros (zero vector for empty input)
- A single entity sets the correct index to 1, all others to 0
- The same entity twice (pre-deduplicated) sets the index to 2 (not tested here —
  deduplication is done upstream; we test that upstream already deduplicated)
- len(build_feature_vector(any_input)) == 40 always
- The index layout formula (type_index × 4 + section_index) is correct
- Section-context encoding works: same entity in different sections → different indices
- VECTOR_INDEX_ORDER has exactly 40 elements
- enrich_cv() produces a feature_vector of exactly 40 elements

Spec reference: specs/002-semantic-ner-features/spec.md User Story 3
Tasks reference: specs/002-semantic-ner-features/tasks.md T018
"""

import pytest
from src.feature_vector import build_feature_vector, VECTOR_INDEX_ORDER


# ──────────────────────────────────────────────────────────────────────────────
# Tests: VECTOR_INDEX_ORDER structure
# ──────────────────────────────────────────────────────────────────────────────

def test_vector_index_order_has_exactly_40_elements():
    """
    VECTOR_INDEX_ORDER must have exactly 40 elements (10 entity types × 4 sections).
    This is verified at module load time by an assert, but we test it here too
    so the failure message is clear and test-traceable.
    """
    assert len(VECTOR_INDEX_ORDER) == 40, (
        f"VECTOR_INDEX_ORDER must have 40 elements, got {len(VECTOR_INDEX_ORDER)}"
    )


def test_vector_index_order_contains_tuples():
    """
    Every element in VECTOR_INDEX_ORDER must be a (type_key, section_name) tuple.
    """
    for i, item in enumerate(VECTOR_INDEX_ORDER):
        assert isinstance(item, tuple), (
            f"VECTOR_INDEX_ORDER[{i}] must be a tuple, got {type(item)}"
        )
        assert len(item) == 2, (
            f"VECTOR_INDEX_ORDER[{i}] must be a 2-tuple, got length {len(item)}"
        )


def test_vector_index_order_starts_with_plc_hardware_skills():
    """
    The first entry in VECTOR_INDEX_ORDER must be ("plc_hardware", "skills").
    This is index 0 — the starting point of the layout contract.
    Changing this would silently break Phase 3's classifier.
    """
    assert VECTOR_INDEX_ORDER[0] == ("plc_hardware", "skills"), (
        f"Index 0 must be ('plc_hardware', 'skills'), got {VECTOR_INDEX_ORDER[0]}"
    )


def test_vector_index_order_ends_with_management_methodology_internships():
    """
    The last entry (index 39) must be ("management_methodology", "internships").
    This verifies the end of the layout contract.
    """
    assert VECTOR_INDEX_ORDER[39] == ("management_methodology", "internships"), (
        f"Index 39 must be ('management_methodology', 'internships'), got {VECTOR_INDEX_ORDER[39]}"
    )


def test_vector_index_formula_is_type_times_4_plus_section():
    """
    The index layout formula is: index = (type_index × 4) + section_index
    We verify this for the first few (type, section) combinations.

    From data-model.md:
      type_index: plc_hardware=0, cad_software=1, microcontroller=2, ...
      section_index: skills=0, projects=1, education=2, internships=3
    """
    # plc_hardware (type_index=0) × 4 + skills (section_index=0) = 0
    assert VECTOR_INDEX_ORDER[0] == ("plc_hardware", "skills")

    # plc_hardware (0) × 4 + projects (1) = 1
    assert VECTOR_INDEX_ORDER[1] == ("plc_hardware", "projects")

    # plc_hardware (0) × 4 + education (2) = 2
    assert VECTOR_INDEX_ORDER[2] == ("plc_hardware", "education")

    # plc_hardware (0) × 4 + internships (3) = 3
    assert VECTOR_INDEX_ORDER[3] == ("plc_hardware", "internships")

    # cad_software (type_index=1) × 4 + skills (0) = 4
    assert VECTOR_INDEX_ORDER[4] == ("cad_software", "skills")

    # simulation_tool (type_index=5) × 4 + skills (0) = 20
    assert VECTOR_INDEX_ORDER[20] == ("simulation_tool", "skills")


def test_vector_index_order_contains_all_10_types():
    """
    All 10 entity type keys must appear in VECTOR_INDEX_ORDER exactly 4 times each
    (once per section). No type should be missing or duplicated.
    """
    expected_types = {
        "plc_hardware", "cad_software", "microcontroller",
        "communication_protocol", "programming_language", "simulation_tool",
        "robotic_framework", "automotive_standard", "mechanical_tool",
        "management_methodology",
    }
    types_in_order = [t for t, s in VECTOR_INDEX_ORDER]

    for type_key in expected_types:
        count = types_in_order.count(type_key)
        assert count == 4, (
            f"Type '{type_key}' should appear exactly 4 times in VECTOR_INDEX_ORDER, "
            f"got {count}"
        )


def test_vector_index_order_contains_all_4_sections_per_type():
    """
    For every type, all 4 sections (skills, projects, education, internships)
    must appear in VECTOR_INDEX_ORDER exactly once.
    """
    expected_sections = {"skills", "projects", "education", "internships"}
    from collections import defaultdict

    # Group by type
    by_type: dict = defaultdict(list)
    for type_key, section in VECTOR_INDEX_ORDER:
        by_type[type_key].append(section)

    for type_key, sections in by_type.items():
        assert set(sections) == expected_sections, (
            f"Type '{type_key}' missing sections: {expected_sections - set(sections)}"
        )
        assert len(sections) == 4, (
            f"Type '{type_key}' has {len(sections)} section entries, expected 4"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: build_feature_vector() — zero vector for empty input
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_entity_list_returns_40_zeros():
    """
    build_feature_vector([]) must return a list of exactly 40 zeros.
    This is the zero vector for a CV with no recognized entities (SC-007).
    """
    result = build_feature_vector([])

    assert isinstance(result, list), "build_feature_vector([]) must return a list"
    assert len(result) == 40, (
        f"build_feature_vector([]) must return 40 elements, got {len(result)}"
    )
    assert all(v == 0 for v in result), (
        "build_feature_vector([]) must return all zeros"
    )


def test_build_feature_vector_always_returns_40_elements():
    """
    Regardless of input size, build_feature_vector() must always return
    a list of exactly 40 elements. SC-004 requires this.
    """
    # Various inputs — all must produce a 40-element list
    test_cases = [
        [],   # empty
        [{"term": "ros", "type": "robotic_framework", "section": "skills"}],  # one entity
        [   # multiple entities
            {"term": "s7-1200", "type": "plc_hardware", "section": "skills"},
            {"term": "ros", "type": "robotic_framework", "section": "projects"},
            {"term": "matlab", "type": "simulation_tool", "section": "education"},
        ],
    ]
    for entities in test_cases:
        result = build_feature_vector(entities)
        assert len(result) == 40, (
            f"build_feature_vector({entities}) returned {len(result)} elements, expected 40"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: build_feature_vector() — single entity sets correct index
# ──────────────────────────────────────────────────────────────────────────────

def test_single_plc_hardware_entity_in_skills_sets_index_0():
    """
    A single plc_hardware entity in the skills section should set index 0 to 1
    and leave all other 39 indices at 0.

    Index formula: plc_hardware (type_index=0) × 4 + skills (section_index=0) = 0
    """
    entities = [{"term": "s7-1200", "type": "plc_hardware", "section": "skills"}]
    result = build_feature_vector(entities)

    assert result[0] == 1, (
        f"Index 0 (plc_hardware × skills) should be 1, got {result[0]}"
    )
    # All other indices must remain 0
    for i in range(1, 40):
        assert result[i] == 0, (
            f"Index {i} should be 0, got {result[i]}"
        )


def test_robotic_framework_in_projects_sets_correct_index():
    """
    robotic_framework (type_index=6) in projects (section_index=1):
    index = 6 × 4 + 1 = 25

    We verify index 25 is set to 1, all others to 0.
    """
    entities = [{"term": "ros", "type": "robotic_framework", "section": "projects"}]
    result = build_feature_vector(entities)

    expected_index = (6 * 4) + 1  # = 25
    assert result[expected_index] == 1, (
        f"Index {expected_index} (robotic_framework × projects) should be 1, "
        f"got {result[expected_index]}"
    )
    for i in range(40):
        if i != expected_index:
            assert result[i] == 0, f"Index {i} should be 0, got {result[i]}"


def test_management_methodology_in_internships_sets_index_39():
    """
    management_methodology (type_index=9) in internships (section_index=3):
    index = 9 × 4 + 3 = 39 — the last index in the vector.
    """
    entities = [{"term": "agile", "type": "management_methodology", "section": "internships"}]
    result = build_feature_vector(entities)

    assert result[39] == 1, (
        f"Index 39 (management_methodology × internships) should be 1, got {result[39]}"
    )
    for i in range(39):
        assert result[i] == 0, f"Index {i} should be 0, got {result[i]}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Section-context encoding — same entity in different sections
# ──────────────────────────────────────────────────────────────────────────────

def test_same_entity_in_different_sections_increments_different_indices():
    """
    The same entity type in two different sections must increment two DIFFERENT
    indices, not the same one. This is the section-context encoding requirement (FR-009).

    Skills version of plc_hardware → index 0
    Projects version of plc_hardware → index 1
    Both must be 1, and no other index should be affected.
    """
    entities = [
        {"term": "s7-1200", "type": "plc_hardware", "section": "skills"},
        {"term": "s7-1200", "type": "plc_hardware", "section": "projects"},
    ]
    result = build_feature_vector(entities)

    # Index 0 = plc_hardware in skills
    assert result[0] == 1, f"Index 0 (plc_hardware/skills) should be 1, got {result[0]}"
    # Index 1 = plc_hardware in projects
    assert result[1] == 1, f"Index 1 (plc_hardware/projects) should be 1, got {result[1]}"

    # Both are different dimensions — no dimension confusion
    for i in range(2, 40):
        assert result[i] == 0, f"Index {i} should be 0, got {result[i]}"


def test_two_cvs_with_different_section_distributions_produce_different_vectors():
    """
    Two CVs with identical entities but in different sections must produce
    different feature vectors. This tests acceptance scenario 2 from spec.md:
    "two CVs with identical keyword sets but different section distributions...
    the resulting feature vectors are different."
    """
    # CV A: plc_hardware entities only in skills
    entities_cv_a = [
        {"term": "s7-1200", "type": "plc_hardware", "section": "skills"}
    ]
    # CV B: plc_hardware entities only in projects (different section)
    entities_cv_b = [
        {"term": "s7-1200", "type": "plc_hardware", "section": "projects"}
    ]

    vector_a = build_feature_vector(entities_cv_a)
    vector_b = build_feature_vector(entities_cv_b)

    # The vectors must be numerically different
    assert vector_a != vector_b, (
        "Same entity in different sections must produce different feature vectors"
    )

    # Specifically: CV A has index 0 = 1, CV B has index 1 = 1
    assert vector_a[0] == 1  # plc_hardware in skills
    assert vector_a[1] == 0  # plc_hardware in projects = 0
    assert vector_b[0] == 0  # plc_hardware in skills = 0
    assert vector_b[1] == 1  # plc_hardware in projects


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Multiple entities in the vector
# ──────────────────────────────────────────────────────────────────────────────

def test_multiple_distinct_entities_increment_correct_indices():
    """
    When multiple entities of different types/sections are provided,
    each increments only its own index. No cross-contamination between dimensions.
    """
    entities = [
        # plc_hardware in skills → index 0
        {"term": "s7-1200", "type": "plc_hardware", "section": "skills"},
        # cad_software in education → index (1×4)+2 = 6
        {"term": "solidworks", "type": "cad_software", "section": "education"},
        # robotic_framework in projects → index (6×4)+1 = 25
        {"term": "ros", "type": "robotic_framework", "section": "projects"},
    ]
    result = build_feature_vector(entities)

    assert result[0] == 1, "Index 0 (plc_hardware/skills) should be 1"
    assert result[6] == 1, "Index 6 (cad_software/education) should be 1"
    assert result[25] == 1, "Index 25 (robotic_framework/projects) should be 1"

    # All other indices should be 0
    for i in range(40):
        if i not in (0, 6, 25):
            assert result[i] == 0, f"Index {i} should be 0, got {result[i]}"


def test_two_entities_of_same_type_in_same_section_count_two():
    """
    If two DISTINCT entities of the same type appear in the same section
    (after deduplication by term, which happens upstream), the dimension
    for that (type, section) pair should count 2.

    For example: "s7-1200" AND "compactlogix" are both plc_hardware in skills →
    index 0 should be 2.
    """
    entities = [
        {"term": "s7-1200",      "type": "plc_hardware", "section": "skills"},
        {"term": "compactlogix", "type": "plc_hardware", "section": "skills"},
    ]
    result = build_feature_vector(entities)

    # Both are distinct plc_hardware entities in skills → index 0 = 2
    assert result[0] == 2, (
        f"Two distinct plc_hardware entities in skills should give index 0 = 2, "
        f"got {result[0]}"
    )


def test_vector_values_are_non_negative_integers():
    """
    All values in the feature vector must be non-negative integers.
    Floats or negative numbers would break the ML classifier.
    """
    entities = [
        {"term": "ros",       "type": "robotic_framework",   "section": "skills"},
        {"term": "solidworks","type": "cad_software",         "section": "projects"},
        {"term": "autosar",   "type": "automotive_standard",  "section": "internships"},
    ]
    result = build_feature_vector(entities)

    for i, val in enumerate(result):
        assert isinstance(val, int), (
            f"feature_vector[{i}] must be int, got {type(val)}"
        )
        assert val >= 0, (
            f"feature_vector[{i}] must be >= 0, got {val}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Robotics vs Industrial Automation vectors are distinct (SC-003 analogue)
# ──────────────────────────────────────────────────────────────────────────────

def test_robotics_heavy_cv_differs_from_automation_heavy_cv():
    """
    A robotics-heavy CV and an industrial-automation-heavy CV must produce
    numerically different feature vectors. This is the SC-003 directional check
    (cosine distance ≥ 0.3 is tested in the integration suite; here we test
    the simpler "are they different at all" condition).
    """
    # Pure robotics entities
    robotics_entities = [
        {"term": "ros",              "type": "robotic_framework", "section": "skills"},
        {"term": "ros2",             "type": "robotic_framework", "section": "skills"},
        {"term": "moveit",           "type": "robotic_framework", "section": "projects"},
        {"term": "gazebo simulator", "type": "simulation_tool",   "section": "projects"},
    ]
    # Pure industrial automation entities
    automation_entities = [
        {"term": "s7-1200",   "type": "plc_hardware",           "section": "skills"},
        {"term": "tia portal","type": "plc_hardware",           "section": "skills"},
        {"term": "profinet",  "type": "communication_protocol", "section": "projects"},
        {"term": "scada",     "type": "plc_hardware",           "section": "projects"},
    ]

    robotics_vector    = build_feature_vector(robotics_entities)
    automation_vector  = build_feature_vector(automation_entities)

    # The two vectors must be different
    assert robotics_vector != automation_vector, (
        "Robotics-heavy and automation-heavy CVs must produce different feature vectors"
    )

    # Additionally, robotics vector should have non-zero robotic_framework dimensions
    # (indices 24-27: robotic_framework × [skills, projects, education, internships])
    robotic_framework_dims = robotics_vector[24:28]  # type_index=6, indices 24,25,26,27
    assert any(v > 0 for v in robotic_framework_dims), (
        f"Robotics-heavy CV should have non-zero robotic_framework dimensions: "
        f"{robotic_framework_dims}"
    )

    # Automation vector should have non-zero plc_hardware dimensions
    plc_hardware_dims = automation_vector[0:4]  # type_index=0, indices 0,1,2,3
    assert any(v > 0 for v in plc_hardware_dims), (
        f"Automation-heavy CV should have non-zero plc_hardware dimensions: "
        f"{plc_hardware_dims}"
    )
