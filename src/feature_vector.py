"""
feature_vector.py — 40-Dimensional Feature Vector Construction

This module defines the fixed index layout for the Phase 2 feature vector and
provides the build_feature_vector() function that encodes a list of NamedEntity
records into that vector.

What the feature vector represents:
  Each of the 40 dimensions corresponds to one (entity_type, section) pair.
  The value at each dimension is the count of DISTINCT entity strings of that
  type found in that section of the CV. This encoding tells Phase 3 not just
  WHAT entities appeared, but WHERE in the CV they appeared — a "SolidWorks"
  listed under Skills signals a different level of expertise than one mentioned
  in passing inside an education block.

Index layout (MUST NOT CHANGE after Phase 3 begins consuming this):
  Index = (type_index × 4) + section_index

  type_index order  (matches ENTITY_TAXONOMY key order):
    0 = plc_hardware         4 = programming_language   8 = mechanical_tool
    1 = cad_software         5 = simulation_tool        9 = management_methodology
    2 = microcontroller      6 = robotic_framework
    3 = communication_protocol  7 = automotive_standard

  section_index order:
    0 = skills    1 = projects    2 = education    3 = internships

  Example:
    feature_vector[0]  → plc_hardware entities in skills
    feature_vector[1]  → plc_hardware entities in projects
    feature_vector[4]  → cad_software entities in skills
    feature_vector[20] → simulation_tool entities in skills

Spec reference: specs/002-semantic-ner-features/data-model.md (FeatureVector)
Contract reference: specs/002-semantic-ner-features/contracts/enricher_contract.md
"""

# ── VECTOR INDEX ORDER ────────────────────────────────────────────────────────
# This constant defines the complete 40-element layout of the feature vector.
# It is built by crossing 10 entity types with 4 sections in fixed order.
# The order here is permanent — changing it after Phase 3 starts would silently
# break the classifier because the model would interpret dimensions incorrectly.

# The 10 entity type keys in their fixed order (matches ENTITY_TAXONOMY key order).
# We declare this explicitly rather than importing ENTITY_TAXONOMY to avoid a
# circular dependency and to make the vector layout self-documenting in one place.
_TYPE_ORDER: list[str] = [
    "plc_hardware",
    "cad_software",
    "microcontroller",
    "communication_protocol",
    "programming_language",
    "simulation_tool",
    "robotic_framework",
    "automotive_standard",
    "mechanical_tool",
    "management_methodology",
]

# The 4 section names in their fixed order.
_SECTION_ORDER: list[str] = ["skills", "projects", "education", "internships"]

# VECTOR_INDEX_ORDER: a 40-element list of (type_key, section_name) tuples.
# Position i in this list tells you what (type, section) combination index i
# of the feature vector counts. Built at import time so it's computed once.
# The assert below is a module-level sanity check — if someone accidentally
# removes a type or section, this will catch it immediately at import time.
VECTOR_INDEX_ORDER: list[tuple[str, str]] = [
    (type_key, section)
    for type_key in _TYPE_ORDER
    for section in _SECTION_ORDER
]

# This is the locked contract: exactly 40 dimensions, always.
assert len(VECTOR_INDEX_ORDER) == 40, (
    f"VECTOR_INDEX_ORDER must have exactly 40 elements "
    f"(10 types × 4 sections), got {len(VECTOR_INDEX_ORDER)}"
)


def build_feature_vector(entities: list[dict]) -> list[int]:
    """
    Encodes a list of NamedEntity records into a fixed 40-dimensional integer vector.

    Parameters
    ----------
    entities : list[dict]
        A list of entity records, each having the shape:
            {"term": str, "type": str, "section": str}
        This is the deduplicated entity list produced by the NER scanner in
        enricher.py — each (term, type, section) triple appears at most once.

    Returns
    -------
    list[int]
        A list of exactly 40 integers. Each integer is the count of distinct
        entity strings of the corresponding type found in the corresponding
        section. All values are ≥ 0.

    Design notes:
    - Deduplication is assumed to have been done BEFORE this function is called
      (by the section-level dedup in enricher.py). So each entity in the input
      list is already unique by (term, type, section) triple.
    - The vector is all zeros for an empty entity list — this is the valid
      zero-vector for a CV with no recognizable entities.
    - We never raise exceptions. An unrecognized type/section pair is silently
      skipped (the corresponding dimension stays at 0). This shouldn't happen
      in normal operation because enricher.py only produces valid type/section
      combos, but it's safer than crashing.
    """
    # Start with a zero vector of exactly 40 elements.
    # We'll increment each dimension as we encounter the matching entities.
    vector: list[int] = [0] * 40

    for entity in entities:
        entity_type = entity.get("type", "")
        entity_section = entity.get("section", "")
        pair = (entity_type, entity_section)

        # Find the index for this (type, section) pair in our fixed layout.
        # If the pair isn't in the index (shouldn't happen in normal flow),
        # we skip it gracefully rather than crashing.
        try:
            index = VECTOR_INDEX_ORDER.index(pair)
        except ValueError:
            # This entity has a type or section that isn't in our fixed layout.
            # We log nothing and skip — the vector dimension stays at 0.
            # In normal operation this should never happen because enricher.py
            # only generates entities with valid types and section names.
            continue

        # Increment the count for this (type, section) slot.
        # Because deduplication was already done upstream, each entity here
        # represents a genuinely distinct term in that section.
        vector[index] += 1

    # Final sanity check — should always be true given the initialization above,
    # but this makes the contract explicit and catches any accidental truncation.
    assert len(vector) == 40, (
        f"Feature vector length should be 40, got {len(vector)}"
    )

    return vector
