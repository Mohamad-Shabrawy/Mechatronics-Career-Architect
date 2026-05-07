"""
enricher.py — Phase 2: Semantic Feature Engineering & NER

This is the sole public entry point for Phase 2. It consumes the structured
output from Phase 1 (parse_cv) and produces an enriched JSON artifact that
Phase 3 (the ML classifier) will consume.

What this module does, from start to finish:
  1. Accepts the Phase 1 dict (four section lists of text blocks + keywords)
  2. If Phase 1 returned an error, passes it through unchanged — no processing
  3. Scans every text block in every section for Mechatronics named entities
     using a case-insensitive whole-word regex lookup against ENTITY_TAXONOMY
  4. Deduplicates entities within each section (same term+type counted once per section)
  5. Maps each entity to one or more semantic clusters via SEMANTIC_MAP
  6. Builds a cluster-frequency profile (7-key dict of cluster counts)
  7. Encodes the entity list as a 40-dimensional feature vector (type × section)
  8. Returns a single enriched dict with all 7 keys: the 4 Phase 1 sections
     plus entities, cluster_profile, and feature_vector

Contract reference: specs/002-semantic-ner-features/contracts/enricher_contract.md
Data model:        specs/002-semantic-ner-features/data-model.md

Author: Phase 2 implementation
"""

import re

from src.entity_types import ENTITY_TAXONOMY
from src.semantic_map import SEMANTIC_MAP
from src.feature_vector import build_feature_vector

# Only expose enrich_cv to callers — all internal helpers stay private
__all__ = ["enrich_cv"]

# The four section names Phase 1 always produces. We iterate over these
# in a fixed order so the processing is deterministic across invocations.
_SECTIONS: list[str] = ["skills", "projects", "education", "internships"]

# The seven cluster keys that must ALWAYS appear in the cluster_profile output.
# Initializing with all seven at zero guarantees the schema is complete even
# when no entities are found.
_CLUSTER_KEYS: list[str] = [
    "industrial_automation",
    "robotics",
    "embedded_systems",
    "automotive",
    "mechanical_design",
    "technical_management",
    "unclassified",
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def enrich_cv(phase1_output: dict) -> dict:
    """
    The single public entry point for Phase 2.

    Accepts the structured output from parse_cv() (Phase 1) and returns an
    enriched dict that adds three new keys on top of the four Phase 1 keys:

        {
            # Phase 1 sections — preserved unchanged
            "skills":      [{"text": str, "keywords": list[str]}, ...],
            "projects":    [{"text": str, "keywords": list[str]}, ...],
            "education":   [{"text": str, "keywords": list[str]}, ...],
            "internships": [{"text": str, "keywords": list[str]}, ...],

            # Phase 2 additions
            "entities": [{"term": str, "type": str, "section": str}, ...],
            "cluster_profile": {
                "industrial_automation": int,
                "robotics": int,
                "embedded_systems": int,
                "automotive": int,
                "mechanical_design": int,
                "technical_management": int,
                "unclassified": int,
            },
            "feature_vector": list[int],  # exactly 40 elements
        }

    On Phase 1 error input:
        Returns the input dict unchanged — no Phase 2 processing occurs.
        This is by contract: if Phase 1 failed, we can't enrich anything.

    Behavioral guarantees (from enricher_contract.md):
        - NEVER raises an exception to the caller
        - feature_vector always has exactly 40 integer elements
        - cluster_profile always has exactly 7 keys
        - Each (term, type, section) triple in entities is unique
        - JSON-serializable output (json.dumps won't raise)
        - Empty Phase 1 input → valid zero-valued enriched output
    """
    # ── Guard 1: Phase 1 error pass-through ─────────────────────────────────
    # If Phase 1 failed, its output already has an "error" key.
    # We return it immediately — there's nothing to enrich.
    # This also handles the case where a non-dict is passed: we check for "error"
    # first, but if phase1_output isn't a dict, the subsequent code will handle it.
    if not isinstance(phase1_output, dict):
        # Someone passed something that's not a dict at all — return a safe error
        return {
            "error": "INVALID_INPUT",
            "message": "enrich_cv() expects a dict (Phase 1 output), got something else.",
        }

    if "error" in phase1_output:
        # Pass Phase 1 error dicts through unchanged — don't touch them
        return phase1_output

    try:
        # ── Step 1: Scan all sections for named entities ─────────────────────
        # Collect all entity records across all sections, then deduplicate within
        # each section by (term, type) pair. The _collect_all_entities helper
        # handles both scanning and deduplication in one pass.
        entities = _collect_all_entities(phase1_output)

        # ── Step 2: Build the cluster frequency profile ──────────────────────
        # Count how many entities map to each of the 7 cluster keys.
        # Multi-cluster entities increment multiple counters.
        cluster_profile = _build_cluster_profile(entities)

        # ── Step 3: Encode as a 40-dimensional feature vector ────────────────
        # Each dimension = count of distinct entities of one type in one section.
        feature_vector = build_feature_vector(entities)

        # ── Step 4: Assemble the enriched output ─────────────────────────────
        # Preserve the four Phase 1 section keys unchanged, then add the three
        # new Phase 2 keys on top.
        result = {
            # Phase 1 sections — copied by reference is fine; callers don't mutate
            "skills":        phase1_output.get("skills", []),
            "projects":      phase1_output.get("projects", []),
            "education":     phase1_output.get("education", []),
            "internships":   phase1_output.get("internships", []),
            # Phase 2 additions
            "entities":      entities,
            "cluster_profile": cluster_profile,
            "feature_vector":  feature_vector,
        }

        return result

    except Exception as exc:
        # Something unexpected went wrong inside Phase 2 processing.
        # We return an error dict instead of letting the exception propagate —
        # this is the "never raise to callers" guarantee from the contract.
        return {
            "error": "ENRICHMENT_FAILED",
            "message": f"Phase 2 enrichment encountered an unexpected error: {exc}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _scan_entities_in_text(text: str, section: str) -> list[dict]:
    """
    Scans a single text block for recognized Mechatronics named entities.

    For each entity string in ENTITY_TAXONOMY, we attempt a case-insensitive
    whole-word regex match against the text. If the entity string is found,
    we emit one entity record for EACH type that entity string belongs to.
    This is how multi-type entities (e.g., "matlab" → programming_language +
    simulation_tool) produce multiple records from a single text match.

    Returns a list of raw entity dicts (before section-level deduplication).
    The same term may appear multiple times if it matches more than once in
    the text — deduplication happens in _collect_all_entities().

    Parameters
    ----------
    text : str
        The raw text of one CV block (e.g. one bullet point or paragraph).
    section : str
        The canonical section name this block belongs to
        ("skills", "projects", "education", or "internships").

    Returns
    -------
    list[dict]
        Raw entity records (possibly with duplicates across blocks):
            [{"term": str, "type": str, "section": str}, ...]
    """
    found: list[dict] = []
    text_lower = text.lower()   # Lowercase once — all entity strings are lowercase

    # Iterate over every type and its entity list in the taxonomy.
    # We use the same approach as Phase 1's _recognize_keywords: whole-word
    # case-insensitive regex matching via re.search with \b boundaries.
    for type_key, entity_strings in ENTITY_TAXONOMY.items():
        for entity_string in entity_strings:
            # Build a whole-word regex. We use \b here because all our entity
            # strings consist of word characters, hyphens, spaces, and parentheses —
            # \b works correctly for alphanumeric boundaries.
            # re.escape() handles the hyphens, parentheses, and dots safely.
            pattern = r"\b" + re.escape(entity_string) + r"\b"

            if re.search(pattern, text_lower, re.IGNORECASE):
                # We found this entity in the text. Emit one record per type
                # that this entity string belongs to. Because we're already
                # inside the loop for type_key, we know this entity maps to
                # at least type_key. But an entity like "matlab" appears in
                # BOTH "programming_language" and "simulation_tool" lists —
                # so it will be found twice (once per type_key iteration) and
                # will correctly produce two records.
                found.append({
                    "term":    entity_string,   # Always lowercase (from ENTITY_TAXONOMY)
                    "type":    type_key,
                    "section": section,
                })

    return found


def _collect_all_entities(phase1_output: dict) -> list[dict]:
    """
    Scans all four CV sections and returns a deduplicated list of entity records.

    The process:
    1. For each section (skills, projects, education, internships):
       a. Collect every entity record from every text block in that section
       b. Deduplicate: within the same section, each (term, type) pair may
          only appear ONCE, regardless of how many blocks it was found in
    2. Combine the deduplicated results from all four sections into one flat list

    This is where the spec's "deduplicated within section" rule is enforced:
    if "SolidWorks" appears in three different bullet points under Skills, it
    should count as ONE entity in the Skills section, not three.

    Parameters
    ----------
    phase1_output : dict
        The Phase 1 structured output with four section lists.

    Returns
    -------
    list[dict]
        The flat, deduplicated list of all NamedEntity records.
        Each (term, type, section) triple appears at most once.
    """
    all_entities: list[dict] = []

    for section_name in _SECTIONS:
        # Get the list of text blocks for this section.
        # If the section key is missing or its value isn't a list (malformed input),
        # we treat it as empty and move on — no crash.
        section_blocks = phase1_output.get(section_name, [])
        if not isinstance(section_blocks, list):
            continue

        # Accumulate all raw entity hits from every block in this section.
        # These may have duplicates (same entity found in multiple blocks).
        raw_hits: list[dict] = []
        for block in section_blocks:
            if not isinstance(block, dict):
                continue
            block_text = block.get("text", "")
            if not isinstance(block_text, str) or not block_text:
                continue
            raw_hits.extend(_scan_entities_in_text(block_text, section_name))

        # Deduplicate within this section: each (term, type) pair counts once.
        # We track seen pairs in a set and only keep the first occurrence of each.
        # The section is already the same for all entries in raw_hits, so the
        # uniqueness key is just (term, type).
        seen_pairs: set[tuple[str, str]] = set()
        for hit in raw_hits:
            pair = (hit["term"], hit["type"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                all_entities.append(hit)

    return all_entities


def _build_cluster_profile(entities: list[dict]) -> dict:
    """
    Counts how many entities from the CV map to each of the seven cluster keys.

    For each entity, we look up its term in SEMANTIC_MAP. If the term is found,
    we increment all the cluster counters it maps to. If the term is NOT in the
    map, we increment "unclassified" — we never drop an entity silently.

    Multi-cluster entities (e.g., "autosar" → ["embedded_systems", "automotive"])
    increment multiple counters, so the sum of all cluster counts can exceed
    the total number of entities.

    Parameters
    ----------
    entities : list[dict]
        The deduplicated entity list from _collect_all_entities().

    Returns
    -------
    dict
        A 7-key dict with all cluster keys always present (values ≥ 0).
        {
            "industrial_automation": int,
            "robotics": int,
            "embedded_systems": int,
            "automotive": int,
            "mechanical_design": int,
            "technical_management": int,
            "unclassified": int,
        }
    """
    # Initialize all seven counters to zero — this guarantees the schema is
    # always complete even when no entities are found or no entity has a cluster.
    profile: dict[str, int] = {key: 0 for key in _CLUSTER_KEYS}

    for entity in entities:
        term = entity.get("term", "")

        # Look up the term in the static semantic map.
        # If it's not there, default to ["unclassified"] per FR-006.
        clusters = SEMANTIC_MAP.get(term, ["unclassified"])

        for cluster in clusters:
            # Increment the counter for this cluster.
            # If a cluster name from the map isn't in our profile keys
            # (shouldn't happen, but defensive programming), skip it.
            if cluster in profile:
                profile[cluster] += 1
            else:
                # Unknown cluster name — count it as unclassified rather than crashing
                profile["unclassified"] += 1

    return profile
