"""
model_io.py — Model Artifact Serialization, Loading, and Schema Version Checking

This module is the single authority for:
  1. The six canonical Mechatronics niche labels (CANONICAL_NICHES)
  2. The current Phase 2 feature vector schema version (current_schema_version)
  3. Saving a trained model artifact to disk (save_model)
  4. Loading a model artifact from disk with full validation (load_model)
  5. Extracting feature importance from a loaded artifact (get_feature_importance)

Why does schema versioning matter?
  The Phase 2 feature vector layout (VECTOR_INDEX_ORDER in feature_vector.py) is a
  fixed 40-element sequence of (type, section) pairs. If that layout changes — even
  slightly — a model trained on the old layout will silently predict garbage on data
  from the new layout, because dimensions 7 and 23 now mean different things.
  By embedding a short hash of VECTOR_INDEX_ORDER in every model artifact and checking
  it at load time, we guarantee that mismatched models are caught immediately, loudly,
  before they can produce any incorrect predictions.

Contract reference: specs/003-ml-niche-classifier/contracts/classifier_contract.md
Data model:         specs/003-ml-niche-classifier/data-model.md
"""

import hashlib
import pathlib

import joblib

from src.feature_vector import VECTOR_INDEX_ORDER


# ── CANONICAL NICHE LABELS ────────────────────────────────────────────────────
# These are the six valid output labels for the classifier. They are lowercase
# with underscores (matching the data model spec). Every other Phase 3 module
# imports this list from here — it is NEVER redefined elsewhere.
#
# The ordering matters: it determines the class-index-to-label mapping when we
# align RandomForestClassifier's .classes_ array with human-readable names.
CANONICAL_NICHES: list[str] = [
    "industrial_automation",
    "robotics",
    "embedded_systems",
    "automotive",
    "mechanical_design",
    "technical_management",
]


# ── CUSTOM EXCEPTION CLASSES ─────────────────────────────────────────────────
# These are raised by load_model() when something is wrong with the artifact.
# They are distinct types so callers can handle each failure mode differently.

class SchemaVersionError(Exception):
    """
    Raised when the model artifact's embedded schema version does not match
    the current Phase 2 vector index layout. This means the model was trained
    on a different feature vector definition and its predictions would be wrong.
    The error message always shows both the artifact version and the current version.
    """
    pass


class ModelLoadError(Exception):
    """
    Raised when the model artifact file cannot be read — either because the file
    is missing, corrupted, or not a valid joblib file. This is a fatal I/O error;
    the caller must supply a different model path.
    """
    pass


class ModelIntegrityError(Exception):
    """
    Raised when the model artifact loads successfully from disk but is missing
    required keys or has incomplete niche labels. This indicates the file was
    produced by an incompatible or corrupted training pipeline.
    """
    pass


# ── SCHEMA VERSION ────────────────────────────────────────────────────────────

def current_schema_version() -> str:
    """
    Returns an 8-character hex string that fingerprints the current Phase 2
    feature vector layout.

    How it works:
      We convert VECTOR_INDEX_ORDER (the locked 40-element list of (type, section)
      tuples) to a string, then take the SHA-256 hash of that string, then truncate
      to 8 hex characters. Short enough to embed in filenames and JSON; long enough
      to make accidental collisions essentially impossible for this use case.

    Why str(VECTOR_INDEX_ORDER)?
      The list of tuples has a stable Python str() representation:
        "[('plc_hardware', 'skills'), ('plc_hardware', 'projects'), ...]"
      Any change to the list's content or order changes this string and therefore
      changes the hash.

    This function is called at model save time (to embed the version) and at model
    load time (to verify the embedded version). Both calls happen in the same Python
    environment with the same VECTOR_INDEX_ORDER, so they must agree unless the
    source code changed between training and inference — which is precisely the
    mismatch we want to catch.
    """
    # Encode the canonical string representation of VECTOR_INDEX_ORDER.
    # We don't need to sort it — the order IS the schema (changing the order
    # changes which dimension means what, and that's exactly what we're hashing).
    content = str(VECTOR_INDEX_ORDER).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:8]


# ── MODEL SERIALIZATION ───────────────────────────────────────────────────────

def save_model(
    model,
    niche_labels: list[str],
    training_record: dict,
    run_id: str,
    output_dir: str,
    niche_centroids: dict | None = None,
    niche_top_skills: dict | None = None,
) -> str:
    """
    Serializes a trained model and its metadata to a single .joblib artifact file.

    The artifact dict structure (documented in data-model.md):
      {
          "model":            <RandomForestClassifier>,   # the trained estimator
          "niche_labels":     list[str],                  # 6 canonical names (class order)
          "training_record":  dict,                       # full TrainingRecord
          "schema_version":   str,                        # 8-char VECTOR_INDEX_ORDER hash
          "model_version":    str,                        # == run_id
          "niche_centroids":  dict[str, list[float]],     # Phase 4: per-niche centroid vectors
      }

    Why store niche_labels in the artifact?
      RandomForestClassifier.classes_ gives us the class labels it was trained on,
      but they're stored as numpy strings. By also storing our clean Python list of
      canonical names (in the same order as the classifier's internal classes_),
      load_model() can validate label completeness without numpy operations.

    Why store niche_centroids in the artifact?
      Phase 4 scorer.py computes cosine similarity between a new CV's feature vector
      and each niche's centroid (mean feature vector from training). Embedding the
      centroids in the artifact keeps training and inference decoupled — inference
      never needs access to the original training dataset (research.md R-005).
      If niche_centroids is None (legacy callers or unit tests that don't need Phase 4),
      the artifact will not include this key and load_model() will raise ModelIntegrityError
      when Phase 4 scoring is attempted — which is the correct behaviour per the spec.

    Parameters
    ----------
    model : RandomForestClassifier
        The fitted scikit-learn classifier to serialize.
    niche_labels : list[str]
        The 6 canonical niche names in class-index order (must match model.classes_).
    training_record : dict
        The complete TrainingRecord dict produced during training.
    run_id : str
        The unique run identifier (format: YYYYMMDD_HHMMSS_seed{N}).
    output_dir : str
        Directory path where the .joblib file will be written.
    niche_centroids : dict | None
        Phase 4 backfill: a dict mapping each canonical niche name to its 40D centroid
        vector (list[float]). If provided, embedded in the artifact under "niche_centroids".
        If None, the key is omitted (legacy artifact — Phase 4 scoring will reject it).

    Returns
    -------
    str
        Absolute path of the saved .joblib file.

    Notes
    -----
    - The output directory is created automatically if it does not exist.
    - This function raises on I/O errors (caller = trainer.py, which handles them).
    """
    # Ensure the output directory exists — create it (and parents) if needed.
    # We do this here rather than assuming it was created at startup, because
    # trainer.py is the only code that calls save_model() and it may not create
    # the directory itself.
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the artifact dict with all required fields.
    artifact = {
        "model":           model,
        "niche_labels":    niche_labels,
        "training_record": training_record,
        "schema_version":  current_schema_version(),
        "model_version":   run_id,
    }

    # Phase 4 backfill: embed per-niche centroid vectors if provided.
    # These are used by scorer.py to compute cosine similarity at scoring time.
    # We only add the key if centroids were computed — omitting it from legacy
    # callers is intentional; load_model() will surface the absence as an error
    # when Phase 4 scoring is attempted, which is safer than silent bad data.
    if niche_centroids is not None:
        artifact["niche_centroids"] = niche_centroids

    # Phase 5 backfill: embed per-niche top-10 skill dimensions if provided.
    # These are used by gap_analyzer.py to map centroid dimensions to human-readable
    # skill names from niche_benchmarks.json. Each entry is a list of 10
    # (dim_index, label) tuples ranked by centroid value descending.
    if niche_top_skills is not None:
        artifact["niche_top_skills"] = niche_top_skills

    # Write the artifact to <output_dir>/<run_id>.joblib
    file_path = output_path / f"{run_id}.joblib"
    joblib.dump(artifact, str(file_path))

    # Return the absolute path so trainer.py can record it in the TrainingRecord.
    return str(file_path.resolve())


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_model(model_path: str) -> dict:
    """
    Loads a serialized model artifact from disk and validates it before returning.

    Validation is performed in three stages, each raising a distinct exception
    if it fails (no silent bad-data passes through):

    Stage 1 — I/O: Can we read the file at all?
      Catches FileNotFoundError (file doesn't exist) and any other exception
      from joblib.load() (corrupted file, wrong format, etc.).
      Raises: ModelLoadError

    Stage 2 — Schema version: Was this model trained on the current feature schema?
      Compares artifact["schema_version"] against current_schema_version().
      If they differ, the model's 40 dimensions don't match our current 40 dimensions —
      predictions would be silently wrong.
      Raises: SchemaVersionError (with both versions in the message)

    Stage 3 — Structural integrity: Does the artifact have all required keys and labels?
      Checks that "model", "niche_labels", "schema_version", "model_version",
      and "training_record" are all present, and that all 6 canonical niches are
      present in niche_labels.
      Raises: ModelIntegrityError

    Parameters
    ----------
    model_path : str
        Path to the .joblib model artifact file.

    Returns
    -------
    dict
        The validated artifact dict, ready for use with predict_niche().

    Raises
    ------
    ModelLoadError
        If the file cannot be read or deserialized.
    SchemaVersionError
        If the artifact's schema version does not match the current Phase 2 schema.
    ModelIntegrityError
        If the artifact is missing required keys or has incomplete niche labels.
    """
    # ── Stage 1: Load from disk ──────────────────────────────────────────────
    # We catch ALL exceptions from joblib.load() and re-raise as ModelLoadError
    # so callers don't need to know about joblib internals.
    try:
        artifact = joblib.load(model_path)
    except FileNotFoundError:
        raise ModelLoadError(
            f"Model file not found: {model_path}\n"
            "Run training first to create a model artifact."
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model artifact from '{model_path}': {exc}\n"
            "The file may be corrupted or from an incompatible version of joblib."
        )

    # ── Stage 2: Schema version check ───────────────────────────────────────
    # This is the most important safety check. If training used a different
    # VECTOR_INDEX_ORDER than inference, every prediction dimension is wrong.
    required_keys = {"model", "niche_labels", "schema_version", "model_version", "training_record"}
    missing_keys = required_keys - set(artifact.keys() if isinstance(artifact, dict) else [])

    if not isinstance(artifact, dict) or missing_keys:
        raise ModelIntegrityError(
            f"Model artifact is not a valid dict or is missing required keys: "
            f"{missing_keys or 'artifact is not a dict'}"
        )

    artifact_schema = artifact.get("schema_version", "")
    live_schema = current_schema_version()
    if artifact_schema != live_schema:
        raise SchemaVersionError(
            f"Schema version mismatch!\n"
            f"  Model artifact schema: '{artifact_schema}'\n"
            f"  Current Phase 2 schema: '{live_schema}'\n"
            f"This model was trained on a different feature vector layout. "
            f"Re-train the model with the current Phase 2 schema to resolve this."
        )

    # ── Stage 3: Niche label completeness ────────────────────────────────────
    # The model's niche_labels list must contain all 6 canonical names.
    # If it's incomplete, the class-index-to-name mapping is broken.
    niche_labels = artifact.get("niche_labels", [])
    if not isinstance(niche_labels, list):
        raise ModelIntegrityError(
            f"'niche_labels' in artifact must be a list, got {type(niche_labels)}"
        )

    canonical_set = set(CANONICAL_NICHES)
    artifact_set = set(niche_labels)
    missing_niches = canonical_set - artifact_set
    if missing_niches:
        raise ModelIntegrityError(
            f"Model artifact is missing canonical niche labels: {missing_niches}\n"
            "This artifact may have been produced by a different version of the trainer."
        )

    # ── Stage 4: Centroid vector completeness (Phase 4 backfill) ────────────
    # Phase 4 scorer.py requires "niche_centroids" — a dict with one 40D float
    # list per canonical niche. We validate it here at load time so any problem
    # is caught immediately and loudly (research.md R-005, tasks.md T008).
    #
    # What we check:
    #   a) "niche_centroids" key is present (artifacts before Phase 4 lack it)
    #   b) The value is a dict — not a list or None
    #   c) All 6 canonical niches appear as keys
    #   d) Each centroid list has exactly 40 elements — matching the feature vector length
    #
    # We raise ModelIntegrityError (not SchemaVersionError) because this is a
    # structural completeness problem, not a schema mismatch.
    if "niche_centroids" not in artifact:
        raise ModelIntegrityError(
            "Model artifact is missing 'niche_centroids'. "
            "This artifact was trained before Phase 4 support was added. "
            "Retrain the model with Phase 4-compatible trainer.py to include centroids."
        )

    niche_centroids = artifact["niche_centroids"]
    if not isinstance(niche_centroids, dict):
        raise ModelIntegrityError(
            f"'niche_centroids' must be a dict, got {type(niche_centroids).__name__}. "
            "The artifact may be corrupted."
        )

    # Every canonical niche must have a centroid entry of exactly 40 floats.
    for niche in CANONICAL_NICHES:
        if niche not in niche_centroids:
            raise ModelIntegrityError(
                f"'niche_centroids' is missing entry for niche '{niche}'. "
                f"Present niches: {list(niche_centroids.keys())}. "
                "Retrain the model to regenerate all 6 centroid vectors."
            )
        centroid = niche_centroids[niche]
        if not isinstance(centroid, list) or len(centroid) != 40:
            actual_len = len(centroid) if isinstance(centroid, list) else "not a list"
            raise ModelIntegrityError(
                f"Centroid for niche '{niche}' must be a list of 40 floats, "
                f"got length {actual_len}. "
                "The artifact may have been saved with a different feature vector schema."
            )

    # ── Stage 5: niche_top_skills validation (Phase 5 backfill) ────────────
    # Phase 5 gap_analyzer.py requires "niche_top_skills" — a dict with one
    # list of 10 (dim_index, label) tuples per canonical niche, ranked by
    # centroid value descending.
    #
    # We validate it here at load time so any problem is caught immediately
    # and loudly, rather than surfacing as a confusing "MISSING_BENCHMARK"
    # error inside analyze_skills_gap() much later.
    #
    # What we check:
    #   a) "niche_top_skills" key is present (artifacts before Phase 5 lack it)
    #   b) The value is a dict
    #   c) All 6 canonical niches appear as keys
    #   d) Each entry is a list of exactly 10 items
    if "niche_top_skills" not in artifact:
        raise ModelIntegrityError(
            "artifact predates Phase 5 — retrain to generate niche_top_skills"
        )
    niche_top_skills = artifact["niche_top_skills"]
    if not isinstance(niche_top_skills, dict):
        raise ModelIntegrityError("'niche_top_skills' must be a dict")
    for niche in CANONICAL_NICHES:
        if niche not in niche_top_skills:
            raise ModelIntegrityError(
                f"'niche_top_skills' missing entry for niche '{niche}'"
            )
        entry = niche_top_skills[niche]
        if not isinstance(entry, list) or len(entry) != 10:
            raise ModelIntegrityError(
                f"'niche_top_skills[{niche}]' must be a list of 10 entries, "
                f"got {len(entry) if isinstance(entry, list) else type(entry)}"
            )

    # All checks passed — return the validated artifact.
    return artifact


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

def get_feature_importance(model_artifact: dict) -> dict[str, float]:
    """
    Extracts and labels the Random Forest feature importance scores from a model artifact.

    What feature importance tells us:
      Each of the 40 dimensions in our feature vector corresponds to one (entity_type,
      section) pair — e.g., index 0 = plc_hardware entities in the skills section.
      The Random Forest's Gini-based feature_importances_ array tells us how much
      each of these 40 dimensions contributed to splitting decisions across all trees.
      Higher score = that (type, section) combination was more useful for distinguishing
      between niches. This is both a model explanation tool (Constitution V) and a
      diagnostic for which kinds of entities matter most in which CV sections.

    Parameters
    ----------
    model_artifact : dict
        A validated artifact dict returned by load_model().

    Returns
    -------
    dict[str, float]
        A dict mapping "type_section" label strings to their importance scores,
        sorted descending by importance (most important first).
        Keys have the format: "<entity_type>_<section_name>",
        e.g. "plc_hardware_skills", "robotic_framework_projects".
        All 40 dimensions are always present in the returned dict.

    Notes
    -----
    - Importance scores sum to approximately 1.0 (floating-point rounding may
      cause slight deviation).
    - If the artifact's model has no feature_importances_ (e.g., it was not fitted),
      returns a dict of all 40 dimensions with 0.0 values.
    """
    model = model_artifact.get("model")

    # Extract the raw importance array from the fitted Random Forest.
    # feature_importances_ is a numpy array of shape (40,) — one value per dimension.
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        # Graceful fallback: model doesn't have importances (shouldn't happen for RF).
        # Return all-zero dict so callers don't crash.
        return {f"{t}_{s}": 0.0 for t, s in VECTOR_INDEX_ORDER}

    # Zip the importance values with the (type, section) labels from VECTOR_INDEX_ORDER.
    # This is the human-readable mapping that satisfies Constitution Principle V
    # (the explanation requirement for AI systems).
    labeled = {
        f"{type_key}_{section}": float(importance)
        for (type_key, section), importance in zip(VECTOR_INDEX_ORDER, importances)
    }

    # Sort descending by importance so the most influential dimensions appear first.
    # This makes the dict immediately useful for reporting without further processing.
    return dict(sorted(labeled.items(), key=lambda x: x[1], reverse=True))
