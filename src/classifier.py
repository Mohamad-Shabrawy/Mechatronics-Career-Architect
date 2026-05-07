"""
classifier.py — Niche Prediction Entry Point and Prediction Logging

This module exposes predict_niche(), the single public inference entry point for Phase 3.

What predict_niche() does:
  1. Validates the input feature vector (must be exactly 40 elements)
  2. Calls the trained Random Forest's predict_proba() to get class probabilities
  3. Returns the predicted niche (highest probability class) and its confidence score
  4. Writes a structured log entry to logs/predictions.jsonl (always — even on error)
  5. NEVER raises an exception — all failures return error dicts

Why "never raise"?
  The caller (Phase 4, 5, or 6) may be processing thousands of CVs in a batch. One
  bad CV should not crash the whole pipeline. Returning {"error": ...} lets the
  caller decide what to do with failures. This is the contract guarantee (Behavioral
  Guarantee #1 in classifier_contract.md).

Why log every prediction?
  The prediction log (logs/predictions.jsonl) enables:
    - Data drift detection: if the input_hash distribution shifts over time, the
      model's training data may be stale
    - Audit: confirm what the model predicted for a given input (using the hash)
    - Performance monitoring: confidence score trends reveal when the model is
      becoming uncertain

Why store only a hash, not the raw vector?
  The feature vector is derived from a real person's CV — their career history,
  skills, and education. Even though it's a 40-number integer vector (not raw text),
  it's still personal data. We store only an 8-character hash (FR-014, PII protection):
  enough to correlate two calls on the same input (same CV → same hash), but
  mathematically impossible to reverse-engineer the vector from.

Contract reference: specs/003-ml-niche-classifier/contracts/classifier_contract.md
Data model:         specs/003-ml-niche-classifier/data-model.md
"""

import hashlib
import json
import pathlib
from datetime import datetime, timezone

from src.model_io import CANONICAL_NICHES, load_model  # re-export load_model for callers


# The path where all prediction log entries are appended.
# Using a JSONL format (one JSON object per line) makes it easy to stream, grep,
# and process with standard tools like jq or Python's json.loads() line by line.
_PREDICTION_LOG_PATH = pathlib.Path("logs") / "predictions.jsonl"


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def predict_niche(feature_vector: list, model_artifact: dict) -> dict:
    """
    Predicts the most likely Mechatronics niche for a CV represented by its
    40-dimensional Phase 2 feature vector.

    This is the sole public inference entry point. It is designed to NEVER raise
    an exception — all failure modes return a dict with an "error" key.

    Success return value:
        {
            "predicted_niche": str,   # One of the 6 canonical niche names
            "confidence":      float, # Class probability in [0.0, 1.0]
        }

    Error return value:
        {
            "error":   str,   # "INVALID_VECTOR" | "MODEL_ERROR"
            "message": str,   # Human-readable explanation
        }

    Behavioral guarantees (from classifier_contract.md):
      - Always returns one of the 6 canonical niche names on success (never null/OOV)
      - Always appends a PredictionLogEntry to logs/predictions.jsonl
      - Confidence is always in [0.0, 1.0]
      - All-zero vector is valid input and returns a valid (not error) prediction
      - NEVER raises — even if the model artifact is wrong or the log write fails

    Parameters
    ----------
    feature_vector : list
        A 40-element list of integers (the Phase 2 feature vector).
        All-zero vectors are valid inputs — they represent CVs with no
        recognized entities and will produce a low-confidence prediction.
    model_artifact : dict
        A validated artifact dict returned by load_model().
        Must have "model" and "niche_labels" keys.

    Returns
    -------
    dict
        Either a success dict (predicted_niche + confidence) or an error dict.
    """
    # Compute the input hash FIRST, before any validation, so we can include it
    # in the log entry even when the vector is invalid. This ensures 100% log coverage
    # including error cases (SC-008 behavioral guarantee).
    input_hash = _compute_input_hash(feature_vector)
    model_version = model_artifact.get("model_version", "unknown") if isinstance(model_artifact, dict) else "unknown"

    # ── Validation: feature vector must have exactly 40 elements ─────────────
    # We check length before passing to the model because sklearn will raise a
    # cryptic dimension error otherwise. Our validation gives a clear message.
    if not isinstance(feature_vector, (list, tuple)) or len(feature_vector) != 40:
        # Even for invalid input, we write a log entry with predicted_niche=null
        # so the audit trail is complete and drift detection can see bad inputs.
        _write_prediction_log(
            input_hash=input_hash,
            model_version=model_version,
            predicted_niche=None,
            confidence=None,
        )
        length_info = len(feature_vector) if hasattr(feature_vector, "__len__") else "N/A"
        return {
            "error":   "INVALID_VECTOR",
            "message": (
                f"Feature vector must have exactly 40 elements, got {length_info}. "
                "Ensure the input comes from Phase 2 enrich_cv()['feature_vector']."
            ),
        }

    # ── Inference: call the trained model ────────────────────────────────────
    # Wrap everything in a try/except so that ANY unexpected error in the model
    # (corrupted weights, numpy version mismatch, etc.) returns an error dict
    # instead of propagating an exception to the caller.
    try:
        model = model_artifact["model"]
        niche_labels = model_artifact["niche_labels"]

        # reshape for sklearn: it expects a 2D array [[...40 features...]]
        # We use a plain list here rather than numpy to avoid an import just for this.
        X = [list(feature_vector)]  # shape: (1, 40)

        # predict_proba returns shape (1, n_classes) — we take the first (and only) row.
        probabilities = model.predict_proba(X)[0]

        # Find the class index with the highest probability.
        # argmax gives us the integer index; we map it to the niche name using
        # niche_labels (which is in the same order as the classifier's .classes_).
        best_index = int(probabilities.argmax())
        predicted_niche = niche_labels[best_index]
        confidence = float(probabilities[best_index])

        # Normalize the predicted_niche to ensure it's lowercase canonical form.
        # In practice it should already be canonical (saved that way in save_model()),
        # but this defensive step prevents any case mismatch from propagating downstream.
        predicted_niche = predicted_niche.lower()

        # Log the successful prediction.
        _write_prediction_log(
            input_hash=input_hash,
            model_version=model_version,
            predicted_niche=predicted_niche,
            confidence=confidence,
        )

        return {
            "predicted_niche": predicted_niche,
            "confidence":      confidence,
        }

    except Exception as exc:
        # Something went wrong inside the model inference — corrupted weights,
        # dimension mismatch, etc. We return an error dict and log the failure.
        _write_prediction_log(
            input_hash=input_hash,
            model_version=model_version,
            predicted_niche=None,
            confidence=None,
        )
        return {
            "error":   "MODEL_ERROR",
            "message": f"Model inference failed: {exc}",
        }


# ── PRIVATE HELPERS ───────────────────────────────────────────────────────────

def _compute_input_hash(feature_vector) -> str:
    """
    Computes an 8-character hash of the feature vector for the prediction log.

    Why this specific hash formula?
      str(sorted(enumerate(feature_vector))) produces a canonical, ordered
      string representation of the vector's values with their indices:
        "[(0, 1), (1, 0), (2, 3), ...]"
      This is stable across Python versions, deterministic, and unique to the vector's
      content (two different vectors won't produce the same string unless they're
      actually identical). We then SHA-256 it and truncate to 8 hex chars.

    Why sorted(enumerate(...))?
      It ensures the hash is consistent regardless of any future changes in how
      the vector is stored internally. The enumerate() preserves value position.

    If the feature_vector is not iterable (e.g., None or wrong type), we return
    a placeholder hash rather than crashing — input validation happens separately.
    """
    try:
        canonical = str(sorted(enumerate(feature_vector)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    except Exception:
        # If hashing fails (e.g., None was passed), return a fixed sentinel.
        # The log entry will still be written, just with a generic hash.
        return "00000000"


def _write_prediction_log(
    input_hash: str,
    model_version: str,
    predicted_niche,   # str | None — None for error cases
    confidence,        # float | None — None for error cases
) -> None:
    """
    Appends one PredictionLogEntry JSON line to logs/predictions.jsonl.

    This function is called for EVERY predict_niche() call — both successes and
    errors. The log entry includes predicted_niche=null and confidence=null for
    error cases, which lets monitoring tools detect and count failures.

    PredictionLogEntry schema (from data-model.md):
        {
            "timestamp":        str,   # ISO 8601 UTC
            "model_version":    str,   # run_id from the model artifact
            "input_hash":       str,   # 8-char SHA-256 of the feature vector
            "predicted_niche":  str | null,
            "confidence":       float | null,
        }

    If the log write itself fails (disk full, permissions), we print a warning to
    stderr and continue — the prediction result is still valid and must be returned
    to the caller. We never let logging failures break inference.
    """
    # Build the log entry dict.
    log_entry = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "model_version":   model_version,
        "input_hash":      input_hash,
        "predicted_niche": predicted_niche,
        "confidence":      confidence,
    }

    # Create the logs/ directory if it doesn't exist.
    # We do this on every call rather than once at import time because the directory
    # might be deleted between calls (e.g., in test environments that clean up).
    try:
        log_path = _PREDICTION_LOG_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode — each call adds exactly one line.
        # We don't hold the file open between calls (no state in this module),
        # which keeps things simple and avoids locking issues.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        # Log write failure is not fatal — print warning but don't raise.
        import sys
        print(
            f"WARNING: Failed to write prediction log entry: {exc}",
            file=sys.stderr,
        )
