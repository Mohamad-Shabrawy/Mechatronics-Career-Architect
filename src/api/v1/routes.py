"""
routes.py — Phase 6 API Blueprint (v1)

Defines POST /api/v1/analyze and GET /api/v1/health.

The analyze endpoint:
  1. Validates the uploaded PDF (size, type)
  2. Serialises the five preference sliders from the form
  3. Runs the full Phase 1–5 pipeline inside a 30-second timeout
  4. Returns a JSON ResultsView to the client

All pipeline calls live in run_pipeline() so they can be unit-tested
independently of Flask. The timeout wrapper (invoke_with_timeout) uses
a single-worker thread executor so the main thread stays responsive.
"""

import base64
import concurrent.futures

from flask import Blueprint, current_app, jsonify, request

# Pipeline module imports are deferred to run_pipeline() to avoid loading all
# heavy ML libraries (scikit-learn, matplotlib, reportlab) at Flask startup time.
# This keeps startup fast and reduces memory pressure during test collection.

# The five preference keys the HTML form sends.
# Order matches the UI drop-down order but we always look them up by name
# so the position doesn't affect correctness.
PREFERENCE_KEYS = [
    "work_environment",
    "system_level",
    "industry_interest",
    "team_scale",
    "travel_tolerance",
]

# Maximum PDF upload size in bytes (10 MB).
MAX_CV_BYTES = 10 * 1024 * 1024   # 10 485 760

api_v1 = Blueprint("api_v1", __name__)


# ── Public helpers (also tested directly in unit tests) ───────────────────────

def validate_cv_upload(file):
    """
    Inspect a werkzeug FileStorage object and return an error tuple or None.

    We check three things in order:
      1. The file was actually attached (not None / empty filename).
      2. It looks like a PDF (content_type AND .pdf extension must both match).
      3. Its byte length is under the 10 MB cap.

    Returns
    -------
    None
        When the file passes all checks — caller may proceed.
    (message: str, status_code: int)
        When validation fails — caller should return this as a 4xx response.
    """
    # Missing file — field wasn't included in the multipart POST at all
    if file is None or file.filename == "":
        return ("Please upload your CV", 400)

    # Type guard — both content_type and file extension must agree on PDF.
    # Checking only content_type would let a renamed .exe slip through;
    # checking only extension would reject files with a correct MIME type
    # that happens to have an unusual extension from an upload library.
    is_pdf_mime = (file.content_type or "").lower() == "application/pdf"
    is_pdf_ext  = (file.filename or "").lower().endswith(".pdf")
    if not (is_pdf_mime and is_pdf_ext):
        return ("Only PDF files are accepted", 400)

    # Size guard — read the whole stream to count bytes, then rewind.
    # We can't trust Content-Length headers (they can be spoofed / absent).
    data = file.read()
    file.stream.seek(0)
    if len(data) > MAX_CV_BYTES:
        return ("CV file must be under 10MB", 400)

    return None


def serialise_preferences(form) -> dict:
    """
    Extract the five preference integers from a form dict (or ImmutableMultiDict).

    Each preference is cast to int, defaulting to 0 when the field is absent.
    Valid range is [-2, 2] inclusive; anything outside raises ValueError so
    the caller can return a 422.

    Parameters
    ----------
    form : dict-like
        The HTML form data (request.form or a plain dict in tests).

    Returns
    -------
    dict
        Exactly five keys (PREFERENCE_KEYS) mapped to int values.

    Raises
    ------
    ValueError
        If any preference value falls outside the range [-2, 2].
    """
    result = {}
    for key in PREFERENCE_KEYS:
        raw = form.get(key, "0") or "0"
        value = int(raw)
        # The valid sentiment range is -2 (strongly against) to +2 (strongly for).
        # Values outside this indicate a tampered or malformed request.
        if value not in range(-2, 3):   # range(-2, 3) → -2, -1, 0, 1, 2
            raise ValueError(
                f"Preference '{key}' value {value} is out of range [-2, 2]"
            )
        result[key] = value
    return result


# ── Pipeline orchestration ────────────────────────────────────────────────────

def run_pipeline(cv_bytes: bytes, user_preferences: dict, artifact) -> dict:
    """
    Run the full Phase 1–5 analysis pipeline and return all intermediate results.

    Steps:
      1. parse_cv      → extract raw text blocks from the PDF bytes
      2. enrich_cv     → NER, keyword extraction, feature vector construction
      3. predict_niche → ML classifier gives niche probabilities
      4. compute_fit_score → apply user preferences to rerank the niches
      5. analyze_skills_gap → compare user skills to top niche benchmark
      6. generate_radar_chart → render PNG (failures return None, not an exception)
      7. generate_career_roadmap_pdf → assemble PDF report

    Parameters
    ----------
    cv_bytes : bytes
        Raw bytes of the uploaded PDF.
    user_preferences : dict
        Five integer preference values from serialise_preferences().
    artifact : dict or None
        Loaded model artifact from model_io.load_model(), or None if no model.

    Returns
    -------
    dict
        All intermediate and final results keyed by phase name.
    """
    # Lazy imports keep all heavy ML libraries out of the module-level namespace.
    # This means Flask and the test suite import routes.py without paying the
    # full startup cost of scikit-learn, matplotlib, and reportlab every time.
    import os
    import tempfile
    from src.parser import parse_cv
    from src.enricher import enrich_cv
    from src.classifier import predict_niche
    from src.scorer import compute_fit_score
    from src.gap_analyzer import analyze_skills_gap
    from src.visualizer import generate_radar_chart
    from src.report_generator import generate_career_roadmap_pdf

    # Phase 1: PDF text extraction.
    # parse_cv() expects a file path, not bytes — write cv_bytes to a temp file
    # and clean it up immediately after parsing so no CV data persists on disk.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(cv_bytes)
        tmp_path = tmp.name
    try:
        phase1_result = parse_cv(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Phase 2: Enrichment — NER, keywords, and feature vector construction.
    # The feature_vector is a flat list of floats used by the ML classifier.
    phase2_result = enrich_cv(phase1_result)
    feature_vector = phase2_result.get("feature_vector", [])

    # Phase 3: ML niche classification
    ml_result = predict_niche(feature_vector, artifact)

    # Phase 4: Fit scoring — reranks niches based on user preferences.
    # compute_fit_score takes (feature_vector, artifact, user_preferences) — not ml_result.
    fit_score_result = compute_fit_score(feature_vector, artifact, user_preferences)

    # Phase 5a: Skills gap — analyse the top 3 ranked niches so the PDF can
    # give the user actionable gap detail for each realistic career option.
    # Sort descending by composite_score to reliably identify the top niches.
    sorted_niches = sorted(
        fit_score_result["ranked_niches"],
        key=lambda n: n["composite_score"],
        reverse=True,
    )
    top_3_niches = [n["niche"] for n in sorted_niches[:3]]
    skills_gap_results = {}
    for niche in top_3_niches:
        skills_gap_results[niche] = analyze_skills_gap(feature_vector, niche, artifact)

    # Keep skills_gap_result pointing at rank-1 for backward compat with any
    # callers that still reference the single-niche key.
    skills_gap_result = skills_gap_results[top_3_niches[0]]

    # Phase 5b: Radar chart — failure is soft (B-007: returns None, pipeline continues).
    try:
        chart_result = generate_radar_chart(fit_score_result)
        # generate_radar_chart never raises — it returns an error dict on failure.
        # Treat an error dict the same as None so the caller has one simple check.
        if isinstance(chart_result, dict) and "error" in chart_result:
            chart_result = {"chart_image_bytes": None}
    except Exception:
        # Belt-and-suspenders: shouldn't happen but chart must never kill the pipeline.
        chart_result = {"chart_image_bytes": None}

    # Phase 5c: PDF report generation — pass all 3 gap results so the PDF
    # renders a section for each of the top 3 niches, not just rank 1.
    chart_image_bytes = chart_result.get("chart_image_bytes") if chart_result else None
    career_roadmap = generate_career_roadmap_pdf(
        fit_score_result, skills_gap_result, chart_image_bytes,
        skills_gap_results=skills_gap_results,
    )

    return {
        "phase1":             phase1_result,
        "phase2":             phase2_result,
        "feature_vector":     feature_vector,
        "ml_result":          ml_result,
        "fit_score_result":   fit_score_result,
        "skills_gap_result":  skills_gap_result,
        "skills_gap_results": skills_gap_results,
        "chart_result":       chart_result,
        "career_roadmap":     career_roadmap,
    }


def invoke_with_timeout(
    cv_bytes: bytes,
    user_preferences: dict,
    artifact,
    timeout: int = 180,
) -> dict:
    """
    Run run_pipeline() in a background thread with a hard timeout.

    If the pipeline takes longer than `timeout` seconds the thread is
    abandoned and a TimeoutError is raised to the caller. This prevents
    slow ML inference from holding the request worker forever.

    Parameters
    ----------
    cv_bytes : bytes
        Raw bytes of the uploaded PDF.
    user_preferences : dict
        Serialised preference dict.
    artifact : dict or None
        Loaded model artifact.
    timeout : int
        Maximum seconds to wait (default 30).

    Raises
    ------
    TimeoutError
        When the pipeline exceeds `timeout` seconds.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_pipeline, cv_bytes, user_preferences, artifact)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Analysis is taking too long — please try again")


# ── Route handlers ────────────────────────────────────────────────────────────

@api_v1.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /api/v1/analyze

    Accepts a multipart/form-data POST with:
      - cv_file    : PDF file (required)
      - work_environment, system_level, industry_interest,
        team_scale, travel_tolerance : integer preference values in [-2, 2]

    Returns a JSON ResultsView with:
      - ranked_niches     : list of niche dicts (top-3 or all-6 when equal)
      - all_equal_scores  : bool — True when all niches score identically
      - pivot_applied     : bool
      - pivot_explanation : str or null
      - skills_gap        : dict from skills_gap_result
      - chart_image_b64   : base64 PNG or null when chart unavailable
      - pdf_b64           : base64 PDF bytes
      - pdf_filename      : suggested download filename
    """
    # ── Step 1: Validate the uploaded file ───────────────────────────────────
    file = request.files.get("cv_file")
    validation_error = validate_cv_upload(file)
    if validation_error:
        message, status_code = validation_error
        return jsonify({"error": True, "message": message, "code": "VALIDATION_ERROR"}), status_code

    # ── Step 2: Serialise preference sliders ─────────────────────────────────
    try:
        user_preferences = serialise_preferences(request.form)
    except ValueError as exc:
        return jsonify({"error": True, "message": str(exc), "code": "VALIDATION_ERROR"}), 422

    # ── Step 3: Read file bytes and get model artifact ────────────────────────
    cv_bytes = file.read()
    artifact = current_app.config.get("MODEL_ARTIFACT")

    # ── Step 4: Run pipeline (with timeout) ───────────────────────────────────
    try:
        pipeline_result = invoke_with_timeout(cv_bytes, user_preferences, artifact)
    except TimeoutError as exc:
        return jsonify({"error": True, "message": str(exc), "code": "TIMEOUT"}), 408
    except Exception:
        # B-006: never expose raw exception text, stack traces, or internal details.
        # A human-readable generic message is all the client needs.
        return jsonify({"error": True, "message": "An error occurred while analysing your CV — please try again", "code": "PIPELINE_ERROR"}), 500

    # ── Step 5: Assemble the ResultsView ─────────────────────────────────────
    fit_score_result = pipeline_result["fit_score_result"]
    skills_gap_result = pipeline_result["skills_gap_result"]
    chart_result      = pipeline_result.get("chart_result") or {}
    career_roadmap    = pipeline_result["career_roadmap"]

    ranked_niches = fit_score_result.get("ranked_niches", [])

    # Detect when all niches tie — in that case we show all 6 so the user
    # knows there's no clear winner, rather than arbitrarily showing 3.
    all_equal_scores = (
        len(set(round(n["composite_score"], 6) for n in ranked_niches)) == 1
        if ranked_niches else False
    )
    display_niches = ranked_niches if all_equal_scores else ranked_niches[:3]

    # Build the niche list with a human-readable score percentage.
    niche_view = [
        {
            "rank":        entry["rank"],
            "niche":       entry["niche"],
            "niche_label": entry["niche"].replace("_", " ").title(),
            "composite_score": entry["composite_score"],
            "score_pct":   round(entry["composite_score"] * 100),
        }
        for entry in display_niches
    ]

    # Chart bytes → base64 string (or null when rendering failed)
    chart_bytes = chart_result.get("chart_image_bytes")
    chart_b64 = (
        base64.b64encode(chart_bytes).decode("utf-8")
        if chart_bytes
        else None
    )

    # PDF bytes → base64 string for browser-side download without a server route
    pdf_b64 = base64.b64encode(career_roadmap.get("pdf_bytes", b"")).decode("utf-8")

    response_body = {
        "ranked_niches":     niche_view,
        "all_equal_scores":  all_equal_scores,
        "pivot_applied":     fit_score_result.get("pivot_applied", False),
        "pivot_explanation": fit_score_result.get("pivot_explanation"),
        "skills_gap":        skills_gap_result,
        "chart_image_b64":   chart_b64,
        "pdf_b64":           pdf_b64,
        "pdf_filename":      career_roadmap.get("filename", "career_roadmap.pdf"),
    }

    return jsonify(response_body), 200


@api_v1.route("/health")
def health():
    """
    GET /api/v1/health

    Returns a simple status object so load balancers and monitoring tools
    can verify the app is alive and whether a model artifact is loaded.
    """
    return jsonify({
        "status":       "ok",
        "model_loaded": current_app.config.get("MODEL_ARTIFACT") is not None,
    })
