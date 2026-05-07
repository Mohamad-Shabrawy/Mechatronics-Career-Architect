"""
app.py — Flask application factory for the Mechatronics Career Architect web UI.

This module is the single entry point for creating a configured Flask app.
It loads the trained model artifact once at startup (avoiding the cost of
reloading it on every request), registers the API blueprint, and wires up
global error handlers for common HTTP errors.

Usage:
    export MODEL_PATH=models/20260418_005030_seed42.joblib
    python src/app.py          # dev server on localhost:5000
    flask --app src.app run    # alternative via Flask CLI
"""

import os

# joblib is imported lazily inside create_app() to avoid pulling numpy/OpenBLAS
# at module-import time. This keeps test collection fast and prevents memory
# pressure when tests don't need the real model artifact.
from flask import Flask, jsonify, request


def _warmup_pipeline_imports():
    """Pre-load all heavy ML/chart/PDF libraries at startup so first request is fast."""
    import src.parser, src.enricher, src.classifier, src.scorer
    import src.gap_analyzer, src.visualizer, src.report_generator


def create_app(test_config: dict | None = None) -> Flask:
    """
    Build and return a configured Flask application.

    Why a factory function instead of a module-level app object?
    Because tests need to create fresh app instances with different configs
    (e.g. MODEL_ARTIFACT=None to skip model loading) without contaminating
    each other through shared module state.

    Parameters
    ----------
    test_config : dict, optional
        Config overrides for testing. When provided, MODEL_PATH env var is
        ignored and the supplied config dict is applied directly.

    Returns
    -------
    Flask
        A fully configured Flask application ready to serve requests.
    """
    # Template and static folders live at the project root, one level above src/.
    # We compute them relative to this file's location so the app works regardless
    # of the current working directory when it's launched.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(src_dir, "../templates")
    static_dir = os.path.join(src_dir, "../static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
    )

    # ── Config ────────────────────────────────────────────────────────────────
    # Default config values — overridden by test_config when supplied.
    app.config.setdefault("MAX_CONTENT_LENGTH", 15 * 1024 * 1024)  # 15MB Werkzeug safety net

    if test_config is not None:
        # Tests pass their own config (e.g. MODEL_ARTIFACT=None, TESTING=True).
        # We apply it without touching MODEL_PATH so CI doesn't need a real artifact.
        app.config.update(test_config)
    else:
        # Production path: load the trained model artifact from MODEL_PATH env var.
        # Failing loudly here (on startup) is intentional — a missing artifact
        # means no analysis is possible and the operator needs to know immediately.
        model_path = os.environ.get("MODEL_PATH", "")
        if model_path:
            import joblib
            artifact = joblib.load(model_path)
            app.config["MODEL_ARTIFACT"] = artifact
            # Pre-warm all heavy pipeline imports so the first HTTP request
            # doesn't pay the DLL loading cost (sklearn/matplotlib/reportlab).
            # On this machine DLL loading takes 2-5 min; doing it at startup
            # means requests are fast from the very first submission.
            _warmup_pipeline_imports()
        else:
            app.config["MODEL_ARTIFACT"] = None

    # ── Cache-busting for the HTML page ──────────────────────────────────────
    # The browser aggressively caches the root page, so dropdown option text
    # updates don't reach users until they hard-refresh. This after_request
    # hook forces a no-store policy on every HTML response so the browser
    # always fetches a fresh copy of index.html.
    @app.after_request
    def no_cache(response):
        if request.path == "/" or request.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.after_request
    def security_headers(response):
        # Prevent browsers from MIME-sniffing the response away from the declared content-type.
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        # Disallow embedding this page in any iframe — blocks clickjacking.
        response.headers.setdefault("X-Frame-Options", "DENY")
        # Only send the origin (no path/query) in the Referer header when crossing origins.
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Content Security Policy:
        #   script-src 'unsafe-inline' — required for the two inline <script> blocks in index.html
        #   style-src  'unsafe-inline' — required for JS-driven inline style mutations (parallax, tilt)
        #   img-src    data:           — required for the base64 radar chart injected by app.js
        #   font-src   fonts.gstatic   — Google Fonts files
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';",
        )
        return response

    # ── Blueprints ────────────────────────────────────────────────────────────
    from src.api.v1.routes import api_v1  # imported here to avoid circular imports
    app.register_blueprint(api_v1, url_prefix="/api/v1")

    # ── Root route ────────────────────────────────────────────────────────────
    from flask import render_template

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Global error handlers ──────────────────────────────────────────────────
    # These catch errors that slip past the route handler — e.g. Werkzeug raising
    # 413 before our validation code even runs if the body is gigantic.

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        # Werkzeug raises 413 when MAX_CONTENT_LENGTH is exceeded.
        # We translate it to our standard validation error shape.
        return (
            jsonify(
                error=True,
                message="CV file must be under 10MB",
                code="VALIDATION_ERROR",
            ),
            413,
        )

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return (
            jsonify(
                error=True,
                message="Method not allowed",
                code="VALIDATION_ERROR",
            ),
            405,
        )

    @app.errorhandler(500)
    def internal_server_error(_error):
        return (
            jsonify(
                error=True,
                message="An unexpected error occurred",
                code="PIPELINE_ERROR",
            ),
            500,
        )

    return app


if __name__ == "__main__":
    # Quick dev-server entry point.
    # In production (or for proper test coverage) use `create_app()` directly.
    create_app().run(debug=False)
