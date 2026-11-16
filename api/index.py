"""
Vercel serverless entry point.

Vercel's Python runtime looks for a module-level WSGI object named `app`
in files under api/. We reuse the exact same Flask factory that run.py
uses locally — no Waitress needed, because Vercel itself is the server.

Two platform quirks are handled here:
  1. The function bundle isn't guaranteed to run with the repo root as the
     working directory, so MODEL_PATH is resolved to an absolute path
     relative to this file instead of relying on cwd.
  2. The repo root must be on sys.path for `import src.*` to work, since
     this file lives one level down in api/.

Everything else (config YAML, niche benchmarks, templates, static files)
already resolves via __file__ inside the src modules, so no changes there.
"""
import os
import pathlib
import sys
from urllib.parse import parse_qsl, urlencode

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault(
    "MODEL_PATH", str(_ROOT / "models" / "20260423_155216_seed42.joblib")
)

from src.app import create_app


class _RestoreOriginalPath:
    """
    Undo Vercel's rewrite before Flask routes the request.

    vercel.json rewrites every URL to /api/index, and the Python runtime
    hands Flask that rewritten path — so every route would 404. The rewrite
    smuggles the real path through as a `__vercel_path` query parameter;
    this middleware moves it back into PATH_INFO and strips the parameter
    so request.args stays clean. Requests without the parameter (local
    testing, direct invocation) pass through untouched.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query = environ.get("QUERY_STRING", "")
        if "__vercel_path" in query:
            original_path = None
            kept_params = []
            for key, value in parse_qsl(query, keep_blank_values=True):
                if key == "__vercel_path" and original_path is None:
                    original_path = value
                else:
                    kept_params.append((key, value))
            environ["PATH_INFO"] = "/" + (original_path or "").lstrip("/")
            environ["QUERY_STRING"] = urlencode(kept_params)
        return self.wsgi_app(environ, start_response)


# Module-level so Vercel's runtime discovers it. The model artifact and all
# heavy ML imports load once per serverless instance (cold start), then every
# request on that instance reuses them — same behaviour as the local server.
app = create_app()
app.wsgi_app = _RestoreOriginalPath(app.wsgi_app)
