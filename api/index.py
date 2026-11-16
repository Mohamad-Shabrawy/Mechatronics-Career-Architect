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

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault(
    "MODEL_PATH", str(_ROOT / "models" / "20260423_155216_seed42.joblib")
)

from src.app import create_app

# Module-level so Vercel's runtime discovers it. The model artifact and all
# heavy ML imports load once per serverless instance (cold start), then every
# request on that instance reuses them — same behaviour as the local server.
app = create_app()
