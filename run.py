"""Standalone launcher — start Flask without -m so __name__ stays as 'src.app'."""
import os
os.environ.setdefault("MODEL_PATH", "models/20260423_155216_seed42.joblib")
from src.app import create_app

app = create_app()

if __name__ == "__main__":
    from waitress import serve
    # Cloud platforms (Render, Railway, etc.) inject PORT and expect 0.0.0.0.
    # Locally we default to 127.0.0.1:5002 so the app isn't exposed on the LAN.
    port = int(os.environ.get("PORT", 5002))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    serve(app, host=host, port=port, threads=4)
