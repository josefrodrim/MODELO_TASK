"""
Vercel entrypoint — envuelve el app FastAPI de src/api/app.py.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("ML_MODEL_PATH", os.path.join(ROOT, "models", "lgbm_model.pkl"))
os.environ.setdefault("PREPROCESSOR_PATH", os.path.join(ROOT, "models", "preprocessor.pkl"))

from src.api.app import app, predictor  # noqa: E402

if not predictor.is_ready:
    predictor.load_models()
