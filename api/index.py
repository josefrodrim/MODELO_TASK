"""
Vercel entrypoint — envuelve el app FastAPI de src/api/app.py.

Vercel busca este archivo en api/index.py y lo sirve como función serverless.
Carga los modelos de forma eager para minimizar latencia en warm starts.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("ML_MODEL_PATH", os.path.join(ROOT, "models", "lgbm_model.pkl"))
os.environ.setdefault("PREPROCESSOR_PATH", os.path.join(ROOT, "models", "preprocessor.pkl"))
# GLM no se despliega en Vercel (37 MB) — se usará solo LGBM
os.environ.setdefault("GLM_MODEL_PATH", os.path.join(ROOT, "models", "glm_model.pkl"))

from src.api.app import app, predictor  # noqa: E402

if not predictor.is_ready:
    predictor.load_models()
