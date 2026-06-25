"""
API REST de scoring crediticio — LightGBM via ONNX Runtime.

Endpoints:
  GET  /health           → Estado del servicio
  GET  /model/info       → Metadata del modelo cargado
  POST /predict          → Predicción individual
  POST /predict/batch    → Predicción en lote (hasta 10K registros)

Uso local:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from src.api.predictor import ModelPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "models/lgbm_model.onnx")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
API_VERSION = "1.0.0"

predictor = ModelPredictor(ONNX_MODEL_PATH, PREPROCESSOR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo ONNX...")
    predictor.load_models()
    if predictor.is_ready:
        logger.info("Modelo listo.")
    else:
        logger.warning("API arrancó sin modelo.")
    yield


app = FastAPI(
    title="API de Scoring de Riesgo Crediticio",
    description=(
        "Modelo LightGBM (ONNX) para estimación de score de riesgo crediticio. "
        "Desarrollado para Scotiabank Perú — caso práctico Risk Data Scientist."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    return HealthResponse(status="ok", model_ready=predictor.is_ready)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Sistema"])
def model_info():
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelo no disponible.")
    n_features = predictor.session.get_inputs()[0].shape[1]
    return ModelInfoResponse(
        model="LightGBM (ONNX)",
        n_features=n_features,
        features=predictor.preprocessor.get_feature_names() if hasattr(predictor.preprocessor, 'get_feature_names') else [],
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict_single(request: PredictionRequest):
    """Predicción individual. Acepta X1–X12 y devuelve el score crediticio predicho."""
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelo no disponible.")
    pred = predictor.predict_single(request.model_dump())
    return PredictionResponse(prediction=pred, request_id=str(uuid.uuid4()))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
def predict_batch(request: BatchPredictionRequest):
    """Predicción en lote — hasta 10,000 registros por llamada."""
    if not predictor.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelo no disponible.")
    records = [r.model_dump() for r in request.records]
    preds = predictor.predict_batch(records)
    return BatchPredictionResponse(predictions=preds, n_records=len(preds))
