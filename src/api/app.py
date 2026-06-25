"""
API REST de predicción de riesgo crediticio — FastAPI.

Endpoints:
  GET  /health            → Estado de salud del servicio y modelos
  GET  /model/info        → Metadata de los modelos cargados
  POST /predict           → Predicción individual
  POST /predict/batch     → Predicción en lote (hasta 10K registros)

Para ejecutar localmente:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Para ejecutar con Docker:
  docker-compose up
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

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GLM_MODEL_PATH = os.getenv("GLM_MODEL_PATH", "models/glm_model.pkl")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/lgbm_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
API_VERSION = "1.0.0"

# Singleton global
predictor = ModelPredictor(GLM_MODEL_PATH, ML_MODEL_PATH, PREPROCESSOR_PATH)


# ---------------------------------------------------------------------------
# Lifecycle: carga modelos al arrancar, libera al apagar
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelos de riesgo...")
    predictor.load_models()
    if predictor.is_ready:
        logger.info("Modelos listos para servir predicciones.")
    else:
        logger.warning("API arrancó sin modelos — los endpoints de predicción fallarán.")
    yield
    logger.info("API apagando...")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de Scoring de Riesgo Crediticio",
    description=(
        "Modelos predictivos para estimación de score de riesgo crediticio. "
        "Expone predicciones de un modelo GLM (OLS log-normal) y un modelo "
        "LightGBM con optimización Optuna. Desarrollado para Scotiabank Perú."
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    """Verificación de salud del servicio. Usada por Docker healthcheck y load balancers."""
    return HealthResponse(status="ok", models_ready=predictor.is_ready)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Sistema"])
def model_info():
    """Metadata de los modelos cargados: features, versión, estado."""
    glm_features = None
    lgbm_features = None

    if predictor.glm_model:
        glm_features = predictor.glm_model.selected_features_

    if predictor.lgbm_model:
        lgbm_features = predictor.lgbm_model.feature_names_

    return ModelInfoResponse(
        glm_loaded=predictor.glm_model is not None,
        lgbm_loaded=predictor.lgbm_model is not None,
        glm_features=glm_features,
        lgbm_features=lgbm_features,
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict_single(request: PredictionRequest):
    """
    Predicción individual de score de riesgo.

    Acepta los features X1–X12 de un cliente y devuelve el score predicho
    por el modelo seleccionado (GLM, LightGBM, o ambos).
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos no disponibles. Verifica que los artefactos existen en /models.",
        )

    record = request.model_dump(exclude={"model"})
    preds = predictor.predict_single(record, model=request.model)

    return PredictionResponse(
        prediction_glm=preds.get("prediction_glm"),
        prediction_lgbm=preds.get("prediction_lgbm"),
        model_used=request.model,
        request_id=str(uuid.uuid4()),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
def predict_batch(request: BatchPredictionRequest):
    """
    Predicción en lote (hasta 10,000 registros por llamada).
    Más eficiente que llamar /predict en loop desde el cliente.
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos no disponibles.",
        )

    model = request.records[0].model if request.records else "lgbm"
    records = [r.model_dump(exclude={"model"}) for r in request.records]
    preds = predictor.predict_batch(records, model=model)

    return BatchPredictionResponse(
        predictions=preds,
        n_records=len(preds),
        model_used=model,
    )
