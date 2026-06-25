"""
Tests de integración para la API FastAPI.

Usan TestClient de Starlette (no hace falta arrancar un servidor real).
Los tests de predicción mockean el predictor para no depender de artefactos
entrenados, lo que permite correr la suite en CI sin necesidad de modelos.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Parchear la carga de modelos ANTES de importar la app para que el predictor
# no intente leer archivos .pkl que no existen en CI.
with patch("src.api.predictor.ModelPredictor.load_models", return_value=None):
    from fastapi.testclient import TestClient
    from src.api.app import app, predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ready_predictor(monkeypatch):
    """Prepara el predictor con modelos mockeados para tests de predicción."""
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.side_effect = lambda df: pd.DataFrame(
        np.zeros((len(df), 5)), columns=["X1", "X2", "X3", "X4", "X5"]
    )

    mock_lgbm = MagicMock()
    mock_lgbm.predict.side_effect = lambda df: np.full(len(df), 8500.0)
    mock_lgbm.feature_names_ = ["X1", "X2", "X3"]

    mock_glm = MagicMock()
    mock_glm.predict.side_effect = lambda df: np.full(len(df), 8200.0)
    mock_glm.selected_features_ = ["X1", "X2"]

    monkeypatch.setattr(predictor, "preprocessor", mock_preprocessor)
    monkeypatch.setattr(predictor, "lgbm_model", mock_lgbm)
    monkeypatch.setattr(predictor, "glm_model", mock_glm)
    return predictor


@pytest.fixture
def no_models_predictor(monkeypatch):
    """Predictor sin modelos cargados (simula primer arranque sin artefactos)."""
    monkeypatch.setattr(predictor, "preprocessor", None)
    monkeypatch.setattr(predictor, "lgbm_model", None)
    monkeypatch.setattr(predictor, "glm_model", None)
    return predictor


VALID_PAYLOAD = {
    "X1": 207137.0,
    "X2": 6427.0,
    "X3": None,
    "X4": None,
    "X5": 3222.0,
    "X6": 1.0,
    "X7": 633.0,
    "X8": 1.0,
    "X9": "SANTIAGO DE SURCO",
    "X10": 30170.0,
    "X11": 146.0,
    "X12": 47.0,
    "model": "lgbm",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_ready" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get("/model/info").json()
        required = {"glm_loaded", "lgbm_loaded", "version"}
        assert required.issubset(data.keys())

    def test_model_info_version(self, client):
        data = client.get("/model/info").json()
        assert data["version"] == "1.0.0"

    def test_model_info_with_loaded_models(self, client, ready_predictor):
        data = client.get("/model/info").json()
        assert data["glm_loaded"] is True
        assert data["lgbm_loaded"] is True
        assert data["glm_features"] is not None
        assert data["lgbm_features"] is not None


# ---------------------------------------------------------------------------
# /predict — sin modelos (503)
# ---------------------------------------------------------------------------

class TestPredictNoModels:
    def test_predict_503_when_no_models(self, client, no_models_predictor):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503

    def test_predict_503_body_has_detail(self, client, no_models_predictor):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "detail" in data


# ---------------------------------------------------------------------------
# /predict — con modelos mockeados
# ---------------------------------------------------------------------------

class TestPredictWithModels:
    def test_predict_lgbm_returns_200(self, client, ready_predictor):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_lgbm_has_prediction(self, client, ready_predictor):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "prediction_lgbm" in data
        assert data["prediction_lgbm"] == pytest.approx(8500.0, abs=1)

    def test_predict_glm_model(self, client, ready_predictor):
        payload = {**VALID_PAYLOAD, "model": "glm"}
        data = client.post("/predict", json=payload).json()
        assert data["prediction_glm"] == pytest.approx(8200.0, abs=1)

    def test_predict_both_models(self, client, ready_predictor):
        # Para 'both', el mock lgbm devuelve [8500], y el glm [8200]
        mock_lgbm = predictor.lgbm_model
        mock_lgbm.predict.return_value = np.array([8500.0])

        payload = {**VALID_PAYLOAD, "model": "both"}
        data = client.post("/predict", json=payload).json()
        assert "model_used" in data
        assert data["model_used"] == "both"

    def test_predict_response_has_request_id(self, client, ready_predictor):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0

    def test_predict_with_sentinel_value_converts_to_none(self, client, ready_predictor):
        # El schema convierte -9999998 → None antes de pasar al modelo
        payload = {**VALID_PAYLOAD, "X3": -9999998, "X4": -9999998}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_all_nulls_accepted(self, client, ready_predictor):
        # Todos los campos opcionales pueden ser None (imputer los manejará)
        payload = {"X6": 1.0, "X8": 1.0, "X10": 30000.0, "model": "lgbm"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /predict — validación de payload
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_invalid_model_field_rejected(self, client):
        payload = {**VALID_PAYLOAD, "model": "xgboost"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_payload_accepted(self, client, ready_predictor):
        # Todos los campos son opcionales (Optional[float])
        response = client.post("/predict", json={"model": "lgbm"})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_returns_200(self, client, ready_predictor):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_n_records_matches_input(self, client, ready_predictor):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        data = client.post("/predict/batch", json=payload).json()
        assert data["n_records"] == 2

    def test_batch_empty_records_rejected(self, client, ready_predictor):
        payload = {"records": []}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_503_when_no_models(self, client, no_models_predictor):
        payload = {"records": [VALID_PAYLOAD]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Rutas inexistentes
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_unknown_route_returns_404(self, client):
        assert client.get("/nonexistent").status_code == 404
