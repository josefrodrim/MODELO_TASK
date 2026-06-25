"""
Tests de integración para la API FastAPI (LightGBM ONNX).
Usan TestClient de Starlette sin arrancar un servidor real.
Los tests mockean el predictor para no depender de artefactos en disco.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

with patch("src.api.predictor.ModelPredictor.load_models", return_value=None):
    from fastapi.testclient import TestClient
    from src.api.app import app, predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ready_predictor(monkeypatch):
    """Predictor con session ONNX y preprocessor mockeados."""
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.side_effect = lambda df: pd.DataFrame(
        np.zeros((len(df), 47)), columns=[f"f{i}" for i in range(47)]
    )

    mock_input = MagicMock()
    mock_input.shape = [None, 47]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.side_effect = lambda out, inputs: [
        np.full((len(list(inputs.values())[0]), 1), 8500.0, dtype=np.float32)
    ]

    monkeypatch.setattr(predictor, "preprocessor", mock_preprocessor)
    monkeypatch.setattr(predictor, "session", mock_session)
    monkeypatch.setattr(predictor, "input_name", "float_input")
    return predictor


@pytest.fixture
def no_models_predictor(monkeypatch):
    """Predictor sin modelos (simula arranque sin artefactos)."""
    monkeypatch.setattr(predictor, "session", None)
    monkeypatch.setattr(predictor, "preprocessor", None)
    return predictor


VALID_PAYLOAD = {
    "X1": 207137.0, "X2": 6427.0, "X3": None, "X4": None,
    "X5": 3222.0, "X6": 1.0, "X7": 633.0, "X8": 1.0,
    "X9": "SANTIAGO DE SURCO", "X10": 30170.0, "X11": 146.0, "X12": 47.0,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_ready" in data

    def test_health_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_model_info_returns_200(self, client, ready_predictor):
        assert client.get("/model/info").status_code == 200

    def test_model_info_schema(self, client, ready_predictor):
        data = client.get("/model/info").json()
        assert {"model", "n_features", "features", "version"}.issubset(data.keys())

    def test_model_info_version(self, client, ready_predictor):
        assert client.get("/model/info").json()["version"] == "1.0.0"

    def test_model_info_n_features(self, client, ready_predictor):
        assert client.get("/model/info").json()["n_features"] == 47

    def test_model_info_503_when_no_models(self, client, no_models_predictor):
        assert client.get("/model/info").status_code == 503


# ---------------------------------------------------------------------------
# /predict — sin modelos (503)
# ---------------------------------------------------------------------------

class TestPredictNoModels:
    def test_predict_503_when_no_models(self, client, no_models_predictor):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503

    def test_predict_503_has_detail(self, client, no_models_predictor):
        assert "detail" in client.post("/predict", json=VALID_PAYLOAD).json()


# ---------------------------------------------------------------------------
# /predict — con modelos mockeados
# ---------------------------------------------------------------------------

class TestPredictWithModels:
    def test_predict_returns_200(self, client, ready_predictor):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_predict_has_prediction(self, client, ready_predictor):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "prediction" in data
        assert data["prediction"] == pytest.approx(8500.0, abs=1)

    def test_predict_has_request_id(self, client, ready_predictor):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "request_id" in data
        assert len(data["request_id"]) > 0

    def test_predict_sentinel_converts_to_none(self, client, ready_predictor):
        payload = {**VALID_PAYLOAD, "X3": -9999998, "X4": -9999998}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_partial_payload_accepted(self, client, ready_predictor):
        assert client.post("/predict", json={"X6": 1.0, "X10": 30000.0}).status_code == 200

    def test_predict_empty_payload_accepted(self, client, ready_predictor):
        assert client.post("/predict", json={}).status_code == 200


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_returns_200(self, client, ready_predictor):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        assert client.post("/predict/batch", json=payload).status_code == 200

    def test_batch_n_records_matches_input(self, client, ready_predictor):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        data = client.post("/predict/batch", json=payload).json()
        assert data["n_records"] == 2

    def test_batch_predictions_list(self, client, ready_predictor):
        payload = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
        data = client.post("/predict/batch", json=payload).json()
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2

    def test_batch_empty_records_rejected(self, client, ready_predictor):
        assert client.post("/predict/batch", json={"records": []}).status_code == 422

    def test_batch_503_when_no_models(self, client, no_models_predictor):
        assert client.post("/predict/batch", json={"records": [VALID_PAYLOAD]}).status_code == 503


# ---------------------------------------------------------------------------
# Rutas inexistentes
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_unknown_route_returns_404(self, client):
        assert client.get("/nonexistent").status_code == 404
