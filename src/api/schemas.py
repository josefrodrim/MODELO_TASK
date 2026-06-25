"""
Esquemas Pydantic para la API de predicción.
Validan y documentan el contrato de entrada/salida de los endpoints.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Payload para una predicción individual."""

    X1: Optional[float] = Field(None, description="Variable numérica X1")
    X2: Optional[float] = Field(None, description="Variable numérica X2")
    X3: Optional[float] = Field(None, description="Variable numérica X3 (puede ser nulo)")
    X4: Optional[float] = Field(None, description="Variable numérica X4 (puede ser nulo)")
    X5: Optional[float] = Field(None, description="Variable numérica X5")
    X6: Optional[float] = Field(None, description="Variable numérica X6")
    X7: Optional[float] = Field(None, description="Variable numérica X7")
    X8: Optional[float] = Field(None, description="Variable numérica X8 (baja cardinalidad)")
    X9: Optional[str] = Field(None, description="Distrito geográfico (categórica)")
    X10: Optional[float] = Field(None, description="Variable numérica X10")
    X11: Optional[float] = Field(None, description="Variable numérica X11")
    X12: Optional[float] = Field(None, description="Variable numérica X12")
    model: Literal["glm", "lgbm", "both"] = Field(
        "lgbm", description="Modelo a usar para la predicción"
    )

    @field_validator("X3", "X4", mode="before")
    @classmethod
    def replace_sentinel(cls, v):
        """Convierte el valor centinela -9999998 a None en la entrada de la API."""
        if v == -9999998:
            return None
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Respuesta de una predicción individual."""

    prediction_glm: Optional[float] = Field(None, description="Score predicho por GLM")
    prediction_lgbm: Optional[float] = Field(None, description="Score predicho por LightGBM")
    model_used: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    """Payload para predicción en lote."""

    records: List[PredictionRequest] = Field(..., min_length=1, max_length=10_000)


class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote."""

    predictions: List[dict]
    n_records: int
    model_used: str


class ModelInfoResponse(BaseModel):
    """Metadata de los modelos cargados."""

    glm_loaded: bool
    lgbm_loaded: bool
    glm_features: Optional[List[str]]
    lgbm_features: Optional[List[str]]
    version: str


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
