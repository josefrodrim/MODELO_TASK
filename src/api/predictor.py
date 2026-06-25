"""
Singleton de predicción: carga el modelo LightGBM una sola vez y expone métodos de inferencia.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Gestiona el ciclo de vida del modelo LightGBM en producción:
      - Carga lazy desde disco al arrancar la API
      - Preprocesamiento centralizado (mismo preprocessor que en entrenamiento)
      - Predicción individual y en lote
    """

    def __init__(self, lgbm_path: str, preprocessor_path: str):
        self.lgbm_path = Path(lgbm_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.lgbm_model = None
        self.preprocessor = None

    def load_models(self):
        """Carga los artefactos desde disco. Llamar una vez al arrancar la API."""
        if self.lgbm_path.exists():
            self.lgbm_model = joblib.load(self.lgbm_path)
            logger.info("LightGBM cargado correctamente")
        else:
            logger.warning(f"Modelo LGBM no encontrado en {self.lgbm_path}")

        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Preprocessor cargado correctamente")
        else:
            logger.warning(f"Preprocessor no encontrado en {self.preprocessor_path}")

    @property
    def is_ready(self) -> bool:
        return self.lgbm_model is not None and self.preprocessor is not None

    def _preprocess(self, records: list) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return self.preprocessor.transform(df)

    def predict_single(self, record: dict) -> float:
        df = self._preprocess([record])
        return round(float(self.lgbm_model.predict(df)[0]), 2)

    def predict_batch(self, records: list) -> list:
        df = self._preprocess(records)
        preds = self.lgbm_model.predict(df)
        return [round(float(p), 2) for p in preds]
