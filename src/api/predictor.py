"""
Singleton de predicción: carga modelos una sola vez y expone métodos de inferencia.
Desacopla la lógica de carga de modelos de los endpoints de FastAPI.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Gestiona el ciclo de vida de los artefactos de ML en producción:
      - Carga lazy de modelos al arrancar la API
      - Preprocesamiento centralizado (mismo preprocessor que en entrenamiento)
      - Predicción individual y en lote
    """

    def __init__(self, glm_path: str, lgbm_path: str, preprocessor_path: str):
        self.glm_path = Path(glm_path)
        self.lgbm_path = Path(lgbm_path)
        self.preprocessor_path = Path(preprocessor_path)

        self.glm_model = None
        self.lgbm_model = None
        self.preprocessor = None

    def load_models(self):
        """Carga todos los artefactos desde disco. Llamar una vez al arrancar la API."""
        if self.glm_path.exists():
            self.glm_model = joblib.load(self.glm_path)
            logger.info("GLM cargado correctamente")
        else:
            logger.warning(f"Modelo GLM no encontrado en {self.glm_path}")

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
        return (self.lgbm_model is not None or self.glm_model is not None) and \
               self.preprocessor is not None

    def _preprocess(self, records: list) -> pd.DataFrame:
        """Convierte lista de dicts en DataFrame procesado listo para inferencia."""
        df = pd.DataFrame(records)
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        return df

    def predict_single(self, record: dict, model: str = "lgbm") -> dict:
        """
        Predicción individual.

        Args:
            record: dict con los features del cliente
            model:  'glm', 'lgbm' o 'both'

        Returns:
            dict con predicción(es) en escala original
        """
        df = self._preprocess([record])
        result = {}

        if model in ("glm", "both") and self.glm_model:
            try:
                result["prediction_glm"] = round(float(self.glm_model.predict(df)[0]), 2)
            except Exception as e:
                logger.error(f"Error GLM: {e}")
                result["prediction_glm"] = None

        if model in ("lgbm", "both") and self.lgbm_model:
            try:
                result["prediction_lgbm"] = round(float(self.lgbm_model.predict(df)[0]), 2)
            except Exception as e:
                logger.error(f"Error LGBM: {e}")
                result["prediction_lgbm"] = None

        return result

    def predict_batch(self, records: list, model: str = "lgbm") -> list:
        """Predicción en lote — más eficiente que llamar predict_single en loop."""
        df = self._preprocess(records)
        results = []

        glm_preds = None
        lgbm_preds = None

        if model in ("glm", "both") and self.glm_model:
            try:
                glm_preds = self.glm_model.predict(df)
            except Exception as e:
                logger.error(f"Error GLM batch: {e}")

        if model in ("lgbm", "both") and self.lgbm_model:
            try:
                lgbm_preds = self.lgbm_model.predict(df)
            except Exception as e:
                logger.error(f"Error LGBM batch: {e}")

        for i in range(len(records)):
            entry = {}
            if glm_preds is not None:
                entry["prediction_glm"] = round(float(glm_preds[i]), 2)
            if lgbm_preds is not None:
                entry["prediction_lgbm"] = round(float(lgbm_preds[i]), 2)
            results.append(entry)

        return results
