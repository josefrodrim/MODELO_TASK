"""
Singleton de predicción: carga el modelo LightGBM (ONNX) y expone métodos de inferencia.
ONNX runtime no requiere libgomp, lo que permite despliegue en Vercel/Lambda.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, onnx_path: str, preprocessor_path: str):
        self.onnx_path = Path(onnx_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.session = None       # onnxruntime.InferenceSession
        self.preprocessor = None
        self.feature_names = None

    def load_models(self):
        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Preprocessor cargado")
        else:
            logger.warning(f"Preprocessor no encontrado: {self.preprocessor_path}")

        if self.onnx_path.exists():
            import onnxruntime as rt
            self.session = rt.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.feature_names = [i.name for i in self.session.get_inputs()[0].shape
                                  if isinstance(i, str)] or None
            logger.info("Modelo ONNX cargado")
        else:
            logger.warning(f"Modelo ONNX no encontrado: {self.onnx_path}")

    @property
    def is_ready(self) -> bool:
        return self.session is not None and self.preprocessor is not None

    def _preprocess(self, records: list) -> np.ndarray:
        df = pd.DataFrame(records)
        df = self.preprocessor.transform(df)
        return df.values.astype(np.float32)

    def predict_single(self, record: dict) -> float:
        X = self._preprocess([record])
        pred = self.session.run(None, {self.input_name: X})[0]
        return round(float(pred.flatten()[0]), 2)

    def predict_batch(self, records: list) -> list:
        X = self._preprocess(records)
        preds = self.session.run(None, {self.input_name: X})[0].flatten()
        return [round(float(p), 2) for p in preds]
