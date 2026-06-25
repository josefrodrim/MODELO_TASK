"""
Modelo de Machine Learning (LightGBM) para regresión de riesgo crediticio.

Justificación de LightGBM:
  - Manejo nativo de valores faltantes (no requiere imputación previa).
  - Gradient boosting basado en histogramas → muy eficiente en datasets medianos (50K rows).
  - Alta performance en datos tabulares con variables mixtas.
  - Compatible con SHAP para interpretabilidad regulatoria.
  - Parámetro num_leaves controla capacidad → fácil de regularizar vs XGBoost.
  - Producción: latencia de inferencia < 5ms por registro (lightweight vs redes neuronales).
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import optuna
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Entrenamiento con Optuna
# ---------------------------------------------------------------------------

def _objective(trial, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """Función objetivo de Optuna: RMSE en CV sobre log(TARGET)."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
    }
    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -scores.mean()


def tune_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50, cv: int = 5
) -> dict:
    """
    Búsqueda de hiperparámetros con Optuna.
    Se optimiza sobre log(TARGET) para alinear con el espacio del GLM.
    """
    y_log = np.log(y_train)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _objective(trial, X_train.values, y_log, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\nMejor RMSE (log-escala) en CV: {study.best_value:.4f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Clase principal del modelo ML
# ---------------------------------------------------------------------------

class LGBMRiskModel:
    """
    Wrapper de LightGBM para el pipeline de riesgo.

    Atributos tras fit():
      - model_:            LGBMRegressor ajustado
      - best_params_:      hiperparámetros óptimos de Optuna
      - feature_names_:    lista de features usadas
      - shap_explainer_:   TreeExplainer de SHAP (lazy-init)
    """

    def __init__(self, best_params: dict = None, log_target: bool = True):
        """
        Args:
            best_params: parámetros de Optuna; si None, usa defaults razonables.
            log_target:  si True, entrena sobre log(TARGET) y predice con exp().
        """
        self.best_params = best_params
        self.log_target = log_target
        self.model_ = None
        self.best_params_ = None
        self.feature_names_ = None
        self.shap_explainer_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose: bool = True):
        self.feature_names_ = X.columns.tolist()
        y_fit = np.log(y) if self.log_target else y.values

        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "random_state": 42,
        }

        params = {**default_params, **(self.best_params or {})}
        self.best_params_ = params

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_fit = np.log(y_val) if self.log_target else y_val.values
            self.model_ = lgb.LGBMRegressor(**params)
            self.model_.fit(
                X, y_fit,
                eval_set=[(X_val, y_val_fit)],
                callbacks=callbacks,
            )
        else:
            self.model_ = lgb.LGBMRegressor(**params)
            self.model_.fit(X, y_fit)

        if verbose:
            print(f"LightGBM entrenado | Features: {len(self.feature_names_)} | "
                  f"Árboles: {self.model_.n_estimators_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción en escala original."""
        raw = self.model_.predict(X[self.feature_names_])
        return np.exp(raw) if self.log_target else raw

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[self.feature_names_])

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Retorna importancia de features por tipo ('gain' recomendado: mide reducción de loss).
        'split' tiende a favorecer variables de alta cardinalidad.
        """
        fi = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model_.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
        return fi.reset_index(drop=True)

    def get_shap_values(self, X: pd.DataFrame, sample_size: int = 2000) -> tuple:
        """
        Calcula SHAP values usando TreeExplainer.
        Usa muestra si el dataset es grande para eficiencia.

        Returns:
            (shap_values, X_sample) para uso directo en shap.plots
        """
        if self.shap_explainer_ is None:
            self.shap_explainer_ = shap.TreeExplainer(self.model_)

        X_sample = X[self.feature_names_].sample(
            min(sample_size, len(X)), random_state=42
        ).reset_index(drop=True)

        shap_values = self.shap_explainer_.shap_values(X_sample)
        return shap_values, X_sample

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """CV para estimar varianza del modelo y detectar overfitting temprano."""
        y_fit = np.log(y) if self.log_target else y.values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model_, X, y_fit, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        return {
            "cv_rmse_mean": round(-scores.mean(), 4),
            "cv_rmse_std": round(scores.std(), 4),
            "cv_folds": n_splits,
        }


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_lgbm_model(model: LGBMRiskModel, path: str):
    joblib.dump(model, path)


def load_lgbm_model(path: str) -> LGBMRiskModel:
    return joblib.load(path)
