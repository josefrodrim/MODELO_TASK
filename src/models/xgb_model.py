"""
Modelo XGBoost para regresión de riesgo crediticio.

XGBoost complementa LightGBM: diferente implementación de gradient boosting,
regularización L1/L2 explícita, y útil como componente de ensemble.
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective_xgb(trial, X: np.ndarray, y: np.ndarray, cv: int = 3) -> float:
    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 1.9),
        "eval_metric": "rmse",
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)
    return -scores.mean()


def tune_xgb(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30, cv: int = 3) -> dict:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _objective_xgb(trial, X_train.values, y_train.values, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nXGB - Mejor RMSE CV: {study.best_value:.4f}")
    return study.best_params


class XGBRiskModel:
    def __init__(self, best_params: dict = None, log_target: bool = False):
        self.best_params = best_params
        self.log_target = log_target
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose: bool = True):
        self.feature_names_ = X.columns.tolist()
        y_fit = np.log(y) if self.log_target else y.values

        default_params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": 1.5,
            "eval_metric": "rmse",
            "verbosity": 0,
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 50,
        }

        params = {**default_params, **(self.best_params or {})}
        if "early_stopping_rounds" not in params:
            params["early_stopping_rounds"] = 50

        self.model_ = xgb.XGBRegressor(**params)

        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_fit = np.log(y_val) if self.log_target else y_val.values
            self.model_.fit(X, y_fit, eval_set=[(X_val, y_val_fit)], verbose=False)
        else:
            params_no_early = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
            self.model_ = xgb.XGBRegressor(**params_no_early)
            self.model_.fit(X, y_fit)

        if verbose:
            n_trees = self.model_.best_iteration if hasattr(self.model_, "best_iteration") and self.model_.best_iteration else params.get("n_estimators", "?")
            print(f"XGBoost entrenado | Features: {len(self.feature_names_)} | Arboles: {n_trees}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model_.predict(X[self.feature_names_])
        return np.exp(raw) if self.log_target else raw

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        scores = self.model_.get_booster().get_score(importance_type=importance_type)
        fi = pd.DataFrame(list(scores.items()), columns=["feature", "importance"])
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
        return fi


def save_xgb_model(model: XGBRiskModel, path: str):
    joblib.dump(model, path)


def load_xgb_model(path: str) -> XGBRiskModel:
    return joblib.load(path)
