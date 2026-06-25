"""
Modelo CatBoost para regresión de riesgo crediticio.

CatBoost: ordenamiento de árboles simétricos, robusto a overfitting con pocos datos,
excelente para variables categóricas. Complementa LightGBM en el ensemble.
"""

import warnings
import numpy as np
import pandas as pd
import optuna
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective_cat(trial, X: np.ndarray, y: np.ndarray, cv: int = 3) -> float:
    params = {
        "loss_function": "Tweedie:variance_power=1.5",
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "random_seed": 42,
        "verbose": 0,
    }
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, val_idx in kf.split(X):
        model = CatBoostRegressor(**params)
        model.fit(X[tr_idx], y[tr_idx], verbose=0)
        pred = model.predict(X[val_idx])
        rmses.append(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
    return float(np.mean(rmses))


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20, cv: int = 3) -> dict:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: _objective_cat(trial, X_train.values, y_train.values, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nCatBoost - Mejor RMSE CV: {study.best_value:.4f}")
    return study.best_params


class CatBoostRiskModel:
    def __init__(self, best_params: dict = None, log_target: bool = False):
        self.best_params = best_params
        self.log_target = log_target
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose: bool = True):
        self.feature_names_ = X.columns.tolist()
        y_fit = np.log(y.values) if self.log_target else y.values

        default_params = {
            "loss_function": "Tweedie:variance_power=1.5",
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50,
        }

        params = {**default_params, **(self.best_params or {})}
        self.model_ = CatBoostRegressor(**params)

        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_fit = np.log(y_val.values) if self.log_target else y_val.values
            eval_pool = Pool(X_val, y_val_fit)
            self.model_.fit(X.values, y_fit, eval_set=eval_pool, verbose=0)
        else:
            self.model_.fit(X.values, y_fit, verbose=0)

        if verbose:
            print(f"CatBoost entrenado | Features: {len(self.feature_names_)} | "
                  f"Iteraciones: {self.model_.tree_count_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model_.predict(X[self.feature_names_].values)
        return np.exp(raw) if self.log_target else raw

    def get_feature_importance(self) -> pd.DataFrame:
        fi = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model_.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(2)
        return fi


def save_catboost_model(model: CatBoostRiskModel, path: str):
    joblib.dump(model, path)


def load_catboost_model(path: str) -> CatBoostRiskModel:
    return joblib.load(path)
