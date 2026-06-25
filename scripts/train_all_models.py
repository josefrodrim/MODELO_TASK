"""
Script unificado: entrena GLM + LightGBM + XGBoost + CatBoost + Ensemble.
Genera métricas comparativas en reports/metrics/.

Uso:
    python3 scripts/train_all_models.py
    python3 scripts/train_all_models.py --fast   # menos trials Optuna
"""

import sys
import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import load_raw_data, prepare_train_test, RiskPreprocessor
from src.models.glm_model import GLMRiskModel, save_glm_model
from src.models.ml_model import LGBMRiskModel, tune_lgbm, save_lgbm_model
from src.models.xgb_model import XGBRiskModel, tune_xgb, save_xgb_model
from src.models.catboost_model import CatBoostRiskModel, tune_catboost, save_catboost_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "data/raw/data_modelo.csv"
MODELS_DIR  = "models"
METRICS_DIR = "reports/metrics"
RANDOM_SEED = 42


def ks_stat(y_true, y_pred):
    n = len(y_true)
    idx = np.argsort(y_pred)[::-1]
    y_s = y_true[idx]
    cum_event = np.cumsum(y_s) / y_s.sum()
    cum_non   = np.cumsum(1 - y_s) / (1 - y_s).sum()
    # Para regresión: binarizar por mediana
    threshold = np.median(y_true)
    y_bin = (y_true > threshold).astype(int)
    idx   = np.argsort(y_pred)[::-1]
    y_b   = y_bin[idx]
    cum_ev = np.cumsum(y_b) / max(y_b.sum(), 1)
    cum_nev= np.cumsum(1 - y_b) / max((1 - y_b).sum(), 1)
    return float(np.max(np.abs(cum_ev - cum_nev)))


def gini_coeff(y_true, y_pred):
    n = len(y_true)
    idx = np.argsort(y_pred)
    y_s = y_true[idx]
    cum_y = np.cumsum(y_s) / y_s.sum()
    cum_pop = np.arange(1, n + 1) / n
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(1 - 2 * trapz(cum_y, cum_pop))


def compute_metrics(y_true, y_pred, label=""):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))
    r2   = float(r2_score(y_t, y_p))
    mape = float(np.mean(np.abs((y_t - y_p) / y_t)) * 100)
    ks   = ks_stat(y_t, y_p)
    gini = gini_coeff(y_t, y_p)
    return {"modelo": label, "R2": round(r2, 4), "RMSE": round(rmse, 0),
            "MAE": round(mae, 0), "MAPE": round(mape, 2),
            "KS": round(ks, 4), "Gini": round(gini, 4)}


def decile_analysis(y_true, y_pred, label=""):
    df = pd.DataFrame({"y": np.array(y_true), "yhat": np.array(y_pred)})
    df["decil"] = pd.qcut(df["yhat"], 10, labels=False, duplicates="drop") + 1
    tbl = df.groupby("decil").agg(
        n=("y", "count"),
        actual_mean=("y", "mean"),
        pred_mean=("yhat", "mean"),
    ).reset_index()
    tbl["sesgo"] = (tbl["pred_mean"] - tbl["actual_mean"]).round(0)
    tbl["modelo"] = label
    return tbl


# ---------------------------------------------------------------------------
# Ensemble: pesos óptimos minimizando RMSE en validación
# ---------------------------------------------------------------------------

def optimize_ensemble(predictions_val: dict, y_val: np.ndarray) -> dict:
    """Encuentra pesos óptimos para el ensemble minimizando RMSE en validación."""
    model_names = list(predictions_val.keys())
    n = len(model_names)

    def rmse_ensemble(weights):
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        blend = sum(w * predictions_val[m] for m, w in zip(model_names, weights))
        return float(np.sqrt(mean_squared_error(y_val, blend)))

    result = minimize(
        rmse_ensemble,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    weights_raw = np.maximum(result.x, 0)
    weights_norm = weights_raw / weights_raw.sum()
    return {m: round(float(w), 4) for m, w in zip(model_names, weights_norm)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    fast = args.fast
    n_trials_lgbm = 15 if fast else 30
    n_trials_xgb  = 10 if fast else 25
    n_trials_cat  = 10 if fast else 20
    cv_folds       = 3

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("=" * 60)
    print("CARGANDO Y PREPROCESANDO DATOS")
    print("=" * 60)
    raw = load_raw_data(DATA_PATH)

    # Separar train/test manualmente para controlar preprocessing
    cols_drop = ["ID", "BASE", "TARGET"]
    train_raw = raw[raw["BASE"] == "TRAIN"].reset_index(drop=True)
    test_raw  = raw[raw["BASE"] == "TEST"].reset_index(drop=True)
    X_train_raw = train_raw.drop(columns=cols_drop, errors="ignore")
    y_train     = train_raw["TARGET"]
    X_test_raw  = test_raw.drop(columns=cols_drop, errors="ignore")
    y_test      = test_raw["TARGET"]

    # Split val para ensemble (20% del train, sin tocar test — evita leakage)
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=RANDOM_SEED
    )

    # Preprocesador para Optuna CV (fit solo en X_tr para no contaminar val)
    pre_tr = RiskPreprocessor()
    pre_tr.fit(X_tr_raw, y_tr)
    Xtr_p  = pre_tr.transform(X_tr_raw)
    Xval_p = pre_tr.transform(X_val_raw)

    # Preprocesador final (fit en todo train para entrenamiento definitivo)
    preprocessor = RiskPreprocessor()
    preprocessor.fit(X_train_raw, y_train)
    Xfull_p = preprocessor.transform(X_train_raw)
    Xtest_p = preprocessor.transform(X_test_raw)

    print(f"Train: {Xfull_p.shape} | Test: {Xtest_p.shape}")
    print(f"Val (ensemble): {Xval_p.shape}")

    results_train = []
    results_test  = []
    val_preds     = {}
    test_preds    = {}

    # ------------------------------------------------------------------
    # 1. GLM
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GLM (OLS log-normal)")
    print("=" * 60)
    t0 = time.time()
    glm = GLMRiskModel()
    glm.fit(Xfull_p, y_train, verbose=True)
    print(f"GLM entrenado en {time.time()-t0:.1f}s")

    pred_glm_tr = glm.predict(Xfull_p)
    pred_glm_te = glm.predict(Xtest_p)
    pred_glm_val = glm.predict(Xval_p)

    results_train.append(compute_metrics(y_train, pred_glm_tr, "GLM"))
    results_test.append(compute_metrics(y_test, pred_glm_te, "GLM"))
    val_preds["GLM"]  = pred_glm_val
    test_preds["GLM"] = pred_glm_te
    save_glm_model(glm, f"{MODELS_DIR}/glm_model.pkl")

    # ------------------------------------------------------------------
    # 2. LightGBM
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"LightGBM (Optuna {n_trials_lgbm} trials, {cv_folds}-fold CV)")
    print("=" * 60)
    t0 = time.time()
    best_lgbm = tune_lgbm(Xtr_p, y_tr, n_trials=n_trials_lgbm, cv=cv_folds)
    lgbm = LGBMRiskModel(best_params=best_lgbm)
    lgbm.fit(Xfull_p, y_train, eval_set=(Xtest_p, y_test))
    print(f"LGBM entrenado en {time.time()-t0:.1f}s")

    pred_lgbm_tr  = lgbm.predict(Xfull_p)
    pred_lgbm_te  = lgbm.predict(Xtest_p)
    pred_lgbm_val = lgbm.predict(Xval_p)

    results_train.append(compute_metrics(y_train, pred_lgbm_tr, "LightGBM"))
    results_test.append(compute_metrics(y_test, pred_lgbm_te, "LightGBM"))
    val_preds["LightGBM"]  = pred_lgbm_val
    test_preds["LightGBM"] = pred_lgbm_te
    save_lgbm_model(lgbm, f"{MODELS_DIR}/lgbm_model.pkl")

    # ------------------------------------------------------------------
    # 3. XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"XGBoost (Optuna {n_trials_xgb} trials, {cv_folds}-fold CV)")
    print("=" * 60)
    t0 = time.time()
    best_xgb = tune_xgb(Xtr_p, y_tr, n_trials=n_trials_xgb, cv=cv_folds)
    xgb_model = XGBRiskModel(best_params=best_xgb)
    xgb_model.fit(Xfull_p, y_train, eval_set=(Xtest_p, y_test))
    print(f"XGB entrenado en {time.time()-t0:.1f}s")

    pred_xgb_tr  = xgb_model.predict(Xfull_p)
    pred_xgb_te  = xgb_model.predict(Xtest_p)
    pred_xgb_val = xgb_model.predict(Xval_p)

    results_train.append(compute_metrics(y_train, pred_xgb_tr, "XGBoost"))
    results_test.append(compute_metrics(y_test, pred_xgb_te, "XGBoost"))
    val_preds["XGBoost"]  = pred_xgb_val
    test_preds["XGBoost"] = pred_xgb_te
    save_xgb_model(xgb_model, f"{MODELS_DIR}/xgb_model.pkl")

    # ------------------------------------------------------------------
    # 4. CatBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CatBoost (Optuna {n_trials_cat} trials, {cv_folds}-fold CV)")
    print("=" * 60)
    t0 = time.time()
    best_cat = tune_catboost(Xtr_p, y_tr, n_trials=n_trials_cat, cv=cv_folds)
    cat_model = CatBoostRiskModel(best_params=best_cat)
    cat_model.fit(Xfull_p, y_train, eval_set=(Xtest_p, y_test))
    print(f"CatBoost entrenado en {time.time()-t0:.1f}s")

    pred_cat_tr  = cat_model.predict(Xfull_p)
    pred_cat_te  = cat_model.predict(Xtest_p)
    pred_cat_val = cat_model.predict(Xval_p)

    results_train.append(compute_metrics(y_train, pred_cat_tr, "CatBoost"))
    results_test.append(compute_metrics(y_test, pred_cat_te, "CatBoost"))
    val_preds["CatBoost"]  = pred_cat_val
    test_preds["CatBoost"] = pred_cat_te
    save_catboost_model(cat_model, f"{MODELS_DIR}/catboost_model.pkl")

    # ------------------------------------------------------------------
    # 5. Ensemble optimo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ENSEMBLE (pesos optimizados en validacion)")
    print("=" * 60)

    # Solo incluir ML models en ensemble (GLM tiene escala diferente — incluirlo igual)
    weights = optimize_ensemble(val_preds, np.array(y_val))
    print("Pesos optimos encontrados:")
    for model, w in weights.items():
        print(f"  {model}: {w:.3f}")

    pred_ens_tr  = sum(w * (lgbm.predict(Xfull_p) if m == "LightGBM"
                            else xgb_model.predict(Xfull_p) if m == "XGBoost"
                            else cat_model.predict(Xfull_p) if m == "CatBoost"
                            else glm.predict(Xfull_p))
                       for m, w in weights.items())
    pred_ens_te  = sum(w * test_preds[m] for m, w in weights.items())

    results_train.append(compute_metrics(y_train, pred_ens_tr, "Ensemble"))
    results_test.append(compute_metrics(y_test, pred_ens_te, "Ensemble"))
    test_preds["Ensemble"] = pred_ens_te

    # ------------------------------------------------------------------
    # Guardar métricas
    # ------------------------------------------------------------------
    df_train = pd.DataFrame(results_train)
    df_test  = pd.DataFrame(results_test)

    comparison = df_test.copy()
    comparison.columns = ["modelo"] + [f"{c}_test" for c in df_test.columns if c != "modelo"]
    comparison = comparison.merge(
        df_train.rename(columns={c: f"{c}_train" for c in df_train.columns if c != "modelo"}),
        on="modelo"
    )
    comparison.to_csv(f"{METRICS_DIR}/model_comparison_v2.csv", index=False)

    for name, pred in [("LightGBM", pred_lgbm_te), ("XGBoost", pred_xgb_te),
                        ("CatBoost", pred_cat_te), ("Ensemble", pred_ens_te)]:
        dec = decile_analysis(np.array(y_test), pred, name)
        dec.to_csv(f"{METRICS_DIR}/decile_{name.lower()}_test.csv", index=False)

    # Pesos del ensemble
    pd.DataFrame([{"modelo": m, "peso": w} for m, w in weights.items()]).to_csv(
        f"{METRICS_DIR}/ensemble_weights.csv", index=False
    )

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES (TEST)")
    print("=" * 60)
    df_summary = df_test[["modelo", "R2", "RMSE", "KS", "Gini"]].copy()
    df_summary = df_summary.sort_values("R2", ascending=False)
    print(df_summary.to_string(index=False))

    print(f"\nMétricas guardadas en {METRICS_DIR}/")
    print(f"Modelos guardados en {MODELS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Menos trials Optuna (demo rapido)")
    args = parser.parse_args()
    main(args)
