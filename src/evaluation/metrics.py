"""
Métricas de evaluación para modelos de regresión en riesgo crediticio.

Incluye métricas estándar de regresión + métricas de discriminación (Gini, KS)
y estabilidad (PSI) usadas en la industria bancaria.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Métricas de regresión estándar
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """
    Calcula RMSE, MAE, R² y MAPE.
    El prefijo permite diferenciar train vs test en comparaciones.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE: evitar división por cero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    results = {
        f"{prefix}RMSE": round(rmse, 2),
        f"{prefix}MAE": round(mae, 2),
        f"{prefix}R2": round(r2, 4),
        f"{prefix}MAPE_%": round(mape, 2),
    }
    return results


# ---------------------------------------------------------------------------
# Métricas de discriminación (rank-ordering) — estándar en riesgo crediticio
# ---------------------------------------------------------------------------

def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Gini normalizado basado en la curva de Lorenz.
    Mide qué tan bien el modelo rank-ordena los clientes.
    Gini = 2 * AUC - 1 cuando se discretiza a binario, pero aquí se computa
    directamente sobre la regresión vía área bajo la curva de Lorenz.

    Rango: 0 (sin discriminación) → 1 (discriminación perfecta).
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values(
        "y_pred", ascending=True
    )
    n = len(df)
    lorenz_actual = df["y_true"].cumsum() / df["y_true"].sum()
    lorenz_equal = np.arange(1, n + 1) / n
    # Área entre la curva de Lorenz y la línea de igualdad perfecta
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz  # NumPy ≥2.0 renamed it
    gini = 1 - 2 * _trapz(lorenz_actual, lorenz_equal)
    return round(abs(gini), 4)


def ks_statistic(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Estadístico KS (Kolmogorov-Smirnov) adaptado a regresión.

    Método: ordena por score predicho, divide en deciles y calcula la diferencia
    máxima entre la distribución acumulada de valores altos y bajos de TARGET.
    Convención bancaria: targets sobre la mediana = "buenos", bajo = "malos".
    KS ∈ [0, 1]; valores > 0.3 se consideran buenos en scoring de crédito.
    """
    threshold = np.median(y_true)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).sort_values(
        "y_pred", ascending=False
    ).reset_index(drop=True)

    df["good"] = (df["y_true"] >= threshold).astype(int)
    df["bad"] = (df["y_true"] < threshold).astype(int)

    n_good = df["good"].sum()
    n_bad = df["bad"].sum()

    if n_good == 0 or n_bad == 0:
        return 0.0

    df["cum_good"] = df["good"].cumsum() / n_good
    df["cum_bad"] = df["bad"].cumsum() / n_bad
    ks = (df["cum_good"] - df["cum_bad"]).abs().max()
    return round(float(ks), 4)


def decile_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Tabla de análisis por decil: real vs predicho.
    Muestra la capacidad de ordenamiento del modelo por tramos de score.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=10, labels=False, duplicates="drop") + 1
    table = (
        df.groupby("decile")
        .agg(
            n=("y_true", "count"),
            mean_pred=("y_pred", "mean"),
            mean_actual=("y_true", "mean"),
            min_pred=("y_pred", "min"),
            max_pred=("y_pred", "max"),
        )
        .round(2)
    )
    table["error_pct"] = ((table["mean_pred"] - table["mean_actual"]) / table["mean_actual"] * 100).round(2)
    return table.reset_index()


def full_metrics_report(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str = "Modelo",
) -> pd.DataFrame:
    """
    Tabla completa de métricas para train y test en una sola llamada.
    Facilita la comparación entre GLM y ML en el notebook de evaluación.
    """
    train_metrics = regression_metrics(y_train, y_train_pred, prefix="train_")
    test_metrics = regression_metrics(y_test, y_test_pred, prefix="test_")

    train_metrics["train_Gini"] = gini_coefficient(y_train, y_train_pred)
    train_metrics["train_KS"] = ks_statistic(y_train, y_train_pred)
    test_metrics["test_Gini"] = gini_coefficient(y_test, y_test_pred)
    test_metrics["test_KS"] = ks_statistic(y_test, y_test_pred)

    all_metrics = {**train_metrics, **test_metrics}
    df = pd.DataFrame(all_metrics, index=[model_name])
    return df


# ---------------------------------------------------------------------------
# Estabilidad del modelo: PSI
# ---------------------------------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).

    Compara la distribución de scores entre dos períodos o poblaciones.
    Interpretación estándar en scoring bancario:
      PSI < 0.10  → Sin cambio significativo
      0.10–0.25   → Cambio moderado, monitorear
      PSI > 0.25  → Cambio severo, reentrenar modelo

    Args:
        expected: distribución de referencia (ej. scores de entrenamiento)
        actual:   distribución nueva (ej. scores del mes actual en producción)
        n_bins:   número de buckets para discretizar
    """
    # Crear bins basados en percentiles de la distribución esperada
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convertir a proporciones (evitar división por cero)
    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / len(actual))

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return round(float(np.sum(psi_values)), 4)


def psi_by_feature(
    df_ref: pd.DataFrame, df_cur: pd.DataFrame, features: list, n_bins: int = 10
) -> pd.DataFrame:
    """
    Calcula PSI para cada feature entre una población de referencia y una actual.
    Útil para monitoreo mensual de deriva de variables de entrada.
    """
    results = []
    for feat in features:
        if feat in df_ref.columns and feat in df_cur.columns:
            psi_val = psi(df_ref[feat].dropna().values, df_cur[feat].dropna().values, n_bins)
            alert = "OK" if psi_val < 0.10 else ("ADVERTENCIA" if psi_val < 0.25 else "ALERTA")
            results.append({"feature": feat, "PSI": psi_val, "status": alert})
    return pd.DataFrame(results).sort_values("PSI", ascending=False)
