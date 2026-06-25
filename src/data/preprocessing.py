"""
Módulo de preprocesamiento de datos para el modelo de riesgo crediticio.

Pipeline compatible con scikit-learn que puede ser serializado y reutilizado
tanto en entrenamiento como en inferencia (API de producción).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

SENTINEL_VALUE = -9999998
NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X10", "X11", "X12"]
CATEGORICAL_FEATURES = ["X9"]
TOP_N_DISTRICTS = 25

# Columnas cuyo centinela (-9999998) indica una condición de negocio
# (el cliente NO TIENE el producto/historial asociado), no un nulo técnico.
# Análisis EDA confirmó: X3 centinela → TARGET -18% / X4 centinela → TARGET -6% (p ≈ 0)
SENTINEL_COLS = ["X3", "X4"]

# Columnas con nulos reales (dato no disponible) que son informativos del riesgo:
# X1→-26.5%, X2→-23.9%, X7→-27% sobre TARGET respecto a clientes con dato.
INFORMATIVE_NULL_COLS = ["X1", "X2", "X5", "X7"]


# ---------------------------------------------------------------------------
# Funciones utilitarias
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str, encoding: str = "latin-1") -> pd.DataFrame:
    """Carga el CSV con encoding correcto y devuelve un DataFrame."""
    return pd.read_csv(filepath, encoding=encoding)


def replace_sentinels(df: pd.DataFrame, sentinel: float = SENTINEL_VALUE) -> pd.DataFrame:
    """Convierte el valor centinela en NaN. Llamar DESPUÉS de crear los flags."""
    return df.replace(sentinel, np.nan)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con conteo y porcentaje de faltantes por columna."""
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    summary = pd.DataFrame({"n_missing": missing, "pct_missing": pct.round(2)})
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def sentinel_summary(df: pd.DataFrame, sentinel: float = SENTINEL_VALUE) -> pd.DataFrame:
    """Tabla específica de valores centinela por columna."""
    counts = (df == sentinel).sum()
    pct = 100 * counts / len(df)
    summary = pd.DataFrame({"n_centinela": counts, "pct_centinela": pct.round(2)})
    return summary[summary["n_centinela"] > 0].sort_values("pct_centinela", ascending=False)


def detect_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """
    Retorna máscara booleana True donde hay outlier extremo.
    Factor=3 preserva variabilidad legítima en variables de ingresos y saldos.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - factor * iqr) | (series > q3 + factor * iqr)


# ---------------------------------------------------------------------------
# Transformer sklearn-compatible
# ---------------------------------------------------------------------------

class RiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador completo para el dataset de riesgo crediticio.

    Orden de pasos (el orden importa para evitar leakage y pérdida de información):

      1. Flags de condicion de negocio (ANTES de reemplazar centinelas)
         El -9999998 no es un nulo técnico — indica que el cliente no tiene
         el producto/historial financiero asociado. Se captura como:
           X3_sin_registro, X4_sin_registro
         Análisis EDA: clientes con centinela tienen TARGET ~18% y ~6% menor.

      2. Reemplazo de centinelas → NaN
         Solo DESPUÉS de capturar los flags para no perder la señal.

      3. Flags de disponibilidad de dato (nulos reales)
         Distintos de los centinelas: indican ausencia de dato (calidad/origen).
         Columnas informativas: X1(-26.5%), X2(-23.9%), X7(-27%) menor TARGET.
           X1_sin_dato, X2_sin_dato, X5_sin_dato, X7_sin_dato

      4. Capping de outliers (límites aprendidos en TRAIN)

      5. Imputación numérica por mediana de cada columna en TRAIN
         Nota: el SimpleImputer aprende la mediana SOLO de valores reales
         (sentinel ya reemplazado en fit), por lo que no contamina la
         distribución de clientes con producto vs sin producto.

      6. Encoding de X9 (distrito) → OHE top-N + OTROS

    Los bounds, medianas y categorías se aprenden SOLO en fit() (anti-leakage).
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        outlier_factor: float = 3.0,
        top_n_districts: int = TOP_N_DISTRICTS,
        add_missing_flags: bool = True,
    ):
        self.impute_strategy = impute_strategy
        self.outlier_factor = outlier_factor
        self.top_n_districts = top_n_districts
        self.add_missing_flags = add_missing_flags

    def fit(self, X: pd.DataFrame, y=None):
        # Reemplazar centinelas ANTES de aprender cualquier estadístico
        # para que la mediana se calcule solo sobre valores reales del producto.
        df = replace_sentinels(X.copy())

        self.numeric_cols_ = [c for c in NUMERIC_FEATURES if c in df.columns]

        # Bounds de capping: aprendidos sobre valores reales (sin sentinelas ni NaN)
        self.clip_bounds_ = {}
        for col in self.numeric_cols_:
            series = df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            self.clip_bounds_[col] = (
                q1 - self.outlier_factor * iqr,
                q3 + self.outlier_factor * iqr,
            )

        # Mediana de cada columna calculada SOLO sobre clientes con valor real.
        # Para X3/X4: la mediana excluye los 40%/30% de clientes con centinela,
        # evitando mezclar distribuciones de clientes "con producto" y "sin producto".
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        self.imputer_.fit(df[self.numeric_cols_])

        if "X9" in df.columns:
            self.top_districts_ = (
                df["X9"].fillna("DESCONOCIDO")
                .value_counts()
                .head(self.top_n_districts)
                .index.tolist()
            )
        else:
            self.top_districts_ = []

        self.x9_dummy_cols_ = None

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # --- Paso 1: Flags de condición de negocio ---
        # Capturar ANTES de replace_sentinels para no perder la señal.
        # Nombre "sin_registro" (no "missing"): refleja que el cliente no tiene
        # el producto, no que el dato sea de mala calidad.
        if self.add_missing_flags:
            for col in SENTINEL_COLS:
                if col in df.columns:
                    df[f"{col}_sin_registro"] = (df[col] == SENTINEL_VALUE).astype(int)

        # --- Paso 2: Reemplazar centinelas → NaN ---
        df = replace_sentinels(df)

        # --- Paso 3: Flags de disponibilidad de dato (nulos reales) ---
        # Distintos de los centinelas: aquí el dato simplemente no existe.
        # Se crean ANTES de imputar para que el modelo pueda distinguir
        # entre valor real e imputado.
        if self.add_missing_flags:
            for col in INFORMATIVE_NULL_COLS:
                if col in df.columns:
                    df[f"{col}_sin_dato"] = df[col].isnull().astype(int)

        # --- Paso 4: Capping de outliers ---
        for col, (lower, upper) in self.clip_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        # --- Paso 5: Imputación numérica ---
        df[self.numeric_cols_] = self.imputer_.transform(df[self.numeric_cols_])

        # --- Paso 6: Encoding de X9 ---
        # Nota: X9 nulo → categoría "DESCONOCIDO" (TARGET +68% en esa categoría,
        # capturado naturalmente por el OHE).
        if "X9" in df.columns:
            df["X9_clean"] = df["X9"].fillna("DESCONOCIDO").apply(
                lambda x: x if x in self.top_districts_ else "OTROS"
            )
            x9_dummies = pd.get_dummies(df["X9_clean"], prefix="X9", dtype=int)
            df = df.drop(columns=["X9", "X9_clean"])

            if self.x9_dummy_cols_ is None:
                self.x9_dummy_cols_ = x9_dummies.columns.tolist()

            for col in self.x9_dummy_cols_:
                if col not in x9_dummies.columns:
                    x9_dummies[col] = 0

            x9_dummies = x9_dummies[self.x9_dummy_cols_]
            df = pd.concat([df, x9_dummies], axis=1)

        return df


# ---------------------------------------------------------------------------
# Función de alto nivel para uso en notebooks
# ---------------------------------------------------------------------------

def prepare_train_test(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    base_col: str = "BASE",
    id_col: str = "ID",
) -> tuple:
    """
    Divide en (X_train, y_train, X_test, y_test) usando la columna BASE.
    Retorna el preprocessor ajustado solo en TRAIN.
    """
    cols_to_drop = [id_col, base_col, target_col]

    train = df[df[base_col] == "TRAIN"].reset_index(drop=True)
    test = df[df[base_col] == "TEST"].reset_index(drop=True)

    X_train = train.drop(columns=cols_to_drop, errors="ignore")
    y_train = train[target_col]
    X_test = test.drop(columns=cols_to_drop, errors="ignore")
    y_test = test[target_col]

    preprocessor = RiskPreprocessor()
    preprocessor.fit(X_train)

    return preprocessor.transform(X_train), y_train, preprocessor.transform(X_test), y_test, preprocessor
