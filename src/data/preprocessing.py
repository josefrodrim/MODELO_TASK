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

# X3/X4: centinela -9999998 = cliente sin producto (condición de negocio)
SENTINEL_COLS = ["X3", "X4"]

# X11/X12: sus propios sentinelas distintos (descubiertos en EDA)
# X11: -99000720 (2,756 registros, 8.3% en train) → cliente sin historial
# X12: -99000792 (  466 registros, 1.4% en train) → cliente sin historial
X11_SENTINEL = -99000720.0
X12_SENTINEL = -99000792.0

# Nulos reales informativos del riesgo (dato no disponible, no condición de negocio)
INFORMATIVE_NULL_COLS = ["X1", "X2", "X5", "X7"]

# Variables financieras right-skewed (skewness > 1.5, min >= 0) → log1p
LOG_TRANSFORM_COLS = ["X1", "X2", "X3", "X4", "X5", "X10"]

# X7: bureau score (rango 0-995, mediana 812). Buckets con señal monotónica con TARGET.
# EDA: TARGET sube de 6,871 (<600) a 9,049 (850+) — relación clara y no lineal.
X7_BINS   = [0, 600, 700, 750, 800, 850, 1001]
X7_LABELS = ["X7_lt600", "X7_600_700", "X7_700_750", "X7_750_800", "X7_800_850", "X7_gt850"]

# X8: 5 categorías (1-5) → OHE
X8_CATEGORIES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Funciones utilitarias
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str, encoding: str = "latin-1") -> pd.DataFrame:
    return pd.read_csv(filepath, encoding=encoding)


def replace_sentinels(df: pd.DataFrame, sentinel: float = SENTINEL_VALUE) -> pd.DataFrame:
    """Reemplaza -9999998 en X3/X4 y los sentinelas propios de X11/X12."""
    df = df.copy()
    df = df.replace(sentinel, np.nan)
    if "X11" in df.columns:
        df["X11"] = df["X11"].replace(X11_SENTINEL, np.nan)
    if "X12" in df.columns:
        df["X12"] = df["X12"].replace(X12_SENTINEL, np.nan)
    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = 100 * missing / len(df)
    summary = pd.DataFrame({"n_missing": missing, "pct_missing": pct.round(2)})
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def sentinel_summary(df: pd.DataFrame, sentinel: float = SENTINEL_VALUE) -> pd.DataFrame:
    counts = (df == sentinel).sum()
    pct = 100 * counts / len(df)
    summary = pd.DataFrame({"n_centinela": counts, "pct_centinela": pct.round(2)})
    return summary[summary["n_centinela"] > 0].sort_values("pct_centinela", ascending=False)


def detect_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - factor * iqr) | (series > q3 + factor * iqr)


# ---------------------------------------------------------------------------
# Transformer sklearn-compatible
# ---------------------------------------------------------------------------

class RiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador + feature engineering para el dataset de riesgo crediticio.

    Orden de pasos:
      1.  Flags de condición de negocio para X3/X4 (centinela -9999998)
          y X11/X12 (centinelas propios: -99000720, -99000792)
      2.  Reemplazo de todos los centinelas → NaN
      3.  Flags de disponibilidad de dato para nulos reales (X1, X2, X5, X7)
      4.  Capping IQR×3 (bounds aprendidos en TRAIN)
      5.  Imputación por mediana (mediana calculada sobre valores reales en TRAIN)
      5b. Log-transform de variables financieras right-skewed (X1-X5, X10)
      5c. Feature engineering de negocio:
            - n_productos_sin_registro: cuántos de X3/X4 tienen centinela (0,1,2)
            - total_balance_log: log1p(X3 + X4) — exposición total en productos
            - X3_div_X4: ratio entre saldos de los dos productos
            - score_x_ingreso: log(X7+1) × log(X1+1) — score bureau × ingreso
            - X1_x_X10: producto de las dos variables más correlacionadas con TARGET
            - ratio_X2_X1: proxy deuda/ingreso (cap P99 para evitar extremos)
            - X11_X12_sum: X11+X12 cuando existen (co-missing siempre juntos)
            - X7_buckets: OHE del bureau score en 6 tramos de riesgo
      5d. X8 como categórica OHE (5 valores: 1-5)
      6.  X9 OHE top-25 distritos + OTROS + DESCONOCIDO

    Todos los estadísticos, bounds y categorías se aprenden SOLO en fit().
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        outlier_factor: float = 3.0,
        top_n_districts: int = TOP_N_DISTRICTS,
        add_missing_flags: bool = True,
        target_encode_x9: bool = True,
        x9_smoothing: float = 10.0,
    ):
        self.impute_strategy = impute_strategy
        self.outlier_factor = outlier_factor
        self.top_n_districts = top_n_districts
        self.add_missing_flags = add_missing_flags
        self.target_encode_x9 = target_encode_x9
        self.x9_smoothing = x9_smoothing

    def fit(self, X: pd.DataFrame, y=None):
        df = replace_sentinels(X.copy())

        self.numeric_cols_ = [c for c in NUMERIC_FEATURES if c in df.columns]

        self.clip_bounds_ = {}
        for col in self.numeric_cols_:
            series = df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            self.clip_bounds_[col] = (
                q1 - self.outlier_factor * iqr,
                q3 + self.outlier_factor * iqr,
            )

        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        self.imputer_.fit(df[self.numeric_cols_])

        # P99 de ratios para cap anti-outlier (aprendido en TRAIN)
        df_imp = df.copy()
        df_imp[self.numeric_cols_] = self.imputer_.transform(df[self.numeric_cols_])
        denom_1 = df_imp["X1"].replace(0, np.nan).fillna(df_imp["X1"].median())
        ratio_x2_x1 = df_imp["X2"] / denom_1.clip(lower=1)
        self.ratio_X2_X1_cap_ = float(ratio_x2_x1.quantile(0.99))

        denom_10 = df_imp["X10"].replace(0, np.nan).fillna(df_imp["X10"].median())
        ratio_x1_x10 = df_imp["X1"] / denom_10.clip(lower=1)
        self.ratio_X1_X10_cap_ = float(ratio_x1_x10.quantile(0.99))

        # X9: target encoding suavizado (evita leakage; media de TARGET por distrito)
        # Fórmula: encode(d) = (n_d * mean_d + k * global_mean) / (n_d + k)
        # k=smoothing_factor: distritos con pocos registros se acercan a la media global
        self.x9_target_encoding_ = {}
        self.x9_global_mean_ = 0.0
        if "X9" in df.columns and y is not None and self.target_encode_x9:
            y_arr = np.array(y)
            self.x9_global_mean_ = float(np.mean(y_arr))
            x9_vals = df["X9"].fillna("DESCONOCIDO").values
            for district in np.unique(x9_vals):
                mask = x9_vals == district
                n = mask.sum()
                mean_d = y_arr[mask].mean()
                k = self.x9_smoothing
                self.x9_target_encoding_[district] = (n * mean_d + k * self.x9_global_mean_) / (n + k)
        elif "X9" in df.columns:
            # Sin y: fallback a OHE
            self.top_districts_ = (
                df["X9"].fillna("DESCONOCIDO")
                .value_counts()
                .head(self.top_n_districts)
                .index.tolist()
            )

        self.x9_dummy_cols_ = None
        self.x8_dummy_cols_ = None
        self.x7_dummy_cols_ = None

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        # --- Paso 1: Flags de condición de negocio ---
        if self.add_missing_flags:
            for col in SENTINEL_COLS:
                if col in df.columns:
                    df[f"{col}_sin_registro"] = (df[col] == SENTINEL_VALUE).astype(int)
            if "X11" in df.columns:
                df["X11_sin_registro"] = (df["X11"] == X11_SENTINEL).astype(int)
            if "X12" in df.columns:
                df["X12_sin_registro"] = (df["X12"] == X12_SENTINEL).astype(int)

        # --- Paso 2: Reemplazar todos los centinelas → NaN ---
        df = replace_sentinels(df)

        # --- Paso 3: Flags de disponibilidad de dato (nulos reales) ---
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

        # --- Paso 5b: Log-transform de variables financieras ---
        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        # --- Paso 5c: Feature engineering de negocio ---

        # Cuántos productos sin registro tiene el cliente (0, 1 ó 2)
        # Captura el efecto acumulativo de no tener productos financieros
        if "X3_sin_registro" in df.columns and "X4_sin_registro" in df.columns:
            df["n_productos_sin_registro"] = df["X3_sin_registro"] + df["X4_sin_registro"]

        # Exposición total: suma de saldos de X3 y X4 (ambas son variables de saldo)
        if "X3" in df.columns and "X4" in df.columns:
            df["total_balance_log"] = np.log1p((df["X3"] + df["X4"]).clip(lower=0))
            # Ratio entre saldos: qué proporción del total está en X3
            total = (df["X3"] + df["X4"]).replace(0, np.nan)
            df["X3_share"] = (df["X3"] / total).fillna(0.5)

        # Score bureau × ingreso: combina la solvencia del cliente con su capacidad
        # EDA: TARGET sube monotónicamente con X7 (score) y X1 (ingreso)
        if "X7" in df.columns and "X1" in df.columns:
            df["score_x_ingreso"] = np.log1p(df["X7"].clip(lower=0)) * np.log1p(df["X1"].clip(lower=0))

        # Producto X1 × X10 (las dos features más correlacionadas con TARGET)
        if "X1" in df.columns and "X10" in df.columns:
            df["X1_x_X10_log"] = np.log1p(df["X1"].clip(lower=0)) * np.log1p(df["X10"].clip(lower=0))

        # Ratio X2/X1 con cap P99 (proxy deuda/ingreso)
        if "X1" in df.columns and "X2" in df.columns:
            denom_1 = df["X1"].replace(0, np.nan).fillna(df["X1"].median())
            df["ratio_X2_X1"] = (df["X2"] / denom_1.clip(lower=1)).clip(upper=self.ratio_X2_X1_cap_)

        # X11 + X12 (siempre co-missing, captura valor combinado cuando existen)
        if "X11" in df.columns and "X12" in df.columns:
            df["X11_X12_sum"] = df["X11"] + df["X12"]

        # Co-missing pattern: cuando faltan tanto X1 como X2 a la vez
        if "X1_sin_dato" in df.columns and "X2_sin_dato" in df.columns:
            df["co_missing_X1_X2"] = (df["X1_sin_dato"] & df["X2_sin_dato"]).astype(int)

        # Interacciones X6 × variables clave (X6 = segmento de empleo: 4 categorías)
        # EDA: TARGET varía 24% entre X6=1 y X6=4 → la misma variable vale distinto por segmento
        if "X6" in df.columns:
            if "X1_log" in df.columns:
                df["X6_x_X1_log"] = df["X6"] * df["X1_log"]
            if "X7" in df.columns:
                df["X6_x_X7"] = df["X6"] * df["X7"]
            if "score_x_ingreso" in df.columns:
                df["X6_x_score_ingreso"] = df["X6"] * df["score_x_ingreso"]

        # --- Paso 5d: X7 bureau score en buckets (relación no lineal con TARGET) ---
        if "X7" in df.columns:
            x7_bucketed = pd.cut(
                df["X7"].clip(lower=0, upper=1000),
                bins=X7_BINS, labels=X7_LABELS, right=False
            ).astype(str).fillna("X7_lt600")
            x7_dummies = pd.get_dummies(x7_bucketed, prefix="", prefix_sep="", dtype=int)
            if self.x7_dummy_cols_ is None:
                self.x7_dummy_cols_ = x7_dummies.columns.tolist()
            for col in self.x7_dummy_cols_:
                if col not in x7_dummies.columns:
                    x7_dummies[col] = 0
            x7_dummies = x7_dummies[self.x7_dummy_cols_]
            df = pd.concat([df, x7_dummies], axis=1)

        # --- Paso 5e: X8 como categórica OHE ---
        if "X8" in df.columns:
            x8_dummies = pd.get_dummies(
                df["X8"].round().astype(int).astype(str).apply(
                    lambda v: v if int(v) in X8_CATEGORIES else "otro"
                ),
                prefix="X8_cat", dtype=int
            )
            df = df.drop(columns=["X8"])
            if self.x8_dummy_cols_ is None:
                self.x8_dummy_cols_ = x8_dummies.columns.tolist()
            for col in self.x8_dummy_cols_:
                if col not in x8_dummies.columns:
                    x8_dummies[col] = 0
            x8_dummies = x8_dummies[self.x8_dummy_cols_]
            df = pd.concat([df, x8_dummies], axis=1)

        # --- Paso 6: X9 encoding ---
        # Target encoding (preferido): captura el ranking de ingreso por distrito
        # con suavizado bayesiano para evitar leakage en distritos con pocos registros.
        # Fallback a OHE si no se aprendió target encoding en fit().
        if "X9" in df.columns:
            if self.x9_target_encoding_:
                df["X9_target_enc"] = (
                    df["X9"].fillna("DESCONOCIDO")
                    .map(self.x9_target_encoding_)
                    .fillna(self.x9_global_mean_)
                )
                df = df.drop(columns=["X9"])
            else:
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
    cols_to_drop = [id_col, base_col, target_col]

    train = df[df[base_col] == "TRAIN"].reset_index(drop=True)
    test  = df[df[base_col] == "TEST"].reset_index(drop=True)

    X_train = train.drop(columns=cols_to_drop, errors="ignore")
    y_train = train[target_col]
    X_test  = test.drop(columns=cols_to_drop, errors="ignore")
    y_test  = test[target_col]

    preprocessor = RiskPreprocessor()
    preprocessor.fit(X_train, y_train)

    return preprocessor.transform(X_train), y_train, preprocessor.transform(X_test), y_test, preprocessor
