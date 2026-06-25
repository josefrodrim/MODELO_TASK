"""
Modelo GLM (OLS con target log-transformado) para regresión de riesgo crediticio.

Justificación del diseño:
  - TARGET es continuo y positivo (1K–40K), distribución right-skewed → log(TARGET) normaliza.
  - OLS sobre log(TARGET) es equivalente a un modelo log-normal, estándar en modelos de scoring.
  - statsmodels proporciona p-valores, intervalos de confianza y tests de supuestos.
  - Se aplica selección de variables por VIF + backward elimination con p-value threshold.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson as dw_test
from scipy import stats
import joblib

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Selección de variables
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de Varianza (VIF) para detectar multicolinealidad.
    VIF > 10 indica multicolinealidad alta; > 5 merece revisión.
    """
    X_const = sm.add_constant(X.select_dtypes(include=[np.number]))
    vif_data = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X_const.values, i)
        vif_data.append({"feature": col, "VIF": round(vif_val, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def remove_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> list:
    """
    Elimina iterativamente la variable con mayor VIF hasta que todas sean < threshold.
    Retorna la lista de features seleccionadas.
    """
    features = X.select_dtypes(include=[np.number]).columns.tolist()

    while True:
        vif_df = compute_vif(X[features])
        max_vif = vif_df["VIF"].max()
        if max_vif < threshold:
            break
        drop_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
        features.remove(drop_feature)

    return features


def backward_elimination(
    X: pd.DataFrame, y: pd.Series, significance_level: float = 0.05
) -> sm.OLS:
    """
    Eliminación hacia atrás por p-valor.
    Remueve la variable menos significativa hasta que todas tengan p < significance_level.
    Retorna el modelo OLS ajustado final.
    """
    features = X.columns.tolist()
    while True:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        # Excluir 'const' del criterio de eliminación
        pvalues = model.pvalues.drop("const", errors="ignore")
        max_pval = pvalues.max()
        if max_pval > significance_level:
            drop_feat = pvalues.idxmax()
            features.remove(drop_feat)
        else:
            break
    return model, features


# ---------------------------------------------------------------------------
# Entrenamiento y predicción
# ---------------------------------------------------------------------------

class GLMRiskModel:
    """
    Wrapper del modelo OLS log-normal para el pipeline de riesgo.

    Flujo:
      1. fit(): selección VIF → backward elimination → OLS en log(TARGET)
      2. predict(): predice en escala original (exp del OLS)
      3. get_summary(): resumen estadístico del modelo
    """

    def __init__(self, vif_threshold: float = 10.0, pvalue_threshold: float = 0.05):
        self.vif_threshold = vif_threshold
        self.pvalue_threshold = pvalue_threshold
        self.model_ = None
        self.selected_features_ = None
        self.log_transform_ = True  # Siempre usamos log(TARGET)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True):
        # Usar solo columnas numéricas (GLM no acepta dummies de alta cardinalidad fácilmente)
        X_num = X.select_dtypes(include=[np.number]).copy()

        if verbose:
            print(f"Features iniciales: {X_num.shape[1]}")

        # Paso 1: eliminar multicolinealidad
        features_after_vif = remove_high_vif(X_num, self.vif_threshold)
        if verbose:
            print(f"Features tras VIF < {self.vif_threshold}: {len(features_after_vif)}")

        # Paso 2: log-transformar TARGET
        y_log = np.log(y)

        # Paso 3: backward elimination
        self.model_, self.selected_features_ = backward_elimination(
            X_num[features_after_vif], y_log, self.pvalue_threshold
        )
        if verbose:
            print(f"Features tras backward elimination: {len(self.selected_features_)}")
            print(self.model_.summary())

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción en escala original mediante exp()."""
        X_sel = sm.add_constant(X[self.selected_features_], has_constant="add")
        log_pred = self.model_.predict(X_sel)
        return np.exp(log_pred)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción en escala logarítmica (salida directa del OLS)."""
        X_sel = sm.add_constant(X[self.selected_features_], has_constant="add")
        return self.model_.predict(X_sel)

    def get_summary(self):
        return self.model_.summary()

    def get_coefficients(self) -> pd.DataFrame:
        """Devuelve coeficientes con IC 95% en forma tabular."""
        params = self.model_.params
        conf = self.model_.conf_int()
        pvals = self.model_.pvalues
        df = pd.DataFrame({
            "coef": params,
            "ci_lower": conf[0],
            "ci_upper": conf[1],
            "p_value": pvals,
        })
        df["significativo"] = df["p_value"] < 0.05
        return df.round(6)


# ---------------------------------------------------------------------------
# Tests de supuestos
# ---------------------------------------------------------------------------

def assumption_tests(model: sm.OLS, X: pd.DataFrame, y_log: pd.Series) -> dict:
    """
    Ejecuta los tests estadísticos de supuestos del modelo lineal.

    Returns:
        dict con resultados de: normalidad de residuos, homocedasticidad,
        autocorrelación y multicolinealidad.
    """
    residuals = model.resid
    results = {}

    # Normalidad de residuos (Jarque-Bera)
    # scipy >= 1.9 retorna solo (statistic, pvalue); skew/kurt se calculan aparte
    jb_result = stats.jarque_bera(residuals)
    jb_stat, jb_pval = jb_result.statistic, jb_result.pvalue
    results["jarque_bera"] = {
        "statistic": round(float(jb_stat), 4),
        "p_value": round(float(jb_pval), 4),
        "skewness": round(float(stats.skew(residuals)), 4),
        "kurtosis": round(float(stats.kurtosis(residuals)), 4),
        "cumple_supuesto": float(jb_pval) > 0.05,
    }

    # Homocedasticidad (Breusch-Pagan)
    X_with_const = sm.add_constant(X)
    bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X_with_const)
    results["breusch_pagan"] = {
        "statistic": round(bp_stat, 4),
        "p_value": round(bp_pval, 4),
        "cumple_supuesto": bp_pval > 0.05,
    }

    # Autocorrelación (Durbin-Watson; ~2 = sin autocorrelación)
    dw_stat = float(dw_test(residuals))
    results["durbin_watson"] = {
        "statistic": round(dw_stat, 4),
        "cumple_supuesto": 1.5 < dw_stat < 2.5,
    }

    return results


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def save_glm_model(model: GLMRiskModel, path: str):
    joblib.dump(model, path)


def load_glm_model(path: str) -> GLMRiskModel:
    return joblib.load(path)
