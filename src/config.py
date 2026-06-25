"""
Configuración centralizada del proyecto. Un único lugar para constantes y rutas.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "data_modelo.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.parquet"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_processed.parquet"

GLM_MODEL_FILE = MODELS_DIR / "glm_model.pkl"
ML_MODEL_FILE = MODELS_DIR / "lgbm_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

# Columnas del dataset
TARGET_COL = "TARGET"
ID_COL = "ID"
BASE_COL = "BASE"
SENTINEL_VALUE = -9999998
NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X10", "X11", "X12"]
CATEGORICAL_FEATURES = ["X9"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Encoding
DATA_ENCODING = "latin-1"

# Reproducibilidad
RANDOM_SEED = 42
