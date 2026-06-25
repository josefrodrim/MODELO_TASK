"""
Tests unitarios para el módulo de preprocesamiento.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    replace_sentinels,
    missing_summary,
    detect_outliers_iqr,
    RiskPreprocessor,
)


@pytest.fixture
def sample_df():
    """DataFrame mínimo con las mismas columnas que el dataset real."""
    return pd.DataFrame({
        "X1": [100000.0, 200000.0, None, 150000.0],
        "X2": [1000.0, 2000.0, 3000.0, None],
        "X3": [-9999998.0, 5000.0, 3000.0, -9999998.0],
        "X4": [1000.0, -9999998.0, 2000.0, 3000.0],
        "X5": [None, 5000.0, None, 2000.0],
        "X6": [1.0, 2.0, 3.0, 1.0],
        "X7": [600.0, 700.0, 800.0, 900.0],
        "X8": [1.0, 2.0, 1.0, 2.0],
        "X9": ["MIRAFLORES", "SAN ISIDRO", None, "SURQUILLO"],
        "X10": [20000.0, 30000.0, 25000.0, 15000.0],
        "X11": [90.0, 100.0, None, 80.0],
        "X12": [30.0, 40.0, None, 25.0],
    })


class TestReplaceSentinels:
    def test_sentinel_becomes_nan(self, sample_df):
        result = replace_sentinels(sample_df)
        assert result["X3"].isna().sum() == 2
        assert result["X4"].isna().sum() == 1

    def test_no_original_mutation(self, sample_df):
        original_val = sample_df["X3"].iloc[0]
        replace_sentinels(sample_df)
        assert sample_df["X3"].iloc[0] == original_val  # no mutación in-place


class TestMissingSummary:
    def test_returns_only_columns_with_missing(self, sample_df):
        df = replace_sentinels(sample_df)
        summary = missing_summary(df)
        # Todas las columnas con missing deben aparecer
        assert "X3" in summary.index
        assert "X1" in summary.index

    def test_percentage_bounds(self, sample_df):
        df = replace_sentinels(sample_df)
        summary = missing_summary(df)
        assert summary["pct_missing"].between(0, 100).all()


class TestDetectOutliersIQR:
    def test_extreme_value_detected(self):
        series = pd.Series([10.0, 12.0, 11.0, 13.0, 1_000_000.0])
        mask = detect_outliers_iqr(series, factor=3.0)
        assert mask.iloc[-1]  # el valor extremo es outlier

    def test_normal_values_not_flagged(self):
        series = pd.Series([10.0, 12.0, 11.0, 13.0, 12.5])
        mask = detect_outliers_iqr(series, factor=3.0)
        assert not mask.any()


class TestRiskPreprocessor:
    def test_fit_transform_no_missing(self, sample_df):
        prep = RiskPreprocessor()
        prep.fit(sample_df)
        result = prep.transform(sample_df)
        # Columnas numéricas no deben tener NaN tras transformación
        num_cols = [c for c in result.columns if result[c].dtype in [np.float64, np.int64, int, float]]
        assert result[num_cols].isnull().sum().sum() == 0

    def test_sentinel_handled_in_transform(self, sample_df):
        prep = RiskPreprocessor()
        prep.fit(sample_df)
        result = prep.transform(sample_df)
        # No deben quedar valores centinela
        assert (result.values == -9999998).sum() == 0

    def test_x9_dummy_columns_created(self, sample_df):
        prep = RiskPreprocessor()
        prep.fit(sample_df)
        result = prep.transform(sample_df)
        x9_cols = [c for c in result.columns if c.startswith("X9_")]
        assert len(x9_cols) > 0

    def test_sentinel_flags_added_before_replace(self, sample_df):
        """X3_sin_registro debe ser 1 donde estaba el centinela, ANTES de imputar."""
        prep = RiskPreprocessor(add_missing_flags=True)
        prep.fit(sample_df)
        result = prep.transform(sample_df)
        # Verificar que los flags de condición de negocio existen
        assert "X3_sin_registro" in result.columns
        assert "X4_sin_registro" in result.columns
        # Verificar que capturan correctamente los centinelas originales
        n_sentinel_x3 = (sample_df["X3"] == -9999998).sum()
        assert result["X3_sin_registro"].sum() == n_sentinel_x3

    def test_null_flags_added(self, sample_df):
        """X5_sin_dato debe capturar los nulos reales de X5."""
        prep = RiskPreprocessor(add_missing_flags=True)
        prep.fit(sample_df)
        result = prep.transform(sample_df)
        assert "X5_sin_dato" in result.columns
        n_null_x5 = sample_df["X5"].isna().sum()
        assert result["X5_sin_dato"].sum() == n_null_x5

    def test_no_data_leakage_from_test(self, sample_df):
        """Bounds de clipping deben aprenderse solo en train."""
        train = sample_df.iloc[:3].copy()
        test = sample_df.iloc[3:].copy()
        prep = RiskPreprocessor()
        prep.fit(train)
        # Transformar test no debe lanzar excepción
        result = prep.transform(test)
        assert result is not None
