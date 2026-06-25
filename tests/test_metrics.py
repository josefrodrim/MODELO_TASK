"""
Tests unitarios para el módulo de métricas.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    regression_metrics,
    gini_coefficient,
    ks_statistic,
    psi,
    decile_table,
)

np.random.seed(42)

Y_TRUE = np.random.uniform(1000, 40000, 500)
Y_PRED_GOOD = Y_TRUE + np.random.normal(0, 500, 500)   # predictor bueno
Y_PRED_RANDOM = np.random.uniform(1000, 40000, 500)    # predictor aleatorio


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        m = regression_metrics(Y_TRUE, Y_TRUE)
        assert m["RMSE"] == 0.0
        assert m["MAE"] == 0.0
        assert m["R2"] == 1.0

    def test_good_predictor_better_than_random(self):
        m_good = regression_metrics(Y_TRUE, Y_PRED_GOOD)
        m_rand = regression_metrics(Y_TRUE, Y_PRED_RANDOM)
        assert m_good["RMSE"] < m_rand["RMSE"]
        assert m_good["R2"] > m_rand["R2"]

    def test_prefix_applied(self):
        m = regression_metrics(Y_TRUE, Y_PRED_GOOD, prefix="train_")
        assert "train_RMSE" in m
        assert "RMSE" not in m


class TestGiniCoefficient:
    def test_gini_perfect_order(self):
        # For uniform y, perfect ordering gives Gini ≈ 0.33 (Lorenz of linear dist).
        # Key property: perfect predictor should beat a random shuffled one.
        y = np.arange(1, 101, dtype=float)
        gini_perfect = gini_coefficient(y, y)
        gini_shuffled = gini_coefficient(y, np.random.permutation(y))
        assert gini_perfect > gini_shuffled  # perfect ordering beats random

    def test_gini_random_near_zero(self):
        y = np.ones(200)  # sin variabilidad → Gini ~0
        pred = np.random.uniform(0, 1, 200)
        gini = gini_coefficient(y, pred)
        assert gini < 0.1

    def test_gini_range(self):
        gini = gini_coefficient(Y_TRUE, Y_PRED_GOOD)
        assert 0.0 <= gini <= 1.0


class TestKSStatistic:
    def test_ks_range(self):
        ks = ks_statistic(Y_TRUE, Y_PRED_GOOD)
        assert 0.0 <= ks <= 1.0

    def test_good_predictor_higher_ks(self):
        ks_good = ks_statistic(Y_TRUE, Y_PRED_GOOD)
        ks_rand = ks_statistic(Y_TRUE, Y_PRED_RANDOM)
        assert ks_good > ks_rand


class TestPSI:
    def test_same_distribution_psi_near_zero(self):
        arr = np.random.normal(0, 1, 1000)
        psi_val = psi(arr, arr)
        assert psi_val < 0.05

    def test_different_distribution_psi_high(self):
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)  # shift significativo
        psi_val = psi(expected, actual)
        assert psi_val > 0.25

    def test_psi_non_negative(self):
        a = np.random.uniform(0, 10, 500)
        b = np.random.uniform(2, 12, 500)
        assert psi(a, b) >= 0


class TestDecileTable:
    def test_returns_dataframe(self):
        table = decile_table(Y_TRUE, Y_PRED_GOOD)
        assert hasattr(table, "columns")

    def test_has_expected_columns(self):
        table = decile_table(Y_TRUE, Y_PRED_GOOD)
        for col in ["decile", "n", "mean_pred", "mean_actual"]:
            assert col in table.columns
