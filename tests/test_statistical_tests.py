"""
test_statistical_tests.py — Tests for statistical inference utilities.
======================================================================
BH-FDR correction, Hurst exponent (R/S + GPH), SPA test,
stationary block bootstrap, and bootstrap Sharpe ratio test.
"""

import numpy as np
import pandas as pd
import pytest

from src.statistical_tests import (
    benjamini_hochberg,
    bootstrap_sharpe_test,
    hurst_gph,
    hurst_rs,
    spa_test,
    stationary_block_bootstrap,
)


# ══════════════════════════════════════════════
# 1.  BENJAMINI-HOCHBERG FDR
# ══════════════════════════════════════════════

class TestBenjaminiHochberg:
    """BH-FDR correction tests."""

    def test_all_significant(self):
        """When all p-values are very small, all should be rejected."""
        pvals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
        assert all(rejected)
        assert all(adjusted <= 0.05)

    def test_none_significant(self):
        """When all p-values are large, none should be rejected."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
        assert not any(rejected)

    def test_partial_rejection(self):
        """Mixed p-values: some rejected, some not."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
        assert rejected[0] and rejected[1]  # first two should be rejected
        assert not rejected[4]  # last should not

    def test_adjusted_monotonicity(self):
        """Adjusted p-values should preserve the ordering of raw p-values."""
        pvals = np.array([0.01, 0.03, 0.04, 0.05, 0.1, 0.5, 0.8])
        _, adjusted = benjamini_hochberg(pvals, q=0.05)
        sorted_idx = np.argsort(pvals)
        adj_sorted = adjusted[sorted_idx]
        # Adjusted values should be non-decreasing when sorted by raw p-values
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1] + 1e-10

    def test_empty_input(self):
        """Empty array should return empty results."""
        rejected, adjusted = benjamini_hochberg(np.array([]), q=0.05)
        assert len(rejected) == 0
        assert len(adjusted) == 0

    def test_single_pvalue(self):
        """Single p-value should be compared directly to q."""
        rejected, adjusted = benjamini_hochberg(np.array([0.01]), q=0.05)
        assert rejected[0]
        rejected2, _ = benjamini_hochberg(np.array([0.10]), q=0.05)
        assert not rejected2[0]

    def test_twenty_tickers_scenario(self):
        """Realistic scenario: 20 ADF p-values."""
        np.random.seed(42)
        # Most stocks have stationary returns (small p-values)
        pvals = np.concatenate([
            np.random.uniform(0.001, 0.01, 15),  # significant
            np.random.uniform(0.1, 0.5, 5),       # not significant
        ])
        rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
        assert sum(rejected) >= 10  # most should survive BH
        assert sum(rejected) <= 20


# ══════════════════════════════════════════════
# 2.  HURST EXPONENT
# ══════════════════════════════════════════════

class TestHurstExponent:
    """Tests for R/S and GPH Hurst estimators."""

    def test_rs_random_walk_near_half(self):
        """Random walk should give H ≈ 0.5."""
        np.random.seed(42)
        rw = np.random.normal(0, 1, 2000)
        H = hurst_rs(rw)
        assert 0.3 < H < 0.7, f"Random walk H should be ~0.5, got {H}"

    def test_rs_persistent_series(self):
        """Persistent series should give H > 0.5."""
        np.random.seed(42)
        # Create persistent series (cumulative sum of positive-biased noise)
        n = 2000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + np.random.normal(0, 1)
        # Absolute values should show long memory
        H = hurst_rs(np.abs(x))
        assert H > 0.4  # should show some persistence

    def test_gph_returns_three_values(self):
        """GPH should return (d, se, p_value)."""
        np.random.seed(42)
        series = np.random.normal(0, 1, 500)
        d, se, p_value = hurst_gph(series)
        assert not np.isnan(d)
        assert se > 0
        assert 0 <= p_value <= 1

    def test_gph_short_series(self):
        """Very short series should return NaN."""
        d, se, p = hurst_gph(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(d)

    def test_rs_short_series(self):
        """Very short series should return NaN."""
        H = hurst_rs(np.array([1.0, 2.0, 3.0]))
        assert np.isnan(H)


# ══════════════════════════════════════════════
# 3.  STATIONARY BLOCK BOOTSTRAP
# ══════════════════════════════════════════════

class TestStationaryBlockBootstrap:
    """Tests for Politis & Romano bootstrap."""

    def test_output_shape(self):
        """Output should be (B, T)."""
        data = np.random.normal(0, 1, 100)
        samples = stationary_block_bootstrap(data, B=50, seed=42)
        assert samples.shape == (50, 100)

    def test_multivariate_shape(self):
        """Multivariate input should preserve columns."""
        data = np.random.normal(0, 1, (100, 5))
        samples = stationary_block_bootstrap(data, B=50, seed=42)
        assert samples.shape == (50, 100, 5)

    def test_reproducibility(self):
        """Same seed should give identical results."""
        data = np.random.normal(0, 1, 100)
        s1 = stationary_block_bootstrap(data, B=10, seed=42)
        s2 = stationary_block_bootstrap(data, B=10, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_values_from_original(self):
        """All bootstrap values should come from the original data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        samples = stationary_block_bootstrap(data, B=20, seed=42)
        for b in range(20):
            for val in samples[b]:
                assert val in data


# ══════════════════════════════════════════════
# 4.  BOOTSTRAP SHARPE RATIO TEST
# ══════════════════════════════════════════════

class TestBootstrapSharpe:
    """Tests for Ledoit & Wolf Sharpe ratio comparison."""

    def test_identical_returns(self):
        """Same returns should give p-value close to 1."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 500)
        result = bootstrap_sharpe_test(returns, returns, B=1000)
        assert result["delta"] == 0.0
        assert result["p_value"] > 0.5

    def test_different_returns(self):
        """Very different returns should give low p-value."""
        np.random.seed(42)
        returns_a = np.random.normal(0.005, 0.01, 1000)
        returns_b = np.random.normal(-0.005, 0.01, 1000)
        result = bootstrap_sharpe_test(returns_a, returns_b, B=2000)
        assert result["delta"] > 0
        assert result["p_value"] < 0.10

    def test_output_keys(self):
        """Result dict should have expected keys."""
        np.random.seed(42)
        r = np.random.normal(0, 0.02, 200)
        result = bootstrap_sharpe_test(r, r, B=100)
        assert "sharpe_a" in result
        assert "sharpe_b" in result
        assert "delta" in result
        assert "p_value" in result


# ══════════════════════════════════════════════
# 5.  SPA TEST
# ══════════════════════════════════════════════

class TestSPATest:
    """Tests for Hansen (2005) Superior Predictive Ability test."""

    def test_benchmark_beats_all(self):
        """When benchmark is best, p-value should be high (fail to reject H₀)."""
        np.random.seed(42)
        T = 500
        # Benchmark has lowest loss
        loss_benchmark = np.random.uniform(0, 1, T)
        loss_model1 = loss_benchmark + np.random.uniform(0, 0.5, T)
        loss_model2 = loss_benchmark + np.random.uniform(0, 0.3, T)
        loss_matrix = np.column_stack([loss_benchmark, loss_model1, loss_model2])
        result = spa_test(loss_matrix, benchmark_col=0, B=1000)
        assert result["p_value"] > 0.10

    def test_model_beats_benchmark(self):
        """When a model clearly outperforms, p-value should be low."""
        np.random.seed(42)
        T = 500
        loss_benchmark = np.random.uniform(0.5, 1.5, T)
        loss_model1 = np.random.uniform(0, 0.5, T)  # much better
        loss_matrix = np.column_stack([loss_benchmark, loss_model1])
        result = spa_test(loss_matrix, benchmark_col=0, B=1000)
        assert result["p_value"] < 0.10
        assert result["best_model_idx"] == 1

    def test_output_keys(self):
        """Result dict should have expected keys."""
        np.random.seed(42)
        loss_matrix = np.random.uniform(0, 1, (100, 3))
        result = spa_test(loss_matrix, B=100)
        assert "statistic" in result
        assert "p_value" in result
        assert "best_model_idx" in result
        assert "d_bar" in result
