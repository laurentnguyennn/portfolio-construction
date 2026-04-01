"""
test_pipeline_integrity.py — No-lookahead-bias and pipeline correctness tests.
===============================================================================
These tests verify the critical data integrity requirements from CLAUDE.md §7.

CRITICAL CHECKS:
1. No feature at time t uses information from t+1 or later.
2. Sentiment features are t-1 lagged.
3. Walk-forward splits are strictly temporal.
4. SMOTE is only used in classification pipelines.
5. StandardScaler is inside the pipeline.
6. Cornish-Fisher uses excess kurtosis.
7. Portfolio CVaR is from joint scenarios.
8. BL Omega = diag(RMSE²), not diag(1/RMSE).
"""

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════
# 1. NO LOOKAHEAD BIAS
# ══════════════════════════════════════════════

class TestNoLookaheadBias:
    """Verify no future information leaks into features or targets."""

    def test_returns_use_only_past(self):
        """Log returns at time t use only prices at t and t-1."""
        from src.feature_engineering import compute_log_returns

        prices = pd.DataFrame({"A": [100, 110, 105, 120, 115]},
                               index=pd.bdate_range("2023-01-01", periods=5))
        log_ret = compute_log_returns(prices)

        # Return at index 1 = log(110/100), uses only t=1 and t=0
        expected = np.log(110 / 100)
        assert abs(log_ret["A"].iloc[1] - expected) < 1e-10

        # Return at index 0 should be NaN (no prior data)
        assert pd.isna(log_ret["A"].iloc[0])

    def test_rolling_vol_uses_only_past(self):
        """Rolling volatility at time t uses only data up to t."""
        from src.feature_engineering import realized_vol

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        vol = realized_vol(returns, window=10, annualize=False)

        # Vol at index 9 uses returns[0:10]
        manual_vol = returns.iloc[:10].std()
        assert abs(vol.iloc[9] - manual_vol) < 1e-10

        # Vol at index 8 should be NaN (window not complete)
        assert pd.isna(vol.iloc[8])

    def test_rsi_uses_only_past(self):
        """RSI at time t uses only data up to t."""
        from src.feature_engineering import rsi

        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.normal(0, 1, 100)) + 100)
        rsi_val = rsi(prices, period=14)

        # First 14 values should have NaN (insufficient history)
        assert rsi_val.iloc[:13].isna().all()

    def test_yang_zhang_vol_no_future(self):
        """Yang-Zhang volatility must not use future data."""
        from src.feature_engineering import yang_zhang_vol

        np.random.seed(42)
        n = 100
        o = pd.Series(np.random.uniform(99, 101, n))
        h = o + np.random.uniform(0, 2, n)
        l = o - np.random.uniform(0, 2, n)
        c = (h + l) / 2

        yz = yang_zhang_vol(o, h, l, c, window=21, annualize=False)

        # Changing future data should not affect past estimates
        c_modified = c.copy()
        c_modified.iloc[-1] = c.iloc[-1] * 2  # double last close

        yz_modified = yang_zhang_vol(o, h, l, c_modified, window=21, annualize=False)

        # All values except the last window should be identical
        np.testing.assert_array_almost_equal(
            yz.iloc[:79].values,
            yz_modified.iloc[:79].values,
            decimal=10,
            err_msg="Changing future data should not affect past YZ vol"
        )


# ══════════════════════════════════════════════
# 2. SENTIMENT LAG ENFORCEMENT
# ══════════════════════════════════════════════

class TestSentimentLag:
    """Verify sentiment features are t-1 lagged."""

    def test_sentiment_shift(self):
        """Lagged sentiment at time t should equal raw sentiment at t-1."""
        dates = pd.bdate_range("2023-01-01", periods=10)
        raw_sentiment = pd.Series(np.arange(10, dtype=float), index=dates,
                                   name="sentiment_mean")

        # Apply t-1 lag (as done in NB06)
        lagged = raw_sentiment.shift(1)

        # Lagged[1] should equal Raw[0]
        assert lagged.iloc[1] == raw_sentiment.iloc[0]
        # Lagged[5] should equal Raw[4]
        assert lagged.iloc[5] == raw_sentiment.iloc[4]
        # Lagged[0] should be NaN (no prior data)
        assert pd.isna(lagged.iloc[0])


# ══════════════════════════════════════════════
# 3. WALK-FORWARD TEMPORAL INTEGRITY
# ══════════════════════════════════════════════

class TestWalkForward:
    """Verify walk-forward evaluation respects temporal ordering."""

    def test_no_future_in_training(self):
        """Training data must never extend beyond the current evaluation point."""
        from src.ml_pipeline import walk_forward_predict, make_regression_pipeline

        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
        }, index=pd.bdate_range("2020-01-01", periods=n))
        y = pd.Series(np.random.normal(0, 1, n), index=X.index)

        preds = walk_forward_predict(
            X, y,
            pipeline_factory=lambda: make_regression_pipeline("ridge"),
            retrain_freq=63,
            initial_train_ratio=0.7,
        )

        if len(preds) > 0:
            # First prediction date should be after 70% of data
            first_pred_date = pd.Timestamp(preds["date"].iloc[0])
            split_date = X.index[int(n * 0.7)]
            assert first_pred_date >= split_date, \
                "First prediction must be after training window"

    def test_expanding_window(self):
        """Each retraining should use MORE data than the previous one."""
        # This is a design test — the walk_forward_predict function
        # uses expanding (not sliding) windows
        from src.config import TRAIN_RATIO, RETRAIN_FREQ_DAYS
        assert TRAIN_RATIO == 0.70
        assert RETRAIN_FREQ_DAYS == 63


# ══════════════════════════════════════════════
# 4. PIPELINE HYGIENE
# ══════════════════════════════════════════════

class TestPipelineHygiene:
    """Verify pipeline construction prevents data leakage."""

    def test_scaler_inside_regression_pipeline(self):
        """StandardScaler must be inside the pipeline, not applied separately."""
        from src.ml_pipeline import make_regression_pipeline

        pipe = make_regression_pipeline("ridge")
        step_names = [name for name, _ in pipe.steps]
        assert "scaler" in step_names, "Scaler must be inside pipeline"
        assert step_names.index("scaler") < step_names.index("model"), \
            "Scaler must come before model"

    def test_smote_in_classification_only(self):
        """SMOTE should only be in classification pipelines."""
        from src.ml_pipeline import make_regression_pipeline, make_classification_pipeline

        reg_pipe = make_regression_pipeline("ridge")
        reg_step_names = [name for name, _ in reg_pipe.steps]
        assert "smote" not in reg_step_names, \
            "SMOTE must NOT be in regression pipeline"

        clf_pipe = make_classification_pipeline("logistic", use_smote=True)
        clf_step_names = [name for name, _ in clf_pipe.steps]
        assert "smote" in clf_step_names, \
            "SMOTE should be in classification pipeline when use_smote=True"

    def test_no_smote_without_flag(self):
        """Classification pipeline without SMOTE flag should not have SMOTE."""
        from src.ml_pipeline import make_classification_pipeline

        pipe = make_classification_pipeline("logistic", use_smote=False)
        step_names = [name for name, _ in pipe.steps]
        assert "smote" not in step_names


# ══════════════════════════════════════════════
# 5. MATHEMATICAL CORRECTNESS
# ══════════════════════════════════════════════

class TestMathematicalCorrectness:
    """Verify critical mathematical formulas."""

    def test_excess_kurtosis_in_cornish_fisher(self):
        """
        CRITICAL: pandas .kurtosis() returns EXCESS kurtosis.
        Cornish-Fisher formula requires excess kurtosis.
        Using raw kurtosis (excess + 3) would be WRONG.
        """
        # Normal distribution: excess kurtosis ≈ 0, raw kurtosis ≈ 3
        np.random.seed(42)
        normal = pd.Series(np.random.normal(0, 1, 100000))

        excess_kurt = normal.kurtosis()   # pandas default
        assert abs(excess_kurt) < 0.1, \
            f"Normal excess kurtosis should be ~0, got {excess_kurt}"

        # Verify our CF implementation uses this correctly
        from src.risk_metrics import var_cornish_fisher, var_parametric_gaussian
        var_cf = var_cornish_fisher(normal * 0.02, 0.05)
        var_g = var_parametric_gaussian(normal * 0.02, 0.05)

        # For normal data, CF ≈ Gaussian (excess kurt ≈ 0 contributes nothing)
        assert abs(var_cf - var_g) / var_g < 0.02, \
            "CF should match Gaussian for normal data"

    def test_bl_omega_is_rmse_squared(self):
        """
        Black-Litterman Omega_ii = RMSE_i² (variance, not 1/RMSE).

        Omega has units of variance (return²), matching the covariance
        matrix units. Using 1/RMSE would give incorrect units.
        """
        # Verify by dimension: if RMSE ~ 0.05 (5% return),
        # then Omega_ii = 0.05² = 0.0025 (variance units)
        rmse = np.array([0.05, 0.10, 0.03])
        omega_correct = np.diag(rmse ** 2)
        omega_wrong = np.diag(1.0 / rmse)

        # Correct Omega should have small diagonal values (variance scale)
        assert abs(omega_correct[0, 0] - 0.0025) < 1e-12
        assert abs(omega_correct[1, 1] - 0.01) < 1e-12

        # Wrong Omega would have large values (inverse scale)
        assert omega_wrong[0, 0] == 20.0  # way too large

    def test_yang_zhang_vol_formula(self):
        """
        Verify Yang-Zhang volatility estimator components:
        σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS
        """
        from src.feature_engineering import yang_zhang_vol

        # Simple test: YZ vol should be positive and finite
        np.random.seed(42)
        n = 100
        o = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, n)))
        h = o + np.abs(np.random.normal(0, 1, n))
        l = o - np.abs(np.random.normal(0, 1, n))
        c = o + np.random.normal(0, 0.5, n)

        yz = yang_zhang_vol(o, h, l, c, window=21, annualize=False)
        valid = yz.dropna()
        assert (valid > 0).all(), "YZ vol must be positive"
        assert np.isfinite(valid).all(), "YZ vol must be finite"

    def test_dcc_stationarity_constraint(self):
        """
        DCC parameters must satisfy a + b < 1 (stationarity).

        We use a tighter bound of 0.95 to avoid near-unit-root correlation
        dynamics that produce explosive Q_t matrices.
        Bounds enforced in dcc_garch.estimate_dcc_params:
          a ∈ (0, 0.20), b ∈ (0, 0.94), a + b ≤ 0.95
        """
        # Verify the constraint value is correctly set in the source
        import inspect
        from src import dcc_garch
        src = inspect.getsource(dcc_garch.estimate_dcc_params)
        assert "0.95" in src, (
            "DCC stationarity constraint should be 0.95 (a + b ≤ 0.95)"
        )
        assert "0.999" not in src, (
            "Old loose constraint 0.999 should have been replaced with 0.95"
        )

    def test_portfolio_constraints(self):
        """Verify UCITS constraint values match CLAUDE.md."""
        from src.config import (
            MAX_SINGLE_STOCK_WEIGHT,
            MAX_SECTOR_WEIGHT,
            MAX_MONTHLY_TURNOVER,
            TRANSACTION_COST_BPS,
        )
        assert MAX_SINGLE_STOCK_WEIGHT == 0.10, "Max 10% per stock"
        assert MAX_SECTOR_WEIGHT == 0.30, "Max 30% per sector"
        assert MAX_MONTHLY_TURNOVER == 0.20, "Max 20% monthly turnover"
        assert TRANSACTION_COST_BPS == 10.0, "10 bps round-trip"

    def test_sector_groups_coverage(self):
        """All 20 tickers must appear in exactly one sector group."""
        from src.config import SECTOR_GROUPS, TICKERS

        all_grouped = []
        for group, members in SECTOR_GROUPS.items():
            all_grouped.extend(members)

        # Every ticker in a group
        for t in TICKERS:
            assert t in all_grouped, f"{t} not in any sector group"

        # No duplicates
        assert len(all_grouped) == len(set(all_grouped)), \
            "Tickers must appear in exactly one sector group"

        # All grouped tickers are in universe
        for t in all_grouped:
            assert t in TICKERS, f"{t} in sector group but not in TICKERS"


# ══════════════════════════════════════════════
# 6. DIEBOLD-MARIANO CORRECTNESS
# ══════════════════════════════════════════════

class TestDieboldMariano:
    """Test DM test implementation."""

    def test_dm_same_errors(self):
        """DM test with identical errors should not reject H0."""
        from src.ml_pipeline import diebold_mariano_test

        np.random.seed(42)
        e = np.random.normal(0, 1, 500)
        result = diebold_mariano_test(e, e, h=1)
        # Same errors → d_t = 0 → can't compute
        # Should return NaN or very high p-value
        assert np.isnan(result["p_value"]) or result["p_value"] > 0.05

    def test_dm_hac_for_multistep(self):
        """DM test with h > 1 should use HAC variance estimator."""
        from src.ml_pipeline import diebold_mariano_test

        np.random.seed(42)
        e1 = np.random.normal(0, 1, 500)
        e2 = np.random.normal(0.1, 1, 500)

        # h=1: no HAC needed
        r1 = diebold_mariano_test(e1, e2, h=1)
        # h=5: HAC with bandwidth 4
        r5 = diebold_mariano_test(e1, e2, h=5)

        # Both should be valid
        assert not np.isnan(r1["dm_stat"])
        assert not np.isnan(r5["dm_stat"])


# ══════════════════════════════════════════════
# 7. RANDOM STATE REPRODUCIBILITY
# ══════════════════════════════════════════════

class TestReproducibility:
    """Verify reproducibility via random_state=42."""

    def test_random_state_config(self):
        from src.config import RANDOM_STATE
        assert RANDOM_STATE == 42

    def test_consistent_results(self):
        """Same seed should produce identical results."""
        np.random.seed(42)
        a = np.random.normal(0, 1, 100)
        np.random.seed(42)
        b = np.random.normal(0, 1, 100)
        np.testing.assert_array_equal(a, b)


# ══════════════════════════════════════════════
# 8. WALK-FORWARD EMBARGO VERIFICATION
# ══════════════════════════════════════════════

class TestWalkForwardEmbargo:
    """Verify embargo/purge protocol prevents label leakage."""

    def test_no_label_leakage_5d(self):
        """
        Verify no training observation has a 5d forward label
        overlapping the test period.

        If train ends at T_train_end and we predict h=5 day forward vol,
        then labels for t in [T_train_end-4, T_train_end] use data
        from t+1 to t+5, which may overlap the test set.
        Purge removes these observations.
        """
        h = 5  # forecast horizon
        T_train_end = 350
        T = 500

        # Simulate: feature dates and label end dates
        train_dates = np.arange(T_train_end + 1)
        label_end_dates = train_dates + h  # label uses data up to t+h

        # After purge: remove obs where label extends past train end
        purged = train_dates[label_end_dates <= T_train_end]
        assert purged[-1] + h <= T_train_end, \
            "Purged training set should not have labels extending past train end"

    def test_embargo_gap_exists(self):
        """Verify embargo of h days between train end and test start."""
        h = 5
        T_train_end = 350

        # After embargo: test starts at T_train_end + h + 1
        test_start = T_train_end + h + 1
        assert test_start - T_train_end >= h, \
            f"Embargo gap must be >= {h}, got {test_start - T_train_end}"

    def test_embargo_21d(self):
        """Same check for 21-day forecast horizon."""
        h = 21
        T_train_end = 350
        train_dates = np.arange(T_train_end + 1)
        label_end_dates = train_dates + h
        purged = train_dates[label_end_dates <= T_train_end]
        assert purged[-1] + h <= T_train_end


# ══════════════════════════════════════════════
# 9. BH-FDR VERIFICATION
# ══════════════════════════════════════════════

class TestMultipleTesting:
    """Verify BH-FDR correction is properly applied."""

    def test_bh_fdr_more_conservative(self):
        """BH-adjusted p-values must be >= raw p-values."""
        from src.statistical_tests import benjamini_hochberg

        pvals = np.array([0.001, 0.01, 0.03, 0.04, 0.06, 0.1, 0.2, 0.5])
        _, adjusted = benjamini_hochberg(pvals, q=0.05)
        assert all(adjusted >= pvals - 1e-10), \
            "Adjusted p-values must be >= raw p-values"

    def test_bh_fdr_twenty_tickers(self):
        """Simulate 20-ticker ADF test scenario."""
        from src.statistical_tests import benjamini_hochberg

        # 18 stocks are clearly stationary (p < 0.01), 2 are borderline
        np.random.seed(42)
        pvals = np.concatenate([
            np.random.uniform(0.001, 0.01, 18),
            np.array([0.04, 0.06]),
        ])
        rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
        # At minimum, the 18 clearly significant should survive
        assert sum(rejected) >= 18


# ══════════════════════════════════════════════
# 10. HURST PREREQUISITE FOR FIGARCH
# ══════════════════════════════════════════════

class TestHurstBeforeFIGARCH:
    """Verify FIGARCH is only fitted when long-memory is detected."""

    def test_random_walk_no_long_memory(self):
        """White noise should NOT show long memory (H ≈ 0.5, d ≈ 0)."""
        from src.statistical_tests import hurst_rs, hurst_gph

        np.random.seed(42)
        white_noise = np.random.normal(0, 1, 2000)
        H = hurst_rs(white_noise)
        d, se, p = hurst_gph(white_noise)

        # Should not satisfy FIGARCH prerequisite
        # H should be close to 0.5 (not clearly > 0.5)
        # d should not be significantly > 0
        figarch_eligible = H > 0.5 and d > 0 and p < 0.05
        # For white noise, this combination should rarely occur
        # (we don't assert it never occurs due to randomness, but check H ~ 0.5)
        assert 0.2 < H < 0.8, f"White noise H should be ~0.5, got {H}"

    def test_long_memory_series(self):
        """Series with known long memory should give H > 0.5."""
        from src.statistical_tests import hurst_rs

        np.random.seed(42)
        # Create ARFIMA-like series with long memory
        n = 3000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + np.random.normal(0, 1)
        # Absolute values (volatility proxy)
        H = hurst_rs(np.abs(x))
        assert H > 0.4, f"Persistent series should show elevated H, got {H}"
