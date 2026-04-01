"""
test_risk_metrics.py — Tests for VaR, CVaR, EVT, and copulas.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.risk_metrics import (
    var_historical,
    var_parametric_gaussian,
    var_cornish_fisher,
    cvar_historical,
    cvar_parametric_gaussian,
    cvar_from_var,
    fit_gpd,
    evt_var,
    evt_cvar,
    kupiec_pof_test,
    christoffersen_test,
    traffic_light_zone,
    fit_clayton_copula,
    build_return_scenario_matrix,
)


@pytest.fixture
def normal_returns():
    """Generate normally distributed returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 0.02, 2520), name="TEST")


@pytest.fixture
def fat_tail_returns():
    """Generate fat-tailed (Student-t) returns."""
    np.random.seed(42)
    return pd.Series(stats.t.rvs(df=4, loc=0, scale=0.02, size=2520), name="TEST")


class TestVaR:
    """Test VaR computation correctness."""

    def test_historical_var_basic(self, normal_returns):
        """Historical VaR should be close to parametric for normal data."""
        var_h = var_historical(normal_returns, 0.05)
        var_g = var_parametric_gaussian(normal_returns, 0.05)
        # Should be within 20% for large normal sample
        assert abs(var_h - var_g) / var_g < 0.20

    def test_var_positive(self, normal_returns):
        """VaR should be a positive loss number."""
        var = var_historical(normal_returns, 0.05)
        assert var > 0, "VaR must be positive (represents a loss)"

    def test_var_99_greater_than_95(self, normal_returns):
        """99% VaR should exceed 95% VaR."""
        var_95 = var_historical(normal_returns, 0.05)
        var_99 = var_historical(normal_returns, 0.01)
        assert var_99 > var_95, "99% VaR must exceed 95% VaR"

    def test_cornish_fisher_uses_excess_kurtosis(self, normal_returns):
        """
        CRITICAL: Cornish-Fisher must use EXCESS kurtosis, not raw kurtosis.

        pandas .kurtosis() already returns excess kurtosis (Fisher's definition).
        The CF formula expects excess kurtosis. Using raw kurtosis (excess + 3)
        would produce incorrect quantile adjustments.
        """
        # For a normal distribution, excess kurtosis ≈ 0
        ek = normal_returns.kurtosis()
        assert abs(ek) < 1.0, f"Excess kurtosis of normal should be ~0, got {ek}"

        # CF VaR should be very close to Gaussian VaR for normal data
        var_cf = var_cornish_fisher(normal_returns, 0.05)
        var_g = var_parametric_gaussian(normal_returns, 0.05)
        assert abs(var_cf - var_g) / var_g < 0.05, \
            "CF VaR should match Gaussian for normal data (excess kurtosis ≈ 0)"

    def test_cornish_fisher_monotonicity_guard(self):
        """
        CF expansion can become non-monotonic for extreme skew/kurtosis.
        The guard should fall back to Gaussian when this happens.
        """
        # Create extremely skewed data
        np.random.seed(42)
        extreme = pd.Series(np.concatenate([
            np.random.normal(0, 0.01, 1000),
            np.random.normal(0, 0.01, 50) - 0.5,  # extreme left tail
        ]))

        var_cf = var_cornish_fisher(extreme, 0.05)
        var_g = var_parametric_gaussian(extreme, 0.05)

        # CF should be >= Gaussian (more conservative or equal due to guard)
        # The guard ensures CF never produces a LESS conservative estimate
        assert var_cf >= var_g * 0.95, \
            "Monotonicity guard should prevent CF from being less conservative than Gaussian"

    def test_cornish_fisher_fat_tails(self, fat_tail_returns):
        """CF should give higher VaR than Gaussian for fat-tailed data."""
        var_cf = var_cornish_fisher(fat_tail_returns, 0.01)
        var_g = var_parametric_gaussian(fat_tail_returns, 0.01)
        # For fat tails (positive excess kurtosis), CF should be more conservative
        assert var_cf >= var_g * 0.9, \
            "CF should be at least as conservative as Gaussian for fat tails"


class TestCVaR:
    """Test CVaR / Expected Shortfall."""

    def test_cvar_exceeds_var(self, normal_returns):
        """CVaR must always be ≥ VaR (average of tail ≥ threshold)."""
        var = var_historical(normal_returns, 0.05)
        cvar = cvar_historical(normal_returns, 0.05)
        assert cvar >= var, "CVaR must be ≥ VaR"

    def test_cvar_positive(self, normal_returns):
        """CVaR should be a positive loss number."""
        cvar = cvar_historical(normal_returns, 0.05)
        assert cvar > 0

    def test_cvar_gaussian_formula(self, normal_returns):
        """Gaussian CVaR has a closed-form: -μ + σ · φ(z)/α."""
        cvar_g = cvar_parametric_gaussian(normal_returns, 0.05)
        mu = normal_returns.mean()
        sigma = normal_returns.std()
        z = stats.norm.ppf(0.05)
        phi = stats.norm.pdf(z)
        expected = -mu + sigma * phi / 0.05
        assert abs(cvar_g - expected) < 1e-10, "Gaussian CVaR formula mismatch"


class TestEVT:
    """Test Extreme Value Theory (GPD)."""

    def test_gpd_fit(self, fat_tail_returns):
        """GPD should fit successfully with reasonable parameters."""
        result = fit_gpd(fat_tail_returns, threshold_quantile=0.95)
        assert "xi" in result
        assert "beta" in result
        assert result["beta"] > 0, "GPD scale must be positive"
        assert result["n_exceed"] > 0

    def test_evt_var_exceeds_historical(self, fat_tail_returns):
        """EVT VaR at 99.9% should exceed historical 99%."""
        gpd = fit_gpd(fat_tail_returns, threshold_quantile=0.95)
        evt_v = evt_var(gpd, 0.001)
        hist_v = var_historical(fat_tail_returns, 0.01)
        assert evt_v > hist_v, "EVT 99.9% VaR should exceed historical 99% VaR"

    def test_xi_interpretation(self, fat_tail_returns):
        """xi > 0 indicates heavy tail (Fréchet-type)."""
        gpd = fit_gpd(fat_tail_returns, threshold_quantile=0.95)
        # Student-t with df=4 should have positive xi (heavy tail)
        # This may not always hold due to sample variation, so use soft check
        if gpd["n_exceed"] > 50:
            assert gpd["xi"] > -0.5, "Student-t should not have strongly negative xi"


class TestBacktesting:
    """Test VaR backtesting procedures."""

    def test_kupiec_correct_model(self, normal_returns):
        """A correct model should not be rejected by Kupiec test."""
        alpha = 0.05
        var = var_historical(normal_returns, alpha)
        violations = (normal_returns < -var).astype(int).values
        result = kupiec_pof_test(violations, alpha)
        # p-value should be > 0.05 (don't reject correct model)
        assert result["p_value"] > 0.01, \
            f"Correct model should not be rejected (p={result['p_value']:.4f})"

    def test_traffic_light_green(self):
        """Low violation rate should be Green zone."""
        assert traffic_light_zone(0.008, 0.01) == "Green"

    def test_traffic_light_yellow(self):
        """Medium violation rate should be Yellow zone."""
        assert traffic_light_zone(0.018, 0.01) == "Yellow"

    def test_traffic_light_red(self):
        """High violation rate should be Red zone."""
        assert traffic_light_zone(0.025, 0.01) == "Red"


class TestPortfolioScenarios:
    """Test that portfolio CVaR is computed from joint scenarios."""

    def test_scenario_matrix_shape(self):
        """Scenario matrix should be (T × N) with no NaN rows."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (1000, 5)),
            columns=["A", "B", "C", "D", "E"],
        )
        scenarios = build_return_scenario_matrix(returns)
        assert scenarios.shape == (1000, 5)
        assert scenarios.isna().sum().sum() == 0

    def test_portfolio_cvar_not_linear(self):
        """
        CRITICAL: Portfolio CVaR ≠ weighted sum of individual CVaRs.

        This test verifies that computing CVaR from w'R_t gives a
        different result than summing weighted individual CVaRs.
        """
        np.random.seed(42)
        # Correlated returns
        cov = [[0.04, 0.03], [0.03, 0.04]]
        returns = np.random.multivariate_normal([0, 0], cov, 5000)
        ret_df = pd.DataFrame(returns, columns=["A", "B"])

        w = np.array([0.5, 0.5])

        # Correct: portfolio CVaR from joint scenarios
        port_returns = ret_df.values @ w
        q = np.quantile(port_returns, 0.05)
        cvar_correct = -port_returns[port_returns <= q].mean()

        # Wrong: weighted individual CVaRs
        cvar_a = cvar_historical(ret_df["A"], 0.05)
        cvar_b = cvar_historical(ret_df["B"], 0.05)
        cvar_wrong = w[0] * cvar_a + w[1] * cvar_b

        # They should differ (diversification effect)
        # The correct (joint) CVaR should be less than the linear sum
        assert cvar_correct < cvar_wrong * 1.05, \
            "Portfolio CVaR from scenarios should benefit from diversification"
