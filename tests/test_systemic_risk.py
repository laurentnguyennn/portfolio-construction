"""
test_systemic_risk.py — Tests for systemic risk measures.
=========================================================
CoVaR, Marginal Expected Shortfall, and Absorption Ratio.
"""

import numpy as np
import pandas as pd
import pytest

from src.systemic_risk import absorption_ratio, covar_quantile_regression, mes


# ══════════════════════════════════════════════
# 1.  ABSORPTION RATIO
# ══════════════════════════════════════════════

class TestAbsorptionRatio:
    """Tests for Kritzman et al. (2011) Absorption Ratio."""

    def test_bounds(self):
        """AR must be between 0 and 1."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(500, 20),
            columns=[f"S{i}" for i in range(20)],
        )
        ar = absorption_ratio(returns, k=4, window=252)
        assert len(ar) > 0
        assert (ar.dropna() > 0).all()
        assert (ar.dropna() < 1).all()

    def test_perfectly_correlated(self):
        """Perfectly correlated assets should give AR close to 1/N * k ≈ k/N."""
        np.random.seed(42)
        base = np.random.randn(400)
        # All assets are the same (perfect correlation)
        returns = pd.DataFrame(
            np.column_stack([base + np.random.randn(400) * 0.01 for _ in range(10)]),
            columns=[f"S{i}" for i in range(10)],
        )
        ar = absorption_ratio(returns, k=2, window=300)
        # With near-perfect correlation, first eigenvalue dominates
        assert ar.iloc[-1] > 0.7

    def test_independent_assets(self):
        """Independent assets should have lower AR."""
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(500, 10),
            columns=[f"S{i}" for i in range(10)],
        )
        ar = absorption_ratio(returns, k=2, window=252)
        # For independent assets, eigenvalues are roughly equal
        # AR ≈ k/N = 2/10 = 0.2
        assert ar.iloc[-1] < 0.5

    def test_output_length(self):
        """Output length should be T - window."""
        np.random.seed(42)
        T = 400
        returns = pd.DataFrame(np.random.randn(T, 5))
        ar = absorption_ratio(returns, k=2, window=252)
        assert len(ar) == T - 252


# ══════════════════════════════════════════════
# 2.  MARGINAL EXPECTED SHORTFALL
# ══════════════════════════════════════════════

class TestMES:
    """Tests for Acharya et al. (2017) MES."""

    def test_cvar_decomposition(self):
        """
        Key property: Portfolio CVaR = −Σ_i w_i · MES_i  (exact decomposition).

        mes() returns E[r_i | r_portfolio ≤ VaR] which is NEGATIVE (returns are
        losses in the tail).  Portfolio CVaR (positive loss number) equals
        −Σ w_i · MES_i, i.e. the negative of the weighted sum.
        """
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(2000, 5))
        w = np.array([0.2] * 5)

        port_ret = returns.values @ w
        alpha = 0.05
        var_threshold = np.quantile(port_ret, alpha)
        tail_mask = port_ret <= var_threshold
        # CVaR as a positive loss number: −E[r_p | r_p ≤ VaR]
        port_cvar = -port_ret[tail_mask].mean()

        mes_values = mes(returns, w, alpha=alpha)
        # MES values are negative (average returns in the left tail).
        # Decomposition: CVaR_portfolio = −Σ w_i · MES_i
        reconstructed = -float(np.sum(w * mes_values))

        assert abs(port_cvar - reconstructed) < 0.05, (
            f"CVaR decomposition failed: port_cvar={port_cvar:.4f}, "
            f"reconstructed={reconstructed:.4f}\n"
            f"MES values: {mes_values}"
        )

    def test_output_shape(self):
        """MES should return one value per asset."""
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(500, 10))
        w = np.ones(10) / 10
        mes_vals = mes(returns, w)
        assert len(mes_vals) == 10

    def test_negative_in_tail(self):
        """MES values should generally be negative (losses in tail)."""
        np.random.seed(42)
        returns = pd.DataFrame(np.random.randn(2000, 5) * 0.02)
        w = np.ones(5) / 5
        mes_vals = mes(returns, w, alpha=0.05)
        # Most MES values should be negative (portfolio is in left tail)
        assert np.mean(mes_vals) < 0


# ══════════════════════════════════════════════
# 3.  CoVaR
# ══════════════════════════════════════════════

class TestCoVaR:
    """Tests for Adrian & Brunnermeier (2016) CoVaR."""

    def test_returns_valid_dict(self):
        """CoVaR should return a dict with expected keys."""
        np.random.seed(42)
        n = 1000
        portfolio = pd.Series(np.random.randn(n) * 0.02, name="portfolio")
        asset = pd.Series(np.random.randn(n) * 0.03, name="asset")

        result = covar_quantile_regression(portfolio, asset, alpha=0.05)
        assert "covar" in result
        assert "delta_covar" in result
        assert "covar_median" in result

    def test_delta_covar_sign(self):
        """
        ΔCoVaR should generally be negative for positively correlated assets.
        (When asset is at its VaR, system VaR is worse than at median.)
        """
        np.random.seed(42)
        n = 2000
        common_factor = np.random.randn(n) * 0.02
        portfolio = pd.Series(common_factor + np.random.randn(n) * 0.01)
        asset = pd.Series(common_factor + np.random.randn(n) * 0.015)

        result = covar_quantile_regression(portfolio, asset, alpha=0.05)
        if not np.isnan(result["delta_covar"]):
            # ΔCoVaR should be negative (worse VaR when asset is stressed)
            assert result["delta_covar"] < 0

    def test_short_series(self):
        """Short series should return NaN gracefully."""
        portfolio = pd.Series([0.01, -0.02, 0.005])
        asset = pd.Series([0.02, -0.01, 0.003])
        result = covar_quantile_regression(portfolio, asset)
        assert np.isnan(result["covar"])
