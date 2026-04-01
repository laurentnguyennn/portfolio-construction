"""
systemic_risk.py — Systemic risk measures.
===========================================
CoVaR (Adrian & Brunnermeier, 2016), Marginal Expected Shortfall
(Acharya et al., 2017), and Absorption Ratio (Kritzman et al., 2011).

These measures capture cross-asset tail dependence and market-wide
fragility — critical for understanding contagion in a concentrated
tech portfolio.

References
----------
* Adrian, T. & Brunnermeier, M. (2016) "CoVaR" — American Econ. Review
* Acharya, V. et al. (2017) "Measuring Systemic Risk" — Rev. Fin. Studies
* Kritzman, M. et al. (2011) "Principal Components as a Measure of
  Systemic Risk" — J. Portfolio Management
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  CoVaR — QUANTILE REGRESSION
# ══════════════════════════════════════════════

def covar_quantile_regression(
    portfolio_returns: pd.Series,
    asset_returns: pd.Series,
    conditioning_vars: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Adrian & Brunnermeier (2016) CoVaR via quantile regression.

    CoVaR^α_{sys|i} = VaR of portfolio conditional on asset i being
    at its VaR level.

    Estimated via quantile regression:
      q_α(r_sys | r_i, M_t) = α₀ + α₁·r_i + γ'M_t

    ΔCoVaR_i = CoVaR(r_i = VaR_i) - CoVaR(r_i = median_i)

    Parameters
    ----------
    portfolio_returns : Series
        System / portfolio return series.
    asset_returns : Series
        Individual asset return series.
    conditioning_vars : DataFrame or None
        State variables (VIX, yield spread, etc.).
    alpha : float
        Quantile level (default 0.05 for 95% VaR).

    Returns
    -------
    dict with keys:
        covar : float — CoVaR value
        delta_covar : float — ΔCoVaR (systemic risk contribution)
        covar_median : float — CoVaR at median conditions
        coefficients : dict — regression coefficients
    """
    try:
        from statsmodels.regression.quantile_regression import QuantReg
    except ImportError:
        logger.error("statsmodels required for CoVaR quantile regression")
        return {"covar": np.nan, "delta_covar": np.nan, "covar_median": np.nan}

    # Align series
    df = pd.DataFrame({
        "port": portfolio_returns,
        "asset": asset_returns,
    }).dropna()

    if conditioning_vars is not None:
        df = df.join(conditioning_vars, how="inner").dropna()

    if len(df) < 50:
        logger.warning("Insufficient data for CoVaR estimation (n=%d)", len(df))
        return {"covar": np.nan, "delta_covar": np.nan, "covar_median": np.nan}

    # Build regressors
    y = df["port"]
    X_cols = ["asset"]
    if conditioning_vars is not None:
        X_cols += [c for c in conditioning_vars.columns if c in df.columns]

    import statsmodels.api as sm
    X = sm.add_constant(df[X_cols])

    # Quantile regression at alpha
    model = QuantReg(y, X)
    result = model.fit(q=alpha, max_iter=1000)

    # CoVaR: portfolio VaR conditional on asset at its VaR
    asset_var = df["asset"].quantile(alpha)
    asset_median = df["asset"].median()

    # Prediction at asset's VaR level
    x_var = np.array([1.0, asset_var] + [df[c].mean() for c in X_cols[1:]])
    x_med = np.array([1.0, asset_median] + [df[c].mean() for c in X_cols[1:]])

    covar = float(result.predict(x_var.reshape(1, -1))[0])
    covar_median = float(result.predict(x_med.reshape(1, -1))[0])
    delta_covar = covar - covar_median

    return {
        "covar": covar,
        "delta_covar": delta_covar,
        "covar_median": covar_median,
        "coefficients": dict(zip(["const"] + X_cols, result.params)),
    }


# ══════════════════════════════════════════════
# 2.  MARGINAL EXPECTED SHORTFALL (MES)
# ══════════════════════════════════════════════

def mes(
    returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Marginal Expected Shortfall per asset.

    MES_i = E[r_i | r_portfolio ≤ VaR_α(portfolio)]

    Key property: Portfolio CVaR = Σ_i w_i · MES_i  (exact decomposition).
    This allows attribution of tail risk to individual positions.

    Parameters
    ----------
    returns : DataFrame (T, N)
        Asset return matrix.
    weights : array (N,)
        Portfolio weights.
    alpha : float
        Tail probability (default 0.05 → worst 5%).

    Returns
    -------
    mes_values : array (N,)
        MES for each asset (negative = expected loss in tail).
        Portfolio CVaR = -Σ w_i · MES_i.
    """
    port_ret = returns.values @ weights
    threshold = np.quantile(port_ret, alpha)
    tail_mask = port_ret <= threshold

    if tail_mask.sum() == 0:
        return np.full(returns.shape[1], np.nan)

    return returns[tail_mask].mean().values


# ══════════════════════════════════════════════
# 3.  ABSORPTION RATIO
# ══════════════════════════════════════════════

def absorption_ratio(
    returns: pd.DataFrame,
    k: int = 4,
    window: int = 252,
) -> pd.Series:
    """
    Kritzman et al. (2011) rolling absorption ratio.

    AR_t = Σ_{j=1}^k λ_j / Σ_{j=1}^N λ_j

    where λ_j = eigenvalues of the rolling correlation matrix,
    k = N/5 (top 4 eigenvalues for N=20 stocks).

    High AR → tightly coupled markets → systemic fragility.
    Sharp AR increases precede market turbulence (leading indicator).
    Typical range: 0.5–0.9 for equity markets.

    Parameters
    ----------
    returns : DataFrame (T, N)
        Asset return matrix.
    k : int
        Number of top eigenvalues to sum (default 4 = N/5 for N=20).
    window : int
        Rolling window for correlation estimation (default 252).

    Returns
    -------
    ar_series : Series
        Rolling absorption ratio, indexed from returns.index[window:].
    """
    values = returns.values
    T, N = values.shape

    if k > N:
        k = N
        logger.warning("k > N, setting k = N = %d", N)

    ar_values = []
    ar_dates = []

    for t in range(window, T):
        window_data = values[t - window: t]

        # Correlation matrix (more stable than covariance for AR)
        # np.corrcoef already normalizes internally, no need to pre-normalize
        corr = np.corrcoef(window_data, rowvar=False)

        # Guard: NaN in correlation (e.g. constant asset in window)
        if np.any(np.isnan(corr)):
            logger.debug(
                "absorption_ratio: NaN correlation at %s "
                "(possible constant or near-constant asset in window)",
                returns.index[t],
            )
            ar_values.append(np.nan)
            ar_dates.append(returns.index[t])
            continue

        eigenvalues = np.linalg.eigvalsh(corr)[::-1]  # descending order
        total_var = eigenvalues.sum()

        # Guard: near-singular correlation (rank-deficient window)
        if total_var < 1e-10:
            logger.debug(
                "absorption_ratio: near-zero total variance at %s "
                "(rank-deficient correlation matrix — window too short?)",
                returns.index[t],
            )
            ar_values.append(np.nan)
        else:
            ar_values.append(eigenvalues[:k].sum() / total_var)

        ar_dates.append(returns.index[t])

    return pd.Series(ar_values, index=ar_dates, name="absorption_ratio")
