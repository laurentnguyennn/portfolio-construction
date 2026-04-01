"""
dcc_garch.py — Custom DCC-GARCH implementation (Engle 2002).
=============================================================
The Python ``arch`` library supports only **univariate** GARCH.
DCC correlation dynamics are estimated separately on the standardized
residuals using the Engle (2002) two-step procedure:

    Step 1: Fit univariate GARCH on each asset → standardized residuals ε_t
    Step 2: Estimate DCC parameters (a, b) on the multivariate ε_t

DCC model
---------
    Q_t = (1 − a − b) · Q̄  +  a · (ε_{t-1} ε_{t-1}')  +  b · Q_{t-1}
    R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

where Q̄ is the unconditional correlation of ε_t.

Reference
---------
Engle, R. (2002). "Dynamic Conditional Correlation."
    *Journal of Business & Economic Statistics*, 20(3), 339–350.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.garch_utils import fit_garch, select_best_distribution

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1.  Step 1 — Univariate GARCH → standardized residuals
# ──────────────────────────────────────────────

def get_standardized_residuals(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit the best univariate GARCH to each column of ``returns``,
    extract standardized residuals.

    Parameters
    ----------
    returns : DataFrame  (T × N)
        Daily log-return panel (one column per ticker).

    Returns
    -------
    std_resids : DataFrame  (T × N)
        Standardized residuals aligned by date.
    """
    resids: Dict[str, pd.Series] = {}
    for col in returns.columns:
        rets = returns[col].dropna()
        try:
            best_dist, result = select_best_distribution(
                rets, vol="GARCH", p=1, o=1, q=1
            )
            sr = result.std_resid
            sr.name = col
            resids[col] = sr
        except RuntimeError:
            logger.warning("DCC step-1: could not fit GARCH for %s", col)
    return pd.DataFrame(resids).dropna()


# ──────────────────────────────────────────────
# 2.  Step 2 — DCC parameter estimation
# ──────────────────────────────────────────────

def _dcc_loglikelihood(
    params: np.ndarray,
    eps: np.ndarray,
    Q_bar: np.ndarray,
) -> float:
    """
    Negative log-likelihood for the DCC model (to be minimised).

    Parameters
    ----------
    params : array [a, b]
        DCC dynamics parameters.  Constraints: a > 0, b > 0, a + b < 1.
    eps : ndarray (T, N)
        Standardized residuals.
    Q_bar : ndarray (N, N)
        Unconditional correlation matrix of eps.

    Returns
    -------
    neg_ll : float
        Negative (quasi) log-likelihood, summed over t.

    Notes
    -----
    The likelihood ignores the constant term and the univariate part
    (already accounted for in step 1).

    L_DCC = −0.5 Σ_t [ ln|R_t| + ε_t' R_t^{-1} ε_t − ε_t' ε_t ]
    """
    a, b = params
    T, N = eps.shape

    # Initialise Q_0 = Q_bar
    Q_t = Q_bar.copy()
    neg_ll = 0.0

    for t in range(1, T):
        # Q_t update
        eps_prev = eps[t - 1].reshape(-1, 1)
        Q_t = (1 - a - b) * Q_bar + a * (eps_prev @ eps_prev.T) + b * Q_t

        # Correlation matrix R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_t), 1e-12)))
        R_t = diag_inv_sqrt @ Q_t @ diag_inv_sqrt

        # Numerical safety
        R_t = (R_t + R_t.T) / 2.0
        np.fill_diagonal(R_t, 1.0)

        # Log-likelihood contribution
        try:
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0:
                return 1e12
            R_inv = np.linalg.solve(R_t, eps[t])
            neg_ll += 0.5 * (logdet + eps[t] @ R_inv - eps[t] @ eps[t])
        except np.linalg.LinAlgError:
            return 1e12

    return neg_ll


def estimate_dcc_params(
    std_resids: np.ndarray,
    Q_bar: np.ndarray,
) -> Tuple[float, float]:
    """
    Maximise DCC log-likelihood to estimate (a, b).

    Constraints: a > 0, b > 0, a + b < 1 (stationarity).

    Returns
    -------
    a, b : floats
    """
    # Tighter bound: a + b ≤ 0.95 (well inside stationary region).
    # 0.999 was too loose — near-unit-root DCC produces explosive correlations.
    # Individual bounds: a ∈ (0, 0.20), b ∈ (0, 0.94) to represent realistic
    # news-impact (a) and decay (b) parameters.
    bounds = [(1e-6, 0.20), (1e-6, 0.94)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.95 - p[0] - p[1]}]

    result = minimize(
        _dcc_loglikelihood,
        x0=np.array([0.01, 0.90]),
        args=(std_resids, Q_bar),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if not result.success:
        logger.warning("DCC optimisation did not converge: %s", result.message)

    a, b = result.x
    logger.info("DCC params: a=%.6f, b=%.6f (a+b=%.6f)", a, b, a + b)
    return a, b


# ──────────────────────────────────────────────
# 3.  Dynamic correlation path
# ──────────────────────────────────────────────

def compute_dcc_correlations(
    std_resids: np.ndarray,
    a: float,
    b: float,
    Q_bar: np.ndarray,
) -> np.ndarray:
    """
    Given estimated DCC params, compute the full path of R_t.

    Returns
    -------
    R : ndarray (T, N, N)
        Time-varying correlation matrices.
    """
    T, N = std_resids.shape
    R = np.zeros((T, N, N))
    Q_t = Q_bar.copy()

    for t in range(T):
        if t > 0:
            eps_prev = std_resids[t - 1].reshape(-1, 1)
            Q_t = (1 - a - b) * Q_bar + a * (eps_prev @ eps_prev.T) + b * Q_t

        diag_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_t), 1e-12)))
        R_t = diag_inv @ Q_t @ diag_inv
        R_t = (R_t + R_t.T) / 2.0
        np.fill_diagonal(R_t, 1.0)
        R[t] = R_t

    return R


# ──────────────────────────────────────────────
# 4.  Dynamic covariance from DCC-R + univariate vols
# ──────────────────────────────────────────────

def dcc_covariance(
    cond_vol_panel: pd.DataFrame,
    dcc_corr: np.ndarray,
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    H_t = D_t · R_t · D_t

    where D_t = diag(σ_{1,t}, ..., σ_{N,t}) from univariate GARCH.

    Parameters
    ----------
    cond_vol_panel : DataFrame (T × N)
        Annualized conditional volatilities.
    dcc_corr : ndarray (T, N, N)
        DCC correlation matrices.
    dates : DatetimeIndex
        Corresponding date index.

    Returns
    -------
    H : ndarray (T, N, N)
        Dynamic covariance matrices.
    """
    # Convert annualized vol to daily
    vols = cond_vol_panel.values / np.sqrt(252)
    T, N = vols.shape
    H = np.zeros((T, N, N))

    for t in range(T):
        D_t = np.diag(vols[t])
        H[t] = D_t @ dcc_corr[t] @ D_t

    return H


# ──────────────────────────────────────────────
# 5.  Convenience: full DCC-GARCH pipeline
# ──────────────────────────────────────────────

def run_dcc_garch(
    returns: pd.DataFrame,
    cond_vol_panel: Optional[pd.DataFrame] = None,
) -> Tuple:
    """
    End-to-end DCC-GARCH (Engle 2002, two-step estimation).

    Step 1: Fit univariate GARCH per asset → standardized residuals.
    Step 2: Estimate DCC parameters (a, b) on the multivariate residuals.
    Step 3: Build full dynamic correlation path R_t (T × N × N).
    Step 4: If ``cond_vol_panel`` provided, compute H_t = D_t R_t D_t.

    Parameters
    ----------
    returns : DataFrame (T × N)
        Daily log returns.
    cond_vol_panel : DataFrame (T × N), optional
        Annualized conditional volatilities from univariate GARCH (NB03).
        If provided, the function returns dynamic covariance matrices H_t.

    Returns
    -------
    Without cond_vol_panel:
        R_path : ndarray (T, N, N), a : float, b : float, dates : DatetimeIndex

    With cond_vol_panel:
        R_path, H_path, a, b, dates
        where H_path : ndarray (T, N, N) = dynamic covariance matrices
    """
    # Step 1
    std_df = get_standardized_residuals(returns)
    eps = std_df.values
    dates = std_df.index

    # Unconditional correlation of standardized residuals
    Q_bar = np.corrcoef(eps, rowvar=False)

    # Step 2
    a, b = estimate_dcc_params(eps, Q_bar)

    # Step 3 — full correlation path
    R_path = compute_dcc_correlations(eps, a, b, Q_bar)

    # Step 4 — optionally build covariance matrices H_t = D_t R_t D_t
    if cond_vol_panel is not None:
        # Align vol panel to the same dates as std_df (inner join)
        vol_aligned = cond_vol_panel.reindex(dates).ffill().dropna()
        common_dates = dates.intersection(vol_aligned.index)
        if len(common_dates) < len(dates):
            logger.warning(
                "cond_vol_panel aligned to %d dates (from %d); "
                "missing vol dates forward-filled",
                len(common_dates), len(dates),
            )
        # Map R_path to common dates
        date_mask = np.isin(dates, common_dates)
        R_aligned = R_path[date_mask]
        H_path = dcc_covariance(vol_aligned.loc[common_dates], R_aligned, common_dates)
        return R_path, H_path, a, b, dates

    return R_path, a, b, dates
