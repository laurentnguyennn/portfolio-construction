"""
statistical_tests.py — Statistical inference utilities.
========================================================
Multiple testing correction (BH-FDR), long-memory detection (Hurst, GPH),
Superior Predictive Ability test, and bootstrap procedures.

References
----------
* Benjamini & Hochberg (1995) — False Discovery Rate
* Hurst (1951) — R/S analysis
* Geweke & Porter-Hudak (1983) — GPH log-periodogram estimator
* Hansen (2005) — SPA test
* Politis & Romano (1994) — Stationary block bootstrap
* Ledoit & Wolf (2008) — Bootstrap Sharpe ratio comparison
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  BENJAMINI-HOCHBERG FDR CORRECTION
# ══════════════════════════════════════════════

def benjamini_hochberg(
    pvalues: np.ndarray,
    q: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    When testing the same hypothesis across m tickers (e.g., ADF across 20
    stocks), controls the expected proportion of false discoveries ≤ q.

    Procedure (Benjamini & Hochberg, 1995):
      1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
      2. Find largest k such that p_(k) ≤ (k/m) · q
      3. Reject H₀ for all i with p_(i) ≤ p_(k)

    Parameters
    ----------
    pvalues : array (m,)
        Raw p-values from m hypothesis tests.
    q : float
        Target False Discovery Rate (default 0.05).

    Returns
    -------
    rejected : bool array (m,)
        True where H₀ is rejected after correction.
    adjusted : float array (m,)
        BH-adjusted p-values (monotonically non-decreasing when sorted).
    """
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    ranks = np.arange(1, m + 1)

    # BH threshold line: (k/m) * q
    thresholds = ranks / m * q

    # Find largest k where p_(k) <= threshold
    below = sorted_pvals <= thresholds

    if not np.any(below):
        # No rejections
        adjusted = np.minimum(sorted_pvals * m / ranks, 1.0)
        # Enforce monotonicity (backward sweep)
        for i in range(m - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        # Map back to original order
        result_adjusted = np.empty(m)
        result_adjusted[sorted_idx] = adjusted
        return np.zeros(m, dtype=bool), result_adjusted

    k = np.max(np.where(below)[0])
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[: k + 1]] = True

    # Adjusted p-values: p_adj_(i) = min(p_(i) * m / i, 1)
    adjusted = np.minimum(sorted_pvals * m / ranks, 1.0)
    # Enforce monotonicity (backward sweep)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    result_adjusted = np.empty(m)
    result_adjusted[sorted_idx] = adjusted
    return rejected, result_adjusted


# ══════════════════════════════════════════════
# 2.  HURST EXPONENT — R/S ANALYSIS
# ══════════════════════════════════════════════

def hurst_rs(series: np.ndarray, min_window: int = 20) -> float:
    """
    Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5 → long memory (persistent / trending)
    H = 0.5 → random walk (no memory)
    H < 0.5 → mean-reverting (anti-persistent)

    Used as prerequisite before fitting FIGARCH: only proceed if H > 0.5.

    Parameters
    ----------
    series : array
        Typically |r_t| or r²_t (volatility proxies) for long-memory
        detection in volatility.
    min_window : int
        Minimum sub-series length.

    Returns
    -------
    H : float
        Estimated Hurst exponent.
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    n = len(series)

    if n < min_window * 2:
        logger.warning("Series too short for Hurst R/S analysis (n=%d)", n)
        return np.nan

    # Test multiple window sizes
    window_sizes = []
    rs_values = []

    for w in range(min_window, n // 2 + 1):
        n_windows = n // w
        if n_windows < 1:
            continue

        rs_list = []
        for i in range(n_windows):
            subseries = series[i * w: (i + 1) * w]
            mean_sub = subseries.mean()
            deviations = subseries - mean_sub
            cumdev = np.cumsum(deviations)
            R = cumdev.max() - cumdev.min()
            S = subseries.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if len(rs_list) > 0:
            window_sizes.append(w)
            rs_values.append(np.mean(rs_list))

    if len(window_sizes) < 3:
        return np.nan

    # OLS on log-log: ln(R/S) = H * ln(n) + c
    log_n = np.log(window_sizes)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)

    return float(slope)


# ══════════════════════════════════════════════
# 3.  HURST EXPONENT — GPH LOG-PERIODOGRAM
# ══════════════════════════════════════════════

def hurst_gph(
    series: np.ndarray,
    bandwidth: float = 0.5,
) -> Tuple[float, float, float]:
    """
    GPH log-periodogram estimator (Geweke & Porter-Hudak, 1983).

    Estimates fractional differencing parameter d from the spectral
    density at low frequencies.

    ln(I(ω_j)) = c - d · ln(4 sin²(ω_j/2)) + ε_j

    Parameters
    ----------
    series : array
        Typically |r_t| or r²_t for detecting long memory in volatility.
    bandwidth : float
        Fraction of frequencies to use (default 0.5 → m = T^0.5).

    Returns
    -------
    d : float
        Fractional differencing parameter. d > 0 → long memory.
        d ∈ (0, 0.5) → stationary long memory.
    se : float
        Standard error of d estimate.
    p_value : float
        p-value for H₀: d = 0 (no long memory).
    """
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    T = len(series)

    if T < 50:
        return np.nan, np.nan, np.nan

    # Number of Fourier frequencies to use
    m = int(T ** bandwidth)
    m = max(m, 3)  # need at least 3 frequencies

    # Fourier frequencies: ω_j = 2πj/T for j = 1, ..., m
    j = np.arange(1, m + 1)
    omega_j = 2 * np.pi * j / T

    # Periodogram at Fourier frequencies
    fft_vals = np.fft.fft(series - series.mean())
    I_omega = (np.abs(fft_vals[1: m + 1]) ** 2) / (2 * np.pi * T)

    # Avoid log(0)
    I_omega = np.maximum(I_omega, 1e-30)

    # Regressor: ln(4 sin²(ω_j / 2))
    x = np.log(4 * np.sin(omega_j / 2) ** 2)
    y = np.log(I_omega)

    # OLS: y = c - d * x + ε  →  d = -slope
    slope, intercept, r_value, p_value_slope, se_slope = stats.linregress(x, y)
    d = -slope
    se = se_slope

    # p-value for H₀: d = 0  (two-sided t-test)
    if se > 0:
        t_stat = d / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=m - 2))
    else:
        p_value = np.nan

    return float(d), float(se), float(p_value)


# ══════════════════════════════════════════════
# 4.  STATIONARY BLOCK BOOTSTRAP
# ══════════════════════════════════════════════

def stationary_block_bootstrap(
    data: np.ndarray,
    B: int = 10_000,
    avg_block_length: float | None = None,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Stationary block bootstrap (Politis & Romano, 1994).

    Block lengths drawn from Geometric(1/q) where q = average block length.
    Preserves serial dependence structure of the original series.

    Parameters
    ----------
    data : array (T,) or (T, K)
        Original time series.
    B : int
        Number of bootstrap samples.
    avg_block_length : float or None
        Average block length q. If None, uses Politis & White (2004)
        automatic selection: q ≈ T^(1/3).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : array (B, T) or (B, T, K)
        Bootstrap resampled series.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    T = data.shape[0]

    if avg_block_length is None:
        # Politis & White (2004) rule of thumb
        avg_block_length = max(T ** (1.0 / 3.0), 2.0)

    p = 1.0 / avg_block_length  # geometric probability

    if data.ndim == 1:
        samples = np.empty((B, T))
    else:
        samples = np.empty((B, T, data.shape[1]))

    for b in range(B):
        idx = []
        pos = rng.randint(0, T)
        while len(idx) < T:
            idx.append(pos % T)
            if rng.random() < p:
                # Start new block
                pos = rng.randint(0, T)
            else:
                pos += 1
        samples[b] = data[np.array(idx[:T])]

    return samples


# ══════════════════════════════════════════════
# 5.  BOOTSTRAP SHARPE RATIO TEST
# ══════════════════════════════════════════════

def bootstrap_sharpe_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    B: int = 10_000,
    avg_block_length: float | None = None,
    seed: int = RANDOM_STATE,
) -> Dict:
    """
    Ledoit & Wolf (2008) studentized bootstrap test for Sharpe ratio
    comparison.

    Tests H₀: SR(A) = SR(B)  vs  H₁: SR(A) ≠ SR(B).

    Accounts for serial correlation and non-normality of returns
    via the stationary block bootstrap.

    Parameters
    ----------
    returns_a, returns_b : arrays (T,)
        Daily return series for two strategies.
    B : int
        Number of bootstrap replications.

    Returns
    -------
    dict with keys:
        sharpe_a, sharpe_b : observed Sharpe ratios (annualized)
        delta : sharpe_a - sharpe_b
        p_value : bootstrap p-value for H₀: delta = 0
    """
    returns_a = np.asarray(returns_a)
    returns_b = np.asarray(returns_b)
    T = len(returns_a)

    def _sharpe(r):
        if r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    sr_a = _sharpe(returns_a)
    sr_b = _sharpe(returns_b)
    delta_obs = sr_a - sr_b

    # Stack for joint bootstrap (preserve cross-sectional dependence)
    joint = np.column_stack([returns_a, returns_b])
    boot_samples = stationary_block_bootstrap(
        joint, B=B, avg_block_length=avg_block_length, seed=seed
    )

    # Center returns under H₀ (common Sharpe)
    boot_deltas = np.empty(B)
    for b in range(B):
        boot_a = boot_samples[b, :, 0]
        boot_b = boot_samples[b, :, 1]
        boot_deltas[b] = _sharpe(boot_a) - _sharpe(boot_b)

    # Center bootstrap distribution at observed delta
    boot_deltas_centered = boot_deltas - boot_deltas.mean()

    # Two-sided p-value
    p_value = np.mean(np.abs(boot_deltas_centered) >= np.abs(delta_obs))

    return {
        "sharpe_a": float(sr_a),
        "sharpe_b": float(sr_b),
        "delta": float(delta_obs),
        "p_value": float(p_value),
    }


# ══════════════════════════════════════════════
# 6.  WHITE'S REALITY CHECK / SPA TEST
# ══════════════════════════════════════════════

def spa_test(
    loss_matrix: np.ndarray,
    benchmark_col: int = 0,
    B: int = 10_000,
    block_length: float | None = None,
    seed: int = RANDOM_STATE,
) -> Dict:
    """
    Hansen (2005) Superior Predictive Ability test.

    Tests H₀: no model outperforms the benchmark, accounting for
    data-snooping across M candidate models.

    T_SPA = max_m (d̄_m / σ̂_m)
    p-value via stationary block bootstrap.

    Parameters
    ----------
    loss_matrix : array (T, M)
        Columns = loss values for each model (e.g., squared errors).
        Column ``benchmark_col`` is the benchmark model.
    benchmark_col : int
        Column index of the benchmark model.
    B : int
        Bootstrap replications.
    block_length : float or None
        Average block length for stationary bootstrap.

    Returns
    -------
    dict with keys:
        statistic : float — SPA test statistic
        p_value : float — bootstrap p-value
        best_model_idx : int — index of best-performing model
    """
    rng = np.random.RandomState(seed)
    T, M = loss_matrix.shape

    # Relative loss: d_{m,t} = L_benchmark_t - L_model_t (positive = model better)
    benchmark_loss = loss_matrix[:, benchmark_col]
    d_matrix = benchmark_loss[:, None] - loss_matrix  # (T, M)
    # Remove benchmark column from comparison
    model_cols = [i for i in range(M) if i != benchmark_col]
    d_matrix = d_matrix[:, model_cols]
    n_models = len(model_cols)

    # Observed means and standard deviations
    d_bar = d_matrix.mean(axis=0)
    d_std = d_matrix.std(axis=0, ddof=1) / np.sqrt(T)
    d_std = np.maximum(d_std, 1e-10)

    # SPA statistic: max standardized mean
    t_spa = np.max(d_bar / d_std)

    # Bootstrap: stationary block bootstrap on d_matrix
    if block_length is None:
        block_length = max(T ** (1.0 / 3.0), 2.0)

    p = 1.0 / block_length
    boot_stats = np.empty(B)

    for b in range(B):
        # Generate bootstrap indices
        idx = []
        pos = rng.randint(0, T)
        while len(idx) < T:
            idx.append(pos % T)
            if rng.random() < p:
                pos = rng.randint(0, T)
            else:
                pos += 1
        idx = np.array(idx[:T])

        boot_d = d_matrix[idx]
        boot_d_bar = boot_d.mean(axis=0) - d_bar  # center under H₀
        boot_d_std = boot_d.std(axis=0, ddof=1) / np.sqrt(T)
        boot_d_std = np.maximum(boot_d_std, 1e-10)
        boot_stats[b] = np.max(boot_d_bar / boot_d_std)

    p_value = np.mean(boot_stats >= t_spa)

    # Best model
    best_rel_idx = np.argmax(d_bar)
    best_model_idx = model_cols[best_rel_idx]

    return {
        "statistic": float(t_spa),
        "p_value": float(p_value),
        "best_model_idx": int(best_model_idx),
        "d_bar": d_bar.tolist(),
    }
