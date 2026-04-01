"""
risk_metrics.py — VaR, CVaR, EVT (GPD), Copulas, CAViaR.
==========================================================
Implements 5 VaR methods, CVaR for each, Extreme Value Theory via POT/GPD,
Clayton & Gumbel copulas for joint tail risk, and backtesting (Kupiec, Christoffersen).

CRITICAL: Portfolio CVaR is NOT linearly aggregable from per-ticker CVaR.
It must be computed from joint portfolio return scenarios (see NB11).

References
----------
* Cornish & Fisher (1937)
* Engle & Manganelli (2004) — CAViaR
* Pickands (1975), Balkema & de Haan (1974) — EVT/GPD
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats, optimize

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 1.  VALUE-AT-RISK (5 methods)
# ══════════════════════════════════════════════

def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical Simulation VaR.

    Non-parametric: VaR_α = −Quantile(returns, α).
    The negative sign makes VaR a positive loss number.
    """
    return -np.quantile(returns.dropna(), alpha)


def var_parametric_gaussian(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Parametric Gaussian VaR.

    VaR_α = −(μ + z_α · σ)
    where z_α = Φ^{-1}(α).

    Expected to underperform for fat-tailed distributions.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(alpha)
    return -(mu + z * sigma)


def var_cornish_fisher(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    Cornish-Fisher VaR with skewness and *excess* kurtosis correction.

    The CF expansion adjusts the Gaussian quantile z_α:

        z_CF = z + (z² − 1)·S/6 + (z³ − 3z)·K/24 − (2z³ − 5z)·S²/36

    where S = skewness, K = EXCESS kurtosis (kurtosis − 3).

    **Monotonicity guard**: For extreme skewness/kurtosis the polynomial
    can become non-monotonic.  We check that the adjusted quantile is
    more extreme than the Gaussian quantile; if not, fall back to Gaussian.
    """
    mu = returns.mean()
    sigma = returns.std()
    s = returns.skew()
    # CRITICAL: use excess kurtosis (pandas .kurtosis() already returns excess)
    k = returns.kurtosis()

    z = stats.norm.ppf(alpha)

    # Full Cornish-Fisher expansion (Cornish & Fisher 1937):
    #   z_CF = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36
    # The last term has a NEGATIVE sign (standard quantile expansion).
    # Matches the formula in CLAUDE.md and Maillard (2012).
    z_cf = (z
            + (z ** 2 - 1) * s / 6.0
            + (z ** 3 - 3 * z) * k / 24.0
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36.0)

    # Monotonicity guard: for the LEFT tail (alpha < 0.5) z < 0 and z_cf should
    # be more negative (more extreme) than z. If the polynomial yields z_cf > z,
    # the expansion is non-monotonic (occurs for extreme S or K) — fall back.
    if z_cf > z:
        logger.warning(
            "Cornish-Fisher non-monotonic at alpha=%.3f "
            "(z_CF=%.4f > z=%.4f, skew=%.2f, excess_kurt=%.2f); "
            "falling back to Gaussian VaR",
            alpha, z_cf, z, s, k,
        )
        z_cf = z

    return -(mu + z_cf * sigma)


def var_garch(
    cond_vol_today: float,
    cond_mean_today: float = 0.0,
    alpha: float = 0.05,
    dist: str = "normal",
    df: float = 5.0,
) -> float:
    """
    GARCH-based conditional VaR.

    VaR_α = −(μ_t + z_α(dist) · σ_t)

    where σ_t is the conditional volatility from a fitted GARCH model.
    """
    if dist == "normal":
        z = stats.norm.ppf(alpha)
    elif dist == "t":
        z = stats.t.ppf(alpha, df=df)
    else:
        z = stats.norm.ppf(alpha)   # fallback
    return -(cond_mean_today + z * cond_vol_today)


def var_caviar_symmetric(
    returns: pd.Series,
    alpha: float = 0.05,
    max_iter: int = 500,
) -> pd.Series:
    """
    CAViaR Symmetric Absolute Value model (Engle & Manganelli 2004).

    q_t = β_0 + β_1 · q_{t-1} + β_2 · |r_{t-1}|

    The quantile q_t evolves autoregressively.  Estimated by
    minimising the quantile regression loss (check function):

        ρ_α(r_t − q_t) = (α − 1(r_t < q_t)) · (r_t − q_t)

    Returns
    -------
    VaR_series : pd.Series
        Time-varying VaR (positive = loss).
    """
    r = returns.dropna().values
    T = len(r)

    # Initial VaR estimate from historical quantile
    q_init = -np.quantile(r, alpha)

    def caviar_path(params):
        b0, b1, b2 = params
        q = np.zeros(T)
        q[0] = q_init
        for t in range(1, T):
            q[t] = b0 + b1 * q[t - 1] + b2 * abs(r[t - 1])
        return q

    def quantile_loss(params):
        q = caviar_path(params)
        # q is VaR (positive = loss), -r are losses
        # Check function: ρ_α(u) = u·(α - I(u<0))
        # where u = (-r) - q = loss - VaR
        resid = -r - q
        hit = (resid > 0).astype(float)  # 1 when loss exceeds VaR
        loss = np.mean((alpha - (1 - hit)) * resid)
        return loss

    result = optimize.minimize(
        quantile_loss,
        x0=[0.01, 0.9, 0.1],
        method="Nelder-Mead",
        options={"maxiter": max_iter},
    )

    q_path = caviar_path(result.x)
    idx = returns.dropna().index
    return pd.Series(q_path, index=idx, name="CAViaR_VaR")


# ══════════════════════════════════════════════
# 2.  EXPECTED SHORTFALL (CVaR)
# ══════════════════════════════════════════════

def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    CVaR (Expected Shortfall) from historical simulation.

    ES_α = −E[r | r < −VaR_α] = mean of losses beyond VaR.
    """
    q = np.quantile(returns.dropna(), alpha)
    tail = returns[returns <= q]
    return -tail.mean()


def cvar_parametric_gaussian(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Gaussian CVaR.

    ES_α = −μ + σ · φ(z_α) / α

    where φ is the standard normal PDF and z_α = Φ^{-1}(α).
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(alpha)
    phi = stats.norm.pdf(z)
    return -mu + sigma * phi / alpha


def cvar_from_var(returns: pd.Series, var_value: float) -> float:
    """
    Generic CVaR: mean of returns below −VaR threshold.

    Works for any VaR method: just pass the VaR estimate.
    """
    threshold = -var_value
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return var_value  # conservative fallback
    return -tail.mean()


# ══════════════════════════════════════════════
# 3.  EXTREME VALUE THEORY — POT / GPD
# ══════════════════════════════════════════════

def fit_gpd(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
) -> Dict:
    """
    Peaks-Over-Threshold (POT) with Generalized Pareto Distribution.

    Fit GPD to exceedances above the threshold.

    Parameters
    ----------
    returns : pd.Series
        Loss series (positive = loss).  Pass ``-returns`` for left-tail analysis.
    threshold_quantile : float
        Quantile of losses to use as threshold (e.g. 0.95 = top 5% of losses).

    Returns
    -------
    dict with: xi (shape), beta (scale), threshold, n_exceed, n_total.

    Notes
    -----
    xi > 0 → heavy tail (Fréchet-type); xi = 0 → exponential tail;
    xi < 0 → bounded tail (Weibull-type).
    """
    losses = (-returns).dropna()
    u = np.quantile(losses, threshold_quantile)
    exceedances = losses[losses > u] - u

    if len(exceedances) < 20:
        logger.warning("Only %d exceedances above threshold; GPD fit unreliable",
                       len(exceedances))

    # Fit GPD via MLE
    xi, loc, beta = stats.genpareto.fit(exceedances, floc=0)

    return {
        "xi": xi,
        "beta": beta,
        "threshold": u,
        "n_exceed": len(exceedances),
        "n_total": len(losses),
    }


def evt_var(gpd_params: Dict, alpha: float = 0.01) -> float:
    """
    EVT-based VaR at level α.

    VaR_α = u + (β/ξ) · [(nα/N_u)^{−ξ} − 1]

    where u = threshold, β = scale, ξ = shape, n = total obs,
    N_u = number of exceedances, α = tail probability (e.g. 0.01).
    """
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]
    n = gpd_params["n_total"]
    n_u = gpd_params["n_exceed"]

    if abs(xi) < 1e-10:
        # Exponential case (ξ → 0 limit): VaR = u − β·ln(nα/N_u)
        return u - beta * np.log(n * alpha / n_u)

    return u + (beta / xi) * ((n * alpha / n_u) ** (-xi) - 1)


def evt_cvar(gpd_params: Dict, alpha: float = 0.01) -> float:
    """
    EVT-based CVaR.

    ES_α = VaR_α / (1 − ξ)  +  (β − ξ·u) / (1 − ξ)

    Valid only for ξ < 1.
    """
    xi = gpd_params["xi"]
    beta = gpd_params["beta"]
    u = gpd_params["threshold"]
    var = evt_var(gpd_params, alpha)

    if xi >= 1:
        logger.warning("xi >= 1 (%.3f); EVT CVaR is infinite", xi)
        return np.inf

    return var / (1 - xi) + (beta - xi * u) / (1 - xi)


# ══════════════════════════════════════════════
# 4.  COPULA-BASED JOINT TAIL RISK
# ══════════════════════════════════════════════

def fit_clayton_copula(u: np.ndarray, v: np.ndarray) -> float:
    """
    Fit Clayton copula parameter θ via maximum pseudo-likelihood.

    Clayton captures lower-tail dependence:
    λ_L = 2^{−1/θ}     (θ > 0)

    Parameters
    ----------
    u, v : arrays of pseudo-observations in [0, 1]
        Obtained from the empirical CDF (probability integral transform).

    Returns
    -------
    theta : float
        Clayton parameter (> 0).
    """
    def neg_ll(theta):
        if theta <= 0:
            return 1e12
        n = len(u)
        term1 = n * np.log(1 + theta)
        term2 = -(1 + theta) * (np.log(u) + np.log(v)).sum()
        term3 = -(2 + 1.0 / theta) * np.log(
            np.maximum(u ** (-theta) + v ** (-theta) - 1, 1e-12)
        ).sum()
        return -(term1 + term2 + term3)

    result = optimize.minimize_scalar(neg_ll, bounds=(0.01, 30), method="bounded")
    return result.x


def fit_gumbel_copula(u: np.ndarray, v: np.ndarray) -> float:
    """
    Fit Gumbel copula parameter θ via Kendall's tau inversion.

    Gumbel captures upper-tail dependence:
    λ_U = 2 − 2^{1/θ}     (θ ≥ 1)

    Relationship: τ = 1 − 1/θ  →  θ = 1/(1−τ)
    """
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u, v)
    # Gumbel requires θ ≥ 1
    theta = max(1.0 / max(1 - tau, 0.01), 1.0)
    return theta


def joint_crash_probability(
    returns_a: pd.Series,
    returns_b: pd.Series,
    var_a: float,
    var_b: float,
) -> float:
    """
    Estimate P(A < -VaR_A AND B < -VaR_B) using empirical copula.

    This is the joint probability that both assets breach their VaR
    simultaneously.
    """
    joint = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    crash = ((joint["a"] < -var_a) & (joint["b"] < -var_b)).mean()
    return crash


# ══════════════════════════════════════════════
# 5.  VaR BACKTESTING
# ══════════════════════════════════════════════

def kupiec_pof_test(
    violations: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    H0: actual violation rate = expected α.

    LR_POF = 2 · [log L(p̂) − log L(α)]
    where p̂ = T_1 / T, T_1 = number of violations.

    Distributed χ²(1) under H0.
    """
    T = len(violations)
    T1 = violations.sum()
    T0 = T - T1
    p_hat = T1 / T if T > 0 else 0

    if p_hat == 0 or p_hat == 1:
        return {"lr_stat": np.nan, "p_value": np.nan, "violations": T1, "expected": alpha * T}

    ll_unrestricted = T1 * np.log(p_hat) + T0 * np.log(1 - p_hat)
    ll_restricted = T1 * np.log(alpha) + T0 * np.log(1 - alpha)
    lr = 2 * (ll_unrestricted - ll_restricted)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "lr_stat": lr,
        "p_value": p_value,
        "violations": int(T1),
        "violation_rate": p_hat,
        "expected_rate": alpha,
        "expected_violations": alpha * T,
    }


def christoffersen_test(
    violations: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Christoffersen (1998) conditional coverage test.

    Tests both unconditional coverage (Kupiec) AND independence
    of violations (no clustering).

    LR_CC = LR_POF + LR_IND   ~  χ²(2)
    """
    T = len(violations)
    T1 = violations.sum()

    # Transition counts
    n00, n01, n10, n11 = 0, 0, 0, 0
    for t in range(1, T):
        prev, curr = int(violations[t - 1]), int(violations[t])
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    # Independence test
    pi_01 = n01 / max(n00 + n01, 1)
    pi_11 = n11 / max(n10 + n11, 1)
    pi = T1 / T if T > 0 else 0

    if pi_01 == 0 or pi_01 == 1 or pi_11 == 0 or pi_11 == 1 or pi == 0 or pi == 1:
        return {"lr_cc": np.nan, "lr_ind": np.nan, "lr_pof": np.nan,
                "p_value": np.nan, "violations": int(T1)}

    ll_ind = (n00 * np.log(1 - pi_01) + n01 * np.log(pi_01)
              + n10 * np.log(1 - pi_11) + n11 * np.log(pi_11))
    ll_0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    lr_ind = 2 * (ll_ind - ll_0)

    kupiec = kupiec_pof_test(violations, alpha)
    lr_cc = kupiec["lr_stat"] + lr_ind if not np.isnan(kupiec["lr_stat"]) else np.nan
    p_value = 1 - stats.chi2.cdf(lr_cc, df=2) if not np.isnan(lr_cc) else np.nan

    return {
        "lr_cc": lr_cc,
        "lr_ind": lr_ind,
        "lr_pof": kupiec["lr_stat"],
        "p_value": p_value,
        "violations": int(T1),
    }


def traffic_light_zone(violation_rate: float, alpha: float = 0.01) -> str:
    """
    Basel traffic-light classification.

    Green:  violation rate in [0, α × 1.5)
    Yellow: violation rate in [α × 1.5, α × 2.0)
    Red:    violation rate ≥ α × 2.0
    """
    if violation_rate < alpha * 1.5:
        return "Green"
    elif violation_rate < alpha * 2.0:
        return "Yellow"
    else:
        return "Red"


# ══════════════════════════════════════════════
# 6.  PORTFOLIO RETURN SCENARIO MATRIX
# ══════════════════════════════════════════════

def build_return_scenario_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the (T × 20) matrix of historical return scenarios for
    portfolio-level CVaR optimisation in NB11.

    This matrix is stored as ``return_scenarios.parquet``.
    Portfolio CVaR = quantile of w' · R_t (NOT sum of weighted ticker CVaRs).

    Note: Uses fillna(0) for pre-IPO NaN values (conservative: no return)
    rather than dropna() which would discard the entire pre-PLTR period,
    leaving too few scenarios for reliable CVaR estimation.
    """
    # fillna(0) for pre-IPO dates (conservative: assume no return)
    # Drop only the first row (NaN from pct_change) rather than all rows
    # with any NaN, which would truncate the dataset severely.
    filled = returns.fillna(0)
    # Drop rows where ALL values are NaN (truly missing data)
    return filled.dropna(how="all")


# ══════════════════════════════════════════════
# 7.  COMPONENT & MARGINAL VaR
# ══════════════════════════════════════════════

def component_var(
    weights: np.ndarray,
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Component VaR — decomposes portfolio VaR into per-asset contributions.

    CVaR_i = w_i · β_i · VaR_p

    where β_i = Cov(R_i, R_p) / Var(R_p). Component VaRs sum to total
    portfolio VaR, useful for risk attribution.
    """
    ret = returns.dropna().values
    port_ret = ret @ weights
    port_var = var_historical(pd.Series(port_ret), alpha)
    port_var_stat = np.var(port_ret, ddof=1)
    if port_var_stat == 0:
        return np.zeros(len(weights))
    betas = np.array([np.cov(ret[:, i], port_ret)[0, 1] / port_var_stat
                       for i in range(ret.shape[1])])
    return weights * betas * port_var


def marginal_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_var: float,
) -> np.ndarray:
    """
    Marginal VaR — sensitivity of portfolio VaR to weight changes.

    MVaR_i = (Σw)_i · VaR_p / σ²_p
    """
    sigma_p = np.sqrt(weights @ cov_matrix @ weights)
    if sigma_p == 0:
        return np.zeros(len(weights))
    return (cov_matrix @ weights) * portfolio_var / (sigma_p ** 2)


# ══════════════════════════════════════════════
# 8.  STRESS VaR (BASEL III)
# ══════════════════════════════════════════════

def stress_var(
    returns: pd.Series,
    alpha: float = 0.01,
    stress_start: str = "2020-02-19",
    stress_end: str = "2020-03-23",
) -> float:
    """
    Stressed VaR — computes VaR using only a stress period.

    Basel III requires Stressed VaR calibrated to a continuous
    12-month period of significant financial stress.
    """
    stress_rets = returns.loc[stress_start:stress_end]
    if len(stress_rets) < 10:
        logger.warning("Stress period has only %d observations", len(stress_rets))
        return var_historical(returns, alpha)
    return var_historical(stress_rets, alpha)


# ══════════════════════════════════════════════
# 9.  CONDITIONAL DRAWDOWN AT RISK (CDaR)
# ══════════════════════════════════════════════

def conditional_drawdown_at_risk(
    prices: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    CDaR — average of worst α% drawdowns.

    CDaR_α = E[DD | DD > DD_{1-α}]

    Analogous to CVaR but on the drawdown distribution.

    Reference: Chekhlov, Uryasev & Zabarankin (2005)
    """
    cummax = prices.cummax()
    dd_losses = -(prices - cummax) / cummax
    threshold = np.quantile(dd_losses.dropna(), 1 - alpha)
    tail = dd_losses[dd_losses >= threshold]
    return float(tail.mean()) if len(tail) > 0 else 0.0


# ══════════════════════════════════════════════
# 10. TAIL DEPENDENCE COEFFICIENTS
# ══════════════════════════════════════════════

def tail_dependence_coefficient(
    theta: float,
    copula_type: str = "clayton",
) -> float:
    """
    Tail dependence coefficient from copula parameter.

    Clayton (lower tail): λ_L = 2^{−1/θ}
    Gumbel (upper tail):  λ_U = 2 − 2^{1/θ}

    λ = 0 → tail independent; λ = 1 → perfect tail dependence.
    """
    if copula_type == "clayton":
        return 2 ** (-1.0 / theta) if theta > 0 else 0.0
    elif copula_type == "gumbel":
        return 2 - 2 ** (1.0 / theta) if theta >= 1 else 0.0
    raise ValueError(f"Unknown copula type: {copula_type}")


# ══════════════════════════════════════════════
# 11. REGIME-CONDITIONAL SCENARIOS
# ══════════════════════════════════════════════

def build_regime_conditional_scenarios(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> Dict[int, pd.DataFrame]:
    """
    Build separate return scenario matrices per regime for
    regime-conditional portfolio CVaR optimization.
    """
    scenarios = {}
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        sub = returns.loc[mask].dropna()
        if len(sub) > 10:
            scenarios[regime] = sub
    return scenarios


# ══════════════════════════════════════════════
# 12. COMPREHENSIVE VaR BACKTEST
# ══════════════════════════════════════════════

def backtest_var_comprehensive(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.05,
) -> Dict:
    """
    Comprehensive VaR backtesting: Kupiec + Christoffersen + traffic light.

    Consolidates the backtesting workflow into a single function.
    """
    aligned = pd.DataFrame({"ret": returns, "var": var_series}).dropna()
    violations = (aligned["ret"] < -aligned["var"]).astype(int).values
    kupiec = kupiec_pof_test(violations, alpha)
    christo = christoffersen_test(violations, alpha)
    violation_rate = violations.mean()
    zone = traffic_light_zone(violation_rate, alpha)
    violation_dates = aligned.index[violations.astype(bool)].tolist()
    return {
        "kupiec": kupiec,
        "christoffersen": christo,
        "traffic_light": zone,
        "violation_rate": violation_rate,
        "n_violations": int(violations.sum()),
        "n_total": len(violations),
        "violation_dates": violation_dates,
    }
