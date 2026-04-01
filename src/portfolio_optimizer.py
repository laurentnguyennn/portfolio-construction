"""
portfolio_optimizer.py — All optimization methods.
====================================================
Mean-Variance (Markowitz), Mean-CVaR (scenario-based), Black-Litterman,
HRP, Risk Budgeting (ERC).  Implements UCITS-inspired constraints.

CRITICAL: Portfolio CVaR is computed from the empirical distribution
of portfolio returns  w' · R_t  (NOT from weighted individual CVaRs).

References
----------
* Markowitz (1952)
* Black & Litterman (1992)
* Lopez de Prado (2016) — HRP
* Rockafellar & Uryasev (2000) — CVaR optimisation
* Engle (2002) — DCC for covariance input
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MAX_SECTOR_WEIGHT,
    MAX_SINGLE_STOCK_WEIGHT,
    RANDOM_STATE,
    RETURN_SCENARIOS_FILE,
    SECTOR_GROUPS,
    TICKERS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  CONSTRAINT HELPERS
# ══════════════════════════════════════════════

def build_sector_bounds(
    tickers: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds per ticker from UCITS constraints.

    Returns (lower_bounds, upper_bounds) arrays of shape (N,).
    """
    tickers = tickers or TICKERS
    n = len(tickers)
    lb = np.zeros(n)
    ub = np.full(n, MAX_SINGLE_STOCK_WEIGHT)

    return lb, ub


def get_sector_group_indices(
    tickers: List[str] | None = None,
) -> Dict[str, List[int]]:
    """Map each sector group to column indices in the ticker list."""
    tickers = tickers or TICKERS
    groups = {}
    for group_name, group_tickers in SECTOR_GROUPS.items():
        indices = [tickers.index(t) for t in group_tickers if t in tickers]
        if indices:
            groups[group_name] = indices
    return groups


# ══════════════════════════════════════════════
# 2.  MEAN-VARIANCE (MARKOWITZ)
# ══════════════════════════════════════════════

def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: List[str] | None = None,
    objective: str = "max_sharpe",
    target_return: Optional[float] = None,
    rf: float = 0.0,
) -> np.ndarray:
    """
    Mean-Variance optimization via pypfopt.

    Parameters
    ----------
    expected_returns : array (N,)
    cov_matrix : array (N, N)
    objective : str
        'max_sharpe', 'min_volatility', 'efficient_return'.
    target_return : float
        Required if objective == 'efficient_return'.
    rf : float
        Annual risk-free rate (decimal).

    Returns
    -------
    weights : array (N,)
    """
    from pypfopt import EfficientFrontier

    tickers = tickers or TICKERS

    mu = pd.Series(expected_returns, index=tickers)
    S = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_SINGLE_STOCK_WEIGHT))

    # Add sector constraints (default-arg capture avoids late-binding closure bug)
    groups = get_sector_group_indices(tickers)
    for indices in groups.values():
        ef.add_constraint(
            lambda w, _g=[tickers[i] for i in indices]: sum(w[t] for t in _g) <= MAX_SECTOR_WEIGHT
        )

    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=rf)
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "efficient_return":
        ef.efficient_return(target_return=target_return)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    cleaned = ef.clean_weights()
    return np.array([cleaned.get(t, 0.0) for t in tickers])


# ══════════════════════════════════════════════
# 3.  MEAN-CVaR (SCENARIO-BASED)
# ══════════════════════════════════════════════

def mean_cvar_optimize(
    scenarios: np.ndarray,
    expected_returns: np.ndarray,
    alpha: float = 0.05,
    target_return: Optional[float] = None,
    tickers: List[str] | None = None,
) -> np.ndarray:
    """
    Scenario-based Mean-CVaR optimization via CVXPY.

    Portfolio CVaR is computed from w' · R_t for each scenario t.
    Formulated as a linear program (Rockafellar & Uryasev 2000).

    min  CVaR_α(w)
    s.t. w' · μ ≥ target_return  (if specified)
         Σ w_i = 1
         0 ≤ w_i ≤ 0.10
         sector constraints

    CVaR_α = ζ + (1/αT) Σ_t max(−w'R_t − ζ, 0)

    Parameters
    ----------
    scenarios : ndarray (T, N)
        Historical return scenarios from NB04.
    expected_returns : ndarray (N,)
    alpha : float
        Tail probability (e.g. 0.05 = worst 5% of scenarios).
    """
    import cvxpy as cp

    tickers = tickers or TICKERS
    T, N = scenarios.shape

    w = cp.Variable(N, nonneg=True)
    zeta = cp.Variable()      # VaR auxiliary variable
    u = cp.Variable(T, nonneg=True)   # max(−w'R_t − ζ, 0)

    # Portfolio returns per scenario
    port_returns = scenarios @ w

    # CVaR constraints
    constraints = [
        u >= -port_returns - zeta,
        cp.sum(w) == 1,
        w <= MAX_SINGLE_STOCK_WEIGHT,
    ]

    # Sector constraints
    groups = get_sector_group_indices(tickers)
    for group_name, indices in groups.items():
        constraints.append(cp.sum(w[indices]) <= MAX_SECTOR_WEIGHT)

    # Target return constraint
    if target_return is not None:
        constraints.append(expected_returns @ w >= target_return)

    # Objective: minimise CVaR
    cvar = zeta + (1.0 / (alpha * T)) * cp.sum(u)
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        logger.warning("CVaR optimization status: %s", problem.status)

    if w.value is None:
        logger.warning("CVaR solver returned None; falling back to equal weights")
        return np.ones(N) / N
    return np.array(w.value).flatten()


# ══════════════════════════════════════════════
# 4.  BLACK-LITTERMAN
# ══════════════════════════════════════════════

def black_litterman_optimize(
    cov_matrix: np.ndarray,
    market_cap_weights: np.ndarray,
    views: np.ndarray,
    view_rmse: np.ndarray,
    rf: float = 0.0,
    tau: float = 0.05,
    tickers: List[str] | None = None,
) -> np.ndarray:
    """
    Black-Litterman with ML-derived views.

    Prior: market-cap-implied equilibrium returns.
    Posterior: blended with ML return forecasts as views.

    View uncertainty: Ω_ii = RMSE_i²  (squared forecast error as
    variance of each view — correct units).

    Parameters
    ----------
    cov_matrix : (N, N)
    market_cap_weights : (N,)
        Market-cap weights as prior.
    views : (N,)
        ML-predicted expected returns per ticker.
    view_rmse : (N,)
        RMSE of each ticker's ML forecast.
    rf : float
        Annual risk-free rate.
    tau : float
        Scaling factor for uncertainty in the prior.

    Returns
    -------
    weights : (N,)
    """
    from pypfopt import BlackLittermanModel, EfficientFrontier

    tickers = tickers or TICKERS
    S = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    mcw = pd.Series(market_cap_weights, index=tickers)

    # Absolute views: each ticker has a predicted return
    viewdict = {t: views[i] for i, t in enumerate(tickers)}

    # Omega = diag(RMSE²)  — correct units (variance)
    omega = np.diag(view_rmse ** 2)

    bl = BlackLittermanModel(
        S,
        pi="market",
        market_caps=mcw,
        absolute_views=viewdict,
        omega=omega,
        tau=tau,
        risk_free_rate=rf,
    )

    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    ef = EfficientFrontier(bl_returns, bl_cov,
                           weight_bounds=(0, MAX_SINGLE_STOCK_WEIGHT))

    # Sector constraints (default-arg capture avoids late-binding closure bug)
    groups = get_sector_group_indices(tickers)
    for indices in groups.values():
        ef.add_constraint(
            lambda w, _g=[tickers[i] for i in indices]: sum(w[t] for t in _g) <= MAX_SECTOR_WEIGHT
        )

    ef.max_sharpe(risk_free_rate=rf)
    cleaned = ef.clean_weights()
    return np.array([cleaned.get(t, 0.0) for t in tickers])


# ══════════════════════════════════════════════
# 5.  HIERARCHICAL RISK PARITY (HRP)
# ══════════════════════════════════════════════

def hrp_optimize(
    returns: pd.DataFrame,
    tickers: List[str] | None = None,
) -> np.ndarray:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).

    No need to invert the covariance matrix — cluster-based allocation.
    More robust to estimation error than Markowitz.
    """
    from pypfopt import HRPOpt

    tickers = tickers or TICKERS
    returns_df = returns[tickers] if isinstance(returns, pd.DataFrame) else returns

    hrp = HRPOpt(returns_df)
    hrp.optimize()
    cleaned = hrp.clean_weights()

    weights = np.array([cleaned.get(t, 0.0) for t in tickers])

    # Enforce constraints by capping and renormalizing
    weights = _enforce_constraints(weights, tickers)

    return weights


# ══════════════════════════════════════════════
# 6.  RISK BUDGETING (EQUAL RISK CONTRIBUTION)
# ══════════════════════════════════════════════

def risk_budgeting_optimize(
    cov_matrix: np.ndarray,
    risk_budgets: Optional[np.ndarray] = None,
    tickers: List[str] | None = None,
) -> np.ndarray:
    """
    Equal Risk Contribution portfolio.

    Each asset contributes equally to total portfolio risk:
        w_i · (Σw)_i / (w' Σ w) = 1/N   for all i.

    Solved via sequential least-squares (SLSQP).
    """
    from scipy.optimize import minimize

    tickers = tickers or TICKERS
    N = len(tickers)

    if risk_budgets is None:
        risk_budgets = np.ones(N) / N  # equal budgets

    def risk_contribution(w):
        sigma = np.sqrt(w @ cov_matrix @ w)
        marginal = cov_matrix @ w
        rc = w * marginal / sigma
        return rc

    def objective(w):
        rc = risk_contribution(w)
        target = risk_budgets * (w @ cov_matrix @ w) ** 0.5
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, MAX_SINGLE_STOCK_WEIGHT)] * N

    result = minimize(
        objective,
        x0=np.ones(N) / N,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    weights = _enforce_constraints(weights, tickers)
    return weights


# ══════════════════════════════════════════════
# 7.  CONSTRAINT ENFORCEMENT
# ══════════════════════════════════════════════

def _enforce_constraints(
    weights: np.ndarray,
    tickers: List[str],
    max_iterations: int = 20,
) -> np.ndarray:
    """
    Post-hoc constraint enforcement: iteratively cap individual and sector
    weights, then renormalize until convergence.

    A single pass of cap + renormalize can re-violate constraints because
    renormalization redistributes weight. This iterates until all constraints
    are satisfied simultaneously.
    """
    groups = get_sector_group_indices(tickers)

    for _ in range(max_iterations):
        # Individual caps
        weights = np.minimum(weights, MAX_SINGLE_STOCK_WEIGHT)
        weights = np.maximum(weights, 0)

        # Sector caps
        for group_name, indices in groups.items():
            group_sum = weights[indices].sum()
            if group_sum > MAX_SECTOR_WEIGHT:
                weights[indices] *= MAX_SECTOR_WEIGHT / group_sum

        # Renormalize to sum to 1
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(tickers)) / len(tickers)
            continue
        weights /= total

        # Check convergence: all constraints satisfied after renormalization
        if (weights <= MAX_SINGLE_STOCK_WEIGHT + 1e-8).all():
            all_sectors_ok = all(
                weights[indices].sum() <= MAX_SECTOR_WEIGHT + 1e-8
                for indices in groups.values()
            )
            if all_sectors_ok:
                break

    return weights


# ══════════════════════════════════════════════
# 8.  EXPECTED RETURN ESTIMATORS
# ══════════════════════════════════════════════

def expected_returns_historical(returns: pd.DataFrame) -> np.ndarray:
    """Annualized historical mean returns (naive estimator)."""
    return (returns.mean() * 252).values


def expected_returns_capm(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    rf: float = 0.0,
) -> np.ndarray:
    """
    CAPM-implied expected returns:  E[R_i] = rf + β_i · (E[R_m] − rf).
    """
    market_premium = market_returns.mean() * 252 - rf
    betas = returns.apply(lambda x: x.cov(market_returns) / market_returns.var())
    return (rf + betas * market_premium).values


# ══════════════════════════════════════════════
# 9.  RISK ESTIMATORS
# ══════════════════════════════════════════════

def covariance_ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator (annualized)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns.dropna())
    return lw.covariance_ * 252


def load_return_scenarios(
    file=None,
    tickers: list | None = None,
) -> np.ndarray:
    """
    Load the (T × N) return scenario matrix produced by NB04.

    Used as input to ``mean_cvar_optimize`` and ``cvar_risk_budgeting``.
    Portfolio CVaR is computed from the empirical distribution of
    w' · R_t — NOT from weighted per-ticker CVaRs.

    Parameters
    ----------
    file : Path or None
        Parquet file path. Defaults to ``RETURN_SCENARIOS_FILE`` from config.
    tickers : list or None
        Column subset to select. If None, uses all columns in the file.

    Returns
    -------
    scenarios : ndarray (T, N)
    """
    path = file or RETURN_SCENARIOS_FILE
    df = pd.read_parquet(path)
    if tickers is not None:
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            logger.warning("load_return_scenarios: missing tickers %s in file", missing)
        df = df[[t for t in tickers if t in df.columns]]
    return df.dropna().values


# ══════════════════════════════════════════════
# 10.  WORST-CASE MEAN-VARIANCE
# ══════════════════════════════════════════════

def worst_case_mv_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    delta: float = 0.1,
    target_return: Optional[float] = None,
    tickers: List[str] | None = None,
) -> Dict:
    """
    Worst-case Mean-Variance optimization (Goldfarb & Iyengar, 2003).

    Robust to estimation error in Σ:
        min_w  w'Σ̂w + δ·||w||₂²
        s.t. w'μ ≥ r_target, Σw_i = 1, 0 ≤ w_i ≤ 0.10

    Equivalent to shrinking Σ̂ toward (Σ̂ + δI).
    δ calibrated via cross-validation or set heuristically.

    Parameters
    ----------
    expected_returns : array (N,)
    cov_matrix : array (N, N)
    delta : float
        Robustness parameter (regularization strength).
    target_return : float or None
        Minimum required return. If None, minimizes risk only.

    Returns
    -------
    dict with 'weights', 'portfolio_vol', 'portfolio_return'
    """
    import cvxpy as cp

    tickers = tickers or TICKERS
    N = len(tickers)

    w = cp.Variable(N, nonneg=True)
    regularized_cov = cov_matrix + delta * np.eye(N)

    objective = cp.Minimize(cp.quad_form(w, regularized_cov))

    constraints = [
        cp.sum(w) == 1,
        w <= MAX_SINGLE_STOCK_WEIGHT,
    ]

    # Sector constraints
    groups = get_sector_group_indices(tickers)
    for group_name, indices in groups.items():
        constraints.append(cp.sum(w[indices]) <= MAX_SECTOR_WEIGHT)

    if target_return is not None:
        constraints.append(expected_returns @ w >= target_return)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if w.value is not None:
        weights = np.array(w.value).flatten()
    else:
        logger.warning("Worst-case MV solver returned None; falling back to equal weights")
        weights = np.ones(N) / N
        if tickers is not None:
            weights = _enforce_constraints(weights, tickers)

    port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
    port_ret = float(expected_returns @ weights)

    return {
        "weights": weights,
        "portfolio_vol": port_vol,
        "portfolio_return": port_ret,
        "delta": delta,
    }


# ══════════════════════════════════════════════
# 11.  MAXIMUM DIVERSIFICATION
# ══════════════════════════════════════════════

def max_diversification_optimize(
    cov_matrix: np.ndarray,
    tickers: List[str] | None = None,
) -> Dict:
    """
    Maximum Diversification portfolio (Choueifaty & Coignard, 2008).

    max_w  DR(w) = w'σ / sqrt(w'Σw)

    where σ = vector of asset volatilities, Σ = covariance matrix.
    DR > 1 always; higher = more diversified.
    Equivalent to min-variance on the correlation matrix.

    Returns
    -------
    dict with 'weights', 'diversification_ratio'
    """
    from scipy.optimize import minimize as scipy_minimize

    tickers = tickers or TICKERS
    N = len(tickers)
    vols = np.sqrt(np.diag(cov_matrix))

    def neg_dr(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(w @ vols) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, MAX_SINGLE_STOCK_WEIGHT)] * N

    # Add sector constraints via penalty (SLSQP supports only eq/ineq)
    groups = get_sector_group_indices(tickers)
    for group_name, indices in groups.items():
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices: MAX_SECTOR_WEIGHT - w[idx].sum()
        })

    result = scipy_minimize(
        neg_dr,
        x0=np.ones(N) / N,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x
    weights = _enforce_constraints(weights, tickers)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    dr = (weights @ vols) / port_vol if port_vol > 0 else 1.0

    return {
        "weights": weights,
        "diversification_ratio": float(dr),
    }


# ══════════════════════════════════════════════
# 12.  RESAMPLED EFFICIENT FRONTIER
# ══════════════════════════════════════════════

def resampled_ef_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    T: int = 252,
    B: int = 1000,
    tickers: List[str] | None = None,
    rf: float = 0.0,
) -> Dict:
    """
    Resampled Efficient Frontier (Michaud, 1998).

    For b = 1,...,B:
        Draw Σ̂_b ~ Wishart(scale=Σ̂/T, df=T) so E[Σ̂_b] = Σ̂
        Draw μ̂_b ~ N(μ̂, Σ̂/T)
        Solve Markowitz with (μ̂_b, Σ̂_b) → w_b
    Final weights: w̄ = (1/B) · Σ_b w_b

    Resampled weights are more stable OOS than single-point Markowitz.

    Parameters
    ----------
    T : int
        Effective sample size for Wishart draws.
    B : int
        Number of bootstrap replications.

    Returns
    -------
    dict with 'weights', 'weight_std'
    """
    from scipy.stats import wishart

    tickers = tickers or TICKERS
    N = len(tickers)
    rng = np.random.RandomState(RANDOM_STATE)

    all_weights = []

    for b in range(B):
        # Draw Σ̂_b ~ Wishart(scale=Σ̂/T, df=T) → E[Σ̂_b] = T * (Σ̂/T) = Σ̂
        try:
            cov_b = wishart.rvs(df=T, scale=cov_matrix / T, random_state=rng)
        except Exception:
            # If Wishart fails (e.g., near-singular), use perturbation
            noise = rng.normal(0, 0.001, (N, N))
            cov_b = cov_matrix + noise @ noise.T

        # Ensure symmetry and positive definiteness
        cov_b = (cov_b + cov_b.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(cov_b))
        if min_eig < 1e-8:
            cov_b += (1e-8 - min_eig) * np.eye(N)

        # Draw μ̂_b ~ N(μ̂, Σ̂/T)
        mu_b = rng.multivariate_normal(expected_returns, cov_matrix / T)

        try:
            w_b = mean_variance_optimize(
                mu_b, cov_b, tickers=tickers, objective="max_sharpe", rf=rf
            )
            all_weights.append(w_b)
        except Exception:
            continue  # skip failed optimizations

    if len(all_weights) == 0:
        logger.warning("No valid resampled solutions; returning equal weights")
        return {"weights": np.ones(N) / N, "weight_std": np.zeros(N)}

    weight_matrix = np.array(all_weights)
    avg_weights = weight_matrix.mean(axis=0)
    std_weights = weight_matrix.std(axis=0)

    # Re-enforce constraints on averaged weights
    avg_weights = _enforce_constraints(avg_weights, tickers)

    # Flag tickers with high weight uncertainty (std > 5%) for disclosure
    high_uncertainty = [tickers[i] for i, s in enumerate(std_weights) if s > 0.05]
    if high_uncertainty:
        logger.info("High weight uncertainty (std > 5%%): %s", high_uncertainty)

    return {
        "weights": avg_weights,
        "weight_std": std_weights,
        "n_valid": len(all_weights),
        "high_uncertainty_tickers": high_uncertainty,
    }


# ══════════════════════════════════════════════
# 13.  CVaR RISK BUDGETING (TAIL RISK PARITY)
# ══════════════════════════════════════════════

def cvar_risk_budgeting(
    return_scenarios: np.ndarray,
    alpha: float = 0.05,
    tickers: List[str] | None = None,
    max_iter: int = 500,
) -> Dict:
    """
    CVaR-ERC: equalize marginal CVaR contributions across assets.

    Analogous to variance-based ERC but using CVaR instead of variance.
    Each asset contributes equally to portfolio tail risk.

    Parameters
    ----------
    return_scenarios : array (T, N)
        Return scenario matrix from NB04.
    alpha : float
        CVaR confidence level tail probability.

    Returns
    -------
    dict with 'weights', 'portfolio_cvar', 'marginal_cvar'
    """
    from scipy.optimize import minimize as scipy_minimize

    tickers = tickers or TICKERS
    N = return_scenarios.shape[1]

    def _portfolio_cvar(w, scenarios, alpha):
        port_ret = scenarios @ w
        threshold = np.quantile(port_ret, alpha)
        tail = port_ret[port_ret <= threshold]
        return -tail.mean() if len(tail) > 0 else 0.0

    def _marginal_cvar(w, scenarios, alpha, eps=1e-4):
        """Numerical marginal CVaR via finite differences."""
        base_cvar = _portfolio_cvar(w, scenarios, alpha)
        m_cvar = np.empty(N)
        for i in range(N):
            w_up = w.copy()
            w_up[i] += eps
            w_up /= w_up.sum()  # renormalize
            m_cvar[i] = (_portfolio_cvar(w_up, scenarios, alpha) - base_cvar) / eps
        return m_cvar

    def objective(w_raw):
        """Minimize dispersion of risk contributions."""
        w_clipped = np.maximum(w_raw, 1e-8)
        w_norm = w_clipped / w_clipped.sum()
        mcvar = _marginal_cvar(w_norm, return_scenarios, alpha)
        rc = w_norm * mcvar  # risk contribution
        target = rc.mean()
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.001, MAX_SINGLE_STOCK_WEIGHT)] * N

    result = scipy_minimize(
        objective,
        x0=np.ones(N) / N,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-10},
    )

    weights = result.x
    weights = _enforce_constraints(weights, tickers)

    port_cvar = _portfolio_cvar(weights, return_scenarios, alpha)
    m_cvar = _marginal_cvar(weights, return_scenarios, alpha)

    rc = weights * m_cvar
    max_imbalance = float(np.max(np.abs(rc - rc.mean()))) if len(rc) > 0 else 0.0

    return {
        "weights": weights,
        "portfolio_cvar": float(port_cvar),
        "marginal_cvar": m_cvar.tolist(),
        "max_imbalance": max_imbalance,
    }


# ══════════════════════════════════════════════
# 14.  CONDITIONAL DIVERSIFICATION BENEFIT
# ══════════════════════════════════════════════

def conditional_diversification_benefit(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Conditional Diversification Benefit (CDB).

    CDB = 1 - σ_portfolio / (w' · σ)

    where σ = vector of individual asset volatilities.
    CDB ∈ [0, 1]: 0 = perfect correlation (no benefit),
                   1 = zero portfolio vol (maximum benefit).

    Compute per regime: typically CDB_bear < CDB_bull
    (diversification fails when most needed).

    Parameters
    ----------
    weights : array (N,)
    cov_matrix : array (N, N)

    Returns
    -------
    cdb : float
    """
    vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol_sum = weights @ vols  # w'σ
    port_vol = np.sqrt(weights @ cov_matrix @ weights)

    if weighted_vol_sum < 1e-10:
        return 0.0

    return 1.0 - port_vol / weighted_vol_sum


# ══════════════════════════════════════════════
# 15.  REGIME-ADAPTIVE ALLOCATION STRATEGY
# ══════════════════════════════════════════════

def regime_adaptive_weights(
    regime: int,
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    expected_returns: np.ndarray,
    tickers: list | None = None,
    rf: float = 0.0,
) -> np.ndarray:
    """
    Regime-adaptive portfolio allocation (CLAUDE.md §NB11).

    Strategy:
        Bear   (regime 0) → HRP — robust to estimation error, no matrix inversion.
        Neutral (regime 1) → Min-Variance — reduce risk without return forecast reliance.
        Bull   (regime 2) → Max-Sharpe — exploit ML return forecasts.

    Empirical basis: ML return forecasts have lower information content during
    high-volatility regimes (wider prediction intervals), so HRP's structural
    diversification is more reliable in bear markets.

    Parameters
    ----------
    regime : int
        HMM regime label. 0 = bear, 1 = neutral/transition, 2 = bull.
        For 2-state models map as: 0 = bear, 1 = bull (treated as regime 2).
    returns : DataFrame (T, N)
        Return history available up to current rebalance date (for HRP).
    cov_matrix : ndarray (N, N)
        Covariance matrix for MV methods (Ledoit-Wolf or DCC).
    expected_returns : ndarray (N,)
        Expected returns for MV optimizers.
    tickers : list or None
    rf : float
        Annual risk-free rate.

    Returns
    -------
    weights : ndarray (N,)
    """
    tickers = tickers or TICKERS

    if regime == 0:
        # Bear: HRP — cluster-based, no covariance inversion needed
        logger.info("Regime-adaptive: BEAR → HRP")
        return hrp_optimize(returns, tickers=tickers)

    elif regime == 1:
        # Neutral: minimum-variance — reduce risk, skip return forecasts
        logger.info("Regime-adaptive: NEUTRAL → Min-Variance")
        return mean_variance_optimize(
            expected_returns, cov_matrix,
            tickers=tickers, objective="min_volatility", rf=rf,
        )

    else:
        # Bull: max-Sharpe — exploit ML return forecasts
        logger.info("Regime-adaptive: BULL → Max-Sharpe")
        try:
            return mean_variance_optimize(
                expected_returns, cov_matrix,
                tickers=tickers, objective="max_sharpe", rf=rf,
            )
        except Exception as e:
            logger.warning("Max-Sharpe failed (%s); falling back to Min-Variance", e)
            return mean_variance_optimize(
                expected_returns, cov_matrix,
                tickers=tickers, objective="min_volatility", rf=rf,
            )
