"""
backtest_engine.py — Walk-forward portfolio backtest logic.
============================================================
Monthly rebalancing using only out-of-sample ML predictions.
Computes performance metrics, benchmark comparison, and
transaction cost impact.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MAX_MONTHLY_TURNOVER,
    TICKERS,
    TRANSACTION_COST_BPS,
)


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  PERFORMANCE METRICS
# ══════════════════════════════════════════════

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return from daily returns."""
    total = (1 + returns).prod()
    n = len(returns)
    if total <= 0:
        return -1.0  # total loss
    return total ** (periods_per_year / n) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    SR = (E[R - Rf] × √252) / (σ(R - Rf) × √252) = E[R-Rf]/σ(R-Rf) × √252
    """
    excess = returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino ratio: like Sharpe but uses downside deviation only.

    Downside deviation = std of min(R - Rf, 0).
    """
    excess = returns - rf_daily
    downside_std = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    if downside_std == 0:
        return 0.0
    return excess.mean() / downside_std * np.sqrt(periods_per_year)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    """Max drawdown from a daily return series."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return dd.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar = annualized return / |max drawdown|."""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown_from_returns(returns))
    if mdd == 0:
        return np.nan
    return ann_ret / mdd


def compute_all_metrics(
    returns: pd.Series,
    rf_daily: float = 0.0,
    name: str = "",
    benchmark_returns: pd.Series | None = None,
) -> Dict:
    """
    Compute the full set of portfolio performance metrics.

    Covers all metrics specified in CLAUDE.md §NB11:
    Sharpe, Sortino, Calmar, Omega, Tail Ratio, Ulcer Index,
    Pain Index, CDaR, Information Ratio, Tracking Error.

    Parameters
    ----------
    returns : Series — daily portfolio returns.
    rf_daily : float — daily risk-free rate.
    name : str — strategy label for reporting.
    benchmark_returns : Series or None — for IR and TE computation.
    """
    metrics = {
        "name": name,
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, rf_daily),
        "sortino_ratio": sortino_ratio(returns, rf_daily),
        "max_drawdown": max_drawdown_from_returns(returns),
        "calmar_ratio": calmar_ratio(returns),
        "total_return": (1 + returns).prod() - 1,
        "omega_ratio": omega_ratio(returns, threshold=rf_daily),
        "tail_ratio": tail_ratio(returns),
        "ulcer_index": ulcer_index(returns, is_returns=True),
        "pain_index": pain_index(returns, is_returns=True),
        "cdar_95": conditional_drawdown_at_risk(returns, alpha=0.05),
        "n_days": len(returns),
    }

    if benchmark_returns is not None:
        aligned = pd.DataFrame({
            "port": returns,
            "bench": benchmark_returns,
        }).dropna()
        if len(aligned) > 10:
            metrics["information_ratio"] = information_ratio(
                aligned["port"], aligned["bench"]
            )
            metrics["tracking_error"] = tracking_error(
                aligned["port"], aligned["bench"]
            )

    return metrics


# ══════════════════════════════════════════════
# 2.  TURNOVER COMPUTATION
# ══════════════════════════════════════════════

def compute_turnover(
    weights_old: np.ndarray,
    weights_new: np.ndarray,
    returns_between: Optional[np.ndarray] = None,
) -> float:
    """
    Portfolio turnover = Σ|w_new − w_drift|.

    If ``returns_between`` is provided, we first drift the old weights
    by the intermediate returns before computing the difference.
    """
    if returns_between is not None:
        # Drift old weights by returns
        w_drift = weights_old * (1 + returns_between)
        w_drift /= w_drift.sum()
    else:
        w_drift = weights_old

    return np.abs(weights_new - w_drift).sum()


# ══════════════════════════════════════════════
# 3.  MONTHLY REBALANCING BACKTEST
# ══════════════════════════════════════════════

def run_backtest(
    returns: pd.DataFrame,
    weight_function: Callable,
    rebalance_dates: List[pd.Timestamp],
    tickers: List[str] | None = None,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    max_turnover: float = MAX_MONTHLY_TURNOVER,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Walk-forward portfolio backtest with monthly rebalancing.

    Parameters
    ----------
    returns : DataFrame (T × N)
        Daily simple returns for each ticker.
    weight_function : callable
        ``f(date, returns_up_to_date) → np.ndarray`` of weights.
        Must use only information available up to ``date`` (no lookahead).
    rebalance_dates : list of Timestamps
        Monthly rebalance dates.
    transaction_cost_bps : float
        Round-trip cost in basis points.
    max_turnover : float
        Maximum allowed turnover per rebalance (20%).

    Returns
    -------
    weights_history : DataFrame
        Columns = tickers, index = rebalance dates.
    portfolio_returns : Series
        Daily portfolio returns (after transaction costs).
    """
    tickers = tickers or TICKERS
    tc = transaction_cost_bps / 10_000

    all_returns = []
    all_weights = []
    current_weights = np.zeros(len(tickers))

    for i, reb_date in enumerate(rebalance_dates):
        # Get target weights using only past data
        target_weights = weight_function(reb_date, returns.loc[:reb_date])

        # Turnover constraint: cap the rebalance if needed
        turnover = compute_turnover(current_weights, target_weights)
        if turnover > max_turnover and current_weights.sum() > 0:
            # Partial rebalance: move max_turnover toward target
            blend = max_turnover / max(turnover, 1e-10)
            target_weights = current_weights + blend * (target_weights - current_weights)
            target_weights = np.maximum(target_weights, 0)
            target_weights /= target_weights.sum()

        # Transaction cost for this rebalance
        actual_turnover = compute_turnover(current_weights, target_weights)
        rebalance_cost = actual_turnover * tc

        # Get daily returns until next rebalance
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = returns.index[-1]

        period_mask = (returns.index > reb_date) & (returns.index <= next_date)
        period_returns = returns.loc[period_mask, tickers]

        if len(period_returns) == 0:
            continue

        # Daily portfolio returns
        daily_port = period_returns.values @ target_weights

        # Subtract transaction cost on first day
        if len(daily_port) > 0:
            daily_port[0] -= rebalance_cost

        port_series = pd.Series(daily_port, index=period_returns.index)
        all_returns.append(port_series)
        all_weights.append({"date": reb_date, **dict(zip(tickers, target_weights))})

        # Update current weights (drift through period)
        cum_ret = (1 + period_returns).prod().values
        current_weights = target_weights * cum_ret
        if current_weights.sum() > 0:
            current_weights /= current_weights.sum()

    portfolio_returns = pd.concat(all_returns) if all_returns else pd.Series(dtype=float)
    weights_history = pd.DataFrame(all_weights).set_index("date") if all_weights else pd.DataFrame()

    return weights_history, portfolio_returns


# ══════════════════════════════════════════════
# 4.  BENCHMARK COMPARISON
# ══════════════════════════════════════════════

def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio returns (1/N)."""
    return returns.mean(axis=1)


def market_cap_weight_returns(
    returns: pd.DataFrame,
    market_caps: pd.Series,
) -> pd.Series:
    """Market-cap-weighted portfolio returns."""
    w = market_caps / market_caps.sum()
    return (returns * w).sum(axis=1)


def compare_strategies(
    strategy_returns: Dict[str, pd.Series],
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    """Compare multiple strategies' performance metrics side by side."""
    rows = []
    for name, rets in strategy_returns.items():
        metrics = compute_all_metrics(rets, rf_daily, name=name)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("name")


# ══════════════════════════════════════════════
# 5.  ROLLING PERFORMANCE METRICS
# ══════════════════════════════════════════════

def rolling_metrics(
    portfolio_returns: pd.Series,
    window: int = 252,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe, rolling volatility, and rolling drawdown.

    Parameters
    ----------
    portfolio_returns : Series of daily returns.
    window : int — rolling window in trading days (default 252 = 1 year).

    Returns
    -------
    DataFrame with columns: rolling_sharpe, rolling_vol, rolling_drawdown.
    """
    excess = portfolio_returns - rf_daily
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    r_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    r_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.rolling(window, min_periods=1).max()
    r_dd = (cumulative - rolling_max) / rolling_max

    return pd.DataFrame({
        "rolling_sharpe": r_sharpe,
        "rolling_vol": r_vol,
        "rolling_drawdown": r_dd,
    }, index=portfolio_returns.index)


# ══════════════════════════════════════════════
# 6.  REGIME-CONDITIONAL PERFORMANCE
# ══════════════════════════════════════════════

def regime_conditional_performance(
    portfolio_returns: pd.Series,
    regime_labels: pd.Series,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    """
    Compute performance metrics separately per regime.

    Aligns regime labels with portfolio returns and computes
    all metrics for each regime period.

    Returns
    -------
    DataFrame with regime as index, metrics as columns.
    """
    aligned = pd.DataFrame({
        "ret": portfolio_returns,
        "regime": regime_labels,
    }).dropna()

    rows = []
    for regime in sorted(aligned["regime"].unique()):
        mask = aligned["regime"] == regime
        rets = aligned.loc[mask, "ret"]
        if len(rets) < 5:
            continue
        metrics = compute_all_metrics(rets, rf_daily, name=f"Regime_{int(regime)}")
        metrics["regime"] = int(regime)
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ══════════════════════════════════════════════
# 7.  INFORMATION RATIO
# ══════════════════════════════════════════════

def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio — risk-adjusted active return.

    IR = mean(R_p − R_b) / std(R_p − R_b) × √252

    Measures consistency of outperformance per unit of tracking error.
    IR > 0.5 is generally considered good; > 1.0 is exceptional.
    """
    active = (portfolio_returns - benchmark_returns).dropna()
    if active.std() == 0:
        return 0.0
    return active.mean() / active.std() * np.sqrt(periods_per_year)


# ══════════════════════════════════════════════
# 8.  OMEGA RATIO
# ══════════════════════════════════════════════

def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio — ratio of probability-weighted gains to losses.

    Ω(L) = ∫_L^∞ (1 − F(r)) dr / ∫_{-∞}^L F(r) dr

    Equivalently: sum of (r - L) for r > L / sum of (L - r) for r < L.

    Ω > 1 indicates the return distribution is favorable above the threshold.
    Unlike Sharpe, Omega uses the entire return distribution (all moments).

    Reference: Keating & Shadwick (2002), "A Universal Performance Measure"
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]

    sum_losses = losses.sum()
    if sum_losses == 0:
        return np.inf
    return gains.sum() / sum_losses


# ══════════════════════════════════════════════
# 9.  TRACKING ERROR
# ══════════════════════════════════════════════

def tracking_error(
    portfolio_returns,
    benchmark_returns,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Tracking Error.

    TE = std(R_p - R_b) × √252

    Measures dispersion of active returns relative to benchmark.
    """
    if isinstance(portfolio_returns, np.ndarray):
        portfolio_returns = pd.Series(portfolio_returns)
    if isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = pd.Series(benchmark_returns)
    active = (portfolio_returns - benchmark_returns).dropna()
    return active.std() * np.sqrt(periods_per_year)


# ══════════════════════════════════════════════
# 10.  ULCER INDEX
# ══════════════════════════════════════════════

def ulcer_index(prices_or_returns, is_returns: bool = False) -> float:
    """
    Ulcer Index (Martin, 1987).

    UI = sqrt(mean(D²)) where D = drawdown percentage.

    Measures the depth and duration of drawdowns from peaks.
    Unlike max drawdown, UI captures the pain of sustained losses.

    Parameters
    ----------
    prices_or_returns : Series or ndarray
        Either price series or return series (set is_returns=True).
    is_returns : bool
        If True, converts returns to cumulative prices first.
    """
    if isinstance(prices_or_returns, np.ndarray):
        prices_or_returns = pd.Series(prices_or_returns)
    if is_returns:
        prices = (1 + prices_or_returns).cumprod()
    else:
        prices = prices_or_returns

    peak = prices.cummax()
    drawdown_pct = ((prices - peak) / peak) * 100  # percentage drawdown
    return np.sqrt((drawdown_pct ** 2).mean())


# ══════════════════════════════════════════════
# 11.  PAIN INDEX
# ══════════════════════════════════════════════

def pain_index(prices_or_returns, is_returns: bool = False) -> float:
    """
    Pain Index — mean absolute drawdown percentage.

    Simpler than Ulcer Index; measures average severity of drawdowns.
    """
    if isinstance(prices_or_returns, np.ndarray):
        prices_or_returns = pd.Series(prices_or_returns)
    if is_returns:
        prices = (1 + prices_or_returns).cumprod()
    else:
        prices = prices_or_returns

    peak = prices.cummax()
    drawdown_pct = (prices - peak) / peak  # negative values
    return float(np.abs(drawdown_pct).mean())


# ══════════════════════════════════════════════
# 12.  CONDITIONAL DRAWDOWN AT RISK (CDaR)
# ══════════════════════════════════════════════

def conditional_drawdown_at_risk(
    returns,
    alpha: float = 0.05,
) -> float:
    """
    CDaR: Conditional Drawdown at Risk (Chekhlov, Uryasev & Zabarankin, 2005).

    CDaR_α = E[DD | DD ≥ quantile(DD, 1-α)]

    Average of the worst α fraction of drawdown episodes.
    CDaR at α=0.05: average of worst 5% of drawdowns.

    Parameters
    ----------
    returns : Series of daily returns.
    alpha : float
        Tail probability (default 0.05 = worst 5%).
        Consistent with risk_metrics.py convention.

    Returns
    -------
    cdar : float (positive value representing loss magnitude)
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdowns = (peak - cumulative) / peak  # positive = loss

    threshold = drawdowns.quantile(1 - alpha)
    tail = drawdowns[drawdowns >= threshold]

    if len(tail) == 0:
        return 0.0

    return float(tail.mean())


# ══════════════════════════════════════════════
# 13.  ALMGREN-CHRISS TRANSACTION COST MODEL
# ══════════════════════════════════════════════

def transaction_cost_impact_model(
    weights_old: np.ndarray,
    weights_new: np.ndarray,
    spreads: np.ndarray,
    sigmas: np.ndarray,
    adv: np.ndarray,
    aum: float,
    eta: float = 0.1,
) -> Dict:
    """
    Almgren-Chriss spread + market impact transaction cost model.

    TC_i = Spread_cost_i + Impact_cost_i

    Spread_cost_i = (spread_i / 2) · |Δw_i| · AUM
    Impact_cost_i = η · σ_{i,daily} · (|Δw_i| · AUM) ·
                    sqrt(|Δw_i| · AUM / ADV_i)

    Parameters
    ----------
    weights_old : array (N,) — pre-trade weights (drifted)
    weights_new : array (N,) — target weights
    spreads : array (N,) — half-spread per ticker (decimal)
    sigmas : array (N,) — daily volatility per ticker
    adv : array (N,) — average daily dollar volume per ticker
    aum : float — portfolio AUM in dollars
    eta : float — market impact coefficient (default 0.1)

    Returns
    -------
    dict with:
        total_cost : float — total dollar cost
        total_cost_bps : float — cost in basis points of AUM
        spread_cost : float — total spread cost
        impact_cost : float — total market impact cost
        per_ticker_cost : array — cost per ticker
    """
    delta_w = np.abs(weights_new - weights_old)
    trade_dollars = delta_w * aum

    # Spread cost
    spread_cost = spreads * trade_dollars
    total_spread = spread_cost.sum()

    # Market impact cost (square-root model)
    participation_rate = np.where(adv > 0, trade_dollars / adv, 0)
    impact_cost = eta * sigmas * trade_dollars * np.sqrt(np.maximum(participation_rate, 0))
    total_impact = impact_cost.sum()

    total_cost = total_spread + total_impact
    total_bps = total_cost / aum * 10_000 if aum > 0 else 0.0

    return {
        "total_cost": float(total_cost),
        "total_cost_bps": float(total_bps),
        "spread_cost": float(total_spread),
        "impact_cost": float(total_impact),
        "per_ticker_cost": (spread_cost + impact_cost).tolist(),
    }


# ══════════════════════════════════════════════
# 14.  TAIL RATIO
# ══════════════════════════════════════════════

def tail_ratio(returns: pd.Series) -> float:
    """
    Tail Ratio = 95th percentile / |5th percentile|.

    Measures asymmetry of the return distribution.
    > 1 → positive skew (larger upside than downside).
    < 1 → negative skew (larger downside than upside).
    """
    p95 = np.percentile(returns.dropna(), 95)
    p5 = abs(np.percentile(returns.dropna(), 5))
    if p5 == 0:
        return np.inf
    return p95 / p5
