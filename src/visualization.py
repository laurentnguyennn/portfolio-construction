"""
visualization.py — Standardized plotting functions.
=====================================================
Consistent style, high-DPI, labelled axes, publication quality.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIG_DPI, FIG_FORMAT, FIGURES_DIR, PALETTE

logger = logging.getLogger(__name__)

# Global plot style
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "figure.figsize": (14, 7),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def save_fig(fig, name: str, tight: bool = True, close: bool = False):
    """Save figure to outputs/figures/.

    Parameters
    ----------
    close : bool
        If True, close the figure after saving. Default False so that
        Jupyter ``plt.show()`` can still display the figure inline.
    """
    path = FIGURES_DIR / f"{name}.{FIG_FORMAT}"
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    logger.info("Saved figure: %s", path)


# ══════════════════════════════════════════════
# 1.  PRICE & RETURN CHARTS
# ══════════════════════════════════════════════

def plot_cumulative_returns(
    prices: pd.DataFrame,
    title: str = "Cumulative Returns (Log Scale)",
    log_scale: bool = True,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """20-stock cumulative return plot."""
    fig, ax = plt.subplots(figsize=(16, 8))
    normalized = prices / prices.iloc[0] * 100
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, linewidth=1)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price (100 = start)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_drawdown_waterfall(
    drawdowns: pd.DataFrame,
    ticker: str,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Drawdown waterfall chart for a single ticker."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdowns.index, drawdowns["drawdown"], 0,
                     alpha=0.4, color="red")
    ax.plot(drawdowns.index, drawdowns["drawdown"], color="darkred", linewidth=0.8)
    ax.set_title(f"Drawdown: {ticker}")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 2.  CORRELATION HEATMAPS
# ══════════════════════════════════════════════

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Publication-quality correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 3.  VOLATILITY PLOTS
# ══════════════════════════════════════════════

def plot_volatility_boxplots(
    vol_data: pd.DataFrame,
    group_labels: Dict[str, List[str]],
    title: str = "Annualized Volatility by Sector",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Sector-grouped box plots of annualized volatility."""
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_data = []
    for group, tickers in group_labels.items():
        for t in tickers:
            if t in vol_data.columns:
                vals = vol_data[t].dropna()
                for v in vals:
                    plot_data.append({"Sector": group, "Ticker": t, "Vol": v})
    df = pd.DataFrame(plot_data)
    if not df.empty:
        sns.boxplot(data=df, x="Sector", y="Vol", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Annualized Volatility")
        plt.xticks(rotation=45, ha="right")
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_conditional_volatility(
    cond_vol: pd.DataFrame,
    tickers: List[str],
    title: str = "GARCH Conditional Volatility",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Time series of conditional volatility for selected tickers."""
    fig, ax = plt.subplots(figsize=(16, 6))
    for t in tickers:
        if t in cond_vol.columns:
            ax.plot(cond_vol.index, cond_vol[t], label=t, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Conditional Volatility (annualized)")
    ax.set_xlabel("Date")
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 4.  REGIME PLOTS
# ══════════════════════════════════════════════

def plot_regime_overlay(
    prices: pd.Series,
    regime_labels: pd.Series,
    title: str = "Price with Regime Overlay",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Full 10-year price chart with regime-colored background shading."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(prices.index, prices.values, color="black", linewidth=0.8)

    colors = {0: "red", 1: "yellow", 2: "green"}
    labels_map = {0: "Bear", 1: "Neutral", 2: "Bull"}

    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        starts = mask & ~mask.shift(1, fill_value=False)
        ends = mask & ~mask.shift(-1, fill_value=False)
        for s, e in zip(prices.index[starts], prices.index[ends]):
            ax.axvspan(s, e, alpha=0.2,
                       color=colors.get(regime, "gray"),
                       label=labels_map.get(regime, f"State {regime}"))

    # Deduplicate legend
    handles, labs = ax.get_legend_handles_labels()
    by_label = dict(zip(labs, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(title)
    ax.set_ylabel("Price")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 5.  MODEL COMPARISON
# ══════════════════════════════════════════════

def plot_model_comparison(
    comparison: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Model Comparison",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing models on a given metric."""
    fig, ax = plt.subplots(figsize=(12, 6))
    comp_sorted = comparison.sort_values(metric)
    ax.barh(comp_sorted.index, comp_sorted[metric])
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 6.  PORTFOLIO PLOTS
# ══════════════════════════════════════════════

def plot_efficient_frontier(
    returns_range: np.ndarray,
    vol_range: np.ndarray,
    optimal_ret: float,
    optimal_vol: float,
    title: str = "Efficient Frontier",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Efficient frontier with optimal portfolio marked."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(vol_range, returns_range, "b-", linewidth=2, label="Efficient Frontier")
    ax.scatter(optimal_vol, optimal_ret, marker="*", s=300, c="red",
               label="Optimal Portfolio", zorder=5)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title(title)
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_weight_evolution(
    weights: pd.DataFrame,
    title: str = "Portfolio Weight Evolution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=(16, 8))
    weights.plot.area(ax=ax, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_backtest_comparison(
    strategy_returns: Dict[str, pd.Series],
    title: str = "Strategy Comparison",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Cumulative return comparison of multiple strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, rets in strategy_returns.items():
        cumulative = (1 + rets).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.set_yscale("log")
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 7.  QQ PLOT & DISTRIBUTION
# ══════════════════════════════════════════════

def plot_qq(
    data: pd.Series,
    title: str = "QQ Plot",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """QQ plot against normal distribution."""
    from scipy import stats
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(data.dropna(), dist="norm", plot=ax)
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_return_distribution(
    returns: pd.Series,
    ticker: str,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Kernel density vs. normal overlay with heavy-tail diagnostics."""
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(10, 6))
    returns.dropna().plot.kde(ax=ax, label="Empirical KDE", linewidth=2)
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, norm.pdf(x, loc=returns.mean(), scale=returns.std()),
            "r--", label="Normal", linewidth=1.5)
    ax.set_title(f"Return Distribution: {ticker}")
    ax.set_xlabel("Log Return")
    ax.legend()
    # Annotate skew/kurtosis
    ax.annotate(
        f"Skew={returns.skew():.2f}  Kurt={returns.kurtosis():.2f}",
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 8.  SHAP EXPLAINABILITY PLOTS
# ══════════════════════════════════════════════

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_names: List[str],
    title: str = "SHAP Feature Importance (Beeswarm)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """SHAP beeswarm plot for tree-based model explainability."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping beeswarm plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "shap not installed", ha="center", va="center")
        return fig

    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_name:
        save_fig(fig, save_name, tight=False)
    return fig


def plot_shap_waterfall(
    shap_values,
    feature_names: List[str],
    sample_idx: int = 0,
    title: str = "SHAP Waterfall (Single Prediction)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """SHAP waterfall for a single prediction explanation."""
    try:
        import shap
    except ImportError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "shap not installed", ha="center", va="center")
        return fig

    fig = plt.figure(figsize=(10, 8))
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title(title)
    if save_name:
        save_fig(fig, save_name, tight=False)
    return fig


# ══════════════════════════════════════════════
# 9.  STRESS TEST & MONTE CARLO
# ══════════════════════════════════════════════

def plot_stress_scenario_waterfall(
    impacts: Dict[str, float],
    title: str = "Stress Scenario P&L Impact",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Waterfall chart of portfolio P&L under a stress scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(impacts.keys())
    values = list(impacts.values())
    colors = ["green" if v >= 0 else "red" for v in values]
    ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("P&L Impact (%)")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.2f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_monte_carlo_distribution(
    portfolio_values: np.ndarray,
    var_95: float,
    cvar_95: float,
    title: str = "Monte Carlo Portfolio Distribution (1-Year Horizon)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Histogram of Monte Carlo simulated portfolio terminal values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(portfolio_values, bins=100, density=True, alpha=0.7,
            color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(var_95, color="orange", linewidth=2, linestyle="--",
               label=f"VaR 95% = {var_95:.2%}")
    ax.axvline(cvar_95, color="red", linewidth=2, linestyle="--",
               label=f"CVaR 95% = {cvar_95:.2%}")
    ax.axvline(np.median(portfolio_values), color="green", linewidth=2,
               label=f"Median = {np.median(portfolio_values):.2%}")
    ax.set_title(title)
    ax.set_xlabel("Portfolio Return")
    ax.set_ylabel("Density")
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 10.  PORTFOLIO WEIGHT PIE CHART
# ══════════════════════════════════════════════

def plot_weight_pie(
    weights: Dict[str, float],
    title: str = "Portfolio Allocation",
    min_pct: float = 0.02,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Pie chart of portfolio weights (small allocations grouped as 'Other')."""
    fig, ax = plt.subplots(figsize=(10, 10))
    labels, sizes = [], []
    other = 0.0
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w >= min_pct:
            labels.append(f"{name} ({w:.1%})")
            sizes.append(w)
        else:
            other += w
    if other > 0:
        labels.append(f"Other ({other:.1%})")
        sizes.append(other)
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 11.  NEWS IMPACT CURVE
# ══════════════════════════════════════════════

def plot_news_impact_curve(
    shock_range: np.ndarray,
    responses: Dict[str, np.ndarray],
    ticker: str,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    News Impact Curve (Engle & Ng 1993): σ² response to return shocks.

    Parameters
    ----------
    shock_range : array of shock values (e.g., -3σ to +3σ)
    responses : dict mapping model name → σ² values at each shock
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, sigma2 in responses.items():
        ax.plot(shock_range, sigma2, label=model_name, linewidth=2)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"News Impact Curve: {ticker}")
    ax.set_xlabel("Return Shock (ε)")
    ax.set_ylabel("Conditional Variance (σ²)")
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 12.  LEARNING CURVE
# ══════════════════════════════════════════════

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    metric_name: str = "RMSE",
    title: str = "Learning Curve",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Learning curve: train vs. validation metric vs. training set size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores, "o-", label=f"Train {metric_name}", linewidth=2)
    ax.plot(train_sizes, val_scores, "s--", label=f"Validation {metric_name}", linewidth=2)
    ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(metric_name)
    ax.legend()
    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 13.  ROLLING METRICS DASHBOARD
# ══════════════════════════════════════════════

def plot_rolling_metrics(
    rolling_df: pd.DataFrame,
    title: str = "Rolling Performance Metrics (252-day)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Three-panel chart: rolling Sharpe, rolling vol, rolling drawdown."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    if "rolling_sharpe" in rolling_df.columns:
        axes[0].plot(rolling_df.index, rolling_df["rolling_sharpe"],
                     color="steelblue", linewidth=1)
        axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
        axes[0].set_ylabel("Rolling Sharpe")
        axes[0].set_title(title)

    if "rolling_vol" in rolling_df.columns:
        axes[1].plot(rolling_df.index, rolling_df["rolling_vol"],
                     color="darkorange", linewidth=1)
        axes[1].set_ylabel("Rolling Volatility (ann.)")

    if "rolling_drawdown" in rolling_df.columns:
        axes[2].fill_between(rolling_df.index, rolling_df["rolling_drawdown"],
                             0, alpha=0.4, color="red")
        axes[2].set_ylabel("Rolling Drawdown")
        axes[2].set_xlabel("Date")

    if save_name:
        save_fig(fig, save_name)
    return fig


# ══════════════════════════════════════════════
# 14.  TRANSITION PROBABILITY HEATMAP
# ══════════════════════════════════════════════

def plot_transition_matrix(
    trans_mat: np.ndarray,
    state_names: Optional[List[str]] = None,
    title: str = "HMM Transition Probability Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of HMM transition probabilities."""
    n = trans_mat.shape[0]
    if state_names is None:
        state_names = [f"State {i}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(trans_mat, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=state_names, yticklabels=state_names,
                square=True, linewidths=1, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title)
    if save_name:
        save_fig(fig, save_name)
    return fig
