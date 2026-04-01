"""
feature_engineering.py — All derived feature computations.
==========================================================
Returns, multi-estimator volatility, momentum indicators, volume metrics,
cross-asset betas, macro composites, and stationarity tests.

Every feature is computed from *past* data only (no lookahead).
Sentiment features are intentionally NOT here — they live in NB06
and are lagged by t-1 before saving.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import VOL_WINDOWS, RANDOM_STATE

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 1.  RETURNS
# ══════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns: (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change()


def compute_excess_returns(
    simple_returns: pd.DataFrame,
    rf_daily: pd.Series,
) -> pd.DataFrame:
    """
    Excess return = simple_return - daily risk-free rate.

    ``rf_daily`` should already be in decimal daily form
    (e.g. annualized 5 % → ~0.05/252 per day).
    """
    return simple_returns.sub(rf_daily, axis=0)


# ══════════════════════════════════════════════
# 2.  VOLATILITY ESTIMATORS
# ══════════════════════════════════════════════

def realized_vol(
    log_returns: pd.Series | pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series | pd.DataFrame:
    """Rolling realized (close-to-close) volatility."""
    rv = log_returns.rolling(window).std()
    if annualize:
        rv *= np.sqrt(252)
    return rv


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.

    σ² = (1/4·ln2) · E[(ln(H/L))²]

    More efficient than close-to-close when only high/low are available.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    pv = np.sqrt(factor * (log_hl ** 2).rolling(window).mean())
    if annualize:
        pv *= np.sqrt(252)
    return pv


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.

    σ² = 0.5·(ln H/L)² − (2·ln2 − 1)·(ln C/O)²

    More efficient than Parkinson; uses all four OHLC prices.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    gk_vol = np.sqrt(gk.rolling(window).mean().clip(lower=0))
    if annualize:
        gk_vol *= np.sqrt(252)
    return gk_vol


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator.

    Combines overnight (close-to-open), Rogers-Satchell, and open-to-close
    components.  Minimum-variance unbiased for GBM with drift & jumps.

    σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS

    where k = 0.34 / (1.34 + (n+1)/(n-1))  and  n = window.
    """
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Overnight variance: ln(O_t / C_{t-1})
    log_oc = np.log(open_ / close.shift(1))
    var_overnight = log_oc.rolling(n).var()

    # Open-to-close variance: ln(C_t / O_t)
    log_co = np.log(close / open_)
    var_close_open = log_co.rolling(n).var()

    # Rogers-Satchell variance
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)
    rs = (log_ho * log_hc + log_lo * log_lc).rolling(n).mean()

    yz_var = var_overnight + k * var_close_open + (1 - k) * rs
    yz_vol = np.sqrt(yz_var.clip(lower=0))
    if annualize:
        yz_vol *= np.sqrt(252)
    return yz_vol


def rogers_satchell_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """
    Rogers-Satchell (1991) volatility estimator.

    σ²_RS = E[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]

    Does not depend on overnight return gaps, making it robust
    when open-to-close dynamics differ from close-to-open dynamics.

    Reference: Rogers & Satchell (1991), "Estimating Variance from
    High, Low and Closing Prices"
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)
    rs = (log_hc * log_ho + log_lc * log_lo).rolling(window).mean()
    rs_vol = np.sqrt(rs.clip(lower=0))
    if annualize:
        rs_vol *= np.sqrt(252)
    return rs_vol


def compute_all_vol_estimators(
    ohlcv: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Compute all five volatility estimators at multiple horizons.

    Estimators: Realized (close-to-close), Parkinson (high-low range),
    Garman-Klass (OHLC), Yang-Zhang (minimum variance), Rogers-Satchell
    (no overnight gap dependence).

    Parameters
    ----------
    ohlcv : DataFrame with columns Open, High, Low, Close, Adj Close.
    windows : list of rolling-window sizes (default from config).
    """
    windows = windows or VOL_WINDOWS
    log_ret = np.log(ohlcv["Adj Close"] / ohlcv["Adj Close"].shift(1))

    frames = {}
    for w in windows:
        frames[f"realized_vol_{w}d"] = realized_vol(log_ret, w)
        frames[f"parkinson_vol_{w}d"] = parkinson_vol(
            ohlcv["High"], ohlcv["Low"], w
        )
        frames[f"garman_klass_vol_{w}d"] = garman_klass_vol(
            ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], w
        )
        frames[f"yang_zhang_vol_{w}d"] = yang_zhang_vol(
            ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], w
        )
        frames[f"rogers_satchell_vol_{w}d"] = rogers_satchell_vol(
            ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], w
        )
    return pd.DataFrame(frames, index=ohlcv.index)


# ══════════════════════════════════════════════
# 3.  MOMENTUM / TECHNICAL INDICATORS
# ══════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder, 1978).

    RSI = 100 − 100 / (1 + RS)
    RS  = EMA(gains, period) / EMA(losses, period)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    MACD_line = EMA(fast) − EMA(slow)
    Signal    = EMA(MACD_line, signal)
    Histogram = MACD_line − Signal
    """
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    }, index=series.index)


def bollinger_pct_b(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger %B = (Price − Lower Band) / (Upper Band − Lower Band).

    Measures where the price is relative to the bands.
    %B > 1 → above upper band; %B < 0 → below lower band.
    """
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = upper - lower
    return (series - lower) / bandwidth.replace(0, np.nan)


def rate_of_change(series: pd.Series, period: int = 20) -> pd.Series:
    """ROC = (P_t − P_{t-n}) / P_{t-n}."""
    return series.pct_change(periods=period)


# ══════════════════════════════════════════════
# 4.  VOLUME INDICATORS
# ══════════════════════════════════════════════

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (Granville, 1963).

    OBV_t = OBV_{t-1} + sign(ΔP_t) × Volume_t
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def vwap_deviation(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Rolling VWAP deviation.

    VWAP = Σ(P_i × V_i) / Σ(V_i)   over the window.
    Deviation = (Close − VWAP) / VWAP
    """
    pv = close * volume
    vwap = pv.rolling(window).sum() / volume.rolling(window).sum()
    return (close - vwap) / vwap.replace(0, np.nan)


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of volume relative to a rolling window."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma.replace(0, np.nan)


# ══════════════════════════════════════════════
# 5.  CROSS-ASSET FEATURES
# ══════════════════════════════════════════════

def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Rolling OLS beta of asset vs. market.

    β = Cov(R_a, R_m) / Var(R_m)
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Pearson rolling correlation."""
    return series_a.rolling(window).corr(series_b)


def information_ratio(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio — risk-adjusted active return.

    IR = mean(R_a − R_b) / std(R_a − R_b) × √252

    Measures the consistency of active (excess-over-benchmark) returns.
    Higher IR → more consistent outperformance per unit of tracking error.
    """
    active = asset_returns - benchmark_returns
    active = active.dropna()
    if active.std() == 0:
        return 0.0
    return active.mean() / active.std() * np.sqrt(periods_per_year)


# ══════════════════════════════════════════════
# 5b. MICROSTRUCTURE / LIQUIDITY FEATURES
# ══════════════════════════════════════════════

def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ_t = (1/N) Σ |R_i| / V_i

    Measures price impact per unit of volume. Higher values indicate
    less liquid assets where trades move prices more.

    Reference: Amihud, Y. (2002). "Illiquidity and stock returns:
    cross-section and time-series effects." JFMA.
    """
    daily_illiq = returns.abs() / volume.replace(0, np.nan)
    return daily_illiq.rolling(window).mean()


def hurst_exponent(
    series: pd.Series,
    max_lag: int = 100,
) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    Delegates to ``statistical_tests.hurst_rs`` for the actual computation.

    H > 0.5 → persistent / trending (momentum regime)
    H = 0.5 → random walk (no memory)
    H < 0.5 → anti-persistent / mean-reverting
    """
    from src.statistical_tests import hurst_rs
    s = series.dropna().values
    return hurst_rs(s, min_window=max(max_lag // 5, 10))


def rolling_hurst(
    returns: pd.Series,
    window: int = 252,
    max_lag: int = 50,
    step: int = 5,
) -> pd.Series:
    """
    Rolling Hurst exponent computed over a sliding window.

    Returns a Series of Hurst values, one per day (NaN until enough data).
    Computes every ``step`` days and forward-fills to reduce O(n*m) cost.
    """
    result = pd.Series(np.nan, index=returns.index)
    for i in range(window, len(returns), step):
        sub = returns.iloc[i - window:i]
        result.iloc[i] = hurst_exponent(sub, max_lag=max_lag)
    # Forward-fill sparse values
    result = result.ffill()
    return result


def rolling_higher_moments(
    returns: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Rolling skewness and excess kurtosis.

    Useful as regime-change indicators: skewness shifts during
    crash regimes; kurtosis spikes indicate fat-tail events.

    Parameters
    ----------
    returns : Series of daily returns
    window : int, rolling window size

    Returns
    -------
    DataFrame with 'rolling_skew' and 'rolling_excess_kurt' columns.
    """
    return pd.DataFrame({
        "rolling_skew": returns.rolling(window).skew(),
        "rolling_excess_kurt": returns.rolling(window).kurt(),
    }, index=returns.index)


# ══════════════════════════════════════════════
# 6.  STATIONARITY TESTS
# ══════════════════════════════════════════════

def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict:
    """
    Augmented Dickey-Fuller test for unit root.

    H0: series has a unit root (non-stationary).
    Reject if p-value < 0.05.
    """
    from statsmodels.tsa.stattools import adfuller
    clean = series.dropna()
    result = adfuller(clean, maxlag=maxlag, autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "nobs": result[3],
        "critical_1pct": result[4]["1%"],
        "critical_5pct": result[4]["5%"],
        "critical_10pct": result[4]["10%"],
    }


def kpss_test(series: pd.Series, regression: str = "c") -> Dict:
    """
    KPSS test for stationarity.

    H0: series is stationary.
    Reject if p-value < 0.05.

    Complementary to ADF: ideally ADF rejects and KPSS fails to reject.
    """
    import warnings
    from statsmodels.tsa.stattools import kpss as _kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    clean = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InterpolationWarning)
        stat, p_value, lags, crit = _kpss(clean, regression=regression, nlags="auto")
    return {
        "kpss_stat": stat,
        "p_value": p_value,
        "lags_used": lags,
        "critical_1pct": crit["1%"],
        "critical_5pct": crit["5%"],
        "critical_10pct": crit["10%"],
    }


def stationarity_table(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Run ADF + KPSS on every column; return summary table."""
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 30:
            continue
        adf = adf_test(s)
        kss = kpss_test(s)
        rows.append({
            "ticker": col,
            "adf_stat": adf["adf_stat"],
            "adf_pvalue": adf["p_value"],
            "kpss_stat": kss["kpss_stat"],
            "kpss_pvalue": kss["p_value"],
            "stationary": (adf["p_value"] < 0.05) and (kss["p_value"] >= 0.05),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 7.  SUMMARY STATISTICS
# ══════════════════════════════════════════════

def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column summary: mean, std, skew, kurtosis, Jarque-Bera p-value, ADF p-value.
    """
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 30:
            continue
        jb_stat, jb_p = stats.jarque_bera(s)
        adf = adf_test(s)
        rows.append({
            "ticker": col,
            "mean_daily": s.mean(),
            "std_daily": s.std(),
            "annualized_return": s.mean() * 252,
            "annualized_vol": s.std() * np.sqrt(252),
            "skewness": s.skew(),
            "excess_kurtosis": s.kurtosis(),      # pandas .kurtosis() = excess
            "jarque_bera_stat": jb_stat,
            "jarque_bera_p": jb_p,
            "adf_stat": adf["adf_stat"],
            "adf_pvalue": adf["p_value"],
        })
    return pd.DataFrame(rows).set_index("ticker")


# ══════════════════════════════════════════════
# 8.  DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════

def compute_drawdowns(prices: pd.Series) -> pd.DataFrame:
    """
    Compute running drawdown, max drawdown, and time-to-recovery.

    Returns DataFrame with columns: cumulative_max, drawdown, duration.
    """
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax          # negative values

    # Duration: consecutive days in drawdown (vectorized)
    in_dd = (dd < 0).values
    dur = np.zeros(len(in_dd), dtype=int)
    for i in range(1, len(dur)):
        if in_dd[i]:
            dur[i] = dur[i - 1] + 1
    duration = pd.Series(dur, index=prices.index, dtype=int)

    return pd.DataFrame({
        "cumulative_max": cummax,
        "drawdown": dd,
        "duration": duration,
    }, index=prices.index)


def max_drawdown(prices: pd.Series) -> float:
    """Scalar max drawdown (most negative value)."""
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd.min()


def calmar_ratio(prices: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio = annualized return / |max drawdown|.

    Measures return per unit of drawdown risk.
    """
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    n_periods = len(prices)
    ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return np.inf
    return ann_return / mdd


# ══════════════════════════════════════════════
# 9.  MASTER FEATURE BUILDER
# ══════════════════════════════════════════════

def build_features_for_ticker(
    ohlcv: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build all engineered features for a single ticker's OHLCV data.

    Parameters
    ----------
    ohlcv : DataFrame
        Must have columns: Open, High, Low, Close, Adj Close, Volume.
    benchmark_returns : Series, optional
        SPY (or XLK) log returns for beta / correlation computation.

    Returns
    -------
    features : DataFrame
        All features aligned to the same DatetimeIndex.
    """
    adj = ohlcv["Adj Close"]
    log_ret = np.log(adj / adj.shift(1))
    simple_ret = adj.pct_change()

    feats = pd.DataFrame(index=ohlcv.index)
    feats["log_return"] = log_ret
    feats["simple_return"] = simple_ret

    # Volatility estimators at multiple horizons
    vol_df = compute_all_vol_estimators(ohlcv)
    feats = feats.join(vol_df)

    # Momentum
    feats["rsi_14"] = rsi(adj, 14)
    macd_df = macd(adj)
    feats = feats.join(macd_df)
    feats["bollinger_pctb"] = bollinger_pct_b(adj)
    feats["roc_20d"] = rate_of_change(adj, 20)

    # Volume
    feats["obv"] = on_balance_volume(adj, ohlcv["Volume"])
    feats["vwap_deviation"] = vwap_deviation(adj, ohlcv["Volume"])
    feats["volume_zscore_20d"] = volume_zscore(ohlcv["Volume"], 20)

    # Cross-asset
    if benchmark_returns is not None:
        aligned = log_ret.align(benchmark_returns, join="inner")
        feats["beta_SPY_63d"] = rolling_beta(aligned[0], aligned[1], 63)
        feats["corr_benchmark_21d"] = rolling_correlation(aligned[0], aligned[1], 21)
        feats["information_ratio_63d"] = (
            (aligned[0].rolling(63).mean() - aligned[1].rolling(63).mean())
            / (aligned[0] - aligned[1]).rolling(63).std().replace(0, np.nan)
        ) * np.sqrt(252)

    # Microstructure / Liquidity
    feats["amihud_illiq_21d"] = amihud_illiquidity(log_ret, ohlcv["Volume"], 21)

    # Higher moments
    moments = rolling_higher_moments(log_ret, 21)
    feats = feats.join(moments)

    return feats


# ══════════════════════════════════════════════
# CROSS-SECTIONAL & COINTEGRATION FEATURES
# ══════════════════════════════════════════════

def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 22,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal: 12-month return minus most recent
    1-month (12-1 month momentum), ranked across tickers.

    This is the standard Jegadeesh-Titman momentum signal used in
    asset pricing research.

    Parameters
    ----------
    prices : DataFrame (T, N)
        Adjusted close prices for N tickers.
    lookback : int
        Total lookback period (default 252 ~ 12 months).
    skip : int
        Recent period to skip (default 22 ~ 1 month).

    Returns
    -------
    DataFrame (T, N) of cross-sectional momentum signals.
    """
    # 12-month return
    ret_full = prices / prices.shift(lookback) - 1
    # 1-month return (to skip)
    ret_skip = prices / prices.shift(skip) - 1
    # 12-1 month momentum
    momentum = ret_full - ret_skip
    return momentum


def engle_granger_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
) -> dict:
    """
    Engle-Granger two-step cointegration test.

    Step 1: OLS regression  price_a = β · price_b + ε
    Step 2: ADF test on residuals ε (must be stationary for cointegration)

    Half-life = -ln(2) / ln(ρ) where ρ = AR(1) coefficient of spread.

    Parameters
    ----------
    price_a, price_b : Series of prices (same index).

    Returns
    -------
    dict with:
        beta : float — hedge ratio
        adf_stat : float — ADF test statistic on residuals
        p_value : float — ADF p-value
        half_life : float — mean-reversion half-life in days
        residuals : Series — cointegration spread
    """
    from statsmodels.tsa.stattools import adfuller

    aligned = pd.DataFrame({"a": price_a, "b": price_b}).dropna()

    if len(aligned) < 50:
        return {"beta": np.nan, "adf_stat": np.nan, "p_value": np.nan,
                "half_life": np.nan, "residuals": pd.Series(dtype=float)}

    # Step 1: OLS  a = β · b + ε
    y = aligned["a"].values
    x = aligned["b"].values
    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    intercept = y.mean() - beta * x.mean()
    residuals = y - (intercept + beta * x)
    residuals_series = pd.Series(residuals, index=aligned.index, name="spread")

    # Step 2: ADF on residuals
    adf_result = adfuller(residuals, autolag="AIC")
    adf_stat = adf_result[0]
    p_value = adf_result[1]

    # Half-life from AR(1) on the spread:  Δε_t = (ρ − 1)·ε_{t-1} + η_t
    # OLS estimate of (ρ − 1), then ρ = slope + 1.
    # WRONG formula was: corr(ε_{t-1},  Δε + ε_{t-1}) = corr(ε_{t-1}, ε_t)
    # which estimates autocorrelation, not the AR(1) mean-reversion coefficient.
    lag_res = residuals[:-1]
    delta_res = np.diff(residuals)         # Δε_t = ε_t − ε_{t-1}
    if len(lag_res) > 2 and np.var(lag_res) > 0:
        # OLS: slope of Δε ~ ε_{t-1}   →   ρ_hat = slope + 1
        slope = np.cov(lag_res, delta_res)[0, 1] / np.var(lag_res, ddof=1)
        rho = slope + 1.0
        # half_life = −ln(2) / ln(|ρ|); guard against |ρ| ≥ 1 (unit root)
        if 0 < abs(rho) < 1:
            half_life = -np.log(2) / np.log(abs(rho))
        else:
            half_life = np.inf  # non-stationary spread
    else:
        half_life = np.nan

    return {
        "beta": float(beta),
        "intercept": float(intercept),
        "adf_stat": float(adf_stat),
        "p_value": float(p_value),
        "half_life": float(half_life),
        "residuals": residuals_series,
    }


def lead_lag_crosscorr(
    returns_a: pd.Series,
    returns_b: pd.Series,
    max_lag: int = 5,
) -> pd.Series:
    """
    Lead-lag cross-correlation between two return series.

    ρ_{a,b}(k) = corr(r_{a,t}, r_{b,t+k}) for k ∈ [-max_lag, +max_lag]

    Positive k: b leads a (b at t predicts a at t+k).
    Negative k: a leads b.

    Useful for identifying systematic lead-lag relationships
    (e.g., NVDA leads AMD by 1 day).

    Parameters
    ----------
    returns_a, returns_b : Series of returns.
    max_lag : int — maximum lag in either direction.

    Returns
    -------
    Series indexed by lag (-max_lag to +max_lag).
    """
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    a = aligned["a"].values
    b = aligned["b"].values

    lags = range(-max_lag, max_lag + 1)
    correlations = {}

    for k in lags:
        if k >= 0:
            corr = np.corrcoef(a[:len(a) - k] if k > 0 else a,
                               b[k:] if k > 0 else b)[0, 1]
        else:
            corr = np.corrcoef(a[-k:], b[:len(b) + k])[0, 1]
        correlations[k] = corr

    return pd.Series(correlations, name=f"{returns_a.name}_vs_{returns_b.name}")


def information_coefficient(
    signal: pd.Series,
    forward_return: pd.Series,
) -> float:
    """
    Information Coefficient — Spearman rank correlation between
    a cross-sectional signal and realized forward returns.

    IC > 0.05 is generally considered economically meaningful.
    IC > 0.10 is strong.

    Parameters
    ----------
    signal : Series — cross-sectional signal at time t (e.g., momentum rank).
    forward_return : Series — realized forward return over next period.

    Returns
    -------
    ic : float — Spearman rank correlation.
    """
    aligned = pd.DataFrame({"signal": signal, "fwd_ret": forward_return}).dropna()
    if len(aligned) < 5:
        return np.nan

    rho, _ = stats.spearmanr(aligned["signal"], aligned["fwd_ret"])
    return float(rho)
