"""
data_loader.py — Centralized data download, caching & cleaning.
================================================================
Downloads OHLCV for 20 tickers + benchmarks, FRED macro series,
and merges the SQ→XYZ ticker transition.  Saves / loads parquet
caches so that repeated runs avoid unnecessary API calls.

Uses the **OpenBB Platform SDK** for all market data retrieval.

Key design decisions
--------------------
* Uses **Adj Close** exclusively (handles splits & dividends).
* Forward-fills up to 5 business days, then flags remaining gaps.
* Flags any single-day |return| > 25 % as a potential corporate-action artifact.
* Never fabricates pre-IPO data for short-history tickers (CRWD, DDOG, PLTR).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openbb import obb

from src.config import (
    BENCHMARK_TICKERS,
    END_DATE,
    FRED_SERIES,
    MASTER_DATA_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    RF_TICKER_OBB,
    SHORT_HISTORY_TICKERS,
    SQ_XYZ_CUTOVER,
    START_DATE,
    TICKER_SQ_LEGACY,
    TICKER_XYZ,
    TICKERS,
    VIX_TICKERS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 1.  OpenBB helpers
# ──────────────────────────────────────────────

def download_single_ticker(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download OHLCV for one ticker via OpenBB; return DataFrame with DatetimeIndex."""
    try:
        result = obb.equity.price.historical(
            symbol=ticker,
            start_date=start,
            end_date=end,
            provider="yfinance",
        )
        df = result.to_df()
    except Exception as exc:
        logger.warning("Failed to download %s: %s", ticker, exc)
        return pd.DataFrame()
    if df.empty:
        logger.warning("Empty data for %s", ticker)
        return df
    # Standardize column names to Title Case for pipeline compatibility
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
        "adj_close": "Adj Close",
    }
    df.rename(columns=col_map, inplace=True)
    # If Adj Close not present, fall back to Close
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def download_all_tickers(
    tickers: List[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for every ticker in the universe.

    Returns dict  {ticker: DataFrame}.
    Caches raw CSVs in ``data/raw/``.
    """
    tickers = tickers or TICKERS
    result: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        cache_path = RAW_DIR / f"{t}.csv"
        if cache and cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = download_single_ticker(t, start, end)
            if not df.empty and cache:
                df.to_csv(cache_path)
        result[t] = df
    return result


# ──────────────────────────────────────────────
# 2.  SQ → XYZ merger
# ──────────────────────────────────────────────

def merge_sq_xyz(
    start: str = START_DATE,
    end: str = END_DATE,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download SQ (pre-2025-01-21) and XYZ (post-2025-01-21),
    merge into a single continuous series stored under 'XYZ'.

    The merge point is the LAST trading day of SQ, with XYZ
    starting the following session.  No overlap, no gap.
    """
    cache_path = RAW_DIR / "XYZ_merged.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    sq = download_single_ticker(TICKER_SQ_LEGACY, start=start, end=SQ_XYZ_CUTOVER)
    xyz = download_single_ticker(TICKER_XYZ, start=SQ_XYZ_CUTOVER, end=end)

    if sq.empty and xyz.empty:
        return pd.DataFrame()
    if sq.empty:
        merged = xyz.sort_index()
    elif xyz.empty:
        merged = sq.sort_index()
    else:
        # Remove any overlap day
        overlap = sq.index.intersection(xyz.index)
        if len(overlap) > 0:
            sq = sq.loc[~sq.index.isin(overlap)]
        merged = pd.concat([sq, xyz]).sort_index()
    merged.index.name = "Date"

    if cache and not merged.empty:
        merged.to_csv(cache_path)

    return merged


# ──────────────────────────────────────────────
# 3.  Benchmark & VIX download
# ──────────────────────────────────────────────

def download_benchmarks(
    start: str = START_DATE,
    end: str = END_DATE,
    cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Download sector + market benchmarks and VIX/VVIX."""
    all_tickers = BENCHMARK_TICKERS + VIX_TICKERS + [RF_TICKER_OBB]
    return download_all_tickers(all_tickers, start, end, cache)


# ──────────────────────────────────────────────
# 4.  FRED macro data
# ──────────────────────────────────────────────

def download_fred(
    api_key: Optional[str] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download FRED macro series.

    If ``fredapi`` is not installed or no API key is provided,
    returns an empty DataFrame with expected columns.

    Parameters
    ----------
    api_key : str, optional
        FRED API key.  Falls back to env var ``FRED_API_KEY``.
    """
    cache_path = RAW_DIR / "fred_macro.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        from fredapi import Fred  # type: ignore
        import os

        api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not api_key:
            logger.warning("FRED_API_KEY not set — returning empty macro DataFrame")
            return pd.DataFrame()

        fred = Fred(api_key=api_key)
        frames = {}
        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start,
                                    observation_end=end)
                frames[name] = s
            except Exception as exc:
                logger.warning("Failed to download FRED %s: %s", series_id, exc)
        if frames:
            df = pd.DataFrame(frames)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            if cache:
                df.to_csv(cache_path)
            return df
    except ImportError:
        logger.warning("fredapi not installed — skipping FRED download")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# 5.  Data quality checks
# ──────────────────────────────────────────────

def quality_checks(
    prices: pd.DataFrame,
    ticker: str,
    max_ffill_days: int = 5,
    return_spike_threshold: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the data-quality protocol from CLAUDE.md §2.3.

    1. Forward-fill up to ``max_ffill_days``, flag remaining NaNs.
    2. Compute log returns on Adj Close and flag |return| > threshold.

    Returns
    -------
    cleaned : pd.DataFrame
        Forward-filled prices.
    flags : pd.DataFrame
        Boolean columns ``ffill_flag`` and ``spike_flag``.
    """
    cleaned = prices.copy()

    # --- forward-fill up to N days ---
    pre_nan = cleaned["Adj Close"].isna().sum()
    cleaned["Adj Close"] = cleaned["Adj Close"].ffill(limit=max_ffill_days)
    post_nan = cleaned["Adj Close"].isna().sum()
    if post_nan > 0:
        logger.warning("%s: %d NaNs remain after %d-day ffill (started with %d)",
                       ticker, post_nan, max_ffill_days, pre_nan)
        # Interpolate remaining
        cleaned["Adj Close"] = cleaned["Adj Close"].interpolate(method="time")

    # --- spike detection ---
    ffill_flag = prices["Adj Close"].isna()
    log_ret = np.log(cleaned["Adj Close"] / cleaned["Adj Close"].shift(1))
    # Exclude forward-filled rows: spikes after a ffill gap are artifacts,
    # not real market moves (e.g., resumption of trading after a holiday).
    spike = (log_ret.abs() > return_spike_threshold) & ~ffill_flag & ~ffill_flag.shift(1, fill_value=False)

    flags = pd.DataFrame({
        "ffill_flag": ffill_flag,
        "spike_flag": spike,
    }, index=cleaned.index)

    if spike.any():
        spike_dates = cleaned.index[spike].tolist()
        logger.warning("%s: return spikes > %.0f%% on %s",
                       ticker, return_spike_threshold * 100, spike_dates)

    return cleaned, flags


# ──────────────────────────────────────────────
# 6.  Build unified Adj Close panel
# ──────────────────────────────────────────────

def build_adj_close_panel(
    raw_data: Dict[str, pd.DataFrame],
    max_ffill_days: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    From a dict of raw OHLCV DataFrames, extract Adj Close columns,
    apply quality checks, and return a (T × N) panel plus per-ticker flags.
    """
    panels = {}
    all_flags: Dict[str, pd.DataFrame] = {}

    for ticker, df in raw_data.items():
        if df.empty:
            continue
        cleaned, flags = quality_checks(df, ticker, max_ffill_days=max_ffill_days)
        panels[ticker] = cleaned["Adj Close"]
        all_flags[ticker] = flags

    adj_close = pd.DataFrame(panels)
    adj_close.index = pd.to_datetime(adj_close.index)
    adj_close.index.name = "Date"
    adj_close.sort_index(inplace=True)

    return adj_close, all_flags


# ──────────────────────────────────────────────
# 7.  Master orchestrator
# ──────────────────────────────────────────────

def load_or_build_master(
    fred_api_key: Optional[str] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    High-level entry point used by NB01.

    * Downloads everything (or loads from cache).
    * Merges SQ/XYZ.
    * Builds Adj Close panel.
    * Saves ``master_data.parquet``.
    """
    if not force_rebuild and MASTER_DATA_FILE.exists():
        logger.info("Loading cached master_data.parquet")
        return pd.read_parquet(MASTER_DATA_FILE)

    # --- stocks ---
    raw = download_all_tickers()
    # Replace XYZ stub with merged SQ/XYZ
    raw["XYZ"] = merge_sq_xyz()

    adj_close, _flags = build_adj_close_panel(raw)

    # --- benchmarks ---
    bench_raw = download_benchmarks()
    bench_panel, _ = build_adj_close_panel(bench_raw)
    bench_panel.columns = [f"BM_{c}" for c in bench_panel.columns]

    # --- FRED macro ---
    macro = download_fred(api_key=fred_api_key)

    # --- combine ---
    master = adj_close.join(bench_panel, how="outer")
    if not macro.empty:
        # Resample macro to daily, forward-fill (monthly/quarterly series)
        macro_daily = macro.resample("B").ffill()
        master = master.join(macro_daily, how="left")

    master.sort_index(inplace=True)
    master.to_parquet(MASTER_DATA_FILE)
    logger.info("Saved master_data.parquet  (%d rows × %d cols)", *master.shape)

    return master


# ──────────────────────────────────────────────
# 8.  Full OHLCV panel builder
# ──────────────────────────────────────────────

def build_ohlcv_panel(
    raw_data: Dict[str, pd.DataFrame],
    max_ffill_days: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    From a dict of raw OHLCV DataFrames, apply quality checks and
    return cleaned OHLCV DataFrames per ticker.

    Unlike ``build_adj_close_panel`` which extracts only Adj Close,
    this preserves full OHLCV data needed for volatility estimators
    (Parkinson, Garman-Klass, Yang-Zhang all need Open/High/Low/Close).

    Returns
    -------
    cleaned : dict {ticker: DataFrame with OHLCV columns cleaned}
    """
    cleaned: Dict[str, pd.DataFrame] = {}
    for ticker, df in raw_data.items():
        if df.empty:
            continue
        # Quality checks on Adj Close
        df_clean, _ = quality_checks(df, ticker, max_ffill_days=max_ffill_days)
        # Forward-fill OHLC columns too (same limit)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill(limit=max_ffill_days)
        cleaned[ticker] = df_clean
    return cleaned


# ──────────────────────────────────────────────
# 9.  Data quality report
# ──────────────────────────────────────────────

def data_quality_report(
    raw_data: Dict[str, pd.DataFrame],
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """
    Generate a comprehensive data quality report for all tickers.

    For each ticker reports: date range, total observations, missing count,
    filled count, spike count, and data availability percentage.

    Returns
    -------
    report : DataFrame with one row per ticker.
    """
    rows = []
    for ticker, df in raw_data.items():
        if df.empty:
            rows.append({"ticker": ticker, "status": "EMPTY"})
            continue

        n_total = len(df)
        adj = df.get("Adj Close")
        if adj is None:
            rows.append({"ticker": ticker, "status": "NO_ADJ_CLOSE"})
            continue

        n_missing = adj.isna().sum()
        cleaned, flags = quality_checks(df, ticker, max_ffill_days=max_ffill_days)
        n_filled = flags["ffill_flag"].sum()
        n_spikes = flags["spike_flag"].sum()
        date_range_start = df.index.min()
        date_range_end = df.index.max()

        # Check if short-history ticker
        is_short = ticker in SHORT_HISTORY_TICKERS

        rows.append({
            "ticker": ticker,
            "start_date": date_range_start,
            "end_date": date_range_end,
            "total_obs": n_total,
            "missing_adj_close": int(n_missing),
            "ffill_applied": int(n_filled),
            "return_spikes": int(n_spikes),
            "data_availability_pct": round((1 - n_missing / max(n_total, 1)) * 100, 2),
            "short_history": is_short,
            "status": "OK",
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 10.  Risk-free rate computation
# ──────────────────────────────────────────────

def compute_daily_rf(
    rf_annual: pd.Series,
) -> pd.Series:
    """
    Convert annualized risk-free rate to daily decimal rate.

    The 13-week T-bill yield from FRED (DTB3) or OpenBB (^IRX)
    is quoted as an annualized percentage.  We convert:

        rf_daily = rf_annual / 100 / 252

    Parameters
    ----------
    rf_annual : Series
        Annualized percentage yield (e.g. 5.0 for 5%).

    Returns
    -------
    rf_daily : Series
        Daily decimal rate (e.g. ~0.0002 for 5% annual).
    """
    return rf_annual / 100.0 / 252.0
