"""
garch_utils.py — GARCH family fitting, model selection, extraction.
====================================================================
Fits GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1) and FIGARCH for each
ticker.  Compares innovation distributions (Normal, Student-t,
Skewed-t, GED) via log-likelihood.  Selects best model by AIC/BIC.

References
----------
* Bollerslev (1986)  — GARCH
* Glosten, Jagannathan & Runkle (1993) — GJR-GARCH
* Nelson (1991) — EGARCH
* Baillie, Bollerslev & Mikkelsen (1996) — FIGARCH
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.config import (
    COND_VOL_FILE,
    GARCH_DIR,
    GARCH_MAX_ITER,
    GARCH_PARAMS_FILE,
    MIN_OBS_FIGARCH,
    SHORT_HISTORY_TICKERS,
    TICKERS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 1.  Model specifications
# ──────────────────────────────────────────────
# Each spec is a dict passed to arch_model() plus a human-readable label.
# GJR-GARCH: vol='GARCH', o=1 adds the asymmetric (leverage) term.

MODEL_SPECS: Dict[str, dict] = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
    # FIGARCH handled separately because of obs-count guard
}

DISTRIBUTIONS: List[str] = ["normal", "t", "skewt", "ged"]

# ──────────────────────────────────────────────
# 2.  Single model fit
# ──────────────────────────────────────────────

def fit_garch(
    returns: pd.Series,
    vol: str = "GARCH",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "skewt",
    max_iter: int = GARCH_MAX_ITER,
    rescale: bool = True,
) -> Optional[ARCHModelResult]:
    """
    Fit a single GARCH-family model.

    Parameters
    ----------
    returns : pd.Series
        Daily return series (NOT prices).  Must be × 100 if ``rescale=True``
        is *not* used (``arch`` works better with percentage returns).
    vol : str
        Variance model type: 'GARCH', 'EGARCH', 'FIGARCH'.
    p, o, q : int
        Lag orders.  ``o`` is the asymmetric-order (set o=1 for GJR/EGARCH).
    dist : str
        Innovation distribution: 'normal', 't', 'skewt', 'ged'.
    rescale : bool
        If True, multiply returns by 100 before fitting (recommended).

    Returns
    -------
    ARCHModelResult or None on convergence failure.
    """
    data = returns.dropna()
    if rescale:
        data = data * 100.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = arch_model(
                data,
                mean="Constant",
                vol=vol,
                p=p, o=o, q=q,
                dist=dist,
                rescale=False,
            )
            result = model.fit(disp="off", options={"maxiter": max_iter})
            return result
        except Exception as exc:
            logger.warning("GARCH fit failed (%s p=%d o=%d q=%d dist=%s): %s",
                           vol, p, o, q, dist, exc)
            return None


# ──────────────────────────────────────────────
# 3.  Best distribution selection
# ──────────────────────────────────────────────

def select_best_distribution(
    returns: pd.Series,
    vol: str = "GARCH",
    p: int = 1,
    o: int = 1,
    q: int = 1,
) -> Tuple[str, ARCHModelResult]:
    """
    Fit the same variance spec under all distributions;
    return the one with lowest BIC (per CLAUDE.md §NB03 model selection).
    """
    best_bic = np.inf
    best_dist = "normal"
    best_result = None

    for dist in DISTRIBUTIONS:
        res = fit_garch(returns, vol=vol, p=p, o=o, q=q, dist=dist)
        if res is not None and res.bic < best_bic:
            best_bic = res.bic
            best_dist = dist
            best_result = res

    if best_result is None:
        raise RuntimeError("All distribution fits failed")
    logger.info("Best dist for %s(%d,%d,%d): %s  (BIC=%.2f)",
                vol, p, o, q, best_dist, best_bic)
    return best_dist, best_result


# ──────────────────────────────────────────────
# 4.  Full model-selection sweep (per ticker)
# ──────────────────────────────────────────────

def fit_all_models_for_ticker(
    returns: pd.Series,
    ticker: str,
    include_figarch: bool = True,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], ARCHModelResult]]:
    """
    Fit GARCH, GJR, EGARCH (and optionally FIGARCH) across all distributions
    for one ticker.  Returns a comparison table and a dict of fitted results.

    FIGARCH is skipped for tickers with < MIN_OBS_FIGARCH observations
    (PLTR, DDOG, CRWD per CLAUDE.md §NB03).

    Returns
    -------
    comparison : DataFrame — summary table of all fits
    fitted_results : dict — {(model_name, dist): ARCHModelResult}
    """
    n_obs = returns.dropna().shape[0]
    skip_figarch = (
        not include_figarch
        or ticker in SHORT_HISTORY_TICKERS
        or n_obs < MIN_OBS_FIGARCH
    )

    rows = []
    fitted_results: Dict[Tuple[str, str], ARCHModelResult] = {}
    for model_name, spec in MODEL_SPECS.items():
        for dist in DISTRIBUTIONS:
            res = fit_garch(returns, dist=dist, **spec)
            if res is None:
                continue
            rows.append(_extract_row(res, ticker, model_name, dist))
            fitted_results[(model_name, dist)] = res

    # FIGARCH
    if not skip_figarch:
        for dist in DISTRIBUTIONS:
            res = fit_garch(returns, vol="FIGARCH", p=1, o=0, q=1, dist=dist)
            if res is not None:
                rows.append(_extract_row(res, ticker, "FIGARCH(1,d,1)", dist))
                fitted_results[("FIGARCH(1,d,1)", dist)] = res
    else:
        logger.info("Skipping FIGARCH for %s (n_obs=%d, threshold=%d)",
                     ticker, n_obs, MIN_OBS_FIGARCH)

    return pd.DataFrame(rows), fitted_results


def _extract_row(
    result: ARCHModelResult,
    ticker: str,
    model_name: str,
    dist: str,
) -> dict:
    """Extract summary row from fitted result."""
    params = result.params.to_dict()
    return {
        "ticker": ticker,
        "model": model_name,
        "distribution": dist,
        "aic": result.aic,
        "bic": result.bic,
        "log_likelihood": result.loglikelihood,
        "num_params": result.num_params,
        "n_obs": result.nobs,
        **{f"param_{k}": v for k, v in params.items()},
    }


# ──────────────────────────────────────────────
# 5.  Best model per ticker
# ──────────────────────────────────────────────

def select_best_model(comparison_df: pd.DataFrame, criterion: str = "bic") -> pd.Series:
    """Return the row with lowest AIC or BIC from the comparison table."""
    idx = comparison_df[criterion].idxmin()
    return comparison_df.loc[idx]


# ──────────────────────────────────────────────
# 6.  Conditional volatility extraction
# ──────────────────────────────────────────────

def extract_conditional_vol(
    result: ARCHModelResult,
    rescale: bool = True,
) -> pd.Series:
    """
    Extract annualized conditional volatility from a fitted model.

    If the model was fit on returns × 100, the conditional variance is
    in units of (% return)²; we divide by 10 000 to get decimal variance,
    then take sqrt and annualize.
    """
    cond_var = result.conditional_volatility ** 2
    if rescale:
        # Model was fit on returns × 100 → variance in (%²)
        cond_var = cond_var / 10_000.0
    cond_vol = np.sqrt(cond_var) * np.sqrt(252)
    cond_vol.name = "cond_vol_annualized"
    return cond_vol


# ──────────────────────────────────────────────
# 7.  Standardized-residual diagnostics
# ──────────────────────────────────────────────

def residual_diagnostics(result: ARCHModelResult) -> Dict:
    """
    Run Ljung-Box on squared standardized residuals and ARCH-LM test.

    A well-specified GARCH model should produce standardized residuals
    whose squares are white noise (no remaining ARCH effects).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    std_resid = result.std_resid.dropna()

    # Ljung-Box on squared residuals
    lb = acorr_ljungbox(std_resid ** 2, lags=[10, 20], return_df=True)

    # ARCH-LM test
    arch_lm_stat, arch_lm_p, _, _ = het_arch(std_resid, nlags=10)

    return {
        "ljung_box_10": lb["lb_pvalue"].iloc[0],
        "ljung_box_20": lb["lb_pvalue"].iloc[1],
        "arch_lm_stat": arch_lm_stat,
        "arch_lm_pvalue": arch_lm_p,
    }


# ──────────────────────────────────────────────
# 8.  Full pipeline (all 20 tickers)
# ──────────────────────────────────────────────

def run_full_garch_pipeline(
    returns_dict: Dict[str, pd.Series],
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run GARCH model selection across all tickers.

    Parameters
    ----------
    returns_dict : dict
        {ticker: pd.Series of daily log returns}.

    Returns
    -------
    params_table : DataFrame
        Full comparison table (20 tickers × ~16 model-dist combos).
    cond_vol_panel : DataFrame
        (T × 20) conditional volatility from each ticker's best model.
    """
    all_tables = []
    vol_series = {}

    for ticker, rets in returns_dict.items():
        logger.info("Fitting GARCH models for %s ...", ticker)
        comp, fitted_results = fit_all_models_for_ticker(rets, ticker)
        all_tables.append(comp)

        # Pick best model by BIC and reuse its cached fitted result
        best = select_best_model(comp, criterion="bic")
        key = (best["model"], best["distribution"])
        best_res = fitted_results.get(key)
        if best_res is not None:
            vol_series[ticker] = extract_conditional_vol(best_res)

    params_table = pd.concat(all_tables, ignore_index=True)
    cond_vol_panel = pd.DataFrame(vol_series)

    if save:
        params_table.to_csv(GARCH_PARAMS_FILE, index=False)
        cond_vol_panel.to_parquet(COND_VOL_FILE)
        logger.info("Saved %s and %s", GARCH_PARAMS_FILE, COND_VOL_FILE)

    return params_table, cond_vol_panel


# ──────────────────────────────────────────────
# 9.  NEWS IMPACT CURVE
# ──────────────────────────────────────────────

def news_impact_curve(
    result: ARCHModelResult,
    model_name: str,
    shock_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    News Impact Curve — shows how σ²_{t+1} responds to a shock ε_t.

    Plots conditional variance as a function of the standardized shock
    at t, holding all other values at their unconditional levels.

    For a symmetric GARCH, the curve is a parabola centered at ε=0.
    For GJR-GARCH, negative shocks produce higher volatility (leverage effect).
    For EGARCH, the asymmetry is exponential.

    Parameters
    ----------
    result : ARCHModelResult
        Fitted GARCH model.
    model_name : str
        'GARCH(1,1)', 'GJR-GARCH(1,1)', or 'EGARCH(1,1)'.
    shock_range : ndarray
        Range of standardized shocks (default: -4 to +4 in 0.1 steps).

    Returns
    -------
    DataFrame with columns 'shock' and 'conditional_variance'.

    Reference
    ---------
    Engle & Ng (1993), "Measuring and Testing the Impact of News on Volatility"
    """
    if shock_range is None:
        shock_range = np.linspace(-4, 4, 81)

    params = result.params
    omega = params.get("omega", 0)

    if "EGARCH" in model_name:
        alpha = params.get("alpha[1]", 0)
        gamma = params.get("gamma[1]", 0)
        beta = params.get("beta[1]", 0)
        # EGARCH: ln(σ²) = ω + α|z| + γ·z + β·ln(σ̄²)
        # E[ln(σ²)] = (ω + α·E[|z|]) / (1 - β)
        # Use empirical E[|z|] from standardized residuals (correct for any distribution)
        e_abs_z = np.mean(np.abs(result.std_resid.dropna()))
        if abs(1 - beta) > 1e-6:
            ln_sigma2_bar = (omega + alpha * e_abs_z) / (1 - beta)
        else:
            ln_sigma2_bar = np.log(result.conditional_volatility.mean() ** 2)
        cond_var = np.exp(omega + alpha * np.abs(shock_range) +
                          gamma * shock_range + beta * ln_sigma2_bar)
    elif "GJR" in model_name:
        alpha = params.get("alpha[1]", 0)
        gamma = params.get("gamma[1]", 0)
        beta = params.get("beta[1]", 0)
        # GJR: σ² = ω + (α + γ·1_{ε<0})·ε² + β·σ̄²
        # shock_range is standardized (z), so ε = z·σ̄ → ε² = z²·σ̄²
        sigma2_bar = omega / max(1 - alpha - beta - gamma / 2, 1e-6)
        indicator = (shock_range < 0).astype(float)
        cond_var = omega + (alpha + gamma * indicator) * (shock_range ** 2 * sigma2_bar) + beta * sigma2_bar
    else:
        # Standard GARCH(1,1): σ² = ω + α·ε² + β·σ̄²
        # shock_range is standardized (z), so ε² = z²·σ̄²
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        sigma2_bar = omega / max(1 - alpha - beta, 1e-6)
        cond_var = omega + alpha * (shock_range ** 2 * sigma2_bar) + beta * sigma2_bar

    return pd.DataFrame({"shock": shock_range, "conditional_variance": cond_var})


# ──────────────────────────────────────────────
# 10. VOLATILITY PERSISTENCE & HALF-LIFE
# ──────────────────────────────────────────────

def volatility_persistence(
    result: ARCHModelResult,
    model_name: str,
) -> float:
    """
    Volatility persistence — rate at which volatility shocks decay.

    GARCH(1,1):     P = α + β
    GJR-GARCH(1,1): P = α + β + γ/2
    EGARCH(1,1):    P = β  (approximate)

    P close to 1 → highly persistent (near IGARCH).
    P = 1 → integrated GARCH (shocks never decay).

    Reference: Engle & Bollerslev (1986)
    """
    params = result.params
    if "EGARCH" in model_name:
        return float(params.get("beta[1]", 0))
    alpha = float(params.get("alpha[1]", 0))
    beta = float(params.get("beta[1]", 0))
    gamma = float(params.get("gamma[1]", 0))
    if "GJR" in model_name:
        return alpha + beta + gamma / 2
    return alpha + beta


def vol_half_life(persistence: float) -> float:
    """
    Half-life of volatility shocks in trading days.

    t_{1/2} = ln(2) / ln(1/P)

    where P = volatility persistence.

    Interpretation: number of days for a shock to decay by 50%.
    E.g., persistence = 0.98 → half-life ≈ 34 days.

    Returns np.inf if persistence >= 1 (integrated process).
    """
    if persistence >= 1.0:
        return np.inf
    if persistence <= 0.0:
        return 0.0
    return np.log(2) / np.log(1.0 / persistence)


def unconditional_variance(
    result: ARCHModelResult,
    model_name: str,
    annualize: bool = True,
    rescale: bool = True,
) -> float:
    """
    Long-run (unconditional) variance from GARCH parameters.

    GARCH(1,1):     σ²_∞ = ω / (1 − α − β)
    GJR-GARCH(1,1): σ²_∞ = ω / (1 − α − β − γ/2)

    Valid only if persistence < 1 (stationary process).

    Parameters
    ----------
    rescale : bool
        If True (model fit on returns×100), convert back to decimal variance.
    annualize : bool
        If True, multiply by 252 and return annualized volatility.

    Returns
    -------
    Annualized unconditional volatility (or daily variance if annualize=False).
    """
    params = result.params
    omega = float(params.get("omega", 0))

    if "EGARCH" in model_name:
        # EGARCH unconditional variance = exp(E[ln(σ²)])
        # E[ln(σ²)] = (ω + α·E[|z|]) / (1 - β)
        # Use empirical E[|z|] from standardized residuals (correct for any distribution)
        alpha_e = float(params.get("alpha[1]", 0))
        beta_e = float(params.get("beta[1]", 0))
        if abs(beta_e) >= 1.0:
            logger.warning("EGARCH persistence >= 1; unconditional variance undefined")
            return np.inf
        e_abs_z = float(np.mean(np.abs(result.std_resid.dropna())))
        ln_var = (omega + alpha_e * e_abs_z) / (1.0 - beta_e)
        var = np.exp(ln_var)
    else:
        p = volatility_persistence(result, model_name)
        if p >= 1.0:
            logger.warning("Persistence >= 1; unconditional variance is undefined")
            return np.inf
        var = omega / (1 - p)

    if rescale:
        var /= 10_000  # model was fit on returns × 100
    if annualize:
        return np.sqrt(var * 252)
    return var


def forecast_volatility(
    result: ARCHModelResult,
    horizon: int = 21,
    rescale: bool = True,
) -> pd.Series:
    """
    Produce h-step ahead conditional volatility forecasts.

    Uses the ``arch`` model's built-in forecast method.

    Parameters
    ----------
    result : ARCHModelResult
    horizon : int
        Number of days to forecast ahead.
    rescale : bool
        Convert from percentage² back to decimal variance.

    Returns
    -------
    Series of length ``horizon`` with annualized vol forecasts.
    """
    fcast = result.forecast(horizon=horizon)
    var_fcast = fcast.variance.iloc[-1].values  # last row = forecast from T
    if rescale:
        var_fcast = var_fcast / 10_000
    vol_fcast = np.sqrt(var_fcast) * np.sqrt(252)
    return pd.Series(vol_fcast, index=range(1, horizon + 1), name="vol_forecast")
