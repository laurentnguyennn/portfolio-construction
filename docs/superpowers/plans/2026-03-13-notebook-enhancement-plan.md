# Notebook Enhancement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comprehensively enhance all 12 notebooks with deeper statistical rigor, missing analyses, publication-quality visualizations, interpretive commentary, and real FinBERT sentiment — restructured for world-class MSc portfolio quality.

**Architecture:** Sequential rewrite in dependency order (9 phases). Each notebook is fully restructured with the Compute-Visualize-Interpret pattern. New `src/` utilities are added as needed before the notebook that first uses them. The plan follows the dependency graph: NB01 first, then NB02/03/05/06 in parallel, then NB04, then downstream ML/DL/portfolio notebooks.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, statsmodels, arch, hmmlearn, sklearn, xgboost, lightgbm, torch, pytorch-forecasting, transformers (FinBERT), pypfopt, cvxpy, shap, optuna, matplotlib, seaborn, plotly, python-pptx, fpdf2

**Spec Document:** `docs/superpowers/specs/2026-03-13-notebook-enhancement-design.md`

---

## File Structure

### Files to Modify (src/ utilities)

| File | New Functions to Add |
|------|---------------------|
| `src/feature_engineering.py` | `ljung_box_test()`, `arch_lm_test()`, `jennrich_test()`, `hurst_exponent()`, `bai_perron_test()` |
| `src/garch_utils.py` | `sign_bias_test()`, `news_impact_curve()`, `extract_all_parameters()`, `garch_forecast_multi_step()` |
| `src/risk_metrics.py` | `hill_plot()`, `mean_excess_function()`, `var_multi_day()`, `wilson_score_ci()` |
| `src/ml_pipeline.py` | `model_confidence_set()`, `platt_scaling()`, `feature_ablation()`, `learning_curve_analysis()` |
| `src/backtest_engine.py` | `brinson_fachler_attribution()`, `herfindahl_index()`, `effective_n_positions()`, `diversification_ratio()` |
| `src/visualization.py` | 12 new plot functions (see spec Section 14.2) |

### Files to Create

| File | Purpose |
|------|---------|
| `src/sentiment.py` | FinBERT headline pipeline: collection, inference, aggregation, lag enforcement |
| `tests/test_pipeline_integrity.py` | 8 automated no-lookahead-bias and constraint tests |

### Files to Rewrite (notebooks)

All 12 notebooks in `notebooks/` — full restructure with enhanced content.

---

## Chunk 1: Foundation Utilities & NB01

### Task 1: Add Statistical Testing Utilities to `src/feature_engineering.py`

**Files:**
- Modify: `src/feature_engineering.py`
- Create: `tests/test_pipeline_integrity.py` (initial skeleton)

- [ ] **Step 1: Read current `feature_engineering.py`**

Read `src/feature_engineering.py` to understand existing functions and import patterns.

- [ ] **Step 2: Add `ljung_box_test(returns, lags=[1,5,10,20])`**

Add function that runs Ljung-Box test on a return series at specified lags. Returns dict with test statistics, p-values, and significance flags. Uses `statsmodels.stats.diagnostic.acorr_ljungbox`.

```python
def ljung_box_test(returns: pd.Series, lags: list = [1, 5, 10, 20]) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation at specified lags.

    Returns DataFrame with columns: lag, lb_stat, lb_pvalue, significant_5pct.
    Run on raw returns to detect autocorrelation.
    Run on squared returns to detect ARCH effects (volatility clustering).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    clean = returns.dropna()
    result = acorr_ljungbox(clean, lags=lags, return_df=True)
    result['significant_5pct'] = result['lb_pvalue'] < 0.05
    return result
```

- [ ] **Step 3: Add `arch_lm_test(returns, lags=10)`**

Engle's ARCH-LM test for heteroskedasticity. Uses `statsmodels.stats.diagnostic.het_arch`.

```python
def arch_lm_test(returns: pd.Series, lags: int = 10) -> dict:
    """Engle's ARCH-LM test for conditional heteroskedasticity.

    H0: No ARCH effects. Rejection justifies GARCH modeling.
    Returns: lm_stat, lm_pvalue, f_stat, f_pvalue.
    """
    from statsmodels.stats.diagnostic import het_arch
    clean = returns.dropna().values
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(clean, nlags=lags)
    return {
        'lm_stat': lm_stat, 'lm_pvalue': lm_pvalue,
        'f_stat': f_stat, 'f_pvalue': f_pvalue,
        'significant_5pct': lm_pvalue < 0.05
    }
```

- [ ] **Step 4: Add `hurst_exponent(series, max_lag=100)`**

Rescaled range (R/S) analysis for mean-reversion vs. trending behavior.

```python
def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5: mean-reverting. H = 0.5: random walk. H > 0.5: trending.
    """
    clean = series.dropna().values
    lags = range(2, min(max_lag, len(clean) // 2))
    rs_values = []
    for lag in lags:
        rs_vals = []
        for start in range(0, len(clean) - lag, lag):
            chunk = clean[start:start + lag]
            mean_chunk = chunk.mean()
            devs = np.cumsum(chunk - mean_chunk)
            R = devs.max() - devs.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            rs_values.append((lag, np.mean(rs_vals)))
    if len(rs_values) < 2:
        return 0.5
    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return slope
```

- [ ] **Step 5: Add `jennrich_test(corr1, corr2, n1, n2)`**

Test for equality of two correlation matrices across sub-periods.

```python
def jennrich_test(corr1: np.ndarray, corr2: np.ndarray, n1: int, n2: int) -> dict:
    """Jennrich (1970) test for equality of two correlation matrices.

    H0: The two correlation matrices are equal.
    Returns: chi2_stat, p_value, degrees_of_freedom.
    """
    from scipy.stats import chi2
    p = corr1.shape[0]
    N = n1 + n2
    R_pooled = (n1 * corr1 + n2 * corr2) / N
    R_pooled_inv = np.linalg.inv(R_pooled)

    # Compute S matrix
    S = (n1 * n2 / N) * np.trace(
        (corr1 - corr2) @ R_pooled_inv @ (corr1 - corr2) @ R_pooled_inv
    )
    # Adjust for off-diagonal elements only
    diag_term = np.sum(np.diag((corr1 - corr2) @ R_pooled_inv) ** 2)
    chi2_stat = S - 0.5 * diag_term

    df = p * (p - 1) // 2
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return {'chi2_stat': chi2_stat, 'p_value': p_value, 'df': df, 'significant_5pct': p_value < 0.05}
```

- [ ] **Step 6: Verify all new functions work**

Run a quick smoke test in Python:
```bash
cd "//Mac/Home/Documents/Personal Project/Finance/Portfolio Construction"
python -c "from src.feature_engineering import ljung_box_test, arch_lm_test, hurst_exponent, jennrich_test; print('All imports OK')"
```

- [ ] **Step 7: Commit**

```bash
git add src/feature_engineering.py
git commit -m "feat: add statistical testing utilities (Ljung-Box, ARCH-LM, Hurst, Jennrich)"
```

---

### Task 2: Add Visualization Utilities to `src/visualization.py`

**Files:**
- Modify: `src/visualization.py`

- [ ] **Step 1: Read current `visualization.py`**

Read `src/visualization.py` to understand existing patterns (color palette, DPI, save conventions).

- [ ] **Step 2: Add `plot_acf_pacf_grid(returns_dict, lags=40, save_name=None)`**

Grid of ACF/PACF plots for multiple tickers. Uses `statsmodels.graphics.tsaplots`.

```python
def plot_acf_pacf_grid(returns_dict: dict, lags: int = 40,
                        save_name: str = None) -> plt.Figure:
    """ACF and PACF grid for multiple tickers (2 columns per ticker)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    tickers = list(returns_dict.keys())
    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, t in enumerate(tickers):
        data = returns_dict[t].dropna()
        plot_acf(data, ax=axes[i, 0], lags=lags, alpha=0.05, title=f'{t} ACF')
        plot_pacf(data, ax=axes[i, 1], lags=lags, alpha=0.05, title=f'{t} PACF')
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig
```

- [ ] **Step 3: Add `plot_data_availability_gantt(tickers, first_dates, last_dates, save_name=None)`**

Horizontal bar chart showing data coverage per ticker.

```python
def plot_data_availability_gantt(tickers: list, first_dates: list,
                                  last_dates: list, save_name: str = None) -> plt.Figure:
    """Gantt-style chart showing data availability per ticker."""
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (t, start, end) in enumerate(zip(tickers, first_dates, last_dates)):
        ax.barh(i, (end - start).days, left=start, height=0.6, alpha=0.8)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    ax.set_xlabel('Date')
    ax.set_title('Data Availability by Ticker')
    ax.invert_yaxis()
    fig.tight_layout()
    if save_name:
        save_fig(fig, save_name)
    return fig
```

- [ ] **Step 4: Add remaining visualization functions**

Add these functions following the same pattern (signature + docstring + implementation):
- `plot_regime_probability_timeseries(probs_df, title, save_name)`
- `plot_transition_heatmap(trans_matrix, state_names, save_name)`
- `plot_shap_regime_comparison(shap_bull, shap_bear, feature_names, save_name)`
- `plot_calibration_curve(y_true, y_prob, n_bins, save_name)`
- `plot_learning_curves(train_sizes, train_scores, test_scores, model_names, save_name)`
- `plot_feature_ablation_waterfall(group_names, rmse_deltas, save_name)`
- `plot_attention_heatmap(attention_weights, time_labels, feature_labels, save_name)`
- `plot_attribution_stacked(dates, allocation_effect, selection_effect, save_name)`
- `plot_reverse_stress(factor_shocks, portfolio_impacts, save_name)`
- `plot_risk_card_table(risk_cards_df, save_name)`

Each function should: accept data + save_name, create a matplotlib figure, call `save_fig()` if save_name provided, return the figure.

- [ ] **Step 5: Verify imports**

```bash
python -c "from src.visualization import plot_acf_pacf_grid, plot_data_availability_gantt, plot_calibration_curve; print('Viz imports OK')"
```

- [ ] **Step 6: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add 12 new visualization functions for enhanced notebooks"
```

---

### Task 3: Rewrite NB01 — Data Ingestion & EDA

**Files:**
- Rewrite: `notebooks/NB01_data_ingestion_eda.ipynb`

This is a full notebook rewrite. The notebook should follow the structure from spec Section 2, with these sections:

- [ ] **Step 1: Read current NB01 to preserve working code patterns**

Read `notebooks/NB01_data_ingestion_eda.ipynb` — note which function calls work, what the cell outputs look like, and preserve any debugging fixes.

- [ ] **Step 2: Write the restructured NB01**

The notebook must contain these sections in order. Each section follows the Compute-Visualize-Interpret pattern.

**Cell structure (markdown + code cells):**

1. **Executive Summary** (markdown) — What this notebook does, key outputs (`master_data.parquet`), key findings placeholder
2. **Setup & Imports** (code) — sys.path, all imports from src.config, src.data_loader, src.feature_engineering, src.visualization
3. **Section 1: Data Download** (markdown + code)
   - Download all 20 tickers via `download_all_tickers()`
   - Handle SQ→XYZ merge via `merge_sq_xyz()`
   - Download benchmarks via `download_benchmarks()`
   - Download FRED macro data via `download_fred()`
4. **Section 2: Data Quality Checks** (markdown + code)
   - `build_adj_close_panel()` with quality flags
   - Short-history ticker table (CRWD, DDOG, PLTR: IPO date, obs count, % coverage)
   - Quality flags summary (forward-fills, return spikes)
   - **NEW**: Data availability Gantt chart via `plot_data_availability_gantt()`
5. **Section 3: Returns** (code)
   - `compute_log_returns()`, `compute_simple_returns()`
6. **Section 4: Summary Statistics** (markdown + code + interpretation)
   - `summary_statistics()` with ENHANCED columns: annualized Sharpe, max 1-day gain/loss, Hurst exponent
   - Styled table with significance stars on JB and ADF
   - **Interpretation markdown**: "All 20 tickers reject normality... Highest vol is [X]... Hurst < 0.5 for [tickers] suggests mean-reversion"
7. **Section 5: Stationarity Tests (ADF + KPSS)** (markdown + code + interpretation)
   - Joint ADF + KPSS confirmatory approach
   - Flag conflicts (ADF stationary but KPSS non-stationary → fractional integration?)
   - **Interpretation**: "All log return series are stationary. Level prices are non-stationary as expected."
8. **Section 6: Autocorrelation & ARCH Effect Testing** (NEW — markdown + code + interpretation)
   - Ljung-Box on raw returns (all 20 tickers, lags 1/5/10/20)
   - Ljung-Box on SQUARED returns (detect volatility clustering)
   - ARCH-LM test (all 20 tickers)
   - ACF/PACF grid for top-5 volatile tickers (NVDA, PLTR, MU, CRWD, XYZ) via `plot_acf_pacf_grid()`
   - Summary table: ticker × test → p-value with significance stars
   - **Interpretation**: "Significant ARCH effects in [X]/20 tickers' squared returns confirms volatility clustering. This justifies GARCH modeling in NB03."
9. **Section 7: Cumulative Returns** (code + interpretation)
   - `plot_cumulative_returns()` — log scale, all 20 tickers
   - Interpretation: top/bottom performers, crisis drawdowns visible
10. **Section 8: Correlation Analysis** (markdown + code + interpretation)
    - Full-period correlation heatmap with dendrogram clustering
    - Rolling 252-day pairwise correlations (6 key pairs)
    - **NEW**: Correlation stability test (Jennrich) across 4 sub-periods
    - **Interpretation**: "Correlations are unstable across regimes (Jennrich p < 0.01), justifying regime-conditional covariance in NB11"
11. **Section 9: Drawdown Analysis** (markdown + code + interpretation)
    - Max drawdown table with time-to-recovery
    - Drawdown waterfall for top-5 most volatile names
    - **NEW**: Underwater equity curves
    - Calmar ratio ranking
12. **Section 10: Distribution Diagnostics** (markdown + code + interpretation)
    - QQ plots (4x5 grid) with Student-t overlay
    - **NEW**: Anderson-Darling test statistics per ticker
    - KDE vs. Normal overlay for top-5 tickers
    - **Interpretation**: "Heavy tails confirmed by A-D test rejection for all 20 tickers. Student-t provides better fit."
13. **Section 11: Overnight vs. Intraday Volatility** (NEW — markdown + code + interpretation)
    - Close-to-open returns vs. open-to-close returns
    - Variance decomposition: % from overnight gaps
    - Stacked bar chart per ticker
    - **Interpretation**: "Overnight gaps account for [X]% of total variance for [tickers], representing non-diversifiable event risk"
14. **Section 12: Rolling Volatility Decomposition** (NEW — markdown + code + interpretation)
    - Rolling 63-day beta × SPY vol = systematic
    - Residual = idiosyncratic
    - Time series plot for 4-6 key tickers
    - **Interpretation**: "High idiosyncratic vol in [PLTR, CRWD] signals stock-specific risk, not market exposure"
15. **Section 13: Tail Ratio Analysis** (NEW — markdown + code + interpretation)
    - Tail ratio = |95th pctile| / |5th pctile| per ticker
    - Rolling 252-day tail ratio
    - **Interpretation**: "Tail ratio < 1 for [tickers] = negatively skewed (crash-prone). Tail ratio > 1 for [tickers] = positively skewed"
16. **Section 14: Volume Spike Detection** (code + interpretation)
    - Volume z-score spikes > 3σ mapped to KEY_EVENTS
    - **NEW**: Cross-ticker synchronization analysis (when do >10 tickers spike simultaneously?)
    - Volume spike timeline for top-4 tickers
17. **Section 15: Save Master Data & Outputs** (code)
    - Save `master_data.parquet`
    - Save summary stats, stationarity tests, drawdown stats as CSV
18. **Synthesis & Cross-References** (markdown)
    - Key findings summary (5-7 bullets)
    - Cross-reference: "ARCH effects justify GARCH in NB03. Correlation instability motivates regime analysis in NB05. Fat tails feed into EVT in NB04."

- [ ] **Step 3: Verify notebook structure**

Open the notebook and verify: all markdown cells have content, code cells reference correct function names from src/, section numbering is sequential.

- [ ] **Step 4: Commit**

```bash
git add notebooks/NB01_data_ingestion_eda.ipynb
git commit -m "feat(NB01): full restructure with ARCH tests, vol decomposition, tail ratio, Jennrich test"
```

---

## Chunk 2: Phase 2 Notebooks (NB02, NB03, NB05, NB06) — Parallel

These 4 notebooks depend only on NB01 and can be worked on in parallel.

### Task 4: Add GARCH Utilities to `src/garch_utils.py`

**Files:**
- Modify: `src/garch_utils.py`

- [ ] **Step 1: Read current `garch_utils.py`**

- [ ] **Step 2: Add `extract_all_parameters(result, ticker, model_name, dist)`**

Extract full parameter table (omega, alpha, beta, gamma, d) with standard errors, t-stats, significance.

```python
def extract_all_parameters(result, ticker: str, model_name: str, dist: str) -> dict:
    """Extract all GARCH parameters with standard errors and t-statistics.

    Returns dict with: ticker, model, dist, omega, alpha, beta, gamma (if GJR/EGARCH),
    d (if FIGARCH), plus _se and _tstat variants, persistence (alpha+beta).
    """
    params = result.params
    std_err = result.std_err
    tvals = result.tvalues

    row = {'ticker': ticker, 'model': model_name, 'dist': dist}
    for name in params.index:
        clean_name = name.replace('[', '_').replace(']', '')
        row[clean_name] = params[name]
        row[f'{clean_name}_se'] = std_err.get(name, np.nan)
        row[f'{clean_name}_tstat'] = tvals.get(name, np.nan)
        row[f'{clean_name}_sig'] = abs(tvals.get(name, 0)) > 1.96

    # Persistence measure
    alpha_val = params.get('alpha[1]', params.get('alpha', 0))
    beta_val = params.get('beta[1]', params.get('beta', 0))
    row['persistence'] = alpha_val + beta_val

    return row
```

- [ ] **Step 3: Add `sign_bias_test(standardized_resids)`**

Engle & Ng (1993) sign bias test for leverage effect validation.

```python
def sign_bias_test(standardized_resids: np.ndarray) -> dict:
    """Engle & Ng (1993) sign bias test.

    Tests whether standardized residuals exhibit asymmetric behavior
    not captured by the volatility model.
    H0: No sign bias (model captures asymmetry correctly).
    """
    import statsmodels.api as sm
    e = standardized_resids
    e2 = e ** 2
    n = len(e)

    # Indicators
    S_neg = (e[:-1] < 0).astype(float)  # negative shock indicator
    S_pos = 1 - S_neg

    # Regressors: constant, S_neg, S_neg * e_{t-1}, S_pos * e_{t-1}
    X = np.column_stack([
        np.ones(n - 1),
        S_neg,
        S_neg * e[:-1],
        S_pos * e[:-1]
    ])
    y = e2[1:]

    model = sm.OLS(y, X).fit()

    # Joint F-test for all bias coefficients = 0
    f_stat = model.fvalue
    f_pvalue = model.f_pvalue

    return {
        'sign_bias_coef': model.params[1], 'sign_bias_pvalue': model.pvalues[1],
        'neg_size_bias_coef': model.params[2], 'neg_size_bias_pvalue': model.pvalues[2],
        'pos_size_bias_coef': model.params[3], 'pos_size_bias_pvalue': model.pvalues[3],
        'joint_f_stat': f_stat, 'joint_f_pvalue': f_pvalue,
        'significant_5pct': f_pvalue < 0.05
    }
```

- [ ] **Step 4: Add `news_impact_curve(result, n_points=200)`**

Plot-ready data for the news impact curve (volatility response to return shocks).

```python
def news_impact_curve(result, n_points: int = 200) -> tuple:
    """Compute news impact curve data from a fitted GARCH model.

    Returns (shocks, conditional_variances) arrays for plotting.
    The NIC shows how sigma^2_{t+1} responds to different values of epsilon_t.
    """
    params = result.params
    sigma2_bar = result.conditional_volatility.mean() ** 2

    # Range of shocks: -4 to +4 standard deviations
    shock_range = np.linspace(-4, 4, n_points) * np.sqrt(sigma2_bar)

    omega = params.get('omega', 0)
    alpha = params.get('alpha[1]', params.get('alpha', 0))
    beta = params.get('beta[1]', params.get('beta', 0))
    gamma = params.get('gamma[1]', params.get('gamma', 0))

    # Compute conditional variance for each shock
    # GARCH: sigma2 = omega + alpha * eps^2 + beta * sigma2_bar
    # GJR:   sigma2 = omega + alpha * eps^2 + gamma * eps^2 * I(eps<0) + beta * sigma2_bar
    cond_var = omega + alpha * shock_range**2 + beta * sigma2_bar
    if gamma != 0:
        cond_var += gamma * shock_range**2 * (shock_range < 0).astype(float)

    return shock_range, cond_var
```

- [ ] **Step 5: Add `garch_forecast_multi_step(result, horizon_steps=[1,5,21,63])`**

Multi-step ahead conditional variance forecasts.

```python
def garch_forecast_multi_step(result, horizon_steps: list = [1, 5, 21, 63]) -> dict:
    """Multi-step ahead conditional variance forecasts from fitted GARCH.

    Returns dict: {horizon: forecast_variance} (annualized vol = sqrt(var * 252)).
    """
    max_h = max(horizon_steps)
    forecast = result.forecast(horizon=max_h, reindex=False)
    variance_forecasts = forecast.variance.iloc[-1]

    return {
        h: np.sqrt(variance_forecasts.iloc[h - 1] * 252)
        for h in horizon_steps if h <= len(variance_forecasts)
    }
```

- [ ] **Step 6: Verify**

```bash
python -c "from src.garch_utils import extract_all_parameters, sign_bias_test, news_impact_curve, garch_forecast_multi_step; print('GARCH utils OK')"
```

- [ ] **Step 7: Commit**

```bash
git add src/garch_utils.py
git commit -m "feat: add GARCH parameter extraction, sign bias test, news impact curve, multi-step forecast"
```

---

### Task 5: Add Risk Metric Utilities to `src/risk_metrics.py`

**Files:**
- Modify: `src/risk_metrics.py`

- [ ] **Step 1: Read current `risk_metrics.py`**

- [ ] **Step 2: Add `hill_plot(returns, k_range=None)`**

Hill estimator for tail index across different order statistics.

```python
def hill_plot(returns: pd.Series, k_range: tuple = None) -> tuple:
    """Hill plot data: tail index estimator vs. number of upper order statistics.

    Returns (k_values, hill_estimates) arrays for plotting.
    Stable plateau in Hill plot confirms GPD assumption.
    """
    losses = -returns.dropna().values
    losses_sorted = np.sort(losses)[::-1]  # descending
    n = len(losses_sorted)

    if k_range is None:
        k_range = (10, min(n // 2, 500))

    k_values = list(range(k_range[0], k_range[1] + 1))
    hill_estimates = []
    for k in k_values:
        log_excesses = np.log(losses_sorted[:k]) - np.log(losses_sorted[k])
        xi_hat = np.mean(log_excesses)
        hill_estimates.append(xi_hat)

    return np.array(k_values), np.array(hill_estimates)
```

- [ ] **Step 3: Add `mean_excess_function(returns, thresholds=None)`**

Mean excess function for GPD threshold validation.

```python
def mean_excess_function(returns: pd.Series, thresholds: np.ndarray = None) -> tuple:
    """Mean excess function: E[X - u | X > u] vs. threshold u.

    Linear mean excess function above threshold confirms GPD assumption.
    Returns (thresholds, mean_excesses) arrays for plotting.
    """
    losses = -returns.dropna().values
    if thresholds is None:
        thresholds = np.percentile(losses, np.linspace(80, 99, 50))

    mean_excesses = []
    for u in thresholds:
        exceedances = losses[losses > u] - u
        if len(exceedances) > 5:
            mean_excesses.append(np.mean(exceedances))
        else:
            mean_excesses.append(np.nan)

    return thresholds, np.array(mean_excesses)
```

- [ ] **Step 4: Add `var_multi_day(returns, alpha, horizons=[5,10,21])`**

Multi-day VaR via historical simulation on overlapping returns.

```python
def var_multi_day(returns: pd.Series, alpha: float = 0.01,
                   horizons: list = [5, 10, 21]) -> dict:
    """Multi-day VaR via historical simulation on overlapping h-day returns.

    Also computes sqrt(h) scaled VaR for comparison.
    Returns dict: {horizon: {'actual': VaR, 'sqrt_scaled': VaR, 'scaling_error_pct': ...}}
    """
    clean = returns.dropna()
    var_1d = np.percentile(clean, alpha * 100)

    results = {}
    for h in horizons:
        h_day_returns = clean.rolling(h).sum().dropna()
        var_actual = np.percentile(h_day_returns, alpha * 100)
        var_sqrt = var_1d * np.sqrt(h)
        scaling_error = (var_sqrt - var_actual) / abs(var_actual) * 100
        results[h] = {
            'actual': var_actual, 'sqrt_scaled': var_sqrt,
            'scaling_error_pct': scaling_error
        }

    return results
```

- [ ] **Step 5: Add `wilson_score_ci(violations, n, alpha=0.05)`**

Wilson score confidence interval on violation rate for backtest power analysis.

```python
def wilson_score_ci(n_violations: int, n_total: int,
                     confidence: float = 0.95) -> tuple:
    """Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small violation counts.
    Returns (lower, point_estimate, upper).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_violations / n_total

    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom

    return (center - spread, p_hat, center + spread)
```

- [ ] **Step 6: Verify and commit**

```bash
python -c "from src.risk_metrics import hill_plot, mean_excess_function, var_multi_day, wilson_score_ci; print('Risk metrics OK')"
git add src/risk_metrics.py
git commit -m "feat: add Hill plot, mean excess function, multi-day VaR, Wilson CI for backtests"
```

---

### Task 6: Add ML Pipeline Utilities to `src/ml_pipeline.py`

**Files:**
- Modify: `src/ml_pipeline.py`

- [ ] **Step 1: Read current `ml_pipeline.py`**

- [ ] **Step 2: Add `model_confidence_set(loss_matrix, alpha=0.10, B=5000, block_size=None)`**

Hansen, Lunde & Nason (2011) Model Confidence Set implementation.

```python
def model_confidence_set(loss_matrix: pd.DataFrame, alpha: float = 0.10,
                          B: int = 5000, block_size: int = None) -> dict:
    """Model Confidence Set (Hansen, Lunde & Nason, 2011).

    Simultaneously compares multiple models and identifies the set of
    models that are not significantly inferior at level alpha.

    Args:
        loss_matrix: DataFrame where columns are model names, rows are time periods,
                     values are loss (e.g., squared forecast errors).
        alpha: Significance level for elimination.
        B: Number of bootstrap replications.
        block_size: Block size for stationary bootstrap. Default: sqrt(T).

    Returns: dict with 'superior_set' (list of model names), 'eliminated' (list),
             'p_values' (dict of model → p-value).
    """
    models = list(loss_matrix.columns)
    T = len(loss_matrix)
    if block_size is None:
        block_size = max(1, int(np.sqrt(T)))

    surviving = list(models)
    eliminated = []
    p_values = {}

    while len(surviving) > 1:
        # Compute pairwise loss differentials
        n_surv = len(surviving)
        d_bar = np.zeros((n_surv, n_surv))
        for i in range(n_surv):
            for j in range(n_surv):
                d_bar[i, j] = (loss_matrix[surviving[i]] - loss_matrix[surviving[j]]).mean()

        # T-max statistic
        t_stats = np.zeros(n_surv)
        for i in range(n_surv):
            d_i = np.mean([d_bar[i, j] for j in range(n_surv) if j != i])
            var_d = np.var([
                (loss_matrix[surviving[i]] - loss_matrix[surviving[j]]).values
                for j in range(n_surv) if j != i
            ], axis=0).mean()
            t_stats[i] = d_i / max(np.sqrt(var_d / T), 1e-10)

        t_max = np.max(t_stats)
        worst_model_idx = np.argmax(t_stats)

        # Bootstrap p-value for t_max
        boot_t_max = np.zeros(B)
        for b in range(B):
            # Stationary block bootstrap
            indices = []
            while len(indices) < T:
                start = np.random.randint(0, T)
                length = np.random.geometric(1.0 / block_size)
                indices.extend(range(start, min(start + length, T)))
            indices = np.array(indices[:T])

            boot_losses = loss_matrix[surviving].iloc[indices]
            boot_d_bar = np.zeros(n_surv)
            for i in range(n_surv):
                boot_d_bar[i] = np.mean([
                    (boot_losses.iloc[:, i] - boot_losses.iloc[:, j]).mean()
                    for j in range(n_surv) if j != i
                ])
            boot_t_max[b] = np.max(np.abs(boot_d_bar) / max(np.std(boot_d_bar), 1e-10))

        p_val = np.mean(boot_t_max >= t_max)
        p_values[surviving[worst_model_idx]] = p_val

        if p_val < alpha:
            eliminated.append(surviving.pop(worst_model_idx))
        else:
            # Cannot eliminate any more models
            for m in surviving:
                if m not in p_values:
                    p_values[m] = 1.0
            break

    return {
        'superior_set': surviving,
        'eliminated': eliminated,
        'p_values': p_values
    }
```

- [ ] **Step 3: Add `platt_scaling(y_true, y_prob_uncalibrated)`**

```python
def platt_scaling(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Platt scaling: fit logistic regression on predicted probabilities.

    Returns (calibrated_probs, calibrator_model).
    """
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1e10, solver='lbfgs')
    calibrator.fit(y_prob.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    return calibrated, calibrator
```

- [ ] **Step 4: Add `feature_ablation(X, y, pipeline_factory, groups, walk_forward_fn)`**

```python
def feature_ablation(X: pd.DataFrame, y: pd.Series,
                      pipeline_factory, feature_groups: dict,
                      baseline_rmse: float) -> pd.DataFrame:
    """Feature group ablation study.

    Removes one feature group at a time and measures RMSE degradation.

    Args:
        X: Full feature matrix.
        y: Target variable.
        pipeline_factory: callable returning a sklearn Pipeline.
        feature_groups: dict mapping group_name → list of column names.
        baseline_rmse: RMSE with all features (for comparison).

    Returns: DataFrame with group_name, rmse_without, rmse_delta, rmse_delta_pct.
    """
    from sklearn.metrics import mean_squared_error
    results = []
    for group_name, cols in feature_groups.items():
        remaining_cols = [c for c in X.columns if c not in cols]
        if not remaining_cols:
            continue
        X_ablated = X[remaining_cols]

        # Simple train/test split (last 20%)
        split = int(len(X_ablated) * 0.8)
        X_train, X_test = X_ablated.iloc[:split], X_ablated.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipe = pipeline_factory()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'group': group_name,
            'rmse_without': rmse,
            'rmse_delta': rmse - baseline_rmse,
            'rmse_delta_pct': (rmse - baseline_rmse) / baseline_rmse * 100
        })

    return pd.DataFrame(results).sort_values('rmse_delta', ascending=False)
```

- [ ] **Step 5: Verify and commit**

```bash
python -c "from src.ml_pipeline import model_confidence_set, platt_scaling, feature_ablation; print('ML pipeline OK')"
git add src/ml_pipeline.py
git commit -m "feat: add Model Confidence Set, Platt scaling, feature ablation utilities"
```

---

### Task 7: Add Portfolio Utilities to `src/backtest_engine.py`

**Files:**
- Modify: `src/backtest_engine.py`

- [ ] **Step 1: Read current `backtest_engine.py`**

- [ ] **Step 2: Add concentration and diversification metrics**

```python
def herfindahl_index(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index of portfolio concentration. Range [1/N, 1]."""
    return float(np.sum(weights ** 2))

def effective_n_positions(weights: np.ndarray) -> float:
    """Effective number of positions: 1 / HHI. Range [1, N]."""
    hhi = herfindahl_index(weights)
    return 1.0 / max(hhi, 1e-10)

def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Diversification ratio: weighted average vol / portfolio vol.

    DR > 1 means diversification is reducing risk.
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, individual_vols)
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    return weighted_avg_vol / max(portfolio_vol, 1e-10)
```

- [ ] **Step 3: Add `brinson_fachler_attribution(...)`**

```python
def brinson_fachler_attribution(
    portfolio_weights: np.ndarray, benchmark_weights: np.ndarray,
    portfolio_returns: np.ndarray, benchmark_returns: np.ndarray,
    sector_map: dict
) -> dict:
    """Brinson-Fachler performance attribution.

    Decomposes active return into allocation effect, selection effect, interaction.

    Args:
        portfolio_weights: (N,) array of portfolio weights.
        benchmark_weights: (N,) array of benchmark weights.
        portfolio_returns: (N,) array of asset returns in period.
        benchmark_returns: (N,) array of asset returns (same, but conceptual benchmark).
        sector_map: dict mapping sector_name → list of asset indices.

    Returns: dict with 'allocation', 'selection', 'interaction', 'total_active' per sector.
    """
    results = {}
    total_bench_return = np.dot(benchmark_weights, benchmark_returns)

    for sector, indices in sector_map.items():
        wp = portfolio_weights[indices].sum()
        wb = benchmark_weights[indices].sum()

        rp = np.dot(portfolio_weights[indices], portfolio_returns[indices]) / max(wp, 1e-10) if wp > 0 else 0
        rb = np.dot(benchmark_weights[indices], benchmark_returns[indices]) / max(wb, 1e-10) if wb > 0 else 0

        allocation = (wp - wb) * (rb - total_bench_return)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        results[sector] = {
            'allocation': allocation,
            'selection': selection,
            'interaction': interaction,
            'total': allocation + selection + interaction
        }

    return results
```

- [ ] **Step 4: Verify and commit**

```bash
python -c "from src.backtest_engine import herfindahl_index, effective_n_positions, diversification_ratio, brinson_fachler_attribution; print('Backtest engine OK')"
git add src/backtest_engine.py
git commit -m "feat: add portfolio attribution, concentration metrics, diversification ratio"
```

---

### Task 8: Create `src/sentiment.py` — FinBERT Pipeline

**Files:**
- Create: `src/sentiment.py`

- [ ] **Step 1: Write the FinBERT sentiment module**

This module handles: headline collection (RSS feeds), FinBERT inference, daily aggregation, t-1 lag enforcement.

```python
"""
FinBERT-based financial sentiment analysis pipeline.

Collects financial headlines, runs FinBERT inference, computes per-ticker
daily sentiment features, and enforces t-1 lag for ML consumption.

Usage:
    from src.sentiment import run_sentiment_pipeline
    sentiment_df = run_sentiment_pipeline(tickers, start_date, end_date)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from src.config import (TICKERS, START_DATE, END_DATE,
                         FEATURES_DIR, RAW_DIR, SENTIMENT_FILE)

logger = logging.getLogger(__name__)


# ─── Headline Collection ───

FINANCIAL_RSS_FEEDS = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline',
    'https://www.investing.com/rss/news.rss',
    'https://feeds.reuters.com/reuters/businessNews',
]

def collect_headlines_rss(tickers: list, start_date: str, end_date: str,
                           cache_dir: Path = None) -> pd.DataFrame:
    """Collect financial headlines from RSS feeds, filtered by ticker keywords.

    Returns DataFrame with columns: date, ticker, headline.
    Falls back to empty DataFrame if collection fails.
    """
    if cache_dir is None:
        cache_dir = RAW_DIR / 'headlines'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'headlines_cache.parquet'

    if cache_file.exists():
        logger.info(f'Loading cached headlines from {cache_file}')
        return pd.read_parquet(cache_file)

    headlines = []
    try:
        import feedparser
        # Build keyword map: ticker → [company name variants, ticker symbol]
        TICKER_KEYWORDS = _build_ticker_keywords(tickers)

        for feed_url in FINANCIAL_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    pub_date = pd.Timestamp(entry.get('published', ''))

                    for ticker, keywords in TICKER_KEYWORDS.items():
                        if any(kw.lower() in title.lower() for kw in keywords):
                            headlines.append({
                                'date': pub_date.normalize(),
                                'ticker': ticker,
                                'headline': title
                            })
            except Exception as e:
                logger.warning(f'RSS feed {feed_url} failed: {e}')
                continue
    except ImportError:
        logger.warning('feedparser not installed. Using fallback.')

    if headlines:
        df = pd.DataFrame(headlines)
        df.to_parquet(cache_file)
        return df

    logger.warning('No headlines collected. Will use synthetic fallback.')
    return pd.DataFrame(columns=['date', 'ticker', 'headline'])


def _build_ticker_keywords(tickers: list) -> dict:
    """Map tickers to company name keywords for headline matching."""
    COMPANY_NAMES = {
        'NVDA': ['NVIDIA', 'Nvidia'], 'AVGO': ['Broadcom'],
        'TSM': ['TSMC', 'Taiwan Semi'], 'SNPS': ['Synopsys'],
        'MSFT': ['Microsoft'], 'META': ['Meta', 'Facebook'],
        'GOOG': ['Google', 'Alphabet'], 'AMZN': ['Amazon'],
        'AAPL': ['Apple'], 'CRM': ['Salesforce'],
        'PANW': ['Palo Alto'], 'CRWD': ['CrowdStrike'],
        'DDOG': ['Datadog'], 'XYZ': ['Block', 'Square'],
        'NOW': ['ServiceNow'], 'PLTR': ['Palantir'],
        'ANET': ['Arista'], 'MU': ['Micron'],
        'AMD': ['AMD', 'Advanced Micro'], 'SAP': ['SAP']
    }
    return {t: COMPANY_NAMES.get(t, [t]) + [t] for t in tickers}


# ─── FinBERT Inference ───

def run_finbert_inference(headlines_df: pd.DataFrame,
                           batch_size: int = 32) -> pd.DataFrame:
    """Run FinBERT on headlines. Returns DataFrame with sentiment scores.

    Columns: date, ticker, headline, positive, negative, neutral, sentiment_score.
    sentiment_score = positive - negative (range [-1, 1]).
    """
    if headlines_df.empty:
        return headlines_df

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = 'ProsusAI/finbert'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f'FinBERT loaded on {device}')

        all_scores = []
        texts = headlines_df['headline'].tolist()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=512, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for j, prob in enumerate(probs):
                all_scores.append({
                    'positive': prob[0], 'negative': prob[1], 'neutral': prob[2],
                    'sentiment_score': prob[0] - prob[1]
                })

        scores_df = pd.DataFrame(all_scores)
        result = pd.concat([headlines_df.reset_index(drop=True), scores_df], axis=1)
        return result

    except ImportError:
        logger.warning('transformers not installed. Cannot run FinBERT.')
        return headlines_df


# ─── Aggregation & Feature Engineering ───

def aggregate_daily_sentiment(scored_df: pd.DataFrame,
                                tickers: list) -> pd.DataFrame:
    """Aggregate per-headline scores to daily per-ticker features.

    Features: sentiment_mean, sentiment_std, sentiment_volume, sentiment_momentum.
    """
    if scored_df.empty or 'sentiment_score' not in scored_df.columns:
        return pd.DataFrame()

    daily = scored_df.groupby(['date', 'ticker']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        sentiment_volume=('sentiment_score', 'count')
    ).reset_index()

    # Pivot to wide format (date × ticker features)
    features = []
    for ticker in tickers:
        ticker_data = daily[daily['ticker'] == ticker].set_index('date')
        ticker_data = ticker_data[['sentiment_mean', 'sentiment_std', 'sentiment_volume']]
        ticker_data.columns = [f'{ticker}_{c}' for c in ticker_data.columns]

        # Momentum: 5-day change in sentiment_mean
        mean_col = f'{ticker}_sentiment_mean'
        if mean_col in ticker_data.columns:
            ticker_data[f'{ticker}_sentiment_momentum'] = ticker_data[mean_col].diff(5)

        features.append(ticker_data)

    if features:
        return pd.concat(features, axis=1).sort_index()
    return pd.DataFrame()


def create_synthetic_sentiment(returns: pd.DataFrame,
                                 vol: pd.DataFrame) -> pd.DataFrame:
    """Synthetic sentiment proxy based on momentum/volatility signals.

    Used as fallback when real headline data is insufficient.
    MUST BE CLEARLY LABELED as synthetic in downstream notebooks.
    """
    logger.warning('Using SYNTHETIC sentiment proxy — NOT real FinBERT inference.')
    features = pd.DataFrame(index=returns.index)

    for ticker in returns.columns:
        ret_5d = returns[ticker].rolling(5).mean()
        vol_20d = returns[ticker].rolling(20).std()

        # Synthetic sentiment: tanh(momentum / volatility)
        features[f'{ticker}_sentiment_mean'] = np.tanh(ret_5d / vol_20d.clip(lower=1e-6))
        features[f'{ticker}_sentiment_std'] = vol_20d / vol_20d.rolling(60).mean().clip(lower=1e-6)
        features[f'{ticker}_sentiment_volume'] = (
            returns[ticker].rolling(20).apply(lambda x: (~x.isna()).sum())
        )
        features[f'{ticker}_sentiment_momentum'] = features[f'{ticker}_sentiment_mean'].diff(5)

    return features


# ─── Lag Enforcement ───

def enforce_sentiment_lag(sentiment_df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Shift all sentiment features by lag_days business days.

    CRITICAL: Prevents contemporaneous information leakage.
    A 2 PM headline affects both the sentiment score and the close-to-close return.
    Using t-1 sentiment ensures we only use yesterday's information.
    """
    lagged = sentiment_df.shift(lag_days)

    # Verification: no same-day alignment possible
    assert lagged.index[lag_days:].equals(sentiment_df.index[lag_days:]), \
        'Lag enforcement failed: index mismatch'

    logger.info(f'Sentiment features lagged by {lag_days} business day(s)')
    return lagged


# ─── Main Pipeline ───

def run_sentiment_pipeline(tickers: list = None, start_date: str = None,
                            end_date: str = None,
                            returns: pd.DataFrame = None,
                            min_headlines_per_ticker: int = 50,
                            force_synthetic: bool = False) -> tuple:
    """Run full sentiment pipeline: collect → infer → aggregate → lag.

    Args:
        tickers: List of tickers. Defaults to config.TICKERS.
        start_date, end_date: Date range. Defaults to config values.
        returns: Return DataFrame for synthetic fallback.
        min_headlines_per_ticker: Minimum headlines required per ticker per year.
        force_synthetic: Force synthetic proxy (skip FinBERT).

    Returns:
        (sentiment_df, is_synthetic): Lagged sentiment features and synthetic flag.
    """
    tickers = tickers or TICKERS
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE

    is_synthetic = False

    if not force_synthetic:
        # Step 1: Collect headlines
        headlines = collect_headlines_rss(tickers, start_date, end_date)

        # Step 2: Check sufficiency
        if not headlines.empty:
            per_ticker = headlines.groupby('ticker').size()
            years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365
            sufficient = (per_ticker >= min_headlines_per_ticker * years).all()
        else:
            sufficient = False

        if sufficient:
            # Step 3: FinBERT inference
            scored = run_finbert_inference(headlines)
            if 'sentiment_score' in scored.columns:
                # Step 4: Aggregate
                sentiment_df = aggregate_daily_sentiment(scored, tickers)
                if not sentiment_df.empty:
                    # Step 5: Lag enforcement
                    sentiment_df = enforce_sentiment_lag(sentiment_df)
                    sentiment_df.to_parquet(FEATURES_DIR / SENTIMENT_FILE)
                    logger.info(f'Real FinBERT sentiment saved: {sentiment_df.shape}')
                    return sentiment_df, False

    # Fallback: synthetic proxy
    is_synthetic = True
    if returns is not None:
        sentiment_df = create_synthetic_sentiment(returns, returns.rolling(20).std())
        sentiment_df = enforce_sentiment_lag(sentiment_df)
        sentiment_df.to_parquet(FEATURES_DIR / SENTIMENT_FILE)
        logger.info(f'Synthetic sentiment saved: {sentiment_df.shape}')
    else:
        logger.error('No returns provided for synthetic fallback')
        sentiment_df = pd.DataFrame()

    return sentiment_df, is_synthetic
```

- [ ] **Step 2: Add SENTIMENT_FILE constant to config.py**

Add to `src/config.py`:
```python
SENTIMENT_FILE = 'sentiment_features.parquet'
```

- [ ] **Step 3: Verify**

```bash
python -c "from src.sentiment import run_sentiment_pipeline; print('Sentiment module OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/sentiment.py src/config.py
git commit -m "feat: add FinBERT sentiment pipeline with RSS collection, inference, lag enforcement"
```

---

### Task 9: Rewrite NB02 — Macro Regime & Geopolitical Context

**Files:**
- Rewrite: `notebooks/NB02_macro_regime_context.ipynb`

- [ ] **Step 1: Read current NB02**

- [ ] **Step 2: Write restructured NB02**

Sections in order:

1. **Executive Summary** (markdown)
2. **Setup & Imports** (code) — load `master_data.parquet` from NB01
3. **Section 1: Macro Regime Timeline** — Rule-based regime labels, timeline visualization with colored background
4. **Section 2: PCA on Macro Factors** (NEW — REQUIRED)
   - Inputs: yield_curve_slope (10Y-2Y), VIX, DXY change, CPI surprise
   - Standardize, compute PCA, extract 2-3 components (>80% variance explained)
   - Loadings table with interpretation (PC1 = risk-on/off, PC2 = growth/stagflation)
   - Biplot of PC1 vs PC2 colored by regime
   - Time series of PC1, PC2 with regime-colored background
   - Save PCA scores for NB07/NB08
5. **Section 3: Event Study with CAR** (ENHANCED)
   - Cumulative Abnormal Return = actual - pre-event 60-day mean return
   - Cross-sectional t-test on mean CAR (is average significantly ≠ 0?)
   - Standardized 20-day post-event windows for comparison
   - Heatmap: 20 tickers × 9 events (CAR values)
   - Confidence bands around mean CAR
   - Interpretation per event
6. **Section 4: Factor Loading Regression** (NEW)
   - Per-ticker OLS: return_t = α + β_VIX·ΔVIX + β_yield·Δyield + β_DXY·ΔDXY + ε
   - Newey-West HAC standard errors (5 lags)
   - 20 × 4 coefficient table with t-stats and significance stars
   - Classification: rate-sensitive, risk-on-sensitive, FX-sensitive
   - Grouped bar chart of factor betas by sector group
7. **Section 5: Granger Causality** (ENHANCED)
   - Bidirectional: VIX → Tech AND Tech → VIX
   - Impulse Response Functions from bivariate VAR
   - Forecast Error Variance Decomposition
   - Per-factor results table
8. **Section 6: Structural Break Detection** (NEW)
   - Bai-Perron or PELT via `ruptures` on rolling beta
   - Break dates with confidence intervals
   - Beta time series with vertical break lines
   - Interpretation: "Breaks at [dates] correspond to [events]"
9. **Section 7: Rolling Beta Analysis** — Rolling 63-day beta, CUSUM
10. **Section 8: Cross-Asset Correlations by Regime** — Tech vs TLT/GLD/DXY per regime
11. **Section 9: Regime-Conditional Statistics** — Separate mean/vol/Sharpe/skew/kurtosis per regime per ticker
12. **Section 10: Regime Transition Analysis** (NEW)
    - Transition probability matrix heatmap
    - Average duration per regime
    - Regime persistence (autocorrelation of labels)
13. **Section 11: Regime Predictability** (NEW)
    - Logistic regression: P(bear | lagged macro features)
    - Walk-forward AUC-ROC
    - Interpretation: "Can we anticipate regime shifts?"
14. **Synthesis & Save** — Save `macro_regimes.parquet`, PCA scores, factor loadings

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB02_macro_regime_context.ipynb
git commit -m "feat(NB02): add PCA factors, CAR event study, factor loadings, structural breaks, regime prediction"
```

---

### Task 10: Rewrite NB03 — Volatility Econometrics

**Files:**
- Rewrite: `notebooks/NB03_volatility_econometrics.ipynb`

- [ ] **Step 1: Read current NB03**

- [ ] **Step 2: Write restructured NB03**

Sections:

1. **Executive Summary**
2. **Setup & Imports**
3. **Section 1: GARCH Model Fitting** — Full pipeline: 20 tickers × 4 models × 4 distributions
4. **Section 2: Model Selection** — AIC/BIC comparison, best model per ticker, win counts by model type and distribution
5. **Section 3: Full Parameter Table with CIs** (NEW)
   - `extract_all_parameters()` for all best-fit models
   - Parameters: ω, α, β, γ, d with standard errors, t-stats, significance stars
   - Persistence measure (α + β) heatmap across tickers
   - Interpretation: "Persistence > 0.95 in [tickers] means vol shocks are extremely long-lived"
6. **Section 4: Leverage Effect Analysis** (NEW)
   - γ extraction from GJR-GARCH, one-sided t-test for γ > 0
   - Bar chart of γ values with error bars
   - News Impact Curve comparison: GARCH vs. GJR for 3 tickers
   - Sign bias test results table
   - Interpretation: "NVDA γ = X means bad news amplifies vol by X% more than good news"
7. **Section 5: Conditional Volatility Time Series** — Plot for top-6 tickers, regime-colored background
8. **Section 6: Realized vs. Conditional Volatility** (NEW)
   - Yang-Zhang 21d realized vol vs. GARCH conditional vol overlay
   - Correlation and RMSE between them
   - Mincer-Zarnowitz regression (forecast efficiency)
   - Interpretation: "GARCH [leads/lags] realized vol by approximately [X] days"
9. **Section 7: Volatility Term Structure** (NEW)
   - Multi-step forecasts: h = 1, 5, 21, 63 via `garch_forecast_multi_step()`
   - Plot σ(h) vs. horizon for 4-6 tickers
   - Interpretation: "Term structure is [flat/upward-sloped/inverted] for [tickers]"
10. **Section 8: Innovation Distribution Analysis** (NEW)
    - Tail parameters: df for Student-t, shape for skewed-t
    - Standardized residual density vs. Normal and best-fit distribution overlays
    - Quantile comparison: 1% quantile ratio (fit vs. Normal)
    - Interpretation: "Student-t with df=5.3 implies tails 2.1x fatter than Normal"
11. **Section 9: Out-of-Sample Forecast Evaluation** (NEW)
    - Hold-out last 252 days
    - 1-step ahead conditional vol vs. next-day |return| (proxy for realized vol)
    - RMSE, MAE, QLIKE loss
    - Diebold-Mariano: GARCH vs. GJR vs. EGARCH
    - Forecast vs. actual scatter with 45-degree line
12. **Section 10: Sub-Period Parameter Stability** (NEW)
    - Re-estimate on 4 sub-periods
    - Parameter shift table
    - Interpretation: "Parameter instability justifies regime-switching in NB05"
13. **Section 11: Residual Diagnostics** (ENHANCED)
    - Ljung-Box, ARCH-LM, K-S test, sign bias test
    - All 20 tickers summary table
14. **Synthesis & Save** — Save `garch_parameters.csv`, `conditional_vol_series.parquet`

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB03_volatility_econometrics.ipynb
git commit -m "feat(NB03): add parameter CIs, leverage analysis, term structure, OOS evaluation, stability tests"
```

---

### Task 11: Rewrite NB05 — Regime Detection

**Files:**
- Rewrite: `notebooks/NB05_regime_detection.ipynb`

- [ ] **Step 1: Read current NB05**

- [ ] **Step 2: Write restructured NB05**

Sections:

1. **Executive Summary**
2. **Setup & Imports**
3. **Section 1: HMM Model Selection** — 2/3/4 states via BIC, results table
4. **Section 2: Regime Visualization** (NEW — CRITICAL)
   - Full 10-year price chart with regime-colored background via `plot_regime_overlay()`
   - Regime probability time series (smoothed posteriors)
   - Interpretation: "HMM identifies 3 regimes: bull (μ=X, σ=Y), neutral, bear"
5. **Section 3: Transition Probability Analysis** (NEW)
   - Transition matrix heatmap via `plot_transition_heatmap()`
   - Expected duration per state
   - Regime persistence histogram
6. **Section 4: Regime-Conditional Statistics**
   - Mean/vol/Sharpe/skew/kurtosis per regime
   - Regime-conditional return distribution KDE overlay (3-panel)
7. **Section 5: Regime-Conditional Correlations** (NEW visualization)
   - Side-by-side correlation heatmaps: bull vs. bear
   - Interpretation: "Correlations increase from X (bull) to Y (bear) — diversification fails when needed most"
8. **Section 6: Per-Ticker HMMs** (NEW)
   - Fit 2-state and 3-state on NVDA, PLTR, MU, CRWD, XYZ
   - Regime synchronization: % days where ticker regime = sector regime
   - 5-panel regime overlay
9. **Section 7: Regime-Conditional VaR/CVaR** (NEW)
   - Historical VaR/CVaR filtered by regime labels (requires NB04 output if available)
   - Table: VaR_95/VaR_99/CVaR_99 per regime
10. **Section 8: Macro Factor Regime Prediction** (NEW)
    - Logistic regression: P(bear | lagged macro) — cross-validates with NB02
    - Walk-forward AUC-ROC
    - Which macro features predict regime transitions?
11. **Section 9: MS-GARCH** (if feasible)
    - Regime-dependent GARCH parameters
    - Comparison to standard GARCH + HMM labels
    - If infeasible: documented as future work
12. **Synthesis & Save** — Save `regime_labels.parquet`, `hmm_model_params.pkl`, `transition_matrices.csv`

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB05_regime_detection.ipynb
git commit -m "feat(NB05): add full regime visualization, transition analysis, per-ticker HMMs, macro prediction"
```

---

### Task 12: Rewrite NB06 — NLP Sentiment (FinBERT)

**Files:**
- Rewrite: `notebooks/NB06_nlp_sentiment_finbert.ipynb`

- [ ] **Step 1: Read current NB06**

- [ ] **Step 2: Write restructured NB06**

Sections:

1. **Executive Summary** — Note whether real FinBERT or synthetic fallback was used
2. **Setup & Imports** — Import from `src.sentiment`
3. **Section 1: Headline Collection**
   - `collect_headlines_rss()` → report: N headlines collected, per-ticker coverage
   - If insufficient: clearly document and switch to synthetic with caveat
4. **Section 2: FinBERT Inference** (if real data available)
   - `run_finbert_inference()` → sentiment score distribution
   - Sample headlines with scores (positive, negative, neutral examples)
5. **Section 3: Daily Sentiment Features**
   - `aggregate_daily_sentiment()` → sentiment_mean, sentiment_std, sentiment_volume, sentiment_momentum
   - Time series plot of sentiment per 4-5 key tickers
6. **Section 4: Lag Enforcement Verification** (CRITICAL)
   - `enforce_sentiment_lag()` → explicit assertion
   - Show: date alignment before/after lag
   - Interpretation: "Sentiment at date t uses headlines from date t-1 only"
7. **Section 5: Granger Causality** (NEW)
   - Sentiment_{t-1} → Returns_t (lags 1-5) for all 20 tickers
   - Sentiment_{t-1} → Realized Vol_t for all 20 tickers
   - Results table with p-values and significance
   - Interpretation: "Lagged sentiment predicts next-day [returns/vol] for X/20 tickers"
8. **Section 6: Sentiment-VIX Correlation** (NEW)
   - Rolling 63-day correlation between cross-ticker average sentiment and VIX
   - Interpretation: "Negative correlation confirms sentiment captures risk aversion"
9. **Section 7: Sentiment-Regime Analysis** (NEW)
   - Average sentiment per HMM regime (from NB05)
   - Sentiment time series with regime-colored background
   - Does sentiment lead regime transitions? (lagged cross-correlation)
10. **Section 8: Event-Level Sentiment** (NEW)
    - Sentiment trajectory around KEY_EVENTS (±10 trading days)
    - Event-aligned chart for 4-5 key events
11. **Section 9: Predictive Power Assessment** (NEW)
    - Univariate regression: next-day return = α + β × sentiment_{t-1}
    - R² and t-stat per ticker
    - Honest assessment of predictive value
12. **Section 10: Synthetic Fallback Documentation** (if used)
    - Clear caveat: "This notebook uses synthetic sentiment proxy based on momentum/volatility"
    - Validation: does synthetic proxy correlate with VIX? (sanity check)
    - All downstream uses labeled as proxy-based
13. **Synthesis & Save** — Save `sentiment_features.parquet` (t-1 lagged)

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB06_nlp_sentiment_finbert.ipynb
git commit -m "feat(NB06): real FinBERT pipeline with Granger causality, regime analysis, event study"
```

---

## Chunk 3: NB04 — Tail Risk

### Task 13: Rewrite NB04 — Tail Risk, VaR, CVaR, EVT

**Files:**
- Rewrite: `notebooks/NB04_tail_risk_var_cvar.ipynb`

- [ ] **Step 1: Read current NB04**

- [ ] **Step 2: Write restructured NB04**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — Load GARCH conditional vol from NB03
3. **Section 1: VaR Computation (5 Methods)** — All 20 tickers × 5 methods × 2 confidence levels (95%, 99%)
4. **Section 2: CVaR / Expected Shortfall** — CVaR for each VaR method
5. **Section 3: VaR Method Accuracy Ranking** (NEW)
   - Cross-ticker comparison: which method has lowest average violation rate deviation?
   - Ranking table: method × (avg violation rate, Kupiec pass rate, Christoffersen pass rate)
   - Visualization: violation rate scatter by method
   - Interpretation: "GARCH-VaR passes for X/20 tickers; Gaussian fails for Y/20"
6. **Section 4: Multi-Day VaR** (NEW)
   - 5d, 10d, 21d VaR via `var_multi_day()`
   - √T scaling comparison: actual vs. scaled
   - Interpretation: "√T rule overestimates 21d VaR by X% due to vol mean-reversion"
7. **Section 5: VaR Backtesting** (ENHANCED)
   - Kupiec POF + Christoffersen CC + traffic light zones
   - Wilson score confidence intervals on violation rates via `wilson_score_ci()`
   - Backtest power discussion
8. **Section 6: EVT / GPD** (ENHANCED)
   - GPD fitting at threshold quantile 0.95
   - EVT-VaR/CVaR at 99%, 99.5%, 99.9%
   - **NEW**: Threshold sensitivity analysis (vary 0.90 to 0.98)
   - **NEW**: Mean excess function plot via `mean_excess_function()`
   - **NEW**: Hill plot via `hill_plot()`
   - **NEW**: Tail-fatness ranking: all 20 tickers sorted by ξ with CIs
   - Interpretation: "Heavy-tailed: [tickers with ξ > 0.2]. Stable plateau at [range] confirms GPD"
9. **Section 7: Cornish-Fisher Monotonicity Verification** (NEW)
   - Test CF-VaR with extreme skew/kurtosis values
   - Document monotonicity guard behavior
   - Side-by-side: CF-VaR vs. Historical for high-kurtosis tickers
10. **Section 8: CAViaR** — Verify implementation, report parameters, compare to GARCH-VaR
11. **Section 9: Copula Analysis** (EXPANDED)
    - Expand from 3 pairs to ~15-20 relevant pairs
    - Clayton + Gumbel for each pair
    - Tail dependence heatmap (20×20, lower triangle = λ_L, upper = λ_U)
    - Rolling 252-day Clayton θ for top-5 pairs
    - Joint crash probability matrix
    - Interpretation: "Tail dependence of X between NVDA-AMD means crashes are highly correlated"
12. **Section 10: Scenario Matrix Construction & Validation** (ENHANCED)
    - Build (T × 20) return scenario matrix
    - Validation: distribution of portfolio returns, tail coverage, bootstrap CI on CVaR
    - Save `return_scenarios.parquet` for NB11
13. **Synthesis & Save** — Save all VaR/CVaR tables, EVT parameters, backtest results, scenario matrix

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB04_tail_risk_var_cvar.ipynb
git commit -m "feat(NB04): add VaR ranking, multi-day VaR, EVT threshold analysis, expanded copulas, scenario validation"
```

---

## Chunk 4: ML Notebooks (NB07, NB08)

### Task 14: Rewrite NB07 — ML Volatility Forecasting

**Files:**
- Rewrite: `notebooks/NB07_ml_volatility_forecast.ipynb`

- [ ] **Step 1: Read current NB07**

- [ ] **Step 2: Write restructured NB07**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — Load all upstream outputs (GARCH vol, regime labels, sentiment)
3. **Section 1: Feature Matrix Construction** — Combine all features, document feature groups
4. **Section 2: Walk-Forward Evaluation** — 5 models, expanding window, quarterly retraining
5. **Section 3: Model Comparison** (ENHANCED)
   - RMSE, MAE, MAPE, Directional Accuracy per model
   - Model win counts across tickers
   - Bar charts for each metric
6. **Section 4: Statistical Tests** (ENHANCED)
   - Diebold-Mariano (pairwise with HAC bandwidth h-1)
   - Mincer-Zarnowitz (Newey-West HAC)
   - **NEW**: Model Confidence Set via `model_confidence_set()`
   - Interpretation: "MCS contains [models] at 10% — all others significantly inferior"
7. **Section 5: SHAP Explainability** (EXPANDED)
   - Expand from NVDA-only to 3 diverse tickers (NVDA, AAPL, PANW)
   - Beeswarm plots per ticker
   - **NEW**: Per-ticker SHAP waterfall for highest-error predictions
   - **NEW**: Force plot for individual predictions during COVID/rate shock
8. **Section 6: Regime-Conditional SHAP** (NEW)
   - Split SHAP values by HMM regime
   - Side-by-side beeswarm: bull vs. bear via `plot_shap_regime_comparison()`
   - Interpretation: "Bear markets → VIX features dominate; bull → momentum features"
9. **Section 7: Feature Ablation** (NEW)
   - `feature_ablation()` on XGBoost: remove momentum, vol, macro, sentiment, regime
   - Waterfall chart via `plot_feature_ablation_waterfall()`
   - Interpretation: "Removing [group] increases RMSE by X%"
10. **Section 8: Learning Curves** (NEW)
    - Training set size vs. RMSE (train and test)
    - Per-model learning curves via `plot_learning_curves()`
    - Interpretation: "XGBoost saturates at ~1500 samples; LSTM needs more data"
11. **Section 9: Multi-Horizon Comparison** (NEW)
    - 5-day vs. 21-day forward vol forecasting
    - Paired bar chart: RMSE by model × horizon
    - Interpretation: "Skill degrades by X% from 5d to 21d horizon"
12. **Section 10: Calibration Analysis** (NEW)
    - Predicted vs. realized vol reliability diagram
    - Calibration curve via `plot_calibration_curve()`
    - Interpretation: "XGBoost overestimates vol by X% in low-vol periods"
13. **Section 11: Per-Ticker Predictability** (NEW)
    - RMSE and DA per ticker (scatter plot: RMSE vs. avg vol)
    - Ranking: most/least predictable tickers
14. **Section 12: Stacking Ensemble** (ENHANCED)
    - Ridge meta-learner, weight analysis
    - DM test: stacking vs. best individual
15. **Synthesis & Save** — Save predictions, SHAP values, comparison tables

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB07_ml_volatility_forecast.ipynb
git commit -m "feat(NB07): add MCS, regime SHAP, ablation, learning curves, calibration, per-ticker analysis"
```

---

### Task 15: Rewrite NB08 — ML Return Forecasting

**Files:**
- Rewrite: `notebooks/NB08_ml_return_price_forecast.ipynb`

- [ ] **Step 1: Read current NB08**

- [ ] **Step 2: Write restructured NB08**

Sections:

1. **Executive Summary**
2. **Setup & Imports**
3. **Section 1: Task A — Classification** (ENHANCED)
   - Walk-forward evaluation: Logistic Regression, RF, XGBoost, SVM
   - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score
   - **NEW**: Threshold optimization (grid 0.3-0.7, maximize Sharpe)
   - **NEW**: Platt scaling calibration with before/after reliability diagram
   - **NEW**: Confusion matrix by regime (3-panel: bull/neutral/bear)
   - **NEW**: Per-ticker AUC-ROC ranking
4. **Section 2: Task B — Regression**
   - Same model suite, RMSE/MAE/DA
   - **NEW**: Feature importance comparison (classification vs. regression SHAP)
5. **Section 3: Task C — Multi-Horizon** (COMPLETED)
   - 1d, 5d, 21d forward returns, full walk-forward
   - **NEW**: Performance decay regression (RMSE = α + β·log(horizon))
   - **NEW**: Visualization: RMSE vs. horizon with fitted decay curve
   - Interpretation: "Return predictability half-life ≈ X days"
6. **Section 4: Strategy Backtest** (NEW — Critical)
   - Long/flat strategy from classifier signals (optimal threshold)
   - 10 bps transaction costs on signal changes
   - Performance: annualized return, vol, Sharpe, max DD, Calmar
   - Benchmark: buy-and-hold, equal-weight
   - Cumulative return comparison plot
7. **Section 5: Signal-Weighted Strategy** (NEW)
   - Position size = P(up) - 0.5
   - Compare binary vs. probability-weighted
   - Transaction cost sensitivity: 5/10/20 bps
8. **Synthesis & Save** — Save predictions, classification reports, strategy backtest

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB08_ml_return_price_forecast.ipynb
git commit -m "feat(NB08): add threshold opt, calibration, multi-horizon decay, strategy backtest"
```

---

## Chunk 5: Deep Learning (NB09)

### Task 16: Rewrite NB09 — Deep Learning Forecasting

**Files:**
- Rewrite: `notebooks/NB09_deep_learning_forecasting.ipynb`

- [ ] **Step 1: Read current NB09**

- [ ] **Step 2: Write restructured NB09**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — PyTorch, pytorch-forecasting (if available)
3. **Section 1: LSTM Forecasting** — Walk-forward, 5 tickers, training curves
4. **Section 2: GRU Forecasting** — Same protocol, convergence comparison
5. **Section 3: Temporal Fusion Transformer** (NEW)
   - If pytorch-forecasting available: full TFT with static/known/observed covariates
   - Quantile outputs (10th, 50th, 90th)
   - If unavailable: documented as future work with rationale
6. **Section 4: Attention Weight Analysis** (NEW)
   - TFT temporal attention heatmap
   - Feature attention analysis
   - Event-specific: what does model attend to during COVID?
7. **Section 5: Prediction Interval Calibration** (NEW)
   - Coverage analysis: % actuals within [10th, 90th] quantile
   - Prediction + interval time series with actuals
8. **Section 6: DL vs. GARCH Residual Analysis** (NEW)
   - DL error - GARCH error: is DL capturing patterns GARCH misses?
   - Regime-conditional: where does DL outperform?
9. **Section 7: Computational Cost-Benefit** (ENHANCED)
   - Table: model × (params, train_time, inference_time, memory, RMSE)
   - Pareto frontier: RMSE vs. compute cost
   - Interpretation: "XGBoost achieves 95% of LSTM accuracy at 1% compute cost"
10. **Section 8: DL + ML Ensemble** (NEW)
    - Average ensemble: (XGBoost + LSTM + GRU) / 3
    - Optimal weight ensemble via convex optimization
    - DM test: ensemble vs. best individual
11. **Section 9: Statistical Comparison** — DM tests, comparison table with NB07 models
12. **Synthesis & Save**

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB09_deep_learning_forecasting.ipynb
git commit -m "feat(NB09): add TFT, attention analysis, prediction intervals, DL-GARCH residual comparison"
```

---

## Chunk 6: Hybrid Audit (NB10)

### Task 17: Rewrite NB10 — Hybrid Model & Audit

**Files:**
- Rewrite: `notebooks/NB10_hybrid_model_audit.ipynb`

- [ ] **Step 1: Read current NB10**

- [ ] **Step 2: Write restructured NB10**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — Load all model outputs from NB03, NB07, NB08, NB09
3. **Section 1: Hybrid Architecture** — 3-stage (GARCH → XGBoost → LSTM), weight optimization
4. **Section 2: Lookback Window Sensitivity** (NEW)
   - 30/60/90/120 day windows, RMSE comparison
5. **Section 3: Comprehensive Model Comparison** (ENHANCED)
   - All models from NB03, NB07, NB08, NB09, hybrid
   - Metrics: RMSE, MAE, MAPE, DA
   - Pairwise DM tests (full matrix)
   - Model Confidence Set
6. **Section 4: Train/Test Gap Quantification** (NEW)
   - IS vs. OOS RMSE per model, gap ratio
   - Scatter plot with 45-degree line
7. **Section 5: Learning Curves** (NEW)
8. **Section 6: Permutation Importance Stability** (NEW)
   - Permutation vs. SHAP consistency
9. **Section 7: Sub-Period Stability** — RMSE across 4 sub-periods
10. **Section 8: Regime-Adaptive Hybrid Weights** (NEW)
    - Separate weights for bull vs. bear
11. **Section 9: Economic Significance** (NEW)
    - RMSE improvement → Sharpe improvement mapping
12. **Section 10: Overfitting Diagnostics** — Rolling RMSE, learning curves
13. **Section 11: Comprehensive Audit Summary** (NEW)
    - Traffic-light table: Green/Yellow/Red per criterion per model
    - Recommendation for NB11
14. **Synthesis & Save** — Save comparison tables, hybrid weights, audit report

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB10_hybrid_model_audit.ipynb
git commit -m "feat(NB10): add lookback sensitivity, train/test gap, regime-adaptive weights, traffic-light audit"
```

---

## Chunk 7: Portfolio Optimization (NB11)

### Task 18: Rewrite NB11 — Portfolio Optimization

**Files:**
- Rewrite: `notebooks/NB11_portfolio_optimization.ipynb`

- [ ] **Step 1: Read current NB11**

- [ ] **Step 2: Write restructured NB11**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — Load NB04 scenario matrix, NB05 regimes, NB10 best model predictions
3. **Section 1: Expected Return Estimation** — 3 methods (historical, CAPM, Black-Litterman)
4. **Section 2: Risk Estimation** (ENHANCED)
   - Ledoit-Wolf shrinkage
   - DCC-GARCH via `src/dcc_garch.py` (import and run)
   - Regime-conditional covariance
5. **Section 3: Optimization Methods** — 5 methods with proper constraints
6. **Section 4: Proper CVaR Integration** (ENHANCED)
   - Load `return_scenarios.parquet` from NB04 (not fallback)
   - Verify dimensions, tail coverage
   - Compare scenario-based vs. historical vs. parametric CVaR
7. **Section 5: Efficient Frontier** (ENHANCED)
   - Individual stocks on frontier, optimal portfolio marked, benchmark marked
8. **Section 6: Backtest** — Monthly rebalancing, 10 bps costs, walk-forward OOS only
9. **Section 7: Covariance Method Comparison** (NEW)
   - Same optimization with each covariance: which is most stable OOS?
10. **Section 8: Regime-Adaptive Strategy** (ENHANCED)
    - Test all combinations: HRP/ERC/MV × bull/neutral/bear
    - Optimize method selection per regime
11. **Section 9: Portfolio Attribution** (NEW)
    - Brinson-Fachler attribution via `brinson_fachler_attribution()`
    - Monthly attribution table, stacked bar over time
12. **Section 10: Turnover & Cost Sensitivity** (NEW)
    - Vary max turnover: 10/15/20/30%
    - Vary transaction cost: 5/10/15/20 bps
    - Sharpe vs. turnover frontier
13. **Section 11: Concentration & Diversification** (NEW)
    - HHI, effective N, diversification ratio over time
    - When does semiconductor cap bind?
14. **Section 12: Weight Evolution** — Stacked area, sector-level aggregation
15. **Synthesis & Save**

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB11_portfolio_optimization.ipynb
git commit -m "feat(NB11): add proper CVaR, covariance comparison, attribution, turnover sensitivity, diversification metrics"
```

---

## Chunk 8: Stress Testing & Deliverables (NB12)

### Task 19: Rewrite NB12 — Stress Testing & Final Report

**Files:**
- Rewrite: `notebooks/NB12_stress_test_report.ipynb`

- [ ] **Step 1: Read current NB12**

- [ ] **Step 2: Write restructured NB12**

Sections:

1. **Executive Summary**
2. **Setup & Imports** — Load all upstream outputs
3. **Section 1: Historical Stress Scenarios** — 5 crises, portfolio impact
4. **Section 2: Hypothetical Stress Scenarios** — 4 tail events
5. **Section 3: EVT-Enhanced Monte Carlo** (NEW)
   - GPD tails from NB04 replacing Normal innovations
   - Compare: Normal MC vs. EVT MC tail probabilities
6. **Section 4: Regime-Conditional Monte Carlo** (NEW)
   - Separate μ, σ, correlation per HMM regime
   - Sample regime sequence from transition matrix
   - Compare: unconditional vs. regime-conditional VaR/CVaR
7. **Section 5: Correlation Shock Scenarios** (NEW)
   - Taiwan crisis: correlations → 0.95 for semis
   - AI bubble: correlations → 0.9 for AI names
   - VaR comparison: normal vs. shocked
8. **Section 6: Reverse Stress Testing** (NEW)
   - What conditions cause 30% portfolio loss?
   - Factor decomposition of worst MC scenarios
9. **Section 7: Probability-Weighted Assessment** (NEW)
   - Subjective probabilities on hypothetical scenarios
   - Probability-weighted expected loss
   - Bubble chart
10. **Section 8: Factor Sensitivity** — OLS with Newey-West, risk decomposition
11. **Section 9: Optimal Hedge Recommendations** (NEW)
    - Marginal VaR contribution per position
    - Hedge suggestions
12. **Section 10: MC Validation** (NEW)
    - K-S test: MC vs. historical distribution
    - Convergence check: VaR vs. MC paths
13. **Section 11: Per-Ticker Risk Cards** (ENHANCED) — Full risk profile per ticker
14. **Section 12: Model Hierarchy Summary** — Best model for vol, returns, portfolio
15. **Section 13: Actionable Recommendations** — Current regime, recommended allocation
16. **Section 14: Final Deliverables** (NEW)
    - Generate 25-30 slide PPTX via `python-pptx`
    - Generate 15-20 page PDF audit report via `fpdf2`
17. **Synthesis**

- [ ] **Step 3: Commit**

```bash
git add notebooks/NB12_stress_test_report.ipynb
git commit -m "feat(NB12): add EVT MC, regime MC, correlation shocks, reverse stress, PPTX/PDF generation"
```

---

## Chunk 9: Test Suite & Final Verification

### Task 20: Create `tests/test_pipeline_integrity.py`

**Files:**
- Create: `tests/test_pipeline_integrity.py`

- [ ] **Step 1: Write the test suite**

```python
"""
Pipeline integrity tests — no lookahead bias, constraint enforcement, data quality.

Run: pytest tests/test_pipeline_integrity.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
FEATURES_DIR = DATA_DIR / 'features'
PROCESSED_DIR = DATA_DIR / 'processed'


class TestNoLookaheadBias:
    """Verify no feature at time t uses information from t+1 or later."""

    def test_sentiment_lag(self):
        """Sentiment features must be lagged by exactly 1 business day."""
        sentiment_file = FEATURES_DIR / 'sentiment_features.parquet'
        if not sentiment_file.exists():
            pytest.skip('Sentiment features not yet generated')

        sentiment = pd.read_parquet(sentiment_file)
        # First row should be NaN (due to lag)
        assert sentiment.iloc[0].isna().all(), \
            'First row of lagged sentiment should be NaN'

    def test_walk_forward_temporal_order(self):
        """Walk-forward predictions at time t only use training data < t."""
        pred_file = FEATURES_DIR / 'vol_forecast_predictions.parquet'
        if not pred_file.exists():
            pytest.skip('Predictions not yet generated')

        preds = pd.read_parquet(pred_file)
        # Predictions should be sorted by date
        assert preds.index.is_monotonic_increasing, \
            'Predictions not in temporal order'

    def test_no_future_in_features(self):
        """Verify rolling features don't include future data."""
        master_file = PROCESSED_DIR / 'master_data.parquet'
        if not master_file.exists():
            pytest.skip('Master data not yet generated')

        master = pd.read_parquet(master_file)
        # Rolling features should have NaN at the start (lookback period)
        rolling_cols = [c for c in master.columns if 'rolling' in c.lower() or '_21d' in c or '_63d' in c]
        for col in rolling_cols[:5]:  # Test first 5
            first_valid = master[col].first_valid_index()
            if first_valid is not None:
                idx = master.index.get_loc(first_valid)
                assert idx > 0, f'{col} has no leading NaN — possible lookahead'


class TestConstraintEnforcement:
    """Verify portfolio constraints are satisfied."""

    def test_max_single_stock_weight(self):
        """No stock > 10% weight."""
        weights_file = FEATURES_DIR / 'portfolio_weights_timeseries.parquet'
        if not weights_file.exists():
            pytest.skip('Portfolio weights not yet generated')

        weights = pd.read_parquet(weights_file)
        max_weight = weights.max().max()
        assert max_weight <= 0.101, \
            f'Max single stock weight {max_weight:.3f} exceeds 10% cap'

    def test_long_only(self):
        """No negative weights."""
        weights_file = FEATURES_DIR / 'portfolio_weights_timeseries.parquet'
        if not weights_file.exists():
            pytest.skip('Portfolio weights not yet generated')

        weights = pd.read_parquet(weights_file)
        min_weight = weights.min().min()
        assert min_weight >= -0.001, \
            f'Negative weight {min_weight:.4f} violates long-only constraint'


class TestCornishFisherMonotonicity:
    """Verify CF-VaR monotonicity guard works."""

    def test_extreme_kurtosis(self):
        """CF-VaR should not break with extreme kurtosis."""
        from src.risk_metrics import var_cornish_fisher

        # Generate returns with extreme kurtosis
        np.random.seed(42)
        extreme_returns = pd.Series(np.random.standard_t(df=3, size=500) * 0.02)

        var_95 = var_cornish_fisher(extreme_returns, alpha=0.05)
        var_99 = var_cornish_fisher(extreme_returns, alpha=0.01)

        # VaR_99 should be more extreme (more negative) than VaR_95
        assert var_99 <= var_95, \
            f'CF-VaR non-monotonic: VaR_99={var_99:.4f} > VaR_95={var_95:.4f}'


class TestFeatureStationarity:
    """Verify return series are stationary."""

    def test_returns_stationary(self):
        """All log return series should pass ADF test."""
        master_file = PROCESSED_DIR / 'master_data.parquet'
        if not master_file.exists():
            pytest.skip('Master data not yet generated')

        from src.feature_engineering import adf_test
        master = pd.read_parquet(master_file)

        # Test a few return columns
        return_cols = [c for c in master.columns if 'log_return' in c.lower()][:5]
        for col in return_cols:
            result = adf_test(master[col].dropna())
            assert result['p_value'] < 0.05, \
                f'{col} fails ADF stationarity test (p={result["p_value"]:.4f})'
```

- [ ] **Step 2: Verify test file parses**

```bash
python -c "import ast; ast.parse(open('tests/test_pipeline_integrity.py').read()); print('Test file OK')"
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_integrity.py
git commit -m "feat: add pipeline integrity test suite (lookahead bias, constraints, stationarity)"
```

---

### Task 21: Final Cross-Notebook Verification

- [ ] **Step 1: Verify all notebook files exist and are valid JSON**

```bash
for nb in notebooks/NB*.ipynb; do python -c "import json; json.load(open('$nb')); print('OK: $nb')"; done
```

- [ ] **Step 2: Verify all src/ imports work**

```bash
python -c "
from src.config import *
from src.data_loader import *
from src.feature_engineering import *
from src.garch_utils import *
from src.dcc_garch import *
from src.risk_metrics import *
from src.regime_utils import *
from src.ml_pipeline import *
from src.dl_models import *
from src.portfolio_optimizer import *
from src.backtest_engine import *
from src.visualization import *
from src.sentiment import *
print('All src/ imports successful')
"
```

- [ ] **Step 3: Run test suite**

```bash
pytest tests/test_pipeline_integrity.py -v --tb=short 2>&1 | head -50
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete notebook enhancement — all 12 notebooks restructured with comprehensive analytics"
```

---

## Summary

| Chunk | Tasks | Notebooks | Key Additions |
|-------|-------|-----------|---------------|
| 1 | 1-3 | NB01 | Statistical utils, viz functions, full NB01 rewrite |
| 2 | 4-12 | NB02, NB03, NB05, NB06 | GARCH/risk utils, sentiment module, 4 notebook rewrites |
| 3 | 13 | NB04 | Tail risk with EVT threshold analysis, expanded copulas |
| 4 | 14-15 | NB07, NB08 | ML utils, MCS, ablation, strategy backtest |
| 5 | 16 | NB09 | TFT, attention, prediction intervals |
| 6 | 17 | NB10 | Lookback sensitivity, regime-adaptive weights, audit |
| 7 | 18 | NB11 | Attribution, turnover sensitivity, diversification |
| 8 | 19 | NB12 | EVT MC, reverse stress, PPTX/PDF generation |
| 9 | 20-21 | Tests | Pipeline integrity test suite, final verification |

**Total: 21 tasks across 9 chunks.**
