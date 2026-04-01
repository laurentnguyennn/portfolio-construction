# CLAUDE.md Merge & Project-Wide Finance Depth Overhaul

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the 22-item design spec into CLAUDE.md AND simultaneously implement all new analytics across src/ modules, 12 notebooks, and tests — producing a postgraduate-ready quantitative finance pipeline.

**Architecture:** Two parallel tracks: (A) CLAUDE.md document merge — inline all spec formulas/sections into the existing 1,227-line file, creating a single authoritative blueprint; (B) Code implementation — add ~15 new functions across 4 new/existing src modules, update all 12 notebooks with new analytical cells, and add comprehensive tests.

**Tech Stack:** Python 3.10+, arch, statsmodels, hmmlearn, scipy, sklearn, xgboost, lightgbm, cvxpy, pypfopt, torch, shap, transformers

**Source spec:** `docs/superpowers/specs/2026-03-18-claude-md-finance-depth-overhaul-design.md`
**Target CLAUDE.md:** `CLAUDE.md` (project root, currently 1,227 lines)

---

## Chunk 1: CLAUDE.md Full Merge

This chunk merges all 22 spec items into the existing CLAUDE.md. No code changes — document only.

### Task 1.1: Add Section 2.4 — Statistical Inference Protocol

**Files:**
- Modify: `CLAUDE.md` (insert after Section 2.3, line ~195)

- [ ] **Step 1: Insert new Section 2.4 header and BH-FDR procedure (spec 1B)**

Insert after line 195 (end of Section 2.3) the complete Statistical Inference Protocol section containing:
- Benjamini-Hochberg FDR at q=0.05 procedure with formula
- Statement: apply wherever testing same hypothesis across 20 tickers
- Report both raw and BH-adjusted p-values

```markdown
### 2.4 Statistical Inference Protocol

#### 2.4.1 Multiple Testing Correction — Benjamini-Hochberg FDR

When testing the same hypothesis across m=20 tickers (ADF, Granger, Kupiec, DM tests), control the False Discovery Rate:
\```
BH procedure (Benjamini & Hochberg, 1995):
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)    (m = number of tests, typically 20)
2. Find largest k such that p_(k) ≤ (k/m) · q      (q = 0.05 target FDR)
3. Reject H₀ for all tests with p ≤ p_(k)
\```
Report both raw and BH-adjusted p-values in ALL multi-ticker test tables. Applies to: ADF (NB01), KPSS (NB01), Granger causality (NB02, NB06), ARCH-LM (NB03), Kupiec/Christoffersen (NB04), Diebold-Mariano (NB07, NB10).
```

- [ ] **Step 2: Insert formal hypothesis test reference table (spec 1D)**

Add the complete 12-row hypothesis test table with H₀, H₁, test statistic, distribution, and notebook location for: ADF, KPSS, Jarque-Bera (using excess kurtosis κ=K-3), Ljung-Box, ARCH-LM, Granger, Kupiec POF, Christoffersen CC, Diebold-Mariano, Bai-Perron, Engle-Granger cointegration, Johansen Trace.

Include convention: "Use α=0.05 throughout. ADF: constant+trend for levels, constant-only for returns. Always run KPSS alongside ADF (confirmatory approach)."

- [ ] **Step 3: Insert bootstrap inference specification (spec 4A)**

Add subsection 2.4.3 covering:
- Stationary Block Bootstrap (Politis & Romano, 1994) for return-based statistics
- Automatic block length selection (Politis & White, 2004)
- Residual Bootstrap for GARCH (Pascual, Romo & Ruiz, 2006)
- Circular Block Bootstrap for DCC parameters
- B=10,000 for 95% CIs, B=50,000 for 99% CIs
- Studentized bootstrap for Sharpe ratio comparison (Ledoit & Wolf, 2008)

---

### Task 1.2: Merge Critical Fixes into Existing Notebook Sections

**Files:**
- Modify: `CLAUDE.md` (multiple locations in Section 3)

- [ ] **Step 4: Add walk-forward embargo/purge to NB07 section (spec 1A)**

In the NB07 walk-forward validation description (around line ~590), insert the embargo/purge protocol:
```
Purge: Remove {t : t + h > T_train_end} from training set
Embargo: Additionally remove {t : T_train_end < t ≤ T_train_end + h}
For h=5: purge last 5 training obs + embargo 5 days; for h=21: purge 21 + embargo 21
```
Reference Lopez de Prado (2018) Ch.7. Add identical note to NB08 and NB09 sections.

- [ ] **Step 5: Add Hurst exponent prerequisite to NB03 section (spec 1C)**

Before the FIGARCH description in NB03 (around line ~490), insert:
```
Mandatory long-memory detection before FIGARCH:
1. R/S Analysis: H = lim ln(R_n/S_n)/ln(n)
2. GPH Log-Periodogram: ln(I(ω_j)) = c - d·ln(4sin²(ω_j/2)) + ε_j on |r_t| and r²_t
3. If H > 0.5 AND d > 0 (95% CI excludes 0) → proceed to FIGARCH
   Otherwise: skip FIGARCH, document reason
```

- [ ] **Step 6: Add realized volatility signature plot to NB03 section (spec 4E)**

Add to NB03 objectives:
```
Realized Volatility Signature Plot (for NVDA, AAPL, MSFT, META, AMZN):
  RV_Δ = Σ r²_{j,Δ}  for Δ ∈ {5min, 15min, 30min, 1hr, daily}
  Using 60-day intraday data at 5-min frequency (OpenBB provider limit).
  Validates daily OHLC-based estimators are appropriate for this universe.
```

- [ ] **Step 7: Add feature selection methodology to NB07/NB08 sections (spec 4C)**

Insert before model training in NB07 and NB08:
```
Feature Selection (pre-modeling):
1. VIF: Remove features with VIF > 10
2. Stability Selection: Lasso on 100 random 50% subsamples, retain features selected in >60%
3. Compare: full vs VIF-filtered vs stability-selected feature sets on OOS RMSE
```

---

### Task 1.3: Merge Systemic Risk & Advanced Analytics into NB Sections

**Files:**
- Modify: `CLAUDE.md` (NB04, NB02, NB08 sections)

- [ ] **Step 8: Add systemic risk measures to NB04 section (spec 2A)**

After copula-based joint tail risk in NB04, add:
- **CoVaR** with quantile regression formula and ΔCoVaR definition
- **MES** with conditional expectation formula and CVaR decomposition property
- **Absorption Ratio** with eigenvalue formula (k=N/5=4)
- Output: `systemic_risk_measures.csv`

- [ ] **Step 9: Add cross-sectional momentum & cointegration to NB08 (spec 2B)**

Add Task D to NB08:
- Cross-sectional momentum signal (12-1 month) with IC measurement
- Engle-Granger cointegration for 5 natural pairs with half-life formula
- Lead-lag cross-correlation ρ_{i,j}(k) for k ∈ {-5,...,+5}
- Output: `cointegration_pairs.csv`, `lead_lag_network.csv`

- [ ] **Step 10: Add options-implied analytics — promote from stretch to core (spec 2C)**

Modify NB03/NB04 sections to include:
- IV Term Structure (30d, 60d, 90d interpolation from OpenBB option chains)
- IV Skew (25Δ put - ATM)
- Variance Risk Premium per ticker (in annualized variance units)
- Implied Correlation with XLK
- Put-Call Ratio
- Data pipeline notes (interpolation method, weekly snapshots, storage)
- Update Section 2.1 data sources table to include options data as core (remove "stretch" label)

- [ ] **Step 11: Add earnings seasonality & PEAD to NB02 section (spec 2D)**

Add to NB02 event studies:
- Pre-earnings drift CAR[-5,-1]
- PEAD with SUE quintile sort
- Earnings vol crush (IV_{t-1} - IV_{t+1})
- Cross-ticker transmission (TSM earnings → NVDA AR)
- Data source note: OpenBB earnings data or seasonal random walk SUE proxy

---

### Task 1.4: Merge Portfolio Depth into NB11/NB12 Sections

**Files:**
- Modify: `CLAUDE.md` (NB11, NB12 sections)

- [ ] **Step 12: Add robust optimization methods 6-8 to NB11 (spec 3A)**

Add after existing method 5 (Risk Budgeting):
- **6. Worst-Case MV** with SDP reformulation: min w'Σ̂w + δ·||w||₂²
- **7. Maximum Diversification** with DR formula
- **8. Resampled EF** with correct Wishart draw: Σ̂_b ~ Wishart(scale=Σ̂/T, df=T)

- [ ] **Step 13: Add advanced performance metrics to NB11 (spec 3B)**

After existing Sharpe/Sortino/Calmar, add:
- Omega Ratio, Information Ratio, Tracking Error, Ulcer Index, Pain Index, CDaR formulas
- All with exact mathematical definitions

- [ ] **Step 14: Replace turnover constraint with formal L1 specification (spec 3C)**

Replace "max 20% monthly rebalance" with:
- L1 formulation: Σ_i |w_{i,new} - w_{i,old}| ≤ τ (τ=0.20)
- LP reformulation with auxiliary variables t_i
- Note: w_{i,old} must be drifted weight

- [ ] **Step 15: Replace flat 10bps with spread + market impact model (spec 3D)**

Replace transaction cost description with:
- Spread_cost_i = (spread_i/2) · |Δw_i| · AUM
- Impact_cost_i = η · σ_i · (|Δw_i|·AUM) · sqrt(|Δw_i|·AUM / ADV_i)
- Per-ticker spread calibration table (1-15 bps by liquidity tier)
- Parameterize AUM at $10M, $100M, $1B

- [ ] **Step 16: Add regime-conditional efficient frontiers to NB11 (spec 3E)**

Add: Overlay 3 frontiers (bull, bear, full-sample) on one plot. Document that bear frontier contracts inward, min-var composition shifts to defensive names.

- [ ] **Step 17: Add CDB, covariance comparison, tail risk budgeting to NB11 (spec 3F, 3G, 3H)**

- CDB formula: 1 - σ_p/(w'σ), computed per regime
- Expand covariance estimation from 3 to 5 methods (add PCA factor model, non-linear shrinkage)
- CVaR-ERC: equalize marginal CVaR contributions

- [ ] **Step 18: Add advanced stress test metrics to NB12 (spec 3B overlap)**

Add Ulcer Index, CDaR, and factor variance decomposition to NB12 final dashboard.

---

### Task 1.5: Merge ADR Premium, White's Reality Check, and Citations

**Files:**
- Modify: `CLAUDE.md` (NB01/NB02, NB10, Section 10)

- [ ] **Step 19: Add ADR premium analysis to NB01/NB02 (spec 4B — optional)**

Add as optional enrichment:
- TSM premium = (P_ADR/5)/(P_2330.TW · FX_TWD/USD) - 1
- SAP premium = P_ADR/(P_SAP.DE · FX_EUR/USD) - 1
- Download 2330.TW, SAP.DE, TWDUSD=X, EURUSD=X via OpenBB

- [ ] **Step 20: Add White's Reality Check / SPA test to NB10 (spec 4D)**

Add to model comparison section:
- H₀: best model = benchmark; T_SPA = max_m(d̄_m/σ̂_m); stationary block bootstrap
- Report SPA p-value for winning model

- [ ] **Step 21: Expand Section 10 citations with all new references**

Add ~20 new references organized into subsection 10.5:
Adrian & Brunnermeier (2016), Acharya et al. (2017), Kritzman et al. (2011), Goldfarb & Iyengar (2003), Choueifaty & Coignard (2008), Michaud (1998), Almgren & Chriss (2000), Lopez de Prado (2018), White (2000), Hansen (2005), Politis & Romano (1994), Politis & White (2004), Pascual et al. (2006), Meinshausen & Bühlmann (2010), Ledoit & Wolf (2008, 2020), Engle & Granger (1987), Geweke & Porter-Hudak (1983), Hurst (1951), Keating & Shadwick (2002), Chekhlov et al. (2005)

- [ ] **Step 22: Verify CLAUDE.md internal consistency and section numbering**

Read the entire merged file. Verify:
- All section numbers are sequential
- All cross-references (e.g., "see NB04") point to correct locations
- No duplicate content between existing formulas and newly merged ones
- Options data removed from "stretch" in Section 2.1

---

## Chunk 2: New src/ Modules & Function Additions

This chunk adds the new analytical functions required by the spec. Follow TDD: write failing test → implement → verify.

### Task 2.1: Create `src/statistical_tests.py` — Inference Utilities

**Files:**
- Create: `src/statistical_tests.py`
- Create: `tests/test_statistical_tests.py`

- [ ] **Step 1: Write failing tests for BH-FDR correction**

```python
# tests/test_statistical_tests.py
import numpy as np
from src.statistical_tests import benjamini_hochberg

def test_bh_fdr_all_significant():
    pvals = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
    assert all(rejected)

def test_bh_fdr_none_significant():
    pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
    assert not any(rejected)

def test_bh_fdr_partial():
    pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
    assert rejected[0] and rejected[1]  # first two should be rejected
    assert not rejected[4]  # last should not be rejected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_statistical_tests.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `src/statistical_tests.py`**

```python
"""Statistical inference utilities: multiple testing correction, bootstrap, hypothesis tests."""
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List

def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.
    Returns: (rejected: bool array, adjusted_pvalues: float array)"""
    m = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    thresholds = np.arange(1, m + 1) / m * q
    # Find largest k where p_(k) <= threshold
    below = sorted_pvals <= thresholds
    if not any(below):
        return np.zeros(m, dtype=bool), np.minimum(sorted_pvals * m / np.arange(1, m+1), 1.0)[np.argsort(sorted_idx)]
    k = np.max(np.where(below)[0])
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[:k+1]] = True
    # Adjusted p-values
    adjusted = np.minimum(sorted_pvals * m / np.arange(1, m+1), 1.0)
    # Enforce monotonicity
    for i in range(m-2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i+1]) if i < m-1 else adjusted[i]
    return rejected, adjusted[np.argsort(sorted_idx)]
```

- [ ] **Step 4: Add Hurst exponent estimation**

Write tests then implement:
```python
def hurst_rs(series: np.ndarray) -> float:
    """R/S analysis Hurst exponent. H>0.5 = long memory."""

def hurst_gph(series: np.ndarray, bandwidth: float = 0.5) -> Tuple[float, float, float]:
    """GPH log-periodogram estimator. Returns (d, se, p_value)."""
```

- [ ] **Step 5: Add White's Reality Check / SPA test**

```python
def spa_test(loss_matrix: np.ndarray, benchmark_col: int = 0,
             B: int = 10000, block_length: int = None) -> Dict:
    """Hansen (2005) Superior Predictive Ability test.
    Returns: {'statistic': float, 'p_value': float, 'best_model_idx': int}"""
```

- [ ] **Step 6: Add stationary block bootstrap**

```python
def stationary_block_bootstrap(data: np.ndarray, B: int = 10000,
                                avg_block_length: float = None) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap.
    Returns: (B, T) array of bootstrap samples."""

def bootstrap_sharpe_test(returns_a: np.ndarray, returns_b: np.ndarray,
                          B: int = 10000) -> Dict:
    """Ledoit & Wolf (2008) Sharpe ratio comparison test."""
```

- [ ] **Step 7: Run all tests, commit**

Run: `pytest tests/test_statistical_tests.py -v`
Expected: ALL PASS

```bash
git add src/statistical_tests.py tests/test_statistical_tests.py
git commit -m "feat: add statistical inference utilities (BH-FDR, Hurst, SPA, bootstrap)"
```

---

### Task 2.2: Create `src/systemic_risk.py` — CoVaR, MES, Absorption Ratio

**Files:**
- Create: `src/systemic_risk.py`
- Create: `tests/test_systemic_risk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_systemic_risk.py
import numpy as np
import pandas as pd
from src.systemic_risk import covar_quantile_regression, mes, absorption_ratio

def test_absorption_ratio_bounds():
    """AR must be between 0 and 1."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(500, 20))
    ar = absorption_ratio(returns, k=4, window=252)
    assert 0 < ar.iloc[-1] < 1

def test_mes_decomposition():
    """Portfolio CVaR = sum(w_i * MES_i)."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.randn(1000, 5))
    w = np.array([0.2] * 5)
    port_ret = returns @ w
    alpha = 0.05
    var_threshold = np.quantile(port_ret, alpha)
    tail_mask = port_ret <= var_threshold
    port_cvar = -port_ret[tail_mask].mean()
    mes_values = mes(returns, w, alpha=alpha)
    reconstructed = -np.sum(w * mes_values)
    assert abs(port_cvar - reconstructed) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `src/systemic_risk.py`**

```python
"""Systemic risk measures: CoVaR, MES, Absorption Ratio."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg

def covar_quantile_regression(portfolio_returns: pd.Series,
                               asset_returns: pd.Series,
                               conditioning_vars: pd.DataFrame = None,
                               alpha: float = 0.05) -> dict:
    """Adrian & Brunnermeier (2016) CoVaR via quantile regression.
    Returns dict with 'covar', 'delta_covar', 'covar_median'."""
    ...

def mes(returns: pd.DataFrame, weights: np.ndarray,
        alpha: float = 0.05) -> np.ndarray:
    """Marginal Expected Shortfall per asset.
    MES_i = E[r_i | r_portfolio <= VaR_alpha(portfolio)]"""
    port_ret = returns.values @ weights
    threshold = np.quantile(port_ret, alpha)
    tail_mask = port_ret <= threshold
    return returns[tail_mask].mean().values

def absorption_ratio(returns: pd.DataFrame, k: int = 4,
                     window: int = 252) -> pd.Series:
    """Kritzman et al. (2011) rolling absorption ratio.
    AR = sum(top k eigenvalues) / sum(all eigenvalues)."""
    ar_series = []
    for t in range(window, len(returns)):
        corr = returns.iloc[t-window:t].corr().values
        eigenvalues = np.linalg.eigvalsh(corr)[::-1]  # descending
        ar_series.append(eigenvalues[:k].sum() / eigenvalues.sum())
    return pd.Series(ar_series, index=returns.index[window:])
```

- [ ] **Step 4: Run tests, commit**

---

### Task 2.3: Add Advanced Portfolio Functions to `src/portfolio_optimizer.py`

**Files:**
- Modify: `src/portfolio_optimizer.py` (add 5 new functions)
- Modify: `tests/test_risk_metrics.py` (add tests)

- [ ] **Step 1: Write failing tests for new optimization methods**

```python
def test_worst_case_mv_returns_valid_weights():
    """Worst-case MV weights must sum to 1, be non-negative, <= 0.10."""
    ...

def test_max_diversification_dr_gt_one():
    """Diversification ratio must be > 1."""
    ...

def test_resampled_ef_weight_stability():
    """Resampled weights should have lower variance than single Markowitz."""
    ...
```

- [ ] **Step 2: Implement new functions**

Add to `src/portfolio_optimizer.py`:
```python
def worst_case_mv_optimize(mu, cov, delta, constraints, constraint_groups) -> Dict:
    """Goldfarb & Iyengar (2003). min w'Σ̂w + δ||w||² s.t. constraints."""

def max_diversification_optimize(cov, constraints) -> Dict:
    """Choueifaty & Coignard (2008). max w'σ / sqrt(w'Σw)."""

def resampled_ef_optimize(mu, cov, T, B=1000, constraints=None) -> Dict:
    """Michaud (1998). Average weights across B bootstrap Markowitz solutions."""

def cvar_risk_budgeting(return_scenarios, alpha=0.95, constraints=None) -> Dict:
    """CVaR-ERC: equalize marginal CVaR contributions."""

def conditional_diversification_benefit(weights, cov) -> float:
    """CDB = 1 - σ_portfolio / (w'σ)."""
```

- [ ] **Step 3: Run tests, commit**

---

### Task 2.4: Add Advanced Metrics to `src/backtest_engine.py`

**Files:**
- Modify: `src/backtest_engine.py` (add metric functions)

- [ ] **Step 1: Write failing tests**

```python
def test_omega_ratio_positive_for_good_portfolio():
    ...
def test_ulcer_index_zero_for_monotonic_increase():
    ...
def test_information_ratio_sign():
    ...
```

- [ ] **Step 2: Implement**

Add to `src/backtest_engine.py`:
```python
def omega_ratio(returns, threshold=0.0) -> float:
    """Keating & Shadwick (2002). Ω(θ) = gains_above_θ / losses_below_θ."""

def information_ratio(returns, benchmark_returns) -> float:
    """IR = (R̄_p - R̄_b) / tracking_error."""

def tracking_error(returns, benchmark_returns) -> float:
    """TE = std(R_p - R_b) * sqrt(252)."""

def ulcer_index(prices) -> float:
    """Martin (1987). UI = sqrt(mean(D²)) where D = drawdown %."""

def pain_index(prices) -> float:
    """Mean absolute drawdown percentage."""

def conditional_drawdown_at_risk(returns, alpha=0.95) -> float:
    """CDaR: E[DD | DD >= quantile(DD, alpha)]."""

def transaction_cost_impact_model(weights_old, weights_new, spreads, sigmas,
                                   adv, aum, eta=0.1) -> float:
    """Almgren-Chriss spread + market impact model."""
```

- [ ] **Step 3: Run tests, commit**

---

### Task 2.5: Add Feature Selection to `src/ml_pipeline.py`

**Files:**
- Modify: `src/ml_pipeline.py` (add 3 functions)

- [ ] **Step 1: Write failing tests**

```python
def test_vif_removes_collinear():
    ...
def test_stability_selection_subset():
    ...
```

- [ ] **Step 2: Implement**

Add to `src/ml_pipeline.py`:
```python
def vif_filter(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Remove features with VIF > threshold."""

def stability_selection(X, y, n_bootstrap=100, threshold=0.6) -> List[str]:
    """Meinshausen & Bühlmann (2010). Lasso on random subsamples."""

def feature_selection_comparison(X_train, y_train, X_test, y_test,
                                  model_builder) -> pd.DataFrame:
    """Compare full vs VIF-filtered vs stability-selected feature sets."""
```

- [ ] **Step 3: Run tests, commit**

---

### Task 2.6: Add Cointegration & Cross-Sectional Functions to `src/feature_engineering.py`

**Files:**
- Modify: `src/feature_engineering.py` (add functions)

- [ ] **Step 1: Write failing tests**

- [ ] **Step 2: Implement**

```python
def cross_sectional_momentum(prices: pd.DataFrame, lookback=252, skip=22) -> pd.DataFrame:
    """12-1 month momentum signal for all tickers."""

def engle_granger_cointegration(price_a, price_b) -> Dict:
    """Returns: beta, residuals, adf_stat, p_value, half_life."""

def lead_lag_crosscorr(returns_a, returns_b, max_lag=5) -> pd.Series:
    """Cross-correlation at lags -max_lag to +max_lag."""

def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """Spearman rank correlation between signal and realized return."""
```

- [ ] **Step 3: Run tests, commit**

---

## Chunk 3: Notebook Updates — NB01 through NB04

### Task 3.1: Update NB01 — Add ADR Premium & BH-FDR to Stationarity Tests

**Files:**
- Modify: `notebooks/NB01_data_ingestion_eda.ipynb`

- [ ] **Step 1: Add BH-FDR correction cell after stationarity tests**

Add new code cell after the ADF/KPSS test table:
```python
from src.statistical_tests import benjamini_hochberg

# Apply BH-FDR to ADF p-values across 20 tickers
adf_pvals = stationarity_df['adf_pvalue'].values
rejected, adjusted = benjamini_hochberg(adf_pvals, q=0.05)
stationarity_df['adf_pvalue_bh'] = adjusted
stationarity_df['adf_reject_bh'] = rejected

# Same for KPSS
kpss_pvals = stationarity_df['kpss_pvalue'].values
rejected_k, adjusted_k = benjamini_hochberg(kpss_pvals, q=0.05)
stationarity_df['kpss_pvalue_bh'] = adjusted_k
```

- [ ] **Step 2: Add ADR premium tracking cell (optional)**

```python
# ADR Premium Analysis for TSM and SAP
from src.data_loader import download_single_ticker
tsm_local = download_single_ticker('2330.TW', START_DATE, END_DATE)
sap_local = download_single_ticker('SAP.DE', START_DATE, END_DATE)
twdusd = download_single_ticker('TWDUSD=X', START_DATE, END_DATE)
eurusd = download_single_ticker('EURUSD=X', START_DATE, END_DATE)

tsm_premium = (adj_close['TSM'] / 5) / (tsm_local['Adj Close'] * twdusd['Adj Close']) - 1
sap_premium = adj_close['SAP'] / (sap_local['Adj Close'] * eurusd['Adj Close']) - 1
```

- [ ] **Step 3: Run NB01 end-to-end, verify outputs saved**

---

### Task 3.2: Update NB02 — Add Earnings Seasonality, Cointegration, Factor Model

**Files:**
- Modify: `notebooks/NB02_macro_regime_context.ipynb`

- [ ] **Step 1: Add Fama-French factor decomposition cell**

```python
import pandas_datareader.data as web
ff_factors = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', START_DATE, END_DATE)[0] / 100
# Merge with returns and run per-ticker regressions
# r^e_{i,t} = α_i + β_mkt·MKT + β_smb·SMB + β_hml·HML + ε
```

- [ ] **Step 2: Add earnings seasonality event study cell**

```python
# Download earnings dates from OpenBB
# Compute CAR[-5,-1] (pre-earnings drift) and CAR[+1,+20] (PEAD)
# Cross-ticker: TSM earnings → NVDA abnormal returns
```

- [ ] **Step 3: Add cointegration testing for 5 natural pairs**

```python
from src.feature_engineering import engle_granger_cointegration
pairs = [('NVDA','AMD'), ('TSM','AVGO'), ('META','GOOG'), ('PANW','CRWD'), ('CRM','NOW')]
coint_results = {}
for a, b in pairs:
    coint_results[f'{a}/{b}'] = engle_granger_cointegration(adj_close[a], adj_close[b])
```

- [ ] **Step 4: Apply BH-FDR to all Granger causality tests**

- [ ] **Step 5: Run NB02 end-to-end**

---

### Task 3.3: Update NB03 — Add Hurst Exponent, Signature Plot

**Files:**
- Modify: `notebooks/NB03_volatility_econometrics.ipynb`

- [ ] **Step 1: Add Hurst exponent cell before FIGARCH**

```python
from src.statistical_tests import hurst_rs, hurst_gph

hurst_results = {}
for ticker in TICKERS:
    ret = returns[ticker].dropna().values
    h_rs = hurst_rs(np.abs(ret))
    d_gph, se_gph, p_gph = hurst_gph(ret**2)
    hurst_results[ticker] = {'H_RS': h_rs, 'd_GPH': d_gph, 'se_GPH': se_gph, 'p_GPH': p_gph}

hurst_df = pd.DataFrame(hurst_results).T
# Only fit FIGARCH where H > 0.5 AND d > 0 AND p < 0.05
figarch_eligible = hurst_df[(hurst_df['H_RS'] > 0.5) & (hurst_df['p_GPH'] < 0.05)].index.tolist()
```

- [ ] **Step 2: Add realized vol signature plot cell**

```python
from openbb import obb
# Download 60 days of 5-min data for 5 liquid tickers
liquid_tickers = ['NVDA', 'AAPL', 'MSFT', 'META', 'AMZN']
for ticker in liquid_tickers:
    result = obb.equity.price.historical(symbol=ticker, start_date=start_60d, end_date=end, interval='5m')
    intraday = result.to_df()
    # Compute RV at 5min, 15min, 30min, 1hr, daily
    # Plot signature: RV vs ln(sampling_freq)
```

- [ ] **Step 3: Apply BH-FDR to ARCH-LM tests across 20 tickers**

- [ ] **Step 4: Run NB03 end-to-end**

---

### Task 3.4: Update NB04 — Add Systemic Risk Measures, Options-Implied

**Files:**
- Modify: `notebooks/NB04_tail_risk_var_cvar.ipynb`

- [ ] **Step 1: Add systemic risk measures cell**

```python
from src.systemic_risk import covar_quantile_regression, mes, absorption_ratio

# Absorption Ratio (rolling 252-day)
ar_ts = absorption_ratio(returns_df[TICKERS], k=4, window=252)

# MES for each ticker
weights = np.ones(20) / 20  # equal-weight for initial analysis
mes_values = mes(returns_df[TICKERS], weights, alpha=0.05)

# CoVaR for top-5 highest-beta names
for ticker in ['NVDA', 'PLTR', 'MU', 'CRWD', 'AMD']:
    covar_result = covar_quantile_regression(
        portfolio_returns=returns_df[TICKERS] @ weights,
        asset_returns=returns_df[ticker],
        alpha=0.05
    )
```

- [ ] **Step 2: Add options-implied analytics cell**

```python
# For 5 liquid tickers: extract IV term structure, skew, VRP
from openbb import obb
for ticker in ['NVDA', 'AAPL', 'MSFT', 'META', 'AMZN']:
    chains = obb.derivatives.options.chains(symbol=ticker).to_df()
    # Group by expiration, take nearest 3 monthly
    # Extract ATM IV, 25-delta put IV, put-call ratio
        # Interpolate to 30d/60d/90d fixed tenors
```

- [ ] **Step 3: Apply BH-FDR to Kupiec/Christoffersen backtests**

- [ ] **Step 4: Run NB04 end-to-end**

---

## Chunk 4: Notebook Updates — NB05 through NB08

### Task 4.1: Update NB05 — Add Absorption Ratio as Regime Indicator

**Files:**
- Modify: `notebooks/NB05_regime_detection.ipynb`

- [ ] **Step 1: Add cell comparing AR to HMM regimes**

```python
from src.systemic_risk import absorption_ratio
ar_ts = absorption_ratio(returns_df[TICKERS], k=4, window=252)
# Plot AR time series with HMM regime shading
# Compute correlation between AR and P(bear)
```

- [ ] **Step 2: Run NB05 end-to-end**

---

### Task 4.2: Update NB06 — Apply BH-FDR to Granger Tests

**Files:**
- Modify: `notebooks/NB06_nlp_sentiment_finbert.ipynb`

- [ ] **Step 1: Add BH-FDR correction to sentiment Granger causality tests**

- [ ] **Step 2: Run NB06 (if data available)**

---

### Task 4.3: Update NB07 — Add Embargo, Feature Selection, QLIKE

**Files:**
- Modify: `notebooks/NB07_ml_volatility_forecast.ipynb`

- [ ] **Step 1: Add embargo/purge to walk-forward protocol**

```python
# In walk_forward_predict call, add embargo parameter
# Purge: remove last h obs from training; Embargo: skip h days after train end
predictions = walk_forward_predict(
    X, y, model_builder,
    retrain_freq=63,
    embargo=5,  # for 5-day forecast horizon
    purge=5
)
```

- [ ] **Step 2: Add feature selection cell before model training**

```python
from src.ml_pipeline import vif_filter, stability_selection, feature_selection_comparison

X_vif = vif_filter(X_train, threshold=10.0)
selected_features = stability_selection(X_train, y_train, n_bootstrap=100, threshold=0.6)
comparison = feature_selection_comparison(X_train, y_train, X_test, y_test, make_xgboost)
```

- [ ] **Step 3: Add QLIKE loss function to evaluation**

```python
def qlike_loss(y_true, y_pred):
    """QLIKE = (1/T) * Σ[ln(ŷ²) + y²/ŷ²]"""
    return np.mean(np.log(y_pred**2) + y_true**2 / y_pred**2)
```

- [ ] **Step 4: Apply BH-FDR to Diebold-Mariano tests across models**

- [ ] **Step 5: Run NB07 end-to-end**

---

### Task 4.4: Update NB08 — Add Cross-Sectional Momentum, Embargo

**Files:**
- Modify: `notebooks/NB08_ml_return_price_forecast.ipynb`

- [ ] **Step 1: Add embargo/purge to walk-forward (same as NB07)**

- [ ] **Step 2: Add Task D: Cross-sectional momentum analysis**

```python
from src.feature_engineering import cross_sectional_momentum, information_coefficient

mom_signals = cross_sectional_momentum(adj_close[TICKERS], lookback=252, skip=22)
# Compute IC per rebalance date
ic_series = []
for t in rebalance_dates:
    signal = mom_signals.loc[t]
    fwd_ret = forward_returns_5d.loc[t]
    ic_series.append(information_coefficient(signal, fwd_ret))
```

- [ ] **Step 3: Add lead-lag cross-correlation analysis**

```python
from src.feature_engineering import lead_lag_crosscorr
# Compute for all pairs, visualize as network
```

- [ ] **Step 4: Run NB08 end-to-end**

---

## Chunk 5: Notebook Updates — NB09 through NB12

### Task 5.1: Update NB09 — Add Embargo to DL Walk-Forward

**Files:**
- Modify: `notebooks/NB09_deep_learning_forecasting.ipynb`

- [ ] **Step 1: Add embargo/purge to DL walk-forward evaluation**

Same protocol as NB07/NB08.

- [ ] **Step 2: Run NB09 end-to-end**

---

### Task 5.2: Update NB10 — Add White's Reality Check / SPA Test

**Files:**
- Modify: `notebooks/NB10_hybrid_model_audit.ipynb`

- [ ] **Step 1: Add SPA test cell to model comparison**

```python
from src.statistical_tests import spa_test

# Build loss matrix: rows = time steps, columns = models
loss_matrix = np.column_stack([
    losses_garch, losses_xgb, losses_lstm, losses_hybrid, losses_benchmark
])
result = spa_test(loss_matrix, benchmark_col=-1, B=10000)
print(f"SPA p-value: {result['p_value']:.4f}")
print(f"Best model index: {result['best_model_idx']}")
```

- [ ] **Step 2: Add BH-FDR to pairwise DM tests**

- [ ] **Step 3: Run NB10 end-to-end**

---

### Task 5.3: Update NB11 — Add Robust Optimization, Advanced Metrics, Regime Frontiers

**Files:**
- Modify: `notebooks/NB11_portfolio_optimization.ipynb`

- [ ] **Step 1: Add robust optimization methods cell**

```python
from src.portfolio_optimizer import (worst_case_mv_optimize,
    max_diversification_optimize, resampled_ef_optimize, cvar_risk_budgeting)

# Method 6: Worst-Case MV
wc_result = worst_case_mv_optimize(mu, cov, delta=0.1, constraints=CONSTRAINTS)

# Method 7: Maximum Diversification
md_result = max_diversification_optimize(cov, constraints=CONSTRAINTS)

# Method 8: Resampled EF
ref_result = resampled_ef_optimize(mu, cov, T=len(returns), B=1000)

# Method 9: CVaR-ERC (tail risk budgeting)
cvar_erc_result = cvar_risk_budgeting(return_scenarios, alpha=0.95)
```

- [ ] **Step 2: Add regime-conditional efficient frontiers cell**

```python
# Compute and overlay 3 frontiers
for regime, label in [(0, 'Bear'), (1, 'Neutral'), (2, 'Bull')]:
    mask = regime_labels == regime
    mu_r = returns[mask].mean() * 252
    cov_r = returns[mask].cov() * 252
    # Trace 50 points along frontier
    frontier_points = []
    for target in np.linspace(mu_r.min(), mu_r.max(), 50):
        w = mean_variance_optimize(mu_r, cov_r, target_return=target)
        frontier_points.append((np.sqrt(w @ cov_r @ w), target))
```

- [ ] **Step 3: Add CDB computation per regime**

```python
from src.portfolio_optimizer import conditional_diversification_benefit
cdb_bull = conditional_diversification_benefit(optimal_weights, cov_bull)
cdb_bear = conditional_diversification_benefit(optimal_weights, cov_bear)
```

- [ ] **Step 4: Add advanced performance metrics to backtest**

```python
from src.backtest_engine import (omega_ratio, information_ratio, tracking_error,
    ulcer_index, pain_index, conditional_drawdown_at_risk, transaction_cost_impact_model)

metrics['omega'] = omega_ratio(portfolio_returns, threshold=rf_daily)
metrics['IR'] = information_ratio(portfolio_returns, xlk_returns)
metrics['TE'] = tracking_error(portfolio_returns, xlk_returns)
metrics['ulcer'] = ulcer_index(portfolio_prices)
metrics['CDaR_95'] = conditional_drawdown_at_risk(portfolio_returns, alpha=0.95)
```

- [ ] **Step 5: Add covariance estimation comparison cell**

```python
# Compare 5 methods: Ledoit-Wolf, DCC, regime-conditional, PCA factor, NL shrinkage
from sklearn.covariance import LedoitWolf
# For each: compute optimal weights, measure OOS Sharpe, weight stability
```

- [ ] **Step 6: Replace flat 10bps with Almgren-Chriss model**

```python
# Calibrated per-ticker spreads
spreads = {'AAPL': 0.0001, 'MSFT': 0.0001, ..., 'SNPS': 0.0012}
for aum in [1e7, 1e8, 1e9]:
    cost = transaction_cost_impact_model(w_old, w_new, spreads, sigmas, adv, aum)
```

- [ ] **Step 7: Run NB11 end-to-end**

---

### Task 5.4: Update NB12 — Add CDaR, Factor Decomposition, Ulcer Index

**Files:**
- Modify: `notebooks/NB12_stress_test_report.ipynb`

- [ ] **Step 1: Add advanced risk metrics to stress test dashboard**

```python
# Per stress scenario: compute Ulcer Index, CDaR, CDB
# Factor variance decomposition: market vs sector vs idiosyncratic
```

- [ ] **Step 2: Run NB12 end-to-end**

---

## Chunk 6: Tests, Validation & Final Verification

### Task 6.1: Update `tests/test_pipeline_integrity.py`

**Files:**
- Modify: `tests/test_pipeline_integrity.py`

- [ ] **Step 1: Add embargo/purge verification test**

```python
class TestWalkForwardEmbargo:
    def test_no_label_leakage_5d(self):
        """Verify no training observation has a 5d label overlapping test period."""
        ...
    def test_embargo_gap_exists(self):
        """Verify embargo of h days between train end and test start."""
        ...
```

- [ ] **Step 2: Add BH-FDR test**

```python
class TestMultipleTesting:
    def test_bh_fdr_applied_to_stationarity(self):
        """Verify BH-adjusted p-values exist in stationarity output."""
        ...
```

- [ ] **Step 3: Add Hurst prerequisite test**

```python
class TestHurstBeforeFIGARCH:
    def test_figarch_only_for_long_memory(self):
        """Verify FIGARCH not fitted for tickers with H <= 0.5."""
        ...
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit all test updates**

---

### Task 6.2: Final CLAUDE.md Verification

**Files:**
- Read: `CLAUDE.md`

- [ ] **Step 1: Read entire merged CLAUDE.md**

Verify:
- Section numbering is sequential (0 through 10)
- All 22 spec items are present
- No duplicate formulas
- All cross-references valid
- Options data listed as core (not stretch) in Section 2.1
- New citations in Section 10

- [ ] **Step 2: Verify line count is ~1,900-2,100 lines (original 1,227 + ~700-800 additions)**

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: merge finance depth overhaul — 22 analytical dimensions across CLAUDE.md, 4 new src modules, 12 notebooks, and tests"
```

---

## File Map Summary

| Action | File | Changes |
|--------|------|---------|
| **Modify** | `CLAUDE.md` | Merge all 22 spec items inline (new Section 2.4, expand NB03-NB12 sections, expand Section 10) |
| **Create** | `src/statistical_tests.py` | BH-FDR, Hurst, SPA test, stationary bootstrap |
| **Create** | `src/systemic_risk.py` | CoVaR, MES, Absorption Ratio |
| **Create** | `tests/test_statistical_tests.py` | Tests for all statistical_tests functions |
| **Create** | `tests/test_systemic_risk.py` | Tests for systemic risk measures |
| **Modify** | `src/portfolio_optimizer.py` | +5 functions: worst-case MV, max diversification, resampled EF, CVaR-ERC, CDB |
| **Modify** | `src/backtest_engine.py` | +7 functions: Omega, IR, TE, Ulcer, Pain, CDaR, Almgren-Chriss TC |
| **Modify** | `src/ml_pipeline.py` | +3 functions: VIF filter, stability selection, feature selection comparison |
| **Modify** | `src/feature_engineering.py` | +4 functions: cross-sectional momentum, cointegration, lead-lag, IC |
| **Modify** | `notebooks/NB01_data_ingestion_eda.ipynb` | BH-FDR on stationarity, ADR premium (optional) |
| **Modify** | `notebooks/NB02_macro_regime_context.ipynb` | Fama-French factors, earnings seasonality, cointegration, BH-FDR |
| **Modify** | `notebooks/NB03_volatility_econometrics.ipynb` | Hurst exponent, signature plot, BH-FDR |
| **Modify** | `notebooks/NB04_tail_risk_var_cvar.ipynb` | Systemic risk (CoVaR, MES, AR), options-implied, BH-FDR |
| **Modify** | `notebooks/NB05_regime_detection.ipynb` | AR as regime indicator |
| **Modify** | `notebooks/NB06_nlp_sentiment_finbert.ipynb` | BH-FDR on Granger tests |
| **Modify** | `notebooks/NB07_ml_volatility_forecast.ipynb` | Embargo, feature selection, QLIKE, BH-FDR |
| **Modify** | `notebooks/NB08_ml_return_price_forecast.ipynb` | Embargo, cross-sectional momentum, lead-lag |
| **Modify** | `notebooks/NB09_deep_learning_forecasting.ipynb` | Embargo |
| **Modify** | `notebooks/NB10_hybrid_model_audit.ipynb` | SPA test, BH-FDR |
| **Modify** | `notebooks/NB11_portfolio_optimization.ipynb` | Robust opt (3 methods), CVaR-ERC, regime frontiers, CDB, advanced metrics, Almgren-Chriss TC, covariance comparison |
| **Modify** | `notebooks/NB12_stress_test_report.ipynb` | CDaR, factor decomposition, Ulcer Index |
| **Modify** | `tests/test_pipeline_integrity.py` | Embargo verification, BH-FDR verification, Hurst prerequisite |
