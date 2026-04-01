# CLAUDE.md Finance Depth Overhaul — Design Specification

**Date**: 2026-03-18
**Author**: Laurent (with Claude)
**Status**: Approved
**Scope**: Full overhaul — 22 items across 4 sections, ~600-800 lines added to CLAUDE.md
**Approach**: Hybrid (inline enrichment + new cross-cutting sections)

---

## Executive Summary

This spec defines a comprehensive upgrade to the CLAUDE.md project blueprint, adding postgraduate-level mathematical rigor, advanced financial analytics, and methodological safeguards. The project is fully implemented (12 notebooks, 13 src modules, 5,730 LOC) and the CLAUDE.md already has basic mathematical formulas. This overhaul adds 22 missing analytical dimensions that elevate the document from "strong" to "exceptional" for MSc admissions at SKEMA, emlyon, and ESCP.

---

## Section 1: Critical Methodological Fixes

### 1A. Walk-Forward Embargo/Purge Period

**Location**: Inline additions to NB07, NB08, NB09 walk-forward descriptions + Section 7 quality standards

**What to add**: When predicting h-day forward targets, purge the last h observations from training (their labels overlap test data) and embargo h additional days.

**Formula**:
```
Purge: Remove {t : t + h > T_train_end} from training set
Embargo: Additionally remove {t : T_train_end < t ≤ T_train_end + h}

For h=5 (5-day forecast): purge last 5 training obs + embargo 5 days after train end
For h=21 (21-day forecast): purge last 21 training obs + embargo 21 days
```

**Reference**: Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapter 7.

**Rationale**: Without purge/embargo, observations near the train/test boundary have labels that include test-period returns, creating subtle data leakage that inflates model performance.

---

### 1B. Multiple Testing Correction

**Location**: New Section 2.4 — Statistical Inference Protocol

**What to add**: Benjamini-Hochberg FDR control at q=0.05 wherever testing the same hypothesis across 20 tickers.

**Formula**:
```
BH procedure:
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)    (m = number of tests, typically 20)
2. Find largest k such that p_(k) ≤ (k/m) · q      (q = 0.05 target FDR)
3. Reject H₀ for all tests with p ≤ p_(k)

Report both raw and BH-adjusted p-values in all tables.
```

**Applies to**: ADF tests (NB01), KPSS tests (NB01), Granger causality (NB02, NB06), ARCH-LM (NB03), Kupiec/Christoffersen (NB04), Diebold-Mariano (NB07, NB10).

**For model comparison across M models**: White's Reality Check or SPA test (see 4D).

---

### 1C. Hurst Exponent Before FIGARCH

**Location**: Inline addition to NB03 before FIGARCH fitting

**What to add**: Mandatory long-memory detection step before committing to FIGARCH.

**Procedure**:
```
1. R/S Analysis:
   H = lim_{n→∞} ln(R_n/S_n) / ln(n)
   where R_n = range of cumulative deviations, S_n = standard deviation over n observations
   H > 0.5 → long memory (persistent), H = 0.5 → random walk, H < 0.5 → mean-reverting

2. GPH Log-Periodogram Regression (Geweke & Porter-Hudak, 1983):
   ln(I(ω_j)) = c - d · ln(4·sin²(ω_j/2)) + ε_j
   where I(ω_j) = periodogram at Fourier frequency ω_j
   Estimate d on |r_t| and r²_t (volatility proxies)
   d > 0 → long memory;  d ∈ (0, 0.5) → stationary long memory

3. If H > 0.5 AND d > 0 with 95% CI excluding 0:
   → Proceed to FIGARCH with d as initialization
   Otherwise: FIGARCH is not justified; skip and document.
```

**Report**: H and d per ticker with bootstrap 95% CIs. Add to `garch_parameters.csv`.

---

### 1D. Formal Hypothesis Test Specification Table

**Location**: New Section 2.4 — Statistical Inference Protocol

**What to add**: A reference table for every statistical test used in the pipeline.

| Test | H₀ | H₁ | Statistic | Distribution | Used In |
|------|----|----|-----------|-------------|---------|
| ADF | Unit root (non-stationary) | Stationary | τ (DF regression) | Dickey-Fuller tables | NB01 |
| KPSS | Stationary | Unit root | LM = Σ S²_t / (T² · σ̂²) | KPSS tables | NB01 |
| Jarque-Bera | Normality (S=0, κ=0) | Non-normal | JB = (T/6)·(S² + κ²/4) where κ = excess kurtosis = K-3 | χ²(2) | NB01 |
| Ljung-Box | No autocorrelation up to lag m | Autocorrelation | Q = T(T+2)·Σ ρ̂²_k/(T-k) | χ²(m) | NB03 |
| ARCH-LM | No ARCH effects | ARCH effects | TR² from auxiliary regression | χ²(q) | NB03 |
| Granger | X does not Granger-cause Y | X Granger-causes Y | F-stat on lagged X coefficients | F(p, T-2p-1) | NB02, NB06 |
| Kupiec POF | Correct unconditional VaR coverage | Incorrect coverage | LR_uc (likelihood ratio) | χ²(1) | NB04 |
| Christoffersen CC | Correct coverage + independent violations | Clustering or wrong coverage | LR_cc = LR_uc + LR_ind | χ²(2) | NB04 |
| Diebold-Mariano | Equal predictive accuracy | Unequal accuracy | DM = d̄/sqrt(V̂_HAC(d̄)) | N(0,1) | NB07, NB10 |
| Bai-Perron | No structural breaks | m breaks at unknown dates | sup-F statistics | Bai-Perron tables | NB02 |
| Engle-Granger | No cointegration | Cointegrated | ADF on OLS residuals | Engle-Granger tables | NB02/NB08 |
| Johansen Trace | rank(Π) ≤ r | rank(Π) > r | -T·Σ ln(1-λ̂_i) | Johansen tables | NB02/NB08 |

**Convention**: Use α = 0.05 throughout unless stated otherwise. Apply BH-FDR when testing across 20 tickers.

**ADF variant**: Use constant + trend for price levels, constant-only for returns. Select lag via BIC.

**KPSS complement**: Always run KPSS alongside ADF (confirmatory approach). If ADF rejects and KPSS does not reject → strong evidence of stationarity. If both reject or neither rejects → inconclusive.

---

## Section 2: Systemic Risk & Advanced Analytics

### 2A. Systemic Risk Measures

**Location**: Inline addition to NB04 after copula-based joint tail risk

**What to add**: Three systemic risk measures with formulas.

**CoVaR** (Adrian & Brunnermeier, 2016):
```
CoVaR^{portfolio|i}_α: the α-VaR of the portfolio conditional on stock i being at its VaR

Estimated via quantile regression (Koenker & Bassett, 1978):
  r_{p,t} = α₀ + α₁·r_{i,t} + α₂·VIX_t + α₃·yield_spread_t + ε_t
  evaluated at q_α (α-quantile)

ΔCoVaR^i = CoVaR^{portfolio|r_i = VaR^i_α} - CoVaR^{portfolio|r_i = median(r_i)}

Interpretation: ΔCoVaR^i > 0 means stock i contributes to portfolio systemic tail risk.
```

**MES — Marginal Expected Shortfall** (Acharya et al., 2017):
```
MES_i,α = E[r_{i,t} | r_{portfolio,t} ≤ -VaR_α(r_portfolio)]

  = (1/|S|) · Σ_{t∈S} r_{i,t}    where S = {t : r_{p,t} ≤ quantile(r_p, 1-α)}

Interpretation: Expected loss of stock i on the portfolio's worst days.
Decomposition: Portfolio CVaR = Σ_i w_i · MES_i  (exact, no approximation)
```

**Absorption Ratio** (Kritzman et al., 2011):
```
AR_t = Σ_{j=1}^{k} λ_{j,t} / Σ_{j=1}^{N} λ_{j,t}

where λ_{j,t} = eigenvalues of rolling 252-day correlation matrix, sorted descending
      k = N/5 = 4 (for N=20 stocks, use top 4 eigenvalues)
      N = 20

Interpretation:
  AR ≈ 0.8+ → high systemic risk (few factors explain most variance → correlated market)
  AR ≈ 0.5  → diversified market (variance spread across many factors)

Track AR_t as time series; spikes precede crises (Kritzman et al. show AR rises before 2008 crash).
Use as regime indicator: AR > 75th percentile → high-risk regime.
```

**Output**: `systemic_risk_measures.csv` (per-ticker ΔCoVaR, MES; time-varying AR)

---

### 2B. Cross-Sectional Momentum & Cointegration

**Location**: New sub-section in NB08 (Task D)

**Cross-sectional momentum**:
```
Signal: ranking by trailing 12-month minus 1-month return (skip most recent month to avoid reversal)
  MOM_{i,t} = Π_{s=t-252}^{t-22} (1 + R_{i,s}) - 1

Strategy: Long top quintile (4 stocks), underweight bottom quintile
IC (Information Coefficient) = Spearman rank correlation(MOM signal, 5-day forward return)
  IC > 0.05 with t-stat > 2 → economically significant signal
```

**Cointegration pairs** (natural tech pairs):
```
Step 1: Engle-Granger two-step:
  ln(P_{i,t}) = α + β · ln(P_{j,t}) + ε_t
  Test ε_t for stationarity via ADF (use Engle-Granger critical values, NOT standard ADF)

Step 2: If cointegrated, compute spread:
  s_t = ln(P_{i,t}) - β̂ · ln(P_{j,t})

Step 3: Half-life of mean reversion:
  Δs_t = φ · s_{t-1} + ε_t    →    HL = -ln(2) / ln(1 + φ̂)  days
  (φ < 0 required for mean-reversion; if φ ≥ 0, spread is non-stationary → pair is not tradeable)

Candidate pairs: NVDA/AMD, TSM/AVGO, META/GOOG, PANW/CRWD, CRM/NOW
Report: cointegration p-value, β̂, half-life, and out-of-sample spread Sharpe
```

**Lead-lag cross-correlation**:
```
ρ_{i,j}(k) = Corr(r_{i,t}, r_{j,t+k})    for k ∈ {-5, -4, ..., 0, ..., +4, +5}

Test significance: |ρ| > 2/sqrt(T) at each lag
Visualize as a directed network graph: edge from i→j if i leads j significantly
```

---

### 2C. Options-Implied Analytics

**Location**: Promote from "stretch" to core in NB03/NB04. Add data source to Section 2.1.

**Scope**: 5 most liquid tickers (NVDA, AAPL, MSFT, META, AMZN) using `OpenBB` options chains.

**IV Term Structure**:
```
For each ticker, extract ATM implied vol at the 3 nearest monthly expirations.
Plot IV(τ) for τ ∈ {30d, 60d, 90d}.
Contango (upward-sloping) = normal; backwardation = market stress.
```

**IV Skew**:
```
Skew = IV(25Δ put) - IV(ATM)

Persistent negative skew → crash fear premium (typical for equities)
Skew flattening → complacency (potential risk signal)
```

**Variance Risk Premium (per-ticker)**:
```
VRP_{i,t} = IV²_{i,t,30d} - RV²_{YZ,i,t,21d}

Expected: VRP > 0 on average (investors pay a premium for vol protection)
High VRP → expected positive future returns (compensation for bearing vol risk)
VRP inverts during crashes (realized > implied)
```

**Implied Correlation**:
```
ρ_implied = (IV²_index - Σ_i w²_i · IV²_i) / (Σ_{i≠j} w_i · w_j · IV_i · IV_j)

Use XLK as index proxy, component weights from ETF holdings.
Correlation Risk Premium = ρ_implied - ρ_realized
Persistently positive → investors overpay for correlation risk
```

**Put-Call Ratio**: Daily aggregate put volume / call volume per ticker. Extreme values (>1.5 or <0.5) as sentiment features.

**Data Pipeline Notes**:
- OpenBB `.option_chain(expiration)` returns raw chains; expirations are irregular (monthly + weeklies)
- To get fixed-tenor IV (30d, 60d, 90d): linearly interpolate ATM IV between the two nearest expirations bracketing the target tenor, weighted by days-to-expiry distance
- ATM strike: nearest strike to current spot price. 25-delta strike: use Black-Scholes delta inversion or select the strike closest to Δ=0.25 from the chain
- Store weekly snapshots in `data/features/options_implied.parquet` (daily is excessive given chain staleness)
- VRP is in **annualized variance units** (IV² - RV²), not volatility units. To convert to vol units: sign(VRP) · sqrt(|VRP|)

---

### 2D. Earnings Seasonality & PEAD

**Location**: Inline addition to NB02 (event studies) and NB08 (features)

**Methodology**:
```
1. Extract earnings dates via OpenBB (.earnings_dates property) or earnings calendar APIs
2. Define event windows: [-5, -1] pre-earnings, [0, 0] announcement day, [+1, +20] post-earnings

Pre-Earnings Drift:
  CAR[-5, -1] = Σ_{t=-5}^{-1} AR_t    where AR_t = r_t - (α̂ + β̂ · r_{m,t})
  Test: t_CAR = CAR / (σ_AR · sqrt(5))

Post-Earnings Announcement Drift (PEAD):
  SUE_i = (EPS_actual - EPS_consensus) / σ(forecast_errors)
  Sort by SUE quintile; measure CAR[+1, +20] for top vs bottom quintile
  PEAD premium = CAR(Q5) - CAR(Q1)

Earnings Vol Crush:
  Crush_i = IV_{i,t-1} - IV_{i,t+1}   (implied vol drops after uncertainty resolves)
  Average crush magnitude across all earnings events per ticker

Cross-Ticker Transmission:
  AR_{NVDA,t} around TSMC earnings date → measures information flow from foundry to designer
  Test: is CAR(NVDA | TSM earnings) statistically significant?
```

**Data source for analyst estimates**: Use `OpenBB` `.earnings_history` for actual vs. estimate EPS. If consensus estimates are unavailable for the full 10-year window, use a simplified SUE: SUE_i = (EPS_actual - EPS_{prev_quarter}) / σ(EPS changes over last 8 quarters). This seasonal random walk SUE is a well-accepted proxy (Foster, Olsen & Shevlin, 1984). Document any data limitations.

---

## Section 3: Portfolio Construction & Performance Depth

### 3A. Robust/Worst-Case Optimization

**Location**: Inline addition to NB11 optimization methods (add as methods 6-8)

**6. Worst-Case Mean-Variance** (Goldfarb & Iyengar, 2003):
```
min_w  max_{Σ ∈ U}  w'Σw    s.t. w'μ ≥ r_target, standard constraints

Uncertainty set (Frobenius ball):
  U = {Σ : ||Σ - Σ̂||_F ≤ δ}

SDP reformulation via S-lemma:
  min_w  w'Σ̂w + δ·||w||₂²    s.t. constraints

This adds a ridge-like penalty proportional to δ, shrinking weights toward equal-weight.
Calibrate δ from bootstrap variance of Σ̂ or cross-validation.
```

**7. Maximum Diversification Portfolio** (Choueifaty & Coignard, 2008):
```
max_w  DR(w) = w'σ / sqrt(w'Σw)

where σ = (σ_1, ..., σ_N)' = vector of individual asset volatilities
DR ≥ 1 always (equality only if all correlations = 1)
Higher DR → more effective diversification

Equivalent to: min_w w'Ρw  s.t. w'1 = 1, w ≥ 0
where Ρ = correlation matrix (not covariance)
```

**8. Resampled Efficient Frontier** (Michaud, 1998):
```
For b = 1, ..., B (B = 1000):
  Draw μ̂_b ~ N(μ̂, Σ̂/T)
  Draw Σ̂_b ~ Wishart(scale=Σ̂/T, df=T)
    E[Σ̂_b] = T · (Σ̂/T) = Σ̂  (correct expected value, no additional rescaling)
  Solve: w*_b = argmin w'Σ̂_b·w  s.t. w'μ̂_b ≥ r_target, constraints
Final weights: w_resampled = (1/B) · Σ_b w*_b

Produces more stable, less extreme allocations than classical Markowitz.
Note: In practice, use np.random.default_rng().multivariate_normal() for μ̂_b
and scipy.stats.wishart(df=T, scale=Σ̂/T).rvs() for Σ̂_b (returns the draw directly).
```

---

### 3B. Advanced Performance Metrics

**Location**: Inline addition to NB11 performance metrics + NB12 stress test dashboard

**Add to existing Sharpe/Sortino/Calmar**:
```
Omega Ratio (Keating & Shadwick, 2002):
  Ω(θ) = ∫_θ^∞ (1-F(r)) dr / ∫_{-∞}^θ F(r) dr    where θ = threshold (use r_f)
  Ω > 1 → positive risk-adjusted performance. Captures full return distribution, not just first 2 moments.

Information Ratio (vs XLK benchmark):
  IR = (R̄_p - R̄_XLK) / TE    where TE = σ(R_p - R_XLK) · sqrt(252)
  IR > 0.5 = good active management; IR > 1.0 = exceptional

Tracking Error:
  TE = std(R_p,t - R_b,t) · sqrt(252)

Ulcer Index (Martin, 1987):
  UI = sqrt((1/T) · Σ_{t=1}^T D²_t)    where D_t = (P_t - max_{s≤t} P_s) / max_{s≤t} P_s
  Captures both depth AND duration of drawdowns (unlike max DD which is a single point)

Pain Index:
  PI = (1/T) · Σ_{t=1}^T |D_t|    (mean of percentage drawdowns, less sensitive to outliers)

Conditional Drawdown at Risk (CDaR, Chekhlov et al., 2005):
  CDaR_α = E[D_t | D_t ≥ quantile(D, α)]
  CVaR analog for drawdowns. More informative than max drawdown for tail risk.
```

---

### 3C. Turnover Constraint Formalization

**Location**: Inline replacement/expansion in NB11 constraints section

**Current**: "Turnover constraint: max 20% monthly rebalance" (ambiguous)

**Replace with**:
```
L1 Turnover Constraint:
  Σ_i |w_{i,new} - w_{i,old}| ≤ τ    where τ = 0.20 (20% monthly)

LP reformulation (for use in cvxpy):
  Introduce auxiliary variables t_i ≥ 0:
    w_{i,new} - w_{i,old} ≤ t_i    ∀i
    -(w_{i,new} - w_{i,old}) ≤ t_i  ∀i    (equivalently: t_i ≥ |w_{i,new} - w_{i,old}|)
    Σ_i t_i ≤ τ

Note: w_{i,old} must be the DRIFTED weight (after market returns), not the prior target:
  w_{i,old,drifted} = w_{i,target,prev} · (1 + r_{i,t}) / Σ_j w_{j,target,prev} · (1 + r_{j,t})
```

---

### 3D. Transaction Cost Model Enhancement

**Location**: Inline replacement in NB11 backtest framework

**Current**: Flat 10 bps round-trip

**Replace with**:
```
Spread + Market Impact Model (per-trade cost in dollars):
  Cost_i = Spread_cost_i + Impact_cost_i

  Spread_cost_i = (spread_i / 2) · |Δw_i| · AUM
    (linear in trade size — half-spread times trade notional)

  Impact_cost_i = η · σ_{i,daily} · (|Δw_i| · AUM) · sqrt(|Δw_i| · AUM / ADV_i)
    (nonlinear — square-root market impact per Almgren-Chriss)
    Equivalently: η · σ_i · |Δw_i|^{3/2} · AUM^{3/2} / sqrt(ADV_i)

where:
  spread_i = estimated bid-ask spread (calibrated per ticker):
    Mega-cap (AAPL, MSFT, AMZN, GOOG, META): 1-2 bps
    Large-cap semi (NVDA, AVGO, TSM, AMD, MU): 2-4 bps
    Mid-cap tech (PLTR, DDOG, CRWD, ANET, XYZ): 5-10 bps
    Low-liquidity (SNPS): 8-15 bps
  η ≈ 0.1 (Almgren-Chriss temporary impact coefficient)
  σ_{i,daily} = daily volatility
  ADV_i = 20-day average daily volume in dollars
  AUM = assumed portfolio size (parameterize: test at $10M, $100M, $1B)

Total rebalancing cost = Σ_i (Spread_cost_i + Impact_cost_i)
Compare backtest results: flat 10bps vs calibrated model at different AUM levels.
Note: Impact cost is negligible at $10M AUM but material at $1B for low-ADV names (SNPS).
```

---

### 3E. Regime-Conditional Efficient Frontiers

**Location**: Inline addition to NB11 regime-adaptive strategy section

**What to add**:
```
Compute and overlay 3 efficient frontiers:
  1. Full-sample frontier: using Σ̂_full, μ̂_full
  2. Bull-regime frontier: using Σ̂_bull, μ̂_bull (HMM state with highest mean)
  3. Bear-regime frontier: using Σ̂_bear, μ̂_bear (HMM state with lowest mean)

For each frontier: trace 50 points from min-variance to max-return portfolio.
Mark the tangency (max Sharpe) portfolio on each frontier.

Key insight to document:
  - Bear frontier shifts DOWN and LEFT (lower returns, but also different risk structure)
  - Correlation increases in bear regimes → frontier contracts inward
  - Min-variance portfolio composition shifts: more weight to defensive names (AAPL, MSFT, SAP)
  - This visually justifies the regime-adaptive strategy (switching from MV to HRP in bears)
```

---

### 3F. Conditional Diversification Benefit

**Location**: Inline addition to NB11 or NB12

```
CDB(w) = 1 - σ_p(w) / w'σ

where σ_p(w) = sqrt(w'Σw) = portfolio volatility
      w'σ = Σ_i w_i · σ_i = weighted average of individual volatilities

CDB ∈ [0, 1]:
  CDB = 0  → no diversification (all correlations = 1)
  CDB = 1  → perfect diversification (all correlations = -1/(N-1))

Compute per regime:
  CDB_bull = 1 - sqrt(w'Σ_bull w) / (w'σ_bull)
  CDB_bear = 1 - sqrt(w'Σ_bear w) / (w'σ_bear)

Expected: CDB_bear << CDB_bull (correlation spike during crises collapses diversification)
This directly motivates regime-adaptive allocation (HRP in bear → preserves diversification)
```

---

### 3G. Covariance Estimation Comparison

**Location**: Inline addition to NB11 risk estimation section

**Expand from 3 to 5 methods**:
```
4. Statistical Factor Model (PCA-based):
   Σ_factor = F · Λ · F' + D
   where F = first k eigenvectors (retain k explaining 90% of variance)
         Λ = diagonal of top k eigenvalues
         D = diagonal matrix of residual variances
   Advantage: Reduces estimation error by projecting onto low-dimensional factor space.

5. Non-Linear Shrinkage (Ledoit & Wolf, 2020):
   Σ_NL = U · diag(d*_1, ..., d*_N) · U'
   where U = sample eigenvectors, d*_i = non-linearly shrunk eigenvalues
   Analytical formula (no bootstrap needed). Optimal under Frobenius loss.
```

**Comparison protocol**: For each covariance estimator, compute the resulting optimal portfolio weights (Markowitz max-Sharpe) and measure:
- Weight stability: avg |Δw_i| across rebalance dates
- Out-of-sample Sharpe ratio
- Realized portfolio variance vs. predicted portfolio variance

---

### 3H. Tail Risk Budgeting

**Location**: Inline addition to NB11 optimization methods

```
CVaR-ERC Portfolio:
  Find w such that: w_i · MCVaR_i(w) = CVaR(w) / N    ∀i

where MCVaR_i = ∂CVaR(w)/∂w_i  (marginal CVaR contribution of asset i)

Estimated from scenarios:
  MCVaR_i ≈ E[r_{i,t} | w'r_t ≤ -VaR_α(w'r_t)]
  Note: MCVaR_i ≈ MES_i (from Section 2A) in the limit for continuous distributions.
  In finite samples, use the scenario-based approximation directly:
    MCVaR_i = (1/|S|) · Σ_{t∈S} r_{i,t}  where S = {t : w'r_t ≤ quantile(w'r, 1-α)}

Solve via iterative optimization (no closed-form unlike vol-ERC).
Compare: vol-ERC vs CVaR-ERC weights and out-of-sample performance.
Key difference: CVaR-ERC allocates less to assets with heavy left tails.
```

---

## Section 4: Statistical Rigor & Remaining Items

### 4A. Bootstrap Inference Specification

**Location**: New Section 2.4 — Statistical Inference Protocol (subsection)

```
1. Stationary Block Bootstrap (Politis & Romano, 1994):
   - For return-based statistics (Sharpe ratio, performance metrics)
   - Block length: automatic selection via Politis & White (2004):
     b_opt estimated from spectral density at frequency zero
   - B = 10,000 replications for 95% CIs; B = 50,000 for 99% CIs
   - Report: bias-corrected bootstrap percentile intervals

2. Residual Bootstrap for GARCH (Pascual, Romo & Ruiz, 2006):
   - Fit GARCH → extract standardized residuals ẑ_t
   - Resample ẑ*_t with replacement (IID, since residuals should be IID after GARCH filtering)
   - Reconstruct: ε*_t = ẑ*_t · σ_t(θ̂)  →  re-estimate θ̂* from reconstructed series
   - Repeat B times → distribution of GARCH parameters
   - Use for: parameter CIs, persistence (α+β) CI, half-life CI

3. Circular Block Bootstrap for DCC parameters:
   - Apply to standardized residual pairs
   - Report: CIs for DCC parameters (a, b) and unconditional correlation Q̄

4. Bootstrap for Sharpe Ratio comparison (Ledoit & Wolf, 2008):
   - Studentized circular block bootstrap
   - H₀: Sharpe(portfolio A) = Sharpe(portfolio B)
   - More powerful than simple DM test for Sharpe comparisons
```

---

### 4B. ADR Premium Analysis (Optional — enrichment, not core)

**Location**: Inline addition to NB01 (data section) or NB02 (macro context)

```
TSM ADR Premium:
  premium_t = (P_{TSM,ADR,t} / 5) / (P_{2330.TW,t} · FX_{TWD/USD,t}) - 1
  where 5 = ADR conversion ratio (1 ADR = 5 ordinary shares)

SAP ADR Premium:
  premium_t = P_{SAP,ADR,t} / (P_{SAP.DE,t} · FX_{EUR/USD,t}) - 1
  where 1 = ADR conversion ratio (1 ADR = 1 ordinary share)

Download: 2330.TW and SAP.DE via OpenBB; TWD/USD via TWDUSD=X; EUR/USD via EURUSD=X

Interpretation:
  Persistent positive premium → capital inflow to US-listed ADR (demand-driven)
  Negative premium → outflow or FX expectations shifting
  Premium mean-reverts → arbitrage forces keep it bounded (typically ±1-2%)

Use as features: ADR_premium_TSM and ADR_premium_SAP as ML inputs (t-1 lagged)
```

---

### 4C. Feature Selection Methodology

**Location**: Inline addition to NB07/NB08 feature engineering description

```
1. Multicollinearity Diagnostics:
   VIF_j = 1 / (1 - R²_j)    where R²_j = R² from regressing feature j on all other features
   Remove features with VIF > 10 (severe multicollinearity)
   Expected high-VIF pairs: realized_vol_5d/21d (correlated), GARCH_vol/realized_vol

2. Stability Selection (Meinshausen & Bühlmann, 2010):
   For b = 1, ..., 100:
     Subsample 50% of features randomly
     Fit Lasso with λ chosen via BIC
     Record which features are selected
   Retain features selected in > 60% of subsamples (stability threshold π_thr = 0.6)
   Provides finite-sample FDR control on selected feature set

3. Feature Set Comparison:
   - Full feature set (~40-50 features)
   - VIF-filtered set (remove VIF > 10)
   - Stability-selected set
   Report: OOS RMSE/MAE for each set. Demonstrate that pruning does not degrade performance.
```

---

### 4D. White's Reality Check / SPA Test

**Location**: Inline addition to NB10 model comparison section

```
Problem: When comparing M models, the best-performing model may appear significant by chance.
The more models tested, the higher the probability of a spurious "winner" (data-snooping).

White's Reality Check (2000):
  H₀: The best model is no better than the benchmark (equal-weight or random walk)
  Test statistic: T_RC = max_m (d̄_m)    where d̄_m = mean loss differential vs benchmark
  Bootstrap: Resample d_{m,t} under H₀ using stationary block bootstrap
  p-value: proportion of bootstrap samples where max_m(d̄*_m) > T_RC

SPA Test (Hansen, 2005) — more powerful refinement:
  Uses studentized test statistic (accounts for different variances across models)
  T_SPA = max_m (d̄_m / σ̂_m)
  Same bootstrap procedure, but with studentized statistics

Report: SPA p-value for winning model. If p < 0.05, the best model is genuinely superior
even after accounting for having tested all M alternatives.
```

---

### 4E. Realized Volatility Signature Plot

**Location**: Inline addition to NB03

```
For 3-5 liquid tickers (NVDA, AAPL, MSFT, META, AMZN):
  Download recent 60 days of intraday data at 5-minute frequency via OpenBB
  (Note: OpenBB limits 1-min data to 7 days; 5-min data available for 60 days)
  Compute realized variance at multiple sampling frequencies:
    RV_Δ = Σ_{j=1}^{T/Δ} r²_{j,Δ}    for Δ ∈ {5min, 15min, 30min, 1hr, daily}

  Plot RV_Δ vs. ln(Δ):
    Flat curve → no microstructure noise → safe to use high-frequency data
    Rising at high frequency → noise dominates → optimal frequency is at the "elbow"
    Typical optimal: 5-15 minute bars for liquid US equities

Purpose: Validates that daily OHLC-based estimators (Yang-Zhang) are appropriate.
If the signature plot shows significant noise below 30-minute bars, document this as a
limitation of daily-only analysis and note that institutional-grade vol estimation
would benefit from 5-minute bars.
```

---

### 4F. Updated Citations

**Add to Section 10**:

| Method | Key Paper | Journal |
|--------|-----------|---------|
| CoVaR | Adrian, T. & Brunnermeier, M. (2016) | *American Economic Review* 106(7) |
| MES | Acharya, V. et al. (2017) | *Review of Financial Studies* 30(1) |
| Absorption Ratio | Kritzman, M. et al. (2011) "Principal Components as a Measure of Systemic Risk" | *J. Portfolio Management* 37(4) |
| Robust Optimization | Goldfarb, D. & Iyengar, G. (2003) "Robust Portfolio Selection Problems" | *Mathematics of Operations Research* 28(1) |
| Maximum Diversification | Choueifaty, Y. & Coignard, Y. (2008) "Toward Maximum Diversification" | *J. Portfolio Management* 35(1) |
| Resampled Efficiency | Michaud, R. (1998) *Efficient Asset Management* | Harvard Business School Press |
| Omega Ratio | Keating, C. & Shadwick, W. (2002) "A Universal Performance Measure" | *J. Performance Measurement* 6(3) |
| CDaR | Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005) "Drawdown Measure in Portfolio Optimization" | *Intern. J. Theoretical & Applied Finance* 8(1) |
| Market Impact | Almgren, R. & Chriss, N. (2000) "Optimal Execution of Portfolio Transactions" | *J. Risk* 3(2) |
| Walk-Forward Purge | Lopez de Prado, M. (2018) *Advances in Financial Machine Learning* | Wiley, Chapter 7 |
| Reality Check | White, H. (2000) "A Reality Check for Data Snooping" | *Econometrica* 68(5) |
| SPA Test | Hansen, P. (2005) "A Test for Superior Predictive Ability" | *J. Business & Econ. Stats* 23(4) |
| Stationary Bootstrap | Politis, D. & Romano, J. (1994) "The Stationary Bootstrap" | *JASA* 89(428) |
| Block Length | Politis, D. & White, H. (2004) "Automatic Block-Length Selection for the Dependent Bootstrap" | *Econometric Reviews* 23(1) |
| GARCH Bootstrap | Pascual, L., Romo, J. & Ruiz, E. (2006) "Bootstrap Prediction for Returns and Volatilities in GARCH Models" | *Computational Stats & Data Analysis* 50(9) |
| Stability Selection | Meinshausen, N. & Bühlmann, P. (2010) "Stability Selection" | *JRSS-B* 72(4) |
| Sharpe Bootstrap | Ledoit, O. & Wolf, M. (2008) "Robust Performance Hypothesis Testing with the Sharpe Ratio" | *J. Empirical Finance* 15(5) |
| Cointegration | Engle, R. & Granger, C. (1987) "Co-integration and Error Correction" | *Econometrica* 55(2) |
| GPH Estimator | Geweke, J. & Porter-Hudak, S. (1983) "The Estimation and Application of Long Memory Time Series Models" | *JRSS-B* 15(1) |
| NL Shrinkage | Ledoit, O. & Wolf, M. (2020) "Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices" | *Annals of Statistics* 48(5) |
| Hurst Exponent | Hurst, H.E. (1951) "Long-term Storage Capacity of Reservoirs" | *Trans. ASCE* 116 |

---

## Implementation Strategy

**Approach**: Hybrid (inline enrichment + new cross-cutting sections)

**New sections to create**:
- Section 2.4 — Statistical Inference Protocol (1B, 1D, 4A)

**Inline additions by notebook**:
- NB01: ADR premium (4B)
- NB02: Earnings seasonality (2D), cointegration (2B)
- NB03: Hurst exponent (1C), volatility signature plot (4E)
- NB04: Systemic risk measures (2A), options-implied (2C)
- NB07/NB08: Embargo (1A), feature selection (4C), cross-sectional momentum (2B)
- NB10: White's Reality Check (4D)
- NB11: Robust optimization (3A), performance metrics (3B), turnover formalization (3C), transaction cost model (3D), regime frontiers (3E), CDB (3F), covariance comparison (3G), tail risk budgeting (3H)
- Section 10: Updated citations (4F)

**Estimated additions**: ~700-800 lines of rigorous content

---

## Success Criteria

1. Every statistical test has H₀, H₁, test statistic, and distribution specified
2. Every mathematical formula is dimensionally consistent and uses correct notation
3. Walk-forward protocol is leakage-free (embargo + purge documented)
4. Multiple testing correction applied wherever testing across 20 tickers
5. At least 3 systemic risk measures with full formulas
6. At least 3 additional portfolio optimization methods beyond current 5
7. Options-implied analytics promoted from stretch to core
8. All new references include author, year, title, journal, and volume
9. Bootstrap methods specified for each inference context
10. Cross-sectional analytics (momentum, cointegration, lead-lag) present
