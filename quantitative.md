# CLAUDE.MD — Tech Sector Risk Management & ML Forecasting Pipeline

> **Project Scope**: Multi-notebook quantitative analysis of 20 actively traded tech stocks across 10 technology sub-sectors, spanning 10 years of daily data (2016-03 → 2026-03). Combines classical econometrics, machine learning, deep learning, regime detection, NLP sentiment, and portfolio optimization into a production-grade risk management framework.

> **Author**: Laurent
> **Target Audience**: Master's admissions committees (SKEMA MSc Financial Markets & Investments, emlyon MSc Data Science & AI Strategy, ESCP MSc Business Analytics & AI), portfolio showcasing, and professional development.

---

## 0 · QUICK START

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate tech-risk-ml

# OR with pip
python -m venv .venv && source .venv/bin/activate   # Unix
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Set FRED API key (required for macro data in NB01/NB02)
export FRED_API_KEY="b70ecbb7d8a4ca619e8a885e5807f9cc"                  # Unix
set FRED_API_KEY=b70ecbb7d8a4ca619e8a885e5807f9cc                      # Windows

# 3. Run notebooks in order (critical path)
jupyter notebook notebooks/NB01_data_ingestion_eda.ipynb

# 4. Run tests (verify no lookahead bias)
pytest tests/ -v

# 5. Run specific test suites
pytest tests/test_pipeline_integrity.py -v   # Lookahead bias checks
pytest tests/test_risk_metrics.py -v         # VaR/CVaR correctness
pytest tests/test_data_loader.py -v          # Data pipeline
```

**Notebook execution order**: Follow the dependency graph in Section 6. After NB01 completes, NB02/NB03/NB05/NB06 can run in parallel.

---

## 1 · STOCK UNIVERSE — 20 Tickers × 10 Technology Sub-Sectors

The universe is deliberately designed to capture the full breadth of the tech ecosystem — from hardware-layer semiconductors to application-layer SaaS — to expose cross-sector contagion, idiosyncratic risk, and regime-dependent correlation structures.

| # | Sub-Sector | Ticker | Company | Constraint Group | Selection Rationale |
|---|---|---|---|---|---|
| 1 | **GPU / AI Semiconductors** | `NVDA` | NVIDIA | Semiconductors | De facto AI compute monopolist; highest vol and beta in the universe; gold-standard GPU for training LLMs |
| 2 | **Broadband / ASIC Semiconductors** | `AVGO` | Broadcom | Semiconductors | Custom ASIC leader for hyperscalers + VMware software acquisition; diversified semi-software hybrid |
| 3 | **Foundry / Chip Manufacturing** | `TSM` | TSMC (ADR) | Semiconductors | World's largest chip foundry; single-point-of-failure geopolitical risk (Taiwan Strait); capex cycle proxy |
| 4 | **EDA / Semiconductor Tools** | `SNPS` | Synopsys | Semiconductors | Upstream enabler of all chip design; recurring license model; lower vol proxy for semi cycle |
| 5 | **Cloud / Enterprise Platforms** | `MSFT` | Microsoft | Cloud & Platforms | Azure + M365 + Copilot AI; mega-cap anchor; low-beta tech bellwether |
| 6 | **Consumer Internet / Advertising** | `META` | Meta Platforms | Consumer Internet | Social media + Reality Labs + Llama AI; ad-cycle sensitivity; high FCF yield |
| 7 | **Search / AI Infrastructure** | `GOOG` | Alphabet (Class C) | Consumer Internet | Search + YouTube + Google Cloud + DeepMind; regulatory overhang (antitrust); defensive ad exposure |
| 8 | **E-Commerce / Cloud** | `AMZN` | Amazon | Cloud & Platforms | AWS + retail + logistics; dual revenue engine; capex-heavy AI buildout phase |
| 9 | **Consumer Hardware / Ecosystem** | `AAPL` | Apple | Consumer Hardware | Hardware + services flywheel; FX-sensitive (>60% international rev); low tech-vol anchor |
| 10 | **Enterprise SaaS / CRM** | `CRM` | Salesforce | Enterprise Software | Enterprise AI (Agentforce); subscription revenue; post-activist restructuring efficiency gains |
| 11 | **Cybersecurity** | `PANW` | Palo Alto Networks | Cybersecurity | Platform consolidation leader; zero-trust + SASE + AI-native security; recession-resistant demand |
| 12 | **Endpoint Security** | `CRWD` | CrowdStrike | Cybersecurity | Cloud-native Falcon platform; high-growth SaaS; recovery narrative post-July 2024 outage event |
| 13 | **Cloud Observability** | `DDOG` | Datadog | Cybersecurity | AI observability + DevSecOps monitoring; 30%+ FCF margins; recurring usage-based revenue |
| 14 | **Fintech / Payments** | `XYZ` | Block (formerly Square) | Fintech | Payments ecosystem + Cash App + Bitcoin exposure; consumer discretionary sensitivity. Note: ticker changed from SQ to XYZ on 2025-01-21 |
| 15 | **Enterprise Workflow SaaS** | `NOW` | ServiceNow | Enterprise Software | Workflow automation + AI agents; 20%+ organic growth; one of highest-quality SaaS compounders |
| 16 | **Data / AI Analytics** | `PLTR` | Palantir | Enterprise Software | Government + commercial AI platforms (AIP, Foundry); defense spending proxy; high valuation risk |
| 17 | **Networking / Data Center** | `ANET` | Arista Networks | Networking | 400G/800G Ethernet switching for AI data centers; hyperscaler capex beneficiary |
| 18 | **Memory / Storage Semiconductors** | `MU` | Micron Technology | Semiconductors | HBM (High Bandwidth Memory) leader for AI GPUs; deeply cyclical; DRAM/NAND pricing power |
| 19 | **GPU / Discrete Compute** | `AMD` | Advanced Micro Devices | Semiconductors | EPYC server CPUs + MI300 AI accelerators; Nvidia's primary competitor; Xilinx FPGA integration |
| 20 | **ERP / Enterprise Software** | `SAP` | SAP SE (ADR) | Enterprise Software | World's largest ERP provider; S/4HANA cloud migration; European tech anchor; EUR/USD FX exposure |

### Sector Constraint Groups (for portfolio optimization)

The 20 sub-sectors roll up into 7 constraint groups for concentration limits:

| Constraint Group | Tickers | Count | Max Group Weight (30%) |
|---|---|---|---|
| **Semiconductors** | NVDA, AVGO, TSM, SNPS, MU, AMD | 6 | 30% (max ~5% avg per stock) |
| **Cloud & Platforms** | MSFT, AMZN | 2 | 30% |
| **Consumer Internet** | META, GOOG | 2 | 30% |
| **Consumer Hardware** | AAPL | 1 | 10% (single stock cap binds first) |
| **Enterprise Software** | CRM, NOW, PLTR, SAP | 4 | 30% |
| **Cybersecurity** | PANW, CRWD, DDOG | 3 | 30% |
| **Fintech** | XYZ | 1 | 10% (single stock cap binds first) |
| **Networking** | ANET | 1 | 10% (single stock cap binds first) |

**Note**: The Semiconductors group has 6 stocks with a 30% group cap and 10% individual cap. This means the optimizer can allocate at most 30% to semiconductors total, averaging ~5% per stock. This is intentionally restrictive to prevent over-concentration in a single cyclical sub-sector.

### Selection Methodology
- **Liquidity filter**: 18 of 20 tickers have average daily volume > 5M shares. Exception: SNPS averages ~2-3M shares/day but is retained for its unique upstream semiconductor design exposure and deep options markets. All tickers have sufficient liquidity for institutional-scale portfolio execution.
- **Sector diversification**: Spans the full tech value chain from physical layer (TSMC, MU) → design layer (NVDA, SNPS) → platform layer (MSFT, AMZN) → application layer (CRM, NOW, PANW) → analytics layer (PLTR, DDOG).
- **Risk profile diversity**: Includes low-beta anchors (AAPL, MSFT, SAP), high-beta momentum names (NVDA, PLTR, CRWD), cyclical plays (MU, TSM), and defensive growers (PANW, NOW).
- **Geopolitical exposure**: TSM (Taiwan Strait), SAP (EU regulation + EUR/USD), AAPL (China supply chain), NVDA (US-China chip export controls).

### Ticker Change Log
| Ticker | Event | Date | Notes |
|---|---|---|---|
| `META` | FB → META | 2022-06-09 | OpenBB handles seamlessly under META for full history |
| `XYZ` | SQ → XYZ | 2025-01-21 | Must use XYZ for data after Jan 2025; use SQ for pre-2025 historical download, then merge |

---

## 2 · DATA ARCHITECTURE

### 2.1 Primary Data Sources

| Data Type | Source | API/Ticker | Frequency | Fields |
|---|---|---|---|---|
| **OHLCV Price** | OpenBB | 20 tickers + benchmarks | Daily | Open, High, Low, Close, Adj Close, Volume |
| **Risk-Free Rate** | FRED API | `DTB3`, `DGS10`, `DGS2` | Daily | 13-week T-bill, 10Y yield, 2Y yield, 2s10s spread |
| **Risk-Free Rate** (alt) | OpenBB | `^IRX` | Daily | 13-week T-bill (alternative source for cross-validation) |
| **VIX / VVIX** | OpenBB | `^VIX`, `^VVIX` | Daily | Implied volatility, vol-of-vol |
| **Sector Benchmarks** | OpenBB | `XLK`, `SMH`, `HACK`, `SKYY` | Daily | Tech ETF, Semi ETF, Cyber ETF, Cloud ETF |
| **Market Benchmarks** | OpenBB | `SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `DX-Y.NYB` | Daily | Cross-asset regime context. Note: DXY uses `DX-Y.NYB` ticker on Yahoo Finance |
| **Macro Indicators** | FRED API | `CPIAUCSL`, `PCEPI`, `FEDFUNDS`, `MANEMP`, `UNRATE`, `GDP` | Monthly/Quarterly | CPI, PCE, Fed Funds Rate, ISM Manufacturing, Unemployment, GDP |
| **News Sentiment** | FinBERT via HuggingFace | `ProsusAI/finbert` | Daily | Per-ticker sentiment scores from financial news headlines |
| **Options Data** | OpenBB options chains | Per-ticker (5 liquid names) | Weekly snapshots | IV term structure (30d/60d/90d), IV skew (25Δ put - ATM), put-call ratio, VRP |

### 2.2 Derived Features (Engineered per Notebook)

```
Returns:       log_return, simple_return, excess_return (over Rf)
Volatility:    realized_vol_5d/21d/63d, Parkinson_vol, Garman-Klass_vol, Yang-Zhang_vol
Momentum:      RSI_14, MACD, Bollinger_%B, rate_of_change_20d
Volume:        OBV, VWAP_deviation, volume_zscore_20d
Cross-asset:   beta_SPY_63d, correlation_XLK_21d, DXY_sensitivity
Macro:         yield_curve_slope, real_rate, CPI_surprise, VIX_term_structure
Regime:        HMM_state, Markov_regime, volatility_regime_label
Sentiment:     FinBERT_score_3d_MA, sentiment_dispersion, news_volume
               (ALL sentiment features MUST be lagged by t-1 when used as ML inputs
                to prevent contemporaneous information leakage — see NB06 for details)
```

#### 2.2.1 Return Definitions

```
Simple return:    R_t = (P_t - P_{t-1}) / P_{t-1}
Log return:       r_t = ln(P_t / P_{t-1})
Excess return:    r^e_t = r_t - r_f,t    where r_f,t = DTB3_t / 252 (daily risk-free)
```

**Convention**: Use log returns for all statistical modeling (additivity over time, better normality approximation). Convert to simple returns only for portfolio P&L aggregation: R_portfolio = sum_i(w_i * R_i,t), since log returns are NOT additive across assets.

#### 2.2.2 Volatility Estimator Formulas (per rolling window of n days)

All estimators below are annualized by multiplying by sqrt(252).

**Close-to-Close (Standard)**:
```
σ²_CC = (1/(n-1)) * Σ_{i=1}^{n} (r_i - r̄)²
```

**Parkinson (1980)** — uses High-Low range, ~5.2x more efficient than close-to-close:
```
σ²_P = (1/(4n·ln2)) * Σ_{i=1}^{n} (ln(H_i/L_i))²
```
Bias: underestimates vol when price gaps between sessions (no overnight component).

**Garman-Klass (1980)** — uses OHLC, ~7.4x more efficient:
```
σ²_GK = (1/n) * Σ_{i=1}^{n} [ 0.5·(ln(H_i/L_i))² - (2ln2-1)·(ln(C_i/O_i))² ]
```
Bias: assumes no overnight jumps (open = prior close). Breaks during earnings/gaps.

**Yang-Zhang (2000)** — PREFERRED estimator. Combines overnight, open-to-close, and Rogers-Satchell components. Handles opening jumps and drift:
```
σ²_YZ = σ²_overnight + k·σ²_close-to-close + (1-k)·σ²_RS

where:
  σ²_overnight = (1/(n-1)) * Σ(ln(O_i/C_{i-1}) - mean(ln(O/C_{-1})))²
  σ²_close     = (1/(n-1)) * Σ(ln(C_i/O_i) - mean(ln(C/O)))²
  σ²_RS        = (1/n) * Σ[ ln(H_i/C_i)·ln(H_i/O_i) + ln(L_i/C_i)·ln(L_i/O_i) ]
  k            = 0.34 / (1.34 + (n+1)/(n-1))
```
The k coefficient minimizes the estimator's variance. Yang-Zhang is minimum-variance among unbiased estimators using OHLC data.

#### 2.2.3 Rolling Beta Estimation

```
β_i,t = Cov(r_i, r_m)_t / Var(r_m)_t

Computed over trailing 63-day (quarterly) windows via OLS:
  r^e_{i,s} = α_i + β_i · r^e_{m,s} + ε_{i,s}    for s ∈ [t-63, t]
```

Report Newey-West t-statistics for α (Jensen's alpha) and β to account for serial correlation in residuals.

### 2.3 Data Quality Protocol

Every notebook must enforce these checks **before** any modeling:

1. **Missing data**: Forward-fill up to 5 business days, then flag and interpolate. Document all fills.
2. **Corporate actions**: Use `Adj Close` exclusively. Verify no split/dividend artifacts via return distribution checks (flag any single-day |return| > 25%).
3. **Survivorship bias**: All 20 tickers have continuous 10-year trading history. If any ticker IPO'd after March 2016 (e.g., CRWD IPO June 2019, DDOG IPO Sept 2019, PLTR DPO Sept 2020), use available data only and document the truncated window. **Never fabricate or backfill pre-IPO prices.**
4. **Ticker changes**: XYZ (formerly SQ) requires downloading SQ data for pre-2025 and XYZ for post-2025, then merging. META handles FB→META seamlessly in OpenBB.
5. **Lookahead bias**: All train/test splits must be strictly temporal. No future information in feature engineering. Rolling windows only. Sentiment features must be lagged t-1.
6. **Stationarity**: ADF and KPSS tests on all return and volatility series. Document p-values. Differencing if needed for level series.
7. **Short-history tickers**: GARCH parameter estimates for CRWD (~1700 days), DDOG (~1630 days), and especially PLTR (~1375 days) will have wider confidence intervals. FIGARCH may not be reliably estimable for tickers with < 1500 observations. Document estimation uncertainty.

### 2.4 Statistical Inference Protocol

#### 2.4.1 Multiple Testing Correction — Benjamini-Hochberg FDR

When testing the same hypothesis across m=20 tickers (ADF, Granger, Kupiec, DM tests), control the False Discovery Rate:
```
BH procedure (Benjamini & Hochberg, 1995):
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)    (m = number of tests, typically 20)
2. Find largest k such that p_(k) ≤ (k/m) · q      (q = 0.05 target FDR)
3. Reject H₀ for all tests with p ≤ p_(k)
```
Report both raw and BH-adjusted p-values in ALL multi-ticker test tables. Applies to: ADF (NB01), KPSS (NB01), Granger causality (NB02, NB06), ARCH-LM (NB03), Kupiec/Christoffersen (NB04), Diebold-Mariano (NB07, NB10).

For model comparison across M models: use White's Reality Check or SPA test (see NB10 section).

#### 2.4.2 Formal Hypothesis Test Reference Table

| Test | H₀ | H₁ | Statistic | Distribution | Used In |
|------|----|----|-----------|-------------|---------|
| ADF | Unit root (non-stationary) | Stationary | τ (DF regression) | Dickey-Fuller tables | NB01 |
| KPSS | Stationary | Unit root | LM = Σ S²_t / (T² · σ̂²) | KPSS tables | NB01 |
| Jarque-Bera | Normality (S=0, κ=0) | Non-normal | JB = (T/6)·(S² + κ²/4) where κ = K-3 (excess kurtosis) | χ²(2) | NB01 |
| Ljung-Box | No autocorrelation | Autocorrelation | Q = T(T+2) · Σ_{k=1}^m ρ̂²_k/(T-k) | χ²(m) | NB03 |
| ARCH-LM | No ARCH effects | ARCH effects | TR² from auxiliary regression | χ²(q) | NB03 |
| Granger | x does not Granger-cause y | x Granger-causes y | F = ((RSS_r - RSS_u)/p) / (RSS_u/(T-2p-1)) | F(p, T-2p-1) | NB02, NB06 |
| Kupiec POF | VaR violation rate = (1-α) | Violation rate ≠ (1-α) | LR_uc (see NB04 formulas) | χ²(1) | NB04 |
| Christoffersen CC | Violations independent + correct rate | Clustering or wrong rate | LR_cc = LR_uc + LR_ind | χ²(2) | NB04 |
| Diebold-Mariano | Equal predictive accuracy | Unequal accuracy | DM = d̄/√(V̂(d̄)) | N(0,1) | NB07, NB10 |
| Bai-Perron | No structural breaks | ≥1 structural break | sup-F statistic | Bai-Perron tables | NB02 |
| Engle-Granger | No cointegration | Cointegration | ADF on OLS residuals | Engle-Granger tables | NB02, NB08 |
| Johansen Trace | Rank(Π) ≤ r | Rank(Π) > r | -T · Σ_{i=r+1}^n ln(1-λ̂_i) | Johansen tables | NB02 |

**Convention**: Use α=0.05 throughout. ADF: constant+trend for levels, constant-only for returns. Always run KPSS alongside ADF (confirmatory approach — ADF tests H₀: unit root; KPSS tests H₀: stationarity).

#### 2.4.3 Bootstrap Inference

- **Stationary Block Bootstrap** (Politis & Romano, 1994) for return-based statistics. Block lengths drawn from Geometric(1/q) where q = average block length.
- **Automatic block length selection**: Politis & White (2004) optimal q via flat-top lag window on autocorrelation.
- **Residual Bootstrap for GARCH**: Pascual, Romo & Ruiz (2006) — resample standardized residuals ẑ_t, recursively reconstruct σ²_t and ε_t. Preserves GARCH dynamics.
- **Circular Block Bootstrap for DCC**: Block bootstrap on standardized residuals from Step 1, re-estimate DCC parameters. Preserves cross-sectional dependence.
- **Sample sizes**: B=10,000 for 95% CIs, B=50,000 for 99% CIs.
- **Studentized bootstrap** for Sharpe ratio comparison (Ledoit & Wolf, 2008) — accounts for serial correlation and non-normality of returns.

---

## 3 · NOTEBOOK PIPELINE — 12 NOTEBOOKS

### NOTEBOOK 01: Data Ingestion & Exploratory Analysis
**File**: `NB01_data_ingestion_eda.ipynb`

**Objectives**:
- Download all 20 tickers + benchmarks + macro data via `OpenBB` and FRED API
- Handle XYZ/SQ ticker transition: download SQ (pre-2025) + XYZ (post-2025), merge into unified series
- Compute all base features (returns, volatility estimators, momentum indicators)
- Produce summary statistics table: mean, std, skewness, kurtosis, Jarque-Bera p-value, ADF p-value per ticker
- Correlation heatmap (full period + rolling 252-day animation)
- Drawdown analysis: max drawdown, Calmar ratio (annualized return / abs(max drawdown)), time-to-recovery per ticker
- Distribution analysis: QQ plots, kernel density vs. normal overlay, heavy-tail diagnostics
- Volume regime identification: z-score volume spikes coinciding with major events (COVID crash, SVB, chip export bans, CrowdStrike outage)
- **ADR Premium Analysis** (optional enrichment for TSM, SAP):
  ```
  TSM premium = (P_ADR / 5) / (P_{2330.TW} · FX_{TWD/USD}) - 1
  SAP premium = P_ADR / (P_{SAP.DE} · FX_{EUR/USD}) - 1
  Download: 2330.TW, SAP.DE, TWDUSD=X, EURUSD=X via OpenBB
  Persistent non-zero ADR premium indicates capital flow restrictions or liquidity preference.
  ```
- Save cleaned master DataFrame as `master_data.parquet`

**Key Visualizations**:
- 20-stock cumulative return plot (log scale)
- Rolling 63-day correlation matrix heatmap (animated or subplot grid)
- Sector-grouped box plots of annualized volatility
- Drawdown waterfall charts for top-5 most volatile names

---

### NOTEBOOK 02: Macro Regime & Geopolitical Context
**File**: `NB02_macro_regime_context.ipynb`

**Objectives**:
- Map the 10-year window onto macro regimes: QE era (2016-2018), rate hike cycle (2018), COVID shock (2020), zero-rate recovery (2020-2021), inflation/rate shock (2022-2023), AI boom (2023-2025), tariff/geopolitical volatility (2025-2026)
- Construct composite macro factor: yield_curve_slope + VIX_level + DXY_change + CPI_surprise → PCA reduction to 2-3 macro factors
- Event study methodology: measure abnormal returns around key events:
  - COVID crash (Feb-Mar 2020)
  - Fed pivot to zero rates (Mar 2020)
  - Inflation shock / rate hikes begin (Jan 2022)
  - ChatGPT launch / AI rally (Nov 2022)
  - SVB collapse (Mar 2023)
  - US chip export controls to China (Oct 2022, Oct 2023 updates)
  - CrowdStrike global outage (Jul 2024)
  - DeepSeek shock / tariff escalation (Jan-Feb 2025)
- Cross-asset correlation analysis: tech stocks vs. TLT, GLD, DXY under each regime
- Output: `macro_regimes.parquet` with labeled regime column

**Analytical Depth**:
- Granger causality tests: Do macro factors (VIX, yields, DXY) Granger-cause tech sector returns?
  ```
  Granger test: r_t = Σ_{j=1}^p α_j·r_{t-j} + Σ_{j=1}^p β_j·x_{t-j} + ε_t
  H₀: β_1 = ... = β_p = 0  (x does not Granger-cause r)
  F-stat ~ F(p, T-2p-1). Select lag p via BIC. Report with Newey-West correction.
  ```
- Structural break detection: Bai-Perron or CUSUM tests on rolling betas
- Regime-conditional statistics: separate mean/vol/skew/kurtosis tables per macro regime

**Factor Model Framework** — Multi-factor return decomposition:
```
Fama-French 3-Factor + Carhart Momentum:
  r^e_{i,t} = α_i + β_{i,mkt}·MKT_t + β_{i,smb}·SMB_t + β_{i,hml}·HML_t + β_{i,mom}·MOM_t + ε_{i,t}

Data source: Kenneth French Data Library via pandas_datareader
  MKT = market excess return (SPY - Rf)
  SMB = small minus big (size factor)
  HML = high minus low (value factor)
  MOM = winners minus losers (momentum, 12-1 month)

Expected results for tech universe:
  - High β_mkt (1.0–1.8 for most names, >2.0 for NVDA/PLTR)
  - Negative β_hml (growth stocks load negatively on value factor)
  - Positive β_mom during momentum regimes, reversal during crashes
  - α_i captures stock-specific return unexplained by factors
```

**Variance Risk Premium (VRP)** — implied vs. realized vol spread:
```
VRP_t = IV²_t - RV²_t    (positive on average — volatility risk is priced)

For the tech sector: VRP_t = VIX²_t - RV²(SPY)_t,21d
Per-ticker (if options data available): VRP_{i,t} = IV²_{i,t} - σ²_{YZ,i,t}

VRP is a well-documented predictor of future returns (Bollerslev, Tauchen & Zhou, 2009):
  - High VRP → expected positive excess returns (compensation for bearing vol risk)
  - VRP collapses or inverts during crashes (realized vol exceeds implied)
```

**Earnings Seasonality & Post-Earnings Announcement Drift (PEAD)**:
```
Pre-earnings drift:  CAR[-5, -1] — abnormal returns in the 5 days before earnings
PEAD:                CAR[+1, +20] — drift continues for ~20 days post-announcement
SUE (Standardized Unexpected Earnings):
  SUE_i = (EPS_actual - EPS_expected) / σ(EPS_surprise)
  Sort into quintiles → Q5-Q1 spread = PEAD magnitude
Earnings vol crush: IV_{t-1} - IV_{t+1} (implied vol drops after uncertainty resolves)
Cross-ticker transmission: TSM earnings → NVDA abnormal returns (supply chain signal)
```
Data source: `obb.equity.calendar.earnings()` or seasonal random walk SUE proxy.

**Event Study Methodology** (formal):
```
Abnormal return: AR_{i,t} = r_{i,t} - E[r_{i,t}]
  where E[r_i,t] = α̂_i + β̂_i · r_{m,t}  (market model, estimated over [-250, -11] window)

Cumulative abnormal return: CAR_{i}[t1,t2] = Σ_{t=t1}^{t2} AR_{i,t}
Test statistic: t_CAR = CAR / (σ_AR · sqrt(t2-t1+1))

Event windows: [-1, +1] for immediate impact, [-5, +10] for full adjustment
Cross-sectional test: t_cross = mean(CAR_i) / (std(CAR_i) / sqrt(N))
```

---

### NOTEBOOK 03: Volatility Modeling — Classical Econometrics
**File**: `NB03_volatility_econometrics.ipynb`

**Objectives**:
- Fit per-ticker volatility models (all 20 stocks):
  - **GARCH(1,1)** — baseline symmetric volatility
  - **GJR-GARCH(1,1)** — asymmetric leverage effect (bad news → more vol than good news)
  - **EGARCH(1,1)** — exponential specification, no positivity constraints
  - **FIGARCH** — long-memory volatility (if fractional integration detected). **Note**: Skip FIGARCH for tickers with < 1500 observations (PLTR, DDOG, CRWD) and document the reason.
- Model selection via AIC/BIC comparison table across all 20 tickers
- Standardized residual diagnostics: Ljung-Box on squared residuals, ARCH-LM test, QQ plot of standardized residuals
- Volatility term structure: fit models at 5d, 21d, 63d horizons; compare implied vs. realized vol
- **Innovation distribution**: Compare Normal, Student-t, Skewed-t, GED for each ticker; select via log-likelihood
- Extract conditional volatility series → save as features for ML notebooks
- **Options-Implied Analytics** (for NVDA, AAPL, MSFT, META, AMZN — 5 most liquid):
  - **IV Term Structure**: Interpolate ATM IV to fixed 30d/60d/90d tenors from nearest 3 monthly expirations via OpenBB option chains
  - **IV Skew**: 25Δ put IV minus ATM IV. Positive skew = crash premium. Compare across regimes.
  - **Variance Risk Premium (VRP)**: VRP_{i,t} = IV²_{i,30d} - σ²_{YZ,i,21d} (annualized variance units). Positive on average — compensation for bearing volatility risk.
  - **Implied Correlation with XLK**: ρ_impl = (σ²_XLK - Σw²_i·σ²_i) / (2·Σ_{i<j} w_i·w_j·σ_i·σ_j)
  - **Put-Call Ratio**: Open interest weighted. High PCR = bearish sentiment.
  - Data pipeline: weekly snapshots stored in `data/features/options_implied.parquet`. Interpolation via cubic spline on total variance (IV² × τ) to avoid calendar arbitrage.
- Output: `garch_parameters.csv`, `conditional_vol_series.parquet`

**Mathematical Specifications**:

All models share the mean equation: r_t = μ + ε_t, where ε_t = σ_t · z_t, z_t ~ D(0,1).

**GARCH(1,1)** — Bollerslev (1986):
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Constraints: ω > 0, α ≥ 0, β ≥ 0
Stationarity: α + β < 1  (ensures finite unconditional variance)
Unconditional variance: σ² = ω / (1 - α - β)
Half-life of vol shock: ln(2) / ln(α + β)  days
Persistence: α + β  (closer to 1 → slower mean-reversion; typical range 0.95–0.99)
```

**GJR-GARCH(1,1)** — Glosten, Jagannathan & Runkle (1993):
```
σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}

where I_{t-1} = 1 if ε_{t-1} < 0 (bad news indicator)
Leverage effect: γ > 0  means negative returns increase vol more than positive returns
Impact of bad news: α + γ    Impact of good news: α
News Impact Ratio: (α + γ) / α    (typically 1.5–3x for tech stocks)
Stationarity: α + β + γ/2 < 1
Unconditional variance: σ² = ω / (1 - α - β - γ/2)
```

**EGARCH(1,1)** — Nelson (1991):
```
ln(σ²_t) = ω + α·|z_{t-1}| + γ·z_{t-1} + β·ln(σ²_{t-1})

where z_{t-1} = ε_{t-1}/σ_{t-1}  (standardized residuals)
No positivity constraints needed (log specification ensures σ² > 0)
Leverage: γ < 0  captures asymmetric response
Stationarity: |β| < 1
```

**Mandatory Long-Memory Detection Before FIGARCH**:
```
1. R/S Analysis:
   H = lim_{n→∞} ln(R_n/S_n) / ln(n)
   where R_n = range of cumulative deviations, S_n = standard deviation
   H > 0.5 → long memory (persistent), H = 0.5 → random walk, H < 0.5 → mean-reverting

2. GPH Log-Periodogram Regression (Geweke & Porter-Hudak, 1983):
   ln(I(ω_j)) = c - d · ln(4·sin²(ω_j/2)) + ε_j
   Estimate d on |r_t| and r²_t (volatility proxies)
   d > 0 → long memory;  d ∈ (0, 0.5) → stationary long memory

3. Decision rule:
   If H > 0.5 AND d > 0 with 95% CI excluding 0 → proceed to FIGARCH with d as initialization
   Otherwise: FIGARCH is not justified; skip and document the reason.
```
Report H and d per ticker with bootstrap 95% CIs. Add to `garch_parameters.csv`.

**FIGARCH(1,d,1)** — Baillie, Bollerslev & Mikkelsen (1996):
```
σ²_t = ω + [1 - (1-βL)^{-1}(1-φL)(1-L)^d] · ε²_t

where d ∈ (0, 1) is the fractional differencing parameter
d = 0 → standard GARCH;  d = 1 → IGARCH (unit root in variance)
0 < d < 0.5 → covariance-stationary long memory in volatility
Hyperbolic decay of volatility autocorrelations: ρ(k) ~ k^{2d-1}
```
**Skip FIGARCH** for tickers with < 1500 observations (PLTR, DDOG, CRWD). Verify long-memory prerequisite (above) before fitting.

**Innovation Distribution Selection**: Compare via log-likelihood and QQ plots:
- Normal: z ~ N(0,1) — baseline, underestimates tails
- Student-t: z ~ t(ν), ν > 2 — fat tails, symmetric. Typical ν ∈ [4, 8] for tech stocks
- Skewed-t: z ~ skewt(ν, λ) — fat tails + asymmetry. λ < 0 indicates left skew (crash risk)
- GED: z ~ GED(ν), ν < 2 → fatter than normal. ν = 2 recovers normal, ν = 1 is Laplace

**Model Selection**: AIC = -2·ln(L) + 2k; BIC = -2·ln(L) + k·ln(T). Prefer BIC for parsimony (penalizes complexity more with large T). If AIC and BIC disagree, report both and select per BIC.

**Realized Volatility Signature Plot** (for NVDA, AAPL, MSFT, META, AMZN):
```
RV_Δ = Σ r²_{j,Δ}  for Δ ∈ {5min, 15min, 30min, 1hr, daily}
Using 60-day intraday data at 5-min frequency (OpenBB provider limit for 5-min bars).
Plot: RV vs ln(sampling_frequency). Stable region validates daily OHLC estimators.
If RV increases at high frequencies → microstructure noise dominates → daily estimators preferred.
```

**Implementation**:
```python
from arch import arch_model
# Example: GJR-GARCH with skewed-t innovations
# vol='GARCH' + o=1 produces the GJR asymmetric term
model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='skewt')
result = model.fit(disp='off', options={'maxiter': 1000})
```

---

### NOTEBOOK 04: Tail Risk — VaR, CVaR, and Extreme Value Theory
**File**: `NB04_tail_risk_var_cvar.ipynb`

**Objectives**:
- Compute Value-at-Risk (95% and 99%) via 5 methods per ticker:
  1. **Historical Simulation** — non-parametric empirical quantile
  2. **Parametric Gaussian** — mean-variance VaR (benchmark, expected to underperform)
  3. **Cornish-Fisher Expansion** — adjusted for skewness and excess kurtosis. **Note**: Must use excess kurtosis (kurtosis - 3), not raw kurtosis. Implement monotonicity guard for extreme parameter values where the polynomial approximation can become non-monotonic.
  4. **GARCH-based VaR** — conditional VaR from NB03's best-fit GARCH model
  5. **CAViaR (Conditional Autoregressive VaR)** — Engle & Manganelli (2004) quantile regression
- **CVaR / Expected Shortfall**: For each VaR method, compute the expected loss beyond VaR threshold
- **Portfolio-Level Scenario Matrix**: Generate a (T × 20) matrix of historical return scenarios for use in NB11's Mean-CVaR optimization. Portfolio CVaR is NOT linearly aggregable from per-ticker CVaR — it must be computed from joint portfolio return scenarios.
- **Backtesting**:
  - Kupiec's POF test (unconditional coverage)
  - Christoffersen's conditional coverage test (independence of violations)
  - Traffic-light system (Basel): Green / Yellow / Red zone classification
- **Extreme Value Theory (EVT)**:
  - Peaks-Over-Threshold (POT) with Generalized Pareto Distribution (GPD)
  - Estimate tail index (xi) per ticker → classify fat-tailedness
  - EVT-based VaR/CVaR at 99.5% and 99.9% for stress-testing
- **Copula-based joint tail risk**:
  - Fit Clayton (lower-tail dependence) and Gumbel (upper-tail) copulas to sector pairs
  - Estimate joint crash probability: P(NVDA < VaR_99 AND AMD < VaR_99)
**Mathematical Definitions**:

**Value-at-Risk** at confidence level α (e.g., α = 0.99):
```
VaR_α = -inf{ x : P(r_t ≤ x) ≥ 1-α } = -F^{-1}(1-α)

Interpretation: The maximum loss not exceeded with probability α over a 1-day horizon.
VaR_99 = -quantile(returns, 0.01)
```

**CVaR / Expected Shortfall** (coherent risk measure, unlike VaR):
```
CVaR_α = -E[r_t | r_t ≤ -VaR_α] = -(1/(1-α)) · ∫₀^{1-α} F^{-1}(p) dp

Equivalently for discrete scenarios: CVaR_α = -(1/|S|) · Σ_{t∈S} r_t
  where S = {t : r_t ≤ -VaR_α}
```
CVaR satisfies subadditivity: CVaR(A+B) ≤ CVaR(A) + CVaR(B). VaR does not.

**Parametric Gaussian VaR**:
```
VaR_α = -(μ + z_α · σ)    where z_{0.99} = -2.3263, z_{0.95} = -1.6449
```

**Cornish-Fisher Expansion** — adjusts Gaussian quantile for skewness (S) and excess kurtosis (K):
```
z_CF = z_α + (1/6)·(z_α² - 1)·S + (1/24)·(z_α³ - 3z_α)·K - (1/36)·(2z_α³ - 5z_α)·S²

VaR_CF = -(μ + z_CF · σ)

CRITICAL: K = kurtosis - 3 (EXCESS kurtosis, not raw).
         S = skewness of return distribution.
```
**Monotonicity guard**: For |S| > 2 or K > 10, the polynomial can become non-monotonic (z_CF may not be monotone in α). In such cases, fall back to historical simulation or EVT.

**GARCH-based VaR** (conditional, from NB03's best model):
```
VaR_{α,t} = -(μ_t + q_α(D) · σ_t)

where σ_t = conditional volatility from GARCH
      q_α(D) = α-quantile of the fitted innovation distribution D (e.g., Student-t)
```

**CAViaR** — Engle & Manganelli (2004), Symmetric Absolute Value specification:
```
q_t(α) = β₁ + β₂ · q_{t-1}(α) + β₃ · |r_{t-1}|

Estimated by minimizing: (1/T) · Σ_{t=1}^T ρ_α(r_t - q_t(α))
  where ρ_α(u) = u·(α - I(u<0))  is the check/tick loss function
```

**Kupiec's POF Test** (unconditional coverage):
```
LR_uc = -2·ln[(1-α)^{T-N} · α^N] + 2·ln[(1-N/T)^{T-N} · (N/T)^N]
LR_uc ~ χ²(1) under H₀: violation rate = (1-α)

where N = number of VaR violations, T = total observations
```

**Christoffersen's Conditional Coverage Test**:
```
LR_cc = LR_uc + LR_ind    where LR_ind tests independence of violations
LR_cc ~ χ²(2)

LR_ind uses a first-order Markov transition matrix of violation/no-violation sequences.
```

**EVT — Generalized Pareto Distribution (GPD)** for Peaks-Over-Threshold:
```
For exceedances y = x - u (where u = threshold, typically 90th-95th percentile of losses):
G_{ξ,β}(y) = 1 - (1 + ξy/β)^{-1/ξ}    if ξ ≠ 0

Tail index ξ (shape parameter):
  ξ > 0  → Fréchet (heavy tails, e.g., tech stock returns — typical ξ ∈ [0.1, 0.4])
  ξ = 0  → Gumbel (exponential tails)
  ξ < 0  → Weibull (bounded tails — rare for financial returns)

EVT-based VaR at extreme levels (99.5%, 99.9%):
  VaR_α = u + (β/ξ) · [(T/N_u · (1-α))^{-ξ} - 1]

EVT-based CVaR:
  CVaR_α = VaR_α / (1-ξ) + (β - ξ·u) / (1-ξ)
  Valid only for ξ < 1 (always satisfied for financial returns)
```

**Threshold selection**: Use mean excess plot (should be linear above u) and parameter stability plot (ξ̂ and β̂ stable across u). Formal test: Danielsson et al. (2001) double-bootstrap.

**Systemic Risk Measures** (cross-asset tail dependence):

**CoVaR** — Adrian & Brunnermeier (2016):
```
CoVaR^α_{sys|i} = VaR of system/portfolio conditional on asset i being at its VaR
Estimated via quantile regression: q_α(r_sys | r_i, M_t) = α_0 + α_1·r_i + γ'M_t
  where M_t = state variables (VIX, yield spread, credit spread)
ΔCoVaR_i = CoVaR(r_i = VaR_i) - CoVaR(r_i = median_i)
```
Compute for top-5 highest-beta names (NVDA, PLTR, MU, CRWD, AMD).

**MES (Marginal Expected Shortfall)** — Acharya et al. (2017):
```
MES_i = E[r_i | r_portfolio ≤ VaR_α(portfolio)]
Key property: Portfolio CVaR = Σ_i w_i · MES_i  (exact decomposition)
```
Compute for all 20 tickers under equal-weight and optimal-weight portfolios.

**Absorption Ratio (AR)** — Kritzman et al. (2011):
```
AR_t = Σ_{j=1}^k λ_j / Σ_{j=1}^N λ_j
  where λ_j = eigenvalues of rolling correlation matrix (252-day window)
  k = N/5 = 4 (top 4 eigenvalues out of 20)

High AR → tightly coupled markets → systemic fragility
Sharp AR increases precede market turbulence (leading indicator)
Typical range: 0.5-0.9 for equity markets
```

- Output: `var_cvar_table.csv`, `evt_parameters.csv`, `backtest_results.csv`, `return_scenarios.parquet`, `systemic_risk_measures.csv`

---

### NOTEBOOK 05: Regime Detection — HMM & Markov Switching
**File**: `NB05_regime_detection.ipynb`

**Dependencies**: NB01 (core HMM fitting), NB02 (optional: macro factor analysis), NB04 (optional: regime-conditional VaR/CVaR)

**Objectives**:
- **Hidden Markov Model (HMM)** — `hmmlearn` GaussianHMM:
  - Fit 2-state (bull/bear) and 3-state (bull/neutral/bear) models on XLK sector returns
  - Fit per-ticker HMMs on the 5 highest-vol names (NVDA, PLTR, MU, CRWD, XYZ)
  - Extract: transition matrix, stationary distribution, state-conditional means/variances
  - Viterbi decoding → most likely regime path
  - Model selection: 2 vs. 3 vs. 4 states via BIC
- **Markov-Switching GARCH (MS-GARCH)**:
  - Regime-dependent volatility dynamics
  - Smoothed regime probabilities
- **Regime features for downstream ML**:
  - `regime_state` (categorical: 0, 1, 2)
  - `regime_probability` (continuous: P(bull), P(bear))
  - `regime_duration` (days in current regime)
  - `regime_transition_signal` (binary: regime changed in last 5 days)
- **Regime-conditional analytics**:
  - Separate correlation matrices per regime
  - Regime-conditional VaR/CVaR (from NB04 methods, filtered by regime) — requires NB04 output
  - Average holding period per regime
  - Which macro factors predict regime transitions? (logistic regression on macro features) — requires NB02 output
**Mathematical Formulation — Gaussian HMM**:

```
Model parameters: λ = (A, B, π)
  A = [a_{ij}]  K×K transition matrix:  a_{ij} = P(S_t = j | S_{t-1} = i)
  B = {(μ_k, σ²_k)} k=1..K  emission parameters:  r_t | S_t=k ~ N(μ_k, σ²_k)
  π = [π_k]    initial state distribution:  π_k = P(S_1 = k)

State inference:
  Forward variable: α_t(k) = P(r_1,...,r_t, S_t=k | λ)
  Backward variable: β_t(k) = P(r_{t+1},...,r_T | S_t=k, λ)
  Smoothed probability: γ_t(k) = P(S_t=k | r_1,...,r_T, λ) = α_t(k)·β_t(k) / P(r|λ)

Viterbi decoding: S*_1,...,S*_T = argmax P(S_1,...,S_T | r_1,...,r_T, λ)
  (dynamic programming on log-probabilities to find most likely state sequence)

EM (Baum-Welch) iterates until convergence:
  E-step: compute γ_t(k) and ξ_t(i,j) = P(S_t=i, S_{t+1}=j | r, λ)
  M-step: â_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
          μ̂_k = Σ_t γ_t(k)·r_t / Σ_t γ_t(k)
          σ̂²_k = Σ_t γ_t(k)·(r_t - μ̂_k)² / Σ_t γ_t(k)
```

**Stationary distribution**: π_∞ satisfying π_∞ = π_∞ · A (left eigenvector of A with eigenvalue 1). Gives long-run fraction of time in each regime.

**Expected regime duration**: E[duration in state k] = 1/(1 - a_{kk}) days.

**Model selection**: BIC = -2·ln(L) + p·ln(T), where p = K² + 2K - 1 free parameters for K-state Gaussian HMM. Run each K with 20+ random initializations (EM is sensitive to initialization) and select K minimizing BIC.

**Practical note**: Run `hmmlearn.GaussianHMM` with `n_init=25, n_iter=500` and sort states post-hoc by μ_k (ascending) so that state 0 = bear (lowest mean), state K-1 = bull (highest mean). This ensures consistent labeling across runs.

- Output: `regime_labels.parquet`, `hmm_model_params.pkl`, `transition_matrices.csv`

**Visualization**:
- Full 10-year price chart with regime-colored background shading
- Regime probability time series (smoothed)
- Transition probability heatmap per model

---

### NOTEBOOK 06: NLP Sentiment — FinBERT
**File**: `NB06_nlp_sentiment_finbert.ipynb`

**Objectives**:
- Collect financial news headlines per ticker (source: `newspaper3k`, `feedparser` on RSS feeds, or pre-built dataset)
- Run FinBERT (`ProsusAI/finbert` from HuggingFace) inference on headlines
- Compute per-ticker daily sentiment features:
  - `sentiment_mean` — average FinBERT score (positive/negative/neutral probabilities)
  - `sentiment_std` — intraday sentiment dispersion (disagreement proxy)
  - `sentiment_volume` — number of articles (attention proxy)
  - `sentiment_momentum` — 5-day change in sentiment
- **CRITICAL: Lag enforcement**: All sentiment features must be lagged by t-1 (use yesterday's sentiment to predict today) when saved for ML consumption. Same-day sentiment contains contemporaneous information (a 2 PM headline affects both the sentiment score and the close-to-close return, creating leakage). The output file `sentiment_features.parquet` must have features already shifted by one business day.
- Validate: Granger-causality of lagged sentiment → next-day returns and volatility
- Correlation analysis: sentiment vs. VIX, sentiment vs. regime state
- Event-level analysis: sentiment spike detection around known events (earnings, product launches, regulatory actions)
- Output: `sentiment_features.parquet` (t-1 lagged)

**Note on Data Availability**: If live headline scraping is impractical, use a pre-built financial news dataset (e.g., Financial PhraseBank, Kaggle financial news datasets) or synthetic proxy from earnings call transcripts. Document any data limitations transparently.

---

### NOTEBOOK 07: Machine Learning — Volatility Forecasting
**File**: `NB07_ml_volatility_forecast.ipynb`

**Objectives**:
- **Target variable**: Realized volatility at 5-day and 21-day forward horizons (Yang-Zhang estimator)
- **Feature set**: Lagged realized vol, GARCH conditional vol, returns, momentum, volume, macro factors, regime state, sentiment scores (t-1 lagged from NB06)
- **Models** (all with walk-forward expanding-window evaluation — consistent with NB08):
  1. **Ridge / Lasso Regression** — linear baseline with regularization
  2. **Random Forest** — `sklearn.ensemble.RandomForestRegressor`, 500 trees
  3. **XGBoost** — `xgboost.XGBRegressor` with Optuna hyperparameter tuning
  4. **LightGBM** — `lightgbm.LGBMRegressor` for comparison
  5. **KNN Regression** — local non-parametric benchmark
- **Walk-Forward Validation** (consistent with NB08): Expanding window, retrain every 63 trading days (quarterly). Initial training window: first 70% of data. This ensures all predictions used in NB11's portfolio backtest are genuinely out-of-sample at each point in time.
- **Embargo/Purge Protocol** (Lopez de Prado, 2018, Ch.7): When predicting h-day forward targets, observations near the train/test boundary have labels overlapping test-period returns, creating subtle leakage:
  ```
  Purge: Remove {t : t + h > T_train_end} from training set
  Embargo: Additionally remove {t : T_train_end < t ≤ T_train_end + h}

  For h=5 (5-day forecast): purge last 5 training obs + embargo 5 days after train end
  For h=21 (21-day forecast): purge last 21 training obs + embargo 21 days
  ```
- **Feature Selection** (pre-modeling):
  1. VIF: Remove features with VIF > 10 (multicollinearity filter)
  2. Stability Selection (Meinshausen & Bühlmann, 2010): Lasso on 100 random 50% subsamples, retain features selected in >60% of runs
  3. Compare: full vs VIF-filtered vs stability-selected feature sets on OOS RMSE
- **Hyperparameter Optimization**: Optuna with `TimeSeriesSplit` (5-fold expanding window) as inner CV within each walk-forward training window
- **Evaluation Metrics**:
  - RMSE = sqrt((1/T)·Σ(σ̂_t - σ_t)²); MAE = (1/T)·Σ|σ̂_t - σ_t|; MAPE = (100/T)·Σ|σ̂_t - σ_t|/σ_t
  - Directional Accuracy = (1/T)·Σ I(sign(Δσ̂_t) = sign(Δσ_t)) — fraction of correct vol direction calls
  - **QLIKE loss** (preferred for vol forecasting — robust to scale):
    ```
    QLIKE = (1/T)·Σ[ ln(σ̂²_t) + σ²_t/σ̂²_t ]
    ```
    QLIKE is the only standard loss function that ranks forecasts consistently regardless of the volatility proxy used (Patton, 2011).
  - **Diebold-Mariano test** — H₀: equal predictive accuracy between models A and B:
    ```
    d_t = L(e^A_t) - L(e^B_t)    where L is the loss function
    DM = d̄ / sqrt(V̂(d̄))  ~ N(0,1) under H₀

    For h-step-ahead forecasts: use HAC variance V̂(d̄) with bandwidth h-1
    (Newey-West or Andrews automatic bandwidth) to handle overlapping windows.
    ```
  - **Mincer-Zarnowitz regression** — forecast efficiency (unbiasedness + optimality):
    ```
    σ²_realized,t = α + β·σ̂²_t + ε_t

    H₀: α=0, β=1 (jointly, via Wald F-test with Newey-West HAC SEs)
    R² of this regression = forecast informativeness
    If β > 1: forecasts underreact to information; β < 1: forecasts overreact
    ```
- **SHAP Explainability**:
  - `shap.TreeExplainer` for XGBoost and Random Forest
  - Global feature importance (beeswarm plot)
  - Per-ticker SHAP waterfall for highest-error predictions
  - Regime-conditional SHAP: Do feature importances shift between bull and bear?
- **Ensemble**: Stacking meta-learner (Ridge on base model predictions)
- Output: `vol_forecast_predictions.parquet`, `shap_values.pkl`, `model_comparison_table.csv`

**Pipeline Hygiene** (critical):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Scaling MUST be inside the pipeline to prevent leakage
# Note: SMOTE/imblearn NOT needed here — this is regression, not classification
# See NB08 for classification-specific pipeline with SMOTE
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(**best_params))
])
```

---

### NOTEBOOK 08: Machine Learning — Return Direction & Price Forecasting
**File**: `NB08_ml_return_price_forecast.ipynb`

**Objectives**:
- **Task A — Return Direction Classification** (binary: up/down next 5 days):
  - Models: Logistic Regression, Random Forest Classifier, XGBoost Classifier, SVM (RBF kernel)
  - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score
  - Threshold optimization: move decision boundary to maximize Sharpe ratio of a simple long/flat strategy
  - Calibration curve: Platt scaling or isotonic regression to ensure predicted probabilities are well-calibrated
- **Task B — Return Magnitude Regression** (continuous: 5-day forward return):
  - Same model suite as NB07 (Ridge, RF, XGBoost, LightGBM)
  - Feature importance comparison: vol-forecasting features vs. return-forecasting features
- **Task C — Multi-horizon forecasting**:
  - 1-day, 5-day, 21-day forward return predictions
  - Performance decay analysis: how quickly does forecast skill degrade with horizon?
- **Walk-forward validation** (consistent with NB07): Expanding window retraining every 63 trading days (quarterly). Initial training window: first 70% of data. Apply embargo/purge protocol (same as NB07) for all forecast horizons.
- **Task D — Cross-Sectional Momentum & Microstructure**:
  - Cross-sectional momentum signal: 12-month return minus most recent 1-month (12-1 month), ranked across 20 tickers
  - Information Coefficient (IC): Spearman rank correlation between signal and realized forward return per rebalance date. IC > 0.05 is economically meaningful.
  - Engle-Granger cointegration for 5 natural pairs: (NVDA, AMD), (TSM, AVGO), (META, GOOG), (PANW, CRWD), (CRM, NOW). Report: cointegration test p-value, β (hedge ratio), half-life = -ln(2)/ln(ρ) where ρ = AR(1) coefficient of spread.
  - Lead-lag cross-correlation: ρ_{i,j}(k) for k ∈ {-5,...,+5}. Identify if any ticker systematically leads others (e.g., NVDA leads AMD by 1 day).
  - Output: `cointegration_pairs.csv`, `lead_lag_network.csv`
- **Transaction cost analysis**: Apply 10 bps round-trip cost to any strategy signals
- Output: `return_predictions.parquet`, `classification_report.csv`, `strategy_backtest.csv`

**Pipeline Hygiene** (classification-specific):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# SMOTE + scaling inside pipeline to prevent leakage (classification only)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(**best_params))
])
```

---

### NOTEBOOK 09: Deep Learning — LSTM, GRU, Temporal Fusion Transformer
**File**: `NB09_deep_learning_forecasting.ipynb`

**Objectives**:
- **LSTM (Long Short-Term Memory)**:
  - Architecture: 2-layer stacked LSTM → Dropout(0.3) → Dense(1)
  - Input: 60-day lookback window of multi-feature sequences
  - Targets: 5-day forward realized vol AND 5-day forward return (dual-head variant)
  - Early stopping on validation loss (patience=15)
- **GRU (Gated Recurrent Unit)**:
  - Same architecture template but with GRU cells (fewer parameters, faster training)
  - Compare convergence speed and test-set performance vs. LSTM
- **Temporal Fusion Transformer (TFT)** — `pytorch-forecasting`:
  - Multi-horizon probabilistic forecasting (quantile outputs: 10th, 50th, 90th percentile)
  - Static covariates: sector, market_cap_bucket
  - Known future inputs: day_of_week, month, earnings_flag
  - Observed inputs: all engineered features from NB01-NB06
  - Attention weight analysis: which time steps and features does the model attend to?
- **Comparison Framework**:
  - Same walk-forward evaluation protocol as NB07-08 with embargo/purge (see NB07 section)
  - Benchmark DL models against best XGBoost from NB07
  - Computational cost analysis: training time, inference time, memory footprint
  - Quantile calibration: are the TFT's prediction intervals reliable?
- **Visualization**:
  - Prediction vs. actual time series with confidence bands (TFT quantiles)
  - Attention heatmaps (TFT)
  - Training loss curves with early stopping markers

**Implementation Notes**:
```python
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# LSTM example — note explicit dropout layer after LSTM output
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        # PyTorch LSTM dropout only applies BETWEEN layers, not after the last layer
        # So we add an explicit dropout before the fully connected layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Apply dropout to last time step output
        return self.fc(out)
```

---

### NOTEBOOK 10: Hybrid Model & Model Audit
**File**: `NB10_hybrid_model_audit.ipynb`

**Objectives**:
- **Hybrid Architecture**: Combine GARCH conditional vol (NB03) + XGBoost residual correction (NB07) + LSTM sequence features (NB09)
  - Stage 1: GARCH produces baseline vol forecast
  - Stage 2: XGBoost corrects GARCH residuals using ML features + regime + sentiment
  - Stage 3: LSTM provides sequence-aware adjustment
  - Final: Weighted ensemble (weights optimized on validation set via scipy.optimize)
- **Comprehensive Model Comparison Table**:
  - All models from NB03, NB04, NB07, NB08, NB09 + hybrid
  - Metrics: RMSE, MAE, MAPE, Directional Accuracy, Sharpe (if strategy-linked)
  - Statistical tests: Diebold-Mariano pairwise, Model Confidence Set (MCS) by Hansen, Lunde & Nason (2011), *Econometrica*
  - **White's Reality Check / SPA Test** — Hansen (2005): Tests H₀ that the best model does not outperform the benchmark, accounting for data-snooping across M models.
    ```
    T_SPA = max_m (d̄_m / σ̂_m)    where d̄_m = mean(L_benchmark - L_model_m)
    p-value via stationary block bootstrap (B=10,000).
    Reject H₀ → best model is genuinely superior (not a spurious discovery from testing many models).
    ```
    Apply BH-FDR correction to pairwise Diebold-Mariano p-values across all model pairs.
  - **Important**: All models must have been evaluated under the same walk-forward protocol for valid comparison
- **Robustness Audit**:
  - Stability across sub-periods: 2016-2019, 2020-2021, 2022-2024, 2024-2026
  - Performance in crisis vs. calm periods (regime-conditional evaluation)
  - Sensitivity to lookback window length (30, 60, 90, 120 days)
  - Feature ablation study: remove feature groups one at a time, measure RMSE degradation
- **Overfitting Diagnostics**:
  - Train vs. test metric gap analysis
  - Learning curves (performance vs. training set size)
  - Permutation importance (shuffle-and-check) vs. SHAP importance consistency
- Output: `final_model_comparison.csv`, `hybrid_weights.json`, `audit_report.md`

---

### NOTEBOOK 11: Portfolio Optimization
**File**: `NB11_portfolio_optimization.ipynb`

**Objectives**:
- **Expected Return Estimation** (3 methods compared):
  1. Historical mean returns (naive)
  2. CAPM-implied expected returns (using SPY beta from NB02)
  3. Black-Litterman with ML-derived views (from NB08 walk-forward return forecasts as views). **View uncertainty**: Omega_ii = RMSE_i^2 (squared forecast error as variance of each view, not 1/RMSE). This ensures Omega has correct units (variance) and properly scales view confidence.
- **Risk Estimation** (5 methods compared — see expanded list in Optimization Methods section below for methods 4-5):
  1. Sample covariance (shrunk via Ledoit-Wolf)
  2. DCC-GARCH dynamic conditional correlation — implemented via custom Python DCC estimation on `arch` univariate GARCH standardized residuals (see `src/dcc_garch.py`). The Python `arch` library supports only univariate GARCH; the DCC correlation dynamics must be estimated separately on standardized residuals using the Engle (2002) two-step procedure.
  3. Regime-conditional covariance (separate matrices per HMM state from NB05)
- **Optimization Methods**:
  1. **Mean-Variance (Markowitz)** — max Sharpe, min variance, target return
  2. **Mean-CVaR** — scenario-based portfolio CVaR optimization using the (T × 20) return scenario matrix from NB04's `return_scenarios.parquet`. Portfolio CVaR is computed from the empirical distribution of portfolio returns (w' * R_t for each scenario t), NOT from weighted individual CVaRs. Formulated as a linear program via `cvxpy`.
  3. **Black-Litterman** — prior = market-cap weights, posterior = blended with ML views (Omega = diag(RMSE^2))
  4. **Hierarchical Risk Parity (HRP)** — Lopez de Prado (2016); no need to invert covariance matrix; cluster-based allocation
  5. **Risk Budgeting** — equal risk contribution (ERC) portfolio
  6. **Worst-Case Mean-Variance** — Goldfarb & Iyengar (2003): robust to estimation error in Σ
     ```
     min_w  w'Σ̂w + δ·||w||₂²    s.t. w'μ ≥ r_target, constraints
     Equivalent to shrinking Σ̂ toward (Σ̂ + δI). δ calibrated via cross-validation.
     ```
  7. **Maximum Diversification** — Choueifaty & Coignard (2008):
     ```
     max_w  DR(w) = w'σ / sqrt(w'Σw)
     where σ = vector of asset volatilities, Σ = covariance matrix
     DR > 1 always; higher = more diversified. Equivalent to min-variance on correlation matrix.
     ```
  8. **Resampled Efficient Frontier** — Michaud (1998):
     ```
     For b = 1,...,B (B=1000):
       Draw Σ̂_b ~ Wishart(scale=Σ̂/T, df=T)  so E[Σ̂_b] = Σ̂
       Draw μ̂_b ~ N(μ̂, Σ̂/T)
       Solve Markowitz with (μ̂_b, Σ̂_b) → w_b
     Final weights: w̄ = (1/B)·Σ_b w_b
     Resampled weights are more stable OOS than single-point Markowitz.
     ```
  9. **CVaR-ERC (Tail Risk Budgeting)**: Equalize marginal CVaR contributions across assets, analogous to ERC but using CVaR instead of variance.
- **Risk Estimation** (expanded to 5 methods):
  1. Sample covariance (shrunk via Ledoit-Wolf)
  2. DCC-GARCH dynamic conditional correlation
  3. Regime-conditional covariance (per HMM state)
  4. PCA factor model: Σ = B·F·B' + D (k factors, idiosyncratic diagonal)
  5. Non-linear shrinkage (Ledoit & Wolf, 2020) — optimal shrinkage of eigenvalues
- **Constraints** (UCITS-inspired where applicable):
  - Max single-stock weight: 10%
  - Max sector concentration: 30% per constraint group (see Section 1 Sector Constraint Groups table)
  - Long-only (no short positions)
  - Turnover constraint (L1 formulation):
    ```
    Σ_i |w_{i,new} - w̃_{i,old}| ≤ τ    (τ = 0.20)
    where w̃_{i,old} = drifted weight = w_{i,old}·(1+r_i) / Σ_j w_{j,old}·(1+r_j)
    LP reformulation: introduce t_i ≥ 0, w_{i,new} - w̃_{i,old} ≤ t_i, w̃_{i,old} - w_{i,new} ≤ t_i, Σ_i t_i ≤ τ
    ```
- **Backtest Framework**:
  - Monthly rebalancing, using only walk-forward out-of-sample ML predictions at each rebalance date
  - **Backtest period**: Determined by when walk-forward predictions become available. With initial training on first 70% of data (~2016-03 to ~2023-01), the first out-of-sample predictions start ~2023-01. Portfolio backtest runs from the first available prediction date through 2026-03.
  - **Transaction Cost Model** — Almgren-Chriss spread + market impact (replaces flat 10bps):
    ```
    TC_i = Spread_cost_i + Impact_cost_i

    Spread_cost_i = (spread_i / 2) · |Δw_i| · AUM
    Impact_cost_i = η · σ_{i,daily} · (|Δw_i| · AUM) · sqrt(|Δw_i| · AUM / ADV_i)

    where η ≈ 0.1 (market impact coefficient), ADV_i = average daily volume ($)
    ```
    Per-ticker spread calibration (by liquidity tier):
    | Tier | Tickers | Half-Spread |
    |------|---------|-------------|
    | Mega-cap liquid | AAPL, MSFT, AMZN, META, GOOG, NVDA | 0.5-1 bps |
    | Large-cap liquid | AVGO, TSM, AMD, CRM, MU | 1-3 bps |
    | Mid-cap / wider | PANW, CRWD, DDOG, PLTR, NOW, ANET, SAP, XYZ, SNPS | 3-8 bps |
    Parameterize AUM at $10M, $100M, $1B to show market impact scaling.
  - Performance metrics: Annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, plus:
    ```
    Omega Ratio (Keating & Shadwick, 2002): Ω(θ) = ∫_θ^∞ (1-F(r))dr / ∫_{-∞}^θ F(r)dr
      Captures entire return distribution, not just first two moments. Ω > 1 is desirable.

    Information Ratio: IR = (R̄_p - R̄_b) / TE
    Tracking Error: TE = σ(R_p - R_b) · sqrt(252)
    Ulcer Index (Martin, 1987): UI = sqrt((1/T)·Σ D²_t) where D_t = drawdown %
    Pain Index: mean(|D_t|) — average drawdown severity
    CDaR (Conditional Drawdown at Risk): E[DD | DD ≥ quantile(DD, α)]
      CDaR at 95%: average of worst 5% drawdown episodes
    ```
  - Benchmark: equal-weight, market-cap weight, XLK ETF
- **Regime-Adaptive Strategy**:
  - Switch allocation method based on HMM regime: HRP in bear markets (robust to estimation error), mean-variance in bull markets (exploit return forecasts)
  - Compare static vs. regime-adaptive performance
- **Regime-Conditional Efficient Frontiers**:
  - Overlay 3 frontiers (bull, bear, full-sample) on one plot
  - Bear frontier contracts inward (lower return per unit risk); min-variance composition shifts to defensive names (AAPL, MSFT, SAP)
  - Document how regime affects the investable set
- **Conditional Diversification Benefit (CDB)**:
  ```
  CDB = 1 - σ_portfolio / (w'σ)
  where σ = vector of individual asset volatilities
  CDB ∈ [0, 1]: 0 = perfect correlation (no benefit), 1 = zero portfolio vol
  Compute per regime: CDB_bear < CDB_bull (diversification fails when most needed)
  ```
- **Covariance Estimation Comparison**: For each of the 5 estimation methods, compute optimal weights and measure OOS Sharpe, weight stability (mean absolute weight change), and turnover. Report as comparison table.
- Output: `portfolio_weights_timeseries.parquet`, `backtest_performance.csv`, `efficient_frontier.png`

**Mathematical Programs**:

**1. Mean-Variance (Markowitz)** — Max Sharpe:
```
max_w  (w'μ - r_f) / sqrt(w'Σw)

s.t.   w'𝟏 = 1          (full investment)
       0 ≤ w_i ≤ 0.10   (long-only + 10% single-stock cap)
       Σ_{i∈G_k} w_i ≤ 0.30   for each constraint group G_k
```
Equivalent to: min w'Σw s.t. w'μ = r_target, constraints. Trace the efficient frontier by sweeping r_target from min-variance to max-return.

**2. Mean-CVaR** — Rockafellar-Uryasev (2000) LP reformulation:
```
min_{w,ζ,s}  ζ + (1/((1-α)·T)) · Σ_{t=1}^T s_t

s.t.   s_t ≥ -(w'R_t) - ζ     ∀t = 1,...,T    (R_t = scenario return vector)
       s_t ≥ 0                  ∀t
       w'μ ≥ r_target                            (minimum expected return)
       w'𝟏 = 1, 0 ≤ w_i ≤ 0.10, group constraints

where ζ = VaR (auxiliary), s_t = max(0, loss_t - ζ) = shortfall beyond VaR
      α = confidence level (e.g., 0.95)
      T = number of scenarios from return_scenarios.parquet
```
This is a linear program — scales efficiently to T=2500 scenarios × 20 assets via `cvxpy`.

**3. Black-Litterman** — posterior expected return:
```
Prior (equilibrium excess returns):  π = δ·Σ·w_mkt
  where δ = (E[R_m] - r_f) / σ²_m  (market risk aversion, typically δ ≈ 2.5)
        w_mkt = market-cap weights of the 20 stocks

Views: P·E[R] = Q + ε,  ε ~ N(0, Ω)
  P = K×N pick matrix (K views on N=20 assets)
  Q = K×1 view return vector (from NB08 walk-forward forecasts)
  Ω = diag(RMSE²_1, ..., RMSE²_K) — view uncertainty from out-of-sample forecast error

Posterior: E[R]_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} · [(τΣ)^{-1}π + P'Ω^{-1}Q]
           Σ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} + Σ

τ ∈ [0.01, 0.05] — scaling factor for uncertainty in equilibrium (set τ = 1/T as default)
```
**Key insight**: When RMSE_i is large (poor forecast), Ω_ii is large → that view is down-weighted toward the prior π. Self-correcting mechanism.

**4. Hierarchical Risk Parity (HRP)** — Lopez de Prado (2016):
```
Step 1: Compute distance matrix D_{ij} = sqrt(0.5·(1 - ρ_{ij}))
Step 2: Single-linkage hierarchical clustering on D
Step 3: Quasi-diagonalization — reorder assets so similar ones are adjacent
Step 4: Recursive bisection — split cluster, allocate inversely to variance:
  w_left = (1/σ²_left) / (1/σ²_left + 1/σ²_right)
  w_right = 1 - w_left
```
No covariance matrix inversion required → robust when Σ is near-singular or poorly estimated.

**5. Equal Risk Contribution (ERC)** — Maillard, Roncalli & Teïlétche (2010):
```
Find w such that: w_i · (Σw)_i / (w'Σw) = 1/N    ∀i

Equivalent to: min_w Σ_{i,j} (w_i·(Σw)_i - w_j·(Σw)_j)²
               s.t. w'𝟏 = 1, w ≥ 0
```
Marginal risk contribution of each asset is equalized.

**DCC-GARCH Two-Step Estimation** — Engle (2002):
```
Step 1: Fit univariate GARCH(1,1) to each asset → extract standardized residuals:
  z_{i,t} = ε_{i,t} / σ_{i,t}

Step 2: DCC correlation dynamics on z_t = (z_{1,t}, ..., z_{20,t})':
  Q_t = (1 - a - b)·Q̄ + a·(z_{t-1}·z'_{t-1}) + b·Q_{t-1}
  R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}

  Q̄ = sample correlation of standardized residuals (unconditional)
  a ≥ 0, b ≥ 0, a + b < 1 (mean-reversion in correlations)
  R_t = time-varying correlation matrix (guaranteed positive definite)

Full covariance: Σ_t = D_t · R_t · D_t
  where D_t = diag(σ_{1,t}, ..., σ_{20,t}) from Step 1
```

**Performance Metrics** (annualized where applicable):
```
Sharpe = (R̄_p - r_f) / σ_p · sqrt(252)
Sortino = (R̄_p - r_f) / σ_downside · sqrt(252)
  where σ_downside = sqrt((1/T)·Σ min(r_t - r_target, 0)²)
Max Drawdown = max_t (peak_t - trough_t) / peak_t
Calmar = annualized_return / |max_drawdown|
Turnover = (1/T_rebal) · Σ_t Σ_i |w_{i,t+1} - w_{i,t}·(1+r_{i,t})/Σ_j w_{j,t}·(1+r_{j,t})|
```

**Implementation**:
```python
from pypfopt import EfficientFrontier, BlackLittermanModel, HRPOpt, risk_models, expected_returns
import cvxpy as cp  # for scenario-based Mean-CVaR optimization
```

---

### NOTEBOOK 12: Stress Testing & Final Report
**File**: `NB12_stress_test_report.ipynb`

**Objectives**:
- **Historical Stress Scenarios**: Replay portfolio through:
  - COVID crash (Feb 19 - Mar 23, 2020): -34% SPY drawdown
  - Rate shock (Jan - Oct 2022): NASDAQ -33%
  - SVB contagion (Mar 2023): banking + tech liquidity crisis
  - Chip export ban escalation (Oct 2022, Oct 2023)
  - DeepSeek / tariff shock (Jan-Feb 2025)
- **Hypothetical Stress Scenarios**:
  - Taiwan Strait crisis: TSM -50%, NVDA -30%, AAPL -25% (supply chain disruption)
  - AI bubble burst: All AI-exposed names (NVDA, PLTR, CRWD, DDOG) drawdown -40%
  - Fed emergency rate hike (+100 bps): duration-sensitive names re-priced
  - Cybersecurity breach at a major platform: PANW/CRWD +15%, META/GOOG -10%
- **Monte Carlo Simulation** (10,000 paths, 252-day horizon):
  ```
  Simulation procedure (correlated GBM with GARCH vol):
  1. Calibrate: Use DCC-GARCH Σ_t from NB11 for current correlation structure
  2. Decompose: Σ_t = L·L' via Cholesky factorization
  3. For each path s = 1,...,10000:
       For each day d = 1,...,252:
         z_d ~ N(0, I_20)            (independent standard normals)
         ε_d = L · z_d               (correlated innovations)
         r_{i,d} = μ_i + σ_{i,d} · ε_{i,d}  (GARCH-filtered returns)
         Update σ²_{i,d+1} via GARCH recursion
       Portfolio return path: r^s_{p,d} = Σ_i w_i · r_{i,d}
  4. Compute: portfolio value paths, drawdown paths, terminal distribution

  Output statistics:
    - P(max_drawdown > 20%) = #{paths with DD > 20%} / 10000
    - Portfolio VaR_99(1Y) = 1st percentile of terminal wealth distribution
    - Portfolio CVaR_99(1Y) = mean of worst 1% of terminal outcomes
  ```
- **Sensitivity Analysis** (factor decomposition):
  ```
  Regression-based factor decomposition:
    r_{p,t} = α + β_mkt·r_{SPY,t} + β_sector·f_{sector,t} + β_macro·f_{macro,t} + ε_t

  Variance decomposition:
    Var(r_p) = β²_mkt·Var(r_SPY) + β²_sector·Var(f_sector) + ... + Var(ε)
    % systematic risk = 1 - Var(ε)/Var(r_p)
  ```
  - Portfolio P&L impact per +1σ move in: VIX, DXY, 10Y yield, oil price
  - Marginal risk contribution per asset: MRC_i = w_i · (Σw)_i / sqrt(w'Σw)
- **Advanced Risk Metrics for Stress Dashboard**:
  - Ulcer Index per stress scenario (sustained drawdown severity)
  - CDaR at 95% per stress scenario
  - Factor variance decomposition: market vs. sector vs. idiosyncratic risk contribution
    ```
    Regression: r_{p,t} = α + β_mkt·r_{SPY,t} + β_sector·f_{sector,t} + β_macro·f_{macro,t} + ε_t
    % market risk = β²_mkt·Var(r_SPY) / Var(r_p)
    % idiosyncratic = Var(ε) / Var(r_p)
    ```
  - CDB (Conditional Diversification Benefit) under each stress scenario
- **Final Summary Dashboard** (consolidated from all notebooks):
  - Executive summary: best model, optimal portfolio, key risk factors
  - Per-ticker risk card: VaR, CVaR, regime, GARCH params, SHAP top features
  - Model hierarchy: which method wins for vol forecasting vs. return forecasting?
  - Actionable insights: regime-dependent allocation recommendations
- Output: `stress_test_results.csv`, `monte_carlo_distribution.png`, `final_report.pdf`

---

## 4 · TECHNICAL STACK & ENVIRONMENT

### 4.1 Python Version & Core Dependencies

```
python                 >= 3.10
numpy                  >= 1.24
pandas                 >= 2.0
scipy                  >= 1.11
matplotlib             >= 3.7
seaborn                >= 0.12
plotly                 >= 5.15
```

### 4.2 Data & Finance

```
openbb                 >= 4.0
fredapi                >= 0.5
pandas-datareader      >= 0.10
```

### 4.3 Econometrics & Time Series

```
arch                   >= 6.0        # GARCH family (univariate only; DCC implemented custom)
statsmodels            >= 0.14       # ADF, KPSS, Granger, ARIMA
hmmlearn               >= 0.3        # Hidden Markov Models
copulas                >= 0.9        # Copula fitting
```

### 4.4 Machine Learning

```
scikit-learn           >= 1.3
xgboost                >= 2.0
lightgbm               >= 4.0
optuna                 >= 3.3        # Bayesian hyperparameter optimization
shap                   >= 0.42       # Explainability
imbalanced-learn       >= 0.11       # SMOTE for NB08 classification tasks
```

### 4.5 Deep Learning

```
torch                  >= 2.1
pytorch-forecasting    >= 1.0        # TFT implementation
pytorch-lightning      >= 2.1
```

### 4.6 NLP

```
transformers           >= 4.35       # FinBERT
```

### 4.7 Portfolio Optimization

```
pypfopt                >= 1.5        # Markowitz, BL, HRP
cvxpy                  >= 1.4        # Custom convex optimization (Mean-CVaR)
riskfolio-lib          >= 4.0        # Advanced portfolio optimization (optional)
```

### 4.8 Reporting

```
python-pptx            >= 0.6        # PowerPoint generation
openpyxl               >= 3.1        # Excel export
fpdf2                  >= 2.7        # PDF generation
```

---

## 5 · DIRECTORY STRUCTURE

```
tech-risk-ml-pipeline/
├── CLAUDE.md                              # THIS FILE — project blueprint
├── requirements.txt
├── environment.yml                        # Conda environment spec
├── data/
│   ├── raw/                               # Downloaded OHLCV, macro, sentiment
│   ├── processed/                         # Cleaned parquets from NB01
│   └── features/                          # Engineered features per notebook
├── notebooks/
│   ├── NB01_data_ingestion_eda.ipynb
│   ├── NB02_macro_regime_context.ipynb
│   ├── NB03_volatility_econometrics.ipynb
│   ├── NB04_tail_risk_var_cvar.ipynb
│   ├── NB05_regime_detection.ipynb
│   ├── NB06_nlp_sentiment_finbert.ipynb
│   ├── NB07_ml_volatility_forecast.ipynb
│   ├── NB08_ml_return_price_forecast.ipynb
│   ├── NB09_deep_learning_forecasting.ipynb
│   ├── NB10_hybrid_model_audit.ipynb
│   ├── NB11_portfolio_optimization.ipynb
│   └── NB12_stress_test_report.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                          # Centralized constants: tickers, dates, paths
│   ├── statistical_tests.py               # BH-FDR, Hurst exponent, SPA test, bootstrap
│   ├── systemic_risk.py                   # CoVaR, MES, Absorption Ratio
│   ├── data_loader.py                     # Centralized data download + caching
│   ├── feature_engineering.py             # All derived feature computations
│   ├── garch_utils.py                     # GARCH fitting, selection, extraction
│   ├── dcc_garch.py                       # Custom DCC-GARCH implementation (Engle 2002)
│   ├── risk_metrics.py                    # VaR, CVaR, EVT, copulas
│   ├── regime_utils.py                    # HMM fitting, decoding, labeling
│   ├── ml_pipeline.py                     # Sklearn/XGBoost pipeline builders
│   ├── dl_models.py                       # LSTM, GRU, TFT wrappers
│   ├── portfolio_optimizer.py             # All optimization methods
│   ├── backtest_engine.py                 # Walk-forward backtest logic
│   └── visualization.py                   # Standardized plotting functions
├── models/
│   ├── garch/                             # Saved GARCH model objects
│   ├── ml/                                # Saved sklearn/XGBoost models
│   ├── dl/                                # Saved PyTorch checkpoints
│   └── hmm/                               # Saved HMM model objects
├── outputs/
│   ├── tables/                            # CSV comparison tables
│   ├── figures/                           # Publication-quality plots
│   ├── presentations/                     # PPTX files
│   └── reports/                           # PDF audit reports
└── tests/
    ├── test_data_loader.py
    ├── test_risk_metrics.py
    ├── test_statistical_tests.py          # BH-FDR, Hurst, SPA, bootstrap tests
    ├── test_systemic_risk.py              # CoVaR, MES, AR tests
    └── test_pipeline_integrity.py         # No-lookahead-bias + embargo checks
```

---

## 6 · EXECUTION SEQUENCE & DEPENDENCY GRAPH

```
NB01 (Data + EDA)
  ├──→ NB02 (Macro Regimes)
  ├──→ NB03 (GARCH Volatility)
  │       └──→ NB04 (Tail Risk — uses GARCH conditional vol)
  ├──→ NB06 (Sentiment — independent data pipeline)
  │
  NB01 ──→ NB05 (Regime Detection — core HMM fitting needs only NB01)
  │           ├── [optional] NB02 output for macro-factor logistic regression
  │           └── [optional] NB04 output for regime-conditional VaR/CVaR
  │
  NB03 + NB04 + NB05 + NB06 ──→ NB07 (ML Vol Forecast — walk-forward)
  NB03 + NB04 + NB05 + NB06 ──→ NB08 (ML Return Forecast — walk-forward)
  │
  NB07 + NB08 ──→ NB09 (Deep Learning — compare against ML baselines)
  │               └──→ NB10 (Hybrid + Audit — combines all models)
  │
  NB04 + NB05 + NB10 ──→ NB11 (Portfolio Optimization — uses walk-forward predictions only)
  │                       └──→ NB12 (Stress Test + Final Report)
```

**Parallelizable**: NB02, NB03, NB05 (core), NB06 can run in parallel after NB01 completes.

**Critical path**: NB01 → NB03 → NB04 → NB07 → NB09 → NB10 → NB11 → NB12

---

## 7 · QUALITY STANDARDS & ANTI-HALLUCINATION PROTOCOL

### 7.1 Data Integrity
- **Every numerical claim must be traceable**: If a table says "NVDA annualized vol = 52.3%", the exact code cell producing that number must be identifiable.
- **No hardcoded results**: All statistics are computed dynamically. If data changes (e.g., new download date), all downstream numbers update automatically.
- **Reproducibility**: Set `random_state=42` for all stochastic operations. Pin all package versions in `requirements.txt`.

### 7.2 Modeling Honesty
- **Report failures**: If a model performs worse than a naive baseline (e.g., random walk for returns), document this clearly. Do not cherry-pick timeframes.
- **Confidence intervals**: All point forecasts must include uncertainty estimates (bootstrap CIs for ML, quantile outputs for TFT, analytical CIs for GARCH).
- **No lookahead bias**: Enforce via automated tests in `tests/test_pipeline_integrity.py`. The test suite should verify that no feature at time t uses information from t+1 or later. Sentiment features must be t-1 lagged.
- **Consistent evaluation protocol**: ALL ML/DL models (NB07, NB08, NB09) must use walk-forward expanding-window evaluation with quarterly retraining. This ensures valid cross-model comparison in NB10 and genuine out-of-sample predictions for NB11's portfolio backtest.
- **Honest train/test separation**: Walk-forward protocol ensures no model sees future data at any prediction point. The final ~17% of data (approximately 2024-09 → 2026-03) serves as the primary held-out evaluation window, but walk-forward predictions are available from ~2023-01 onward.

### 7.3 Presentation Standards
- All figures: high-DPI (300+), consistent color palette, labeled axes, titled, with source annotation
- All tables: formatted with consistent decimal places (4 for ratios, 2 for percentages, 0 for counts)
- Code cells: clear markdown headers, docstrings on all functions in `src/`, type hints

---

## 8 · MACRO CONTEXT FRAMEWORK (as of March 2026)

The following macro themes should inform interpretation of all results:

1. **AI Infrastructure Spending Cycle**: Hyperscaler capex (MSFT, AMZN, GOOG, META) exceeding $200B annually on AI data centers. Semiconductor demand (NVDA, AVGO, TSM, MU) directly correlated. Risk: capex pullback if AI monetization disappoints.

2. **US Monetary Policy**: Fed rate trajectory post-2022 tightening cycle. Impact on growth stock valuations (duration effect), cost of capital for high-P/E names (PLTR, CRWD, DDOG). Yield curve shape as recession signal.

3. **US-China Technology Decoupling**: Chip export controls affecting NVDA, AMD (restricted China sales), TSM (geopolitical foundry risk). Potential for further restrictions. Beneficiaries: domestic semi equipment (SNPS).

4. **Tariff Escalation (2025)**: Broad tariff implementation affecting supply chains, particularly hardware-dependent names (AAPL, TSM). Inflationary pressure on input costs. FX volatility (DXY, EUR/USD) impacting SAP, AAPL.

5. **Cybersecurity Structural Growth**: Post-breach regulation tightening. Zero-trust adoption acceleration. PANW, CRWD, DDOG benefit from mandatory compliance spending — relatively recession-resistant demand.

6. **AI Bubble Risk**: Concentration of returns in a small number of AI-linked names. DeepSeek-style disruption risk (open-source models undermining proprietary moats). Valuation compression if revenue growth decelerates.

---

## 9 · DELIVERABLES CHECKLIST

| Deliverable | Source Notebook | Format |
|---|---|---|
| Cleaned master dataset | NB01 | `.parquet` |
| EDA summary statistics | NB01 | `.csv` + figures |
| Macro regime timeline | NB02 | `.parquet` + annotated chart |
| GARCH parameter table (20 stocks x 4 models) | NB03 | `.csv` |
| VaR/CVaR comparison table with backtests | NB04 | `.csv` |
| Return scenario matrix (T x 20) for portfolio CVaR | NB04 | `.parquet` |
| Regime labels + transition matrices | NB05 | `.parquet` + `.csv` |
| Sentiment features (t-1 lagged) | NB06 | `.parquet` |
| ML vol forecast comparison | NB07 | `.csv` + SHAP plots |
| ML return forecast + strategy backtest | NB08 | `.csv` |
| DL forecast comparison + attention maps | NB09 | `.csv` + figures |
| Hybrid model + full audit report | NB10 | `.csv` + `.md` |
| Portfolio weights + backtest performance | NB11 | `.parquet` + `.csv` |
| Stress test results + Monte Carlo | NB12 | `.csv` + figures |
| Systemic risk measures (CoVaR, MES, AR) | NB04 | `.csv` |
| Cointegration pairs + lead-lag network | NB08 | `.csv` |
| Options-implied analytics | NB03 | `.parquet` |
| SPA test results | NB10 | `.csv` |
| Covariance estimation comparison | NB11 | `.csv` |
| **Final PowerPoint Presentation** | NB12 | `.pptx` (25-30 slides) |
| **Final PDF Audit Report** | NB10 + NB12 | `.pdf` (15-20 pages) |

---

## 10 · CITATION & METHODOLOGY REFERENCES

### 10.1 Core Methodology Papers

| Method | Key Paper | Journal/Venue | Implementation |
|---|---|---|---|
| GARCH(1,1) | Bollerslev, T. (1986) "Generalized Autoregressive Conditional Heteroskedasticity" | *J. Econometrics* 31(3), 307–327 | `arch` library |
| GJR-GARCH | Glosten, L., Jagannathan, R. & Runkle, D. (1993) "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks" | *J. Finance* 48(5), 1779–1801 | `arch` library |
| EGARCH | Nelson, D. (1991) "Conditional Heteroskedasticity in Asset Returns: A New Approach" | *Econometrica* 59(2), 347–370 | `arch` library |
| FIGARCH | Baillie, R., Bollerslev, T. & Mikkelsen, H. (1996) "Fractionally Integrated Generalized Autoregressive Conditional Heteroskedasticity" | *J. Econometrics* 74(1), 3–30 | `arch` library |
| DCC | Engle, R. (2002) "Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models" | *J. Business & Econ. Stats* 20(3), 339–350 | Custom Python |
| CAViaR | Engle, R. & Manganelli, S. (2004) "CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles" | *J. Business & Econ. Stats* 22(4), 367–381 | Custom implementation |
| EVT / GPD | Pickands, J. (1975) "Statistical Inference Using Extreme Order Statistics" | *Annals of Statistics* 3(1), 119–131 | `scipy.stats.genpareto` |
| Cornish-Fisher VaR | Cornish, E. & Fisher, R. (1937) "Moments and Cumulants in the Specification of Distributions" | *Revue de l'Institut Intern. de Stat.* 5(4), 307–320 | `scipy.stats` + monotonicity guard |
| VaR Backtesting | Kupiec, P. (1995) "Techniques for Verifying the Accuracy of Risk Measurement Models" | *J. Derivatives* 3(2), 73–84 | Custom implementation |
| Conditional Coverage | Christoffersen, P. (1998) "Evaluating Interval Forecasts" | *Intern. Econ. Review* 39(4), 841–862 | Custom implementation |
| CVaR Optimization | Rockafellar, R.T. & Uryasev, S. (2000) "Optimization of Conditional Value-at-Risk" | *J. Risk* 2(3), 21–42 | `cvxpy` |
| Hidden Markov Models | Rabiner, L. (1989) "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" | *Proc. IEEE* 77(2), 257–286 | `hmmlearn` |
| Black-Litterman | Black, F. & Litterman, R. (1992) "Global Portfolio Optimization" | *Financial Analysts Journal* 48(5), 28–43 | `pypfopt` |
| HRP | Lopez de Prado, M. (2016) "Building Diversified Portfolios that Outperform Out-of-Sample" | *J. Portfolio Management* 42(4), 59–69 | `pypfopt` |
| Risk Parity / ERC | Maillard, S., Roncalli, T. & Teïlétche, J. (2010) "The Properties of Equally Weighted Risk Contribution Portfolios" | *J. Portfolio Management* 36(4), 60–70 | `riskfolio-lib` |
| Ledoit-Wolf Shrinkage | Ledoit, O. & Wolf, M. (2004) "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices" | *J. Multivariate Analysis* 88(2), 365–411 | `sklearn.covariance` |
| Model Confidence Set | Hansen, P., Lunde, A. & Nason, J. (2011) "The Model Confidence Set" | *Econometrica* 79(2), 453–497 | Custom implementation |
| Diebold-Mariano | Diebold, F. & Mariano, R. (1995) "Comparing Predictive Accuracy" | *J. Business & Econ. Stats* 13(3), 253–263 | Custom implementation |
| Mincer-Zarnowitz | Mincer, J. & Zarnowitz, V. (1969) "The Evaluation of Economic Forecasts" | *Econ. Forecasts & Expectations*, NBER | Custom implementation |
| QLIKE Loss | Patton, A. (2011) "Volatility Forecast Comparison Using Imperfect Volatility Proxies" | *J. Econometrics* 160(1), 246–256 | Custom implementation |

### 10.2 Machine Learning & Deep Learning

| Method | Key Paper | Venue | Implementation |
|---|---|---|---|
| XGBoost | Chen, T. & Guestrin, C. (2016) "XGBoost: A Scalable Tree Boosting System" | *KDD 2016* | `xgboost` |
| LightGBM | Ke, G. et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" | *NeurIPS 2017* | `lightgbm` |
| SHAP | Lundberg, S. & Lee, S. (2017) "A Unified Approach to Interpreting Model Predictions" | *NeurIPS 2017* | `shap` |
| LSTM | Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory" | *Neural Computation* 9(8), 1735–1780 | `torch.nn.LSTM` |
| GRU | Cho, K. et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder" | *EMNLP 2014* | `torch.nn.GRU` |
| TFT | Lim, B. et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" | *Intern. J. Forecasting* 37(4), 1748–1764 | `pytorch-forecasting` |
| FinBERT | Araci, D. (2019) "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" | *arXiv:1908.10063* | `transformers` (HuggingFace) |
| Optuna | Akiba, T. et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework" | *KDD 2019* | `optuna` |

### 10.3 Volatility Estimators

| Estimator | Key Paper | Notes |
|---|---|---|
| Parkinson | Parkinson, M. (1980) "The Extreme Value Method for Estimating the Variance of the Rate of Return" | *J. Business* 53(1), 61–65 |
| Garman-Klass | Garman, M. & Klass, M. (1980) "On the Estimation of Security Price Volatilities from Historical Data" | *J. Business* 53(1), 67–78 |
| Yang-Zhang | Yang, D. & Zhang, Q. (2000) "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices" | *J. Business* 73(3), 477–492 |
| Rogers-Satchell | Rogers, L. & Satchell, S. (1991) "Estimating Variance from High, Low and Closing Prices" | *Annals of Applied Probability* 1(4), 504–512 |

### 10.4 Additional References — Core Extensions

| Topic | Paper | Relevance |
|---|---|---|
| Variance Risk Premium | Bollerslev, T., Tauchen, G. & Zhou, H. (2009) "Expected Stock Returns and Variance Risk Premia" | *Rev. Financial Studies* 22(11) — VRP as return predictor |
| Fama-French Factors | Fama, E. & French, K. (1993) "Common Risk Factors in the Returns on Stocks and Bonds" | *J. Financial Economics* 33(1) — 3-factor model |
| Carhart Momentum | Carhart, M. (1997) "On Persistence in Mutual Fund Performance" | *J. Finance* 52(1) — 4th momentum factor |
| Copula-GARCH | Patton, A. (2006) "Modelling Asymmetric Exchange Rate Dependence" | *Intern. Econ. Review* 47(2) — time-varying copulas |
| Structural Breaks | Bai, J. & Perron, P. (2003) "Computation and Analysis of Multiple Structural Change Models" | *J. Applied Econometrics* 18(1) |
| GPD Threshold | Danielsson, J., de Haan, L., Peng, L. & de Vries, C. (2001) "Using a Bootstrap Method to Choose the Sample Fraction in Tail Index Estimation" | *J. Multivariate Analysis* 76(2) |

### 10.5 New References — Finance Depth Overhaul

| Topic | Paper | Relevance |
|---|---|---|
| CoVaR | Adrian, T. & Brunnermeier, M. (2016) "CoVaR" | *American Econ. Review* 106(7) — systemic risk via quantile regression |
| MES / SRISK | Acharya, V., Pedersen, L., Philippon, T. & Richardson, M. (2017) "Measuring Systemic Risk" | *Rev. Financial Studies* 30(1) — marginal expected shortfall |
| Absorption Ratio | Kritzman, M., Li, Y., Page, S. & Rigobon, R. (2011) "Principal Components as a Measure of Systemic Risk" | *J. Portfolio Management* 37(4) — eigenvalue-based fragility |
| Robust Optimization | Goldfarb, D. & Iyengar, G. (2003) "Robust Portfolio Selection Problems" | *Mathematics of Operations Research* 28(1) — worst-case MV |
| Max Diversification | Choueifaty, Y. & Coignard, Y. (2008) "Toward Maximum Diversification" | *J. Portfolio Management* 35(1) — diversification ratio |
| Resampled EF | Michaud, R. (1998) *Efficient Asset Management* | Oxford Univ. Press — bootstrap-stabilized Markowitz |
| Market Impact | Almgren, R. & Chriss, N. (2000) "Optimal Execution of Portfolio Transactions" | *J. Risk* 3(2) — permanent + temporary impact |
| Walk-Forward Embargo | Lopez de Prado, M. (2018) *Advances in Financial Machine Learning* | Ch. 7 — purge/embargo for cross-validation |
| SPA Test | Hansen, P. (2005) "A Test for Superior Predictive Ability" | *J. Business & Econ. Stats* 23(4) — data-snooping correction |
| Reality Check | White, H. (2000) "A Reality Check for Data Snooping" | *Econometrica* 68(5) — bootstrap-based model comparison |
| Stationary Bootstrap | Politis, D. & Romano, J. (1994) "The Stationary Bootstrap" | *JASA* 89(428) — block bootstrap with random lengths |
| Block Length Selection | Politis, D. & White, H. (2004) "Automatic Block-Length Selection" | *Econometric Reviews* 23(1) — optimal bootstrap block |
| GARCH Bootstrap | Pascual, L., Romo, J. & Ruiz, E. (2006) "Bootstrap Prediction for Returns and Volatilities" | *Computational Stats & Data Analysis* 50(9) |
| Stability Selection | Meinshausen, N. & Bühlmann, P. (2010) "Stability Selection" | *JRSS Series B* 72(4) — robust feature selection |
| Sharpe Bootstrap | Ledoit, O. & Wolf, M. (2008) "Robust Performance Hypothesis Testing with the Sharpe Ratio" | *J. Empirical Finance* 15(5) |
| NL Shrinkage | Ledoit, O. & Wolf, M. (2020) "Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices" | *Annals of Statistics* 48(5) |
| Cointegration | Engle, R. & Granger, C. (1987) "Co-Integration and Error Correction" | *Econometrica* 55(2) — cointegration testing |
| GPH Estimator | Geweke, J. & Porter-Hudak, S. (1983) "The Estimation and Application of Long Memory Time Series Models" | *JTSA* 4(4) — long-memory detection |
| Hurst Exponent | Hurst, H. (1951) "Long-Term Storage Capacity of Reservoirs" | *Trans. ASCE* 116 — R/S analysis |
| Omega Ratio | Keating, C. & Shadwick, W. (2002) "A Universal Performance Measure" | *J. Performance Measurement* 6(3) |
| CDaR | Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005) "Drawdown Measure in Portfolio Optimization" | *Intern. J. Theoretical & Applied Finance* 8(1) |
| BH-FDR | Benjamini, Y. & Hochberg, Y. (1995) "Controlling the False Discovery Rate" | *JRSS Series B* 57(1) — multiple testing correction |

---

*This document serves as the single source of truth for the entire pipeline. Every notebook should import this file's conventions, ticker list, and feature definitions. Update this document as the project evolves.*
