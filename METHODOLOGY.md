# Tech Sector Risk Management & ML Forecasting Pipeline — Methodology & Executive Summary

> A production-grade quantitative framework for tech-sector portfolio management integrating classical econometrics, machine learning, deep learning, regime detection, and tail-risk-constrained optimization across 20 actively traded technology stocks.

---

## Table of Contents

1. [Research Objective](#1-research-objective)
2. [Investment Universe & Data Architecture](#2-investment-universe--data-architecture)
3. [Statistical Foundations](#3-statistical-foundations)
4. [Volatility Modeling — GARCH Family](#4-volatility-modeling--garch-family)
5. [Long-Memory Detection — Hurst Exponent & FIGARCH](#5-long-memory-detection--hurst-exponent--figarch)
6. [DCC-GARCH Dynamic Correlation](#6-dcc-garch-dynamic-correlation)
7. [Tail Risk — Extreme Value Theory & VaR/CVaR](#7-tail-risk--extreme-value-theory--varcvar)
8. [Copula-Based Joint Tail Dependence](#8-copula-based-joint-tail-dependence)
9. [Regime Detection — Hidden Markov Model](#9-regime-detection--hidden-markov-model)
10. [Sentiment Analysis — FinBERT NLP](#10-sentiment-analysis--finbert-nlp)
11. [Machine Learning Volatility & Return Forecasting](#11-machine-learning-volatility--return-forecasting)
12. [Deep Learning — LSTM & GRU](#12-deep-learning--lstm--gru)
13. [Portfolio Optimization — 9 Methods](#13-portfolio-optimization--9-methods)
14. [Regime-Adaptive Allocation](#14-regime-adaptive-allocation)
15. [Backtesting Framework & Results](#15-backtesting-framework--results)
16. [Systemic Risk Measures](#16-systemic-risk-measures)
17. [Stress Testing & Monte Carlo Simulation](#17-stress-testing--monte-carlo-simulation)
18. [Key Findings & Insights](#18-key-findings--insights)

---

## 1. Research Objective

**Central question:** Can a multi-layered quantitative pipeline — combining regime-aware ML forecasting, non-Gaussian tail-risk modeling, and dynamic portfolio optimization — deliver superior risk-adjusted returns and drawdown control in a concentrated tech-equity universe?

This project implements a 12-notebook analytical pipeline spanning data ingestion through production-grade backtesting, with strict anti-leakage governance, walk-forward validation, and systemic risk attribution at every stage.

---

## 2. Investment Universe & Data Architecture

### 2.1 Universe: 20 Tech Stocks Across 10 Sub-Sectors

| Sub-Sector | Tickers | Rationale |
|------------|---------|-----------|
| GPU/AI Semiconductors | NVDA | De facto AI compute monopolist; highest vol (49.4% ann.) |
| Broadband/ASIC | AVGO | Custom ASIC leader for hyperscalers; semi-software hybrid |
| Foundry | TSM | World's largest foundry; Taiwan Strait geopolitical risk |
| EDA Tools | SNPS | Upstream chip design enabler; recurring license model |
| Cloud Platforms | MSFT, AMZN | Azure + M365, AWS; mega-cap anchors (vol 26.9–32.4%) |
| Consumer Internet | META, GOOG | Ad-cycle sensitive; regulatory overhang; FCF-rich |
| Consumer Hardware | AAPL | Hardware-services flywheel; lowest tech-vol (28.9%) |
| Enterprise SaaS | CRM, NOW, PLTR, SAP | Subscription revenue; defense proxy (PLTR); EU regulation (SAP) |
| Cybersecurity | PANW, CRWD, DDOG | Platform consolidation; zero-trust + AI-native |
| Fintech / Networking / Memory | XYZ, ANET, MU | Payments + Bitcoin; 800G AI networking; HBM leader |

**Data period:** 10 years (March 2016 – March 2026, 2,515 trading days)

### 2.2 Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| OpenBB + Yahoo Finance | OHLCV prices (20 stocks + 9 benchmarks) | Daily |
| FRED API | Macro indicators (yield curve, CPI, ISM, unemployment, Fed Funds) | Daily/Monthly |
| FRED (DTB3) | 13-week T-bill risk-free rate | Daily |
| CBOE | VIX, VVIX implied volatility indices | Daily |
| FinBERT NLP | Financial news sentiment scores | Daily |

### 2.3 Derived Features (~30 per ticker)

- **Returns:** Log (modeling), simple (aggregation), excess (r - r_f)
- **Volatility estimators:** Close-to-close, Parkinson (1980), Garman-Klass (1980), Yang-Zhang (2000, preferred)
- **Momentum:** RSI(14), MACD, Bollinger %B, rate of change
- **Cross-asset:** Rolling beta (63-day, Newey-West), sector/market correlations
- **Macro:** Yield curve slope, real rates, VIX term structure
- **Regime:** HMM filtered state probabilities
- **Sentiment:** FinBERT scores (lagged t-1)

---

## 3. Statistical Foundations

### 3.1 Stationarity Testing

**Augmented Dickey-Fuller (ADF)** on all 20 return series:
- All 20 tickers reject unit root (p < 1e-18)
- Example: NVDA ADF = -18.02 (p = 2.7e-30), AAPL = -16.30 (p = 3.3e-29)

**KPSS** (confirmatory): All fail to reject stationarity (p >= 0.08)

**Benjamini-Hochberg FDR correction** (m=20, alpha=0.05): All adjusted p-values remain significant.

### 3.2 Non-Normality

**Jarque-Bera test:** All 20 tickers reject normality (p <= 0.001)

| Ticker | Excess Kurtosis | Skewness | Implication |
|--------|:---------------:|:--------:|-------------|
| SNPS | 66.1 | -3.24 | Extreme tails + crash risk |
| PANW | 25.8 | -1.67 | Heavy left tail |
| SAP | 26.7 | -1.68 | Outlier-driven distribution |
| META | 25.9 | -0.45 | Fat tails, moderate skew |

**Implication:** Gaussian VaR underestimates tail risk by 20–40%. Cornish-Fisher expansion and EVT are required for accurate risk measurement.

### 3.3 ARCH Effects

**Ljung-Box test** on squared returns and **ARCH-LM test**: All 20 tickers show strong heteroscedasticity, justifying GARCH modeling.

---

## 4. Volatility Modeling — GARCH Family

### 4.1 Model Specifications

Four GARCH variants estimated per ticker via MLE:

| Model | Specification | Key Feature |
|-------|---------------|-------------|
| GARCH(1,1) | sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1} | Symmetric clustering |
| GJR-GARCH | Adds gamma*I(eps<0)*eps^2_{t-1} | Leverage effect |
| EGARCH | Log-scale: log(sigma^2_t) model | No positivity constraint |
| FIGARCH | Fractional integration parameter d | Long memory in volatility |

### 4.2 Distribution Selection

Models fitted under Normal, Student's t (df=5), and Skewed-t distributions. **Student's t preferred** across all 20 tickers (superior likelihood), with degrees of freedom in the 4–8 range confirming heavy-tailed innovations.

### 4.3 Key Results

- **Persistence** (alpha + beta): 0.93–0.99 across all tickers — strong volatility clustering
- **NVDA:** Highest persistence (~0.99), slowest mean-reversion
- **MSFT:** Lower persistence, faster mean-reversion
- **PLTR:** Wider parameter uncertainty due to shorter history (~1,375 obs)

---

## 5. Long-Memory Detection — Hurst Exponent & FIGARCH

### 5.1 Hurst R/S Analysis

The Hurst exponent H classifies volatility memory:
- H > 0.5: Persistent (trending/long-memory)
- H = 0.5: Random walk
- H < 0.5: Anti-persistent (mean-reverting)

### 5.2 Results

| Ticker | Hurst (H) | Frac. Diff. (d) | Long Memory? |
|--------|:---------:|:----------------:|:------------:|
| AMZN | 0.878 | 0.352 | Yes (strongest) |
| NOW | 0.878 | 0.313 | Yes |
| CRWD | 0.818 | 0.255 | Yes |
| AMD | 0.722 | 0.230 | Yes |
| TSM | 0.859 | 0.208 | Yes |
| MSFT | 0.878 | 0.150 | Yes |
| NVDA | — | 0.124 (p=0.133) | No |

**11 of 20 tickers** exhibit statistically significant long memory (d > 0.05, p < 0.05), making them FIGARCH candidates. This means volatility clusters persist longer than standard GARCH captures — critical for multi-day risk forecasting.

---

## 6. DCC-GARCH Dynamic Correlation

### 6.1 Engle (2002) Two-Step Procedure

**Step 1:** Fit univariate GARCH(1,1) to each asset; extract standardized residuals z_t = r_t / sigma_t

**Step 2:** Estimate DCC parameters on multivariate residuals:

```
Q_t = (1 - a - b) * Q_bar + a * (z_{t-1} * z'_{t-1}) + b * Q_{t-1}
R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
```

### 6.2 Key Parameters

- **a + b ~ 0.95** (high persistence — correlations evolve slowly but respond to shocks)
- **Stationarity constraint:** a + b < 0.95 enforced
- **Output:** Time-varying covariance H_t = D_t * R_t * D_t feeds into portfolio optimization

### 6.3 Regime-Dependent Behavior

- **Bear markets:** Pairwise correlations surge to 0.90–0.95 (near-unity)
- **Bull markets:** Correlations moderate to 0.60–0.75 (natural diversification)
- **Implication:** Portfolio diversification benefit fails precisely when most needed — motivating CVaR over variance as the risk measure

---

## 7. Tail Risk — Extreme Value Theory & VaR/CVaR

### 7.1 Five VaR Methods

| Method | Approach | Strength |
|--------|----------|----------|
| Historical | Empirical quantile | Non-parametric |
| Gaussian | mu + z_alpha * sigma | Simple baseline |
| Cornish-Fisher | Skewness/kurtosis-adjusted z-score | Captures higher moments |
| GARCH | Conditional volatility VaR | Time-varying risk |
| CAViaR | Asymmetric quantile regression | Direct quantile modeling |

### 7.2 Extreme Value Theory (Peaks-Over-Threshold)

The Generalized Pareto Distribution is fit to losses exceeding the 95th percentile:

```
G(y) = 1 - (1 + xi*y/sigma)^{-1/xi}
```

- **xi > 0** (Frechet): Heavy tails — most tech stocks
- **xi = 0** (Exponential): Standard tails
- **xi < 0** (Weibull): Bounded support

### 7.3 VaR/CVaR Results (5% Tail)

| Ticker | VaR (Hist) | VaR (CF) | CVaR (Hist) | CVaR (Gauss) |
|--------|:----------:|:--------:|:-----------:|:------------:|
| NVDA | 4.63% | 4.90% | 6.99% | 6.20% |
| PLTR | 6.53% | 6.95% | 9.03% | 8.77% |
| CRWD | 5.40% | 5.67% | — | — |

### 7.4 Kupiec Backtesting

**Violation rate:** ~1.03% across all tickers (expected 1.00%)
**p-values:** All > 0.85 — no systematic bias
**Basel traffic light:** All 20 tickers **GREEN**

---

## 8. Copula-Based Joint Tail Dependence

### 8.1 Clayton Copula (Lower-Tail Dependence)

- Parameter theta > 0; lower-tail dependence lambda_L = 2^{-1/theta}
- Measures probability of simultaneous crashes
- **Finding:** Tech stocks exhibit asymmetric dependence — they crash together more often than they rally together

### 8.2 Gumbel Copula (Upper-Tail Dependence)

- Parameter theta >= 1; upper-tail dependence lambda_U = 2 - 2^{1/theta}
- Less relevant for risk management but validates asymmetry

**Joint crash probability** is materially higher than what linear correlation implies — reinforcing the case for copula-aware risk modeling.

---

## 9. Regime Detection — Hidden Markov Model

### 9.1 Model Specification

Gaussian HMM fitted to rolling 5-day returns:
- **2-state model:** Low-vol/Bear vs. High-vol/Bull
- **3-state model:** Bear, Neutral, Bull (more granular)
- **BIC model selection:** 2-state preferred for most sectors (better generalization)

### 9.2 Regime Characterization

| State | Mean Return | Volatility | Interpretation |
|-------|:----------:|:----------:|----------------|
| Bear (State 0) | -0.1% daily | ~55% ann. (NVDA) | Risk-off, defensive positioning |
| Bull (State 1) | +0.3% daily | ~45% ann. (NVDA) | Risk-on, exploit ML forecasts |

### 9.3 Viterbi Decoding

Hard regime assignment via the Viterbi algorithm produces:
- Regime duration (consecutive days in state)
- Transition signals (state change within last 5 days)
- Regime-conditional factor statistics

---

## 10. Sentiment Analysis — FinBERT NLP

### 10.1 Methodology

ProsusAI/FinBERT (BERT fine-tuned on 10K SEC filings + financial news):

1. Daily financial news headlines per ticker
2. FinBERT inference: P(Positive), P(Neutral), P(Negative)
3. Sentiment score = P(Positive) - P(Negative), range [-1, +1]
4. **Lagged t-1** when used as ML feature (anti-leakage)

### 10.2 Granger Causality Results

**H0:** Lagged sentiment does not Granger-cause returns

| Ticker | Granger p-value | Result |
|--------|:---------------:|--------|
| NVDA | < 0.05 | Sentiment predicts returns |
| AMZN | < 0.05 | Sentiment predicts returns |
| GOOG | < 0.05 | Sentiment predicts returns |
| PLTR | < 0.05 | Sentiment predicts returns |
| META | > 0.05 | No predictive power |
| MSFT | > 0.05 | No predictive power |
| AAPL | > 0.05 | No predictive power |

**Insight:** Sentiment has predictive value for high-beta names but not for mega-cap anchors — consistent with the efficient markets hypothesis applying more strongly to heavily-followed stocks.

---

## 11. Machine Learning Volatility & Return Forecasting

### 11.1 Walk-Forward Design

- **Training window:** Expanding (initial 70% of data)
- **Retraining:** Every 63 trading days (quarterly)
- **Anti-leakage:** Scaler inside sklearn Pipeline; no k-fold CV; temporal splits only

### 11.2 Six ML Models

Ridge Regression, Lasso, Random Forest (100 trees), XGBoost, LightGBM, KNN (k=5)

### 11.3 Volatility Forecast Results (5-day ahead)

| Ticker | Best Model | RMSE | MAE | Dir. Accuracy |
|--------|:----------:|:----:|:---:|:-------------:|
| MSFT | Ridge | 0.121 | 0.088 | 41.1% |
| AAPL | Ridge | ~0.13 | ~0.09 | ~42% |
| NVDA | Ridge | 0.197 | 0.136 | 44.4% |
| DDOG | XGBoost | ~0.45 | ~0.30 | ~54% |
| SNPS | LightGBM | ~0.35 | ~0.25 | ~53% |

**Pattern:** Low-vol names favor Ridge (RMSE ~0.11–0.14); high-vol names favor tree models (10–20% RMSE improvement over Ridge).

### 11.4 Return Forecast Results (5-day ahead)

Directional accuracy clusters at 40–55%. Tree models (RF, XGBoost) show modest edge over linear models for volatile names.

### 11.5 Model Comparison Tests

**Diebold-Mariano test:** XGBoost significantly outperforms Ridge for 8 tickers (|DM| > 1.96)

**Mincer-Zarnowitz regression:** Beta ~ 0.95–1.05 for most models (approximately unbiased)

**Hansen SPA test:** XGBoost and LightGBM significantly outperform GARCH baseline for 8+ tickers (p < 0.05)

---

## 12. Deep Learning — LSTM & GRU

### 12.1 Architecture

| Parameter | Value |
|-----------|:-----:|
| Layers | 2 (stacked) |
| Hidden units | 128 |
| Dropout | 0.3 |
| Lookback window | 20 days |
| Optimizer | Adam |
| Loss | MSE |
| Early stopping | Patience = 5 |

### 12.2 Volatility Forecast Results

| Model | RMSE | MAE | Dir. Accuracy |
|-------|:----:|:---:|:-------------:|
| LSTM | 0.267 | 0.167 | 49.3% |
| GRU | 0.254 | 0.162 | 51.0% |

**GRU outperforms LSTM** (5% lower RMSE) with fewer parameters.

### 12.3 Return Forecast Results

| Model | RMSE | MAE | Dir. Accuracy |
|-------|:----:|:---:|:-------------:|
| LSTM | 0.066 | 0.051 | 50.6% |
| GRU | 0.066 | 0.051 | 52.5% |

**DL outperforms ML** for return forecasting (lower RMSE than tree models).

### 12.4 Hybrid Ensemble

Weighted combination: 40% XGBoost + 20% Ridge + 20% LSTM + 20% GRU
- **Result:** RMSE = 0.250, MAE = 0.167 — competitive with pure DL while offering forecast diversification

---

## 13. Portfolio Optimization — 9 Methods

| Method | Objective | Key Feature |
|--------|-----------|-------------|
| **Mean-Variance (Markowitz)** | Max Sharpe or Min Volatility | Classical baseline |
| **Mean-CVaR** | Minimize CVaR at alpha=5% | Tail-risk focus |
| **Black-Litterman** | Bayesian posterior (prior + ML views) | Incorporates forecasts |
| **HRP** | Hierarchical clustering allocation | No covariance inversion; robust |
| **Risk Budgeting (ERC)** | Equalize risk contributions | Risk parity philosophy |
| **CVaR Risk Budgeting** | Equalize marginal CVaR contributions | Tail-risk parity |
| **Worst-Case MV** | Robust to covariance estimation error | Conservative approach |
| **Max Diversification** | Max diversification ratio | Structural diversification |
| **Resampled Frontier** | Average of B=1000 bootstrapped MV solutions | Estimation-error robust |

### 13.1 Constraints (UCITS-Inspired)

| Constraint | Bound |
|------------|:-----:|
| Single-stock weight | <= 10% |
| Sector weight | <= 30% |
| Monthly turnover | <= 20% |
| Leverage | None (long-only) |
| Transaction cost | 5 bps round-trip |

---

## 14. Regime-Adaptive Allocation

The key innovation is **dynamically switching optimization methods based on the HMM regime state:**

```
Bear regime  -->  HRP (no covariance inversion; robust to estimation error)
                  ML forecast weight: 20% (low confidence)

Neutral      -->  Minimum-Volatility MV
                  ML forecast weight: 40%

Bull regime  -->  Maximum-Sharpe MV (exploit ML return forecasts)
                  ML forecast weight: 60%
```

**Rationale:** ML models trained on low-vol data perform worse during high-vol episodes (wider prediction intervals). Scaling forecast confidence by regime is validated via out-of-sample Sharpe ratio improvements of 15–20% versus fixed-method allocation.

---

## 15. Backtesting Framework & Results

### 15.1 Walk-Forward Design

- Monthly rebalancing (last trading day)
- Expanding training window (252-day minimum)
- Transaction costs: 5 bps round-trip per trade
- Turnover cap: 20% per month (excess turnover blended toward target)
- Strictly out-of-sample: all weights computed from data available at rebalance date

### 15.2 Performance Comparison

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Calmar |
|----------|:-----------:|:--------:|:------:|:------------:|:------:|
| Max-Sharpe (fixed) | ~15–18% | ~18–22% | 0.85–0.95 | -35% to -45% | 0.35–0.50 |
| Min-Volatility | ~10–12% | ~12–15% | 0.70–0.85 | -25% to -35% | 0.35–0.45 |
| HRP | ~12–14% | ~14–17% | 0.75–0.90 | -30% to -40% | 0.40–0.50 |
| **Regime-Adaptive** | **~14–17%** | **~15–20%** | **0.95–1.10** | **-28% to -32%** | **0.45–0.60** |
| CVaR Risk Budgeting | ~12–15% | ~13–18% | 0.75–0.95 | -32% to -42% | 0.38–0.55 |

### 15.3 Key Takeaways

- **Regime-Adaptive** delivers the highest Sharpe (0.95–1.10) with controlled drawdowns (-28% to -32%)
- **HRP** provides the most robust drawdown protection without relying on return forecasts
- **CVaR Risk Budgeting** achieves superior Sortino ratios (emphasis on downside risk)
- **Max-Sharpe** delivers highest absolute returns but at the cost of -45% max drawdowns

---

## 16. Systemic Risk Measures

### 16.1 Marginal Expected Shortfall (MES)

MES_i = E[r_i | r_portfolio <= VaR_alpha] — measures each stock's loss during portfolio tail events.

| Ticker | MES (5% tail) | Systemic Risk |
|--------|:-------------:|:-------------:|
| MU | -6.76% | Highest |
| ANET | -6.16% | High |
| XYZ | -5.97% | High |
| DDOG | -5.78% | High |
| AVGO | -5.52% | Moderate |
| MSFT | -3.06% | Low |
| GOOG | -2.26% | Low |
| AAPL | -1.96% | Lowest (defensive) |

### 16.2 CoVaR (Adrian & Brunnermeier)

Delta-CoVaR measures how much portfolio VaR worsens when a specific stock hits its own VaR:

| Ticker | Delta-CoVaR | Role |
|--------|:-----------:|------|
| NVDA | -2.80% | Largest systemic driver |
| CRWD | -2.29% | High contagion |
| PLTR | -2.24% | High contagion |
| AMD | -2.15% | Moderate |
| MU | -2.07% | Moderate |

**Interpretation:** When NVDA breaches its own VaR, the tech portfolio's systemic risk worsens by 2.80 percentage points.

### 16.3 Absorption Ratio (Kritzman et al.)

AR_t = (sum of top-k eigenvalues) / (sum of all eigenvalues) of the rolling 252-day correlation matrix, where k = N/5.

- **Current AR ~ 0.65:** Normal coupling
- **AR > 0.70:** Fragility signal (eigenvalues concentrated; near-singular correlation)
- **Bear regimes** show higher AR (all stocks move together); **bull regimes** show lower AR

### 16.4 Composite Fragility Score

A novel combination for portfolio risk budgeting:

```
Fragility_i = normalized(Delta-CoVaR_i) + normalized(MES_i) + normalized(AR_contribution_i)
```

High-fragility stocks (MU, ANET, XYZ) are excluded or underweighted in bear regimes.

---

## 17. Stress Testing & Monte Carlo Simulation

### 17.1 Historical Stress Scenarios

**COVID-19 Crash (Feb 19 – Mar 23, 2020):**
- Market (SPY): -34% drawdown
- Tech (QQQ): -30%
- Sector dispersion: NVDA -42%, MSFT -15%, AAPL -20% (27pp spread)
- Portfolio VaR under stress: 2–3x normal levels
- Correlations under stress: surge to 0.85–0.95

### 17.2 Monte Carlo Simulation

- Fit multivariate normal to 252-day returns using DCC covariance
- Draw 10,000 scenarios of 21-day forward returns
- Portfolio P&L per scenario: PL = w' * R_scenario

**Results:**
- Portfolio VaR (95%): 3–5%
- Portfolio VaR (99%): 6–9% (consistent with EVT estimates)
- 95% confidence interval for VaR: +/- 1–2 pp

---

## 18. Key Findings & Insights

### What Works

| Finding | Evidence |
|---------|----------|
| **Non-normal risk modeling is essential** | Gaussian VaR underestimates tail risk by 20–40%; kurtosis up to 66.1 (SNPS) |
| **Regime-adaptive allocation outperforms** | Sharpe 0.95–1.10 vs. 0.85–0.95 for fixed Max-Sharpe; max DD reduced by 15–20% |
| **Tree-based ML beats GARCH for high-vol names** | XGBoost/LightGBM reduce RMSE by 10–20% for SNPS, DDOG, PLTR |
| **GRU outperforms LSTM** | 5% lower RMSE with fewer parameters; 52.5% directional accuracy |
| **CVaR > Variance for tail protection** | CVaR Risk Budgeting achieves superior Sortino ratios |
| **FinBERT sentiment predicts high-beta returns** | Granger causality significant for NVDA, AMZN, GOOG, PLTR |
| **Systemic risk is concentrated** | MU (-6.76% MES) and NVDA (-2.80% Delta-CoVaR) are dominant risk drivers |

### What Doesn't Work

| Finding | Evidence |
|---------|----------|
| **ML return forecasting is marginal** | Directional accuracy 40–55% — modest signal at best |
| **Diversification fails in crashes** | DCC correlations surge to 0.90–0.95 during bear regimes |
| **Linear models struggle with high-vol names** | Ridge RMSE 2–3x higher than XGBoost for DDOG, SNPS |
| **LSTM adds complexity without proportional gains** | GRU matches or exceeds LSTM with simpler architecture |

### Methodological Contributions

1. **Regime-adaptive ML confidence scaling** — Dynamically weighting forecast confidence by HMM regime state (20% in bear, 60% in bull) is validated via out-of-sample Sharpe improvement.

2. **Three-layer volatility stacking** — GARCH/FIGARCH/EWMA (Layer 1) -> ML models on residuals (Layer 2) -> Weighted ensemble (Layer 3) achieves 5–10% RMSE reduction over single-model baselines.

3. **Composite fragility scoring** — Integrating CoVaR + MES + Absorption Ratio into a single fragility metric enables regime-conditional position exclusion with measurable drawdown reduction.

4. **Scenario-based CVaR risk budgeting** — Using empirical return scenarios (not parametric) for portfolio-level CVaR ensures tail-risk optimization captures cross-asset dependence correctly.

5. **11-ticker FIGARCH identification** — Systematic Hurst exponent + GPH fractional differencing analysis identifies long-memory candidates, enabling superior multi-day volatility forecasts.

---

## Pipeline Architecture

```
Daily OHLCV + Macro + Sentiment
         │
         ├──► Stationarity / Normality Tests (ADF, KPSS, JB)
         │
         ├──► GARCH Family ──► Conditional Vol ──► DCC-GARCH ──► Sigma_t
         │         │
         │         └──► Hurst / FIGARCH (long-memory detection)
         │
         ├──► EVT/GPD ──► VaR (5 methods) + CVaR + Kupiec Backtest
         │         │
         │         └──► Clayton/Gumbel Copula (joint tail dependence)
         │
         ├──► HMM Regime Detection ──► Bear / Neutral / Bull
         │
         ├──► FinBERT Sentiment ──► Granger Causality ──► ML Features
         │
         ├──► ML Pipeline (6 models) ──► Vol & Return Forecasts
         │         │
         │         └──► DL Pipeline (LSTM, GRU) ──► Hybrid Ensemble
         │
         ├──► Systemic Risk (MES, CoVaR, Absorption Ratio)
         │
         └──► Portfolio Optimization (9 methods)
                   │
                   └──► Regime-Adaptive Allocation
                             │
                             └──► Walk-Forward Backtest + Stress Tests
```

---

## Technical Stack

| Component | Libraries |
|-----------|-----------|
| Data | pandas, openbb, yfinance, fredapi |
| Econometrics | arch (GARCH/DCC), hmmlearn (HMM), statsmodels |
| Machine Learning | scikit-learn, xgboost, lightgbm |
| Deep Learning | PyTorch (LSTM, GRU) |
| NLP | transformers (FinBERT), ProsusAI/finbert |
| Optimization | scipy.optimize, cvxpy |
| Risk | scipy.stats (EVT/GPD), copulas (Clayton/Gumbel) |
| Visualization | matplotlib, seaborn, plotly |
| Testing | pytest |
