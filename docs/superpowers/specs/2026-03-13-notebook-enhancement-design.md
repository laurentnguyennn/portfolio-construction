# Comprehensive Notebook Enhancement Design Specification

**Date**: 2026-03-13
**Author**: Laurent (with Claude)
**Scope**: Full restructure and analytical deepening of all 12 notebooks
**Target**: World-class quant finance portfolio for MSc admissions (SKEMA, emlyon, ESCP)

---

## 1. Design Principles

Every notebook will follow this structure:

1. **Executive Summary** — 3-5 bullet markdown cell: what this notebook does, key findings, outputs
2. **Section Headers** — Clear H2/H3 markdown with rationale before every code block
3. **Compute → Visualize → Interpret** — Every analytical block follows this triple: compute the metric, plot it, then a markdown cell explaining what it means in business/risk context
4. **Statistical Rigor** — Every claim backed by formal test (test statistic, p-value, CI, effect size)
5. **Cross-References** — Explicit links to upstream inputs and downstream consumers
6. **Key Takeaways** — End-of-section summaries and end-of-notebook synthesis

### Notebook Template Structure

```
[Executive Summary markdown]
[Setup & Imports cell]
[Section 1: markdown rationale]
  [Compute cell]
  [Visualization cell]
  [Interpretation markdown]
[Section 2: ...]
  ...
[Synthesis & Key Takeaways markdown]
[Save Outputs cell]
[Cross-Reference to Next Notebooks markdown]
```

---

## 2. NB01 — Data Ingestion & Exploratory Analysis

### Current State
- Basic EDA: summary stats, correlation heatmap, QQ plots, volume spikes, drawdowns
- ~75% complete, ~65% analytical rigor

### Enhancements

#### 2.1 New Sections to Add

**A. Autocorrelation & ARCH Effect Testing (CRITICAL — justifies NB03)**
- Ljung-Box test on raw returns (lags 1, 5, 10, 20) for all 20 tickers
- Ljung-Box test on squared returns (detects volatility clustering / ARCH effects)
- ARCH-LM test on returns (Engle 1982)
- ACF/PACF plots for top-5 most volatile tickers (NVDA, PLTR, MU, CRWD, XYZ)
- Interpretation: "Significant ARCH effects in squared returns justify GARCH modeling in NB03"
- Summary table: ticker × test → p-value with significance stars

**B. Open-to-Close vs. Close-to-Open Volatility Decomposition**
- Compute overnight returns (close-to-open) vs. intraday returns (open-to-close) for all 20 tickers
- Ratio analysis: what % of total variance comes from overnight gaps vs. trading hours?
- Visualization: stacked bar chart (overnight vol% vs. intraday vol%) per ticker
- Interpretation: "Overnight gaps are non-diversifiable event risk" — matters for position sizing

**C. Rolling Volatility Decomposition (Systematic vs. Idiosyncratic)**
- Rolling 63-day beta × SPY vol = systematic component
- Residual vol = idiosyncratic component
- Time series plot: systematic vs. idiosyncratic for 4-6 key tickers
- Cross-sectional: which tickers have highest idiosyncratic risk? (investment opportunity)
- Interpretation: "High idiosyncratic risk = potential alpha; high systematic = beta exposure"

**D. Tail Ratio Analysis**
- Tail ratio = |95th percentile return| / |5th percentile return|
- Rolling 252-day tail ratio per ticker
- Interpretation: tail ratio > 1 = positive skew (gains > losses); < 1 = negative skew (crash-prone)

**E. Correlation Stability Testing**
- Jennrich test (1970): H0 = correlation matrix is stable across sub-periods
- Split data into 4 sub-periods (2016-2019, 2020-2021, 2022-2024, 2024-2026)
- Test each pair of adjacent sub-periods
- Visualization: heatmap of correlation changes between sub-periods
- Interpretation: "Correlation instability justifies regime-conditional covariance in NB11"

**F. Survivorship & Short-History Analysis**
- Explicit table: ticker, IPO/DPO date, first available date, number of observations, % of full window
- Impact analysis: how do correlation estimates change when computed only on overlapping periods?
- Visualization: data availability Gantt chart (horizontal bars showing each ticker's coverage)

#### 2.2 Enhanced Existing Sections

**Summary Statistics Table** — Add columns:
- Annualized Sharpe ratio (vs. risk-free)
- Maximum 1-day gain, maximum 1-day loss
- Hurst exponent (mean-reversion vs. trending)
- Significance stars on Jarque-Bera and ADF

**Correlation Heatmap** — Add:
- Dendrogram-clustered correlation (hierarchical clustering)
- This preview of HRP clustering (NB11) shows natural stock groupings

**Drawdown Analysis** — Add:
- Time-to-recovery (days from trough to previous peak)
- Underwater equity curve per ticker
- Conditional drawdown analysis: average drawdown during bear regimes

**QQ Plots** — Add:
- Student-t QQ overlay (not just Normal) — demonstrates better fit
- Anderson-Darling test statistics for each ticker

**Volume Spike Detection** — Add:
- Cross-ticker volume spike synchronization: when do >10 stocks spike simultaneously?
- This identifies systemic events vs. idiosyncratic news

---

## 3. NB02 — Macro Regime & Geopolitical Context

### Current State
- Rule-based regime labels, event study, Granger causality, cross-asset correlations
- ~70% complete, ~60% rigor

### Enhancements

#### 3.1 New Sections

**A. PCA on Macro Factors (REQUIRED by CLAUDE.md)**
- Input: yield_curve_slope (10Y-2Y), VIX level, DXY change, CPI surprise (actual - consensus)
- Standardize all factors, compute PCA
- Extract first 2-3 principal components (target >80% variance explained)
- Loadings table: what does PC1 represent? (typically risk-on/risk-off)
- Visualization: biplot of first 2 PCs colored by regime
- Time series of PC1, PC2 with regime-colored background
- Save PCA scores as features for ML notebooks (NB07, NB08)

**B. Bai-Perron Structural Break Test**
- Apply to rolling 63-day beta (EW Tech vs. SPY)
- Report: break dates, confidence intervals, number of breaks (BIC-selected)
- Visualization: beta time series with vertical lines at break points + CIs
- Interpretation: "Structural breaks at [dates] correspond to [macro events]"
- If `ruptures` library available, use PELT algorithm as alternative

**C. Factor Loading Regression (20 tickers × macro factors)**
- Per-ticker OLS: return_t = α + β_VIX · ΔVIX_t + β_yield · Δyield_t + β_DXY · ΔDXY_t + ε_t
- Newey-West HAC standard errors (5 lags)
- Output: 20 × 4 table with coefficients, t-stats, significance stars
- Classification: "rate-sensitive" (|β_yield| high), "risk-on-sensitive" (|β_VIX| high), "FX-sensitive" (|β_DXY| high)
- Visualization: grouped bar chart of factor betas by sector group

**D. Regime Transition Analysis**
- Transition probability matrix from rule-based regimes
- Average duration per regime (in trading days)
- Regime persistence metric: autocorrelation of regime labels at lag 1, 5, 21
- Visualization: Sankey diagram or heatmap of transition probabilities

**E. Regime Predictability Assessment**
- Logistic regression: P(regime_t = bear | macro_features_{t-1})
- Features: lagged VIX, yield curve slope, DXY, CPI momentum, Fed Funds rate
- Out-of-sample accuracy (walk-forward)
- Interpretation: "Can we actually anticipate regime shifts? Out-of-sample AUC = X"

#### 3.2 Enhanced Existing Sections

**Event Study** — Add:
- Cumulative Abnormal Return (CAR) with pre-event baseline (60-day mean return)
- Cross-sectional t-test: is average CAR significantly ≠ 0?
- Standardized event windows (normalize all to 20 trading days post-event for comparison)
- Confidence bands around mean CAR

**Granger Causality** — Add:
- Reverse direction: Tech returns → VIX (feedback effect)
- Impulse Response Functions (IRF) from bivariate VAR
- Forecast Error Variance Decomposition (FEVD): what % of tech return variance is explained by VIX shocks?

---

## 4. NB03 — Volatility Econometrics

### Current State
- Full GARCH pipeline (4 models × 4 distributions × 20 tickers)
- ~80% complete, ~70% rigor

### Enhancements

#### 4.1 New Sections

**A. Parameter Table with Confidence Intervals**
- Full parameter extraction: ω, α, β, γ (GJR/EGARCH), d (FIGARCH) per ticker
- Standard errors and t-statistics from model.summary()
- Significance stars (*, **, ***)
- Persistence measure: α + β for each model (close to 1.0 = high persistence)
- Visualization: persistence heatmap (tickers × models)

**B. Leverage Effect Analysis**
- Extract γ (asymmetry parameter) from GJR-GARCH for all 20 tickers
- Statistical significance test: is γ > 0? (one-sided t-test)
- Visualization: bar chart of γ values with error bars
- Interpretation: "NVDA γ = 0.15 means negative shocks amplify volatility 15% more than positive shocks of equal magnitude"
- News Impact Curve: plot σ²_{t+1} vs. ε_t for symmetric GARCH vs. GJR to visualize asymmetry

**C. Realized vs. Conditional Volatility Comparison**
- Yang-Zhang 21-day realized vol (from NB01 feature engineering)
- Best-fit GARCH 21-day conditional vol forecast
- Time series overlay: realized vs. conditional for top-6 tickers
- Correlation and RMSE between realized and conditional
- Lead/lag analysis: does GARCH lead or lag realized vol?
- Mincer-Zarnowitz regression: conditional vol vs. realized vol (forecast efficiency)

**D. Out-of-Sample Forecast Evaluation**
- Hold-out: last 252 trading days (~1 year)
- 1-step ahead conditional vol predictions vs. next-day squared returns (proxy for realized vol)
- Metrics: RMSE, MAE, QLIKE loss function (asymmetric — penalizes underestimation more)
- Diebold-Mariano test: pairwise comparison of GARCH vs. GJR vs. EGARCH forecast accuracy
- Visualization: forecast vs. actual scatter plot with 45-degree line

**E. Innovation Distribution Tail Analysis**
- For best-fit distribution per ticker, extract tail parameters (df for Student-t, shape for skewed-t)
- Compare: how much fatter are tails vs. Normal? (ratio of 1% quantile)
- Visualization: standardized residual density plot with Normal overlay and best-fit distribution overlay
- Interpretation: "Student-t with df=5.3 for NVDA implies tails 2.1x fatter than Normal"

**F. Volatility Term Structure**
- Multi-step ahead forecasts from best model: h = 1, 5, 21, 63 days
- Plot: σ(h) vs. horizon h (should converge to unconditional vol for large h)
- Compare term structure shape across tickers: flat (MU, cyclical) vs. steep (AAPL, mean-reverting)

**G. Sub-Period Parameter Stability**
- Re-estimate best model on 4 sub-periods: 2016-2019, 2020-2021, 2022-2024, 2024-2026
- Parameter stability table: do α, β, γ shift significantly across periods?
- Visualization: parameter trajectory plots
- Interpretation: "Parameters unstable across regimes justifies regime-switching GARCH in NB05"

#### 4.2 Enhanced Existing Sections

**Model Comparison Table** — Restructure to show:
- All 20 tickers × best model per distribution
- Grand summary: how often does GJR beat GARCH? (win count)
- Which distribution wins most? (t vs. skewt vs. ged)

**Residual Diagnostics** — Add:
- Sign bias test (Engle & Ng, 1993): does model capture asymmetry correctly?
- Kolmogorov-Smirnov test on standardized residuals vs. assumed distribution
- Weighted Ljung-Box (WLB) for robustness

---

## 5. NB04 — Tail Risk: VaR, CVaR, EVT

### Current State
- 5 VaR methods, CVaR, EVT/GPD, backtesting, 3 copula pairs
- ~75% complete, ~65% rigor

### Enhancements

#### 5.1 New Sections

**A. VaR Method Accuracy Ranking**
- Cross-ticker comparison: which method has lowest average violation rate deviation from target α?
- Ranking table: method × (avg violation rate, Kupiec pass rate, Christoffersen pass rate)
- Visualization: violation rate scatter plot by method (target line at α)
- Interpretation: "GARCH-based VaR passes backtests for 18/20 tickers; Gaussian fails for 14/20"

**B. Multi-Day VaR (5d, 10d, 21d)**
- Historical simulation at 5d and 21d horizons (use overlapping returns)
- √T scaling from 1-day Gaussian VaR → compare to actual multi-day quantile
- Visualization: scaling factor actual vs. √T (quantify scaling error)
- Interpretation: "√T rule overestimates 21-day VaR by X% on average due to mean-reversion in vol"

**C. Backtest Power Analysis**
- Confidence intervals on violation rate: Wilson score interval
- Required sample size for 99% VaR backtest to achieve 80% statistical power
- Visualization: power curve (sample size vs. detection probability)
- Interpretation: "With 2,500 observations, Kupiec test has only X% power to detect a 50% misspecification at 99% level"

**D. EVT Threshold Sensitivity**
- Vary threshold quantile from 0.90 to 0.98 in steps of 0.01
- Plot: EVT-VaR estimate vs. threshold for each ticker
- Mean excess function plot: E[X - u | X > u] vs. u (should be linear for GPD)
- Hill plot: tail index estimator vs. number of order statistics
- Interpretation: "Stable plateau in Hill plot at [range] confirms GPD assumption"

**E. Systematic Copula Analysis (all relevant pairs)**
- Expand from 3 pairs to all within-sector pairs + key cross-sector pairs (~15-20 pairs)
- For each pair: Clayton θ, Gumbel θ, lower-tail λ_L, upper-tail λ_U, joint crash P
- Visualization: tail dependence heatmap (20 × 20 matrix, lower triangle = λ_L, upper = λ_U)
- Time-varying copula: rolling 252-day Clayton θ for top-5 pairs
- Interpretation: "NVDA-AMD lower-tail dependence = 0.45 means crashes are highly correlated"

**F. Scenario Matrix Validation**
- Distribution of portfolio returns from scenario matrix (histogram with VaR/CVaR lines)
- Compare scenario-based CVaR to analytical CVaR (should be close)
- Tail coverage: are extreme events (>3σ) adequately represented in scenarios?
- Bootstrap confidence interval on scenario-based CVaR

**G. Tail-Fatness Ranking**
- Rank all 20 tickers by GPD shape parameter ξ
- Classification: heavy-tailed (ξ > 0.2), moderate (0.05 < ξ < 0.2), thin-tailed (ξ < 0.05)
- Visualization: ranked bar chart of ξ values with confidence intervals
- Cross-reference: do high-ξ tickers also have high kurtosis in NB01?

#### 5.2 Enhanced Existing Sections

**Cornish-Fisher VaR** — Add:
- Explicit monotonicity guard verification: test with extreme skew/kurtosis values
- Fallback documentation: if CF becomes non-monotonic, use historical VaR instead
- Side-by-side comparison: CF-VaR vs. Historical VaR for high-kurtosis tickers

**CAViaR** — Verify implementation exists. If incomplete:
- Implement symmetric absolute value CAViaR: q_t = β_0 + β_1 · q_{t-1} + β_2 · |r_{t-1}|
- Quantile regression estimation via scipy.optimize
- Compare CAViaR to GARCH-VaR: which adapts faster to regime changes?

---

## 6. NB05 — Regime Detection (HMM & Markov Switching)

### Current State
- Basic HMM fitting, BIC selection, transition matrix extraction
- ~40% complete, ~30% rigor, ZERO visualizations

### Enhancements (Major Expansion Required)

#### 6.1 New Sections

**A. Full 10-Year Regime Visualization (CRITICAL)**
- Price chart (XLK or EW Tech) with regime-colored background shading
- Using plot_regime_overlay() from visualization.py
- Side panel: regime probability time series (smoothed posteriors from HMM)
- Interpretation: "HMM identifies 3 distinct regimes: bull (μ=0.08%, σ=0.7%), neutral (μ=0.01%, σ=1.2%), bear (μ=-0.12%, σ=2.1%)"

**B. Per-Ticker HMMs (Top-5 Volatile Names)**
- Fit 2-state and 3-state HMMs separately on NVDA, PLTR, MU, CRWD, XYZ
- Compare: do individual ticker regimes align with sector-wide regimes?
- Regime synchronization metric: % of days where ticker regime = sector regime
- Visualization: 5-panel regime overlay for each ticker

**C. Regime-Conditional Risk Metrics**
- Separate VaR/CVaR per regime (using NB04 methods filtered by regime labels)
- Separate correlation matrices per regime (already in regime_utils.py)
- Visualization: side-by-side heatmaps (bull correlation vs. bear correlation)
- Interpretation: "Correlations increase from 0.35 (bull) to 0.72 (bear) — diversification fails when needed most"

**D. Macro Factor Regime Prediction**
- Logistic regression: P(regime_t = bear | macro_{t-1}) using NB02 macro features
- Walk-forward evaluation: can we predict regime 1 day ahead?
- AUC-ROC score, confusion matrix
- Which macro features are most predictive? (coefficient analysis)

**E. Transition Probability Heatmap**
- Visualize the transition matrix as annotated heatmap
- Expected duration per state (from stationary distribution)
- Regime persistence: average consecutive days in each regime
- Visualization: duration distribution histogram per regime

**F. Regime-Conditional Return Distribution**
- KDE plots of returns in each regime (overlaid)
- Regime-conditional skewness and kurtosis
- Regime-conditional Sharpe ratio
- Visualization: 3-panel density overlay (bull vs. neutral vs. bear)

**G. MS-GARCH (Markov-Switching GARCH)**
- If feasible: regime-dependent GARCH parameters (separate α, β per state)
- Compare: MS-GARCH vs. standard GARCH + HMM regime labels
- If library support insufficient: document as future work with rationale

---

## 7. NB06 — NLP Sentiment (FinBERT)

### Current State
- Synthetic momentum proxy only, no real NLP, no validation
- ~30% complete, ~20% rigor

### Enhancements (Major Rebuild — Real FinBERT)

#### 7.1 Data Collection Pipeline

**A. Financial News Headlines**
- Primary: `feedparser` on major financial RSS feeds (Reuters, Bloomberg terminal-free feeds, Yahoo Finance RSS)
- Secondary: `newspaper3k` for headline extraction from financial news sites
- Tertiary fallback: Financial PhraseBank dataset (Malo et al., 2014) for validation
- Per-ticker keyword filtering: match headlines to tickers by company name / ticker symbol
- Date range: full 10-year window (2016-03 to 2026-03)
- Store raw headlines in `data/raw/headlines/`

**B. FinBERT Inference**
- Model: `ProsusAI/finbert` from HuggingFace Transformers
- Per-headline: positive, negative, neutral probability scores
- Batch inference with GPU if available, CPU fallback
- Output: per-headline DataFrame (date, ticker, headline_text, pos, neg, neu, sentiment_score)

#### 7.2 Feature Engineering

**A. Daily Sentiment Aggregation**
- `sentiment_mean`: average sentiment score per ticker per day
- `sentiment_std`: intraday disagreement (high std = conflicting news)
- `sentiment_volume`: number of headlines per ticker per day (attention proxy)
- `sentiment_momentum`: 5-day change in sentiment_mean
- `sentiment_extreme`: binary flag for sentiment_mean > 2σ or < -2σ

**B. Lag Enforcement (t-1)**
- ALL sentiment features shifted by 1 business day
- Explicit verification cell: assert no same-day alignment between sentiment and returns
- Documentation: why t-1 lag prevents contemporaneous information leakage

#### 7.3 Validation & Analysis

**A. Granger Causality (Sentiment → Returns/Vol)**
- Test: does lagged sentiment Granger-cause next-day returns? (lags 1-5)
- Test: does lagged sentiment Granger-cause next-day realized volatility?
- Per-ticker results table with p-values
- Interpretation: "Sentiment has predictive power for 8/20 tickers at 5% level"

**B. Sentiment-VIX Correlation**
- Rolling 63-day correlation between average cross-ticker sentiment and VIX
- Interpretation: negative correlation expected (bad sentiment → high VIX)

**C. Sentiment-Regime Analysis**
- Average sentiment per HMM regime (from NB05)
- Does sentiment lead regime transitions? (lagged cross-correlation)
- Visualization: sentiment time series with regime-colored background

**D. Event-Level Sentiment Spikes**
- Map sentiment extremes to KEY_EVENTS from config.py
- Event study: sentiment trajectory around major events (±10 days)
- Visualization: event-aligned sentiment chart for 4-5 key events

**E. Sentiment Predictive Power Assessment**
- Simple univariate regression: next-day return = α + β × sentiment_{t-1}
- R² and t-stat per ticker
- Honest assessment: "Sentiment alone explains X% of return variance — modest but non-zero"

#### 7.4 Fallback Protocol
- If real headline data is insufficient (<50 headlines per ticker per year):
  - Use synthetic proxy as in current implementation
  - Document clearly: "Synthetic sentiment proxy based on momentum/volatility signals"
  - Label all downstream uses as "proxy-based" in NB07, NB08

---

## 8. NB07 — ML Volatility Forecasting

### Current State
- Walk-forward with 5 models, DM test, MZ test, SHAP for NVDA only
- ~75% complete, ~70% rigor

### Enhancements

#### 8.1 New Sections

**A. Regime-Conditional SHAP Analysis**
- Split SHAP values by HMM regime (bull vs. bear)
- Comparison: do feature importances change across regimes?
- Visualization: side-by-side SHAP beeswarm plots (bull regime vs. bear regime)
- Interpretation: "In bear markets, VIX-related features dominate; in bull markets, momentum features are more important"

**B. Feature Ablation Study (Single-Model Scope)**
- Remove feature groups one at a time: (1) momentum, (2) volatility, (3) macro, (4) sentiment, (5) regime
- Measure RMSE degradation per group removal for best single model (XGBoost)
- Visualization: waterfall chart of feature group contributions
- Interpretation: "Removing macro features increases RMSE by X%, confirming their importance for vol forecasting"
- Note: NB10 performs cross-model ablation; this is single-model ablation for interpretability

**C. Learning Curves**
- Train on increasing fractions of data (10%, 20%, ..., 100%)
- Plot: RMSE vs. training set size (train and test curves)
- Diagnose: are models data-starved (curves still declining) or saturated (plateau)?
- Per-model learning curves (Ridge vs. XGBoost vs. RF)

**D. Multi-Horizon Comparison (5d vs. 21d)**
- Run full walk-forward pipeline for both 5-day and 21-day forward vol
- Comparison table: RMSE, DA for each model × each horizon
- Skill decay analysis: how much does performance degrade from 5d to 21d?
- Visualization: paired bar chart (5d vs. 21d RMSE per model)

**E. Model Confidence Set (Hansen et al., 2011)**
- Simultaneously compare all 6+ models (not just pairwise DM)
- T-max statistic with stationary block bootstrap
- Output: which models are in the "superior set" at 10% significance?
- Interpretation: "The MCS contains XGBoost and Stacking; all other models are significantly inferior"

**F. Calibration Analysis**
- Predicted vol distribution vs. realized vol distribution
- Reliability diagram: binned predicted vol vs. average realized vol
- Interpretation: "XGBoost overestimates vol by 12% in low-vol periods and underestimates by 8% in high-vol periods"

**G. Per-Ticker Predictability Ranking**
- RMSE and DA per ticker (not just averaged)
- Which tickers are most/least predictable?
- Visualization: scatter plot (average RMSE vs. average vol per ticker)
- Interpretation: "NVDA is hardest to forecast (high vol + nonlinear dynamics); AAPL is easiest (stable, low vol)"

#### 8.2 Enhanced Existing Sections

**SHAP Analysis** — Expand from NVDA-only to:
- Top-3 diverse tickers: one high-vol (NVDA), one low-vol (AAPL/MSFT), one mid-vol (PANW)
- SHAP waterfall for highest-error predictions
- Force plot for individual predictions during key events (COVID, rate shock)

**Stacking Ensemble** — Add:
- Weight analysis: which base models contribute most?
- Does stacking significantly outperform best individual? (DM test)

---

## 9. NB08 — ML Return Direction & Price Forecasting

### Current State
- Classification + regression tasks, but incomplete: no threshold optimization, no calibration, no strategy backtest
- ~50% complete, ~40% rigor

### Enhancements (Major Completion Required)

#### 9.1 Task A — Classification Enhancements

**A. Threshold Optimization**
- Grid search on decision boundary (0.3 to 0.7 in 0.01 steps)
- At each threshold: compute Sharpe ratio of simple long/flat strategy
- Optimal threshold = maximize out-of-sample Sharpe
- Visualization: Sharpe vs. threshold curve with optimal point marked

**B. Calibration Curve (Platt Scaling)**
- Apply Platt scaling (logistic regression on predicted probabilities)
- Before/after calibration: reliability diagram (predicted P vs. observed frequency)
- Brier score improvement
- Visualization: calibration curve with perfect-calibration diagonal

**C. Confusion Matrix by Regime**
- Separate confusion matrices for bull vs. bear vs. neutral regimes
- Which regime is hardest to predict? (lowest accuracy)
- Visualization: 3-panel confusion matrix

**D. Per-Ticker Predictability**
- AUC-ROC per ticker (not just averaged)
- Ranking: which tickers have directionally predictable returns?
- Interpretation: "Momentum-driven tickers (NVDA, MU) show higher directional predictability"

#### 9.2 Task B — Regression Enhancements

**A. Feature Importance Comparison**
- SHAP for return prediction vs. SHAP for vol prediction (NB07)
- Which features matter for returns but not vol, and vice versa?
- Visualization: side-by-side feature importance bar charts

#### 9.3 Task C — Multi-Horizon (Complete Implementation)

**A. Performance Decay Regression**
- Formal regression: RMSE = α + β × log(horizon)
- Slope β quantifies how fast predictive skill degrades
- Visualization: RMSE vs. horizon (1d, 5d, 21d) with fitted decay curve
- Interpretation: "Return predictability half-life ≈ X days"

#### 9.4 Strategy Backtest (NEW — Critical Missing Piece)

**A. Simple Long/Flat Strategy**
- Signal: if P(up) > optimal threshold → long; else flat (cash)
- Apply 10 bps transaction costs on each signal change
- Performance metrics: annualized return, vol, Sharpe, max DD, Calmar
- Benchmark: buy-and-hold, equal-weight
- Visualization: cumulative return comparison

**B. Signal-Weighted Strategy**
- Position size = P(up) - 0.5 (scaled from -0.5 to +0.5, long-only → 0 to 1)
- Compare: binary signal vs. probability-weighted
- Transaction cost sensitivity: 5 bps, 10 bps, 20 bps

---

## 10. NB09 — Deep Learning Forecasting

### Current State
- LSTM/GRU implemented, walk-forward, DM tests
- ~70% complete, no TFT, no attention analysis

### Enhancements

#### 10.1 New Sections

**A. Temporal Fusion Transformer (TFT)**
- Implementation via `pytorch-forecasting`
- Static covariates: sector group, market_cap_bucket
- Known future inputs: day_of_week, month, quarter, earnings_flag
- Observed inputs: all engineered features
- Quantile outputs: 10th, 50th, 90th percentile
- Multi-horizon: 1d, 5d, 21d simultaneous predictions

**B. Attention Weight Analysis (TFT)**
- Temporal attention: which past time steps does the model focus on?
- Feature attention: which input features receive highest attention?
- Visualization: attention heatmap (time steps × features)
- Event-specific attention: what does the model attend to during COVID crash?

**C. Prediction Interval Calibration**
- TFT quantile outputs: are 80% prediction intervals actually 80% reliable?
- Coverage analysis: observed % of actuals within [10th, 90th] quantile
- Visualization: prediction + interval time series with actuals

**D. Residual Analysis (DL vs. GARCH)**
- Compute: DL prediction error - GARCH prediction error
- Is DL capturing patterns GARCH misses? (test: are residuals significantly different?)
- Regime-conditional: where does DL outperform GARCH? (bull? bear? transitions?)

**E. Computational Cost-Benefit Analysis**
- Detailed table: model × (params, train_time, inference_time, memory, RMSE)
- Cost-benefit metric: RMSE improvement per unit of compute
- Visualization: Pareto frontier (RMSE vs. compute cost)
- Interpretation: "XGBoost achieves 95% of LSTM accuracy at 1% of the computational cost"

**F. Ensemble with ML Models**
- Simple average ensemble: (XGBoost + LSTM + GRU) / 3
- Optimal weight ensemble: minimize OOS RMSE via convex optimization
- Does DL ensemble significantly outperform ML-only ensemble?

---

## 11. NB10 — Hybrid Model & Comprehensive Audit

### Current State
- 3-stage hybrid, DM tests, MCS, sub-period analysis
- ~85% complete, ~90% rigor (strongest notebook)

### Enhancements

#### 11.1 New Sections

**A. Lookback Window Sensitivity Analysis**
- Vary lookback window: 30, 60, 90, 120 days for feature computation
- Re-run best model (XGBoost) with each lookback
- Compare RMSE, DA across window lengths
- Visualization: RMSE vs. lookback window (line plot with CI bands)
- Interpretation: "Optimal lookback = X days; shorter windows capture regime shifts, longer windows reduce noise"

**B. Train/Test Gap Quantification**
- Per model: in-sample RMSE vs. out-of-sample RMSE
- Gap ratio = (OOS RMSE / IS RMSE) - 1
- Large gap = overfitting; small gap = good generalization
- Visualization: scatter plot (IS RMSE vs. OOS RMSE, 45-degree line)

**C. Learning Curves**
- Vary training set size: 500, 1000, 1500, 2000, all available
- Plot train/test RMSE vs. training samples
- Identify: which models are data-hungry? (LSTM vs. Ridge)

**D. Permutation Importance Stability**
- Permutation importance on 3 random shuffles → mean ± std
- Compare: does permutation importance agree with SHAP?
- Flag inconsistencies: features where SHAP says "important" but permutation says "not"

**E. Regime-Adaptive Hybrid Weights**
- Optimize separate ensemble weights for bull vs. bear regimes
- Does regime-adaptive weighting improve OOS performance?
- Visualization: weight comparison (bull weights vs. bear weights)

**F. Economic Significance**
- Translation: does a 5% RMSE improvement lead to a measurable Sharpe improvement?
- Link to NB11: map forecast accuracy to portfolio performance
- Visualization: scatter (RMSE improvement % vs. Sharpe improvement %)

**G. Comprehensive Audit Summary**
- Traffic-light table: each model gets Green/Yellow/Red for each criterion
- Criteria: OOS accuracy, stability, efficiency (MZ), overfitting risk, compute cost
- Recommendation: which model to use in NB11 and why

---

## 12. NB11 — Portfolio Optimization

### Current State
- 5 optimization methods, backtest, regime-adaptive strategy
- ~80% complete, ~80% rigor

### Enhancements

#### 12.1 New Sections

**A. Proper CVaR Integration (from NB04 scenario matrix)**
- Load `return_scenarios.parquet` from NB04
- Use actual scenario matrix (not "last 2 years" fallback)
- Verify: scenario matrix dimensions, tail coverage, date alignment
- Compare: scenario-based CVaR vs. historical CVaR vs. parametric CVaR for portfolio

**B. Portfolio Attribution Analysis**
- Brinson-Fachler attribution: allocation effect + selection effect + interaction
- Factor attribution: how much return comes from market beta vs. sector tilt vs. stock selection?
- Monthly attribution table for backtest period
- Visualization: stacked bar of attribution components over time

**C. Turnover Sensitivity Analysis**
- Vary max turnover constraint: 10%, 15%, 20%, 30%, unconstrained
- Plot: Sharpe ratio vs. turnover constraint
- Find optimal turnover cap (best risk-adjusted return after costs)
- Visualization: efficient turnover frontier

**D. Transaction Cost Sensitivity**
- Vary cost assumption: 5, 10, 15, 20 bps
- Impact on each strategy's annualized return and Sharpe
- At what cost level does active management become unprofitable vs. equal-weight?

**E. Covariance Method Comparison**
- Run best optimization (max-Sharpe) with each covariance method
- Compare: Ledoit-Wolf vs. DCC-GARCH vs. Regime-Conditional backtest performance
- Which covariance method produces most stable out-of-sample performance?

**F. Regime-Adaptive Strategy Deep Dive**
- Current: switch HRP (bear) vs. mean-variance (bull)
- Expand: test all combinations (HRP/ERC/MV × bull/neutral/bear)
- Optimize: which method works best in each regime? (backtest each combination)
- Add neutral regime handling (currently may be missing)

**G. Concentration & Diversification Metrics**
- Herfindahl-Hirschman Index (HHI) of portfolio weights over time
- Effective number of positions: 1 / Σ w_i²
- Diversification ratio: weighted average vol / portfolio vol
- Visualization: time series of diversification metrics

#### 12.2 Enhanced Existing Sections

**Efficient Frontier** — Add:
- Individual stocks plotted on the frontier (risk-return scatter)
- Current portfolio marked, benchmark marked
- Tangency portfolio (max Sharpe) highlighted

**Weight Evolution** — Add:
- Sector-level weight evolution (aggregate by constraint group)
- Identify: when does semiconductor allocation hit 30% cap?

---

## 13. NB12 — Stress Testing & Final Report

### Current State
- Historical + hypothetical stress, Monte Carlo, factor sensitivity
- ~80% complete, ~75% rigor

### Enhancements

#### 13.1 New Sections

**A. EVT-Enhanced Monte Carlo**
- Replace Normal innovation distribution with GPD tails (from NB04)
- Compare: Normal MC vs. EVT MC tail probabilities
- How much does EVT change P(drawdown > 20%)?
- Visualization: overlay of Normal vs. EVT return distributions from MC

**B. Regime-Conditional Monte Carlo**
- Separate calibration by HMM regime: different μ, σ, correlation per regime
- Sample regime sequence from transition matrix, then generate returns conditional on regime
- More realistic tail behavior than unconditional MC
- Compare: unconditional vs. regime-conditional VaR, CVaR, P(DD>20%)

**C. Correlation Shock Scenarios**
- Taiwan crisis: force correlations to 0.95 for semiconductor names
- AI bubble: force correlations among AI-exposed names to 0.9
- How much does portfolio VaR increase under correlation breakdown?
- Visualization: VaR comparison (normal vs. shocked correlation)

**D. Reverse Stress Testing**
- Question: "What market conditions cause a 30% portfolio loss?"
- Method: identify historical or simulated scenarios closest to -30% return
- Factor decomposition of worst scenarios
- Interpretation: "A 30% loss requires simultaneous: VIX > 50, 10Y yield spike > 100bps, and DXY > 5%"

**E. Probability-Weighted Scenario Assessment**
- Assign subjective probabilities to hypothetical scenarios (e.g., Taiwan crisis: 5%, AI bubble: 15%)
- Probability-weighted expected loss across all scenarios
- Visualization: risk contribution by scenario (bubble chart: x=probability, y=impact, size=expected loss)

**F. Optimal Hedge Recommendations**
- If portfolio has max drawdown risk from [factor]: which position change minimizes it?
- Marginal VaR contribution per position
- Visualization: marginal risk contribution bar chart
- Interpretation: "Reducing NVDA from 10% to 5% decreases portfolio VaR by X%"

**G. Monte Carlo Validation**
- Kolmogorov-Smirnov test: MC distribution vs. historical return distribution
- Are MC paths statistically consistent with observed data?
- Convergence check: VaR estimate vs. number of MC paths (stability after ~5000?)

#### 13.2 Enhanced Final Dashboard

**A. Per-Ticker Risk Card (Enhanced)**
- Each ticker gets: annualized vol, VaR 99%, CVaR 99%, max DD, beta, HMM regime duration, top-3 SHAP features, sector group, constraint binding status
- Formatted as styled HTML table in notebook

**B. Model Hierarchy Summary**
- Best model for vol forecasting: [model] (NB07/NB10)
- Best model for return forecasting: [model] (NB08)
- Best portfolio method: [method] (NB11)
- Key insight: what combination of methods gives best risk-adjusted return?

**C. Actionable Recommendations**
- Current regime assessment: bull/neutral/bear based on latest HMM state
- Recommended allocation method for current regime
- Top-3 overweight and underweight positions with rationale
- Risk monitoring dashboard: what to watch (VIX threshold, yield curve inversion, etc.)

#### 13.3 Final Deliverables Generation

**A. PowerPoint Presentation (25-30 slides)**
- Generated via `python-pptx`
- Structure: Executive Summary (2 slides) → Data & EDA Highlights (3) → Macro Regimes (2) → Volatility Modeling (3) → Tail Risk (2) → Regime Detection (2) → Sentiment (1) → ML Forecasting (3) → Deep Learning (2) → Hybrid Audit (2) → Portfolio Optimization (3) → Stress Testing (3) → Conclusions & Recommendations (2)
- Each slide: title, key finding, one visualization, one supporting metric
- Consistent branding: dark blue header, white background, Calibri font

**B. PDF Audit Report (15-20 pages)**
- Generated via `fpdf2`
- Sections: Executive Summary, Methodology Overview, Model Comparison Results, Portfolio Backtest Performance, Stress Test Findings, Risk Factor Decomposition, Limitations & Caveats, Appendix (parameter tables)
- All key tables and figures embedded
- Cross-referenced to notebook cells for full reproducibility

---

## 14. Cross-Notebook Enhancements

### 14.1 New `src/` Utilities Needed

| Utility | Module | Purpose |
|---------|--------|---------|
| `ljung_box_test(returns, lags)` | `feature_engineering.py` | Autocorrelation test for NB01 |
| `arch_lm_test(returns, lags)` | `feature_engineering.py` | ARCH effect test for NB01 |
| `jennrich_test(corr1, corr2, n1, n2)` | `feature_engineering.py` | Correlation stability for NB01 |
| `hurst_exponent(series)` | `feature_engineering.py` | Mean-reversion/trending for NB01 |
| `bai_perron_test(series)` | `feature_engineering.py` | Structural breaks for NB02 |
| `sign_bias_test(residuals)` | `garch_utils.py` | Leverage validation for NB03 |
| `news_impact_curve(model_result)` | `garch_utils.py` | Asymmetry visualization for NB03 |
| `hill_plot(returns, k_range)` | `risk_metrics.py` | EVT threshold selection for NB04 |
| `mean_excess_function(returns, thresholds)` | `risk_metrics.py` | GPD validation for NB04 |
| `model_confidence_set(losses, alpha)` | `ml_pipeline.py` | MCS for NB07/NB10 |
| `platt_scaling(y_true, y_prob)` | `ml_pipeline.py` | Calibration for NB08 |
| `brinson_fachler_attribution(...)` | `backtest_engine.py` | Attribution for NB11 |
| `sentiment_pipeline(tickers, dates)` | NEW: `sentiment.py` | FinBERT pipeline for NB06 |

### 14.2 Visualization Additions

| Plot Function | Module | Used In |
|---------------|--------|---------|
| `plot_acf_pacf_grid(...)` | `visualization.py` | NB01 |
| `plot_data_availability_gantt(...)` | `visualization.py` | NB01 |
| `plot_regime_probability_timeseries(...)` | `visualization.py` | NB05 |
| `plot_transition_heatmap(...)` | `visualization.py` | NB05 |
| `plot_shap_regime_comparison(...)` | `visualization.py` | NB07 |
| `plot_calibration_curve(...)` | `visualization.py` | NB08 |
| `plot_learning_curves(...)` | `visualization.py` | NB07, NB10 |
| `plot_feature_ablation_waterfall(...)` | `visualization.py` | NB07 |
| `plot_attention_heatmap(...)` | `visualization.py` | NB09 |
| `plot_attribution_stacked(...)` | `visualization.py` | NB11 |
| `plot_reverse_stress(...)` | `visualization.py` | NB12 |
| `plot_risk_card_table(...)` | `visualization.py` | NB12 |

---

### 14.3 Test Suite (`tests/test_pipeline_integrity.py`)

CLAUDE.md requires automated no-lookahead-bias verification. Add/update these tests:

| Test | Purpose |
|------|---------|
| `test_no_future_features()` | Verify no feature at time t uses data from t+1 or later. Check all feature columns against their computation window. |
| `test_sentiment_lag()` | Verify sentiment features in `sentiment_features.parquet` are lagged by exactly 1 business day. |
| `test_walk_forward_temporal_order()` | Verify walk-forward predictions at time t only use training data from times < t. |
| `test_train_test_no_overlap()` | Verify train/test splits are strictly temporal with no overlap. |
| `test_portfolio_backtest_uses_oos_only()` | Verify NB11 portfolio weights at time t use only out-of-sample ML predictions available at t. |
| `test_feature_stationarity()` | Verify ADF test passes for all return and volatility features (p < 0.05). |
| `test_cornish_fisher_monotonicity()` | Verify CF-VaR monotonicity guard activates for extreme skew/kurtosis. |
| `test_constraint_enforcement()` | Verify portfolio weights satisfy: max 10% per stock, max 30% per sector, long-only. |

### 14.4 DCC-GARCH Status in NB11

The `src/dcc_garch.py` module (fully implemented, Engle 2002 two-step procedure) provides `run_dcc_garch()`. NB11 should:
1. Import and run DCC-GARCH on the return panel
2. Use the time-varying covariance matrices at each rebalance date
3. Compare against Ledoit-Wolf and regime-conditional covariance in backtest

### 14.5 KPSS Verification in NB01

The existing `src/feature_engineering.py` contains `kpss_test()`. NB01 should:
1. Run both ADF and KPSS on all return series (confirmatory approach)
2. Present joint results: ADF rejects unit root AND KPSS fails to reject stationarity = confirmed stationary
3. Flag any conflicts (ADF says stationary but KPSS says non-stationary → fractionally integrated?)

---

## 15. Implementation Order

Based on the dependency graph:

| Phase | Notebooks | Dependency |
|-------|-----------|------------|
| **Phase 1** | NB01 | None (data foundation) |
| **Phase 2** | NB02, NB03, NB05 (core), NB06 | NB01 (parallelizable) |
| **Phase 3** | NB04 | NB03 (GARCH conditional vol) |
| **Phase 4** | NB05 (optional sections) | NB02, NB04 |
| **Phase 5** | NB07, NB08 | NB03, NB04, NB05, NB06 |
| **Phase 6** | NB09 | NB07, NB08 (benchmarks) |
| **Phase 7** | NB10 | NB03, NB07, NB08, NB09 |
| **Phase 8** | NB11 | NB04, NB05, NB10 |
| **Phase 9** | NB12 | NB11 (all upstream) |

Within each phase, notebooks can be worked on in parallel (via subagent dispatch).

---

## 16. Quality Gates Per Notebook

Before marking any notebook "complete":

1. All code cells execute without errors
2. All statistical tests report: test statistic, p-value, confidence level, interpretation
3. All visualizations have: title, axis labels, legend, 300 DPI, consistent color palette
4. All sections end with interpretation markdown (≥2 sentences of business context)
5. Executive summary accurately reflects computed results
6. Output files saved to correct paths (data/processed/, data/features/, outputs/)
7. Cross-references to upstream/downstream notebooks are accurate

---

## 17. Success Criteria

The enhanced pipeline will be considered world-class when:

- Every metric claimed has a formal statistical test backing it
- Every visualization tells a story with interpretation
- A reader can open any single notebook and understand: what was done, why, what was found, and what it means
- The model hierarchy is honest: failures are documented as clearly as successes
- Portfolio recommendations are backed by evidence across multiple regimes and stress scenarios
- An admissions committee member can follow the analytical narrative from NB01 through NB12 without gaps
