"""
Tech Sector Risk Management & ML Forecasting Pipeline
=====================================================
Source modules for the 12-notebook quantitative analysis pipeline.

Modules
-------
config               : Centralized constants (tickers, dates, paths, parameters)
data_loader          : OpenBB/FRED download, SQ→XYZ merge, quality checks, caching
feature_engineering  : Returns, volatility estimators, momentum, volume, cross-asset, stationarity
garch_utils          : GARCH/GJR/EGARCH/FIGARCH fitting, model selection, conditional vol extraction
dcc_garch            : DCC-GARCH two-step procedure (Engle 2002)
risk_metrics         : VaR (5 methods), CVaR, EVT/GPD, copulas, CAViaR, backtesting
regime_utils         : HMM fitting, Viterbi decoding, regime features, regime-conditional stats
ml_pipeline          : Walk-forward evaluation, Optuna HPO, Diebold-Mariano, Model Confidence Set
dl_models            : LSTM, GRU architectures, training loops with early stopping
portfolio_optimizer  : Markowitz, Mean-CVaR, Black-Litterman, HRP, ERC, constraint enforcement
backtest_engine      : Walk-forward portfolio backtest, performance metrics, benchmark comparison
visualization        : Publication-quality plotting (cumulative returns, heatmaps, regimes, SHAP)
statistical_tests    : BH-FDR correction, Hurst exponent, SPA test, stationary block bootstrap
systemic_risk        : CoVaR, Marginal Expected Shortfall, Absorption Ratio
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Laurent"
