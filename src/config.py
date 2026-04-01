"""
config.py — Centralized constants: tickers, dates, paths, sector groups.
============================================================================
Single source of truth for the entire pipeline.  Every notebook and src module
imports from here so that a single change propagates everywhere.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────
# 1.  PROJECT PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR  = DATA_DIR / "features"

MODELS_DIR    = PROJECT_ROOT / "models"
GARCH_DIR     = MODELS_DIR / "garch"
ML_DIR        = MODELS_DIR / "ml"
DL_DIR        = MODELS_DIR / "dl"
HMM_DIR       = MODELS_DIR / "hmm"

OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
TABLES_DIR    = OUTPUTS_DIR / "tables"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
REPORTS_DIR   = OUTPUTS_DIR / "reports"
PRES_DIR      = OUTPUTS_DIR / "presentations"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure all directories exist on import
for _d in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR,
           GARCH_DIR, ML_DIR, DL_DIR, HMM_DIR,
           TABLES_DIR, FIGURES_DIR, REPORTS_DIR, PRES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2.  TIME WINDOW
# ──────────────────────────────────────────────
START_DATE: str = "2016-03-01"
END_DATE:   str = "2026-03-13"          # update to latest available

# ──────────────────────────────────────────────
# 3.  STOCK UNIVERSE — 20 tickers × 10 sub-sectors
# ──────────────────────────────────────────────
TICKERS: List[str] = [
    "NVDA", "AVGO", "TSM", "SNPS",     # Semiconductors (design / foundry / EDA)
    "MSFT", "AMZN",                      # Cloud & Platforms
    "META", "GOOG",                      # Consumer Internet
    "AAPL",                              # Consumer Hardware
    "CRM", "NOW", "PLTR", "SAP",        # Enterprise Software
    "PANW", "CRWD", "DDOG",             # Cybersecurity
    "XYZ",                               # Fintech  (was SQ before 2025-01-21)
    "ANET",                              # Networking
    "MU", "AMD",                         # Memory + GPU compute
]

# Historical ticker for Block Inc (SQ → XYZ on 2025-01-21)
TICKER_SQ_LEGACY: str = "SQ"
TICKER_XYZ:       str = "XYZ"
SQ_XYZ_CUTOVER:   str = "2025-01-21"

# Tickers with truncated history (IPO / DPO after START_DATE)
SHORT_HISTORY_TICKERS: Dict[str, str] = {
    "CRWD": "2019-06-12",   # IPO date
    "DDOG": "2019-09-19",   # IPO date
    "PLTR": "2020-09-30",   # DPO date
}

# Minimum observations for FIGARCH estimation
MIN_OBS_FIGARCH: int = 1500

# ──────────────────────────────────────────────
# 4.  SECTOR CONSTRAINT GROUPS
# ──────────────────────────────────────────────
SECTOR_GROUPS: Dict[str, List[str]] = {
    "Semiconductors":       ["NVDA", "AVGO", "TSM", "SNPS", "MU", "AMD"],
    "Cloud & Platforms":    ["MSFT", "AMZN"],
    "Consumer Internet":    ["META", "GOOG"],
    "Consumer Hardware":    ["AAPL"],
    "Enterprise Software":  ["CRM", "NOW", "PLTR", "SAP"],
    "Cybersecurity":        ["PANW", "CRWD", "DDOG"],
    "Fintech":              ["XYZ"],
    "Networking":           ["ANET"],
}

MAX_SINGLE_STOCK_WEIGHT: float = 0.10      # 10 %
MAX_SECTOR_WEIGHT:       float = 0.30      # 30 %
MAX_MONTHLY_TURNOVER:    float = 0.20      # 20 %
TRANSACTION_COST_BPS:    float = 10.0      # 10 bps round-trip

# ──────────────────────────────────────────────
# 5.  BENCHMARKS & MACRO TICKERS
# ──────────────────────────────────────────────
SECTOR_BENCHMARKS: List[str] = ["XLK", "SMH", "HACK", "SKYY"]

MARKET_BENCHMARKS: List[str] = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DX-Y.NYB"]

BENCHMARK_TICKERS: List[str] = SECTOR_BENCHMARKS + MARKET_BENCHMARKS

VIX_TICKERS: List[str] = ["^VIX", "^VVIX"]

# FRED API series IDs
FRED_SERIES: Dict[str, str] = {
    "tbill_3m":    "DTB3",
    "yield_10y":   "DGS10",
    "yield_2y":    "DGS2",
    "cpi":         "CPIAUCSL",
    "pce":         "PCEPI",
    "fed_funds":   "FEDFUNDS",
    "mfg_employment": "MANEMP",
    "unemployment": "UNRATE",
    "gdp":         "GDP",
}

# Alternative risk-free from OpenBB
RF_TICKER_OBB: str = "^IRX"   # 13-week T-bill

# ──────────────────────────────────────────────
# 6.  WALK-FORWARD & MODELLING PARAMS
# ──────────────────────────────────────────────
TRAIN_RATIO:          float = 0.70   # first 70 % for initial training
RETRAIN_FREQ_DAYS:    int   = 63     # quarterly retraining (~63 trading days)
INNER_CV_SPLITS:      int   = 5      # TimeSeriesSplit folds inside walk-forward
RANDOM_STATE:         int   = 42

# GARCH defaults
GARCH_MAX_ITER:       int   = 1000

# LSTM / DL defaults
LSTM_LOOKBACK:        int   = 60     # 60-day input window
LSTM_HIDDEN_DIM:      int   = 128
LSTM_NUM_LAYERS:      int   = 2
LSTM_DROPOUT:         float = 0.3
EARLY_STOP_PATIENCE:  int   = 15

# Monte-Carlo (NB12)
MC_PATHS:             int   = 10_000

# ──────────────────────────────────────────────
# 7.  VOLATILITY HORIZONS (trading days)
# ──────────────────────────────────────────────
VOL_WINDOWS: List[int] = [5, 21, 63]       # 1-week, 1-month, 1-quarter

# ──────────────────────────────────────────────
# 8.  KEY EVENT DATES (for NB02 event studies)
# ──────────────────────────────────────────────
KEY_EVENTS: Dict[str, Tuple[str, str]] = {
    "COVID Crash":              ("2020-02-19", "2020-03-23"),
    "Fed Zero Rate":            ("2020-03-15", "2020-03-15"),
    "Inflation / Rate Hikes":   ("2022-01-03", "2022-10-13"),
    "ChatGPT / AI Rally":       ("2022-11-30", "2022-11-30"),
    "SVB Collapse":             ("2023-03-08", "2023-03-15"),
    "Chip Export Ban v1":       ("2022-10-07", "2022-10-07"),
    "Chip Export Ban v2":       ("2023-10-17", "2023-10-17"),
    "CrowdStrike Outage":      ("2024-07-19", "2024-07-19"),
    "DeepSeek / Tariff Shock":  ("2025-01-27", "2025-02-10"),
}

# ──────────────────────────────────────────────
# 9.  PRESENTATION / PLOTTING DEFAULTS
# ──────────────────────────────────────────────
FIG_DPI:     int = 300
FIG_FORMAT:  str = "png"
PALETTE:     str = "tab20"       # seaborn / matplotlib palette name
DECIMAL_RATIO:   int = 4         # decimal places for ratios
DECIMAL_PCT:     int = 2         # decimal places for percentages
DECIMAL_COUNT:   int = 0         # decimal places for counts

# ──────────────────────────────────────────────
# 10. OUTPUT FILE NAMES (canonical)
# ──────────────────────────────────────────────
# NB01
MASTER_DATA_FILE        = PROCESSED_DIR / "master_data.parquet"
# NB02
MACRO_REGIMES_FILE      = FEATURES_DIR  / "macro_regimes.parquet"
# NB03
GARCH_PARAMS_FILE       = TABLES_DIR    / "garch_parameters.csv"
COND_VOL_FILE           = FEATURES_DIR  / "conditional_vol_series.parquet"
# NB04
VAR_CVAR_FILE           = TABLES_DIR    / "var_cvar_table.csv"
EVT_PARAMS_FILE         = TABLES_DIR    / "evt_parameters.csv"
BACKTEST_VAR_FILE       = TABLES_DIR    / "backtest_results.csv"
RETURN_SCENARIOS_FILE   = FEATURES_DIR  / "return_scenarios.parquet"
# NB05
REGIME_LABELS_FILE      = FEATURES_DIR  / "regime_labels.parquet"
HMM_PARAMS_FILE         = HMM_DIR      / "hmm_model_params.pkl"
TRANSITION_MATRIX_FILE  = TABLES_DIR    / "transition_matrices.csv"
# NB06
SENTIMENT_FILE          = FEATURES_DIR  / "sentiment_features.parquet"
# NB07
VOL_FORECAST_FILE       = FEATURES_DIR  / "vol_forecast_predictions.parquet"
SHAP_VALUES_FILE        = ML_DIR        / "shap_values.pkl"
MODEL_COMPARISON_FILE   = TABLES_DIR    / "model_comparison_table.csv"
# NB08
RETURN_PRED_FILE        = FEATURES_DIR  / "return_predictions.parquet"
CLASS_REPORT_FILE       = TABLES_DIR    / "classification_report.csv"
STRATEGY_BACKTEST_FILE  = TABLES_DIR    / "strategy_backtest.csv"
# NB10
FINAL_COMPARISON_FILE   = TABLES_DIR    / "final_model_comparison.csv"
HYBRID_WEIGHTS_FILE     = ML_DIR        / "hybrid_weights.json"
# NB11
PORTFOLIO_WEIGHTS_FILE  = FEATURES_DIR  / "portfolio_weights_timeseries.parquet"
BACKTEST_PERF_FILE      = TABLES_DIR    / "backtest_performance.csv"
# NB12
STRESS_TEST_FILE        = TABLES_DIR    / "stress_test_results.csv"

# ──────────────────────────────────────────────
# 11. MACRO REGIME PERIODS (for NB02, NB05, NB11)
# ──────────────────────────────────────────────
MACRO_REGIME_PERIODS: Dict[str, Tuple[str, str]] = {
    "QE_ERA":              ("2016-03-01", "2018-09-30"),
    "RATE_HIKE_2018":      ("2018-10-01", "2019-06-30"),
    "PRE_COVID":           ("2019-07-01", "2020-02-18"),
    "COVID_CRASH":         ("2020-02-19", "2020-03-23"),
    "ZERO_RATE_RECOVERY":  ("2020-03-24", "2021-12-31"),
    "INFLATION_SHOCK":     ("2022-01-01", "2022-12-31"),
    "AI_RALLY":            ("2023-01-01", "2024-06-30"),
    "TARIFF_VOLATILITY":   ("2024-07-01", "2026-03-13"),
}

# ──────────────────────────────────────────────
# 12. SUB-SECTOR MAP (per-ticker sub-sector label)
# ──────────────────────────────────────────────
SUB_SECTOR_MAP: Dict[str, str] = {
    "NVDA": "GPU / AI Semiconductors",
    "AVGO": "Broadband / ASIC Semiconductors",
    "TSM":  "Foundry / Chip Manufacturing",
    "SNPS": "EDA / Semiconductor Tools",
    "MSFT": "Cloud / Enterprise Platforms",
    "META": "Consumer Internet / Advertising",
    "GOOG": "Search / AI Infrastructure",
    "AMZN": "E-Commerce / Cloud",
    "AAPL": "Consumer Hardware / Ecosystem",
    "CRM":  "Enterprise SaaS / CRM",
    "PANW": "Cybersecurity",
    "CRWD": "Endpoint Security",
    "DDOG": "Cloud Observability",
    "XYZ":  "Fintech / Payments",
    "NOW":  "Enterprise Workflow SaaS",
    "PLTR": "Data / AI Analytics",
    "ANET": "Networking / Data Center",
    "MU":   "Memory / Storage Semiconductors",
    "AMD":  "GPU / Discrete Compute",
    "SAP":  "ERP / Enterprise Software",
}

# ──────────────────────────────────────────────
# 13. VaR CONFIDENCE LEVELS
# ──────────────────────────────────────────────
VAR_ALPHAS: List[float]  = [0.05, 0.01]          # 95% and 99% VaR
EVT_ALPHAS: List[float]  = [0.005, 0.001]        # 99.5% and 99.9% for stress testing
