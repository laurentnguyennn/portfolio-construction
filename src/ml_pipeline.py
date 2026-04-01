"""
ml_pipeline.py — Sklearn/XGBoost/LightGBM pipeline builders.
=============================================================
Walk-forward expanding-window evaluation, Optuna hyperparameter
optimization, Diebold-Mariano tests, Mincer-Zarnowitz regression.

Walk-Forward Protocol (CLAUDE.md §NB07-08)
------------------------------------------
  • Expanding window, retrain every 63 trading days (quarterly).
  • Initial training window: first 70 % of data.
  • ALL ML/DL models use the same protocol for valid comparison.
  • Predictions are genuinely out-of-sample at each point in time.

Pipeline Hygiene
----------------
  • StandardScaler INSIDE the pipeline (no data leakage).
  • SMOTE only for classification tasks (NB08), never for regression (NB07).
  • TimeSeriesSplit for inner CV within each walk-forward training window.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score,
    roc_auc_score, brier_score_loss,
)

from src.config import (
    INNER_CV_SPLITS,
    RANDOM_STATE,
    RETRAIN_FREQ_DAYS,
    TRAIN_RATIO,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  PIPELINE FACTORIES
# ══════════════════════════════════════════════

def make_regression_pipeline(model_name: str, params: Dict | None = None) -> Pipeline:
    """
    Build a regression pipeline: StandardScaler → Model.

    No SMOTE — regression tasks do not need class balancing.
    """
    params = params or {}

    if model_name == "ridge":
        model = Ridge(**params)
    elif model_name == "lasso":
        model = Lasso(max_iter=5000, **params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, **params
        )
    elif model_name == "knn":
        model = KNeighborsRegressor(**params)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", **params
        )
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params
        )
    else:
        raise ValueError(f"Unknown regression model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_classification_pipeline(
    model_name: str,
    params: Dict | None = None,
    use_smote: bool = True,
) -> Pipeline:
    """
    Build a classification pipeline.

    If ``use_smote=True``, uses imblearn.pipeline.Pipeline with SMOTE
    between scaler and model.  SMOTE is appropriate for NB08's
    return-direction classification only.
    """
    params = params or {}

    if model_name == "logistic":
        model = LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000, **params
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, **params
        )
    elif model_name == "svm":
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **params)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
            tree_method="hist", **params
        )
    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    if use_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        # SMOTE is applied after train/test split (inside the pipeline), so it
        # only synthesises samples within the training fold.  In time-series
        # walk-forward evaluation verify class imbalance justifies SMOTE:
        # synthetic samples interpolate between existing training observations,
        # which can blend information from different time periods.  If the
        # imbalance is mild (< 60/40 split), set use_smote=False.
        logger.debug(
            "Classification pipeline built WITH SMOTE. "
            "Verify class imbalance warrants synthetic oversampling in your "
            "time-series context before using walk-forward evaluation."
        )
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ])
    else:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])


# ══════════════════════════════════════════════
# 2.  WALK-FORWARD EXPANDING-WINDOW ENGINE
# ══════════════════════════════════════════════

def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline_factory: Callable,
    retrain_freq: int = RETRAIN_FREQ_DAYS,
    initial_train_ratio: float = TRAIN_RATIO,
    task: str = "regression",
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward expanding-window evaluation with embargo/purge.

    1. Train on the first ``initial_train_ratio`` of data.
    2. Predict the next ``retrain_freq`` days.
    3. Expand training window by ``retrain_freq`` days, retrain.
    4. Repeat until end.

    Parameters
    ----------
    X : DataFrame (T × features)
    y : Series (T,)
    pipeline_factory : callable
        Function that returns a fresh (unfitted) Pipeline.
    retrain_freq : int
        Number of trading days between retrains.
    initial_train_ratio : float
        Fraction of data for the initial training window.
    task : str
        'regression' or 'classification'.
    horizon : int
        Forecast horizon in days. Used for purge/embargo to prevent
        label leakage at train/test boundaries (Lopez de Prado, 2018).

    Returns
    -------
    predictions : DataFrame
        Columns: date, y_true, y_pred [, y_prob for classification].
    """
    T = len(X)
    initial_train_end = int(T * initial_train_ratio)

    results = []
    train_end = initial_train_end

    while train_end < T:
        test_end = min(train_end + retrain_freq, T)

        # Purge: remove last `horizon` training obs (labels overlap test period)
        purge_end = max(train_end - horizon, 0)
        X_train = X.iloc[:purge_end]
        y_train = y.iloc[:purge_end]
        # Embargo: skip first `horizon` test obs after train boundary
        embargo_start = min(train_end + horizon, T)
        X_test = X.iloc[embargo_start:test_end]
        y_test = y.iloc[embargo_start:test_end]

        # Drop NaN rows
        valid_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        # Align X_test and y_test on the same index before filtering NaNs
        # so that .values arrays stay in sync (prevents silent row misalignment).
        test_df = pd.DataFrame({"__y__": y_test}).join(X_test, how="inner")
        test_df = test_df.dropna()
        X_test = test_df.drop(columns="__y__")
        y_test = test_df["__y__"]

        if len(X_train) < 50 or len(X_test) == 0:
            train_end = test_end
            continue

        pipe = pipeline_factory()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # X_test and y_test share the same index after alignment above
        batch = pd.DataFrame({
            "date": X_test.index,
            "y_true": y_test.values,
            "y_pred": preds,
        })

        if task == "classification" and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)[:, 1]
                batch["y_prob"] = proba
            except Exception:
                pass

        results.append(batch)
        train_end = test_end

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ══════════════════════════════════════════════
# 3.  OPTUNA HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════

def optuna_xgboost_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    n_splits: int = INNER_CV_SPLITS,
) -> Dict:
    """
    Optuna Bayesian optimization for XGBoost regressor.

    Inner CV uses TimeSeriesSplit to respect temporal ordering.
    """
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(
                    random_state=RANDOM_STATE, n_jobs=-1,
                    tree_method="hist", **params
                )),
            ])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best Optuna RMSE: %.6f", study.best_value)
    return study.best_params


# ══════════════════════════════════════════════
# 4.  EVALUATION METRICS
# ══════════════════════════════════════════════

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """RMSE, MAE, MAPE, Directional Accuracy."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE with protection against zero values
    nonzero = np.abs(y_true) > 1e-10
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = np.nan

    # Directional accuracy
    if len(y_true) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        da = (true_dir == pred_dir).mean() * 100
    else:
        da = np.nan

    # QLIKE loss (preferred for volatility forecasting — Patton, 2011)
    # Inputs are annualized volatilities (σ); convert to variance (σ²) for QLIKE.
    # QLIKE = mean( ln(σ̂²_t) + σ²_t / σ̂²_t )
    if np.all(y_pred > 0) and np.all(y_true > 0):
        h_pred = y_pred ** 2
        h_true = y_true ** 2
        qlike = np.mean(np.log(h_pred) + h_true / h_pred)
    else:
        qlike = np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "directional_accuracy": da, "qlike": qlike}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """Accuracy, Precision, Recall, F1, AUC-ROC, Brier Score."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            m["auc_roc"] = roc_auc_score(y_true, y_prob)
            m["brier_score"] = brier_score_loss(y_true, y_prob)
        except ValueError:
            m["auc_roc"] = np.nan
            m["brier_score"] = np.nan
    return m


# ══════════════════════════════════════════════
# 5.  DIEBOLD-MARIANO TEST
# ══════════════════════════════════════════════

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss_fn: str = "squared",
) -> Dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[d_t] = 0  where d_t = L(e1_t) − L(e2_t).

    For multi-step forecasts (h > 1), uses HAC variance estimator
    with bandwidth h-1 to account for overlapping forecast windows.

    Parameters
    ----------
    e1, e2 : arrays of forecast errors from model 1 and model 2.
    h : int
        Forecast horizon.  For h > 1, applies Newey-West HAC correction.
    loss_fn : str
        'squared' for MSE-based, 'absolute' for MAE-based.

    Returns
    -------
    dict with dm_stat, p_value.
    """
    if loss_fn == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss_fn == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    T = len(d)
    d_bar = d.mean()

    # HAC variance estimation (Newey-West)
    bandwidth = max(h - 1, 0)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1 - k / (bandwidth + 1)   # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {"dm_stat": dm_stat, "p_value": p_value}


# ══════════════════════════════════════════════
# 6.  MINCER-ZARNOWITZ REGRESSION
# ══════════════════════════════════════════════

def mincer_zarnowitz_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    h: int = 1,
) -> Dict:
    """
    Mincer-Zarnowitz (1969) forecast efficiency test.

    Regress actual on forecast: y_t = α + β · ŷ_t + ε_t

    Joint H0: α = 0 and β = 1  (efficient forecast).
    Uses Newey-West HAC standard errors for h > 1.

    Returns
    -------
    dict with alpha, beta, f_stat, f_pvalue, r_squared.
    """
    import statsmodels.api as sm

    X = sm.add_constant(y_pred)

    if h > 1:
        model = sm.OLS(y_true, X).fit(cov_type="HAC",
                                        cov_kwds={"maxlags": h - 1})
    else:
        model = sm.OLS(y_true, X).fit()

    alpha = model.params[0]
    beta = model.params[1]

    # Joint F-test: α=0, β=1
    # Uses (R, q) form: R @ params = q  =>  [[1,0],[0,1]] @ [α,β] = [0,1]
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    f_test = model.f_test((R, q))

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_pvalue": model.pvalues[0],
        "beta_pvalue": model.pvalues[1],
        "f_stat": float(f_test.fvalue),
        "f_pvalue": float(f_test.pvalue),
        "r_squared": model.rsquared,
    }


# ══════════════════════════════════════════════
# 7.  STACKING META-LEARNER
# ══════════════════════════════════════════════

def build_stacking_ensemble(
    base_predictions: pd.DataFrame,
    y_true: pd.Series,
) -> Pipeline:
    """
    Train a Ridge meta-learner on base model predictions.

    Parameters
    ----------
    base_predictions : DataFrame
        Each column is a base model's out-of-sample predictions.
    y_true : Series
        True target values aligned with base_predictions.

    Returns
    -------
    Fitted Pipeline (scaler + Ridge).
    """
    valid = ~(base_predictions.isna().any(axis=1) | y_true.isna())
    X = base_predictions.loc[valid]
    y = y_true.loc[valid]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("meta", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


# ══════════════════════════════════════════════
# 8.  MODEL CONFIDENCE SET (Hansen, Lunde & Nason 2011)
# ══════════════════════════════════════════════

def model_confidence_set(
    errors: Dict[str, np.ndarray],
    alpha: float = 0.10,
    loss_fn: str = "squared",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Model Confidence Set (MCS) — Hansen, Lunde & Nason (2011), *Econometrica*.

    Sequentially eliminates the worst-performing model using pairwise
    Diebold-Mariano tests until no model is significantly outperformed
    at level ``alpha``.

    The surviving set M* contains all models whose forecasting ability
    is not statistically distinguishable from the best model.

    Parameters
    ----------
    errors : dict {model_name: array of forecast errors}
    alpha : float
        Significance level for elimination (default 0.10 as in original paper).
    loss_fn : str
        'squared' for MSE-based, 'absolute' for MAE-based.

    Returns
    -------
    mcs_set : list of str
        Model names in the surviving confidence set.
    elimination_log : DataFrame
        Log of eliminated models with test statistics and p-values.
    """
    remaining = list(errors.keys())
    log_rows = []

    while len(remaining) > 1:
        # For each model, count how many times it is significantly
        # outperformed (dm_stat > 0 means model m has higher loss,
        # i.e., model m is worse than the other model).
        outperformed_counts = {}

        for m in remaining:
            count = 0
            min_pvalue = 1.0
            for other in remaining:
                if m == other:
                    continue
                dm = diebold_mariano_test(errors[m], errors[other], h=1, loss_fn=loss_fn)
                p = dm["p_value"] if not np.isnan(dm["p_value"]) else 1.0
                stat = dm["dm_stat"] if not np.isnan(dm["dm_stat"]) else 0.0
                # dm_stat > 0 means L(m) > L(other), i.e., m is worse
                if stat > 0 and p < alpha:
                    count += 1
                    min_pvalue = min(min_pvalue, p)
            outperformed_counts[m] = (count, -min_pvalue)  # sort by count desc, then lowest p

        # The model most frequently outperformed is eliminated
        worst_model = max(outperformed_counts, key=outperformed_counts.get)
        worst_count, neg_worst_p = outperformed_counts[worst_model]

        if worst_count > 0:
            log_rows.append({
                "eliminated": worst_model,
                "times_outperformed": worst_count,
                "min_dm_pvalue": -neg_worst_p,
                "step": len(log_rows) + 1,
            })
            remaining.remove(worst_model)
        else:
            break  # No model is significantly outperformed

    log_df = pd.DataFrame(log_rows) if log_rows else pd.DataFrame()
    return remaining, log_df


# ══════════════════════════════════════════════
# 9.  FEATURE ABLATION STUDY
# ══════════════════════════════════════════════

def feature_ablation_study(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline_factory: Callable,
    feature_groups: Dict[str, List[str]],
    retrain_freq: int = RETRAIN_FREQ_DAYS,
    initial_train_ratio: float = TRAIN_RATIO,
) -> pd.DataFrame:
    """
    Measure RMSE degradation when each feature group is removed.

    For each group, drops its columns from X, runs walk-forward
    evaluation, and compares against the full-feature baseline.

    Parameters
    ----------
    feature_groups : dict {group_name: list of column names}
        E.g., {"momentum": ["rsi_14", "macd_line", "bollinger_pctb"],
               "volatility": ["realized_vol_21d", "parkinson_vol_21d"]}

    Returns
    -------
    DataFrame with columns: group, rmse_full, rmse_without, rmse_change_pct.
    """
    # Full model baseline
    preds_full = walk_forward_predict(
        X, y, pipeline_factory, retrain_freq, initial_train_ratio
    )
    if preds_full.empty:
        return pd.DataFrame()
    rmse_full = np.sqrt(mean_squared_error(preds_full["y_true"], preds_full["y_pred"]))

    rows = []
    for group_name, cols in feature_groups.items():
        cols_to_drop = [c for c in cols if c in X.columns]
        if not cols_to_drop:
            continue
        X_ablated = X.drop(columns=cols_to_drop)
        preds = walk_forward_predict(
            X_ablated, y, pipeline_factory, retrain_freq, initial_train_ratio
        )
        if preds.empty:
            continue
        rmse_without = np.sqrt(mean_squared_error(preds["y_true"], preds["y_pred"]))
        rows.append({
            "group": group_name,
            "n_features_removed": len(cols_to_drop),
            "rmse_full": rmse_full,
            "rmse_without": rmse_without,
            "rmse_change_pct": (rmse_without - rmse_full) / rmse_full * 100,
        })

    return pd.DataFrame(rows).sort_values("rmse_change_pct", ascending=False)


# ══════════════════════════════════════════════
# 10. LEARNING CURVE
# ══════════════════════════════════════════════

def compute_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline_factory: Callable,
    fractions: List[float] | None = None,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Compute training and validation RMSE at increasing training set sizes.

    Helps diagnose overfitting (train ≪ val) vs. underfitting (both high).

    Parameters
    ----------
    fractions : list of float
        Fractions of training data to use (default [0.1, 0.2, ..., 1.0]).
    val_ratio : float
        Last fraction of data held out for validation.

    Returns
    -------
    DataFrame with columns: train_fraction, n_samples, train_rmse, val_rmse.
    """
    fractions = fractions or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Split into train pool and validation
    split = int(len(X) * (1 - val_ratio))
    X_pool, y_pool = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    # Drop NaN
    valid_val = ~(X_val.isna().any(axis=1) | y_val.isna())
    X_val = X_val.loc[valid_val]
    y_val = y_val.loc[valid_val]

    rows = []
    for frac in fractions:
        n = max(int(len(X_pool) * frac), 50)
        X_sub = X_pool.iloc[:n]
        y_sub = y_pool.iloc[:n]

        valid = ~(X_sub.isna().any(axis=1) | y_sub.isna())
        X_sub = X_sub.loc[valid]
        y_sub = y_sub.loc[valid]

        if len(X_sub) < 30 or len(X_val) < 10:
            continue

        pipe = pipeline_factory()
        pipe.fit(X_sub, y_sub)
        train_rmse = np.sqrt(mean_squared_error(y_sub, pipe.predict(X_sub)))
        val_rmse = np.sqrt(mean_squared_error(y_val, pipe.predict(X_val)))

        rows.append({
            "train_fraction": frac,
            "n_samples": len(X_sub),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 11. OPTUNA LIGHTGBM
# ══════════════════════════════════════════════

def optuna_lightgbm_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    n_splits: int = INNER_CV_SPLITS,
) -> Dict:
    """
    Optuna Bayesian optimization for LightGBM regressor.

    Inner CV uses TimeSeriesSplit to respect temporal ordering.
    """
    import optuna
    from lightgbm import LGBMRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LGBMRegressor(
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params
                )),
            ])
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Best Optuna LightGBM RMSE: %.6f", study.best_value)
    return study.best_params


# ══════════════════════════════════════════════
# 12. CLASSIFICATION THRESHOLD OPTIMIZATION
# ══════════════════════════════════════════════

def optimize_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
    forward_returns: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Optimize the classification decision threshold.

    Instead of using the default 0.5 cutoff, search for the threshold
    that maximizes the chosen metric.

    Parameters
    ----------
    y_true : array of binary labels (0/1)
    y_prob : array of predicted probabilities
    strategy : str
        'f1' — maximize F1 score
        'sharpe' — maximize Sharpe ratio of a long/flat strategy
            (long when prob > threshold, flat otherwise).
            Requires ``forward_returns``.
    forward_returns : array, optional
        Actual forward returns (not labels). Required for strategy='sharpe'.

    Returns
    -------
    best_threshold : float
    best_metric : float
    """
    thresholds = np.linspace(0.30, 0.70, 41)
    best_threshold = 0.5
    best_metric = -np.inf

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if strategy == "f1":
            metric = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == "sharpe":
            if forward_returns is None:
                raise ValueError("strategy='sharpe' requires forward_returns")
            # Long/flat: earn actual return when predicted up, 0 when flat
            signal_returns = forward_returns * y_pred
            if np.std(signal_returns) > 0:
                metric = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252)
            else:
                metric = 0.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if metric > best_metric:
            best_metric = metric
            best_threshold = t

    return best_threshold, best_metric


# ══════════════════════════════════════════════
# FEATURE SELECTION UTILITIES
# ══════════════════════════════════════════════

def vif_filter(
    X: pd.DataFrame,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Remove features with Variance Inflation Factor > threshold.

    VIF_j = 1 / (1 - R²_j) where R²_j is from regressing feature j
    on all other features. VIF > 10 indicates severe multicollinearity.

    Parameters
    ----------
    X : DataFrame (T, K) of features.
    threshold : float — VIF threshold (default 10.0).

    Returns
    -------
    X_filtered : DataFrame with collinear features removed.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_clean = X.dropna()
    if len(X_clean) < X_clean.shape[1] + 2:
        logger.warning("Insufficient observations for VIF; returning all features")
        return X

    cols = list(X_clean.columns)
    dropped = []

    while True:
        X_arr = X_clean[cols].values
        if X_arr.shape[1] <= 1:
            break

        vifs = np.array([
            variance_inflation_factor(X_arr, i) for i in range(X_arr.shape[1])
        ])

        max_vif_idx = np.argmax(vifs)
        max_vif = vifs[max_vif_idx]

        if max_vif > threshold:
            dropped_col = cols[max_vif_idx]
            dropped.append(dropped_col)
            cols.remove(dropped_col)
            logger.info("VIF filter: dropped %s (VIF=%.1f)", dropped_col, max_vif)
        else:
            break

    if dropped:
        logger.info("VIF filter removed %d features: %s", len(dropped), dropped)

    return X[cols]


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 100,
    threshold: float = 0.6,
    alpha_range: Tuple = (0.001, 0.1),
    seed: int = RANDOM_STATE,
    random_state: int | None = None,
) -> List[str]:
    """
    Stability Selection (Meinshausen & Bühlmann, 2010).

    Runs Lasso on 100 random 50% subsamples of the data. Features
    selected in more than ``threshold`` fraction of runs are retained.

    This provides robust feature selection that is less sensitive to
    the choice of regularization parameter than a single Lasso run.

    Parameters
    ----------
    X : DataFrame (T, K) of features.
    y : Series (T,) of targets.
    n_bootstrap : int — number of random subsamples.
    threshold : float — selection frequency threshold (default 0.6).

    Returns
    -------
    selected : list of str — selected feature names.
    """
    rng = np.random.RandomState(random_state if random_state is not None else seed)
    feature_names = list(X.columns)
    K = len(feature_names)
    selection_count = np.zeros(K)

    X_vals = X.values
    y_vals = y.values
    T = len(y_vals)

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    for b in range(n_bootstrap):
        # Contiguous 50% subsample: random start point, preserves temporal ordering
        # (random index permutation would introduce lookahead bias in time series)
        start = rng.randint(0, T // 2 + 1)
        idx = np.arange(start, start + T // 2)
        X_sub = scaler.fit_transform(X_vals[idx])
        y_sub = y_vals[idx]

        # Random alpha from range (log-uniform)
        alpha = np.exp(rng.uniform(
            np.log(alpha_range[0]), np.log(alpha_range[1])
        ))

        lasso = Lasso(alpha=alpha, max_iter=5000, random_state=random_state if random_state is not None else seed)
        try:
            lasso.fit(X_sub, y_sub)
            selected_mask = np.abs(lasso.coef_) > 1e-6
            selection_count += selected_mask
        except Exception:
            continue

    selection_freq = selection_count / n_bootstrap
    selected = [
        name for name, freq in zip(feature_names, selection_freq)
        if freq >= threshold
    ]

    logger.info(
        "Stability selection: %d/%d features selected (threshold=%.2f)",
        len(selected), K, threshold
    )
    return selected


def feature_selection_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline_factory: Callable,
) -> pd.DataFrame:
    """
    Compare full vs VIF-filtered vs stability-selected feature sets.

    Trains the model on each feature subset and reports OOS RMSE.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    pipeline_factory : callable returning a fitted pipeline

    Returns
    -------
    DataFrame with columns: feature_set, n_features, rmse_oos
    """
    results = []

    # Full feature set
    pipe_full = pipeline_factory()
    pipe_full.fit(X_train, y_train)
    pred_full = pipe_full.predict(X_test)
    rmse_full = np.sqrt(mean_squared_error(y_test, pred_full))
    results.append({
        "feature_set": "full",
        "n_features": X_train.shape[1],
        "rmse_oos": rmse_full,
    })

    # VIF-filtered
    X_train_vif = vif_filter(X_train, threshold=10.0)
    X_test_vif = X_test[X_train_vif.columns]
    pipe_vif = pipeline_factory()
    pipe_vif.fit(X_train_vif, y_train)
    pred_vif = pipe_vif.predict(X_test_vif)
    rmse_vif = np.sqrt(mean_squared_error(y_test, pred_vif))
    results.append({
        "feature_set": "vif_filtered",
        "n_features": X_train_vif.shape[1],
        "rmse_oos": rmse_vif,
    })

    # Stability-selected
    selected = stability_selection(X_train, y_train)
    if len(selected) > 0:
        X_train_ss = X_train[selected]
        X_test_ss = X_test[selected]
        pipe_ss = pipeline_factory()
        pipe_ss.fit(X_train_ss, y_train)
        pred_ss = pipe_ss.predict(X_test_ss)
        rmse_ss = np.sqrt(mean_squared_error(y_test, pred_ss))
    else:
        rmse_ss = np.nan
    results.append({
        "feature_set": "stability_selected",
        "n_features": len(selected),
        "rmse_oos": rmse_ss,
    })

    return pd.DataFrame(results)
