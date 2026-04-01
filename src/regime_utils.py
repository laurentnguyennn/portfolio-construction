"""
regime_utils.py — HMM fitting, Markov-Switching, regime labelling.
==================================================================
Fits 2-state and 3-state Gaussian HMMs using hmmlearn, extracts
transition matrices, stationary distributions, and regime-conditional
statistics.  Provides Viterbi decoding for most-likely-path analysis.

References
----------
* Baum & Welch (1970), Rabiner (1989) — HMM
* Hamilton (1989) — Markov-Switching
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.config import (
    HMM_DIR,
    HMM_PARAMS_FILE,
    RANDOM_STATE,
    REGIME_LABELS_FILE,
    TRANSITION_MATRIX_FILE,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1.  HMM FITTING
# ══════════════════════════════════════════════

def fit_hmm(
    returns: np.ndarray,
    n_states: int = 2,
    n_iter: int = 500,
    n_restarts: int = 25,
    covariance_type: str = "full",
    random_state: int = RANDOM_STATE,
) -> Tuple[GaussianHMM, float]:
    """
    Fit a Gaussian HMM with multiple random restarts.

    Multiple restarts mitigate local-optima issues in the EM algorithm.

    Parameters
    ----------
    returns : ndarray (T,) or (T, 1)
        Return series (univariate or multivariate).
    n_states : int
        Number of hidden states.
    n_restarts : int
        Number of random initialisations; best (highest log-likelihood) is kept.

    Returns
    -------
    best_model : GaussianHMM
    best_ll : float
        Log-likelihood of the best model.
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    best_model = None
    best_ll = -np.inf

    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state + i,
        )
        try:
            model.fit(returns)
            ll = model.score(returns)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception as exc:
            logger.debug("HMM restart %d failed: %s", i, exc)
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed")

    logger.info("Best HMM (%d states): log-likelihood=%.2f", n_states, best_ll)
    return best_model, best_ll


# ══════════════════════════════════════════════
# 2.  MODEL SELECTION (BIC)
# ══════════════════════════════════════════════

def hmm_bic(model: GaussianHMM, X: np.ndarray) -> float:
    """
    BIC = −2·log L + k·ln(T)

    where k = number of free parameters in the HMM:
      transition matrix:  n_states × (n_states − 1)     (rows sum to 1)
      means:              n_states × n_features
      covariances:        depends on covariance_type
    """
    T = X.shape[0]
    n = model.n_components
    n_feat = X.shape[1] if X.ndim > 1 else 1

    # Transition parameters
    k_trans = n * (n - 1)
    # Mean parameters
    k_mean = n * n_feat
    # Covariance parameters
    if model.covariance_type == "full":
        k_cov = n * n_feat * (n_feat + 1) // 2
    elif model.covariance_type == "diag":
        k_cov = n * n_feat
    elif model.covariance_type == "spherical":
        k_cov = n
    else:
        k_cov = n * n_feat

    k_initial = n - 1  # initial state distribution (sum-to-1 constraint)
    k = k_initial + k_trans + k_mean + k_cov
    ll = model.score(X)
    return -2 * ll + k * np.log(T)


def select_n_states(
    returns: np.ndarray,
    state_range: List[int] = [2, 3, 4],
    **fit_kwargs,
) -> Tuple[int, Dict[int, float]]:
    """
    Select optimal number of HMM states via BIC.

    Returns
    -------
    best_n : int
    bic_scores : dict {n_states: BIC}
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    bics = {}
    for n in state_range:
        model, _ = fit_hmm(returns, n_states=n, **fit_kwargs)
        bics[n] = hmm_bic(model, returns)
        logger.info("HMM %d states: BIC=%.2f", n, bics[n])

    best_n = min(bics, key=bics.get)
    return best_n, bics


# ══════════════════════════════════════════════
# 3.  VITERBI DECODING & REGIME LABELS
# ══════════════════════════════════════════════

def decode_regimes(
    model: GaussianHMM,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Viterbi-decode the most likely regime path and compute
    smoothed regime probabilities.

    Returns DataFrame with:
      - regime_state (int)
      - regime_prob_0, regime_prob_1, [regime_prob_2]
      - regime_duration (consecutive days in current regime)
      - regime_transition_signal (1 if regime changed in last 5 days)
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    states = model.predict(returns)
    probs = model.predict_proba(returns)

    df = pd.DataFrame(index=dates[:len(states)])
    df["regime_state"] = states

    for s in range(model.n_components):
        df[f"regime_prob_{s}"] = probs[:, s]

    # Duration: consecutive days in current regime (1-based: first day = 1)
    duration = np.ones(len(states), dtype=int)
    for t in range(1, len(states)):
        if states[t] == states[t - 1]:
            duration[t] = duration[t - 1] + 1
        else:
            duration[t] = 1
    df["regime_duration"] = duration

    # Transition signal: regime changed in last 5 days
    changes = np.zeros(len(states), dtype=int)
    for t in range(1, len(states)):
        if states[t] != states[t - 1]:
            changes[t] = 1
    df["regime_transition_signal"] = (
        pd.Series(changes, index=df.index).rolling(5, min_periods=1).max().astype(int)
    )

    return df


# ══════════════════════════════════════════════
# 4.  REGIME-CONDITIONAL STATISTICS
# ══════════════════════════════════════════════

def regime_conditional_stats(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Compute mean, vol, skew, kurtosis per regime.

    Parameters
    ----------
    returns : DataFrame (T × N)
    regime_labels : Series of int regime states

    Returns
    -------
    DataFrame with multi-level index (regime, statistic) × tickers.
    """
    rows = []
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        sub = returns.loc[mask]
        for col in sub.columns:
            s = sub[col].dropna()
            rows.append({
                "regime": regime,
                "ticker": col,
                "mean_daily": s.mean(),
                "annualized_return": s.mean() * 252,
                "annualized_vol": s.std() * np.sqrt(252),
                "skewness": s.skew(),
                "excess_kurtosis": s.kurtosis(),
                "n_days": len(s),
            })
    return pd.DataFrame(rows)


def regime_conditional_correlation(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> Dict[int, pd.DataFrame]:
    """Separate correlation matrices per regime."""
    corr_dict = {}
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        corr_dict[regime] = returns.loc[mask].corr()
    return corr_dict


# ══════════════════════════════════════════════
# 5.  TRANSITION MATRIX ANALYSIS
# ══════════════════════════════════════════════

def extract_transition_info(model: GaussianHMM) -> Dict:
    """
    Extract transition matrix, stationary distribution, and
    expected durations from a fitted HMM.

    Stationary distribution π satisfies π = π · A.
    Expected duration in state i = 1 / (1 − A_{ii}).
    """
    A = model.transmat_

    # Stationary distribution: left eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)  # eigenvector can have arbitrary sign
    pi = pi / pi.sum()

    # Expected durations (guard against absorbing states where A_ii = 1)
    diag_A = np.diag(A)
    durations = np.where(diag_A < 1.0 - 1e-10, 1.0 / (1.0 - diag_A), np.inf)

    return {
        "transition_matrix": A,
        "stationary_distribution": pi,
        "expected_durations": durations,
        "means": model.means_.flatten(),
        "variances": np.array([np.diag(model.covars_[i]).sum() if model.covars_[i].ndim > 1
                                else float(model.covars_[i])
                                for i in range(model.n_components)]),
    }


# ══════════════════════════════════════════════
# 6.  LABEL ORDERING (ensure state 0 = bear, last = bull)
# ══════════════════════════════════════════════

def order_states_by_mean(
    model: GaussianHMM,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-label states so that state 0 has the lowest mean (bear)
    and the highest-numbered state has the highest mean (bull).

    This ensures consistent labelling across different HMM fits.
    """
    means = model.means_.flatten()
    order = np.argsort(means)  # ascending: bear first
    mapping = {old: new for new, old in enumerate(order)}

    regime_df = regime_df.copy()
    regime_df["regime_state"] = regime_df["regime_state"].map(mapping)

    # Re-order probability columns
    n_states = model.n_components
    prob_cols = [f"regime_prob_{i}" for i in range(n_states)]
    new_probs = regime_df[prob_cols].values[:, order]
    for i in range(n_states):
        regime_df[f"regime_prob_{i}"] = new_probs[:, i]

    return regime_df


# ══════════════════════════════════════════════
# 7.  SAVE / LOAD
# ══════════════════════════════════════════════

def save_hmm_model(model: GaussianHMM, name: str = "sector_hmm"):
    """Pickle the HMM model."""
    path = HMM_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved HMM model to %s", path)


def load_hmm_model(name: str = "sector_hmm") -> GaussianHMM:
    """Load a pickled HMM model."""
    path = HMM_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
