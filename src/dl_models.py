"""
dl_models.py — LSTM, GRU, Temporal Fusion Transformer wrappers.
================================================================
PyTorch deep-learning models for multi-horizon volatility and
return forecasting.  Includes sequence creation, training loops
with early stopping, and evaluation helpers.

References
----------
* Hochreiter & Schmidhuber (1997) — LSTM
* Cho et al. (2014) — GRU
* Lim et al. (2021) — Temporal Fusion Transformer
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DL_DIR,
    EARLY_STOP_PATIENCE,
    LSTM_DROPOUT,
    LSTM_HIDDEN_DIM,
    LSTM_LOOKBACK,
    LSTM_NUM_LAYERS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 1.  SEQUENCE CREATION
# ══════════════════════════════════════════════

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int = LSTM_LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) sequences for LSTM/GRU.

    X[i] = features[i : i + lookback]     shape (lookback, n_features)
    y[i] = targets[i + lookback]           shape (1,) or scalar

    Parameters
    ----------
    features : ndarray (T, n_features)
    targets : ndarray (T,)
    lookback : int
        Number of past time steps in each input window.

    Returns
    -------
    X : ndarray (N, lookback, n_features)
    y : ndarray (N,)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


# ══════════════════════════════════════════════
# 2.  LSTM MODEL
# ══════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn

    class LSTMForecaster(nn.Module):
        """
        2-layer stacked LSTM → Dropout → Dense(1).

        Note: PyTorch LSTM's ``dropout`` parameter only applies BETWEEN
        layers, not after the last layer.  We add an explicit nn.Dropout
        before the fully connected layer (per CLAUDE.md §NB09).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = LSTM_HIDDEN_DIM,
            num_layers: int = LSTM_NUM_LAYERS,
            dropout: float = LSTM_DROPOUT,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            # Explicit dropout after the last LSTM layer
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, (h_n, c_n) = self.lstm(x)
            out = self.dropout(out[:, -1, :])  # last time step
            return self.fc(out).squeeze(-1)

    # ══════════════════════════════════════════
    # 3.  GRU MODEL
    # ══════════════════════════════════════════

    class GRUForecaster(nn.Module):
        """
        2-layer stacked GRU → Dropout → Dense(1).

        Fewer parameters than LSTM (no cell state), potentially
        faster convergence.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = LSTM_HIDDEN_DIM,
            num_layers: int = LSTM_NUM_LAYERS,
            dropout: float = LSTM_DROPOUT,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden_dim, num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, h_n = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    # ══════════════════════════════════════════
    # 4.  TRAINING LOOP WITH EARLY STOPPING
    # ══════════════════════════════════════════

    class EarlyStopping:
        """Early stopping monitor."""

        def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-6):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = np.inf
            self.should_stop = False

        def step(self, val_loss: float) -> bool:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            return self.should_stop

    def train_model(
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = EARLY_STOP_PATIENCE,
        device: str = "cpu",
    ) -> Dict:
        """
        Train a PyTorch model with early stopping.

        Returns dict with training history (train_loss, val_loss per epoch).
        """
        torch.manual_seed(RANDOM_STATE)
        model = model.to(device)

        X_tr = torch.FloatTensor(X_train).to(device)
        y_tr = torch.FloatTensor(y_train).to(device)
        X_v = torch.FloatTensor(X_val).to(device)
        y_v = torch.FloatTensor(y_val).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=patience)
        best_state_dict = None

        history = {"train_loss": [], "val_loss": [], "stopped_epoch": epochs}

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_tr), batch_size):
                X_batch = X_tr[i:i + batch_size]
                y_batch = y_tr[i:i + batch_size]

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train = total_loss / max(n_batches, 1)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, y_v).item()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)

            # Save best model weights
            if val_loss < early_stop.best_loss:
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())

            if early_stop.step(val_loss):
                history["stopped_epoch"] = epoch + 1
                logger.info("Early stopping at epoch %d (val_loss=%.6f)",
                            epoch + 1, val_loss)
                break

        # Restore best model weights
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return history

    def predict_model(
        model: nn.Module,
        X: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference and return numpy predictions."""
        model.eval()
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        return preds

except ImportError:
    logger.warning("PyTorch not installed — DL models unavailable")

    class LSTMForecaster:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for LSTMForecaster")

    class GRUForecaster:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for GRUForecaster")
