"""PyTorch LSTM volatility forecaster.

Architecture:
- 2 LSTM layers, hidden size 64, dropout 0.2 between layers
- Input: a (seq_len=21, n_features) window of past features ending at time t
- Output: a single scalar prediction of the horizon-h log-vol target at t

Training:
- MSE loss in log-vol space (matches the project's target scale)
- Adam optimizer, early stopping on inner-validation loss
- Inner train/val split is chronological with the same `gap=horizon` rule
- StandardScaler is fit on inner-train features only and reused for val/test
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
SEQ_LEN = 21
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 8
LR = 1e-3


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VolLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take the last timestep representation; it summarizes the sequence.
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice (n_samples, n_features) into (n_windows, seq_len, n_features) and align targets.

    Returns (X_seq, y_seq, end_indices) where end_indices[i] is the original-row index
    that the i-th window's prediction corresponds to (i.e. the last row in the window).
    """
    n = len(X)
    if n < seq_len:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    n_windows = n - seq_len + 1
    X_seq = np.empty((n_windows, seq_len, X.shape[1]), dtype=np.float32)
    for i in range(n_windows):
        X_seq[i] = X[i : i + seq_len]
    y_seq = y[seq_len - 1 :].astype(np.float32)
    end_indices = np.arange(seq_len - 1, n, dtype=np.int64)
    return X_seq, y_seq, end_indices


@dataclass
class LSTMTrainResult:
    model: VolLSTM
    scaler: StandardScaler
    n_features: int
    best_val_loss: float
    epochs_trained: int


def _train_one(
    model: VolLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
) -> tuple[VolLSTM, float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state: dict | None = None
    epochs_since_improve = 0
    last_epoch = 0

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, last_epoch


def fit_predict_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    inner_train_idx_in_train: np.ndarray,
    inner_val_idx_in_train: np.ndarray,
    seq_len: int = SEQ_LEN,
    seed: int = SEED,
    hidden: int = HIDDEN,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Train the LSTM on the train fold and return predictions for the val fold.

    The caller supplies *positions within the training fold* for an inner train /
    inner validation split; this lets us do early stopping without leaking outer
    validation data. We also return the indices of the validation rows that
    actually receive a prediction (the first `seq_len-1` val rows are skipped
    because they don't yet have a complete past-window of features).

    Inputs are unscaled DataFrames; this function fits a StandardScaler on the
    inner-train rows only.
    """
    _set_seed(seed)

    feature_cols = list(X_train.columns)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)

    inner_train_X = X_train_np[inner_train_idx_in_train]
    scaler = StandardScaler().fit(inner_train_X)

    X_train_scaled = scaler.transform(X_train_np).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_np).astype(np.float32)

    # Build inner-train and inner-val sequences from the *scaled* train fold.
    inner_train_X_seq, inner_train_y_seq, _ = _build_sequences(
        X_train_scaled[inner_train_idx_in_train],
        y_train_np[inner_train_idx_in_train],
        seq_len,
    )
    inner_val_X_seq, inner_val_y_seq, _ = _build_sequences(
        X_train_scaled[inner_val_idx_in_train],
        y_train_np[inner_val_idx_in_train],
        seq_len,
    )

    if inner_train_X_seq.shape[0] < batch_size or inner_val_X_seq.shape[0] == 0:
        # Not enough history this fold; emit NaNs so downstream metrics skip cleanly.
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.from_numpy(inner_train_X_seq), torch.from_numpy(inner_train_y_seq))
    val_ds = TensorDataset(torch.from_numpy(inner_val_X_seq), torch.from_numpy(inner_val_y_seq))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VolLSTM(
        n_features=len(feature_cols),
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model, _, _ = _train_one(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
    )

    # For val predictions we need a window of past features ending at each val row.
    # Concatenate the tail of the train fold with the val fold so val rows have a
    # full `seq_len` look-back. Targets are taken from val only; the last
    # `len(X_val)` predictions correspond to val rows.
    n_val = X_val_scaled.shape[0]
    history = X_train_scaled[-(seq_len - 1):] if seq_len > 1 else np.empty((0, X_train_scaled.shape[1]), dtype=np.float32)
    combined = np.concatenate([history, X_val_scaled], axis=0)

    val_X_seq = np.empty((n_val, seq_len, X_val_scaled.shape[1]), dtype=np.float32)
    for i in range(n_val):
        val_X_seq[i] = combined[i : i + seq_len]

    model.eval()
    with torch.no_grad():
        preds = (
            model(torch.from_numpy(val_X_seq).to(device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    val_positions = np.arange(n_val, dtype=np.int64)
    return val_positions, preds
