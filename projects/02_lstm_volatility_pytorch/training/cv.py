"""Time-series cross-validation utilities.

Wraps `sklearn.model_selection.TimeSeriesSplit` so callers can pass a forecast
horizon and automatically get a `gap=horizon` to prevent leakage between train
and validation windows. The expanding-window behaviour is inherited directly
from `TimeSeriesSplit` (the train segment grows each fold).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_N_SPLITS = 5


@dataclass(frozen=True)
class FoldIndices:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_splits(
    n_samples: int,
    horizon: int,
    n_splits: int = DEFAULT_N_SPLITS,
) -> Iterator[FoldIndices]:
    """Yield (train_idx, val_idx) pairs with an `horizon`-day gap.

    The gap ensures that the last `horizon` observations of training data are
    excluded so their forward-looking targets cannot peek into the validation set.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    indices = np.arange(n_samples)
    for i, (train_idx, val_idx) in enumerate(splitter.split(indices)):
        yield FoldIndices(fold=i, train_idx=train_idx, val_idx=val_idx)


def chronological_train_val_split(
    train_idx: np.ndarray,
    val_fraction: float = 0.15,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into inner train / inner val for early stopping.

    Uses the *last* `val_fraction` of the fold as inner validation (chronological,
    never random). Drops `horizon` observations between inner train and inner val
    to mirror the outer leakage rule.
    """
    n = len(train_idx)
    if n < 10:
        raise ValueError(f"Train fold too small for inner split: n={n}")

    val_n = max(1, int(round(n * val_fraction)))
    val_start = n - val_n
    inner_train_end = max(0, val_start - horizon)
    inner_train = train_idx[:inner_train_end]
    inner_val = train_idx[val_start:]
    return inner_train, inner_val
