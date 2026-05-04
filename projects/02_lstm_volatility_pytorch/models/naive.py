"""Naive persistence baseline.

For each validation timestamp t with target horizon h, the prediction is the
last realized vol observed at-or-before time t (i.e. `vol_{h}d` in the feature
frame). This is the standard "the future will look like right now" baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_naive(
    df: pd.DataFrame,
    val_idx: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Naive forecast == most recent realized vol of matching window.

    `vol_{h}d` is constructed past-only in `features.engineer`, so reading it at
    timestamp t never leaks future information.
    """
    feature_col = f"vol_{horizon}d"
    if feature_col not in df.columns:
        raise KeyError(f"Missing column {feature_col} required by naive model.")
    return df.iloc[val_idx][feature_col].to_numpy(dtype=float)
