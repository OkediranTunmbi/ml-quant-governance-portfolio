"""LightGBM regressor for volatility forecasting + SHAP explanations."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

SEED = 42
DEFAULT_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> lgb.LGBMRegressor:
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    model = lgb.LGBMRegressor(**cfg)
    model.fit(X_train, y_train)
    return model


def predict_lgbm(model: lgb.LGBMRegressor, X_val: pd.DataFrame) -> np.ndarray:
    return model.predict(X_val)


def save_shap_summary(
    model: lgb.LGBMRegressor,
    X_sample: pd.DataFrame,
    out_path: Path,
    horizon: int,
    max_display: int = 10,
) -> None:
    """Save a SHAP bar-summary plot of the trained model on `X_sample`.

    Uses the `TreeExplainer` for fast exact SHAP values on tree models.
    `X_sample` should be a representative slice (e.g. the last training fold
    plus its validation set) to avoid a misleading single-fold view.
    """
    explainer = shap.TreeExplainer(model)
    # Use the modern Explanation API to avoid legacy summary_plot internals
    # that depend on NumPy's global RNG behavior.
    explanation = explainer(X_sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(f"LightGBM SHAP feature importance — horizon {horizon}d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
