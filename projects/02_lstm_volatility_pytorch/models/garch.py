"""GARCH(1,1) volatility forecasts (rolled per fold).

For each validation timestamp we use a single fit on all training-fold returns
(plus everything observed up to t) and then forecast `horizon` steps ahead.
The aggregated forecast variance over the horizon is converted to annualized
log-vol so it is directly comparable to the project-wide target.

Notes:
- We use `arch_model(returns * 100)` because the `arch` library is numerically
  better-behaved with percentage returns; we rescale on the way out.
- Re-fitting the GARCH model at every val timestamp would be ideal but is too
  slow; instead we fit once per fold on the train slice and update predictions
  using `forecast(reindex=False)` with the rolling history available in arch.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252
PCT_SCALE = 100.0


def _annualized_log_vol_from_var(variance_pct_squared: float) -> float:
    """Convert per-day variance (in pct^2) to annualized log-vol (decimal space)."""
    if not np.isfinite(variance_pct_squared) or variance_pct_squared <= 0:
        return np.nan
    sigma_daily_decimal = np.sqrt(variance_pct_squared) / PCT_SCALE
    return float(np.log(sigma_daily_decimal * np.sqrt(TRADING_DAYS)))


def _aggregate_horizon_variance(forecast_variances: np.ndarray) -> float:
    """Average forecast variance across the horizon to mirror realized vol target.

    Realized target at horizon h is std of the next h returns (approximately
    sqrt(mean(returns^2)) when returns are roughly zero-mean), so the matching
    GARCH summary is the *mean* of the per-step forecast variances.
    """
    return float(np.mean(forecast_variances))


def predict_garch(
    returns: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Roll a GARCH(1,1) forecast over each validation timestamp.

    Strategy: fit once on the training-fold returns, then for each validation
    timestamp t, refit using `last_obs` advanced through history so the model
    state always reflects info available at t (no peek into future).
    """
    if returns.isna().any():
        # GARCH cannot be fit on NaN; drop a leading NaN if present.
        returns = returns.dropna()

    preds = np.full(len(val_idx), np.nan, dtype=float)
    if len(train_idx) < 50:
        return preds

    pct_returns = returns * PCT_SCALE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Refit per validation step is expensive; instead we fit once on the train
        # window and then update by extending `last_obs` to each val index.
        full_series = pct_returns.reset_index(drop=True)
        train_end = int(train_idx[-1])

        try:
            base_model = arch_model(
                full_series,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                rescale=False,
            )
            base_res = base_model.fit(
                last_obs=train_end + 1,
                disp="off",
                show_warning=False,
            )
        except Exception:
            return preds

        for i, vi in enumerate(val_idx):
            try:
                fc = base_res.forecast(
                    horizon=horizon,
                    start=int(vi),
                    reindex=False,
                )
                # `variance` is a DataFrame with rows indexed by start position and
                # columns h.1, h.2, ... h.horizon
                var_row = fc.variance.iloc[0].to_numpy(dtype=float)
                preds[i] = _annualized_log_vol_from_var(_aggregate_horizon_variance(var_row))
            except Exception:
                preds[i] = np.nan
    return preds
