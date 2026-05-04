"""Feature engineering and target construction for SPY volatility forecasting.

All features use **past-only** information; all targets use **future-only** returns
relative to each row's index, so there is no look-ahead leakage when the resulting
frame is split chronologically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (1, 5, 21)
HAR_WINDOWS = (1, 5, 21)
# Practical floor to avoid pathological near-zero daily vol in 1d targets/features.
MIN_DAILY_VOL = 1e-4


@dataclass(frozen=True)
class FeatureSpec:
    """Names of the engineered columns; useful for downstream model code."""

    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    return_col: str = "log_return"
    date_col: str = "Date"


def compute_log_returns(close: pd.Series) -> pd.Series:
    """log(P_t / P_{t-1}) — first row is NaN by construction."""
    return np.log(close).diff()


def realized_vol_log_annualized(returns: pd.Series, window: int) -> pd.Series:
    """Trailing realized vol (annualized log-vol) using the past `window` returns inclusive."""
    if window == 1:
        # For a 1-day horizon, realized vol reduces to |r_t| scaled to annualized units.
        sigma = returns.abs()
    else:
        # ddof=1 matches Series.std default; min_periods=window forces full windows only.
        sigma = returns.rolling(window=window, min_periods=window).std(ddof=1)
    sigma = sigma.clip(lower=MIN_DAILY_VOL)
    return np.log(sigma * np.sqrt(TRADING_DAYS))


def forward_realized_vol_log_annualized(returns: pd.Series, horizon: int) -> pd.Series:
    """Forward realized vol target at time t using returns from t+1 .. t+horizon.

    Algebra:
      Let s = returns.shift(-horizon). Then s[t] = returns[t + horizon].
      A rolling window of size `horizon` evaluated at t covers s[t-horizon+1 .. t],
      which equals returns[t+1 .. t+horizon] -- exactly the *future* window we want,
      with no overlap with the current return at time t.
    """
    if horizon == 1:
        # Forward 1-day realized vol at t uses absolute next-day return |r_{t+1}|.
        sigma = returns.shift(-1).abs()
    else:
        shifted = returns.shift(-horizon)
        sigma = shifted.rolling(window=horizon, min_periods=horizon).std(ddof=1)
    sigma = sigma.clip(lower=MIN_DAILY_VOL)
    return np.log(sigma * np.sqrt(TRADING_DAYS))


def add_har_features(df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """HAR-style realized vol features at 1/5/21 day windows (past-only)."""
    out = df.copy()
    for w in HAR_WINDOWS:
        out[f"vol_{w}d"] = realized_vol_log_annualized(returns, w)
    return out


def add_return_features(df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Squared and absolute return features (past-only by definition)."""
    out = df.copy()
    out["ret_sq"] = returns ** 2
    out["ret_abs"] = returns.abs()
    # Lag-1 versions because the same-day return is technically "current" information;
    # using only lagged values keeps everything strictly past.
    out["ret_sq_lag1"] = out["ret_sq"].shift(1)
    out["ret_abs_lag1"] = out["ret_abs"].shift(1)
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Day-of-week and month, encoded as integers (model side handles categoricals)."""
    out = df.copy()
    out["dow"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    return out


def add_targets(df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Forward realized vol targets for each forecast horizon."""
    out = df.copy()
    for h in HORIZONS:
        out[f"target_{h}d"] = forward_realized_vol_log_annualized(returns, h)
    return out


def build_features(spy: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Assemble the full feature/target frame from raw SPY OHLCV.

    Returns the dataframe (rows with any NaN feature/target dropped) plus a `FeatureSpec`
    listing which columns are features vs. targets.
    """
    # Use adjusted close so returns reflect splits/dividends and reduce discretization artifacts.
    price_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    df = spy[["Date", price_col]].copy().sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={price_col: "Price"})
    df["log_return"] = compute_log_returns(df["Price"])

    df = add_har_features(df, df["log_return"])
    df = add_return_features(df, df["log_return"])
    df = add_calendar_features(df, "Date")
    df = add_targets(df, df["log_return"])

    feature_cols = (
        "vol_1d",
        "vol_5d",
        "vol_21d",
        "ret_sq_lag1",
        "ret_abs_lag1",
        "dow",
        "month",
    )
    target_cols = tuple(f"target_{h}d" for h in HORIZONS)

    # Drop any rows where features or *any* target is missing so all horizons share the same index.
    required = list(feature_cols) + list(target_cols)
    df = df.dropna(subset=required).reset_index(drop=True)

    return df, FeatureSpec(feature_cols=feature_cols, target_cols=target_cols)


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    spy_path = project_dir / "data" / "spy.csv"
    out_path = project_dir / "data" / "features.csv"

    spy = pd.read_csv(spy_path, parse_dates=["Date"])
    df, spec = build_features(spy)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows / {len(spec.feature_cols)} features to {out_path}")


if __name__ == "__main__":
    main()
