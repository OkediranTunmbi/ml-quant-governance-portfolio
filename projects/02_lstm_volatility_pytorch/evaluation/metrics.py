"""Forecast-evaluation metrics for log-volatility predictions.

All inputs are expected to be **annualized log-volatility** (the same space as the
targets produced by `features.engineer`). For QLIKE we transform back to variance
space, since QLIKE is defined on positive variances.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _to_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    """Root mean squared error directly in log-vol space."""
    yt, yp = _to_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def qlike(y_true, y_pred) -> float:
    """QLIKE loss in variance space.

    QLIKE = mean( log(sigma_pred^2) + sigma_true^2 / sigma_pred^2 )

    where sigma is non-annualized vol. Since the inputs are *log-annualized vol*,
    we convert back via sigma = exp(log_vol_annualized) / sqrt(252). The annualization
    factor cancels inside QLIKE up to an additive constant, so we simply use the
    annualized variances throughout for consistency with the training scale.
    """
    yt, yp = _to_arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    var_true = np.exp(2.0 * yt)
    var_pred = np.exp(2.0 * yp)
    var_pred = np.maximum(var_pred, EPS)
    return float(np.mean(np.log(var_pred) + var_true / var_pred))


def qlike_terms(y_true, y_pred) -> np.ndarray:
    """Per-observation QLIKE terms for diagnostics/robust summaries."""
    yt, yp = _to_arrays(y_true, y_pred)
    if yt.size == 0:
        return np.array([], dtype=float)
    var_true = np.exp(2.0 * yt)
    var_pred = np.exp(2.0 * yp)
    var_pred = np.maximum(var_pred, EPS)
    return np.log(var_pred) + var_true / var_pred


def qlike_median(y_true, y_pred) -> float:
    terms = qlike_terms(y_true, y_pred)
    if terms.size == 0:
        return float("nan")
    return float(np.median(terms))


def qlike_trimmed(y_true, y_pred, trim: float = 0.01) -> float:
    """Trim tail outliers from both ends before averaging QLIKE terms."""
    terms = qlike_terms(y_true, y_pred)
    if terms.size == 0:
        return float("nan")
    lo = np.quantile(terms, trim)
    hi = np.quantile(terms, 1 - trim)
    kept = terms[(terms >= lo) & (terms <= hi)]
    if kept.size == 0:
        return float("nan")
    return float(np.mean(kept))


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of steps where forecast direction (rise/fall) matches realized.

    Compares sign of first-difference of consecutive predictions vs realized.
    Zero-change steps (rare in vol space) are excluded from the denominator.
    """
    yt, yp = _to_arrays(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    dt = np.diff(yt)
    dp = np.diff(yp)
    nonzero = dt != 0
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.sign(dt[nonzero]) == np.sign(dp[nonzero])))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "qlike": qlike(y_true, y_pred),
        "qlike_median": qlike_median(y_true, y_pred),
        "qlike_trimmed_1pct": qlike_trimmed(y_true, y_pred, trim=0.01),
        "dir_acc": directional_accuracy(y_true, y_pred),
    }
