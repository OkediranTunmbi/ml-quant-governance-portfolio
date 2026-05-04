"""
calibration.py — Probability calibration for credit risk models.

WHY CALIBRATION MATTERS IN CREDIT RISK:
Under Basel II/III, a Probability of Default (PD) model must produce
well-calibrated outputs: a score of 0.30 should mean that roughly 30% of
loans with that score will default.  This is not just a modelling nicety —
it is a regulatory requirement.  Poorly calibrated PD scores lead to:

  1. Mispriced loans (interest rate doesn't reflect true risk)
  2. Incorrect regulatory capital calculations (Basel IRB approach)
  3. Fairness violations (groups may have systematically biased scores)

SR 11-7 expects validators to test calibration as part of independent
model validation.  ECE (Expected Calibration Error) quantifies the
average gap between predicted probability and observed frequency.

Platt scaling (sigmoid calibration) is a simple and widely used technique:
  logit(P_calibrated) = A * logit(P_raw) + B
where A and B are fit on a holdout calibration set.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from netcal.metrics import ECE


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error via the netcal library.
    ECE = E[ |P(Y=1|score=s) - s| ] — the average absolute gap between
    predicted probability and empirical default frequency.
    """
    ece = ECE(n_bins)
    return float(ece.measure(y_prob, y_true))


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    label: str = "Model",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot a reliability diagram (calibration curve).
    Perfect calibration: points fall on the diagonal.
    Above diagonal: model is under-confident (predicts lower prob than actual).
    Below diagonal: model is over-confident (predicts higher prob than actual).
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(mean_predicted, fraction_of_positives, "o-", label=label)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title(f"Reliability Diagram — {label}")
    ax.legend(loc="upper left")
    return ax


class PlattScaledModel:
    """
    Thin wrapper that combines a prefit base model with a Platt sigmoid
    learned on a held-out calibration set.

    sklearn removed cv='prefit' in 1.4.  We replicate the same behaviour
    manually: the base model is frozen; only the sigmoid (A, B) parameters
    of a 1-D logistic regression are fitted on the calibration holdout.
    This is algebraically identical to what CalibratedClassifierCV(cv='prefit')
    was doing internally.
    """

    def __init__(self, base_model, platt_lr: LogisticRegression):
        self.base_model = base_model
        self.platt_lr   = platt_lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt_lr.predict_proba(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def fit_platt_scaling(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
) -> PlattScaledModel:
    """
    Fit Platt scaling on a holdout calibration set.

    The base model is already fitted — we only learn the sigmoid (A, B)
    parameters on the new holdout data.  This separation prevents overfitting:
    the calibration data was not used during model training.

    Platt scaling fits: logit(P_calibrated) = A * logit(P_raw) + B
    which is equivalent to a 1-D logistic regression with raw scores as input.
    """
    raw_scores = model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
    # C=1e10 makes the logistic regression nearly unconstrained — we want
    # the sigmoid to fit the data, not be regularised away
    platt_lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt_lr.fit(raw_scores, y_calib)
    calibrated = PlattScaledModel(model, platt_lr)
    print("[calibration] Fitted Platt scaling on calibration holdout.")
    return calibrated


def compare_calibration(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    n_bins: int = 15,
    save_dir: str = "data/processed",
) -> dict:
    """
    Side-by-side reliability diagrams and ECE comparison.
    Returns ECE metrics for the governance report.
    """
    ece_raw = compute_ece(y_true, y_prob_raw, n_bins)
    ece_cal = compute_ece(y_true, y_prob_cal, n_bins)

    print(f"[calibration] ECE before calibration: {ece_raw:.4f}")
    print(f"[calibration] ECE after  calibration: {ece_cal:.4f}")
    print(f"[calibration] ECE improvement: {ece_raw - ece_cal:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    reliability_diagram(
        y_true, y_prob_raw, n_bins=n_bins,
        label=f"Raw LR (ECE={ece_raw:.3f})", ax=axes[0]
    )
    axes[0].set_title("Before Calibration")

    reliability_diagram(
        y_true, y_prob_cal, n_bins=n_bins,
        label=f"Platt Scaled (ECE={ece_cal:.3f})", ax=axes[1]
    )
    axes[1].set_title("After Platt Scaling")

    plt.suptitle("Calibration Comparison — Test Set", fontsize=13, y=1.01)
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "calibration_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[calibration] Saved reliability diagram comparison to {out}")

    return {
        "ece_before": round(ece_raw, 4),
        "ece_after":  round(ece_cal, 4),
        "ece_delta":  round(ece_raw - ece_cal, 4),
    }


def monitor_calibration(
    y_true_monitor: np.ndarray,
    y_prob_monitor: np.ndarray,
    ece_test: float,
    n_bins: int = 15,
    save_dir: str = "data/processed",
) -> dict:
    """
    Evaluate calibration on the monitor (post-2015) set.
    Reports whether calibration has degraded over time — a sign of concept
    drift or distributional shift that post-training calibration cannot fix.
    """
    ece_mon = compute_ece(y_true_monitor, y_prob_monitor, n_bins)
    degraded = ece_mon > ece_test * 1.5  # flag if ECE grew by 50%+

    fig, ax = plt.subplots(figsize=(6, 6))
    reliability_diagram(
        y_true_monitor, y_prob_monitor, n_bins=n_bins,
        label=f"Monitor (ECE={ece_mon:.3f})", ax=ax
    )
    ax.set_title("Reliability Diagram — Monitor Set (post-2015)")
    plt.tight_layout()
    out = Path(save_dir) / "calibration_monitor.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

    status = "DEGRADED" if degraded else "OK"
    print(f"[calibration] Monitor ECE={ece_mon:.4f} vs test ECE={ece_test:.4f} [{status}]")

    return {
        "ece_monitor":        round(ece_mon, 4),
        "ece_test_reference": round(ece_test, 4),
        "calibration_status": status,
    }
