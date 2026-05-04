"""
model.py — Logistic Regression credit default model: training, evaluation, SHAP.

Credit risk performance is reported in bank-standard metrics:
  - AUC-ROC    : overall discrimination
  - AUC-PR     : discrimination under class imbalance (more informative than AUC-ROC
                 when defaults are rare, since it focuses on the positive class)
  - Gini       : Gini = 2 * AUC - 1.  Industry standard for scorecards.
                 A Gini of 0 = random model; 1 = perfect; typical retail credit
                 models target Gini > 0.40.
  - KS Statistic: maximum separation between the cumulative distribution of
                 scores for defaults vs non-defaults.  Regulators and validators
                 often require KS > 0.20 for a model to pass initial review.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from scipy.stats import ks_2samp

RANDOM_STATE = 42
MODEL_PATH   = "data/processed/logistic_model.pkl"


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    KS statistic: max(TPR - FPR) across all thresholds.
    Equivalently, the two-sample KS between the score distributions of
    positives and negatives.
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    ks_stat, _ = ks_2samp(pos_scores, neg_scores)
    return float(ks_stat)


def evaluate(y_true: np.ndarray, y_score: np.ndarray, label: str = "") -> dict:
    """Return standard credit risk performance metrics."""
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr  = average_precision_score(y_true, y_score)
    gini    = 2 * auc_roc - 1
    ks      = compute_ks(y_true, y_score)
    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr":  round(auc_pr, 4),
        "gini":    round(gini, 4),
        "ks":      round(ks, 4),
    }
    tag = f"[{label}] " if label else ""
    print(f"{tag}AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
          f"Gini={gini:.4f}  KS={ks:.4f}")
    return metrics


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = MODEL_PATH,
) -> LogisticRegression:
    """
    Train logistic regression with balanced class weights.
    class_weight='balanced' compensates for the minority class (defaults) by
    up-weighting each default observation by n_samples / (2 * n_defaults).
    Without this, the model would maximise accuracy by predicting 'no default'
    for nearly all loans, because defaults are the minority class.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    print(f"[model] Trained on {X_train.shape[0]:,} samples, {X_train.shape[1]} features.")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[model] Saved model to {save_path}")
    return model


def load_model(path: str = MODEL_PATH) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "Test",
    save_dir: str = "data/processed",
) -> None:
    """Side-by-side ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {label}")
    axes[0].legend()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    axes[1].plot(rec, prec, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {label}")
    axes[1].legend()

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / f"roc_pr_{label.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved ROC/PR plot to {out}")


def compute_shap(
    model: LogisticRegression,
    X: np.ndarray,
    feature_names: list[str],
    save_dir: str = "data/processed",
    n_background: int = 500,
) -> shap.Explanation:
    """
    Compute SHAP values using LinearExplainer (exact for linear models).
    SHAP (SHapley Additive exPlanations) decomposes each prediction into
    additive feature contributions, so you can explain WHY the model scored
    a specific loan the way it did — a regulatory expectation under SR 11-7
    and the EU AI Act.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Use a background sample to speed up computation
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X), size=min(n_background, len(X)), replace=False)
    background = X[idx]

    explainer   = shap.LinearExplainer(model, background)
    shap_values = explainer(X)

    # --- Global feature importance: mean |SHAP| per feature ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False, ax=ax)
    plt.title("Global Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- Beeswarm: direction and magnitude of each feature's effect ---
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Feature Effects")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[model] Saved SHAP bar and beeswarm plots to {save_dir}")
    return shap_values


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_dir: str = "data/processed",
) -> None:
    """
    Waterfall plots for three representative predictions:
      - True Positive  : predicted default, actually defaulted
      - False Positive : predicted default, actually repaid
      - False Negative : predicted repaid, actually defaulted
    These help a model risk officer understand specific failure modes.
    """
    threshold = 0.5
    y_pred_bin = (y_pred_proba >= threshold).astype(int)

    tp_idx = np.where((y_pred_bin == 1) & (y_true == 1))[0]
    fp_idx = np.where((y_pred_bin == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred_bin == 0) & (y_true == 1))[0]

    cases = [
        (tp_idx, "True Positive",  "tp"),
        (fp_idx, "False Positive", "fp"),
        (fn_idx, "False Negative", "fn"),
    ]

    for idx_arr, label, suffix in cases:
        if len(idx_arr) == 0:
            print(f"[model] No {label} examples found for waterfall plot.")
            continue
        i = idx_arr[0]
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i], max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {label} (index {i})")
        plt.tight_layout()
        out = Path(save_dir) / f"shap_waterfall_{suffix}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[model] Saved waterfall ({label}) to {out}")
