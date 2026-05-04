"""Reliability diagrams + ECE plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netcal.metrics import ECE

from config import LABELS, PLOTS_DIR, PREDICTIONS_DIR, ensure_dirs


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """Return (mean_confidence, mean_accuracy, bin_counts) for the predicted-class confidence."""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mean_conf = np.zeros(n_bins)
    mean_acc = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.any():
            mean_conf[i] = confidences[mask].mean()
            mean_acc[i] = correct[mask].mean()
            counts[i] = mask.sum()
    return mean_conf, mean_acc, counts


def plot_reliability_per_model(out_path: Path | None = None) -> Path:
    ensure_dirs()
    pred_files = sorted(PREDICTIONS_DIR.glob("*_test.csv"))
    if not pred_files:
        raise RuntimeError("No *_test.csv predictions to plot.")
    n = len(pred_files)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, pred_files):
        df = pd.read_csv(path)
        probs = df[[f"prob_{c}" for c in LABELS]].to_numpy()
        y_true = df["label_id"].to_numpy()
        mean_conf, mean_acc, counts = reliability_curve(probs, y_true)
        ece = float(ECE(bins=15).measure(probs, y_true))
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        # Only draw bars where the bin actually had observations to avoid misleading zeros.
        mask = counts > 0
        ax.bar(mean_conf[mask], mean_acc[mask], width=0.05, alpha=0.7, edgecolor="black")
        model_name = df["model"].iloc[0] if "model" in df.columns else path.stem
        ax.set_title(f"{model_name}\nECE={ece:.3f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Reliability diagrams (test set)")
    fig.tight_layout()
    out = out_path or (PLOTS_DIR / "reliability_diagrams.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    out = plot_reliability_per_model()
    print(f"[calibration] saved {out}")


if __name__ == "__main__":
    main()
