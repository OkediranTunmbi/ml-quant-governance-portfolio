"""Reliability diagrams for all classifiers — one unified figure.

A reliability diagram answers the question: "When my model says it is X%
confident, is it actually correct X% of the time?"

HOW TO READ THEM
  - X axis : mean predicted confidence in a bin
  - Y axis : fraction of predictions in that bin that were correct
  - Diagonal: perfect calibration (confidence = accuracy)
  - Bar above diagonal -> model is UNDERconfident (predicts less than it knows)
  - Bar below diagonal -> model is OVERconfident (more sure than it should be)

WHY ECE MATTERS OPERATIONALLY
  A trading signal that ranks stocks by model confidence is only useful if
  confidence is meaningful. If the model says 0.95 but is right only 70% of
  the time, you'll systematically over-size positions on its highest-confidence
  (and most-wrong) calls. ECE quantifies this miscalibration in a single number.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netcal.metrics import ECE

from data.load import ROOT

PREDS_DIR = ROOT / "data" / "predictions"
PLOTS_DIR = ROOT / "artifacts" / "plots"

MODELS = {
    "TF-IDF + LogReg":    "tfidf",
    "FinBERT fine-tuned": "finbert",
    "GPT-4 few-shot":     "gpt4",
}
N_BINS = 15


def _reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> tuple[list[float], list[float], list[int]]:
    """Bin predictions and return (mean_conf, fraction_correct, counts)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_confs.append(float(confidences[mask].mean()))
        bin_accs.append(float(correct[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return bin_confs, bin_accs, bin_counts


def main() -> None:
    available = {
        name: prefix
        for name, prefix in MODELS.items()
        if (PREDS_DIR / f"{prefix}_test.csv").exists()
    }
    if not available:
        print("[calibration] no test prediction CSVs found — run models first")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (display_name, prefix) in zip(axes, available.items()):
        df = pd.read_csv(PREDS_DIR / f"{prefix}_test.csv")
        if "parse_error" in df.columns:
            df = df[~df["parse_error"]]

        probs  = df[["prob_negative", "prob_neutral", "prob_positive"]].values
        labels = df["y_true"].values
        ece    = float(ECE(bins=N_BINS).measure(probs, labels))

        bin_confs, bin_accs, _ = _reliability_bins(probs, labels, N_BINS)
        width = 1.0 / N_BINS

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.bar(
            bin_confs, bin_accs, width=width * 0.9,
            alpha=0.7, color="steelblue", label="Model",
        )
        # Gap-fill bars show where the model is over/underconfident.
        for conf, acc in zip(bin_confs, bin_accs):
            if acc < conf:
                ax.bar(conf, conf - acc, width=width * 0.9, bottom=acc,
                       alpha=0.25, color="red")
            else:
                ax.bar(conf, acc - conf, width=width * 0.9, bottom=conf,
                       alpha=0.25, color="green")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"{display_name}\nECE = {ece:.4f}")
        ax.legend(fontsize=8)

    plt.suptitle("Reliability Diagrams — Test Set (15 bins)", fontsize=13)
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "reliability_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[calibration] reliability diagrams -> {out}")


if __name__ == "__main__":
    main()
