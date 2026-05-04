"""Confusion matrices and high-confidence error inspection."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import LABELS, PLOTS_DIR, PREDICTIONS_DIR, SUMMARY_DIR, ensure_dirs


def plot_confusion_matrices(out_path: Path | None = None) -> Path:
    ensure_dirs()
    pred_files = sorted(PREDICTIONS_DIR.glob("*_test.csv"))
    if not pred_files:
        raise RuntimeError("No *_test.csv predictions to plot.")
    n = len(pred_files)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, pred_files):
        df = pd.read_csv(path)
        y_true = df["label_id"].to_numpy()
        y_pred = df["pred_id"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            cbar=False,
            ax=ax,
        )
        model_name = df["model"].iloc[0] if "model" in df.columns else path.stem
        ax.set_title(model_name)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true")
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    out = out_path or (PLOTS_DIR / "confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def top_confident_errors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-n highest-confidence misclassifications for one model."""
    wrong = df[df["pred_id"] != df["label_id"]].copy()
    wrong = wrong.sort_values("confidence", ascending=False).head(n)
    cols = ["text", "label_str", "pred_str", "confidence"]
    return wrong[cols].reset_index(drop=True)


def collect_top_errors(n: int = 10) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(PREDICTIONS_DIR.glob("*_test.csv")):
        df = pd.read_csv(path)
        model = df["model"].iloc[0] if "model" in df.columns else path.stem
        out[model] = top_confident_errors(df, n=n)
        out[model].to_csv(SUMMARY_DIR / f"top_errors_{model}.csv", index=False)
    return out


def main() -> None:
    cm_path = plot_confusion_matrices()
    errors = collect_top_errors()
    print(f"[errors] confusion matrices -> {cm_path}")
    for model, frame in errors.items():
        print(f"[errors] top-10 confident errors for {model}: saved {len(frame)} rows")


if __name__ == "__main__":
    main()
