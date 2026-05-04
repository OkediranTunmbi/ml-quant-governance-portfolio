"""Classification metrics + per-model summary table.

All summaries are written to ``artifacts/summary/`` so notebooks and the README
can render the same numbers without recomputing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score

from config import LABELS, PREDICTIONS_DIR, SUMMARY_DIR, ensure_dirs


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    f1s = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))))
    return {f"f1_{lbl}": float(s) for lbl, s in zip(LABELS, f1s)}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    """ECE on the predicted-class confidence; one number per model.

    netcal's ECE supports multiclass directly when given a probability matrix.
    """
    return float(ECE(bins=bins).measure(probs, y_true))


def model_metrics_from_predictions(df: pd.DataFrame) -> dict[str, float]:
    """Compute the full metric bundle from a saved predictions CSV."""
    y_true = df["label_id"].to_numpy()
    y_pred = df["pred_id"].to_numpy()
    probs = df[[f"prob_{c}" for c in LABELS]].to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        **per_class_f1(y_true, y_pred),
        "ece": expected_calibration_error(probs, y_true),
    }


def collect_test_summary() -> pd.DataFrame:
    """Aggregate metrics for every model that has a *_test.csv prediction file."""
    ensure_dirs()
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob("*_test.csv")):
        df = pd.read_csv(path)
        model = df["model"].iloc[0] if "model" in df.columns else path.stem.replace("_test", "")
        metrics = model_metrics_from_predictions(df)
        metrics["model"] = model
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("model")
    cols = ["accuracy", "macro_f1", "f1_negative", "f1_neutral", "f1_positive", "ece"]
    out = out[cols].sort_values("macro_f1", ascending=False)
    out.to_csv(SUMMARY_DIR / "classification_summary.csv")
    return out


def main() -> None:
    summary = collect_test_summary()
    if summary.empty:
        print("[metrics] no *_test.csv predictions found yet")
        return
    print(summary.round(4).to_string())
    print(f"\n[metrics] wrote {SUMMARY_DIR / 'classification_summary.csv'}")


if __name__ == "__main__":
    main()
