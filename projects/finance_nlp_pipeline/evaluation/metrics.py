"""Final test-set classification metrics for all models.

Reads pre-saved prediction CSVs — never re-runs any model.
Produces artifacts/summary/classification_summary.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score

from data.load import ROOT
from features.engineer import LABEL_ORDER

PREDS_DIR   = ROOT / "data" / "predictions"
SUMMARY_DIR = ROOT / "artifacts" / "summary"

# Models to evaluate and the CSV prefix they use.
MODELS = {
    "TF-IDF + LogReg":        "tfidf",
    "FinBERT fine-tuned":     "finbert",
    "GPT-4 few-shot":         "gpt4",
}


def _load_test(prefix: str) -> pd.DataFrame:
    """Load test-split predictions for a model, filtering GPT-4 parse errors."""
    df = pd.read_csv(PREDS_DIR / f"{prefix}_test.csv")
    if "parse_error" in df.columns:
        n_err = df["parse_error"].sum()
        if n_err:
            print(f"  [{prefix}] excluding {n_err} parse errors from metrics")
        df = df[~df["parse_error"]].reset_index(drop=True)
    return df


def compute_row(display_name: str, df: pd.DataFrame) -> dict:
    """Compute all metrics for one model's test predictions."""
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    probs  = df[["prob_negative", "prob_neutral", "prob_positive"]].values

    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])

    return {
        "model":        display_name,
        "accuracy":     accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro"),
        "f1_negative":  per_class[0],
        "f1_neutral":   per_class[1],
        "f1_positive":  per_class[2],
        "ece":          float(ECE(bins=15).measure(probs, y_true)),
        "n_samples":    len(df),
    }


def main() -> None:
    rows = []
    for display_name, prefix in MODELS.items():
        path = PREDS_DIR / f"{prefix}_test.csv"
        if not path.exists():
            print(f"[metrics] skipping {display_name} — {path.name} not found")
            continue
        df  = _load_test(prefix)
        row = compute_row(display_name, df)
        rows.append(row)
        print(
            f"[metrics] {display_name:25s} | "
            f"acc={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}  ECE={row['ece']:.4f}"
        )

    summary = pd.DataFrame(rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / "classification_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n[metrics] summary -> {out}")

    # Pretty-print the leaderboard
    print("\n" + "=" * 75)
    print(f"{'Model':25s} {'Acc':>7} {'F1-mac':>7} {'F1-neg':>7} {'F1-neu':>7} {'F1-pos':>7} {'ECE':>7}")
    print("-" * 75)
    for row in rows:
        print(
            f"{row['model']:25s} "
            f"{row['accuracy']:7.4f} {row['macro_f1']:7.4f} "
            f"{row['f1_negative']:7.4f} {row['f1_neutral']:7.4f} "
            f"{row['f1_positive']:7.4f} {row['ece']:7.4f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
