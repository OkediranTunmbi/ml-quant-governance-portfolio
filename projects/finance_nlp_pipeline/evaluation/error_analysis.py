"""Confusion matrices, top-10 confident errors, and TF-IDF SHAP on errors.

SYSTEMATIC FAILURE PATTERNS TO LOOK FOR
-----------------------------------------
When you run this and inspect the output, look for these cross-model patterns:

1. NEUTRAL IS THE HARDEST CLASS
   All three models will likely show more off-diagonal confusion in the neutral
   row/column than negative or positive. Neutral sentences are genuinely
   ambiguous — they report facts without clear directional language. Models
   that lean toward the majority class (neutral ~59%) may over-predict it.

2. NEGATIVE IS THE RAREST AND MOST CONFUSED
   With ~13% of PhraseBank, negative is the minority class. Even with
   class_weight='balanced', models trained on limited negative examples
   often confuse negative with neutral (hedged language looks similar).
   FinBERT, pre-trained on financial text, should handle this better than
   TF-IDF — the confusion matrix will show whether that's true.

3. GPT-4 ERRORS CLUSTER ON SHORT OR AMBIGUOUS SENTENCES
   Few-shot GPT-4 tends to fail on sentences that require deep domain
   knowledge (specific accounting terms, regulatory language) or sentences
   so short they lack context. The top-10 error table for GPT-4 will show
   the reasoning it generated — often revealing where in-context examples
   failed to cover an edge case.

4. SHAP REVEALS LEXICAL ANCHORS FOR TF-IDF FAILURES
   The SHAP plot on TF-IDF's top errors shows which tokens pushed the model
   toward the wrong class. Common pattern: the token "expects" appearing in
   a negative sentence ("expects a loss") triggers a positive prediction
   because "expects" is correlated with forward guidance in positive contexts.
   FinBERT handles this because it reads the full phrase in context.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from data.load import ROOT, SPLITS_DIR
from features.engineer import (
    LABEL_ORDER,
    LABEL2ID,
    encode_labels,
    load_tfidf,
    transform,
)

PREDS_DIR    = ROOT / "data" / "predictions"
PLOTS_DIR    = ROOT / "artifacts" / "plots"
SUMMARY_DIR  = ROOT / "artifacts" / "summary"

MODELS = {
    "TF-IDF + LogReg":    "tfidf",
    "FinBERT fine-tuned": "finbert",
    "GPT-4 few-shot":     "gpt4",
}
TOP_N = 10


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(model_dfs: dict[str, pd.DataFrame]) -> Path:
    """Side-by-side normalised confusion matrices for all models."""
    n = len(model_dfs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (display_name, df) in zip(axes, model_dfs.items()):
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1, 2])
        # Normalise by row (true class) to show recall per class.
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm,
            annot=cm,               # show raw counts as annotations
            fmt="d",
            cmap="Blues",
            vmin=0, vmax=1,
            xticklabels=LABEL_ORDER,
            yticklabels=LABEL_ORDER,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(display_name)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=13)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Top-10 confident errors
# ---------------------------------------------------------------------------

def top_errors(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Return the n highest-confidence wrong predictions, sorted by confidence."""
    errors = df[df["y_true"] != df["y_pred"]].copy()
    errors["true_label"]  = errors["y_true"].map({i: l for i, l in enumerate(LABEL_ORDER)})
    errors["pred_label"]  = errors["y_pred"].map({i: l for i, l in enumerate(LABEL_ORDER)})
    return (
        errors[["text", "true_label", "pred_label", "confidence"]]
        .sort_values("confidence", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SHAP on TF-IDF errors
# ---------------------------------------------------------------------------

def plot_shap_errors(error_texts: pd.Series) -> Path:
    """SHAP values for TF-IDF's top errors — reveals which tokens caused each failure.

    We re-fit a base LogisticRegression (same hyperparameters as Stage 2)
    on the training set to get a model with a coef_ attribute that
    LinearExplainer requires. The calibrated model saved in Stage 2 wraps
    5 inner models and has no direct coef_, so we need this re-fit.
    The re-fit is deterministic (same data, same seed) and takes <10 seconds.
    """
    # Load saved vectorizer and training data.
    vec      = load_tfidf()
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    y_train  = np.array(encode_labels(train_df["label"]))
    X_train  = transform(vec, train_df["text"])

    # Re-fit base LR (fast — sparse linear model on 20k features).
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=42, solver="lbfgs",
    )
    lr.fit(X_train, y_train)

    X_errors = transform(vec, error_texts)
    explainer = shap.LinearExplainer(lr, X_train, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_errors)   # list of [n_errors, n_features] per class

    feature_names = vec.get_feature_names_out().tolist()
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(18, 5))
    for ax, cls_shap, cls_name in zip(axes, shap_vals, LABEL_ORDER):
        # Mean absolute SHAP across the error cases for this class.
        mean_abs  = np.abs(cls_shap).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[-15:][::-1]
        top_feats = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]
        ax.barh(top_feats[::-1], top_vals[::-1], color="tomato")
        ax.set_title(f"Tokens driving errors toward '{cls_name}'", fontsize=10)
        ax.set_xlabel("Mean |SHAP| on error cases")

    plt.suptitle("TF-IDF SHAP — Top Error Cases (test set)", fontsize=13)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "shap_tfidf_errors.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load test predictions for all available models
    # ------------------------------------------------------------------
    model_dfs: dict[str, pd.DataFrame] = {}
    for display_name, prefix in MODELS.items():
        path = PREDS_DIR / f"{prefix}_test.csv"
        if not path.exists():
            print(f"[error_analysis] skipping {display_name} — {path.name} not found")
            continue
        df = pd.read_csv(path)
        if "parse_error" in df.columns:
            df = df[~df["parse_error"]].reset_index(drop=True)
        model_dfs[display_name] = df

    if not model_dfs:
        print("[error_analysis] no test CSVs found — run models first")
        return

    # ------------------------------------------------------------------
    # 2. Confusion matrices
    # ------------------------------------------------------------------
    cm_path = plot_confusion_matrices(model_dfs)
    print(f"[error_analysis] confusion matrices -> {cm_path}")

    # ------------------------------------------------------------------
    # 3. Top-10 confident errors per model
    # ------------------------------------------------------------------
    for display_name, df in model_dfs.items():
        errors_df = top_errors(df)
        prefix    = MODELS[display_name]
        out_path  = SUMMARY_DIR / f"top_errors_{prefix}.csv"
        errors_df.to_csv(out_path, index=False)
        print(f"[error_analysis] {display_name}: {len(errors_df)} top errors -> {out_path}")

    # ------------------------------------------------------------------
    # 4. SHAP on TF-IDF errors
    # ------------------------------------------------------------------
    tfidf_key = "TF-IDF + LogReg"
    if tfidf_key in model_dfs:
        tfidf_errors = model_dfs[tfidf_key]
        tfidf_errors = tfidf_errors[tfidf_errors["y_true"] != tfidf_errors["y_pred"]]
        if len(tfidf_errors) > 0:
            print(f"[error_analysis] computing SHAP on {len(tfidf_errors)} TF-IDF errors ...")
            shap_path = plot_shap_errors(tfidf_errors["text"].reset_index(drop=True))
            print(f"[error_analysis] SHAP error plot -> {shap_path}")
        else:
            print("[error_analysis] TF-IDF has no errors on test set — SHAP plot skipped")


if __name__ == "__main__":
    main()
