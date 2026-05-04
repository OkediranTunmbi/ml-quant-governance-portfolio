"""TF-IDF + Logistic Regression baseline with isotonic calibration.

WHY MLFLOW? (Learning goal)
---------------------------
Every time you tweak a hyperparameter — C, ngram_range, calibration method —
you produce a slightly different model. Without experiment tracking you'd need
a manual spreadsheet of "what parameters gave what F1." MLflow records every
run automatically: parameters, metrics, artifact paths, and timestamps. Later,
when FinBERT and GPT-4 results come in, a single `mlflow ui` command opens a
browser dashboard that compares all three models side-by-side. That comparison
is the core deliverable of experiment tracking — reproducibility and visibility
with zero manual bookkeeping.

To view the UI after running this script:
    mlflow ui --backend-store-uri mlruns
    # then open http://localhost:5000 in your browser
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from netcal.metrics import ECE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data.load import ROOT, SPLITS_DIR
from features.engineer import (
    LABEL_ORDER,
    encode_labels,
    fit_tfidf,
    save_tfidf,
    transform,
)

ARTIFACTS_DIR = ROOT / "models" / "artifacts" / "tfidf"
PLOTS_DIR = ROOT / "artifacts" / "plots"
PREDS_DIR = ROOT / "data" / "predictions"

# ---------------------------------------------------------------------------
# Hyperparameters — every value here is logged to MLflow so any future
# re-run with different settings is tracked separately and comparable.
# ---------------------------------------------------------------------------
C = 1.0
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 20_000
CALIB_METHOD = "isotonic"
CALIB_CV = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, labels: np.ndarray) -> float:
    """Expected Calibration Error via netcal, 15 bins."""
    return float(ECE(bins=15).measure(probs, labels))


def _save_predictions(
    df: pd.DataFrame,
    X,
    y: np.ndarray,
    model: CalibratedClassifierCV,
    split_name: str,
) -> None:
    """Persist per-row predictions + calibrated probabilities as CSV.

    WHY SAVE PREDICTIONS?
    Downstream evaluation code (evaluation/metrics.py, error_analysis.py)
    needs predictions from every model in the same format. Saving them now
    means evaluation never has to re-load or re-run the model — it just reads
    a CSV. This also makes the test-set evaluation a pure read-only operation,
    which enforces the no-leakage rule: the model never 'sees' the test set
    labels during training.
    """
    probs = model.predict_proba(X)          # shape [n, 3]
    preds = probs.argmax(axis=1)
    out = df[["text", "label"]].copy()
    out["y_true"] = y
    out["y_pred"] = preds
    out["confidence"] = probs.max(axis=1)   # max probability across classes
    for i, cls in enumerate(LABEL_ORDER):
        out[f"prob_{cls}"] = probs[:, i]
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDS_DIR / f"tfidf_{split_name}.csv", index=False)


def plot_shap_importance(
    base_lr: LogisticRegression,
    X_train,
    feature_names: list[str],
    top_k: int = 15,
) -> Path:
    """SHAP bar chart: top-k most influential tokens per sentiment class.

    WHY SHAP FOR A LINEAR MODEL?
    For Logistic Regression, SHAP values equal the feature's coefficient
    multiplied by its value for that sample — exactly what you'd expect from
    a linear model. But expressing them as SHAP makes them *additive*: you can
    sum up each token's contribution to arrive at the final log-odds, which is
    an intuitive and auditable explanation.

    For financial NLP this matters: regulators or stakeholders can ask
    "why did the model call this sentence positive?" and you can point to the
    top contributing tokens with a precise numerical answer.

    WHY USE base_lr INSTEAD OF THE CALIBRATED MODEL?
    CalibratedClassifierCV wraps five inner models (from 5-fold CV) and
    averages their probability outputs. LinearExplainer requires a single
    linear model with a coef_ attribute. base_lr — trained on all of train
    with the same hyperparameters — is the right proxy: calibration adjusts
    probabilities but does not change which tokens drive the decision.
    """
    explainer = shap.LinearExplainer(
        base_lr, X_train, feature_perturbation="interventional"
    )
    # shap_values is a list of [n_samples, n_features] arrays, one per class.
    shap_values = explainer.shap_values(X_train)

    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=(18, 6))
    for ax, cls_shap, cls_name in zip(axes, shap_values, LABEL_ORDER):
        mean_abs = np.abs(cls_shap).mean(axis=0)        # average contribution per token
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]   # top-k by importance
        top_features = [feature_names[i] for i in top_idx]
        top_values = mean_abs[top_idx]
        ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
        ax.set_title(f"Top {top_k} tokens — {cls_name}", fontsize=11)
        ax.set_xlabel("Mean |SHAP value|")

    plt.suptitle("TF-IDF LogReg — SHAP Feature Importance (train set)", fontsize=13)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "shap_tfidf.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the pre-saved splits (same CSV every model reads)
    # ------------------------------------------------------------------
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

    y_train = np.array(encode_labels(train_df["label"]))
    y_val   = np.array(encode_labels(val_df["label"]))
    y_test  = np.array(encode_labels(test_df["label"]))

    # ------------------------------------------------------------------
    # 2. TF-IDF: fit on train, transform all splits
    # ------------------------------------------------------------------
    print("[tfidf] fitting vectorizer on train only ...")
    vec = fit_tfidf(train_df["text"])
    save_tfidf(vec)

    X_train = transform(vec, train_df["text"])   # sparse [n_train, 20000]
    X_val   = transform(vec, val_df["text"])
    X_test  = transform(vec, test_df["text"])

    # ------------------------------------------------------------------
    # 3. Base LogReg — used for SHAP (requires a plain coef_ attribute)
    # ------------------------------------------------------------------
    print("[tfidf] fitting base LogReg for SHAP ...")
    base_lr = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",    # up-weight rare 'negative' class (~13% of data)
        random_state=42,
        solver="lbfgs",
    )
    base_lr.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Calibrated model — this is what we serve and evaluate
    #
    # WHY CALIBRATION? (Learning goal)
    # A raw LogisticRegression can be overconfident: it might output 0.95
    # probability for a prediction that is only correct 70% of the time.
    # Downstream use cases care deeply about this:
    #   - A trading system using confidence to size positions will
    #     over-bet on overconfident wrong predictions.
    #   - An alert system thresholding at P > 0.8 will trigger too often.
    #
    # Isotonic calibration fits a monotone step function (the 'isotonic'
    # regression) that maps raw model scores -> calibrated probabilities.
    # Unlike Platt scaling (sigmoid fit), isotonic makes no shape assumption,
    # which is better when the score distribution is non-sigmoid.
    # cv=5 means it uses 5-fold cross-validation to prevent the calibrator
    # from over-fitting to the training set it's calibrating on.
    # ------------------------------------------------------------------
    print("[tfidf] calibrating with isotonic regression (cv=5) ...")
    calibrated = CalibratedClassifierCV(
        estimator=LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced",
            random_state=42, solver="lbfgs",
        ),
        cv=CALIB_CV,
        method=CALIB_METHOD,
    )
    calibrated.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate on val set (test set is untouched until final evaluation)
    # ------------------------------------------------------------------
    val_probs = calibrated.predict_proba(X_val)
    val_preds = val_probs.argmax(axis=1)

    val_f1  = f1_score(y_val, val_preds, average="macro")
    val_acc = accuracy_score(y_val, val_preds)
    val_ece = _ece(val_probs, y_val)

    print(f"[tfidf] val macro-F1={val_f1:.4f}  acc={val_acc:.4f}  ECE={val_ece:.4f}")

    # ------------------------------------------------------------------
    # 6. Save predictions for all three splits
    # ------------------------------------------------------------------
    for split_name, df, X, y in [
        ("train", train_df, X_train, y_train),
        ("val",   val_df,   X_val,   y_val),
        ("test",  test_df,  X_test,  y_test),
    ]:
        _save_predictions(df, X, y, calibrated, split_name)
    print(f"[tfidf] predictions saved -> {PREDS_DIR}")

    # ------------------------------------------------------------------
    # 7. SHAP feature importance plot
    # ------------------------------------------------------------------
    print("[tfidf] computing SHAP values (this may take ~30s) ...")
    feature_names = vec.get_feature_names_out().tolist()
    shap_path = plot_shap_importance(base_lr, X_train, feature_names)
    print(f"[tfidf] SHAP plot saved -> {shap_path}")

    # ------------------------------------------------------------------
    # 8. Persist model artifacts
    # ------------------------------------------------------------------
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(calibrated, model_path)
    print(f"[tfidf] calibrated model saved -> {model_path}")

    # ------------------------------------------------------------------
    # 9. Log run to MLflow
    #
    # set_tracking_uri points MLflow at this project's own mlruns/ folder.
    # set_experiment groups all classification model runs under one name
    # so they show up together in the UI comparison view.
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("finance_nlp_classification")

    with mlflow.start_run(run_name="tfidf_logreg"):
        mlflow.log_params({
            "model":                "tfidf_logreg",
            "C":                    C,
            "ngram_range":          str(NGRAM_RANGE),
            "max_features":         MAX_FEATURES,
            "calibration_method":   CALIB_METHOD,
            "calibration_cv":       CALIB_CV,
            "class_weight":         "balanced",
        })
        mlflow.log_metrics({
            "val_macro_f1":  val_f1,
            "val_accuracy":  val_acc,
            "val_ece":       val_ece,
        })
        # Log artifacts so the MLflow UI links directly to the saved files.
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "vectorizer.joblib"))
        mlflow.log_artifact(str(shap_path))

    print("\n[tfidf] done.")
    print("  -> View MLflow UI: mlflow ui --backend-store-uri mlruns")
    print(f"  -> val macro-F1 : {val_f1:.4f}")
    print(f"  -> val ECE      : {val_ece:.4f}")


if __name__ == "__main__":
    main()
