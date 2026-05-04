"""TF-IDF + Logistic Regression baseline with isotonic calibration.

This is the cheapest model in the comparison and acts as a sanity floor for
FinBERT. The same train/val/test splits as the deep models are reused so
metrics are directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from config import (
    LABELS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEED,
    ensure_dirs,
    set_seed,
)
from data.load import load_phrasebank_splits
from features.engineer import TfidfArtifacts, fit_tfidf, transform_tfidf

MODEL_NAME = "tfidf_logreg"


def _build_estimator() -> LogisticRegression:
    # class_weight='balanced' inversely weights the loss by class frequency so
    # the model does not collapse to predicting the majority "neutral" class.
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )


def fit_calibrated_tfidf(
    train_df: pd.DataFrame,
) -> tuple[TfidfArtifacts, CalibratedClassifierCV]:
    """Fit TF-IDF on train, then a 5-fold isotonic-calibrated logistic regressor."""
    set_seed(SEED)
    artifacts = fit_tfidf(train_df["text"])
    X_train = transform_tfidf(artifacts, train_df["text"])
    y_train = train_df["label_id"].to_numpy()

    base = _build_estimator()
    # CalibratedClassifierCV wraps the base estimator in 5-fold CV and learns an
    # isotonic mapping from raw scores to calibrated probabilities.
    calibrated = CalibratedClassifierCV(estimator=base, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)
    return artifacts, calibrated


def predict_proba(
    artifacts: TfidfArtifacts,
    model: CalibratedClassifierCV,
    texts: pd.Series,
) -> np.ndarray:
    X = transform_tfidf(artifacts, texts)
    return model.predict_proba(X)


def _save_predictions(probs: np.ndarray, df: pd.DataFrame, split_name: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(probs, columns=[f"prob_{c}" for c in LABELS])
    out.insert(0, "text", df["text"].values)
    out.insert(1, "label_id", df["label_id"].values)
    out.insert(2, "label_str", df["label_str"].values)
    out["pred_id"] = probs.argmax(axis=1)
    out["pred_str"] = out["pred_id"].map({i: LABELS[i] for i in range(len(LABELS))})
    out["confidence"] = probs.max(axis=1)
    out["model"] = MODEL_NAME
    out["split"] = split_name
    path = PREDICTIONS_DIR / f"{MODEL_NAME}_{split_name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    ensure_dirs()
    splits = load_phrasebank_splits()
    artifacts, model = fit_calibrated_tfidf(splits["train"])

    for name in ("val", "test"):
        probs = predict_proba(artifacts, model, splits[name]["text"])
        path = _save_predictions(probs, splits[name], name)
        print(f"[{MODEL_NAME}] saved {name} predictions -> {path}")

    joblib.dump(model, MODELS_DIR / f"{MODEL_NAME}.joblib")
    joblib.dump(artifacts.vectorizer, MODELS_DIR / f"{MODEL_NAME}_vectorizer.joblib")
    print(f"[{MODEL_NAME}] saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
