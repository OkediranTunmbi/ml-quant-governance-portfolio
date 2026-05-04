"""FinBERT zero-shot inference.

Zero-shot here means we use ProsusAI/finbert as-is (no fine-tuning). The model
already has a domain-pretrained finance vocabulary, so this is a strong
"upper baseline" relative to TF-IDF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, PREDICTIONS_DIR, SEED, ensure_dirs, set_seed
from data.load import load_phrasebank_splits

MODEL_NAME = "finbert_zero"
HF_ID = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LEN = 256


def _label_remap(model_label_names: list[str]) -> np.ndarray:
    """Build a permutation that maps FinBERT's label order onto our LABELS order.

    FinBERT publishes labels as ['positive','negative','neutral']; we always
    store columns in the canonical (negative, neutral, positive) order to
    match `config.LABELS` so downstream metric code can stay identical across
    every model.
    """
    perm = np.array([model_label_names.index(label) for label in LABELS])
    return perm


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_proba(texts: pd.Series) -> np.ndarray:
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_ID)
    device = _device()
    model.to(device)
    model.eval()

    # FinBERT's id2label uses lowercase strings; normalize for safety.
    id2label = {int(i): str(lbl).lower() for i, lbl in model.config.id2label.items()}
    perm = _label_remap([id2label[i] for i in range(len(id2label))])

    probs_chunks: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts.iloc[start : start + BATCH_SIZE].tolist()
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Reorder so columns are (negative, neutral, positive) regardless of HF order.
        probs_chunks.append(probs[:, perm])

    return np.concatenate(probs_chunks, axis=0)


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
    for name in ("val", "test"):
        probs = _predict_proba(splits[name]["text"])
        path = _save_predictions(probs, splits[name], name)
        print(f"[{MODEL_NAME}] saved {name} predictions -> {path}")


if __name__ == "__main__":
    main()
