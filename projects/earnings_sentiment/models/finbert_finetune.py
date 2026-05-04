"""FinBERT fine-tuning with early stopping and temperature-scaled calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import (
    CHECKPOINTS_DIR,
    LABELS,
    PREDICTIONS_DIR,
    SEED,
    ensure_dirs,
    set_seed,
)
from data.load import load_phrasebank_splits
from models.finbert_zero import _label_remap

MODEL_NAME = "finbert_finetune"
HF_ID = "ProsusAI/finbert"
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
MAX_LEN = 256
PATIENCE = 2


class _PhraseBankDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, perm: np.ndarray):
        # Re-map dataset labels (negative/neutral/positive) -> the model's native id space
        # so cross-entropy lines up with FinBERT's classification head ordering.
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        # perm[c] tells us the model's id for canonical label c, which is exactly
        # the value we need to feed into the loss.
        self.labels = torch.tensor([int(perm[label]) for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(model, loader, device, perm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (macro_f1, probs_in_canonical_order, true_canonical_labels)."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            labels_native = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs[:, perm])
            # Convert native-label-space -> canonical-label-space for metric comparison.
            all_labels.append(np.array([perm.tolist().index(int(l)) for l in labels_native]))
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    f1 = f1_score(labels, probs.argmax(axis=1), average="macro")
    return float(f1), probs, labels


def _temperature_scale(val_logits: np.ndarray, val_labels_native: np.ndarray) -> float:
    """Single-parameter Platt-style calibration on logits.

    Operates in the model's native label space (matches the saved logits) so we
    don't need to re-run inference. Returns the optimal temperature T that
    minimizes NLL on validation.
    """
    logits = torch.from_numpy(val_logits.astype(np.float32))
    labels = torch.from_numpy(val_labels_native.astype(np.int64))
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def _step():
        optimizer.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(_step)
    return float(T.detach().clamp(min=1e-3).item())


def _save_predictions(
    probs: np.ndarray,
    df: pd.DataFrame,
    split_name: str,
    suffix: str,
) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(probs, columns=[f"prob_{c}" for c in LABELS])
    out.insert(0, "text", df["text"].values)
    out.insert(1, "label_id", df["label_id"].values)
    out.insert(2, "label_str", df["label_str"].values)
    out["pred_id"] = probs.argmax(axis=1)
    out["pred_str"] = out["pred_id"].map({i: LABELS[i] for i in range(len(LABELS))})
    out["confidence"] = probs.max(axis=1)
    out["model"] = f"{MODEL_NAME}_{suffix}"
    out["split"] = split_name
    path = PREDICTIONS_DIR / f"{MODEL_NAME}_{suffix}_{split_name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    ensure_dirs()
    set_seed(SEED)
    device = _device()

    splits = load_phrasebank_splits()
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_ID)
    model.to(device)

    id2label = {int(i): str(lbl).lower() for i, lbl in model.config.id2label.items()}
    perm = _label_remap([id2label[i] for i in range(len(id2label))])

    train_ds = _PhraseBankDataset(
        splits["train"]["text"].tolist(),
        splits["train"]["label_id"].tolist(),
        tokenizer,
        perm,
    )
    val_ds = _PhraseBankDataset(
        splits["val"]["text"].tolist(),
        splits["val"]["label_id"].tolist(),
        tokenizer,
        perm,
    )
    test_ds = _PhraseBankDataset(
        splits["test"]["text"].tolist(),
        splits["test"]["label_id"].tolist(),
        tokenizer,
        perm,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_f1 = -math.inf
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

        val_f1, _, _ = _evaluate(model, val_loader, device, perm)
        print(f"[ft] epoch={epoch}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[ft] early stopping at epoch={epoch} (best val_macro_f1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = CHECKPOINTS_DIR / "finbert_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    (ckpt_dir / "best_val_f1.json").write_text(json.dumps({"val_macro_f1": best_f1}, indent=2))
    print(f"[ft] best checkpoint saved to {ckpt_dir}")

    # Collect raw val logits to fit a temperature on; we only consume val here, never test.
    model.eval()
    val_logits: list[np.ndarray] = []
    val_native_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            labels_native = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            val_logits.append(logits)
            val_native_labels.append(labels_native)
    val_logits_arr = np.concatenate(val_logits, axis=0)
    val_native_labels_arr = np.concatenate(val_native_labels, axis=0)
    temperature = _temperature_scale(val_logits_arr, val_native_labels_arr)
    print(f"[ft] temperature scaling T={temperature:.3f}")
    (ckpt_dir / "temperature.json").write_text(json.dumps({"T": temperature}, indent=2))

    # Final predictions for both val and test, before and after calibration.
    for split_name, loader, df in (("val", val_loader, splits["val"]), ("test", test_loader, splits["test"])):
        all_probs_uncal: list[np.ndarray] = []
        all_probs_cal: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits.cpu().numpy()
                probs_uncal = _softmax(logits)[:, perm]
                probs_cal = _softmax(logits / temperature)[:, perm]
                all_probs_uncal.append(probs_uncal)
                all_probs_cal.append(probs_cal)
        probs_uncal = np.concatenate(all_probs_uncal, axis=0)
        probs_cal = np.concatenate(all_probs_cal, axis=0)
        path_u = _save_predictions(probs_uncal, df, split_name, suffix="uncal")
        path_c = _save_predictions(probs_cal, df, split_name, suffix="cal")
        print(f"[ft] saved {split_name}: {path_u.name}, {path_c.name}")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
