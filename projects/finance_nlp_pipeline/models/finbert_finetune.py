"""FinBERT fine-tuning with early stopping and temperature scaling.

HOW THE EMBEDDING PIPELINE WORKS (Learning goal)
-------------------------------------------------
When a sentence enters FinBERT, it goes through four distinct steps before
we ever see a probability. Understanding each step is the core of modern NLP.

STEP 1 — WORDPIECE TOKENIZATION
  Raw text is split into *subword* pieces, not whole words:
    "impairment" -> ["impair", "##ment"]
    "write-down"  -> ["write", "-", "down"]
  The '##' prefix means "this piece continues the previous token."
  Why subwords? They handle rare words and typos gracefully: a word the model
  has never seen can still be broken into familiar pieces. A pure word-level
  vocabulary would need millions of entries; WordPiece uses ~30,000.

STEP 2 — SPECIAL TOKENS + PADDING
  The tokenizer wraps every sentence with:
    [CLS] token1 token2 ... tokenN [SEP] [PAD] [PAD] ...
  [CLS] ("classification") is a learnable token placed at position 0.
  [SEP] marks end of sentence (used for sentence-pair tasks).
  [PAD] fills shorter sentences to the fixed max_length so every batch
  has the same shape (required for GPU tensor operations).

STEP 3 — ATTENTION MASK
  A binary mask tells the model which tokens are real (1) vs padding (0).
    real tokens -> 1  (attend to these)
    [PAD] tokens -> 0  (ignore these)
  Without the mask, the transformer would mix real information with
  padding noise, degrading representations.

STEP 4 — TRANSFORMER LAYERS -> [CLS] EMBEDDING
  The 12 transformer layers each run *self-attention*: every token looks at
  every other token (respecting the mask) and updates its representation.
  After 12 layers, the [CLS] token's 768-dimensional vector has "seen" the
  entire sentence in context. We pass this [CLS] vector through a linear
  classification head (768 -> 3) to get the raw scores (logits) per class.

STEP 5 — TEMPERATURE SCALING (post-training calibration)
  Raw logits -> softmax -> probabilities, but these probabilities are often
  overconfident (e.g., 0.97 when the model is only right 80% of the time).
  Temperature scaling divides all logits by a scalar T before softmax:
    P(class_i) = exp(logit_i / T) / Σ exp(logit_j / T)
  Geometrically:
    T > 1 -> softens the distribution (probabilities closer to uniform)
    T < 1 -> sharpens the distribution (probabilities more extreme)
    T = 1 -> no change
  Crucially, argmax(logits / T) = argmax(logits) for any T > 0, so
  temperature scaling never changes the predicted class — only the
  confidence. We find T by minimizing cross-entropy on the *validation* set,
  then apply it to the test set. This is calibration without data leakage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.load import ROOT, SPLITS_DIR
from features.engineer import LABEL_ORDER, LABEL2ID, ID2LABEL, encode_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = ROOT / "models" / "checkpoints" / "finbert_best"
PLOTS_DIR      = ROOT / "artifacts" / "plots"
PREDS_DIR      = ROOT / "data" / "predictions"

# ---------------------------------------------------------------------------
# Hyperparameters — all logged to MLflow for reproducibility
# ---------------------------------------------------------------------------
MODEL_NAME   = "ProsusAI/finbert"
MAX_LEN      = 128       # PhraseBank sentences average ~14 words; 128 is ample
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LR           = 2e-5      # standard fine-tuning LR for BERT-based models
WEIGHT_DECAY = 0.01      # L2 regularisation via AdamW
PATIENCE     = 2         # early stopping: stop if val F1 doesn't improve for 2 epochs
SEED         = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhraseBankDataset(Dataset):
    """PyTorch Dataset that tokenizes text and pairs it with integer labels.

    The tokenizer call does steps 1-3 from the module docstring:
    WordPiece splitting -> adding [CLS]/[SEP]/[PAD] -> building the attention mask.
    Everything is pre-computed here (not inside __getitem__) so tokenization
    happens once and the DataLoader can batch without repeated overhead.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        # tokenizer() returns a BatchEncoding dict with keys:
        #   input_ids      — integer token IDs, shape [n, max_len]
        #   attention_mask — 1 for real tokens, 0 for padding, shape [n, max_len]
        self.encodings = tokenizer(
            texts,
            truncation=True,        # cut sentences longer than max_len
            padding="max_length",   # pad shorter sentences with [PAD] tokens
            max_length=max_len,
            return_tensors="pt",    # return PyTorch tensors directly
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon MPS is another option on Mac; ignored here for simplicity.
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One full pass over the DataLoader with gradient updates. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        # The model returns a SequenceClassifierOutput; .loss is cross-entropy
        # computed internally when labels are provided.
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output.loss.backward()

        # Gradient clipping prevents exploding gradients — common in fine-tuning.
        # Without it, a large gradient step can destroy pre-trained weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += output.loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader. Returns (macro_f1, logits, preds, labels)."""
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # output.logits shape: [batch_size, num_labels] — raw unnormalised scores
        all_logits.append(output.logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits)     # [n, 3]
    labels = np.concatenate(all_labels)     # [n]
    preds  = logits.argmax(axis=1)          # [n]
    macro_f1 = f1_score(labels, preds, average="macro")
    return macro_f1, logits, preds, labels


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def find_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find the scalar T that minimises NLL on val logits.

    We use LBFGS (a quasi-Newton optimiser) with 50 iterations — it converges
    much faster than SGD for a single-parameter optimisation problem like this.
    LBFGS requires a closure (a callable that recomputes loss and calls backward)
    because it may evaluate the function multiple times per step.
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)

    # Start temperature at 1.0 (no scaling) and optimise.
    temperature = nn.Parameter(torch.ones(1))
    optimiser   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimiser.zero_grad()
        scaled_logits = logits_t / temperature.clamp(min=1e-8)  # prevent div-by-zero
        loss = F.cross_entropy(scaled_logits, labels_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(temperature.item())


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_bins: int = 15,
) -> None:
    """Side-by-side reliability diagrams before and after temperature scaling.

    A perfectly calibrated model's curve falls on the diagonal: when the model
    says P=0.8, it should be correct 80% of the time. Points above the
    diagonal mean the model is underconfident; below means overconfident.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, title in [
        (axes[0], probs_before, "Before temperature scaling"),
        (axes[1], probs_after,  "After temperature scaling"),
    ]:
        confidences = probs.max(axis=1)     # top predicted probability
        predictions = probs.argmax(axis=1)
        correct     = (predictions == labels).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

        ece = float(ECE(bins=n_bins).measure(probs, labels))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(bin_confs, bin_accs, width=1 / n_bins, alpha=0.6, label="Model")
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend()

    plt.suptitle("FinBERT Reliability Diagram — Validation Set", fontsize=13)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Prediction saver
# ---------------------------------------------------------------------------

def _save_predictions(
    df: pd.DataFrame,
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    split_name: str,
) -> None:
    """Apply temperature scaling and save predictions in the standard schema."""
    scaled_logits = logits / max(temperature, 1e-8)
    probs  = torch.softmax(torch.tensor(scaled_logits, dtype=torch.float32), dim=-1).numpy()
    preds  = probs.argmax(axis=1)

    out = df[["text", "label"]].copy()
    out["y_true"]      = labels
    out["y_pred"]      = preds
    out["confidence"]  = probs.max(axis=1)
    for i, cls in enumerate(LABEL_ORDER):
        out[f"prob_{cls}"] = probs[:, i]

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDS_DIR / f"finbert_{split_name}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()
    print(f"[finbert] device: {device}")

    # ------------------------------------------------------------------
    # 1. Load splits
    # ------------------------------------------------------------------
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

    y_train = encode_labels(train_df["label"])
    y_val   = encode_labels(val_df["label"])
    y_test  = encode_labels(test_df["label"])

    # ------------------------------------------------------------------
    # 2. Tokenizer — converts raw strings to input_ids + attention_mask
    # ------------------------------------------------------------------
    print(f"[finbert] loading tokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PhraseBankDataset(train_df["text"].tolist(), y_train, tokenizer, MAX_LEN)
    val_ds   = PhraseBankDataset(val_df["text"].tolist(),   y_val,   tokenizer, MAX_LEN)
    test_ds  = PhraseBankDataset(test_df["text"].tolist(),  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Model
    # ignore_mismatched_sizes=True reinitialises the classification head
    # so our label order (negative=0, neutral=1, positive=2) is used,
    # not ProsusAI's original ordering. The transformer backbone
    # (all 12 attention layers) is kept intact with its pre-trained weights.
    # ------------------------------------------------------------------
    print(f"[finbert] loading model from {MODEL_NAME} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # AdamW decouples weight decay from the gradient update — standard for transformers.
    # We exclude bias and LayerNorm parameters from weight decay (they shouldn't be penalised).
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR)

    # ------------------------------------------------------------------
    # 4. Fine-tuning loop with early stopping
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("finance_nlp_classification")

    best_val_f1   = -1.0
    epochs_no_imp = 0
    best_state    = None

    with mlflow.start_run(run_name="finbert_finetune") as run:
        mlflow.log_params({
            "model":        MODEL_NAME,
            "max_len":      MAX_LEN,
            "batch_size":   BATCH_SIZE,
            "num_epochs":   NUM_EPOCHS,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
            "patience":     PATIENCE,
        })

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, optimizer, device)
            val_f1, val_logits, val_preds, val_labels = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            # Log val F1 as a metric *curve* — step=epoch lets MLflow plot
            # F1 vs epoch, making it easy to see whether training is still
            # improving or has plateaued.
            mlflow.log_metric("val_macro_f1", val_f1, step=epoch)
            mlflow.log_metric("train_loss",   train_loss, step=epoch)

            print(
                f"  epoch {epoch+1}/{NUM_EPOCHS} | "
                f"loss={train_loss:.4f} | val_f1={val_f1:.4f} | {elapsed:.0f}s"
            )

            if val_f1 > best_val_f1:
                best_val_f1   = val_f1
                epochs_no_imp = 0
                # Deep-copy the state dict so we can restore later.
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                best_val_logits  = val_logits.copy()
                best_val_labels  = val_labels.copy()
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= PATIENCE:
                    print(f"[finbert] early stopping at epoch {epoch+1}")
                    break

        # ------------------------------------------------------------------
        # 5. Restore best checkpoint weights
        # ------------------------------------------------------------------
        model.load_state_dict(best_state)
        print(f"[finbert] best val macro-F1: {best_val_f1:.4f}")

        # ------------------------------------------------------------------
        # 6. Temperature scaling on val logits
        # ------------------------------------------------------------------
        print("[finbert] fitting temperature scalar on val logits ...")
        temperature = find_temperature(best_val_logits, best_val_labels)
        print(f"[finbert] temperature T = {temperature:.4f}")

        # Uncalibrated val probs (T=1)
        probs_before = torch.softmax(
            torch.tensor(best_val_logits, dtype=torch.float32), dim=-1
        ).numpy()
        # Calibrated val probs (T=temperature)
        probs_after = torch.softmax(
            torch.tensor(best_val_logits / temperature, dtype=torch.float32), dim=-1
        ).numpy()

        val_ece_before = float(ECE(bins=15).measure(probs_before, best_val_labels))
        val_ece_after  = float(ECE(bins=15).measure(probs_after,  best_val_labels))
        val_acc        = accuracy_score(best_val_labels, probs_after.argmax(axis=1))

        mlflow.log_metrics({
            "val_ece_before_calibration": val_ece_before,
            "val_ece":                    val_ece_after,
            "val_accuracy":               val_acc,
            "temperature":                temperature,
        })

        print(f"[finbert] ECE before={val_ece_before:.4f}  after={val_ece_after:.4f}")

        # ------------------------------------------------------------------
        # 7. Reliability diagram
        # ------------------------------------------------------------------
        reliability_path = PLOTS_DIR / "reliability_finbert.png"
        plot_reliability(probs_before, probs_after, best_val_labels, reliability_path)
        mlflow.log_artifact(str(reliability_path))
        print(f"[finbert] reliability diagram saved -> {reliability_path}")

        # ------------------------------------------------------------------
        # 8. Save predictions for all splits (using calibrated probabilities)
        # ------------------------------------------------------------------
        for split_name, df, loader, labels in [
            ("train", train_df, train_loader, np.array(y_train)),
            ("val",   val_df,   val_loader,   np.array(y_val)),
            ("test",  test_df,  test_loader,  np.array(y_test)),
        ]:
            _, logits, _, _ = evaluate(model, loader, device)
            _save_predictions(df, logits, labels, temperature, split_name)
        print(f"[finbert] predictions saved -> {PREDS_DIR}")

        # ------------------------------------------------------------------
        # 9. Save checkpoint: model weights + tokenizer + temperature
        # ------------------------------------------------------------------
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(CHECKPOINT_DIR)
        tokenizer.save_pretrained(CHECKPOINT_DIR)
        (CHECKPOINT_DIR / "temperature.json").write_text(
            json.dumps({"temperature": temperature})
        )
        mlflow.log_artifact(str(CHECKPOINT_DIR / "temperature.json"))
        print(f"[finbert] checkpoint saved -> {CHECKPOINT_DIR}")

        print(f"\n[finbert] run_id: {run.info.run_id}")
        print("  -> View MLflow UI: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
