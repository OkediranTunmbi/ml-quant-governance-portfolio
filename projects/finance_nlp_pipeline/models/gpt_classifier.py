"""GPT-4 few-shot financial sentiment classifier.

WHAT FEW-SHOT PROMPTING IS (Learning goal)
------------------------------------------
Fine-tuning (Stage 3) updates a model's *weights* via gradient descent.
The model sees thousands of labelled examples, adjusts its parameters,
and permanently encodes that task knowledge. It requires a GPU, a training
loop, and time.

Few-shot prompting does something fundamentally different: we give the model
a handful of labelled examples *in the input text itself*, and the model
generalises to new inputs without any weight updates at all. This is called
**in-context learning** — the model has already seen enough language patterns
during pre-training to recognise "here are examples of a task, now do it
for this new input" and comply, even on tasks it was never explicitly trained on.

Why does it work? Large language models trained on internet-scale text implicitly
learn patterns like "when text is formatted as Examples -> Answer, produce an Answer."
Few-shot prompting exploits that pattern at inference time.

The trade-offs vs fine-tuning:
  PROS  — no training cost, no GPU, generalises to new tasks in minutes
  CONS  — latency per call (~1–3s), API cost per call, confidence scores are
          self-reported (poorly calibrated), context window limits the number
          of examples

For this project GPT-4 is the *challenger* — we want to see whether paying per
token for a frontier model beats the free, already-trained FinBERT on the same
test set. The answer is often surprising.

CACHING STRATEGY
----------------
Every API call costs money and time. The cache maps SHA256(input_text) -> raw
API response JSON. Before any API call, we check the cache. On a hit, we skip
the call entirely. This means:
  - Re-running the script costs zero additional API calls.
  - The evaluation is fully reproducible from the cache files alone.
  - If you change the prompt (different few-shot examples), the cache is still
    valid for any unchanged inputs — the SHA256 key is based on input text only,
    so different prompts with the same inputs can share cached raw responses.
    (We re-parse with the new prompt format if needed.)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from netcal.metrics import ECE

from data.load import ROOT, SPLITS_DIR
from features.engineer import LABEL_ORDER, LABEL2ID, ID2LABEL, encode_labels

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = ROOT / "data" / "cache"
PREDS_DIR = ROOT / "data" / "predictions"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GPT_MODEL      = "gpt-4o"      # current GPT-4 family model
N_PER_CLASS    = 2             # few-shot examples per class -> 6 total
SEED           = 42
TEMPERATURE    = 0             # deterministic output — same input always gives same response
MAX_TOKENS     = 200

# Only run on val + test to avoid expensive train-set API calls (~2,800 sentences).
# Evaluation code reads val and test CSVs; train is optional.
SPLITS_TO_RUN  = ["val", "test"]


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------

def select_examples(train_df: pd.DataFrame, n_per_class: int = N_PER_CLASS) -> list[dict]:
    """Pick n examples per class from train, seeded for reproducibility.

    These examples are embedded directly in the system prompt and never change
    between test samples. Using train examples only (never val/test) ensures
    no data leakage even through the prompt.
    """
    examples = []
    for label in LABEL_ORDER:                       # fixed order: neg, neu, pos
        subset = (
            train_df[train_df["label"] == label]
            .sample(n=n_per_class, random_state=SEED)
        )
        for _, row in subset.iterrows():
            examples.append({"text": row["text"], "label": label})
    return examples


def build_system_prompt(examples: list[dict]) -> str:
    """Build the system prompt with labelled examples and output format instructions.

    The output format is strict JSON so we can parse it programmatically.
    Asking for 'confidence' lets GPT self-report its certainty — this is
    imperfect (LLMs are not well calibrated on self-reported confidence)
    but gives us a scalar we can use for ECE and error analysis.
    """
    lines = [
        "You are a financial sentiment classifier.",
        "Classify the sentiment of financial news sentences as negative, neutral, or positive.",
        "",
        "Return ONLY valid JSON with no extra text, using exactly this format:",
        '{"label": "negative|neutral|positive", "confidence": <float 0-1>, "reasoning": "<one sentence>"}',
        "",
        "Labelled examples:",
    ]
    for ex in examples:
        lines.append(f'Text: "{ex["text"]}"')
        # Placeholder confidence to show the expected format; GPT will supply its own.
        lines.append(
            f'{{"label": "{ex["label"]}", '
            f'"confidence": 0.95, '
            f'"reasoning": "Training example — {ex["label"]} sentiment."}}'
        )
        lines.append("")
    lines.append("Now classify the sentence provided by the user.")
    return "\n".join(lines)


def prompt_hash(system_prompt: str) -> str:
    """SHA256 of the system prompt — logged to MLflow to identify which prompt was used."""
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_path(text: str) -> Path:
    """Deterministic file path for caching a given input text's API response."""
    key = hashlib.sha256(text.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def load_from_cache(text: str) -> dict | None:
    """Return cached API response dict, or None if not cached."""
    path = cache_path(text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_to_cache(text: str, response_dict: dict) -> None:
    """Persist raw API response to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(text).write_text(
        json.dumps(response_dict, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# API call + parsing
# ---------------------------------------------------------------------------

def call_api(
    client: OpenAI,
    system_prompt: str,
    text: str,
    retries: int = 3,
) -> dict:
    """Call GPT with caching. Returns the raw API response as a dict."""
    cached = load_from_cache(text)
    if cached is not None:
        return cached                   # zero cost, zero latency on cache hit

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": text},
                ],
                temperature=TEMPERATURE,    # 0 = deterministic; important for reproducibility
                max_tokens=MAX_TOKENS,
            )
            result = {"content": response.choices[0].message.content}
            save_to_cache(text, result)
            return result
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)    # exponential back-off on transient errors
            else:
                raise exc

    return {}   # unreachable; satisfies type checker


def parse_response(raw: dict) -> tuple[str, float, str, bool]:
    """Parse GPT JSON response. Returns (label, confidence, reasoning, parse_error).

    WHY EXPLICIT ERROR HANDLING?
    GPT-4 occasionally produces malformed JSON — truncated, with prose before
    the JSON, or with non-standard keys. If we let json.loads crash, the whole
    evaluation run fails. Instead we catch the error, record parse_error=True,
    and exclude those rows from metrics while reporting the count. This is
    production-style fault tolerance.
    """
    content = raw.get("content", "")
    try:
        # Strip any leading/trailing prose in case GPT adds a sentence before the JSON.
        start = content.find("{")
        end   = content.rfind("}") + 1
        parsed = json.loads(content[start:end])

        label      = parsed["label"].strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning  = parsed.get("reasoning", "")

        if label not in LABEL2ID:
            return "parse_error", 0.0, f"unrecognised label: {label}", True

        return label, confidence, reasoning, False

    except (json.JSONDecodeError, KeyError, ValueError):
        return "parse_error", 0.0, content, True


# ---------------------------------------------------------------------------
# Probability vector helper
# ---------------------------------------------------------------------------

def confidence_to_probs(label: str, confidence: float) -> dict[str, float]:
    """Convert a single (label, confidence) into a 3-class probability vector.

    GPT-4 returns one confidence score for the predicted class. We distribute
    the remaining probability equally over the other two classes. This is an
    approximation — GPT-4 does not produce per-class logits — but it lets us
    compute ECE using the same machinery as TF-IDF and FinBERT.
    """
    other = (1.0 - confidence) / 2.0
    probs = {cls: other for cls in LABEL_ORDER}
    probs[label] = confidence
    return probs


# ---------------------------------------------------------------------------
# Prediction saver
# ---------------------------------------------------------------------------

def _save_predictions(rows: list[dict], split_name: str) -> None:
    """Save predictions in the standard schema + GPT-specific extra columns."""
    df = pd.DataFrame(rows)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDS_DIR / f"gpt4_{split_name}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # 1. Build few-shot prompt (fixed for all test samples)
    # ------------------------------------------------------------------
    train_df  = pd.read_csv(SPLITS_DIR / "train.csv")
    examples  = select_examples(train_df)
    sys_prompt = build_system_prompt(examples)
    p_hash     = prompt_hash(sys_prompt)
    print(f"[gpt4] system prompt hash: {p_hash}")
    print(f"[gpt4] few-shot examples ({N_PER_CLASS} per class):")
    for ex in examples:
        print(f"  [{ex['label']}] {ex['text'][:70]}...")

    # ------------------------------------------------------------------
    # 2. Run inference on each split
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("finance_nlp_classification")

    split_metrics: dict[str, dict] = {}

    with mlflow.start_run(run_name="gpt4_fewshot") as run:
        mlflow.log_params({
            "model":            GPT_MODEL,
            "few_shot_count":   N_PER_CLASS * len(LABEL_ORDER),
            "prompt_hash":      p_hash,
            "temperature":      TEMPERATURE,
        })

        for split_name in SPLITS_TO_RUN:
            df      = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
            y_true  = np.array(encode_labels(df["label"]))
            rows    = []
            n_errors = 0
            cache_hits = 0

            print(f"\n[gpt4] running {split_name} split ({len(df)} samples) ...")
            for i, (_, row) in enumerate(df.iterrows()):
                text = row["text"]

                # Cache check — prints a dot for each call, 'C' for cache hit
                was_cached = load_from_cache(text) is not None
                raw = call_api(client, sys_prompt, text)
                if was_cached:
                    cache_hits += 1

                label, confidence, reasoning, is_error = parse_response(raw)
                if is_error:
                    n_errors += 1

                probs = (
                    confidence_to_probs(label, confidence)
                    if not is_error
                    else {cls: 1 / 3 for cls in LABEL_ORDER}     # uniform on error
                )
                rows.append({
                    "text":          text,
                    "label":         row["label"],
                    "y_true":        int(y_true[i]),
                    "y_pred":        LABEL2ID.get(label, -1),
                    "confidence":    confidence,
                    "prob_negative": probs["negative"],
                    "prob_neutral":  probs["neutral"],
                    "prob_positive": probs["positive"],
                    "reasoning":     reasoning,
                    "parse_error":   is_error,
                })

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(df)} | cache_hits={cache_hits} | errors={n_errors}")

            _save_predictions(rows, split_name)
            print(f"[gpt4] {split_name}: {n_errors} parse errors ({n_errors/len(df)*100:.1f}%), "
                  f"{cache_hits} cache hits")

            # ------------------------------------------------------------------
            # 3. Compute metrics on valid (non-error) rows only
            # ------------------------------------------------------------------
            valid = [r for r in rows if not r["parse_error"]]
            if valid:
                yt = np.array([r["y_true"] for r in valid])
                yp = np.array([r["y_pred"] for r in valid])
                probs_arr = np.array([
                    [r["prob_negative"], r["prob_neutral"], r["prob_positive"]]
                    for r in valid
                ])
                macro_f1 = f1_score(yt, yp, average="macro")
                acc      = accuracy_score(yt, yp)
                ece      = float(ECE(bins=15).measure(probs_arr, yt))
                split_metrics[split_name] = {
                    "macro_f1": macro_f1, "accuracy": acc, "ece": ece,
                    "n_valid": len(valid), "n_errors": n_errors,
                }
                print(f"[gpt4] {split_name} macro-F1={macro_f1:.4f}  acc={acc:.4f}  ECE={ece:.4f}")

        # ------------------------------------------------------------------
        # 4. Log val metrics to MLflow (val = held-out from training, like other models)
        # ------------------------------------------------------------------
        if "val" in split_metrics:
            m = split_metrics["val"]
            mlflow.log_metrics({
                "val_macro_f1": m["macro_f1"],
                "val_accuracy": m["accuracy"],
                "val_ece":      m["ece"],
                "val_n_errors": m["n_errors"],
            })

        print(f"\n[gpt4] run_id: {run.info.run_id}")
        print("  -> View MLflow UI: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
