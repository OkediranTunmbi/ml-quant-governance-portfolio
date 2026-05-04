# CLAUDE.md — Project Context for finance_nlp_pipeline

Read this file at the start of every session or after context compaction.
It contains the full implementation contract so any stage can be built
consistently with what came before.

---

## Project location
`ml-quant-governance-portfolio/ml-quant-governance-portfolio/projects/finance_nlp_pipeline/`

---

## Build stages and status

| Stage | Description | Status |
|---|---|---|
| 1 | Scaffold + `.env` + `data/load.py` + `features/engineer.py` | DONE |
| 2 | `models/tfidf_baseline.py` + MLflow logging | DONE |
| 3 | `models/finbert_finetune.py` — FinBERT fine-tuning + temp scaling | DONE |
| 4 | `models/gpt_classifier.py` — GPT-4 few-shot classification | DONE |
| 5 | `models/gpt_summarizer.py` — GPT-3.5 summarization, 2 prompt variants | DONE |
| 6 | `evaluation/metrics.py` + `calibration.py` + `error_analysis.py` | DONE |
| 7 | `serving/app.py` FastAPI + `serving/model_card.md` | DONE |
| 8 | Notebooks + `README.md` | DONE |

---

## Key constants (used across every file)

```python
SEED = 42
LABEL_ORDER = ["negative", "neutral", "positive"]   # fixed canonical order
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
ROOT = Path(__file__).parent.parent   # finance_nlp_pipeline/
```

---

## Directory layout and what each path is for

```
finance_nlp_pipeline/
├── data/
│   ├── load.py                  # DONE — loads PhraseBank + ECTSum, writes splits
│   ├── splits/train.csv         # 70% of PhraseBank, stratified, seed=42
│   ├── splits/val.csv           # 10%
│   ├── splits/test.csv          # 20% — NEVER touched until final evaluation
│   ├── predictions/             # tfidf_{split}.csv, finbert_{split}.csv, etc.
│   ├── cache/                   # GPT API response cache (SHA256-keyed JSON)
│   └── ectsum.csv               # full ECTSum, all splits, with 'split' column
├── features/
│   └── engineer.py              # DONE — clean_text, fit_tfidf, transform, encode_labels
├── models/
│   ├── tfidf_baseline.py        # DONE
│   ├── finbert_finetune.py      # TODO (Stage 3)
│   ├── gpt_classifier.py        # TODO (Stage 4)
│   ├── gpt_summarizer.py        # TODO (Stage 5)
│   ├── artifacts/tfidf/         # vectorizer.joblib, model.joblib
│   └── checkpoints/finbert_best/# saved FinBERT weights + tokenizer + temperature
├── evaluation/
│   ├── metrics.py               # TODO (Stage 6)
│   ├── calibration.py           # TODO (Stage 6)
│   └── error_analysis.py        # TODO (Stage 6)
├── serving/
│   ├── app.py                   # TODO (Stage 7)
│   └── model_card.md            # TODO (Stage 7)
├── notebooks/                   # TODO (Stage 8)
├── artifacts/
│   ├── plots/                   # shap_tfidf.png, reliability_*.png, etc.
│   └── summary/                 # classification_summary.csv, summarization_summary.csv
├── mlruns/                      # MLflow tracking (project-local, gitignored)
├── requirements.txt
├── .env                         # OPENAI_API_KEY — gitignored
└── .gitignore
```

---

## Implemented APIs (what future stages must import from)

### `data/load.py`
```python
ROOT: Path                      # finance_nlp_pipeline/
SPLITS_DIR: Path                # finance_nlp_pipeline/data/splits/
LABEL_NAMES: list[str]          # ["negative", "neutral", "positive"]
SEED: int                       # 42
def load_phrasebank() -> pd.DataFrame           # columns: text, label (string)
def make_splits(df) -> tuple[df, df, df]        # train, val, test
def load_ectsum() -> pd.DataFrame               # columns: id, text, summary, split
def main() -> None                              # writes all CSVs
```

### `features/engineer.py`
```python
LABEL_ORDER: list[str]          # ["negative", "neutral", "positive"]
LABEL2ID: dict                  # {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict                  # {0: "negative", 1: "neutral", 2: "positive"}
ARTIFACTS_DIR: Path             # models/artifacts/tfidf/
def clean_text(text: str) -> str
def fit_tfidf(train_texts: pd.Series) -> TfidfVectorizer
def transform(vec, texts: pd.Series) -> sparse matrix
def encode_labels(labels: pd.Series) -> list[int]
def decode_labels(ids: list[int]) -> list[str]
def save_tfidf(vec) -> None
def load_tfidf() -> TfidfVectorizer
```

### `models/tfidf_baseline.py`
```python
# Saves to:
#   models/artifacts/tfidf/model.joblib         (CalibratedClassifierCV)
#   models/artifacts/tfidf/vectorizer.joblib    (TfidfVectorizer)
#   artifacts/plots/shap_tfidf.png
#   data/predictions/tfidf_{train,val,test}.csv
#     columns: text, label, y_true, y_pred, confidence, prob_negative,
#              prob_neutral, prob_positive
# MLflow experiment: "finance_nlp_classification"
# MLflow run name:   "tfidf_logreg"
# Logged params: model, C, ngram_range, max_features, calibration_method, calibration_cv
# Logged metrics: val_macro_f1, val_accuracy, val_ece
```

---

## Prediction CSV schema (ALL models must match this exactly)

Every file saved to `data/predictions/{model}_{split}.csv` must have these columns:

| Column | Type | Description |
|---|---|---|
| `text` | str | original sentence |
| `label` | str | true label string ("negative" / "neutral" / "positive") |
| `y_true` | int | true label integer (0 / 1 / 2) |
| `y_pred` | int | predicted label integer |
| `confidence` | float | max probability across classes |
| `prob_negative` | float | P(negative) |
| `prob_neutral` | float | P(neutral) |
| `prob_positive` | float | P(positive) |

GPT-4 additionally has: `reasoning` (str), `parse_error` (bool)

---

## MLflow conventions

```python
mlflow.set_tracking_uri(str(ROOT / "mlruns"))    # project-local mlruns/
mlflow.set_experiment("finance_nlp_classification")  # all classifiers here
# summarization models log to experiment: "finance_nlp_summarization"
```

Run names:
- `"tfidf_logreg"` — Stage 2 (DONE)
- `"finbert_finetune"` — Stage 3
- `"gpt4_fewshot"` — Stage 4

---

## Stage 3 (FinBERT) — what to build next

Model: `ProsusAI/finbert`
- 3 epochs, lr=2e-5, batch_size=16, weight_decay=0.01, max_len=128
- Early stopping on val macro-F1, patience=2
- Save best checkpoint to `models/checkpoints/finbert_best/` (weights + tokenizer + temperature scalar)
- Temperature scaling fit on val logits, applied at inference
- Log per-epoch val F1 as MLflow metric curve (step = epoch number)
- Log final val ECE as scalar
- Save predictions to `data/predictions/finbert_{train,val,test}.csv` (same schema above)
- Save reliability diagram (before/after calibration) to `artifacts/plots/reliability_finbert.png`

Key learning annotations to include:
- What WordPiece tokenization does (subword splitting)
- What the [CLS] token is and why we classify from it
- What temperature scaling does geometrically (sharpens/softens softmax without changing argmax)
- What attention masks are (tell the model which tokens are real vs. padding)

---

## Stage 4 (GPT-4) — contract

- 6 few-shot examples in system prompt (2 per class, drawn from train, fixed)
- Output format: JSON `{"label": ..., "confidence": ..., "reasoning": ...}`
- Cache: `data/cache/{sha256_of_input}.json` — never re-call API for same input
- Malformed JSON → `parse_error=True`, excluded from metrics with count reported
- Log to MLflow: model_name, few_shot_count, prompt_hash, macro_f1

---

## Stage 5 (GPT-3.5 summarization) — contract

- Variant A: "Summarize this earnings call in 3 bullet points for a buy-side analyst."
- Variant B: "Extract the key financial guidance, risks, and outlook from this earnings call."
- Same cache mechanism as Stage 4
- Evaluate on ECTSum test split only
- Metrics: ROUGE-1/2/L, BERTScore F1
- Save 5 qualitative examples to `data/qualitative_examples.csv`

---

## Environment variables

All loaded via `python-dotenv` from `.env` at project root.

| Variable | Used in |
|---|---|
| `OPENAI_API_KEY` | `models/gpt_classifier.py`, `models/gpt_summarizer.py`, `serving/app.py` |

---

## FastAPI serving (Stage 7) — contract

- Loads FinBERT checkpoint at startup (not TF-IDF, not GPT-4)
- `POST /classify` → `{"label": str, "confidence": float, "probabilities": dict}`
- `POST /summarize` → `{"summary": str}` (calls GPT-3.5 with best-performing variant)
- `GET /health` → `{"status": "ok", "model": "finbert-finetuned"}`
- Input validation via Pydantic: reject empty strings, truncate at 512 tokens
- Run with: `uvicorn serving.app:app --reload`
