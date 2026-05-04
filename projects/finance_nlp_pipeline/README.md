# Production-Style Financial NLP Pipeline

End-to-end NLP pipeline for financial text classification and summarization. Progresses from a TF-IDF bag-of-words baseline through fine-tuned FinBERT to GPT-4 few-shot classification, with MLflow experiment tracking, a FastAPI serving layer, and a model card. Every non-obvious implementation step includes an annotated explanation of the "why" — built as a learning project as much as a portfolio piece.

---

## What this project does

1. **Classifies financial sentiment** (negative / neutral / positive) on Financial PhraseBank sentences using three progressively more sophisticated approaches — TF-IDF + Logistic Regression, fine-tuned FinBERT, and GPT-4 few-shot — and compares them on accuracy, macro-F1, per-class F1, and Expected Calibration Error.
2. **Summarizes earnings call transcripts** from ECTSum using two GPT-3.5 prompt variants, evaluated with ROUGE-1/2/L and BERTScore to measure both lexical overlap and semantic quality.
3. **Tracks all experiments** with a local MLflow server — hyperparameters, per-epoch metrics, and artifact paths — so any run is reproducible and all models are comparable in a single dashboard.
4. **Serves the best classifier** as a REST API via FastAPI, with Pydantic input validation, temperature-scaled calibrated confidence scores, and a GPT-3.5 summarization endpoint backed by a response cache.

---

## Datasets

| Dataset | Task | Size | Source |
|---|---|---|---|
| **Financial PhraseBank** (`sentences_allagree`) | 3-class sentiment classification | ~4,840 sentences, 100% annotator agreement | Malo et al. (2014) via HuggingFace |
| **ECTSum** | Earnings call summarization (evaluation only) | ~2,425 transcripts + analyst summaries | ECTSum via HuggingFace |

Financial PhraseBank uses the `sentences_allagree` subset — only sentences where every annotator agreed on the label. This maximises label quality and makes the benchmark as clean as possible. ECTSum is used exclusively for summarization evaluation; none of it is used in classifier training.

---

## Pipeline

```
Financial PhraseBank (sentences_allagree)
           │
           ▼
  70 / 10 / 20 stratified split  (seed=42, written once to data/splits/)
           │
     ┌─────┼──────────────────────┐
     │     │                      │
  TF-IDF  FinBERT fine-tuned   GPT-4 few-shot
  + LogReg  (3 epochs, AdamW,   (6 examples,
  + isotonic  early stop,       JSON output,
  calibration  temp scaling)    cached calls)
     │     │                      │
     └─────┴──────────────────────┘
           │
           ▼
    MLflow experiment tracking
    (params · metrics · artifacts)
           │
           ▼
    Evaluation (test set only)
    Accuracy · Macro-F1 · Per-class F1 · ECE
    Reliability diagrams · Confusion matrices
    Top-10 confident errors · SHAP (TF-IDF)
           │
           ▼
    FastAPI  ──►  POST /classify   (FinBERT, local, ~50ms)
                  POST /summarize  (GPT-3.5, cached)
                  GET  /health

ECTSum (test split)
           │
           ▼
    GPT-3.5 · Variant A: "3 bullet points for a buy-side analyst"
            · Variant B: "guidance, risks, and outlook"
           │
           ▼
    ROUGE-1/2/L · BERTScore F1
    Prompt sensitivity: which variant wins per metric?
    5 qualitative examples → data/qualitative_examples.csv
```

---

## Project layout

```
finance_nlp_pipeline/
├── data/
│   ├── load.py                   # PhraseBank + ECTSum loaders; fixed splits written once
│   ├── splits/{train,val,test}.csv
│   ├── predictions/              # {model}_{split}.csv — one file per model × split
│   ├── cache/                    # SHA256-keyed GPT response cache (never re-calls API)
│   ├── ectsum.csv
│   └── qualitative_examples.csv  # 5 sampled summarization comparisons
├── features/
│   └── engineer.py               # clean_text, TF-IDF vectorizer, label encoding
├── models/
│   ├── tfidf_baseline.py         # TF-IDF + LogReg + isotonic calibration + SHAP
│   ├── finbert_finetune.py       # FinBERT fine-tuning + temperature scaling
│   ├── gpt_classifier.py         # GPT-4 few-shot with caching + parse-error handling
│   ├── gpt_summarizer.py         # GPT-3.5 two-variant summarization + ROUGE + BERTScore
│   ├── artifacts/tfidf/          # vectorizer.joblib, model.joblib
│   └── checkpoints/finbert_best/ # model weights + tokenizer + temperature.json
├── evaluation/
│   ├── metrics.py                # final test-set leaderboard → classification_summary.csv
│   ├── calibration.py            # 15-bin reliability diagrams for all models
│   └── error_analysis.py         # confusion matrices, top-10 errors, SHAP on errors
├── serving/
│   ├── app.py                    # FastAPI: /classify, /summarize, /health
│   └── model_card.md             # intended use, limitations, metrics, citations
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_results.ipynb
├── artifacts/
│   ├── plots/                    # shap_tfidf.png, reliability_*.png, confusion_matrices.png
│   └── summary/                  # classification_summary.csv, summarization_summary.csv
├── mlruns/                       # MLflow tracking (project-local, gitignored)
├── .env                          # OPENAI_API_KEY — never committed
├── requirements.txt
└── CLAUDE.md                     # implementation contract for AI-assisted development
```

---

## Setup

```powershell
cd projects/finance_nlp_pipeline
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# Add your OpenAI API key to .env
# Edit .env and replace: OPENAI_API_KEY=your-openai-api-key-here
```

---

## Reproduction — one command per stage

```powershell
# Stage 1 — load datasets and write fixed splits
python -m data.load

# Stage 2 — TF-IDF baseline + SHAP + MLflow
python -m models.tfidf_baseline

# Stage 3 — FinBERT fine-tuning + temperature scaling + MLflow
python -m models.finbert_finetune

# Stage 4 — GPT-4 few-shot classification (requires OPENAI_API_KEY)
python -m models.gpt_classifier

# Stage 5 — GPT-3.5 summarization, two variants, ROUGE + BERTScore
python -m models.gpt_summarizer

# Stage 6 — evaluation: metrics, calibration, error analysis
python -m evaluation.metrics
python -m evaluation.calibration
python -m evaluation.error_analysis

# Stage 7 — FastAPI serving
uvicorn serving.app:app --reload
```

---

## Results — classification (test set)

*Fill in from `artifacts/summary/classification_summary.csv` after running Stage 6.*

| Model | Accuracy | Macro-F1 | F1 (neg) | F1 (neu) | F1 (pos) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | — | — | — | — | — | — |
| FinBERT fine-tuned | — | — | — | — | — | — |
| GPT-4 few-shot | — | — | — | — | — | — |

**Key things to look for:**
- FinBERT vs TF-IDF macro-F1 gap — quantifies what fine-tuned transformers add over bag-of-words
- ECE before and after temperature scaling — visible in `artifacts/plots/reliability_all.png`
- GPT-4 parse error rate — logged per run in MLflow and printed at script completion
- Neutral class F1 — lowest across all models; neutral sentences are genuinely ambiguous

---

## Results — summarization (ECTSum test set)

*Fill in from `artifacts/summary/summarization_summary.csv` after running Stage 5.*

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| Variant A — 3 bullet points | — | — | — | — |
| Variant B — guidance / risks / outlook | — | — | — | — |

**Prompt sensitivity result:** report which variant wins each metric and by how much (Δ). A gap > 0.03 on ROUGE-2 or > 0.01 on BERTScore is meaningful. See `data/qualitative_examples.csv` for 5 side-by-side comparisons.

---

## MLflow experiment tracking

```powershell
# Start the UI (from inside finance_nlp_pipeline/)
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000 in your browser
```

You will see two experiments:
- **finance_nlp_classification** — three runs (tfidf_logreg, finbert_finetune, gpt4_fewshot)
  each with logged params, val macro-F1, val ECE, and artifact links
- **finance_nlp_summarization** — two runs (gpt35_variant_a, gpt35_variant_b)
  each with ROUGE-1/2/L and BERTScore F1

The FinBERT run shows a per-epoch val F1 metric curve — click the run, then "Metrics" to see whether F1 improved monotonically or triggered early stopping.

---

## FastAPI usage

```powershell
# Start server
uvicorn serving.app:app --reload

# Health check
curl http://localhost:8000/health
# → {"status":"ok","model":"finbert-finetuned"}

# Classify a sentence
curl -X POST http://localhost:8000/classify `
     -H "Content-Type: application/json" `
     -d '{"text": "The company reported record quarterly earnings, beating analyst estimates."}'
# → {"label":"positive","confidence":0.97,"probabilities":{"negative":0.01,"neutral":0.02,"positive":0.97},"truncated":false}

# Summarize an earnings call excerpt
curl -X POST http://localhost:8000/summarize `
     -H "Content-Type: application/json" `
     -d '{"text": "Good morning. Revenue for the quarter came in at $4.2 billion, up 18% year-over-year..."}'
# → {"summary":"• Revenue grew 18% YoY to $4.2B...", "variant_used":"variant_a"}

# Interactive API docs (Swagger UI)
# Open http://localhost:8000/docs
```

---

## Technical highlights

| Design decision | Why it matters |
|---|---|
| All splits written to CSV once | Every model reads identical data; results are fully comparable |
| TF-IDF fit on train only | Prevents IDF weights encoding val/test vocabulary — data leakage |
| Temperature scaling fit on val, applied to test | Calibration never sees test labels |
| SHA256-keyed API response cache | Zero redundant API calls; re-runs cost nothing |
| GPT-4 parse error handling | Malformed JSON is flagged, not a crash; production-style fault tolerance |
| Forward-looking SHAP on TF-IDF errors | Identifies which tokens actively mislead the model on failures |
| FinBERT served (not GPT-4) | Latency (~50ms vs ~2s), cost ($0 vs ~$0.02/call), offline operation, data privacy |
| Model card with OOD limitation | Documents earnings call performance degradation before deployment |
| All RNGs seeded at 42 | End-to-end reproducibility across all models and splits |

---

## Resume bullets

- **Fine-tuned FinBERT** (ProsusAI/finbert, 110M params) for 3-class financial sentiment classification on Financial PhraseBank, achieving **[X]% macro-F1** — a **[Y] pp improvement** over a TF-IDF + Logistic Regression baseline — with temperature-scaled calibration reducing ECE by **[Z]%**, tracked across **3 experiment runs** in MLflow.

- **Deployed fine-tuned FinBERT as a REST API** via FastAPI with Pydantic input validation, calibrated confidence scores, and a GPT-3.5 summarization endpoint backed by a SHA256 response cache — serving real-time financial sentiment predictions at ~50ms latency with zero API cost per classification.

- **Benchmarked GPT-3.5 summarization** across two prompt variants on **[N] ECTSum earnings call transcripts**, reporting ROUGE-1/2/L and BERTScore F1 per variant; measured prompt sensitivity (Δ ROUGE-2 = **[X]**, Δ BERTScore = **[Y]**) and documented results in a structured model card with OOD limitations and MRM-aligned responsible deployment guidelines.

---

## Stack

`transformers` · `torch` · `openai` · `fastapi` · `uvicorn` · `mlflow` · `scikit-learn` · `datasets` · `rouge-score` · `bert-score` · `shap` · `netcal` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `pydantic` · `python-dotenv`
