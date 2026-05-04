# Production-Style Financial NLP Pipeline

End-to-end NLP pipeline for financial text classification and summarization. Progresses from a TF-IDF bag-of-words baseline through fine-tuned FinBERT to GPT-4 few-shot classification, with MLflow experiment tracking, a FastAPI serving layer, and a model card. Every non-obvious implementation step includes an annotated explanation of the "why" â€” built as a learning project as much as a portfolio piece.

---

## What this project does

1. **Classifies financial sentiment** (negative / neutral / positive) on Financial PhraseBank sentences using three progressively more sophisticated approaches â€” TF-IDF + Logistic Regression, fine-tuned FinBERT, and GPT-4 few-shot â€” and compares them on accuracy, macro-F1, per-class F1, and Expected Calibration Error.
2. **Summarizes earnings call transcripts** from ECTSum using two GPT-3.5 prompt variants, evaluated with ROUGE-1/2/L and BERTScore to measure both lexical overlap and semantic quality.
3. **Tracks all experiments** with a local MLflow server â€” hyperparameters, per-epoch metrics, and artifact paths â€” so any run is reproducible and all models are comparable in a single dashboard.
4. **Serves the best classifier** as a REST API via FastAPI, with Pydantic input validation, temperature-scaled calibrated confidence scores, and a GPT-3.5 summarization endpoint backed by a response cache.

---

## Datasets

| Dataset | Task | Size | Source |
|---|---|---|---|
| **Financial PhraseBank** (`sentences_allagree`) | 3-class sentiment classification | ~4,840 sentences, 100% annotator agreement | Malo et al. (2014) via HuggingFace |
| **ECTSum** | Earnings call summarization (evaluation only) | ~2,425 transcripts + analyst summaries | ECTSum via HuggingFace |

Financial PhraseBank uses the `sentences_allagree` subset â€” only sentences where every annotator agreed on the label. This maximises label quality and makes the benchmark as clean as possible. ECTSum is used exclusively for summarization evaluation; none of it is used in classifier training.

---

## Pipeline

```
Financial PhraseBank (sentences_allagree)
           â”‚
           â–¼
  70 / 10 / 20 stratified split  (seed=42, written once to data/splits/)
           â”‚
     â”Œâ”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
     â”‚     â”‚                      â”‚
  TF-IDF  FinBERT fine-tuned   GPT-4 few-shot
  + LogReg  (3 epochs, AdamW,   (6 examples,
  + isotonic  early stop,       JSON output,
  calibration  temp scaling)    cached calls)
     â”‚     â”‚                      â”‚
     â””â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
           â”‚
           â–¼
    MLflow experiment tracking
    (params Â· metrics Â· artifacts)
           â”‚
           â–¼
    Evaluation (test set only)
    Accuracy Â· Macro-F1 Â· Per-class F1 Â· ECE
    Reliability diagrams Â· Confusion matrices
    Top-10 confident errors Â· SHAP (TF-IDF)
           â”‚
           â–¼
    FastAPI  â”€â”€â–º  POST /classify   (FinBERT, local, ~50ms)
                  POST /summarize  (GPT-3.5, cached)
                  GET  /health

ECTSum (test split)
           â”‚
           â–¼
    GPT-3.5 Â· Variant A: "3 bullet points for a buy-side analyst"
            Â· Variant B: "guidance, risks, and outlook"
           â”‚
           â–¼
    ROUGE-1/2/L Â· BERTScore F1
    Prompt sensitivity: which variant wins per metric?
    5 qualitative examples â†’ data/qualitative_examples.csv
```

---

## Project layout

```
finance_nlp_pipeline/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ load.py                   # PhraseBank + ECTSum loaders; fixed splits written once
â”‚   â”œâ”€â”€ splits/{train,val,test}.csv
â”‚   â”œâ”€â”€ predictions/              # {model}_{split}.csv â€” one file per model Ã— split
â”‚   â”œâ”€â”€ cache/                    # SHA256-keyed GPT response cache (never re-calls API)
â”‚   â”œâ”€â”€ ectsum.csv
â”‚   â””â”€â”€ qualitative_examples.csv  # 5 sampled summarization comparisons
â”œâ”€â”€ features/
â”‚   â””â”€â”€ engineer.py               # clean_text, TF-IDF vectorizer, label encoding
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ tfidf_baseline.py         # TF-IDF + LogReg + isotonic calibration + SHAP
â”‚   â”œâ”€â”€ finbert_finetune.py       # FinBERT fine-tuning + temperature scaling
â”‚   â”œâ”€â”€ gpt_classifier.py         # GPT-4 few-shot with caching + parse-error handling
â”‚   â”œâ”€â”€ gpt_summarizer.py         # GPT-3.5 two-variant summarization + ROUGE + BERTScore
â”‚   â”œâ”€â”€ artifacts/tfidf/          # vectorizer.joblib, model.joblib
â”‚   â””â”€â”€ checkpoints/finbert_best/ # model weights + tokenizer + temperature.json
â”œâ”€â”€ evaluation/
â”‚   â”œâ”€â”€ metrics.py                # final test-set leaderboard â†’ classification_summary.csv
â”‚   â”œâ”€â”€ calibration.py            # 15-bin reliability diagrams for all models
â”‚   â””â”€â”€ error_analysis.py         # confusion matrices, top-10 errors, SHAP on errors
â”œâ”€â”€ serving/
â”‚   â”œâ”€â”€ app.py                    # FastAPI: /classify, /summarize, /health
â”‚   â””â”€â”€ model_card.md             # intended use, limitations, metrics, citations
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_eda.ipynb
â”‚   â”œâ”€â”€ 02_error_analysis.ipynb
â”‚   â””â”€â”€ 03_results.ipynb
â”œâ”€â”€ artifacts/
â”‚   â”œâ”€â”€ plots/                    # shap_tfidf.png, reliability_*.png, confusion_matrices.png
â”‚   â””â”€â”€ summary/                  # classification_summary.csv, summarization_summary.csv
â”œâ”€â”€ mlruns/                       # MLflow tracking (project-local, gitignored)
â”œâ”€â”€ .env                          # OPENAI_API_KEY â€” never committed
â”œâ”€â”€ requirements.txt
â””â”€â”€ CLAUDE.md                     # implementation contract for AI-assisted development
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

## Reproduction â€” one command per stage

```powershell
# Stage 1 â€” load datasets and write fixed splits
python -m data.load

# Stage 2 â€” TF-IDF baseline + SHAP + MLflow
python -m models.tfidf_baseline

# Stage 3 â€” FinBERT fine-tuning + temperature scaling + MLflow
python -m models.finbert_finetune

# Stage 4 â€” GPT-4 few-shot classification (requires OPENAI_API_KEY)
python -m models.gpt_classifier

# Stage 5 â€” GPT-3.5 summarization, two variants, ROUGE + BERTScore
python -m models.gpt_summarizer

# Stage 6 â€” evaluation: metrics, calibration, error analysis
python -m evaluation.metrics
python -m evaluation.calibration
python -m evaluation.error_analysis

# Stage 7 â€” FastAPI serving
uvicorn serving.app:app --reload
```

---

## Results â€” classification (test set)

*Fill in from `artifacts/summary/classification_summary.csv` after running Stage 6.*

| Model | Accuracy | Macro-F1 | F1 (neg) | F1 (neu) | F1 (pos) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | â€” | â€” | â€” | â€” | â€” | â€” |
| FinBERT fine-tuned | â€” | â€” | â€” | â€” | â€” | â€” |
| GPT-4 few-shot | â€” | â€” | â€” | â€” | â€” | â€” |

**Key things to look for:**
- FinBERT vs TF-IDF macro-F1 gap â€” quantifies what fine-tuned transformers add over bag-of-words
- ECE before and after temperature scaling â€” visible in `artifacts/plots/reliability_all.png`
- GPT-4 parse error rate â€” logged per run in MLflow and printed at script completion
- Neutral class F1 â€” lowest across all models; neutral sentences are genuinely ambiguous

---

## Results â€” summarization (ECTSum test set)

*Fill in from `artifacts/summary/summarization_summary.csv` after running Stage 5.*

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| Variant A â€” 3 bullet points | â€” | â€” | â€” | â€” |
| Variant B â€” guidance / risks / outlook | â€” | â€” | â€” | â€” |

**Prompt sensitivity result:** report which variant wins each metric and by how much (Î”). A gap > 0.03 on ROUGE-2 or > 0.01 on BERTScore is meaningful. See `data/qualitative_examples.csv` for 5 side-by-side comparisons.

---

## MLflow experiment tracking

```powershell
# Start the UI (from inside finance_nlp_pipeline/)
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000 in your browser
```

You will see two experiments:
- **finance_nlp_classification** â€” three runs (tfidf_logreg, finbert_finetune, gpt4_fewshot)
  each with logged params, val macro-F1, val ECE, and artifact links
- **finance_nlp_summarization** â€” two runs (gpt35_variant_a, gpt35_variant_b)
  each with ROUGE-1/2/L and BERTScore F1

The FinBERT run shows a per-epoch val F1 metric curve â€” click the run, then "Metrics" to see whether F1 improved monotonically or triggered early stopping.

---

## FastAPI usage

```powershell
# Start server
uvicorn serving.app:app --reload

# Health check
curl http://localhost:8000/health
# â†’ {"status":"ok","model":"finbert-finetuned"}

# Classify a sentence
curl -X POST http://localhost:8000/classify `
     -H "Content-Type: application/json" `
     -d '{"text": "The company reported record quarterly earnings, beating analyst estimates."}'
# â†’ {"label":"positive","confidence":0.97,"probabilities":{"negative":0.01,"neutral":0.02,"positive":0.97},"truncated":false}

# Summarize an earnings call excerpt
curl -X POST http://localhost:8000/summarize `
     -H "Content-Type: application/json" `
     -d '{"text": "Good morning. Revenue for the quarter came in at $4.2 billion, up 18% year-over-year..."}'
# â†’ {"summary":"â€¢ Revenue grew 18% YoY to $4.2B...", "variant_used":"variant_a"}

# Interactive API docs (Swagger UI)
# Open http://localhost:8000/docs
```

---

## Technical highlights

| Design decision | Why it matters |
|---|---|
| All splits written to CSV once | Every model reads identical data; results are fully comparable |
| TF-IDF fit on train only | Prevents IDF weights encoding val/test vocabulary â€” data leakage |
| Temperature scaling fit on val, applied to test | Calibration never sees test labels |
| SHA256-keyed API response cache | Zero redundant API calls; re-runs cost nothing |
| GPT-4 parse error handling | Malformed JSON is flagged, not a crash; production-style fault tolerance |
| Forward-looking SHAP on TF-IDF errors | Identifies which tokens actively mislead the model on failures |
| FinBERT served (not GPT-4) | Latency (~50ms vs ~2s), cost ($0 vs ~$0.02/call), offline operation, data privacy |
| Model card with OOD limitation | Documents earnings call performance degradation before deployment |
| All RNGs seeded at 42 | End-to-end reproducibility across all models and splits |

---

## Stack

`transformers` Â· `torch` Â· `openai` Â· `fastapi` Â· `uvicorn` Â· `mlflow` Â· `scikit-learn` Â· `datasets` Â· `rouge-score` Â· `bert-score` Â· `shap` Â· `netcal` Â· `pandas` Â· `numpy` Â· `matplotlib` Â· `seaborn` Â· `pydantic` Â· `python-dotenv`

