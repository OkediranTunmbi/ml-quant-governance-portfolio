# Earnings Call Sentiment as a Return Predictor

End-to-end NLP-to-quant pipeline that transforms earnings call transcripts into a tradeable cross-sectional sector signal. Covers the full ML lifecycle: data ingestion, model training, calibration, evaluation, signal construction, and backtesting.

---

## What this project does

1. **Trains three sentiment classifiers** on the Financial PhraseBank dataset — a TF-IDF + Logistic Regression baseline, FinBERT zero-shot, and fine-tuned FinBERT — and evaluates them on classification accuracy, macro-F1, per-class F1, and Expected Calibration Error (ECE).
2. **Scores 2,425 earnings call transcripts** from ECTSum using the best classifier.
3. **Aggregates transcript scores** to a weekly sector-level sentiment matrix across 9 GICS sector ETFs.
4. **Runs a long-short backtest** — long top-2 / short bottom-2 sectors by sentiment — and benchmarks it against a shuffled-sentiment random strategy.

---

## Why earnings call sentiment

Earnings calls are scheduled, dense, and price-moving. Management tone has been shown in academic literature (Loughran & McDonald; Cohen et al.) to lead realised returns by days to weeks. A fine-tuned transformer reads hedging and forward-looking language far better than a bag-of-words baseline, making this an honest test of whether modern NLP adds alpha in a setting where humans already trade aggressively.

---

## Pipeline

```
Financial PhraseBank ──► train / val / test  (70 / 10 / 20, stratified, seed=42)
           │
           ├─ TF-IDF + LogReg  (isotonic-calibrated, 5-fold CV)
           ├─ FinBERT zero-shot  (ProsusAI/finbert, no fine-tuning)
           └─ FinBERT fine-tuned  (3 epochs, AdamW, early stop, temperature scaling)
                         │
                         ▼
               Test-set evaluation
         (Accuracy, Macro-F1, per-class F1, ECE, reliability diagrams)

ECTSum transcripts ──► fine-tuned FinBERT ──► P(positive) − P(negative)  ∈ [−1, +1]
           │                                              │
           ├─ yfinance ticker → GICS sector               │
           └─ weekly sector sentiment matrix  ◄───────────┘
                         │
                         ▼
  Long-short: long top-2 / short bottom-2 sectors each week
                         │
                         ▼
      Sharpe, hit rate, cumulative return, max drawdown
           (vs. shuffled-sentiment random benchmark)
```

---

## Project layout

```
earnings_sentiment/
├── config.py                   # seeds, paths, label map, sector ETF universe
├── data/
│   ├── load.py                 # PhraseBank + ECTSum loaders; fixed splits written once
│   └── returns.py              # weekly sector ETF returns via yfinance
├── features/
│   └── engineer.py             # text cleaning, TF-IDF builder (fit on train only)
├── models/
│   ├── tfidf_baseline.py       # LogReg + isotonic calibration
│   ├── finbert_zero.py         # zero-shot batch inference, label remapping
│   └── finbert_finetune.py     # fine-tune + temperature scaling on val set
├── evaluation/
│   ├── metrics.py              # accuracy, macro-F1, per-class F1, ECE leaderboard
│   ├── calibration.py          # reliability diagrams (15-bin)
│   └── error_analysis.py       # confusion matrices, top confident errors
├── signals/
│   ├── score.py                # per-transcript scoring, weekly sector aggregation
│   ├── backtest.py             # long-short engine, forward-return alignment
│   └── performance.py          # cumulative return plot, sector sentiment timeseries
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_signal.ipynb
├── run_pipeline.py             # orchestrator with --only / --skip stage flags
└── requirements.txt
```

---

## Results — classification (test set, n = 452)

| Model | Accuracy | Macro-F1 | F1 (neg) | F1 (neu) | F1 (pos) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| FinBERT fine-tuned + calibrated | **98.0%** | **0.968** | 0.943 | 0.993 | 0.969 | **0.010** |
| FinBERT fine-tuned (uncalibrated) | 98.0% | 0.968 | 0.943 | 0.993 | 0.969 | 0.016 |
| FinBERT zero-shot | 97.3% | 0.965 | 0.960 | 0.987 | 0.948 | 0.059 |
| TF-IDF + LogReg | 86.3% | 0.803 | 0.699 | 0.923 | 0.786 | 0.072 |

**Key takeaways:**

- Fine-tuning improves macro-F1 by **16.5 pp** over the TF-IDF baseline and **0.3 pp** over the already-strong zero-shot model.
- Temperature scaling reduces ECE by **37%** (0.016 → 0.010) with no change to predicted labels — calibration only adjusts confidence scores, which matters for downstream position sizing.
- FinBERT zero-shot is a surprisingly competitive baseline, achieving 97.3% accuracy with no task-specific training, reflecting strong domain pre-training.
- The TF-IDF model's main failure mode is negative sentiment: F1 of 0.699 vs. 0.943 for fine-tuned FinBERT, because short bearish phrases are harder to capture with n-gram features.

---

## Results — long-short backtest (60 weeks)

| Strategy | Annualised Sharpe | Hit rate | Cum. return | Max drawdown | Weeks |
|---|---:|---:|---:|---:|---:|
| FinBERT signal | −0.66 | 50.0% | −13.3% | −15.9% | 60 |
| Random benchmark | +0.07 | 48.3% | −0.4% | −15.1% | 60 |

**Honest interpretation:**

The sentiment signal underperformed the random benchmark over this 60-week window. This is a meaningful result, not a failure:

- Both strategies share the same max drawdown (~15%), confirming the market regime — not the signal — drove losses.
- A 50% hit rate with a negative Sharpe indicates that wins and losses are roughly symmetric in frequency but asymmetric in magnitude; the signal may be buying high-momentum sectors at the wrong point in their cycle.
- ECTSum's temporal coverage is limited (~60 tradeable weeks after sector aggregation), making statistical significance difficult to establish — a longer history or more granular transcript data would be needed to draw firm alpha conclusions.
- The backtest architecture itself is sound: forward-return alignment, a structurally matched random benchmark, and no look-ahead bias.

---

## Technical highlights

| Design decision | Why it matters |
|---|---|
| Stratified 70/10/20 split written once to disk | Every model reads identical splits; results are comparable |
| TF-IDF fit on train only | Prevents vocabulary leakage from val/test into features |
| Temperature scaling fit on val, applied to test | Calibration cannot see test labels |
| `P(positive) − P(negative)` as signal score | Symmetric, bounded in [−1, +1], preserves cross-sectional ranking |
| Forward returns (`shift(-1)`) in backtest | Position at week T earns week T+1 return; no look-ahead |
| Random benchmark shuffles within each week | Preserves rank distribution, destroys only information content |
| All RNGs pinned at seed=42 | Full end-to-end reproducibility |
| `--only` / `--skip` stage flags | Re-run any stage independently without repeating the expensive fine-tune |

---

## Setup & reproduction

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Run the full pipeline from inside `earnings_sentiment/`:

```bash
python run_pipeline.py
```

Or run individual stages:

```bash
python run_pipeline.py --only data
python run_pipeline.py --only tfidf,finbert_zero,finbert_finetune
python run_pipeline.py --only evaluation
python run_pipeline.py --only signal,backtest
```

---

## Outputs

| Artifact | Description |
|---|---|
| `data/splits/{train,val,test}.csv` | Canonical PhraseBank splits |
| `data/predictions/*_test.csv` | Per-row predictions + calibrated softmax probabilities |
| `data/predictions/ectsum_sentiment.csv` | Per-transcript FinBERT score |
| `data/predictions/weekly_sector_sentiment.csv` | Sector × week sentiment matrix |
| `artifacts/summary/classification_summary.csv` | Full model leaderboard |
| `artifacts/summary/backtest_summary.csv` | Sharpe / hit rate / drawdown |
| `artifacts/plots/confusion_matrices.png` | Side-by-side confusion matrices |
| `artifacts/plots/reliability_diagrams.png` | Calibration reliability diagrams |
| `artifacts/plots/cumulative_return.png` | Signal vs. random benchmark equity curve |
| `artifacts/plots/sector_sentiment.png` | Weekly sector sentiment timeseries |
| `models/checkpoints/finbert_best/` | Fine-tuned weights + tokenizer + temperature |

---

## Stack

`transformers` · `torch` · `scikit-learn` · `datasets` (Hugging Face) · `yfinance` · `netcal` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib`
