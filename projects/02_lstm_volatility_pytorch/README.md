# Time-Series Volatility Forecasting — SPY (S&P 500 ETF)

End-to-end volatility forecasting pipeline comparing four models — Naive, GARCH(1,1), LightGBM, and a PyTorch LSTM — across three forecast horizons on 14 years of SPY daily data (2010–2024). Covers feature engineering, strict leakage-free cross-validation, hyperparameter tuning, and multi-metric evaluation.

---

## What this project does

Realized volatility is one of the most important quantities in quantitative finance — it drives options pricing, risk budgeting, portfolio rebalancing, and margin calculations. This project asks a simple question: **can a sequence model (LSTM) forecast next-day, next-week, or next-month volatility better than classical econometric and tree-based alternatives?**

The answer is nuanced: LSTM consistently wins on point-forecast accuracy (RMSE), GARCH is more reliable under the likelihood-based QLIKE loss, and LightGBM leads on directional accuracy. No single model dominates all metrics — a finding that itself carries practical implications for ensemble design.

---

## Pipeline

```
SPY daily OHLCV (yfinance, 2010–2024)
           │
           ▼
  Log returns → HAR features (1d / 5d / 21d realized vol)
              + lagged squared & absolute returns
              + calendar features (day-of-week, month)
           │
           ▼
  Three forward targets (annualized log-vol):
      target_1d  |  target_5d  |  target_21d
           │
           ▼
  TimeSeriesSplit (5 folds, expanding window, gap = horizon)
           │
     ┌─────┼──────────┬──────────┐
     │     │          │          │
  Naive  GARCH(1,1) LightGBM   LSTM
   (persistence)  (rolling)  (SHAP)  (seq_len=21, 2-layer)
     │     │          │          │
     └─────┴──────────┴──────────┘
           │
           ▼
  Evaluation: RMSE · QLIKE · QLIKE (median) · Directional Accuracy
```

---

## Models

| Model | Description |
|---|---|
| **Naive** | Persistence baseline — predicts last observed realized volatility |
| **GARCH(1,1)** | Classical conditional heteroskedasticity model; rolled and re-fit each fold |
| **LightGBM** | Gradient-boosted regression on HAR + return features; SHAP feature importance |
| **LSTM** | PyTorch sequence model (seq_len=21, 2 layers, hidden=64, dropout=0.2, early stopping) |

---

## Results

All targets are in **annualized log-volatility space**: `log(rolling_std_of_returns × √252)`.  
Metrics are averaged across 5 time-series cross-validation folds. **Bold = best per metric.**

### 1-day horizon

| Model | RMSE | QLIKE | QLIKE (median) | QLIKE (trimmed 1%) | Dir. Acc. |
|---|---:|---:|---:|---:|---:|
| Naive | 1.6210 | 130.934 | -2.737 | 39.647 | 0.327 |
| GARCH | 1.4150 | **-2.999** | -3.527 | **-3.116** | 0.294 |
| LightGBM | 1.2881 | 3.430 | -3.488 | 1.064 | **0.528** |
| LSTM | **1.1926** | 0.187 | **-3.857** | -0.721 | 0.448 |

### 5-day horizon

| Model | RMSE | QLIKE | QLIKE (median) | QLIKE (trimmed 1%) | Dir. Acc. |
|---|---:|---:|---:|---:|---:|
| Naive | 0.6475 | -2.000 | -2.924 | -2.395 | 0.406 |
| GARCH | 0.6017 | **-2.922** | -3.272 | **-2.993** | 0.333 |
| LightGBM | 0.6456 | -2.275 | -3.180 | -2.596 | **0.483** |
| LSTM | **0.5746** | -2.549 | **-3.295** | -2.718 | 0.424 |

### 21-day horizon

| Model | RMSE | QLIKE | QLIKE (median) | QLIKE (trimmed 1%) | Dir. Acc. |
|---|---:|---:|---:|---:|---:|
| Naive | 0.4783 | -2.456 | -3.014 | -2.627 | 0.357 |
| GARCH | 0.4435 | **-2.738** | -3.146 | **-2.882** | 0.317 |
| LightGBM | 0.5421 | -2.285 | -3.006 | -2.586 | **0.472** |
| LSTM | **0.4229** | -2.606 | **-3.204** | -2.843 | 0.394 |

### Key takeaways

- **LSTM wins RMSE at every horizon** — 26% better than Naive at 1d, 12% better at 5d, 12% better at 21d.
- **GARCH leads on mean and trimmed QLIKE** at 5d and 21d, reflecting its well-specified distributional assumptions. Under a likelihood-based loss, the classic econometric model holds its own.
- **LightGBM leads directional accuracy** at all horizons — useful for regime-classification or convex/concave vol overlays where direction matters more than magnitude.
- **1d QLIKE is noisy** (Naive's 130.9 vs. all others near zero) due to occasional near-zero variance blowups; median and trimmed QLIKE are the reliable measures at short horizons.
- No single model dominates all three metrics — the natural implication is a metric-conditioned ensemble or use-case-driven model selection.

---

## Technical highlights

| Design decision | Why it matters |
|---|---|
| Targets computed from **future** returns only | Prevents any forward-looking information entering model inputs |
| `TimeSeriesSplit` with `gap = horizon` | Ensures val window never overlaps the training tail by at least one forecast horizon |
| Expanding window (train set grows each fold) | Simulates real deployment: models always use all available history |
| Scalers fit on **train fold only** | Prevents val/test distribution information leaking into feature normalization |
| LSTM early stopping on inner chronological split | Hyperparameter selection uses no future val-fold data |
| Log-vol target + volatility floor (1e-4) | Stabilizes QLIKE computation; avoids numerical issues from near-zero variance days |
| GARCH returns scaled ×100 | Improves optimizer convergence; standard practice for ARCH family models |
| SHAP on last fold's val set | Representative importance plot without aggregation artifacts |
| All RNGs seeded at 42 | Full cross-fold reproducibility |

---

## Feature engineering

| Feature | Description |
|---|---|
| `vol_1d`, `vol_5d`, `vol_21d` | HAR realized volatility (1-, 5-, 21-day windows, past-only) |
| `ret_sq_lag1`, `ret_abs_lag1` | Lagged squared and absolute returns |
| `dow`, `month` | Calendar features (day-of-week, month) |
| `target_1d/5d/21d` | Forward annualized log-realized-volatility (label, not input) |

HAR (Heterogeneous Autoregressive) features are a standard volatility forecasting benchmark from Corsi (2009), capturing the daily, weekly, and monthly components of the volatility term structure.

---

## Project layout

```
02_lstm_volatility_pytorch/
├── data/
│   ├── fetch.py                # SPY download via yfinance; cached to data/spy.csv
│   ├── features.csv            # Engineered features + all three forward targets
│   └── predictions/            # Per-fold predictions (model × horizon × fold)
├── features/
│   └── engineer.py             # Log returns, HAR features, targets
├── models/
│   ├── naive.py                # Persistence baseline
│   ├── garch.py                # GARCH(1,1) rolling forecast
│   ├── lgbm.py                 # LightGBM + SHAP explanations
│   └── lstm.py                 # PyTorch LSTM (shared architecture, per-horizon)
├── training/
│   └── cv.py                   # TimeSeriesSplit with gap, expanding window
├── evaluation/
│   └── metrics.py              # RMSE, QLIKE (mean/median/trimmed), directional accuracy
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
├── artifacts/
│   ├── plots/                  # SHAP feature-importance plots (per horizon)
│   └── summary/                # metrics_summary.csv, metrics_summary_wide.csv, tuning_summary.csv
├── run_experiments.py          # Orchestrator: all models × horizons × folds
└── requirements.txt
```

---

## Setup & reproduction

```bash
cd projects/02_lstm_volatility_pytorch
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt

# 1. Fetch raw SPY data (cached after first run)
python -m data.fetch

# 2. Run all experiments — 4 models × 3 horizons × 5 folds
python run_experiments.py

# Optional: control tuning
python run_experiments.py --tune lgbm,lstm   # tune specific models
python run_experiments.py --tune ""          # skip tuning entirely
```

---

## Outputs

| Artifact | Description |
|---|---|
| `data/spy.csv` | Raw SPY OHLCV |
| `data/features.csv` | Engineered features + forward targets |
| `data/predictions/{model}_{horizon}_fold{n}.csv` | Per-fold predictions with `date, y_true, y_pred, fold, model, horizon` |
| `artifacts/summary/metrics_summary.csv` | Long-format: model × horizon × metric |
| `artifacts/summary/metrics_summary_wide.csv` | Pivoted leaderboard |
| `artifacts/summary/tuning_summary.csv` | Per-fold best hyperparameters + inner-CV RMSE |
| `artifacts/plots/shap_lgbm_{horizon}.png` | SHAP feature-importance plots |

---

## Stack

`torch` · `lightgbm` · `arch` (GARCH) · `scikit-learn` · `yfinance` · `shap` · `pandas` · `numpy` · `matplotlib`
