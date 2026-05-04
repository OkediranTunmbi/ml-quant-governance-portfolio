# ML Quant Governance Portfolio

Portfolio of production-style projects at the intersection of:

- Quant Research
- AI/ML Engineering (Financial Services)
- AI Governance / Model Risk Management

Each project is designed to show end-to-end ownership: data pipelines, modeling, evaluation, reproducibility, and risk-aware communication.

## Role Fit

| Target role | Evidence in this repo |
|---|---|
| Quant Research | Signal construction, backtesting, volatility forecasting, horizon-based evaluation |
| AI/ML Engineer (Finance) | Train/serve pipelines, FastAPI deployment, MLflow tracking, calibration, error analysis |
| AI Governance / Model Risk | SR 11-7 aligned validation workflow, drift/fairness/calibration monitoring, model cards and reports |

## Projects

| Project | Focus | Key outputs |
|---|---|---|
| [`finance_nlp_pipeline`](projects/finance_nlp_pipeline) | Financial sentiment + summarization with TF-IDF, FinBERT, GPT few-shot, FastAPI serving | Classification and summarization benchmarks, model card, API endpoints |
| [`earnings_sentiment`](projects/earnings_sentiment) | Earnings-call sentiment transformed into sector long/short trading signal | Classifier leaderboard, weekly sector sentiment, backtest performance |
| [`02_lstm_volatility_pytorch`](projects/02_lstm_volatility_pytorch) | Multi-horizon volatility forecasting with naive, GARCH, LightGBM, LSTM | Horizon-level RMSE/QLIKE/Direction metrics, SHAP analyses |
| [`model_risk_governance`](projects/model_risk_governance) | Credit risk model governance toolkit (SR 11-7 style) | Drift, calibration, fairness, threshold governance, validation report artifacts |

## Key Results Snapshot

### Financial NLP classification (test set, n=453)

Source: `projects/finance_nlp_pipeline/artifacts/summary/classification_summary.csv`

| Model | Accuracy | Macro-F1 | ECE |
|---|---:|---:|---:|
| FinBERT fine-tuned | **0.9801** | **0.9683** | **0.0151** |
| GPT-4 few-shot | 0.9691 | 0.9675 | 0.0566 |
| TF-IDF + Logistic Regression | 0.8521 | 0.7854 | 0.0569 |

### Financial NLP summarization (ECTSum test split)

Source: `projects/finance_nlp_pipeline/artifacts/summary/summarization_summary.csv`

| Prompt variant | ROUGE-1 | ROUGE-2 | ROUGE-L | BertScore F1 |
|---|---:|---:|---:|---:|
| Variant A | **0.2022** | **0.0717** | **0.1349** | **0.0559** |
| Variant B | 0.1486 | 0.0526 | 0.0938 | -0.0365 |

### Volatility forecasting (best RMSE by horizon)

Source: `projects/02_lstm_volatility_pytorch/artifacts/summary/metrics_summary.csv`

| Horizon | Best model | RMSE |
|---|---|---:|
| 1-day | LSTM | **1.1926** |
| 5-day | LSTM | **0.5746** |
| 21-day | LSTM | **0.4229** |

### Quant signal backtest (earnings sentiment)

Source: `projects/earnings_sentiment/README.md`

| Strategy | Annualized Sharpe | Hit rate | Cumulative return |
|---|---:|---:|---:|
| FinBERT signal | -0.66 | 50.0% | -13.3% |
| Random benchmark | 0.07 | 48.3% | -0.4% |

The underperformance is intentionally documented as part of honest research communication and model-risk-aware reporting.

## Governance and Reproducibility Themes

- Reproducible train/validation/test splits and seeded experiments
- Calibration (ECE, reliability curves, temperature or Platt scaling)
- Drift monitoring and fairness diagnostics
- Clear model limitations via model cards and validation reports
- MLflow-tracked experimentation for auditability

## How To Navigate

1. Start with project-level READMEs in [`projects/`](projects).
2. Review `artifacts/summary` and `artifacts/plots` folders for outcomes.
3. Open model cards and governance reports for deployment context and risk controls.

## Next Step

A recruiter-facing project website is planned to showcase these projects, results, and demos in one place.
