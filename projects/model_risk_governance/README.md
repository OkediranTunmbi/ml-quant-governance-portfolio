# Model Risk Governance Toolkit â€” Lending Club Credit Risk

A production-oriented model risk management toolkit built around a Logistic Regression
credit default classifier trained on Lending Club data. Covers the full SR 11-7 model
lifecycle: data validation, model development, independent validation, drift detection,
calibration monitoring, fairness analysis, threshold governance, and a bank-style
validation report.

---

## Project Motivation: SR 11-7 Model Risk Management in Practice

SR 11-7 (Supervisory Guidance on Model Risk Management, Federal Reserve / OCC, 2011)
is the primary US regulatory framework governing model risk at banks. It requires:

1. **Model development** â€” documented conceptual soundness, data quality, performance evaluation
2. **Model validation** â€” independent review by a team that did not build the model
3. **Ongoing monitoring** â€” drift detection, performance tracking, periodic review

This toolkit implements each of those requirements programmatically, so every stage is
reproducible, auditable, and backed by statistical evidence.

---

## Pipeline Diagram

```
data/raw/lending_club.csv
        â”‚
        â–¼
[toolkit/data_validation.py]   â† Schema checks, leakage exclusions, target binarisation
        â”‚
        â–¼
[toolkit/preprocessing.py]     â† Feature engineering, OrdinalEncoder, OHE, StandardScaler
        â”‚           (fit on train only)
        â–¼
[toolkit/model.py]             â† Logistic Regression, AUC/Gini/KS evaluation, SHAP
        â”‚
        â”œâ”€â”€â–º [toolkit/threshold_governance.py]  â† Threshold sweep, 3 candidate thresholds
        â”‚
        â”œâ”€â”€â–º [toolkit/drift.py]                 â† PSI, KS test, prediction drift, target drift
        â”‚    [monitoring/evidently_dashboard.py] â† Visual drift/quality/performance dashboards
        â”‚
        â”œâ”€â”€â–º [toolkit/calibration.py]            â† Platt scaling, ECE, reliability diagrams
        â”‚
        â”œâ”€â”€â–º [toolkit/fairness.py]               â† 80% rule, equalized odds, predictive parity
        â”‚
        â””â”€â”€â–º [toolkit/report.py]                 â† Jinja2 HTML validation report
                        â”‚
                        â–¼
             reports/output/validation_report_YYYY-MM-DD.html
```

---

## How to Run

### Setup

```bash
# From the model_risk_governance/ directory
pip install -r requirements.txt
```


### Stage-by-stage

| Stage | Command |
|---|---|
| 1. EDA | `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb` |
| 2. Model development | `jupyter nbconvert --to notebook --execute notebooks/02_model_development.ipynb` |
| 3. Validation | `jupyter nbconvert --to notebook --execute notebooks/03_validation.ipynb` |
| 4. Governance report | `jupyter nbconvert --to notebook --execute notebooks/04_governance_report.ipynb` |

Or run interactively in JupyterLab:

```bash
jupyter lab
```

### Individual toolkit modules (from Python)

```python
# Data validation
from toolkit.data_validation import validate
df_clean = validate(raw_df)

# PSI drift
from toolkit.drift import run_drift_report
results = run_drift_report(train_df, monitor_df, ...)

# Threshold governance
from toolkit.threshold_governance import run_threshold_governance
results = run_threshold_governance(y_true, y_scores)

# Render report
from toolkit.report import render_report
path = render_report(results)
```

---

## Viewing Outputs

### Evidently Dashboards (open in browser)

```
reports/output/evidently_drift.html        â† Feature distribution drift (train vs monitor)
reports/output/evidently_quality.html      â† Data quality on monitor set
reports/output/evidently_performance.html  â† Classification performance comparison
```

### Validation Report

```
reports/output/validation_report_YYYY-MM-DD.html
```

Open in any browser. The report is fully self-contained (all images are base64-embedded).

---

## Project Structure

```
model_risk_governance/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/lending_club.csv          â† Source data (place here)
â”‚   â””â”€â”€ processed/                    â† Train/monitor parquets, model pkl, metrics JSON
â”œâ”€â”€ toolkit/
â”‚   â”œâ”€â”€ data_validation.py            â† Schema checks, leakage exclusions
â”‚   â”œâ”€â”€ preprocessing.py              â† Feature engineering, encoding, scaling
â”‚   â”œâ”€â”€ model.py                      â† LR training, evaluation, SHAP
â”‚   â”œâ”€â”€ drift.py                      â† PSI, KS, prediction/target drift
â”‚   â”œâ”€â”€ calibration.py                â† Platt scaling, ECE, reliability diagrams
â”‚   â”œâ”€â”€ fairness.py                   â† Disparate impact, equalized odds
â”‚   â”œâ”€â”€ threshold_governance.py       â† Threshold sweep, 3 candidate thresholds
â”‚   â””â”€â”€ report.py                     â† Jinja2 report generator
â”œâ”€â”€ monitoring/
â”‚   â””â”€â”€ evidently_dashboard.py        â† Evidently drift + quality + performance
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_eda.ipynb                  â† EDA, class balance, temporal patterns
â”‚   â”œâ”€â”€ 02_model_development.ipynb    â† Training, SHAP, threshold selection
â”‚   â”œâ”€â”€ 03_validation.ipynb           â† Drift, calibration, fairness
â”‚   â””â”€â”€ 04_governance_report.ipynb    â† Evidently + HTML report rendering
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ templates/validation_report.html
â”‚   â””â”€â”€ output/                       â† Generated reports
â”œâ”€â”€ model_card.md
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## Key Concepts Covered

| Concept | Where |
|---|---|
| SR 11-7 three-lines-of-defense | `notebooks/03_validation.ipynb` |
| Data leakage taxonomy | `toolkit/data_validation.py` |
| Train-only preprocessing | `toolkit/preprocessing.py` + notebook 02 |
| Gini coefficient & KS statistic | `toolkit/model.py` |
| SHAP explainability | `toolkit/model.py` + notebook 02 |
| PSI thresholds (0.10 / 0.25) | `toolkit/drift.py` + notebook 03 |
| Platt scaling / ECE | `toolkit/calibration.py` + notebook 03 |
| Fairness impossibility theorem | `notebooks/03_validation.ipynb` Section 5 |
| ECOA 80% disparate impact rule | `toolkit/fairness.py` |
| Threshold as business decision | `toolkit/threshold_governance.py` + notebook 02 |
| Independent model validation | `toolkit/report.py` + notebook 04 |

