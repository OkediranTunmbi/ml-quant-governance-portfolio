# Model Risk Governance Toolkit — Lending Club Credit Risk

A production-oriented model risk management toolkit built around a Logistic Regression
credit default classifier trained on Lending Club data. Covers the full SR 11-7 model
lifecycle: data validation, model development, independent validation, drift detection,
calibration monitoring, fairness analysis, threshold governance, and a bank-style
validation report.

---

## Project Motivation: SR 11-7 Model Risk Management in Practice

SR 11-7 (Supervisory Guidance on Model Risk Management, Federal Reserve / OCC, 2011)
is the primary US regulatory framework governing model risk at banks. It requires:

1. **Model development** — documented conceptual soundness, data quality, performance evaluation
2. **Model validation** — independent review by a team that did not build the model
3. **Ongoing monitoring** — drift detection, performance tracking, periodic review

This toolkit implements each of those requirements programmatically, so every stage is
reproducible, auditable, and backed by statistical evidence.

---

## Pipeline Diagram

```
data/raw/lending_club.csv
        │
        ▼
[toolkit/data_validation.py]   ← Schema checks, leakage exclusions, target binarisation
        │
        ▼
[toolkit/preprocessing.py]     ← Feature engineering, OrdinalEncoder, OHE, StandardScaler
        │           (fit on train only)
        ▼
[toolkit/model.py]             ← Logistic Regression, AUC/Gini/KS evaluation, SHAP
        │
        ├──► [toolkit/threshold_governance.py]  ← Threshold sweep, 3 candidate thresholds
        │
        ├──► [toolkit/drift.py]                 ← PSI, KS test, prediction drift, target drift
        │    [monitoring/evidently_dashboard.py] ← Visual drift/quality/performance dashboards
        │
        ├──► [toolkit/calibration.py]            ← Platt scaling, ECE, reliability diagrams
        │
        ├──► [toolkit/fairness.py]               ← 80% rule, equalized odds, predictive parity
        │
        └──► [toolkit/report.py]                 ← Jinja2 HTML validation report
                        │
                        ▼
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
reports/output/evidently_drift.html        ← Feature distribution drift (train vs monitor)
reports/output/evidently_quality.html      ← Data quality on monitor set
reports/output/evidently_performance.html  ← Classification performance comparison
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
├── data/
│   ├── raw/lending_club.csv          ← Source data (place here)
│   └── processed/                    ← Train/monitor parquets, model pkl, metrics JSON
├── toolkit/
│   ├── data_validation.py            ← Schema checks, leakage exclusions
│   ├── preprocessing.py              ← Feature engineering, encoding, scaling
│   ├── model.py                      ← LR training, evaluation, SHAP
│   ├── drift.py                      ← PSI, KS, prediction/target drift
│   ├── calibration.py                ← Platt scaling, ECE, reliability diagrams
│   ├── fairness.py                   ← Disparate impact, equalized odds
│   ├── threshold_governance.py       ← Threshold sweep, 3 candidate thresholds
│   └── report.py                     ← Jinja2 report generator
├── monitoring/
│   └── evidently_dashboard.py        ← Evidently drift + quality + performance
├── notebooks/
│   ├── 01_eda.ipynb                  ← EDA, class balance, temporal patterns
│   ├── 02_model_development.ipynb    ← Training, SHAP, threshold selection
│   ├── 03_validation.ipynb           ← Drift, calibration, fairness
│   └── 04_governance_report.ipynb    ← Evidently + HTML report rendering
├── reports/
│   ├── templates/validation_report.html
│   └── output/                       ← Generated reports
├── model_card.md
├── requirements.txt
└── README.md
```

---

## Resume-Ready Bullets

- **Built end-to-end model risk toolkit** for a credit default classifier on 1M+ Lending Club
  loans, covering data validation, drift detection, calibration, and fairness analysis
  aligned with SR 11-7 model risk management guidance

- **Implemented PSI and KS-based feature drift monitoring** across train/monitor temporal
  splits, flagging [N] features with significant distributional shift post-2015 and producing
  auto-generated Evidently HTML dashboards for risk committee review

- **Conducted fair lending analysis** using geographic and credit-age proxies, identifying
  [X]pp approval rate disparity across Census regions under the 80% ECOA disparate impact
  rule, and documented the fairness impossibility result and its regulatory implications

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
