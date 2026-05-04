# Model Card — Logistic Regression Credit Default Model

| Field | Value |
|---|---|
| **Model Name** | Logistic Regression Credit Default Model |
| **Version** | 1.0 |
| **Date** | 2026-05-02 |
| **Owner** | Model Development Team (1st Line) |
| **Validator** | Model Risk Management (2nd Line) |

---

## Intended Use

**Primary use:** Predict the probability of default for personal unsecured loans at the point of origination. Score is used to:
- Approve or decline loan applications
- Set interest rates commensurate with risk
- Calculate regulatory capital under Basel IRB approach (after regulatory approval)

**In-scope:**
- Personal (consumer) unsecured loans
- U.S.-domiciled borrowers
- Loan amounts consistent with Lending Club origination range ($1,000–$40,000)
- Applications processed at origination only — score uses only information available at the time of application

## Out-of-Scope Use

- **Post-origination monitoring**: this model scores at origination only; it is not valid for predicting further deterioration of existing loans
- **Non-US loans**: trained exclusively on US borrowers; geographic, regulatory, and credit-bureau differences make cross-border application invalid
- **Business / commercial loans**: trained on personal loans; business credit dynamics differ materially
- **Fraud detection**: this model predicts credit default, not fraudulent application
- **Pricing of non-standard products**: HELOC, auto, mortgage — different collateral and repayment structures

---

## Training Data

| Attribute | Detail |
|---|---|
| **Source** | Lending Club public loan data |
| **Period** | 2007–2014 (issue_d < 2015-01-01) |
| **N loans (training split)** | ~383,400 (85% of pre-2015 loans after filtering) |
| **N loans (evaluation set)** | ~47,400 (stratified holdout from pre-2015) |
| **N loans (monitor set)** | ~894,300 (issue_d >= 2015-01-01) |
| **Total dataset** | ~1,345,000 loans across all splits |
| **Target definition** | `default = 1` if `loan_status == 'Charged Off'`; `default = 0` if `loan_status == 'Fully Paid'`. Ambiguous statuses (Current, Late, In Grace Period) excluded and documented in `toolkit/data_validation.py`. |
| **Train default rate** | 16.96% |
| **Monitor default rate** | 21.48% (+4.52pp shift — below 5pp flag threshold) |
| **Leakage controls** | 32 post-origination columns explicitly excluded with documented justification in `toolkit/data_validation.py::LEAKAGE_EXCLUSIONS` |
| **Train/test split** | 85% train / 15% stratified holdout (of holdout: 30% calibration, 70% evaluation) |

---

## Performance

| Split | AUC-ROC | Gini | KS | AUC-PR |
|---|---|---|---|---|
| **Train** | 0.7001 | 0.4001 | 0.2921 | 0.3164 |
| **Test (eval)** | 0.6962 | 0.3925 | 0.2844 | 0.3115 |
| **Monitor (post-2015)** | 0.7142 | 0.4285 | 0.3110 | 0.3959 |

**Calibration (Platt scaling on calibration holdout):**

| Stage | ECE |
|---|---|
| Raw Logistic Regression | 0.2907 |
| After Platt scaling | 0.0093 |
| Monitor (post-calibration) | 0.0093 — Status: OK |

---

## Threshold Governance

Three candidate thresholds are documented. Final selection requires risk committee approval.

| Policy | Threshold | Approval Rate | Default Rate (Approved) | F1 | F2 |
|---|---|---|---|---|---|
| **Conservative** (max F2) | 0.37 | 32.9% | 7.1% | — | 0.5415 |
| **Balanced** (max F1) | 0.51 | 63.3% | 10.7% | 0.3782 | — |
| **Liberal** (max approval, default rate <= 20%) | 0.90 | 99.9% | 16.9% | — | — |

---

## Drift Monitoring (Train vs. Monitor)

| Indicator | Result | Status |
|---|---|---|
| PSI significant (>0.25) features | 0 | Stable |
| PSI moderate (0.10-0.25) features | 1 (`mths_since_last_record` = 0.2163) | Monitor |
| KS-drifted features (p < 0.05) | 19 of 19 tested | See note |
| Prediction score PSI | 0.00092 | Stable |
| Target drift (default rate delta) | +4.52pp (16.96% to 21.48%) | OK (below 5pp threshold) |

> **Note on KS results:** With 400K+ training samples, the KS test flags nearly every feature as statistically significant even for economically trivial distributional differences. The PSI analysis (which uses business-calibrated thresholds) is the primary operational drift indicator. Zero significant PSI features and score PSI near zero confirm model stability over the monitoring period.

---

## Fairness Findings

Proxy variables used (Lending Club data contains no explicit demographic fields).

### Census Region (race/ethnicity proxy)

| Region | Approval Rate | Disparate Impact Ratio | Fails 80% Rule |
|---|---|---|---|
| West | 64.3% | 1.000 (reference) | No |
| Northeast | 63.8% | 0.992 | No |
| South | 62.6% | 0.973 | No |
| Midwest | 62.1% | 0.966 | No |

Max disparity: 2.2pp (West vs. Midwest). All groups pass the 80% rule. No fair lending flag.

### Credit Age Group (age proxy)

| Group | Approval Rate | Disparate Impact Ratio | Fails 80% Rule |
|---|---|---|---|
| Senior (oldest credit) | 68.9% | 1.000 (reference) | No |
| Mature | 64.2% | 0.931 | No |
| Established | 61.9% | 0.898 | No |
| Young (newest credit) | 58.1% | 0.843 | No |

Max disparity: 10.8pp (Senior vs. Young). Young group disparate impact ratio of 0.843 is above the 80% threshold but close — **flagged as a watchpoint for ongoing monitoring.**

### Equalized Odds Summary (Balanced Threshold = 0.51)

| Group | TPR (Recall) | FPR |
|---|---|---|
| Midwest (region) | 0.612 | 0.332 |
| South (region) | 0.607 | 0.325 |
| Young (credit age) | 0.631 | 0.371 |
| Senior (credit age) | 0.559 | 0.267 |

Younger credit-history borrowers show the highest recall and highest FPR simultaneously — the model catches more of their defaults but also incorrectly flags more of their non-defaults. This is consistent with the fairness impossibility result: groups with higher base default rates cannot simultaneously achieve equal TPR, FPR, and approval rates.

**Disclaimer:** Lending Club data contains no explicit demographic fields. Proxy-based fairness findings are indicative screening measures. Definitive fair lending analysis requires actual demographic data.

---

## Known Limitations

1. **Proxy-based fairness analysis only** — no demographic ground truth available; findings cannot confirm or rule out individual discrimination
2. **Borderline Gini on test set** — Gini of 0.3925 is just below the common 0.40 industry minimum. The monitor Gini (0.4285) exceeds it. A tree-based model (LightGBM) would likely add 10-15 Gini points at the cost of interpretability
3. **Severe raw miscalibration** — ECE of 0.2907 on the raw model means uncalibrated scores cannot be used as PD estimates; Platt scaling is mandatory before any Basel IRB capital calculation
4. **KS test over-sensitivity** — with large sample sizes, KS flags all 19 numeric features as statistically drifted; PSI (not KS) is the operational trigger for model review
5. **Young credit group proximity to 80% threshold** — disparate impact ratio of 0.843 is above 80% but warrants quarterly monitoring as the portfolio evolves
6. **Concept drift post-2015** — the model was trained on 2007-2014 data; the monitor period reflects different economic conditions, higher default rates, and Lending Club's rapid loan-book growth

---

## Risk Controls

| Control | Implementation | Result |
|---|---|---|
| **Leakage exclusions** | 32 columns excluded with documented business justification | Applied |
| **Train-only fitting** | All transformers fit on training data only; applied to test/monitor | Applied |
| **Calibration** | Platt scaling on calibration holdout; ECE 0.2907 to 0.0093 | Pass |
| **Drift monitoring** | PSI/KS monitoring; 0 significant PSI features, score PSI near 0 | Pass |
| **Fairness screening** | Proxy analysis; 0 groups fail 80% rule; Young age group at watchpoint | Pass with monitoring |
| **Threshold governance** | 3 candidate thresholds documented; final threshold requires committee approval | Pending |

---

## Validation Status

| Item | Status |
|---|---|
| Independent model validation | Completed — see `reports/output/validation_report_*.html` |
| Validator recommendation | Approve with Conditions |
| Approval conditions | (1) Deploy only calibrated model — raw scores not permitted; (2) Quarterly fairness re-testing for Young credit age group; (3) Confirm KS flags are statistical artefacts via PSI before escalating |
| Next review date | 2027-05-02 (12 months from approval) |
