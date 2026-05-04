"""
data_validation.py — Schema checks, missing-rate audit, and leakage exclusions.

SR 11-7 requires that all data used in a model be validated for completeness,
accuracy, and representativeness before training. This module performs those
checks and enforces the leakage exclusion policy.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# LEAKAGE EXCLUSION REGISTRY
# Each entry documents the column, the category of leakage, and the business
# justification for exclusion. This registry is the auditable record required
# under SR 11-7 Section IV: "Data Management and Aggregation."
# ---------------------------------------------------------------------------
LEAKAGE_EXCLUSIONS = {
    # POST-ORIGINATION PAYMENT OUTCOMES
    # These columns are populated only after the loan has been active for some
    # time. At origination (the prediction point), none of these values exist.
    # Including them would allow the model to see the future — a classic
    # "look-ahead" leakage that inflates AUC dramatically.
    "total_pymnt":              "Post-origination: total amount paid — unknown at origination",
    "total_pymnt_inv":          "Post-origination: investor-portion of payment — unknown at origination",
    "total_rec_prncp":          "Post-origination: principal recovered — unknown at origination",
    "total_rec_int":            "Post-origination: interest recovered — unknown at origination",
    "total_rec_late_fee":       "Post-origination: late fees recovered — unknown at origination",
    "recoveries":               "Post-origination: post-charge-off collections — unknown at origination",
    "collection_recovery_fee":  "Post-origination: collection fees — unknown at origination",
    "out_prncp":                "Post-origination: remaining principal — unknown at origination",
    "out_prncp_inv":            "Post-origination: investor remaining principal — unknown at origination",
    "last_pymnt_d":             "Post-origination: date of last payment — unknown at origination",
    "last_pymnt_amnt":          "Post-origination: amount of last payment — unknown at origination",
    "next_pymnt_d":             "Post-origination: scheduled next payment — unknown at origination",

    # ADMINISTRATIVE / ID COLUMNS
    # These carry no predictive signal — they are unique identifiers or
    # free-text fields that would cause memorisation rather than generalisation.
    "id":                       "Administrative: unique loan ID — no predictive signal",
    "member_id":                "Administrative: unique member ID — no predictive signal",
    "url":                      "Administrative: URL to loan listing — no predictive signal",
    "desc":                     "Administrative: free-text description — high-cardinality, no signal",
    "title":                    "Administrative: loan title free text — high-cardinality, no signal",

    # POST-ORIGINATION STATUS FLAGS
    # These flags indicate hardship or settlement programmes initiated after
    # origination — they are consequences of default risk, not causes.
    "hardship_flag":            "Post-origination: hardship programme flag — consequence of default",
    "hardship_type":            "Post-origination: hardship type — post-origination status",
    "hardship_reason":          "Post-origination: hardship reason — post-origination status",
    "hardship_status":          "Post-origination: hardship status — post-origination status",
    "debt_settlement_flag":     "Post-origination: settlement flag — consequence of default",
    "settlement_status":        "Post-origination: settlement status — post-origination status",
}


def apply_leakage_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all leakage columns that are present in the dataframe."""
    cols_to_drop = [c for c in LEAKAGE_EXCLUSIONS if c in df.columns]
    print(f"[data_validation] Dropping {len(cols_to_drop)} leakage columns:")
    for c in cols_to_drop:
        print(f"  - {c}: {LEAKAGE_EXCLUSIONS[c]}")
    return df.drop(columns=cols_to_drop)


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert loan_status → binary default target (1 = default, 0 = repaid).

    We keep only 'Charged Off' and 'Fully Paid' rows.  All other statuses
    (Current, Late (31-120 days), In Grace Period, etc.) are dropped because:
      - 'Current': the loan has not reached its final state — labelling it
        non-default would be premature and biased toward 0.
      - 'Late' / 'In Grace Period': ambiguous mid-outcome; including them as
        either class would mislabel a meaningful fraction of eventual defaults.
      - Including ambiguous statuses would create label noise that degrades
        model performance and misrepresents true default rates.
    Dropped rows are documented here for the validation record.
    """
    valid_statuses = {"Charged Off", "Fully Paid"}
    n_before = len(df)
    other_statuses = df.loc[~df["loan_status"].isin(valid_statuses), "loan_status"].value_counts()

    print("[data_validation] Dropping ambiguous loan statuses:")
    for status, count in other_statuses.items():
        print(f"  - '{status}': {count:,} rows dropped (ambiguous outcome)")

    df = df[df["loan_status"].isin(valid_statuses)].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    n_after = len(df)
    print(f"[data_validation] Retained {n_after:,} / {n_before:,} rows "
          f"({n_after/n_before:.1%}). Default rate: {df['default'].mean():.2%}")
    return df


def schema_check(df: pd.DataFrame) -> dict:
    """Return a dict of schema diagnostics: dtypes, nunique, missing rates."""
    report = {}
    for col in df.columns:
        report[col] = {
            "dtype":       str(df[col].dtype),
            "n_missing":   int(df[col].isna().sum()),
            "pct_missing": float(df[col].isna().mean()),
            "nunique":     int(df[col].nunique()),
        }
    return report


def missing_rate_report(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """
    Flag columns whose missing rate exceeds threshold.
    SR 11-7 expects documentation of data quality issues — high missing rates
    may indicate a data sourcing problem or a feature not available at origination.
    """
    rates = df.isna().mean().sort_values(ascending=False)
    report = pd.DataFrame({"pct_missing": rates})
    report["flag"] = report["pct_missing"] > threshold
    flagged = report[report["flag"]]
    if not flagged.empty:
        print(f"[data_validation] WARNING: {len(flagged)} columns exceed "
              f"{threshold:.0%} missing-rate threshold:")
        print(flagged.to_string())
    return report


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full validation pipeline:
      1. Binarize target and drop ambiguous statuses
      2. Apply leakage exclusions
      3. Run schema and missing-rate checks
    Returns the cleaned dataframe.
    """
    df = binarize_target(df)
    df = apply_leakage_exclusions(df)
    _ = missing_rate_report(df)
    print("[data_validation] Validation complete.")
    return df
