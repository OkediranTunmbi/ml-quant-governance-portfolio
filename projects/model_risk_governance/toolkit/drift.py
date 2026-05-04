"""
drift.py — Manual statistical drift detection: PSI, KS test, prediction drift,
           and target drift.

WHY DRIFT MONITORING MATTERS (SR 11-7 context):
Under SR 11-7, banks must monitor deployed models on an ongoing basis.
"Model drift" — the phenomenon where the real-world distribution of inputs
or outputs diverges from the training-time distribution — is one of the most
common reasons models fail silently in production.

PSI (Population Stability Index) is the industry-standard drift metric in
retail credit.  It measures how much a feature's distribution has shifted
between reference (train) and current (monitor) periods.

Interpretation:
  PSI < 0.10  → No significant change. Model likely still valid.
  0.10–0.25   → Moderate shift. Investigate; consider recalibration.
  PSI > 0.25  → Major shift. Model likely invalid. Trigger review/retraining.

This threshold is embedded in bank model risk policies and SR 11-7 guidance
on model performance monitoring.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pathlib import Path


# ---------------------------------------------------------------------------
# PSI CALCULATION
# ---------------------------------------------------------------------------

def compute_psi_single(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index for a single numeric feature.

    PSI = sum( (P_current - P_ref) * ln(P_current / P_ref) )

    Bins are created from the reference distribution; the current distribution
    is mapped into the same bins.  This ensures a fair comparison.
    """
    # Define bin edges from reference quantiles — robust to outliers
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.nanpercentile(reference, quantiles)
    # Ensure unique edges (degenerate distributions can have repeated quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0  # can't compute PSI for constant features

    # Count observations in each bin, normalise to proportions
    ref_counts, _  = np.histogram(reference, bins=bin_edges)
    cur_counts, _  = np.histogram(current,   bins=bin_edges)

    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    # Replace zeros with a small epsilon to avoid log(0)
    eps = 1e-6
    ref_props = np.where(ref_props == 0, eps, ref_props)
    cur_props = np.where(cur_props == 0, eps, cur_props)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def compute_psi_all_features(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for every numeric feature.  Returns a DataFrame sorted by
    PSI descending, with a severity flag column.
    """
    results = []
    for feat in numeric_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        psi = compute_psi_single(ref_vals, cur_vals, n_bins=n_bins)
        if psi < 0.10:
            severity = "Stable"
        elif psi < 0.25:
            severity = "Moderate"
        else:
            severity = "Significant"
        results.append({"feature": feat, "psi": round(psi, 4), "severity": severity})

    psi_df = pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)
    return psi_df


# ---------------------------------------------------------------------------
# KS TEST — two-sample
# ---------------------------------------------------------------------------

def compute_ks_all_features(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Two-sample KS test for each numeric feature.
    p-value < alpha indicates the two distributions are significantly different.
    """
    results = []
    for feat in numeric_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue
        stat, pval = ks_2samp(ref_vals, cur_vals)
        results.append({
            "feature":  feat,
            "ks_stat":  round(float(stat), 4),
            "p_value":  round(float(pval), 6),
            "drifted":  pval < alpha,
        })
    ks_df = (
        pd.DataFrame(results)
        .sort_values("ks_stat", ascending=False)
        .reset_index(drop=True)
    )
    n_drifted = ks_df["drifted"].sum()
    print(f"[drift] KS test: {n_drifted} / {len(ks_df)} features drifted (p < {alpha})")
    return ks_df


# ---------------------------------------------------------------------------
# PREDICTION DRIFT — score-level PSI
# ---------------------------------------------------------------------------

def compute_prediction_drift(
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    n_bins: int = 10,
    save_dir: str = "data/processed",
) -> float:
    """
    PSI on model output probabilities.  This is the model-level drift indicator.
    Even if feature distributions are stable, calibration or concept drift
    can shift the score distribution.
    """
    psi = compute_psi_single(ref_scores, cur_scores, n_bins=n_bins)
    severity = (
        "Stable" if psi < 0.10 else
        "Moderate" if psi < 0.25 else
        "Significant"
    )
    print(f"[drift] Prediction score PSI = {psi:.4f} ({severity})")

    # Plot score distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ref_scores, bins=40, alpha=0.6, label="Train (reference)", density=True)
    ax.hist(cur_scores, bins=40, alpha=0.6, label="Monitor (current)", density=True)
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution Drift (PSI = {psi:.4f})")
    ax.legend()
    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_dir) / "prediction_drift.png", dpi=120, bbox_inches="tight")
    plt.close()

    return psi


# ---------------------------------------------------------------------------
# TARGET DRIFT — default rate comparison
# ---------------------------------------------------------------------------

def compute_target_drift(
    ref_labels: np.ndarray,
    cur_labels: np.ndarray,
    threshold_pp: float = 5.0,
) -> dict:
    """
    Compare default rates between reference and current periods.
    A delta > threshold_pp percentage points triggers a flag.
    Target drift indicates concept drift: the relationship between features
    and the outcome may have changed, requiring model retraining.
    """
    ref_rate = float(np.mean(ref_labels)) * 100
    cur_rate = float(np.mean(cur_labels)) * 100
    delta_pp = cur_rate - ref_rate
    flagged  = abs(delta_pp) > threshold_pp

    result = {
        "ref_default_rate_pct":  round(ref_rate, 2),
        "cur_default_rate_pct":  round(cur_rate, 2),
        "delta_pp":              round(delta_pp, 2),
        "flagged":               flagged,
    }
    flag_str = "FLAGGED" if flagged else "OK"
    print(f"[drift] Target drift: ref={ref_rate:.1f}%  cur={cur_rate:.1f}%  "
          f"Δ={delta_pp:+.1f}pp  [{flag_str}]")
    return result


# ---------------------------------------------------------------------------
# FULL DRIFT REPORT
# ---------------------------------------------------------------------------

def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    ref_labels: np.ndarray,
    cur_labels: np.ndarray,
    numeric_features: list[str],
    save_dir: str = "data/processed",
) -> dict:
    """
    End-to-end drift report.  Saves PSI and KS tables as CSVs.
    Returns a dict of all results for the governance report.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    psi_df = compute_psi_all_features(reference_df, current_df, numeric_features)
    ks_df  = compute_ks_all_features(reference_df, current_df, numeric_features)
    pred_psi = compute_prediction_drift(ref_scores, cur_scores, save_dir=save_dir)
    target   = compute_target_drift(ref_labels, cur_labels)

    psi_df.to_csv(Path(save_dir) / "psi_table.csv", index=False)
    ks_df.to_csv(Path(save_dir) / "ks_table.csv",  index=False)
    print(f"[drift] Saved PSI and KS tables to {save_dir}")

    n_sig_psi  = int((psi_df["severity"] == "Significant").sum())
    n_mod_psi  = int((psi_df["severity"] == "Moderate").sum())
    n_drifted_ks = int(ks_df["drifted"].sum())

    print(f"[drift] Summary: PSI significant={n_sig_psi}, moderate={n_mod_psi}; "
          f"KS drifted={n_drifted_ks}")

    return {
        "psi_table":         psi_df,
        "ks_table":          ks_df,
        "prediction_psi":    pred_psi,
        "target_drift":      target,
        "n_sig_psi":         n_sig_psi,
        "n_moderate_psi":    n_mod_psi,
        "n_drifted_ks":      n_drifted_ks,
    }
