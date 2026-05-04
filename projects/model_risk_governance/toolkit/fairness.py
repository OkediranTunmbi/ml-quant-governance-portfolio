"""
fairness.py — Fair lending analysis for the Lending Club credit model.

IMPORTANT — PROXY VARIABLE DISCLAIMER:
Lending Club data contains no explicit demographic fields (race, ethnicity,
age, sex).  We therefore use proxy variables, which are imperfect but
standard practice in fair lending analysis when direct attributes are absent:

  1. Geographic proxy (region): addr_state → U.S. Census region.
     Census regions correlate with racial/ethnic composition (e.g., the South
     has a higher proportion of Black residents).  This is an IMPERFECT proxy
     because individual borrowers within a region vary enormously.  A positive
     finding here triggers investigation, not automatic enforcement.

  2. Credit age proxy: credit_age_months quartiled → Young/Established/
     Mature/Senior credit histories.  Younger credit files correlate with
     younger borrowers (age proxy), since credit history accumulates over time.
     Again, an imperfect individual-level proxy.

WHY THIS MATTERS (legal context):
Under the Equal Credit Opportunity Act (ECOA) and the Fair Housing Act (FHA),
lenders cannot discriminate on the basis of race, national origin, sex, age,
or other protected characteristics.  The "80% rule" (disparate impact test)
requires that the approval rate for any protected group be at least 80% of
the highest group's approval rate.

The FAIRNESS IMPOSSIBILITY result (Chouldechova 2017, Kleinberg et al. 2016):
  You cannot simultaneously satisfy all three of:
    1. Demographic parity (equal approval rates)
    2. Equalized odds  (equal TPR and FPR across groups)
    3. Predictive parity (equal precision across groups)
  ...unless the base rates (default rates) are equal across groups.
  Since default rates differ by region and credit age in practice, a model
  risk officer must choose which criterion to prioritise — and document why.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CENSUS REGION MAPPING
# ---------------------------------------------------------------------------
STATE_TO_REGION = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",
    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}


def assign_region(df: pd.DataFrame) -> pd.Series:
    """Map addr_state to U.S. Census region."""
    return df["addr_state"].map(STATE_TO_REGION).fillna("Unknown")


def assign_credit_age_group(df: pd.DataFrame) -> pd.Series:
    """
    Quartile credit_age_months into four named groups.
    Labels chosen to be interpretable in a governance report.
    """
    labels = ["Young", "Established", "Mature", "Senior"]
    return pd.qcut(
        df["credit_age_months"],
        q=4,
        labels=labels,
        duplicates="drop",
    ).astype(str)


# ---------------------------------------------------------------------------
# FAIRNESS METRICS
# ---------------------------------------------------------------------------

def approval_rate_by_group(
    y_true: pd.Series,
    y_pred_bin: pd.Series,
    group: pd.Series,
) -> pd.DataFrame:
    """
    Demographic parity: approval rate per group.
    In credit, 'approved' = predicted non-default (y_pred_bin == 0).
    The 80% rule: approval_rate_group / max_approval_rate >= 0.80.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_bin, "group": group})
    # approved = predicted to repay
    df["approved"] = (df["y_pred"] == 0).astype(int)
    rates = df.groupby("group")["approved"].mean().rename("approval_rate").reset_index()
    overall_rate = df["approved"].mean()
    max_rate     = rates["approval_rate"].max()
    rates["disparate_impact_ratio"] = rates["approval_rate"] / max_rate
    rates["flag_80_rule"] = rates["disparate_impact_ratio"] < 0.80
    rates["overall_rate"] = overall_rate
    rates["delta_vs_overall_pp"] = (rates["approval_rate"] - overall_rate) * 100
    return rates.sort_values("approval_rate", ascending=False).reset_index(drop=True)


def equalized_odds_by_group(
    y_true: pd.Series,
    y_pred_bin: pd.Series,
    group: pd.Series,
) -> pd.DataFrame:
    """
    Equalized odds: TPR and FPR per group.
    TPR (recall) = fraction of true defaults correctly flagged.
    FPR          = fraction of true non-defaults incorrectly flagged as default.
    Equalized odds requires TPR and FPR to be equal across groups.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred_bin, "g": group})
    results = []
    for g, gdf in df.groupby("g"):
        tp  = ((gdf["p"] == 1) & (gdf["y"] == 1)).sum()
        fn  = ((gdf["p"] == 0) & (gdf["y"] == 1)).sum()
        fp  = ((gdf["p"] == 1) & (gdf["y"] == 0)).sum()
        tn  = ((gdf["p"] == 0) & (gdf["y"] == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results.append({"group": g, "tpr": round(tpr, 4), "fpr": round(fpr, 4), "n": len(gdf)})
    return pd.DataFrame(results).sort_values("tpr", ascending=False).reset_index(drop=True)


def predictive_parity_by_group(
    y_true: pd.Series,
    y_pred_bin: pd.Series,
    group: pd.Series,
) -> pd.DataFrame:
    """
    Predictive parity: precision per group.
    Among borrowers predicted to DEFAULT, what fraction actually defaulted?
    Equal precision across groups means the model is equally 'correct' about
    predicted defaults regardless of group membership.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred_bin, "g": group})
    results = []
    for g, gdf in df.groupby("g"):
        predicted_default = gdf[gdf["p"] == 1]
        precision = predicted_default["y"].mean() if len(predicted_default) > 0 else 0.0
        results.append({"group": g, "precision_defaults": round(float(precision), 4),
                         "n_predicted_default": len(predicted_default)})
    return pd.DataFrame(results).sort_values("precision_defaults", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def plot_approval_rates(
    approval_df: pd.DataFrame,
    group_name: str,
    save_dir: str = "data/processed",
) -> None:
    """Bar chart of approval rates by group with 80% threshold line."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        approval_df["group"].astype(str),
        approval_df["approval_rate"],
        color=["#e74c3c" if f else "#2ecc71" for f in approval_df["flag_80_rule"]],
        edgecolor="black", linewidth=0.5,
    )
    max_rate = approval_df["approval_rate"].max()
    ax.axhline(max_rate * 0.80, color="red", linestyle="--", linewidth=1.5,
               label="80% rule threshold")
    ax.axhline(approval_df["overall_rate"].iloc[0], color="navy", linestyle=":",
               linewidth=1.5, label="Overall approval rate")
    ax.set_xlabel(f"Group ({group_name})")
    ax.set_ylabel("Approval Rate")
    ax.set_title(f"Approval Rates by {group_name} (Disparate Impact Analysis)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, approval_df["approval_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = Path(save_dir) / f"approval_rates_{group_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fairness] Saved approval rate chart to {out}")


def plot_equalized_odds(
    eo_df: pd.DataFrame,
    group_name: str,
    save_dir: str = "data/processed",
) -> None:
    """Grouped bar chart of TPR and FPR per group."""
    groups = eo_df["group"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, eo_df["tpr"], width, label="TPR (Recall)", color="#3498db")
    ax.bar(x + width / 2, eo_df["fpr"], width, label="FPR",          color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_xlabel(f"Group ({group_name})")
    ax.set_ylabel("Rate")
    ax.set_title(f"Equalized Odds by {group_name}")
    ax.legend()
    plt.tight_layout()
    out = Path(save_dir) / f"equalized_odds_{group_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[fairness] Saved equalized odds chart to {out}")


# ---------------------------------------------------------------------------
# FULL FAIRNESS REPORT
# ---------------------------------------------------------------------------

def run_fairness_report(
    df: pd.DataFrame,
    y_pred_bin: np.ndarray,
    threshold: float,
    save_dir: str = "data/processed",
) -> dict:
    """
    End-to-end fairness analysis.  Saves tables as CSVs.
    Returns results dict for the governance report.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["region"]    = assign_region(df)
    df["age_group"] = assign_credit_age_group(df)

    y_true     = pd.Series(df["default"].values)
    y_pred_ser = pd.Series(y_pred_bin)

    results = {}

    for group_col, group_name in [("region", "Census Region"), ("age_group", "Credit Age Group")]:
        group_series = pd.Series(df[group_col].values)

        apr = approval_rate_by_group(y_true, y_pred_ser, group_series)
        eo  = equalized_odds_by_group(y_true, y_pred_ser, group_series)
        pp  = predictive_parity_by_group(y_true, y_pred_ser, group_series)

        key = group_col
        apr.to_csv(Path(save_dir) / f"fairness_approval_{key}.csv", index=False)
        eo.to_csv(Path(save_dir)  / f"fairness_eo_{key}.csv",       index=False)
        pp.to_csv(Path(save_dir)  / f"fairness_pp_{key}.csv",       index=False)

        plot_approval_rates(apr, group_name, save_dir)
        plot_equalized_odds(eo,  group_name, save_dir)

        n_flagged = int(apr["flag_80_rule"].sum())
        print(f"[fairness] {group_name}: {n_flagged} group(s) fail 80% rule.")

        results[key] = {
            "approval_rates":    apr,
            "equalized_odds":    eo,
            "predictive_parity": pp,
            "n_flagged_80_rule": n_flagged,
        }

    return results
