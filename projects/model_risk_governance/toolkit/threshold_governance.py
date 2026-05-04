"""
threshold_governance.py — Threshold sweep and governance framework.

WHY THRESHOLD SELECTION IS A BUSINESS DECISION (SR 11-7 context):
A classification model outputs a continuous probability score.  Converting
that score to a binary decision (approve / deny) requires choosing a
threshold.  This threshold is NOT a technical optimisation problem alone —
it embeds fundamental business trade-offs:

  • Conservative threshold (low probability → deny): fewer defaults, but
    also fewer approved loans and lower revenue.  Appropriate for risk-averse
    lenders, stressed capital environments, or when regulator scrutiny is high.

  • Liberal threshold (high probability → deny only when very likely to
    default): maximises loan volume, but accepts more defaults.  Appropriate
    for growth-phase lenders with high credit appetite.

  • Balanced threshold: maximises F1 — the harmonic mean of precision and
    recall.  A common starting point when no explicit risk appetite is given.

SR 11-7 Section V ("Model Validation") requires that "threshold selection,
including the rationale," be documented in the validation report.  A model
risk officer reviewing this toolkit should be able to trace every threshold
choice to an explicit business or risk objective.

F2 score (β=2): weights recall twice as much as precision.  In credit risk,
a missed default (false negative) is typically more costly than a wrongly
rejected application (false positive), hence the recall-weighted F2 is a
natural performance measure for a conservative credit policy.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    max_default_rate_liberal: float = 0.20,
) -> pd.DataFrame:
    """
    Sweep classification thresholds from 0.10 to 0.90 and compute, per threshold:
      - precision   : of those denied (predicted default), fraction truly defaulted
      - recall      : of all true defaults, fraction correctly flagged
      - f1           : harmonic mean of precision and recall
      - f2           : recall-weighted F-beta (β=2)
      - approval_rate: fraction of applicants approved (predicted non-default)
      - default_rate_approved: fraction of approved applicants who default
                               (true measure of credit quality among approvals)
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    records = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)  # 1 = predicted default = denied
        n_approved = int((y_pred == 0).sum())
        approved_mask = y_pred == 0

        precision  = precision_score(y_true, y_pred, zero_division=0)
        recall     = recall_score(y_true, y_pred, zero_division=0)
        f1         = f1_score(y_true, y_pred, zero_division=0)
        f2         = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        appr_rate  = float(approved_mask.mean())

        # Default rate among approved: what fraction of approved loans actually default?
        if n_approved > 0:
            def_rate_appr = float(y_true[approved_mask].mean())
        else:
            def_rate_appr = 0.0

        records.append({
            "threshold":            round(float(t), 2),
            "precision":            round(precision, 4),
            "recall":               round(recall, 4),
            "f1":                   round(f1, 4),
            "f2":                   round(f2, 4),
            "approval_rate":        round(appr_rate, 4),
            "default_rate_approved":round(def_rate_appr, 4),
        })

    return pd.DataFrame(records)


def identify_candidate_thresholds(
    sweep_df: pd.DataFrame,
    max_default_rate_liberal: float = 0.20,
) -> dict:
    """
    Identify three governance-relevant threshold candidates:

    Conservative: maximise F2 (minimises missed defaults — recall-weighted).
                  Appropriate when cost of default >> cost of rejection.

    Balanced:     maximise F1 (balanced precision and recall).
                  Neutral starting point; typical model validation default.

    Liberal:      maximise approval rate subject to default_rate_approved ≤ 20%.
                  Maximum growth while keeping portfolio quality above a floor.
    """
    conservative_row = sweep_df.loc[sweep_df["f2"].idxmax()]
    balanced_row     = sweep_df.loc[sweep_df["f1"].idxmax()]
    liberal_subset   = sweep_df[sweep_df["default_rate_approved"] <= max_default_rate_liberal]
    if not liberal_subset.empty:
        liberal_row = liberal_subset.loc[liberal_subset["approval_rate"].idxmax()]
    else:
        # Fallback: lowest threshold
        liberal_row = sweep_df.iloc[0]

    candidates = {
        "conservative": {
            "threshold":     float(conservative_row["threshold"]),
            "criterion":     "Max F2 (recall-weighted)",
            "f2":            float(conservative_row["f2"]),
            "precision":     float(conservative_row["precision"]),
            "recall":        float(conservative_row["recall"]),
            "approval_rate": float(conservative_row["approval_rate"]),
            "default_rate_approved": float(conservative_row["default_rate_approved"]),
        },
        "balanced": {
            "threshold":     float(balanced_row["threshold"]),
            "criterion":     "Max F1 (balanced)",
            "f1":            float(balanced_row["f1"]),
            "precision":     float(balanced_row["precision"]),
            "recall":        float(balanced_row["recall"]),
            "approval_rate": float(balanced_row["approval_rate"]),
            "default_rate_approved": float(balanced_row["default_rate_approved"]),
        },
        "liberal": {
            "threshold":     float(liberal_row["threshold"]),
            "criterion":     f"Max approval rate | default_rate ≤ {max_default_rate_liberal:.0%}",
            "approval_rate": float(liberal_row["approval_rate"]),
            "default_rate_approved": float(liberal_row["default_rate_approved"]),
            "precision":     float(liberal_row["precision"]),
            "recall":        float(liberal_row["recall"]),
        },
    }

    print("[threshold] Candidate thresholds:")
    for name, info in candidates.items():
        print(f"  {name:12s}: threshold={info['threshold']:.2f}  "
              f"approval={info['approval_rate']:.1%}  "
              f"default_rate_approved={info['default_rate_approved']:.1%}")
    return candidates


def plot_threshold_sweep(
    sweep_df: pd.DataFrame,
    candidates: dict,
    save_dir: str = "data/processed",
) -> None:
    """
    Single figure with all metrics vs threshold.
    Vertical dashed lines mark the three candidate thresholds.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top panel: classification metrics
    axes[0].plot(sweep_df["threshold"], sweep_df["precision"],  label="Precision",  linewidth=1.8)
    axes[0].plot(sweep_df["threshold"], sweep_df["recall"],     label="Recall",     linewidth=1.8)
    axes[0].plot(sweep_df["threshold"], sweep_df["f1"],         label="F1",         linewidth=1.8)
    axes[0].plot(sweep_df["threshold"], sweep_df["f2"],         label="F2 (β=2)",   linewidth=1.8, linestyle="--")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Metrics vs. Decision Threshold")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # Bottom panel: business metrics
    axes[1].plot(sweep_df["threshold"], sweep_df["approval_rate"],
                 label="Approval Rate", linewidth=1.8, color="#2ecc71")
    axes[1].plot(sweep_df["threshold"], sweep_df["default_rate_approved"],
                 label="Default Rate (Approved)", linewidth=1.8, color="#e74c3c")
    axes[1].set_xlabel("Decision Threshold")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Business Metrics vs. Decision Threshold")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    # Mark candidate thresholds
    colours = {"conservative": "purple", "balanced": "navy", "liberal": "green"}
    for name, info in candidates.items():
        t = info["threshold"]
        for ax in axes:
            ax.axvline(t, color=colours[name], linestyle=":", linewidth=1.2,
                       label=f"{name} ({t:.2f})")
        axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "threshold_sweep.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[threshold] Saved sweep plot to {out}")


def run_threshold_governance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_dir: str = "data/processed",
) -> dict:
    """Full threshold governance pipeline. Saves sweep CSV and plot."""
    sweep_df   = threshold_sweep(y_true, y_score)
    candidates = identify_candidate_thresholds(sweep_df)
    plot_threshold_sweep(sweep_df, candidates, save_dir)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(Path(save_dir) / "threshold_sweep.csv", index=False)
    print(f"[threshold] Saved threshold sweep table to {save_dir}/threshold_sweep.csv")

    return {"sweep": sweep_df, "candidates": candidates}
