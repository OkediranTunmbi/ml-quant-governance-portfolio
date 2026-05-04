"""
evidently_dashboard.py — Evidently AI drift and quality dashboards.

WHAT EVIDENTLY DOES UNDER THE HOOD:
Evidently generates statistical drift tests per column, choosing the test
automatically based on column type:
  - Numeric columns:    Wasserstein distance (by default) or KS test
  - Categorical columns: Chi-squared test for distribution shift
  - Target / prediction: Jensen-Shannon divergence or PSI

Evidently's reports are HTML dashboards that make these statistical results
interpretable for non-technical stakeholders: risk committees, model
sponsors, and regulators.  While the toolkit's drift.py computes PSI and KS
manually (for full auditability and explainability), Evidently provides a
visual layer on top that communicates findings at a glance.

WHY VISUAL DASHBOARDS MATTER FOR GOVERNANCE:
Model risk committees and senior stakeholders rarely review raw tables.
A self-contained HTML dashboard that shows distribution comparisons, drift
alerts, and data quality metrics allows non-quantitative reviewers to engage
with model performance evidence — a requirement under SR 11-7's emphasis on
communication and escalation of model risk findings.

Three reports are generated:
  1. DataDriftReport    — feature-level distribution comparison (train vs monitor)
  2. DataQualityReport  — missing values, unique counts, outliers on monitor set
  3. ClassificationReport — performance comparison on train vs monitor
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Evidently v0.4.x API
try:
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        ClassificationPreset,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("[evidently] WARNING: evidently not installed. Run: pip install evidently")


OUTPUT_DIR = Path("reports/output")


def _check_evidently():
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "evidently is required. Install with: pip install evidently"
        )


def run_drift_report(
    train_df: pd.DataFrame,
    monitor_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "default",
    save_path: str = "reports/output/evidently_drift.html",
) -> None:
    """
    DataDriftReport: compare feature distributions between train and monitor.
    Each feature gets its own drift test and a visual distribution overlay.
    """
    _check_evidently()

    cols = [c for c in feature_cols if c in train_df.columns and c in monitor_df.columns]
    ref  = train_df[cols + [target_col]].copy() if target_col in train_df.columns else train_df[cols].copy()
    cur  = monitor_df[cols + [target_col]].copy() if target_col in monitor_df.columns else monitor_df[cols].copy()

    # Evidently expects a 'target' column named consistently
    if target_col in ref.columns:
        ref = ref.rename(columns={target_col: "target"})
        cur = cur.rename(columns={target_col: "target"})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(save_path)
    print(f"[evidently] DataDriftReport saved to {save_path}")


def run_quality_report(
    monitor_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "default",
    save_path: str = "reports/output/evidently_quality.html",
) -> None:
    """
    DataQualityReport on the monitor (current production) set.
    Shows missing values, distributions, and data quality issues —
    the kind of table a data steward reviews before model retraining.
    """
    _check_evidently()

    cols = [c for c in feature_cols if c in monitor_df.columns]
    cur  = monitor_df[cols + [target_col]].copy() if target_col in monitor_df.columns else monitor_df[cols].copy()
    if target_col in cur.columns:
        cur = cur.rename(columns={target_col: "target"})

    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=None, current_data=cur)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(save_path)
    print(f"[evidently] DataQualityReport saved to {save_path}")


def run_performance_report(
    train_df: pd.DataFrame,
    monitor_df: pd.DataFrame,
    train_scores: np.ndarray,
    monitor_scores: np.ndarray,
    target_col: str = "default",
    save_path: str = "reports/output/evidently_performance.html",
) -> None:
    """
    ClassificationReport: compare model performance on train vs monitor.
    Evidently computes accuracy, precision, recall, F1, and ROC on both
    sets and flags performance gaps.
    """
    _check_evidently()

    ref = pd.DataFrame({
        "target":     train_df[target_col].values,
        "prediction": (train_scores >= 0.5).astype(int),
        "prediction_score": train_scores,
    })
    cur = pd.DataFrame({
        "target":     monitor_df[target_col].values,
        "prediction": (monitor_scores >= 0.5).astype(int),
        "prediction_score": monitor_scores,
    })

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=ref, current_data=cur)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(save_path)
    print(f"[evidently] ClassificationReport saved to {save_path}")


def run_all(
    train_df: pd.DataFrame,
    monitor_df: pd.DataFrame,
    train_scores: np.ndarray,
    monitor_scores: np.ndarray,
    feature_cols: list[str],
    target_col: str = "default",
    output_dir: str = "reports/output",
) -> None:
    """Run all three Evidently reports in sequence."""
    run_drift_report(
        train_df, monitor_df, feature_cols, target_col,
        save_path=f"{output_dir}/evidently_drift.html",
    )
    run_quality_report(
        monitor_df, feature_cols, target_col,
        save_path=f"{output_dir}/evidently_quality.html",
    )
    run_performance_report(
        train_df, monitor_df, train_scores, monitor_scores, target_col,
        save_path=f"{output_dir}/evidently_performance.html",
    )
    print(f"[evidently] All reports saved to {output_dir}/")
