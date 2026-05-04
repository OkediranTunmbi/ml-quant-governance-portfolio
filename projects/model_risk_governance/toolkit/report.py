"""
report.py — Bank-style model validation report generator using Jinja2.

WHAT IS INDEPENDENT MODEL VALIDATION (IMV)?
Under SR 11-7, banks must have a model risk management framework with
"three lines of defense":
  1st line: Model developers (build and own the model)
  2nd line: Model Risk Management / Independent Validators (review, challenge,
            approve or reject models built by 1st line)
  3rd line: Internal Audit (periodic review of the MRM framework itself)

The Independent Model Validation (IMV) report is the 2nd-line artifact:
a formal document produced by a team that did NOT build the model, that
assesses its conceptual soundness, data quality, performance, stability,
and risk controls.  This script auto-generates that report from results
computed by the other toolkit modules.

The Validator Recommendation logic mirrors real bank policy:
  - Any High-risk finding → "Approve with Conditions" or "Reject"
  - Multiple Medium-risk findings → "Approve with Conditions"
  - All findings Low-risk → "Approve"
"""

import base64
import json
from datetime import date
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def _embed_image(path: str) -> str:
    """Base64-encode a PNG so it can be embedded inline in the HTML report."""
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""


def _auto_recommendation(findings: list[dict]) -> str:
    """
    Derive validator recommendation from findings.
    findings: list of dicts with keys 'title', 'risk_level', 'description'
    """
    high_count   = sum(1 for f in findings if f.get("risk_level") == "High")
    medium_count = sum(1 for f in findings if f.get("risk_level") == "Medium")

    if high_count >= 2:
        return "Reject"
    elif high_count == 1 or medium_count >= 3:
        return "Approve with Conditions"
    elif medium_count >= 1:
        return "Approve with Conditions"
    else:
        return "Approve"


def _performance_flag(train_auc: float, monitor_auc: float, threshold: float = 0.05) -> str:
    delta = train_auc - monitor_auc
    if delta > threshold:
        return f"DEGRADED (Δ={delta:.3f})"
    return "OK"


def build_findings(results: dict) -> list[dict]:
    """
    Auto-generate structured findings from results dict.
    Each finding maps to a risk level used in the recommendation logic.
    """
    findings = []

    # --- PSI findings ---
    psi_table = results.get("psi_table")
    if psi_table is not None:
        n_sig = int((psi_table["severity"] == "Significant").sum())
        n_mod = int((psi_table["severity"] == "Moderate").sum())
        if n_sig > 0:
            top_feat = psi_table[psi_table["severity"] == "Significant"]["feature"].iloc[0]
            findings.append({
                "title":       f"{n_sig} feature(s) with significant PSI (>0.25)",
                "risk_level":  "High" if n_sig >= 3 else "Medium",
                "description": (f"{n_sig} input features show major distributional shift "
                                f"between train and monitor periods. Most affected: {top_feat}. "
                                "Model may no longer be representative of current population."),
            })
        if n_mod > 0:
            findings.append({
                "title":       f"{n_mod} feature(s) with moderate PSI (0.10–0.25)",
                "risk_level":  "Low",
                "description": (f"{n_mod} features show moderate distributional shift. "
                                "Monitor closely; no immediate model action required."),
            })

    # --- Prediction drift ---
    pred_psi = results.get("prediction_psi", 0)
    if pred_psi > 0.25:
        findings.append({
            "title":       f"Prediction score PSI = {pred_psi:.3f} (Significant)",
            "risk_level":  "High",
            "description": ("Model output distribution has shifted significantly. "
                            "Calibration and threshold may require recalibration."),
        })
    elif pred_psi > 0.10:
        findings.append({
            "title":       f"Prediction score PSI = {pred_psi:.3f} (Moderate)",
            "risk_level":  "Medium",
            "description": "Moderate shift in model score distribution. Watch trend.",
        })

    # --- Target drift ---
    target = results.get("target_drift", {})
    if target.get("flagged"):
        findings.append({
            "title":       f"Target drift: Δ={target.get('delta_pp', 0):+.1f}pp default rate",
            "risk_level":  "High",
            "description": ("Default rate has shifted by more than 5pp between train and monitor "
                            "periods, indicating possible concept drift or portfolio mix change."),
        })

    # --- Calibration ---
    cal_status = results.get("calibration_monitor", {}).get("calibration_status", "OK")
    ece_after  = results.get("ece_after", 0)
    if cal_status == "DEGRADED":
        findings.append({
            "title":       "Calibration degraded on monitor set",
            "risk_level":  "Medium",
            "description": ("ECE has increased on the monitor set relative to the test set. "
                            "PD scores may no longer reflect true default probabilities in production."),
        })
    if ece_after > 0.05:
        findings.append({
            "title":       f"Post-calibration ECE = {ece_after:.3f} (above 0.05 threshold)",
            "risk_level":  "Low",
            "description": "Model calibration is acceptable but not tight. Further isotonic regression could help.",
        })

    # --- Fairness ---
    for key in ["region", "age_group"]:
        fair = results.get("fairness", {}).get(key, {})
        n_flagged = fair.get("n_flagged_80_rule", 0)
        if n_flagged > 0:
            findings.append({
                "title":       f"Disparate impact: {n_flagged} group(s) fail 80% rule ({key})",
                "risk_level":  "High",
                "description": (f"{n_flagged} demographic group(s) have approval rates below 80% "
                                f"of the highest group rate, triggering the ECOA disparate impact test "
                                f"(proxy variable: {key}). Legal review recommended."),
            })

    if not findings:
        findings.append({
            "title":       "No material findings",
            "risk_level":  "Low",
            "description": "All monitored indicators are within acceptable bounds.",
        })

    return findings


def render_report(
    results: dict,
    template_dir: str = "reports/templates",
    output_dir:   str = "reports/output",
    processed_dir: str = "data/processed",
) -> str:
    """
    Render the validation report HTML using Jinja2.
    Returns the path to the saved report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    findings       = build_findings(results)
    recommendation = _auto_recommendation(findings)

    # Embed images as base64 so the HTML is fully self-contained
    images = {
        "roc_pr":          _embed_image(f"{processed_dir}/roc_pr_test.png"),
        "shap_bar":        _embed_image(f"{processed_dir}/shap_bar.png"),
        "calibration":     _embed_image(f"{processed_dir}/calibration_comparison.png"),
        "calibration_mon": _embed_image(f"{processed_dir}/calibration_monitor.png"),
        "threshold_sweep": _embed_image(f"{processed_dir}/threshold_sweep.png"),
        "pred_drift":      _embed_image(f"{processed_dir}/prediction_drift.png"),
        "approval_region": _embed_image(f"{processed_dir}/approval_rates_census_region.png"),
        "approval_age":    _embed_image(f"{processed_dir}/approval_rates_credit_age_group.png"),
        "eo_region":       _embed_image(f"{processed_dir}/equalized_odds_census_region.png"),
        "eo_age":          _embed_image(f"{processed_dir}/equalized_odds_credit_age_group.png"),
    }

    # Build psi_table and fairness tables as HTML strings
    psi_table_html = ""
    if results.get("psi_table") is not None:
        top20 = results["psi_table"].head(20)
        psi_table_html = top20.to_html(index=False, classes="table", border=0)

    ks_table_html = ""
    if results.get("ks_table") is not None:
        top20_ks = results["ks_table"].head(20)
        ks_table_html = top20_ks.to_html(index=False, classes="table", border=0)

    threshold_candidates = results.get("threshold_candidates", {})

    env      = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template("validation_report.html")

    html = template.render(
        report_date=date.today().isoformat(),
        model_name="Logistic Regression Credit Default Model",
        model_version="1.0",
        dataset="Lending Club 2007–2014",
        train_period="< 2015-01-01",
        monitor_period=">= 2015-01-01",

        # Performance
        train_auc=results.get("train_metrics", {}).get("auc_roc", "N/A"),
        train_gini=results.get("train_metrics", {}).get("gini", "N/A"),
        train_ks=results.get("train_metrics", {}).get("ks", "N/A"),
        test_auc=results.get("test_metrics", {}).get("auc_roc", "N/A"),
        test_gini=results.get("test_metrics", {}).get("gini", "N/A"),
        test_ks=results.get("test_metrics", {}).get("ks", "N/A"),
        monitor_auc=results.get("monitor_metrics", {}).get("auc_roc", "N/A"),
        monitor_gini=results.get("monitor_metrics", {}).get("gini", "N/A"),
        monitor_ks=results.get("monitor_metrics", {}).get("ks", "N/A"),
        perf_flag=_performance_flag(
            results.get("train_metrics", {}).get("auc_roc", 0.5),
            results.get("monitor_metrics", {}).get("auc_roc", 0.5),
        ),

        # Calibration
        ece_before=results.get("ece_before", "N/A"),
        ece_after=results.get("ece_after", "N/A"),
        calibration_monitor=results.get("calibration_monitor", {}),

        # Drift
        psi_table_html=psi_table_html,
        ks_table_html=ks_table_html,
        prediction_psi=results.get("prediction_psi", "N/A"),
        target_drift=results.get("target_drift", {}),
        n_sig_psi=results.get("n_sig_psi", 0),
        n_moderate_psi=results.get("n_moderate_psi", 0),
        n_drifted_ks=results.get("n_drifted_ks", 0),

        # Fairness
        fairness=results.get("fairness", {}),

        # Threshold
        threshold_candidates=threshold_candidates,

        # Findings and recommendation
        findings=findings,
        recommendation=recommendation,

        # Images
        images=images,
    )

    out_path = Path(output_dir) / f"validation_report_{date.today().isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[report] Validation report saved to {out_path}")
    return str(out_path)
