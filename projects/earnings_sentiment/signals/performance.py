"""Performance plots and pretty-printed summary blocks for the backtest."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOTS_DIR, ensure_dirs


def plot_cumulative(
    signal: pd.DataFrame,
    benchmark: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(signal.index, signal["cum_ret"], label="FinBERT signal", lw=2)
    ax.plot(benchmark.index, benchmark["cum_ret"], label="Random benchmark", lw=1.5, ls="--", color="grey")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Cumulative return: FinBERT long-short vs random benchmark")
    ax.set_ylabel("cumulative return")
    ax.set_xlabel("week")
    ax.legend()
    fig.tight_layout()
    out = out_path or (PLOTS_DIR / "cumulative_return.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sentiment_timeseries(weekly_sentiment: pd.DataFrame, out_path: Path | None = None) -> Path:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in weekly_sentiment.columns:
        ax.plot(weekly_sentiment.index, weekly_sentiment[col], lw=1, label=col, alpha=0.85)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_title("Weekly sector sentiment (P(positive) - P(negative))")
    ax.set_ylabel("sentiment score")
    ax.set_xlabel("week")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = out_path or (PLOTS_DIR / "sector_sentiment.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def pretty_summary(summary: dict[str, float], title: str = "Backtest summary") -> str:
    lines = [
        f"=== {title} ===",
        f"Annualized Sharpe : {summary.get('sharpe', float('nan')):.3f}",
        f"Hit rate          : {summary.get('hit_rate', float('nan')):.2%}",
        f"Cumulative return : {summary.get('cum_return', float('nan')):+.2%}",
        f"Max drawdown      : {summary.get('max_drawdown', float('nan')):+.2%}",
        f"Weeks traded      : {int(summary.get('n_weeks', 0))}",
    ]
    return "\n".join(lines)
