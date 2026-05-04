"""End-to-end orchestrator: data -> models -> evaluation -> signal/backtest.

Each stage is opt-in/out via the ``--skip`` flag so it can be re-run
independently without re-doing the expensive fine-tune.
"""
from __future__ import annotations

import argparse

from config import ensure_dirs


STAGES = ("data", "tfidf", "finbert_zero", "finbert_finetune", "evaluation", "signal", "backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip", type=str, default="", help="comma-separated stages to skip")
    p.add_argument("--only", type=str, default="", help="comma-separated stages to run (overrides --skip)")
    return p.parse_args()


def _resolve_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(",") if s.strip()}
        return [s for s in STAGES if s in wanted]
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    return [s for s in STAGES if s not in skip]


def main() -> None:
    args = parse_args()
    ensure_dirs()

    stages = _resolve_stages(args)
    print(f"[pipeline] running stages: {stages}")

    if "data" in stages:
        from data.load import main as data_main

        data_main()

    if "tfidf" in stages:
        from models.tfidf_baseline import main as tfidf_main

        tfidf_main()

    if "finbert_zero" in stages:
        from models.finbert_zero import main as finbert_zero_main

        finbert_zero_main()

    if "finbert_finetune" in stages:
        from models.finbert_finetune import main as finbert_ft_main

        finbert_ft_main()

    if "evaluation" in stages:
        from evaluation.metrics import main as metrics_main
        from evaluation.calibration import main as calibration_main
        from evaluation.error_analysis import main as errors_main

        metrics_main()
        calibration_main()
        errors_main()

    if "signal" in stages:
        from data.returns import main as returns_main
        from signals.score import main as score_main

        returns_main()
        score_main()

    if "backtest" in stages:
        import pandas as pd

        from config import PREDICTIONS_DIR, RAW_DIR
        from signals.backtest import run as backtest_run
        from signals.performance import plot_cumulative, plot_sentiment_timeseries, pretty_summary

        sentiment = pd.read_csv(PREDICTIONS_DIR / "weekly_sector_sentiment.csv", parse_dates=["week"]).set_index("week")
        returns = pd.read_csv(RAW_DIR / "sector_returns_weekly.csv", parse_dates=["Date"]).set_index("Date")
        results = backtest_run(sentiment, returns)
        plot_cumulative(results["signal"].weekly, results["random_benchmark"].weekly)
        plot_sentiment_timeseries(sentiment)
        print(pretty_summary(results["signal"].summary, "FinBERT signal"))
        print(pretty_summary(results["random_benchmark"].summary, "Random benchmark"))


if __name__ == "__main__":
    main()
