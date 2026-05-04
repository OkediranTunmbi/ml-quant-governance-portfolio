"""Long-short sector ETF backtest driven by weekly sentiment.

Conventions:
- ``sentiment.iloc[t]`` is the score at week T.
- ``returns.iloc[t]`` is the realized return for week T+1 (we shift returns
  back by one week so a row labelled "week T" already holds the future return).
  This mirrors a tradable rule: at week T's close, decide positions; collect
  the week T+1 return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import PREDICTIONS_DIR, SECTOR_ETFS, SEED, SUMMARY_DIR, ensure_dirs


@dataclass
class BacktestResult:
    weekly: pd.DataFrame  # columns: long_ret, short_ret, port_ret, cum_ret, sector picks
    summary: dict[str, float]


def align_sentiment_to_forward_returns(
    sentiment_weekly: pd.DataFrame,
    returns_weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align sentiment(T) with return(T+1) by shifting returns one week earlier.

    Returns indexed at week T means "the return realised between week T's close
    and week T+1's close" so the trade is fully out-of-sample relative to the
    score that triggered it.
    """
    common_cols = [c for c in SECTOR_ETFS if c in sentiment_weekly.columns and c in returns_weekly.columns]
    sent = sentiment_weekly[common_cols].copy()
    rets = returns_weekly[common_cols].shift(-1)  # next-week returns at row T
    idx = sent.index.intersection(rets.index)
    return sent.loc[idx].copy(), rets.loc[idx].copy()


def _pick_top_bottom(sent_row: pd.Series, k: int = 2) -> tuple[list[str], list[str]]:
    valid = sent_row.dropna()
    if len(valid) < 2 * k:
        return [], []
    ranked = valid.sort_values(ascending=False)
    return list(ranked.index[:k]), list(ranked.index[-k:])


def long_short_backtest(
    sentiment: pd.DataFrame,
    forward_returns: pd.DataFrame,
    k: int = 2,
) -> BacktestResult:
    rows: list[dict] = []
    for week, sent_row in sentiment.iterrows():
        if week not in forward_returns.index:
            continue
        long_picks, short_picks = _pick_top_bottom(sent_row, k=k)
        if not long_picks:
            continue
        rets_row = forward_returns.loc[week]
        long_ret = rets_row[long_picks].mean()
        short_ret = rets_row[short_picks].mean()
        port_ret = long_ret - short_ret
        rows.append(
            {
                "week": week,
                "long_picks": ",".join(long_picks),
                "short_picks": ",".join(short_picks),
                "long_ret": float(long_ret),
                "short_ret": float(short_ret),
                "port_ret": float(port_ret),
            }
        )
    weekly = pd.DataFrame(rows).set_index("week").sort_index()
    weekly = weekly.dropna(subset=["port_ret"])
    weekly["cum_ret"] = (1.0 + weekly["port_ret"]).cumprod() - 1.0
    summary = _summarize(weekly["port_ret"])
    return BacktestResult(weekly=weekly, summary=summary)


def shuffled_benchmark(
    sentiment: pd.DataFrame,
    forward_returns: pd.DataFrame,
    k: int = 2,
    seed: int = SEED,
) -> BacktestResult:
    """Null hypothesis: same long-short rule with sentiment shuffled within each week.

    Shuffling per row preserves the cross-sectional rank distribution but
    destroys any information content tied to the actual sentiment score, so any
    edge our model has should beat this benchmark.
    """
    rng = np.random.default_rng(seed)
    sent_shuffled = sentiment.copy()
    for week in sent_shuffled.index:
        row = sent_shuffled.loc[week].to_numpy(copy=True)
        rng.shuffle(row)
        sent_shuffled.loc[week] = row
    return long_short_backtest(sent_shuffled, forward_returns, k=k)


def _summarize(returns: pd.Series, periods_per_year: int = 52) -> dict[str, float]:
    if returns.empty:
        return {"sharpe": float("nan"), "hit_rate": float("nan"), "cum_return": float("nan"), "max_drawdown": float("nan"), "n_weeks": 0}
    sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(periods_per_year) if returns.std(ddof=1) > 0 else float("nan")
    hit_rate = float((returns > 0).mean())
    cum = (1.0 + returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "sharpe": float(sharpe),
        "hit_rate": hit_rate,
        "cum_return": float(cum.iloc[-1] - 1.0),
        "max_drawdown": max_dd,
        "n_weeks": int(len(returns)),
    }


def run(
    sentiment_weekly: pd.DataFrame,
    returns_weekly: pd.DataFrame,
    k: int = 2,
) -> dict[str, BacktestResult]:
    ensure_dirs()
    sent, fwd = align_sentiment_to_forward_returns(sentiment_weekly, returns_weekly)
    signal = long_short_backtest(sent, fwd, k=k)
    benchmark = shuffled_benchmark(sent, fwd, k=k, seed=SEED)
    signal.weekly.to_csv(PREDICTIONS_DIR / "backtest_signal_weekly.csv")
    benchmark.weekly.to_csv(PREDICTIONS_DIR / "backtest_random_weekly.csv")
    summary = pd.DataFrame({"signal": signal.summary, "random_benchmark": benchmark.summary}).T
    summary.to_csv(SUMMARY_DIR / "backtest_summary.csv")
    return {"signal": signal, "random_benchmark": benchmark}
