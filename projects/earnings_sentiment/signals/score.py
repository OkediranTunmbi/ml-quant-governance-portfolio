"""Score ECTSum transcripts with the fine-tuned FinBERT and aggregate to sectors.

Per-transcript score: P(positive) - P(negative).  This compresses three softmax
probabilities into a single value in [-1, +1]; positive means net-bullish.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CHECKPOINTS_DIR,
    LABELS,
    PREDICTIONS_DIR,
    SECTOR_ETFS,
    SECTOR_KEYWORDS,
    SEED,
    ensure_dirs,
    set_seed,
)
from data.load import load_ectsum
from data.sectors import build_ticker_sector_map
from models.finbert_zero import HF_ID, _label_remap

BATCH_SIZE = 16
MAX_LEN = 512  # transcripts are long; we truncate generously and rely on the [CLS] head


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_finetuned_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, np.ndarray]:
    ckpt = CHECKPOINTS_DIR / "finbert_best"
    if ckpt.exists() and any(ckpt.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    else:
        # Fallback to base FinBERT so the signal pipeline works even if fine-tuning
        # has not been run yet (useful for end-to-end smoke tests).
        print("[signal] WARNING: no fine-tuned checkpoint found; falling back to base FinBERT")
        tokenizer = AutoTokenizer.from_pretrained(HF_ID)
        model = AutoModelForSequenceClassification.from_pretrained(HF_ID)
    id2label = {int(i): str(lbl).lower() for i, lbl in model.config.id2label.items()}
    perm = _label_remap([id2label[i] for i in range(len(id2label))])
    return tokenizer, model, perm


def _score_texts(texts: list[str]) -> np.ndarray:
    set_seed(SEED)
    tokenizer, model, perm = _load_finetuned_model()
    device = _device()
    model.to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [t if isinstance(t, str) else "" for t in texts[start : start + BATCH_SIZE]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        chunks.append(probs[:, perm])
    return np.concatenate(chunks, axis=0)


def _assign_sectors_from_tickers(df: pd.DataFrame) -> pd.Series:
    """Map each row's ticker to a sector ETF using the cached yfinance lookup.

    Falls back to a keyword scan over the transcript text only if a ticker is
    missing or unresolvable; that branch should rarely fire on the GitHub
    source of ECTSum.
    """
    sector_map = build_ticker_sector_map(df["ticker"].dropna().unique().tolist())
    sector_lookup = dict(zip(sector_map["ticker"].astype(str), sector_map["sector_etf"]))

    def _resolve(row: pd.Series) -> str:
        ticker = str(row.get("ticker", "")).strip()
        etf = sector_lookup.get(ticker)
        if isinstance(etf, str) and etf in SECTOR_ETFS:
            return etf
        haystack = str(row.get("transcript", "")).lower()
        for cand_etf, keywords in SECTOR_KEYWORDS.items():
            if any(kw in haystack for kw in keywords):
                return cand_etf
        return "OTHER"

    return df.apply(_resolve, axis=1)


def score_ectsum() -> pd.DataFrame:
    """Score every ECTSum transcript and return a per-row sentiment table."""
    ensure_dirs()
    df = load_ectsum().copy()
    if not {"transcript", "date", "ticker"}.issubset(df.columns):
        raise RuntimeError(
            "ECTSum frame is missing required columns (transcript/date/ticker). "
            "Run `python -c \"from data.load import load_ectsum; load_ectsum(force=True)\"` "
            "to refresh the cache from the GitHub source."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "transcript", "ticker"]).reset_index(drop=True)

    probs = _score_texts(df["transcript"].tolist())

    out = pd.DataFrame(probs, columns=[f"prob_{c}" for c in LABELS])
    out["sentiment_score"] = out["prob_positive"] - out["prob_negative"]
    out["date"] = df["date"].values
    out["ticker"] = df["ticker"].astype(str).values
    out["sector_etf"] = _assign_sectors_from_tickers(df).values
    out["text"] = df["transcript"].astype(str).str.slice(0, 500).values

    out.to_csv(PREDICTIONS_DIR / "ectsum_sentiment.csv", index=False)
    return out


def aggregate_weekly_sector_sentiment(transcript_scores: pd.DataFrame) -> pd.DataFrame:
    """Average each sector's transcript scores within the same Friday-week bucket."""
    df = transcript_scores.copy()
    df = df[df["sector_etf"].isin(SECTOR_ETFS)]
    # ``end_time`` returns a Series of Timestamps; use the ``.dt`` accessor to
    # normalize so we land on midnight Friday and align cleanly with the Friday-
    # closing weekly returns produced upstream.
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    weekly = (
        df.groupby(["week", "sector_etf"])["sentiment_score"]
        .mean()
        .unstack("sector_etf")
        .sort_index()
    )
    weekly = weekly.reindex(columns=list(SECTOR_ETFS))
    weekly.to_csv(PREDICTIONS_DIR / "weekly_sector_sentiment.csv")
    return weekly


def main() -> None:
    scores = score_ectsum()
    weekly = aggregate_weekly_sector_sentiment(scores)
    print(f"[signal] {len(scores):,} transcript scores; weekly sector matrix shape={weekly.shape}")


if __name__ == "__main__":
    main()
