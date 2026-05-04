"""Project-wide constants and helpers.

Centralizing seeds, paths, and label maps keeps every module in lock-step so
results are reproducible and the same splits feed every model.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
RAW_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SUMMARY_DIR = ARTIFACTS_DIR / "summary"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = PROJECT_DIR / "models" / "checkpoints"

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_ID.items()}

# Sector ETF universe used for the long-short backtest. Order is fixed for
# reproducible random benchmark shuffling.
SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE")

# Lightweight keyword -> sector mapping. Used only when transcript-level sector
# metadata is missing in ECTSum; downstream code falls back to "OTHER" if no
# keyword matches.
SECTOR_KEYWORDS = {
    "XLK": ("technology", "software", "semiconductor", "cloud", "internet", "platform"),
    "XLF": ("bank", "financial", "insurance", "asset management", "credit", "lending"),
    "XLE": ("oil", "gas", "energy", "petroleum", "drilling", "refining"),
    "XLV": ("health", "pharma", "biotech", "medical", "hospital", "device"),
    "XLI": ("industrial", "manufacturing", "aerospace", "defense", "logistics", "machinery"),
    "XLP": ("consumer staples", "beverage", "tobacco", "household", "grocery", "snack"),
    "XLU": ("utility", "electric power", "water utility", "natural gas utility"),
    "XLB": ("materials", "chemical", "mining", "metals", "paper", "construction materials"),
    "XLRE": ("real estate", "reit", "property", "lease", "occupancy"),
}


def ensure_dirs() -> None:
    for d in (
        SPLITS_DIR,
        RAW_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
        SUMMARY_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch so PhraseBank splits and model training are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
