"""Text cleaning and TF-IDF feature engineering.

HOW TEXT BECOMES A NUMBER — TWO PATHS
--------------------------------------
This module covers the *classical* NLP path. Understanding it makes the
transformer (FinBERT) path in Stage 3 easier to appreciate by contrast.

  TF-IDF path (this file):
    raw text
      → clean (lowercase, strip noise)
      → tokenize by whitespace / punctuation
      → count n-grams in a fixed vocabulary
      → weight by inverse document frequency
      → sparse float vector  (e.g. shape: [n_samples, 20000])

  FinBERT / transformer path (models/finbert_finetune.py):
    raw text
      → WordPiece tokenizer  →  integer token IDs + attention mask
      → embedding lookup layer  →  each token becomes a 768-d float vector
      → 12 transformer layers refine each vector using *all other tokens*
      → [CLS] token vector used as the sentence representation
      → linear classification head on top
      → dense float vector  (shape: [n_samples, 768])

THE KEY DIFFERENCE
  TF-IDF treats every word independently — it's a "bag of words."
  "The company's revenue did not grow" and "The company's revenue grew"
  have almost identical TF-IDF vectors because both contain "revenue" and "grow."

  A transformer embedding encodes *meaning in context*: the word "grow" after
  "did not" produces a different vector than "grow" alone. That context-
  sensitivity is what makes transformers far more powerful for language tasks.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "models" / "artifacts" / "tfidf"

# Canonical label order — must match data/load.py.
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_ORDER)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_ORDER)}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize raw financial text for TF-IDF.

    WHY KEEP $, %, -, .?
    Financial text is full of domain-specific signals: '-5%' is bearish,
    '$2.3B revenue' is a magnitude signal, '52-week high' is a price anchor.
    Stripping these away loses real information. We keep them and strip
    everything else that adds noise without signal (HTML, brackets, etc.).
    """
    text = str(text).lower()
    # Keep letters, digits, spaces, and a small set of financially meaningful chars.
    text = re.sub(r"[^a-z0-9\s\$\%\.\,\-]", " ", text)
    # Collapse any sequence of whitespace into a single space.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# TF-IDF vectorizer
# ---------------------------------------------------------------------------

def fit_tfidf(train_texts: pd.Series) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on training text only and return it.

    WHY (1, 2)-GRAMS?
    Bigrams capture negation and multi-word expressions that unigrams miss:
      'not profitable' → tokens: ['not', 'profitable', 'not profitable']
    Without bigrams, 'not profitable' and 'profitable' would look almost
    identical to the model. Trigrams add too much sparsity for little gain
    on short sentences (PhraseBank sentences average ~14 words).

    WHY max_features=20000?
    Uncapped, a 20k-sentence corpus with bigrams can produce 100k+ features.
    Most of them appear only once (hapax legomena) and are pure noise.
    Capping at 20k keeps the vocabulary to the most informative tokens
    and keeps memory manageable.

    WHY sublinear_tf=True?
    Raw term frequency gives "very very good" a TF of 2 for "very" and 1 for
    "good." With sublinear scaling we use 1 + log(tf) instead, so repetition
    has diminishing returns. This reduces the influence of filler words that
    happen to repeat and is standard practice for sentence-level tasks.

    WHY FIT ON TRAIN ONLY?
    If we fit the vocabulary on val or test sentences, the IDF weights would
    reflect how common words are across the *entire* dataset including unseen
    data — that's information leakage. Fitting only on train and then
    transforming val/test simulates what happens in real deployment: the model
    has never seen those sentences before.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",   # normalize accented characters consistently
    )
    cleaned = train_texts.map(clean_text)
    vec.fit(cleaned)
    return vec


def transform(vec: TfidfVectorizer, texts: pd.Series) -> np.ndarray:
    """Apply a fitted vectorizer to any split (val, test, ECTSum, etc.).

    Returns a sparse matrix; downstream models that need dense input
    should call .toarray() themselves.
    """
    cleaned = texts.map(clean_text)
    return vec.transform(cleaned)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def encode_labels(labels: pd.Series) -> list[int]:
    """Convert string labels to integers in the fixed canonical order.

    WHY A FIXED ORDER?
    If we let LabelEncoder infer the order from data, alphabetical order gives
    negative=0, neutral=1, positive=2. That happens to match our LABEL_ORDER
    here, but making it *explicit* means we are never surprised if the data
    arrives in a different order in a future run.
    """
    return [LABEL2ID[label] for label in labels]


def decode_labels(ids: list[int]) -> list[str]:
    return [ID2LABEL[i] for i in ids]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_tfidf(vec: TfidfVectorizer) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, ARTIFACTS_DIR / "vectorizer.joblib")
    print(f"[features] vectorizer saved: {ARTIFACTS_DIR / 'vectorizer.joblib'}")


def load_tfidf() -> TfidfVectorizer:
    return joblib.load(ARTIFACTS_DIR / "vectorizer.joblib")
