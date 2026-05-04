"""Text feature engineering: cleaning helpers and TF-IDF builder.

Vectorizers fit on the train split only; this module exposes a small API so
downstream model code never accidentally fits on validation or test data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Minimal cleaner: lowercase, collapse whitespace, strip non-letters except basic
# punctuation. We deliberately avoid heavy stemming so token-level SHAP later
# remains interpretable.
_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s.,%$\-]")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = _NON_TEXT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


@dataclass
class TfidfArtifacts:
    vectorizer: TfidfVectorizer
    feature_names: np.ndarray


def fit_tfidf(texts: pd.Series, max_features: int = 20000) -> TfidfArtifacts:
    """Fit a (1,2)-gram TF-IDF vectorizer on the cleaned training texts only."""
    cleaned = texts.map(clean_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    vectorizer.fit(cleaned)
    return TfidfArtifacts(vectorizer=vectorizer, feature_names=vectorizer.get_feature_names_out())


def transform_tfidf(artifacts: TfidfArtifacts, texts: pd.Series):
    return artifacts.vectorizer.transform(texts.map(clean_text))
