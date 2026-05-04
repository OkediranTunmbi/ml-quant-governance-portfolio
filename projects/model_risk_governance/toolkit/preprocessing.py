"""
preprocessing.py — Feature engineering, encoding, and scaling pipeline.

All transformers are fit on the training set ONLY and then applied to test/
monitor sets.  This mirrors production deployment: the production system
receives raw origination data and applies the same fitted transformers.
Fitting on test or monitor data would be "data leakage via preprocessing."
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# SAFE ORIGINATION-TIME FEATURES
# These are the only fields observable at loan origination — the moment the
# model would run in production.  Post-origination columns are excluded in
# data_validation.py.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mths_since_last_delinq", "mths_since_last_record",
    "credit_age_months", "fico_mid",
]

# OrdinalEncoder: grade (A→G) and sub_grade (A1→G5) have a meaningful order —
# A-grade is the lowest risk, G-grade is the highest.  Preserving this order
# lets the linear model exploit the monotone relationship without needing 35
# dummy columns for sub_grade alone.
ORDINAL_FEATURES = ["grade", "sub_grade"]
ORDINAL_CATEGORIES = {
    "grade":     ["A", "B", "C", "D", "E", "F", "G"],
    "sub_grade": [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)],
}

# OneHotEncoder: remaining categoricals have no natural order.
CATEGORICAL_FEATURES = [
    "term", "emp_length", "home_ownership", "verification_status",
    "purpose", "initial_list_status", "application_type",
]

TARGET = "default"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive new origination-time features.
    - credit_age_months: number of months between the borrower's oldest
      account and the loan issue date.  Longer credit history generally
      predicts lower default risk.
    - fico_mid: midpoint of the reported FICO range.  Avoids choosing
      between _low and _high arbitrarily and reduces feature collinearity.
    """
    df = df.copy()

    # Parse issue_d if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df["issue_d"]):
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    # Parse earliest_cr_line and compute credit age in months
    if "earliest_cr_line" in df.columns:
        ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["credit_age_months"] = (
            (df["issue_d"].dt.year - ecl.dt.year) * 12
            + (df["issue_d"].dt.month - ecl.dt.month)
        )
    else:
        df["credit_age_months"] = np.nan

    # FICO midpoint
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    else:
        df["fico_mid"] = np.nan

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct the sklearn ColumnTransformer.  The object is returned unfitted
    so callers can explicitly fit on training data only — a deliberate design
    choice to prevent accidental leakage.
    """
    numeric_pipeline = Pipeline([
        # Median imputation is robust to outliers and sensible for financial
        # data (e.g. mths_since_last_delinq is NaN when no delinquency exists).
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[ORDINAL_CATEGORIES["grade"], ORDINAL_CATEGORIES["sub_grade"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",  # new categories in monitor set are ignored safely
            sparse_output=False,
        )),
    ])

    # Filter to only use ordinal/categorical features that are actually present
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,     NUMERIC_FEATURES),
            ("ord",  ordinal_pipeline,     ORDINAL_FEATURES),
            ("cat",  categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def filter_available_features(df: pd.DataFrame, feature_list: list) -> list:
    """Return only features that actually exist in the dataframe."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"[preprocessing] WARNING: {len(missing)} features not found in data: {missing}")
    return available


def fit_and_save(
    train_df: pd.DataFrame,
    save_path: str = "data/processed/preprocessor.pkl",
) -> tuple[ColumnTransformer, np.ndarray]:
    """
    Fit the preprocessor on training data and save it.
    Returns (fitted_preprocessor, X_train_transformed).
    """
    train_df = engineer_features(train_df)

    # Use only features present in this dataset
    num_feats = filter_available_features(train_df, NUMERIC_FEATURES)
    ord_feats = filter_available_features(train_df, ORDINAL_FEATURES)
    cat_feats = filter_available_features(train_df, CATEGORICAL_FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), num_feats),
            ("ord",  Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    categories=[
                        ORDINAL_CATEGORIES.get(f, "auto") for f in ord_feats
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), ord_feats),
            ("cat",  Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    X_train = preprocessor.fit_transform(train_df)
    print(f"[preprocessing] Fitted preprocessor on {X_train.shape[0]:,} rows, "
          f"{X_train.shape[1]} features.")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"[preprocessing] Saved preprocessor to {save_path}")

    return preprocessor, X_train


def transform(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> np.ndarray:
    """Apply a FITTED preprocessor to new data (test or monitor)."""
    df = engineer_features(df)
    return preprocessor.transform(df)


def load_preprocessor(path: str = "data/processed/preprocessor.pkl") -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract output feature names from the fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())
