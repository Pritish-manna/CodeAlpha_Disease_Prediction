"""Cleaning, splitting and the scikit-learn preprocessing pipeline.

Anything that *learns* from data (imputation values, scaling, one-hot categories) lives inside
a ``ColumnTransformer`` that is fitted on the training split only and saved together with the
model, so there is no train/test leakage and prediction uses exactly the same steps.
Stateless cleaning (removing duplicates, marking impossible values as missing) happens before
the split because it uses no statistics from the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET, TEST_SIZE, DiseaseConfig


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Basic facts about a raw dataset: shape, dtypes, missing values, duplicates."""
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items()},
        "duplicates": int(df.duplicated().sum()),
    }


def clean_dataset(cfg: DiseaseConfig, df: pd.DataFrame):
    """Return (clean dataframe, log). Uses no statistics learnt from the data."""
    needed = cfg.feature_names + [TARGET]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(f"{cfg.csv_name} is missing expected columns: {absent}")

    out = df[needed].copy()
    for col in cfg.feature_names:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    log = {"rows_raw": int(len(out)), "zero_marked_missing": {}, "out_of_range_marked_missing": {}}

    # 0 is physiologically impossible for these columns -> treat as missing
    for col in cfg.zero_invalid:
        mask = out[col] == 0
        if mask.any():
            log["zero_marked_missing"][col] = int(mask.sum())
            out.loc[mask, col] = np.nan

    # values outside the plausible range defined in the config -> treat as missing
    for feat in cfg.features:
        if feat.kind == "choice" or feat.min is None or feat.max is None:
            continue
        col = out[feat.name]
        mask = col.notna() & ((col < feat.min) | (col > feat.max))
        if mask.any():
            log["out_of_range_marked_missing"][feat.name] = int(mask.sum())
            out.loc[mask, feat.name] = np.nan

    before = len(out)
    out = out.dropna(subset=[TARGET])
    log["rows_without_target_dropped"] = int(before - len(out))
    out[TARGET] = out[TARGET].astype(int)

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    log["duplicates_removed"] = int(before - len(out))
    log["rows_clean"] = int(len(out))
    log["missing_after_cleaning"] = {c: int(n) for c, n in out[cfg.feature_names].isna().sum().items() if n}
    return out, log


def build_preprocessor(cfg: DiseaseConfig) -> ColumnTransformer:
    """Median-impute + standardise numeric columns; mode-impute + one-hot encode categorical ones."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    transformers = [("num", numeric, cfg.numeric_features)]
    if cfg.categorical_features:
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", categorical, cfg.categorical_features))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(cfg: DiseaseConfig, df: pd.DataFrame):
    """80:20 stratified split. Returns X_train, X_test, y_train, y_test."""
    X = df[cfg.feature_names]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
