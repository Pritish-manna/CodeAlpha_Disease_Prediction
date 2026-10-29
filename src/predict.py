"""Load a saved model bundle, validate user input and return a risk estimate."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import DISEASES


def load_bundle(key: str) -> dict:
    cfg = DISEASES[key]
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for {cfg.title}. Train it first with: "
            f"python -m src.train --disease {key}")
    return joblib.load(cfg.model_path)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def validate_inputs(key: str, values: dict) -> list:
    """Return a list of human-readable problems (empty list = inputs are valid)."""
    errors = []
    for feat in DISEASES[key].features:
        value = values.get(feat.name)
        if _is_missing(value):
            if not feat.allow_missing:
                errors.append(f"{feat.label} is required.")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{feat.label} must be a number.")
            continue
        if feat.kind == "choice":
            if number not in feat.options:
                errors.append(f"{feat.label}: '{value}' is not one of the allowed options.")
        elif feat.min is not None and not (feat.min <= number <= feat.max):
            errors.append(f"{feat.label} must be between {feat.min:g} and {feat.max:g}.")
    return errors


def predict_risk(key: str, values: dict, bundle: dict | None = None) -> dict:
    """Predict the probability of the positive class for one patient.

    ``values`` maps feature name -> number (or None when 'not available'). Missing values are
    filled by the imputer fitted during training, exactly as in the training pipeline.
    """
    errors = validate_inputs(key, values)
    if errors:
        raise ValueError("; ".join(errors))
    bundle = bundle or load_bundle(key)
    row = pd.DataFrame([[np.nan if _is_missing(values.get(f)) else float(values[f])
                         for f in bundle["features"]]], columns=bundle["features"], dtype=float)
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    higher = probability >= bundle["threshold"]

    notes = []
    for feat, (lo, hi) in bundle["training_ranges"].items():
        value = row.at[0, feat]
        if not np.isnan(value) and not (lo <= value <= hi):
            notes.append(f"{feat} = {value:g} is outside the range seen in the training data "
                         f"({lo:g} to {hi:g}); the estimate may be less reliable.")
    return {
        "probability": probability,
        "label": "Higher predicted risk" if higher else "Lower predicted risk",
        "predicted_class": int(higher),
        "threshold": bundle["threshold"],
        "model_name": bundle["model_name"],
        "warnings": notes,
    }
