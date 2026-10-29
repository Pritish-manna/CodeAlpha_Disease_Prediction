"""Create / load the three CSV datasets.

* Breast cancer ships inside scikit-learn (the UCI WDBC data), so it works offline.
* Heart disease and diabetes are downloaded once from public sources and cached in data/.
  If you have no internet access, download the files manually and save them as
  data/heart.csv and data/diabetes.csv (see the README for the expected columns).
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, DISEASES, TARGET

HEART_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
HEART_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/data.csv",
]
PIMA_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def _build_heart() -> pd.DataFrame:
    last_error = None
    for url in HEART_URLS:
        try:
            df = pd.read_csv(url, header=None, names=HEART_COLUMNS, na_values="?")
            if "num" not in df.columns or df["age"].isna().all():  # file with a header row
                df = pd.read_csv(url, na_values="?")
            df = df[HEART_COLUMNS].apply(pd.to_numeric, errors="coerce")
            df[TARGET] = (df["num"] > 0).astype(int)  # 0 = no disease, 1-4 = disease present
            return df.drop(columns="num")
        except Exception as exc:  # try the next source
            last_error = exc
    raise RuntimeError(f"Could not download the heart disease data: {last_error}")


def _build_diabetes() -> pd.DataFrame:
    try:
        df = pd.read_csv(PIMA_URL, header=None, names=PIMA_COLUMNS)
        return df.rename(columns={"Outcome": TARGET})
    except Exception as first_error:
        try:  # fallback: OpenML copy of the same dataset
            from sklearn.datasets import fetch_openml

            frame = fetch_openml("diabetes", version=1, as_frame=True).frame
            frame = frame.rename(columns={"preg": "Pregnancies", "plas": "Glucose",
                                          "pres": "BloodPressure", "skin": "SkinThickness",
                                          "insu": "Insulin", "mass": "BMI",
                                          "pedi": "DiabetesPedigreeFunction", "age": "Age"})
            frame[TARGET] = (frame["class"] == "tested_positive").astype(int)
            return frame.drop(columns="class")
        except Exception as second_error:
            raise RuntimeError(f"Could not download the diabetes data: {first_error}; "
                               f"fallback failed: {second_error}")


def _build_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.data.copy()
    df[TARGET] = 1 - data.target  # scikit-learn: 0 = malignant; flipped so 1 = malignant
    return df


_BUILDERS = {"heart": _build_heart, "diabetes": _build_diabetes, "breast_cancer": _build_breast_cancer}


def ensure_dataset(key: str) -> None:
    """Create data/<disease>.csv if it does not exist yet."""
    cfg = DISEASES[key]
    if cfg.csv_path.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _BUILDERS[key]().to_csv(cfg.csv_path, index=False)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"{cfg.csv_path.name} is missing and could not be downloaded ({exc}). "
            f"Download the dataset manually ({cfg.source_url}) and save it as {cfg.csv_path} "
            f"with columns {cfg.feature_names + [TARGET]}."
        ) from exc


def ensure_all_datasets() -> None:
    for key in DISEASES:
        ensure_dataset(key)


def load_dataset(key: str) -> pd.DataFrame:
    """Load a disease CSV (creating it first if needed)."""
    ensure_dataset(key)
    return pd.read_csv(DISEASES[key].csv_path)
