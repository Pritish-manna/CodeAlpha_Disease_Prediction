"""Central configuration: paths, constants and the description of each disease module.

Every disease (dataset + model) is described by a ``DiseaseConfig``. The training code,
the prediction code and the Streamlit app all read the same configuration, which keeps
feature names, valid ranges and labels consistent everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# --------------------------------------------------------------------------- constants
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 80:20 stratified split
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5  # probability >= threshold -> "Higher predicted risk"
TARGET = "target"         # 1 = disease present / positive class, 0 = absent

DISCLAIMER = (
    "This application is developed for educational and machine learning demonstration "
    "purposes only. Predictions are statistical estimates from the trained models and are "
    "not medical diagnoses. Consult a qualified healthcare professional for medical advice."
)


@dataclass
class Feature:
    """One model input. ``kind`` is 'int', 'float' or 'choice' (a fixed set of codes)."""

    name: str
    label: str
    kind: str = "float"
    unit: str = ""
    help: str = ""
    min: Optional[float] = None       # smallest plausible value (also used for validation)
    max: Optional[float] = None       # largest plausible value
    default: Optional[float] = None   # starting value shown in the app (not a recommendation)
    step: Optional[float] = None
    fmt: str = ""                     # number format used by the app
    options: Optional[dict] = None    # code -> label (only for kind == 'choice')
    categorical: bool = False         # True -> one-hot encoded, False -> imputed + scaled
    allow_missing: bool = False       # the original dataset has missing values here
    group: str = ""                   # used to group inputs in the app


@dataclass
class DiseaseConfig:
    key: str
    title: str
    csv_name: str
    features: list
    class_names: tuple                # (negative class, positive class)
    positive_meaning: str
    source: str
    source_url: str
    description: str
    zero_invalid: tuple = field(default_factory=tuple)  # columns where 0 is physiologically impossible

    @property
    def feature_names(self) -> list:
        return [f.name for f in self.features]

    @property
    def numeric_features(self) -> list:
        return [f.name for f in self.features if not f.categorical]

    @property
    def categorical_features(self) -> list:
        return [f.name for f in self.features if f.categorical]

    @property
    def csv_path(self) -> Path:
        return DATA_DIR / self.csv_name

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"{self.key}_model.pkl"

    @property
    def results_path(self) -> Path:
        return REPORTS_DIR / f"{self.key}_results.json"


# --------------------------------------------------------------------------- Heart disease
# Feature descriptions follow the UCI "Heart Disease" (Cleveland) documentation.
HEART = DiseaseConfig(
    key="heart",
    title="Heart Disease",
    csv_name="heart.csv",
    class_names=("No heart disease", "Heart disease"),
    positive_meaning="Heart disease present (UCI attribute 'num' > 0)",
    source="UCI Machine Learning Repository - Heart Disease (Cleveland database)",
    source_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
    description="303 patient records from the Cleveland Clinic with 13 clinical attributes.",
    features=[
        Feature("age", "Age", "int", "years", "Age of the patient.", 20, 100, 55, 1, "%d"),
        Feature("sex", "Sex", "choice", options={1: "Male", 0: "Female"}, default=1),
        Feature("cp", "Chest pain type", "choice", categorical=True, default=4,
                options={1: "Typical angina", 2: "Atypical angina",
                         3: "Non-anginal pain", 4: "Asymptomatic"}),
        Feature("trestbps", "Resting blood pressure", "int", "mm Hg",
                "Resting blood pressure on admission to the hospital.", 80, 220, 130, 1, "%d"),
        Feature("chol", "Serum cholesterol", "int", "mg/dl", "", 100, 600, 240, 1, "%d"),
        Feature("fbs", "Fasting blood sugar > 120 mg/dl", "choice", default=0,
                options={1: "Yes", 0: "No"}),
        Feature("restecg", "Resting ECG result", "choice", categorical=True, default=0,
                options={0: "Normal", 1: "ST-T wave abnormality",
                         2: "Probable or definite left ventricular hypertrophy"}),
        Feature("thalach", "Maximum heart rate achieved", "int", "", "", 60, 220, 150, 1, "%d"),
        Feature("exang", "Exercise-induced angina", "choice", default=0,
                options={1: "Yes", 0: "No"}),
        Feature("oldpeak", "ST depression induced by exercise (relative to rest)", "float", "",
                "", 0.0, 7.0, 1.0, 0.1, "%.1f"),
        Feature("slope", "Slope of the peak exercise ST segment", "choice", categorical=True,
                default=2, options={1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
        Feature("ca", "Number of major vessels coloured by fluoroscopy", "choice", default=0,
                options={0: "0", 1: "1", 2: "2", 3: "3"}, allow_missing=True),
        Feature("thal", "Thalassemia (thallium stress test)", "choice", categorical=True,
                default=3, options={3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
                allow_missing=True),
    ],
)

# --------------------------------------------------------------------------- Diabetes
# Feature descriptions follow the Pima Indians Diabetes dataset documentation (UCI / NIDDK).
DIABETES = DiseaseConfig(
    key="diabetes",
    title="Diabetes",
    csv_name="diabetes.csv",
    class_names=("Non-diabetic", "Diabetic"),
    positive_meaning="Outcome = 1 (tested positive for diabetes)",
    source="Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and "
           "Kidney Diseases), available via UCI / OpenML",
    source_url="https://www.openml.org/search?type=data&id=37",
    description="768 records of Pima Indian women aged 21+ with 8 diagnostic measurements.",
    zero_invalid=("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"),
    features=[
        Feature("Pregnancies", "Number of pregnancies", "int", "", "", 0, 20, 2, 1, "%d"),
        Feature("Glucose", "Plasma glucose concentration (2 h oral glucose tolerance test)",
                "int", "", "", 40, 300, 120, 1, "%d"),
        Feature("BloodPressure", "Diastolic blood pressure", "int", "mm Hg", "", 20, 150, 70, 1, "%d"),
        Feature("SkinThickness", "Triceps skin fold thickness", "int", "mm", "", 1, 100, 25, 1,
                "%d", allow_missing=True),
        Feature("Insulin", "2-hour serum insulin", "int", "mu U/ml", "", 1, 900, 80, 1, "%d",
                allow_missing=True),
        Feature("BMI", "Body mass index", "float", "kg/m\u00b2", "weight in kg / (height in m)\u00b2",
                10.0, 70.0, 30.0, 0.1, "%.1f"),
        Feature("DiabetesPedigreeFunction", "Diabetes pedigree function", "float", "",
                "A score summarising diabetes family history.", 0.05, 2.5, 0.4, 0.01, "%.3f"),
        Feature("Age", "Age", "int", "years", "", 18, 100, 33, 1, "%d"),
    ],
)


# --------------------------------------------------------------------------- Breast cancer
def _breast_cancer_features() -> list:
    """Build the 30 WDBC features from scikit-learn's bundled copy of the UCI dataset.

    The dataset documentation does not define units, so none are shown.
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True).data
    feats = []
    for col in data.columns:
        if col.startswith("mean "):
            group = "Mean values"
        elif col.endswith(" error"):
            group = "Standard error values"
        else:
            group = "Worst (largest) values"
        hi = float(data[col].max()) * 1.5
        step = float(f"{hi / 100:.1g}")
        feats.append(Feature(col, col.capitalize(), "float", "", "", 0.0, hi,
                             float(data[col].median()), step, "%.5f", group=group))
    return feats


BREAST = DiseaseConfig(
    key="breast_cancer",
    title="Breast Cancer",
    csv_name="breast_cancer.csv",
    class_names=("Benign", "Malignant"),
    positive_meaning="Malignant tumour (label flipped so that 1 = malignant)",
    source="UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic), "
           "bundled with scikit-learn",
    source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    description="569 fine-needle-aspirate samples described by 30 features computed from "
                "digitised cell-nucleus images.",
    features=_breast_cancer_features(),
)

DISEASES = {d.key: d for d in (HEART, DIABETES, BREAST)}
