# Disease Prediction from Medical Data

**Machine Learning-Based Health Risk Prediction System**

> **Medical disclaimer.** This project is for educational and machine learning demonstration purposes only. Predictions are statistical estimates from trained models and are **not medical diagnoses**. Consult a qualified healthcare professional for medical advice.

## Overview
An end-to-end machine learning project that estimates the risk of three diseases from structured patient data and serves the models through an interactive Streamlit dashboard.

`Medical dataset -> Cleaning -> EDA -> Feature engineering -> Training -> Model comparison -> Best model -> Prediction -> Web dashboard`

## Objective
Demonstrate the complete ML workflow (preprocessing, EDA, several classifiers, evaluation, tuning, explainability, saved models, deployment as a dashboard) on real public medical datasets, with no leakage and no hard-coded results.

## Diseases predicted
| Module | Dataset | Positive class (target = 1) |
|---|---|---|
| Heart Disease | UCI Heart Disease, Cleveland (303 rows, 13 features) | heart disease present |
| Diabetes | Pima Indians Diabetes (768 rows, 8 features) | tested positive for diabetes |
| Breast Cancer | UCI Breast Cancer Wisconsin Diagnostic (569 rows, 30 features) | malignant tumour |

## Dataset sources
* Heart disease: https://archive.ics.uci.edu/dataset/45/heart+disease
* Diabetes (Pima Indians, NIDDK): https://www.openml.org/search?type=data&id=37 (also on UCI/Kaggle)
* Breast cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (bundled with scikit-learn, works offline)

Heart and diabetes data are downloaded automatically the first time you train (`src/data_loader.py`) and cached in `data/`. **If you are offline or a download fails**, save the files manually:
* `data/heart.csv` with columns `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target` (target: 0 = no disease, 1 = disease; i.e. UCI `num > 0`; missing values empty or `?`).
* `data/diabetes.csv` with columns `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, target` (target: 1 = diabetic).

## Features
* Separate dataset and model per disease, all driven by one configuration file (`src/config.py`).
* Cleaning: duplicates removed, impossible values (e.g. glucose = 0) marked missing, out-of-range values flagged.
* Leakage-free preprocessing: imputation, scaling and one-hot encoding live in a scikit-learn `Pipeline` fitted on the training split only and saved with the model.
* EDA with computed interpretation notes (class balance, correlations, age comparison).
* Four classifiers, grid-search tuning with stratified 5-fold CV, class weighting for imbalance.
* Permutation importance for the selected model, plus Random Forest importances and Logistic Regression coefficients.
* Streamlit app with Home, Predict and Analytics pages.

## Algorithms
Logistic Regression (interpretable baseline), SVM with RBF kernel (on standardised features), Random Forest, XGBoost. If `xgboost` is not installed, training prints a clear warning and the app/notebook report that XGBoost was skipped - nothing is silently substituted.

## Project workflow
1. Load and inspect each dataset.
2. Clean (stateless steps) and split 80:20 with stratification (`random_state=42`).
3. Fit the preprocessing + model pipeline on the training split.
4. Train a baseline model, then tune with `GridSearchCV` (5-fold stratified CV on the training split).
5. Select the best model per disease by the mean of cross-validated **ROC-AUC, recall and F1** (training data only - the test set never influences selection).
6. Report test-set metrics, confusion matrices, ROC curves and importances; save the best pipeline to `models/<disease>_model.pkl`.

## Project structure
```
Disease-Prediction/
├── data/                  # heart.csv, diabetes.csv, breast_cancer.csv (created on first run)
├── notebooks/
│   └── disease_prediction_analysis.ipynb
├── models/                # <disease>_model.pkl (saved pipelines, created by training)
├── outputs/
│   ├── figures/           # EDA and result plots (PNG)
│   └── reports/           # <disease>_results.json and <disease>_comparison.csv
├── src/
│   ├── config.py          # paths, constants, feature definitions per disease
│   ├── data_loader.py     # download / load datasets
│   ├── preprocessing.py   # cleaning, split, preprocessing pipeline
│   ├── eda.py             # EDA plots + computed notes
│   ├── plots.py           # result plots
│   ├── train.py           # training, tuning, selection, saving
│   └── predict.py         # load model, validate input, predict
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
Python 3.10+ is recommended. From the project folder (Windows PowerShell shown; use `source .venv/bin/activate` on macOS/Linux):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the models
```powershell
python -m src.train                    # all diseases, with tuning (a few minutes)
python -m src.train --disease heart    # one disease
python -m src.train --no-tune          # quick run with default parameters
```
This creates the models in `models/`, the reports in `outputs/reports/` and the figures in `outputs/figures/`.

## How to run the notebook
Open `notebooks/disease_prediction_analysis.ipynb` in VS Code (select the `.venv` interpreter as the kernel) or Jupyter and run all cells from top to bottom. Hyper-parameter tuning is in section 17; set `TUNE = False` there for a fast run.

## How to run the Streamlit application
Train the models first (or run the notebook), then:
```powershell
streamlit run app.py
```
Pick a disease on the **Predict** page, enter values and press **Predict**. The result is shown as *Lower predicted risk* or *Higher predicted risk* with the model's predicted probability. Fields marked "Not available" use the typical training value (the median / most frequent value learnt during training).

## Model evaluation explanation
* **Accuracy** - share of correct predictions; misleading with imbalanced classes.
* **Precision** - of the patients flagged as positive, how many truly are.
* **Recall (sensitivity)** - of the truly positive patients, how many are caught; usually the most important metric in health screening because a missed case is costly.
* **F1-score** - harmonic mean of precision and recall.
* **ROC-AUC** - how well the model ranks positive cases above negative ones, independent of threshold.
* **Confusion matrix** - counts of true/false positives and negatives.

All metrics use predicted probabilities with a 0.5 threshold (the same rule the app uses). Because the test sets are small, differences of a few percentage points between models are within noise.

## Screenshots
_Add screenshots here after running the app, e.g._
* `docs/home.png` - Home page
* `docs/predict.png` - Prediction result
* `docs/analytics.png` - Analytics dashboard

## Limitations
* Small datasets from specific populations and time periods; results may not generalise.
* A single stratified split plus cross-validation - no external validation set.
* Probabilities are not guaranteed to be calibrated.
* Feature importance reflects association learnt by the model, not causation.
* The Pima data has many physiologically impossible zeros (treated as missing); the heart data uses a binarised target (any disease vs none).
* Saved `.pkl` files depend on the scikit-learn version used for training - retrain if you upgrade it.
* Not a medical device.

## Future improvements
* Optional SHAP explanations (`pip install shap`) for per-prediction contributions.
* Probability calibration and threshold tuning for a chosen sensitivity target.
* Additional datasets/diseases, external validation, repeated cross-validation.
* Deployment to Streamlit Community Cloud.
