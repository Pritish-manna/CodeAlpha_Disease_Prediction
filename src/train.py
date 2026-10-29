"""Train, tune, compare and save the models for every disease.

Run from the project root:

    python -m src.train                         # all three diseases (with tuning)
    python -m src.train --disease heart         # one disease
    python -m src.train --no-tune               # quick run with default hyper-parameters

Model selection uses cross-validation on the TRAINING split only. The test split is touched
once, for reporting, so it never influences which model is chosen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import data_loader, eda, plots, preprocessing
from src.config import (CV_FOLDS, DECISION_THRESHOLD, DISEASES, MODELS_DIR, RANDOM_STATE,
                        REPORTS_DIR, TARGET, DiseaseConfig)

try:
    from xgboost import XGBClassifier
    XGBOOST_ERROR = None
except ImportError as exc:  # reported clearly, never silently replaced by another model
    XGBClassifier = None
    XGBOOST_ERROR = (f"XGBoost could not be imported ({exc}). Install it with "
                     "`pip install xgboost` and re-run training to include this model.")

SCORING = {"roc_auc": "roc_auc", "recall": "recall", "f1": "f1"}
N_JOBS = -1


def make_models(scale_pos_weight: float) -> dict:
    """Return {name: (estimator, small hyper-parameter grid)} for the available algorithms."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1, 10]}),
        "SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
            {"n_estimators": [200, 400], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]}),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight,
                          random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
            {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
    return models


def build_pipeline(cfg: DiseaseConfig, estimator) -> Pipeline:
    """Preprocessing + classifier in one object, so both are saved and applied together."""
    return Pipeline([("prep", preprocessing.build_preprocessor(cfg)), ("clf", clone(estimator))])


def evaluate(pipe: Pipeline, X, y, class_names=("Negative", "Positive")) -> dict:
    """Metrics from predicted probabilities, using the same threshold as the app."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y, pred, target_names=list(class_names),
                                        labels=[0, 1], zero_division=0),
    }


def _tune(pipe, grid, X, y, tune: bool):
    """Hyper-parameter search (or plain CV) on the training split. Returns (pipe, params, cv)."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if tune:
        search = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()}, scoring=SCORING,
                              refit="roc_auc", cv=cv, n_jobs=N_JOBS,
                              error_score="raise")  # surface failures instead of silent NaN scores
        search.fit(X, y)
        i = search.best_index_
        scores = {m: float(search.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        return search.best_estimator_, params, scores
    res = cross_validate(pipe, X, y, scoring=SCORING, cv=cv, n_jobs=N_JOBS, error_score="raise")
    scores = {m: float(np.mean(res[f"test_{m}"])) for m in SCORING}
    return clone(pipe).fit(X, y), {}, scores


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def train_disease(key: str, tune: bool = True, make_figures: bool = True) -> dict:
    """Full workflow for one disease. Saves the model bundle, results JSON and figures."""
    cfg = DISEASES[key]
    t0 = time.time()
    raw = data_loader.load_dataset(key)
    raw_info = preprocessing.inspect_dataset(raw)
    df, clean_log = preprocessing.clean_dataset(cfg, raw)
    X_train, X_test, y_train, y_test = preprocessing.split_data(cfg, df)
    spw = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    models = make_models(spw)
    print(f"\n=== {cfg.title}: {len(df)} clean rows, train {len(X_train)} / test {len(X_test)} ===")

    baseline, tuned, selection, params, pipes = {}, {}, {}, {}, {}
    for name, (est, grid) in models.items():
        pipe = build_pipeline(cfg, est)
        baseline[name] = evaluate(clone(pipe).fit(X_train, y_train), X_test, y_test, cfg.class_names)
        fitted, params[name], cv_scores = _tune(pipe, grid, X_train, y_train, tune)
        pipes[name] = fitted
        tuned[name] = evaluate(fitted, X_test, y_test, cfg.class_names)
        # selection score: mean of CV ROC-AUC, recall and F1 (recall matters in a health setting)
        score = float(np.mean([cv_scores["roc_auc"], cv_scores["recall"], cv_scores["f1"]]))
        selection[name] = {**{f"cv_{m}": v for m, v in cv_scores.items()}, "selection_score": score}
        print(f"  {name:<20} cv-score {score:.3f} | test recall {tuned[name]['metrics']['recall']:.3f}"
              f" | test ROC-AUC {tuned[name]['metrics']['roc_auc']:.3f}")

    best_name = max(selection, key=lambda n: selection[n]["selection_score"])
    best_pipe = pipes[best_name]
    print(f"  -> selected: {best_name}")

    # ---- explainability (association, not causation) ------------------------------------
    perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=1)
    order = np.argsort(perm.importances_mean)[::-1]
    perm_list = [{"feature": cfg.feature_names[i], "mean": float(perm.importances_mean[i]),
                  "std": float(perm.importances_std[i])} for i in order]
    names_out = list(pipes[next(iter(pipes))].named_steps["prep"].get_feature_names_out())
    extra = {}
    if "Random Forest" in pipes:
        imp = pipes["Random Forest"].named_steps["clf"].feature_importances_
        extra["random_forest_importance"] = [{"feature": names_out[i], "importance": float(imp[i])}
                                             for i in np.argsort(imp)[::-1][:15]]
    if "Logistic Regression" in pipes:
        coef = pipes["Logistic Regression"].named_steps["clf"].coef_[0]
        extra["logistic_regression_coefficients"] = [
            {"feature": names_out[i], "coefficient": float(coef[i])}
            for i in np.argsort(np.abs(coef))[::-1][:15]]

    # ---- EDA figures + data-driven notes -------------------------------------------------
    notes = eda.run_eda(cfg, df, show=False, save=make_figures)

    # ---- save the model bundle (pipeline = preprocessing + classifier) ---------------------
    ranges = {c: [float(X_train[c].min()), float(X_train[c].max())] for c in cfg.feature_names}
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "features": cfg.feature_names,
        "threshold": DECISION_THRESHOLD,
        "training_ranges": ranges,
        "test_metrics": tuned[best_name]["metrics"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.model_path)

    results = {
        "disease": key, "title": cfg.title, "class_names": list(cfg.class_names),
        "source": cfg.source, "source_url": cfg.source_url, "positive_meaning": cfg.positive_meaning,
        "raw_info": raw_info, "cleaning_log": clean_log, "eda_notes": notes,
        "class_counts": {"negative": int((df[TARGET] == 0).sum()), "positive": int((df[TARGET] == 1).sum())},
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "tuned": tuned, "baseline": baseline, "best_params": params, "selection": selection,
        "best_model": best_name, "selection_rule": "highest mean of cross-validated ROC-AUC, recall "
        "and F1 on the training split (test set not used for selection)",
        "permutation_importance": perm_list, **extra,
        "skipped_models": {"XGBoost": XGBOOST_ERROR} if XGBClassifier is None else {},
        "tuning_enabled": tune, "threshold": DECISION_THRESHOLD,
        "trained_at": bundle["trained_at"], "seconds": round(time.time() - t0, 1),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({n: r["metrics"] for n, r in tuned.items()}).T
    table.round(4).to_csv(REPORTS_DIR / f"{key}_comparison.csv", index_label="model")
    with open(cfg.results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_jsonable)

    if make_figures:
        plots.plot_model_comparison(table, cfg.title, key)
        plots.plot_confusion_matrices(tuned, cfg.class_names, cfg.title, key)
        plots.plot_roc_curves(pipes, X_test, y_test, cfg.title, key)
        plots.plot_feature_importance(perm_list, f"{cfg.title} ({best_name})", key)
    return results


def main() -> int:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")  # no pop-up windows when training from the command line
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Train the disease prediction models.")
    parser.add_argument("--disease", choices=["all", *DISEASES], default="all")
    parser.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    parser.add_argument("--no-figures", action="store_true", help="skip saving figures")
    args = parser.parse_args()

    if XGBOOST_ERROR:
        print(f"WARNING: {XGBOOST_ERROR}\n")
    keys = list(DISEASES) if args.disease == "all" else [args.disease]
    failures = 0
    for key in keys:
        try:
            train_disease(key, tune=not args.no_tune, make_figures=not args.no_figures)
        except Exception as exc:  # keep going so one missing dataset does not block the others
            failures += 1
            print(f"\nERROR while training '{key}': {exc}")
    print("\nDone." if not failures else f"\nFinished with {failures} error(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
