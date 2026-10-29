"""Streamlit dashboard: Disease Prediction from Medical Data.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import DISCLAIMER, DISEASES, FIGURES_DIR
from src.predict import load_bundle, predict_risk, validate_inputs

st.set_page_config(page_title="Disease Prediction from Medical Data", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.2rem 1.5rem; border-radius: 14px;
           background: linear-gradient(135deg, #1f4e79 0%, #2E86AB 100%); color: white;}
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; font-size: 1.05rem; opacity: .92;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner=False)
def get_bundle(key: str):
    return load_bundle(key)


@st.cache_data(show_spinner=False)
def get_results(key: str, mtime: float):
    """Read the results JSON (mtime in the cache key refreshes it after re-training)."""
    with open(DISEASES[key].results_path, encoding="utf-8") as fh:
        return json.load(fh)


def results_or_none(key: str):
    path = DISEASES[key].results_path
    return get_results(key, path.stat().st_mtime) if path.exists() else None


def hero(subtitle: str):
    st.markdown(f"<div class='hero'><h1>Disease Prediction from Medical Data</h1>"
                f"<p>{subtitle}</p></div>", unsafe_allow_html=True)
    st.write("")


# --------------------------------------------------------------------------- Home
def page_home():
    hero("Machine Learning-Based Health Risk Prediction System")
    st.write(
        "This project trains classic machine learning classifiers (Logistic Regression, SVM, "
        "Random Forest and XGBoost) on public medical datasets to estimate the risk of "
        "**heart disease**, **diabetes** and **breast cancer** from structured patient data. "
        "Use **Predict** to try a model and **Analytics** to see how each model was built and evaluated."
    )
    st.warning(DISCLAIMER)
    cols = st.columns(len(DISEASES))
    for col, (key, cfg) in zip(cols, DISEASES.items()):
        res = results_or_none(key)
        with col:
            st.subheader(cfg.title)
            st.caption(cfg.description)
            if res:
                m = res["tuned"][res["best_model"]]["metrics"]
                st.metric("Selected model", res["best_model"])
                st.write(f"Test recall **{m['recall']:.2f}** · ROC-AUC **{m['roc_auc']:.2f}**")
            else:
                st.info(f"Not trained yet. Run `python -m src.train --disease {key}`.")


# --------------------------------------------------------------------------- Predict
def feature_widget(feat, prefix: str):
    """Render the right Streamlit control for one feature and return its value (or None)."""
    key = f"{prefix}_{feat.name}"
    label = f"{feat.label} ({feat.unit})" if feat.unit else feat.label
    help_text = feat.help or None
    if feat.kind == "choice":
        codes = list(feat.options)
        if feat.allow_missing:
            codes.append(None)
        index = codes.index(feat.default) if feat.default in codes else 0
        return st.selectbox(
            label, codes, index=index, key=key, help=help_text,
            format_func=lambda c: "Not available (typical training value is used)" if c is None
            else feat.options[c])
    if feat.kind == "int":
        value = st.number_input(label, min_value=int(feat.min), max_value=int(feat.max),
                                value=int(feat.default), step=int(feat.step or 1), key=key,
                                help=help_text)
    else:
        value = st.number_input(label, min_value=float(feat.min), max_value=float(feat.max),
                                value=float(feat.default), step=float(feat.step), format=feat.fmt,
                                key=key, help=help_text)
    if feat.allow_missing and st.checkbox("Not available", key=f"{key}_na",
                                          help="Tick if you do not have this value; the model "
                                               "substitutes the typical value seen in training."):
        return None
    return value


def page_predict():
    hero("Estimate health risk from medical measurements")
    key = st.selectbox("Choose a disease model", list(DISEASES),
                       format_func=lambda k: DISEASES[k].title)
    cfg = DISEASES[key]
    try:
        bundle = get_bundle(key)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.caption("Starting values are only placeholders - replace them with the measurements you "
               "want to test.")
    values = {}
    with st.form(f"form_{key}"):
        groups = list(dict.fromkeys(f.group for f in cfg.features))
        for group in groups:
            members = [f for f in cfg.features if f.group == group]
            box = st.expander(group, expanded=(group == groups[0])) if group else st.container()
            with box:
                columns = st.columns(2)
                for i, feat in enumerate(members):
                    with columns[i % 2]:
                        values[feat.name] = feature_widget(feat, key)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        errors = validate_inputs(key, values)
        if errors:
            for message in errors:
                st.error(message)
            return
        result = predict_risk(key, values, bundle)
        st.divider()
        st.subheader("Result")
        if result["predicted_class"]:
            st.warning(f"**{result['label']}** for {cfg.title.lower()}")
        else:
            st.info(f"**{result['label']}** for {cfg.title.lower()}")
        left, right = st.columns(2)
        left.metric(f"Predicted probability of '{cfg.class_names[1]}'",
                    f"{100 * result['probability']:.1f}%")
        right.metric("Model used", result["model_name"])
        st.progress(min(max(result["probability"], 0.0), 1.0))
        st.caption(f"Risk is labelled 'higher' when the probability is at least "
                   f"{result['threshold']:.0%}. This is a statistical estimate, not a diagnosis.")
        for note in result["warnings"]:
            st.warning(note)
        m = bundle["test_metrics"]
        st.caption(f"Held-out test performance of this model: recall {m['recall']:.2f}, "
                   f"precision {m['precision']:.2f}, ROC-AUC {m['roc_auc']:.2f}.")
    st.info(DISCLAIMER)


# --------------------------------------------------------------------------- Analytics
def show_image(name: str):
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path))


def page_analytics():
    hero("Model performance, explainability and dataset information")
    key = st.selectbox("Disease", list(DISEASES), format_func=lambda k: DISEASES[k].title,
                       key="analytics_disease")
    cfg = DISEASES[key]
    res = results_or_none(key)
    if res is None:
        st.error(f"No results found. Train first with: python -m src.train --disease {key}")
        return
    for model, reason in res.get("skipped_models", {}).items():
        st.warning(f"{model} was skipped: {reason}")

    tabs = st.tabs(["Model comparison", "Confusion matrix", "Metrics", "Important features",
                    "Dataset information"])
    tuned = res["tuned"]
    table = pd.DataFrame({n: r["metrics"] for n, r in tuned.items()}).T

    with tabs[0]:
        st.write(f"Selected model: **{res['best_model']}** ({res['selection_rule']}).")
        st.dataframe(table.style.format("{:.3f}"))
        st.bar_chart(table)
        with st.expander("Baseline (default hyper-parameters) results"):
            base = pd.DataFrame({n: r["metrics"] for n, r in res["baseline"].items()}).T
            st.dataframe(base.style.format("{:.3f}"))
        with st.expander("Selected hyper-parameters and cross-validation scores"):
            st.json({"best_params": res["best_params"], "cross_validation": res["selection"]})
        show_image(f"{key}_roc_curves.png")

    with tabs[1]:
        model = st.selectbox("Model", list(tuned), index=list(tuned).index(res["best_model"]),
                             key="cm_model")
        neg, pos = res["class_names"]
        cm = pd.DataFrame(tuned[model]["confusion_matrix"],
                          index=[f"Actual: {neg}", f"Actual: {pos}"],
                          columns=[f"Predicted: {neg}", f"Predicted: {pos}"])
        st.dataframe(cm)
        st.text(tuned[model]["report"])
        show_image(f"{key}_confusion_matrices.png")

    with tabs[2]:
        model = st.selectbox("Model", list(tuned), index=list(tuned).index(res["best_model"]),
                             key="metric_model")
        cols = st.columns(5)
        for col, (name, label) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                             ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]):
            col.metric(label, f"{tuned[model]['metrics'][name]:.3f}")
        st.caption("Recall (sensitivity) is the share of real positive cases the model catches; in a "
                   "health setting missed cases are usually costlier than false alarms.")

    with tabs[3]:
        perm = pd.DataFrame(res["permutation_importance"][:15]).set_index("feature")
        st.write(f"Permutation importance of the selected model ({res['best_model']}): how much "
                 "test ROC-AUC drops when a feature is shuffled.")
        st.bar_chart(perm["mean"])
        if "random_forest_importance" in res:
            with st.expander("Random Forest built-in importance"):
                st.bar_chart(pd.DataFrame(res["random_forest_importance"]).set_index("feature"))
        if "logistic_regression_coefficients" in res:
            with st.expander("Logistic Regression coefficients (standardised features)"):
                st.bar_chart(pd.DataFrame(res["logistic_regression_coefficients"]).set_index("feature"))
        st.caption("Importance shows which inputs the model relies on. It does not mean those "
                   "factors cause the disease.")

    with tabs[4]:
        info = res["raw_info"]
        log = res["cleaning_log"]
        st.write(f"**Source:** [{res['source']}]({res['source_url']})")
        st.write(f"**Target:** {res['positive_meaning']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows x columns", f"{info['shape'][0]} x {info['shape'][1]}")
        c2.metric("Rows after cleaning", log["rows_clean"])
        c3.metric("Train / test rows", f"{res['split']['train']} / {res['split']['test']}")
        st.write(f"Class counts: {res['class_names'][0]} = {res['class_counts']['negative']}, "
                 f"{res['class_names'][1]} = {res['class_counts']['positive']}")
        st.write("**Cleaning log**")
        st.json(log)
        st.write("**Features**")
        st.dataframe(pd.DataFrame([{"feature": f.name, "description": f.label, "unit": f.unit or "-"}
                                   for f in cfg.features]))
        st.write("**EDA**")
        for note in res["eda_notes"]:
            st.write(f"- {note}")
        for name in ("class_distribution", "age", "histograms", "boxplots", "correlation"):
            show_image(f"{key}_eda_{name}.png")


# --------------------------------------------------------------------------- router
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Analytics"])
st.sidebar.caption("Educational demonstration only - not medical advice.")
{"Home": page_home, "Predict": page_predict, "Analytics": page_analytics}[page]()
