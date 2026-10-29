"""Plots for model results (comparison bars, confusion matrices, ROC curves, importances)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from src.config import FIGURES_DIR

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _save(fig, key, name, show=False):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{key}_{name}.png", dpi=130, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig


def plot_model_comparison(table, title, key=None, show=False):
    """Grouped bar chart. ``table`` is a DataFrame indexed by model name with metric columns."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(METRICS)
    x = np.arange(len(table))
    for i, metric in enumerate(METRICS):
        ax.bar(x + i * width, table[metric], width, label=metric)
    ax.set_xticks(x + width * (len(METRICS) - 1) / 2, table.index)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}: model comparison on the held-out test set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(ncol=len(METRICS), loc="lower center", fontsize=8)
    return _save(fig, key, "model_comparison", show) if key else fig


def plot_confusion_matrices(results, class_names, title, key=None, show=False):
    """One confusion matrix per model. ``results`` maps model name -> evaluate() output."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
    for ax, (name, res) in zip(axes.ravel(), results.items()):
        cm = np.array(res["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], class_names, rotation=20)
        ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"{title}: confusion matrices (test set)")
    return _save(fig, key, "confusion_matrices", show) if key else fig


def plot_roc_curves(pipes, X_test, y_test, title, key=None, show=False):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, pipe in pipes.items():
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_title(f"{title}: ROC curves (test set)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(fontsize=8)
    return _save(fig, key, "roc_curves", show) if key else fig


def plot_feature_importance(importance, title, key=None, show=False, top=12):
    """``importance`` is a list of {'feature', 'mean', 'std'} dicts sorted by importance."""
    items = importance[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1.8))
    ax.barh([i["feature"] for i in items], [i["mean"] for i in items],
            xerr=[i["std"] for i in items], color="#2E86AB")
    ax.set_title(f"{title}: permutation importance (drop in ROC-AUC)")
    ax.set_xlabel("Mean decrease in test ROC-AUC when the feature is shuffled")
    ax.set_ylabel("Feature")
    return _save(fig, key, "feature_importance", show) if key else fig
