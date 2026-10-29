"""Exploratory data analysis helpers.

Each plot uses the cleaned data. ``run_eda`` also returns short interpretation notes that are
*computed from the data* (class balance, strongest correlations, ...), never hard-coded.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, TARGET, DiseaseConfig

COLORS = ("#2E86AB", "#D1495B")  # negative class, positive class


def _finish(fig, cfg, name, save, show):
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{cfg.key}_eda_{name}.png", dpi=130, bbox_inches="tight")
    if not show:
        plt.close(fig)


def top_features(df: pd.DataFrame, cfg: DiseaseConfig, k: int = 6) -> pd.Series:
    """Features ranked by absolute correlation with the target (Pearson)."""
    corr = df[cfg.feature_names].corrwith(df[TARGET]).dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(k)


def plot_missing(raw: pd.DataFrame, cfg: DiseaseConfig, show: bool = True):
    """Bar chart of missing values (including impossible zeros) in the raw data."""
    counts = raw[cfg.feature_names].isna().sum()
    for col in cfg.zero_invalid:
        if col in raw.columns:
            counts[col] += int((raw[col] == 0).sum())
    counts = counts[counts > 0].sort_values()
    if counts.empty:
        print(f"{cfg.title}: no missing or impossible values found.")
        return None
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(counts) + 1.5))
    ax.barh(counts.index, counts.values, color=COLORS[1])
    ax.set_title(f"{cfg.title}: missing / impossible values per feature")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    if not show:
        plt.close(fig)
    return fig


def run_eda(cfg: DiseaseConfig, df: pd.DataFrame, show: bool = False, save: bool = True) -> list:
    """Create the EDA figures and return a list of computed interpretation notes."""
    neg, pos = cfg.class_names
    y = df[TARGET]
    notes = []

    # 1. class distribution -------------------------------------------------------------
    counts = y.value_counts().sort_index()
    n_neg, n_pos = int(counts.get(0, 0)), int(counts.get(1, 0))
    ratio = max(n_neg, n_pos) / max(1, min(n_neg, n_pos))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar([neg, pos], [n_neg, n_pos], color=COLORS)
    for bar, n in zip(bars, (n_neg, n_pos)):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n} ({100 * n / len(df):.1f}%)",
                ha="center", va="bottom")
    ax.set_title(f"{cfg.title}: class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of patients")
    _finish(fig, cfg, "class_distribution", save, show)
    level = "roughly balanced" if ratio < 1.5 else "moderately imbalanced" if ratio < 3 else "strongly imbalanced"
    notes.append(f"Class balance: {n_pos} of {len(df)} records ({100 * n_pos / len(df):.1f}%) are "
                 f"'{pos}' and {n_neg} are '{neg}' - the data is {level} (ratio {ratio:.2f}:1). "
                 "Recall, F1 and ROC-AUC are reported alongside accuracy for this reason.")

    # 2. age distribution ---------------------------------------------------------------
    age_col = next((c for c in cfg.feature_names if c.lower() == "age"), None)
    if age_col:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for label, color, grp in ((neg, COLORS[0], df[y == 0]), (pos, COLORS[1], df[y == 1])):
            ax.hist(grp[age_col].dropna(), bins=15, alpha=0.65, color=color, label=label)
        ax.set_title(f"{cfg.title}: age distribution by class")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Number of patients")
        ax.legend()
        _finish(fig, cfg, "age", save, show)
        med = df.groupby(TARGET)[age_col].median()
        notes.append(f"Median age is {med.get(1, float('nan')):.1f} years for '{pos}' versus "
                     f"{med.get(0, float('nan')):.1f} years for '{neg}'.")

    # 3. feature histograms and 4. box plots for the strongest features -------------------
    top = top_features(df, cfg, 6)
    n = len(top)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.4 * rows), squeeze=False)
    for ax, feat in zip(axes.ravel(), top.index):
        for label, color, code in ((neg, COLORS[0], 0), (pos, COLORS[1], 1)):
            ax.hist(df.loc[y == code, feat].dropna(), bins=20, alpha=0.65, color=color, label=label)
        ax.set_title(feat)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    axes.ravel()[0].legend()
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{cfg.title}: histograms of the features most associated with the target")
    _finish(fig, cfg, "histograms", save, show)

    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.4 * rows), squeeze=False)
    for ax, feat in zip(axes.ravel(), top.index):
        groups = [df.loc[y == 0, feat].dropna(), df.loc[y == 1, feat].dropna()]
        bp = ax.boxplot(groups, tick_labels=[neg, pos], patch_artist=True)
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(feat)
        ax.set_xlabel("Class")
        ax.set_ylabel("Value")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{cfg.title}: feature values by class (box plots)")
    _finish(fig, cfg, "boxplots", save, show)
    pretty = ", ".join(f"{f} (r={r:+.2f})" for f, r in top.head(4).items())
    notes.append(f"Features most correlated with the target: {pretty}. Correlation describes "
                 "association in this dataset, not cause and effect; categorical codes (e.g. chest "
                 "pain type) are treated as numbers here, so their correlation is only approximate.")

    # 5. correlation heatmap --------------------------------------------------------------
    cols = list(top_features(df, cfg, 12).index) + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3, 0.7 * len(cols) + 2.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    if len(cols) <= 14:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(f"{cfg.title}: correlation heatmap (top features + target)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    _finish(fig, cfg, "correlation", save, show)
    feats = corr.drop(index=TARGET, columns=TARGET)
    upper = feats.where(np.triu(np.ones(feats.shape, dtype=bool), k=1)).stack()
    if len(upper):
        a, b = upper.abs().idxmax()
        notes.append(f"Strongest correlation between two shown features: {a} and {b} "
                     f"(r={feats.loc[a, b]:+.2f}). Highly correlated features carry overlapping "
                     "information, which matters when reading coefficients or importances.")
    return notes
