"""Figure generation for the experiments (static PNGs, matplotlib).

Palette follows the validated light-mode reference palette (categorical slots
in fixed order, blue sequential ramp for magnitude heatmaps).  All figures are
300 dpi with titles, axis labels and legends.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe in scripts and CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- palette (validated light-mode reference) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in FIXED order, assigned once per model (never by rank).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MODEL_COLORS = {
    "knn_1": CATEGORICAL[0],
    "logreg": CATEGORICAL[1],
    "lda": CATEGORICAL[2],
    "svm_linear": CATEGORICAL[3],
    "svm_rbf": CATEGORICAL[4],
}

# Blue sequential ramp (magnitude).
_SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _SEQ_STEPS)

TRIAL_LINESTYLE = {1: "-", 2: "--"}
TRIAL_MARKER = {1: "o", 2: "s"}


def _style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: light gridlines, muted ticks, clean spines."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _new_figure(figsize=(7.5, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    return fig, ax


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_model_comparison(trials_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart of mean test accuracy with the two per-trial points overlaid.

    The two trials are shown as points (not error bars): they are two fixed
    writer-separated splits, not two independent samples of a distribution.
    """
    fig, ax = _new_figure(figsize=(8.0, 4.5))
    labels = summary_df["model_name"].tolist()
    configs = summary_df["config_id"].tolist()
    means = summary_df["mean_test_accuracy"].to_numpy()
    x = np.arange(len(labels))
    colors = [MODEL_COLORS[c] for c in configs]

    ax.bar(x, means, width=0.62, color=colors, zorder=3, label=None)
    for i, m in enumerate(means):
        ax.text(
            x[i], m + 0.0025, f"{m * 100:.2f}", ha="center", va="bottom",
            fontsize=9, color=INK_SECONDARY,
        )
    for trial, marker in TRIAL_MARKER.items():
        accs = trials_df.loc[trials_df["trial"] == trial]
        accs = accs.set_index("config_id").loc[configs, "test_accuracy"].to_numpy()
        ax.scatter(
            x, accs, s=42, marker=marker, facecolor=INK, edgecolor=SURFACE,
            linewidth=1.5, zorder=4, label=f"Trial {trial}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", color=INK)
    ax.set_ylabel("Test accuracy", color=INK)
    ax.set_ylim(max(0.80, float(means.min()) - 0.03), min(1.0, float(means.max()) + 0.015))
    ax.set_title("Model comparison: test accuracy over the two course trials", color=INK, fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    return _save(fig, out_path)


def plot_pca_accuracy(pca_df: pd.DataFrame, out_path: Path) -> Path:
    """Line chart: test accuracy vs. PCA dimensions for LR and RBF SVM.

    Model = hue, trial = line style.  The x-axis is categorical (the four
    discrete dimension settings), spaced evenly.
    """
    fig, ax = _new_figure(figsize=(7.5, 4.5))
    _plot_pca_lines(ax, pca_df, "test_accuracy")
    ax.set_ylabel("Test accuracy", color=INK)
    ax.set_title("PCA dimensionality vs. test accuracy (LR and RBF SVM)", color=INK, fontsize=11)
    return _save(fig, out_path)


def plot_pca_time(pca_df: pd.DataFrame, out_path: Path) -> Path:
    """Line chart: training (CV search) wall time vs. PCA dimensions.

    Timing scope: 3-fold GridSearchCV on the training set including the PCA
    fit inside the pipeline, plus the final refit of the best configuration.
    """
    fig, ax = _new_figure(figsize=(7.5, 4.5))
    _plot_pca_lines(ax, pca_df, "fit_time_sec")
    ax.set_ylabel("Training time (s)", color=INK)
    ax.set_title(
        "PCA dimensionality vs. CV training time (3-fold grid search incl. PCA fit)",
        color=INK, fontsize=11,
    )
    return _save(fig, out_path)


def _plot_pca_lines(ax: plt.Axes, pca_df: pd.DataFrame, value_col: str) -> None:
    # derive the x-axis order from the data: no-PCA first, then ascending dims
    labels = list(pca_df["n_components"].astype(str).drop_duplicates())
    none_labels = sorted(l for l in labels if "None" in l)
    dim_labels = sorted((l for l in labels if "None" not in l), key=lambda v: float(v))
    order = none_labels + dim_labels
    x = np.arange(len(order))
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("None ", "None\n") for l in order], color=INK)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for config_id, name in (("logreg", "LR"), ("svm_rbf", "RBF SVM")):
        for trial in (1, 2):
            sub = pca_df[(pca_df["config_id"] == config_id) & (pca_df["trial"] == trial)].copy()
            sub["n_components"] = sub["n_components"].astype(str)
            sub = sub.set_index("n_components").loc[order, value_col].to_numpy(dtype=float)
            ax.plot(
                x, sub,
                color=MODEL_COLORS[config_id],
                linestyle=TRIAL_LINESTYLE[trial],
                marker=TRIAL_MARKER[trial],
                markersize=6,
                linewidth=2,
                label=f"{name} — Trial {trial}",
                zorder=3,
            )
    ax.legend(loc="best", frameon=False, fontsize=9)


def _cell_text_color(value: float, vmin: float, vmax: float) -> str:
    """White text on dark cells, dark ink on light cells."""
    if vmax <= vmin:
        return INK
    frac = (value - vmin) / (vmax - vmin)
    return "#ffffff" if frac > 0.6 else INK


def plot_rbf_cv_heatmap(cv_df: pd.DataFrame, out_path: Path) -> Path:
    """3x3 CV-accuracy heatmaps (C x gamma) of RBF SVM, one per trial.

    These are training-set CV scores only; test accuracy is never used for
    parameter visualization.
    """
    sub = cv_df[cv_df["config_id"] == "svm_rbf"].copy()
    # params may be a dict in memory or its str() form after a CSV round-trip.
    import ast

    def parse(p) -> dict:
        d = p if isinstance(p, dict) else ast.literal_eval(p)
        return {"C": float(d["C"]), "gamma": float(d["gamma"])}

    sub[["C", "gamma"]] = sub["params"].apply(lambda p: pd.Series(parse(p)))
    cs = sorted(sub["C"].unique())
    gammas = sorted(sub["gamma"].unique())
    trials = sorted(sub["trial"].unique())
    vmin = float(sub["mean_cv_accuracy"].min())
    vmax = float(sub["mean_cv_accuracy"].max())

    fig, axes = plt.subplots(1, len(trials), figsize=(4.6 * len(trials), 3.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    if len(trials) == 1:
        axes = [axes]
    for ax, trial in zip(axes, trials):
        grid = np.full((len(gammas), len(cs)), np.nan)
        for (_, row) in sub[sub["trial"] == trial].iterrows():
            i = gammas.index(row["gamma"])
            j = cs.index(row["C"])
            grid[i, j] = row["mean_cv_accuracy"]
        im = ax.imshow(grid, cmap=SEQ_BLUE, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([f"{c:g}" for c in cs], color=INK)
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([f"{g:g}" for g in gammas], color=INK)
        ax.set_xlabel("C", color=INK)
        ax.set_ylabel("gamma", color=INK)
        ax.set_title(f"RBF SVM CV accuracy — Trial {trial}", color=INK, fontsize=10)
        for i in range(len(gammas)):
            for j in range(len(cs)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v * 100:.2f}",
                        ha="center", va="center", fontsize=8,
                        color=_cell_text_color(v, vmin, vmax),
                    )
        # mark the best cell
        bi, bj = np.unravel_index(np.nanargmax(grid), grid.shape)
        ax.add_patch(
            plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                          edgecolor=INK, linewidth=1.8)
        )
        ax.tick_params(colors=INK_MUTED, labelsize=9)
    fig.suptitle(
        "RBF SVM: training-set 3-fold CV accuracy over the C x gamma grid (boxed = best)",
        color=INK, fontsize=11, y=1.02,
    )
    return _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, trial: int, out_path: Path) -> Path:
    """10x10 confusion matrix for one trial (rows = true, columns = predicted)."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    vmax = max(cm.max(), 1)
    n = cm.shape[0]
    im = ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), color=INK)
    ax.set_yticklabels(range(n), color=INK)
    ax.set_xlabel("Predicted label", color=INK)
    ax.set_ylabel("True label", color=INK)
    ax.set_title(f"RBF SVM confusion matrix — Trial {trial}", color=INK, fontsize=11)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(
                j, i, f"{v}", ha="center", va="center", fontsize=8,
                color=_cell_text_color(v, 0, vmax),
            )
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    return _save(fig, out_path)


def plot_error_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trial: int,
    out_path: Path,
    max_examples: int = 12,
) -> Path | None:
    """Plot up to ``max_examples`` misclassified test images for one trial.

    Returns None (and writes no file) when there are no misclassifications.
    Images are reshaped with MATLAB column-major order ('F').
    """
    wrong = np.nonzero(y_pred != y_true)[0]
    if wrong.size == 0:
        return None
    wrong = wrong[:max_examples]
    n = wrong.size
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.4 * nrows), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for k, idx in enumerate(wrong):
        ax = axes[k]
        img = X[idx].reshape(28, 28, order="F")
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(f"true {y_true[idx]}  pred {y_pred[idx]}", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, axes.size):
        axes[k].axis("off")
    fig.suptitle(
        f"RBF SVM misclassified examples — Trial {trial}", color=INK, fontsize=11
    )
    return _save(fig, out_path)
