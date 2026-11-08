"""CLI entry point: run the full experiment pipeline from raw data to outputs.

Usage:
    python -m digit_project.run --data-dir data/digits4000_txt --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import dataset_summary, load_data
from .experiments import (
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
)
from .plots import (
    plot_confusion_matrix,
    plot_error_examples,
    plot_model_comparison,
    plot_pca_accuracy,
    plot_pca_time,
    plot_rbf_cv_heatmap,
)

log = logging.getLogger("digit_project")


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="digit_project.run",
        description="CS5487 project: classical classifiers on digits4000 (two trials).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/digits4000_txt",
        help="Directory with the four digits4000 TXT files (or its parent).",
    )
    parser.add_argument("--output-dir", default="outputs", help="Where tables/figures are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (CV splits, models).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for GridSearchCV (default 1 to keep the machine responsive).",
    )
    return parser.parse_args(argv)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(df))
    return path


def _write_metadata(
    args: argparse.Namespace,
    source_files: dict[str, str],
    wall_sec: float,
    n_tables: int,
    n_figures: int,
    output_dir: Path,
) -> Path:
    """Write reproducibility metadata (versions, seed, file fingerprints)."""
    import matplotlib
    import sklearn

    metadata = {
        "project": "CS5487 handwritten digit classification (comparative study)",
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "data_dir": str(Path(args.data_dir).resolve()),
        "data_file_sha256": source_files,
        "python_version": sys.version.split()[0],
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "wall_time_sec": round(wall_sec, 1),
        "experiment_config": {
            "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=seed) on training set only",
            "scaling": "X.astype(float32) / 255 (deterministic, applied before splitting)",
            "experiment_a_models": [
                "knn_1 (Euclidean 1-NN, fixed)",
                "logreg (L2, C in {0.1, 1, 10})",
                "lda (lsqr, shrinkage=auto, fixed)",
                "svm_linear (C in {0.1, 1, 10})",
                "svm_rbf (C in {0.1, 1, 10} x gamma in {0.001, 0.01, 0.1})",
            ],
            "experiment_b": {
                "models": ["logreg", "svm_rbf"],
                "n_components": [None, 50, 100, 200],
                "pca": "PCA(svd_solver='auto', random_state=seed) inside sklearn Pipeline",
            },
            "experiment_c_target": "svm_rbf (fixed before seeing results)",
        },
        "outputs": {"tables": n_tables, "figures": n_figures},
    }
    path = output_dir / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Wrote %s", path)
    return path


def _write_experiment_notes(
    exp_a, exp_b, exp_c, output_dir: Path, baseline_expected: dict[int, float]
) -> Path:
    """Auto-generated objective notes based on the real computed numbers."""
    lines = [
        "# Experiment notes (auto-generated from real results)",
        "",
        "Observations below are computed from the run outputs only.",
        "",
        "## 1-NN baseline check",
        "",
        "| Trial | Reproduced accuracy | Instructor reference |",
        "|---|---:|---:|",
    ]
    for trial in sorted(baseline_expected):
        row = exp_a.trials_df[
            (exp_a.trials_df["trial"] == trial) & (exp_a.trials_df["config_id"] == "knn_1")
        ]
        acc = float(row["test_accuracy"].iloc[0])
        lines.append(f"| {trial} | {acc * 100:.2f}% | {baseline_expected[trial] * 100:.2f}% |")

    lines += [
        "",
        "## Model comparison (two-trial mean)",
        "",
        "| Model | Mean test acc. | Trial 1 | Trial 2 | Mean Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in exp_a.summary_df.iterrows():
        lines.append(
            f"| {r['model_name']} | {r['mean_test_accuracy'] * 100:.2f}% | "
            f"{r['trial1_accuracy'] * 100:.2f}% | {r['trial2_accuracy'] * 100:.2f}% | "
            f"{r['mean_macro_f1']:.4f} |"
        )

    lines += ["", "## PCA dimensionality (per model, two-trial mean)", ""]
    lines.append("| Model | n_components | Mean test acc. | Mean cum. explained variance |")
    lines.append("|---|---:|---:|---:|")
    for (config_id, name) in (("logreg", "LR"), ("svm_rbf", "RBF SVM")):
        for nc in ("None (784)", 50, 100, 200):
            sub = exp_b.pca_df[
                (exp_b.pca_df["config_id"] == config_id)
                & (exp_b.pca_df["n_components"] == nc)
            ]
            acc = float(sub["test_accuracy"].mean())
            evr = sub["cum_explained_variance_ratio"].mean()
            evr_txt = f"{evr:.3f}" if pd.notna(evr) else "—"
            lines.append(f"| {name} | {nc} | {acc * 100:.2f}% | {evr_txt} |")

    lines += ["", "## Most frequent RBF-SVM confusion pairs (per trial)", ""]
    for trial in (1, 2):
        lines.append(f"Trial {trial}:")
        sub = exp_c.error_pairs[exp_c.error_pairs["trial"] == trial].head(5)
        if sub.empty:
            lines.append("- no errors")
        for _, r in sub.iterrows():
            lines.append(
                f"- true {int(r['true_label'])} -> predicted {int(r['pred_label'])} "
                f"({int(r['count'])} test samples)"
            )
    lines += [
        "",
        "Note: statements are limited to what the computed numbers show; no causal",
        "claims beyond the data.",
        "",
    ]
    path = output_dir / "experiment_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)
    return path


def _print_final_summary(exp_a, exp_b) -> None:
    log.info("=== Final summary ===")
    for trial in (1, 2):
        sub = exp_a.trials_df[exp_a.trials_df["trial"] == trial].sort_values("test_accuracy")
        log.info("Trial %d test accuracy:", trial)
        for _, r in sub.iterrows():
            log.info("  %-28s %.2f%%", r["model_name"], r["test_accuracy"] * 100)
    log.info("PCA (two-trial mean test accuracy):")
    for (config_id, name) in (("logreg", "LR"), ("svm_rbf", "RBF SVM")):
        for nc in ("None (784)", 50, 100, 200):
            sub = exp_b.pca_df[
                (exp_b.pca_df["config_id"] == config_id) & (exp_b.pca_df["n_components"] == nc)
            ]
            log.info("  %-8s %-11s %.2f%%", name, nc, float(sub["test_accuracy"].mean()) * 100)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    t_start = time.perf_counter()

    # --- data loading and validation ---
    log.info("Loading data from %s", args.data_dir)
    ds = load_data(args.data_dir)
    log.info("Data loaded and validated:\n%s", dataset_summary(ds))

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    # --- Experiment A: five classifiers x two trials ---
    log.info("Running Experiment A (classifier benchmark, seed=%d, n_jobs=%d)", args.seed, args.n_jobs)
    exp_a = run_experiment_a(ds, seed=args.seed, n_jobs=args.n_jobs)

    # --- Experiment B: PCA dimensionality ---
    log.info("Running Experiment B (PCA dimensionality for LR and RBF SVM)")
    exp_b = run_experiment_b(ds, seed=args.seed, n_jobs=args.n_jobs)

    # --- Experiment C: RBF SVM error analysis ---
    log.info("Running Experiment C (RBF SVM error analysis)")
    rbf_preds = {t: exp_a.predictions[(t, "svm_rbf")] for t in (1, 2)}
    exp_c = run_experiment_c(ds, rbf_preds)

    # --- tables ---
    n_tables = 0
    for df, name in (
        (exp_a.trials_df, "model_comparison_trials.csv"),
        (exp_a.summary_df, "model_comparison_summary.csv"),
        (exp_b.pca_df, "pca_comparison_trials.csv"),
        (exp_a.cv_df, "cv_parameter_results.csv"),
        (exp_c.error_pairs, "error_summary.csv"),
    ):
        _write_csv(df, tables_dir / name)
        n_tables += 1

    # --- figures ---
    n_figures = 0
    for fn, out_name in (
        (
            lambda: plot_model_comparison(exp_a.trials_df, exp_a.summary_df,
                                          figures_dir / "model_comparison.png"),
            "model_comparison.png",
        ),
        (
            lambda: plot_pca_accuracy(exp_b.pca_df, figures_dir / "pca_accuracy.png"),
            "pca_accuracy.png",
        ),
        (
            lambda: plot_pca_time(exp_b.pca_df, figures_dir / "pca_time.png"),
            "pca_time.png",
        ),
        (
            lambda: plot_rbf_cv_heatmap(exp_a.cv_df, figures_dir / "rbf_cv_heatmap.png"),
            "rbf_cv_heatmap.png",
        ),
    ):
        path = fn()
        log.info("Wrote %s", path)
        n_figures += 1
    for trial in (1, 2):
        path = plot_confusion_matrix(
            exp_c.confusion_matrices[trial], trial, figures_dir / f"confusion_matrix_trial{trial}.png"
        )
        log.info("Wrote %s", path)
        n_figures += 1
        split = ds.trials[trial]
        X_test = ds.X[split.test_idx]
        y_test = ds.y[split.test_idx]
        path = plot_error_examples(
            X_test, y_test, exp_a.predictions[(trial, "svm_rbf")], trial,
            figures_dir / f"error_examples_trial{trial}.png",
        )
        if path is None:
            log.info("Trial %d: no misclassifications, error-example figure skipped.", trial)
        else:
            log.info("Wrote %s", path)
            n_figures += 1

    # --- metadata and notes ---
    _write_metadata(args, ds.source_files, time.perf_counter() - t_start, n_tables, n_figures, output_dir)
    _write_experiment_notes(exp_a, exp_b, exp_c, output_dir, {1: 0.9135, 2: 0.9185})

    _print_final_summary(exp_a, exp_b)
    log.info("All done in %.1fs. Outputs in %s", time.perf_counter() - t_start, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
