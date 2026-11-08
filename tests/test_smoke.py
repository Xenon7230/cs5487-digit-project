"""End-to-end smoke tests on small synthetic data + visualization smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digit_project.data import DigitDataset, TrialSplit
from digit_project.experiments import run_experiment_a, run_experiment_b
from digit_project.plots import plot_confusion_matrix, plot_error_examples


def make_synthetic_ds(n_samples: int = 120, n_features: int = 16, n_classes: int = 10, seed: int = 5) -> DigitDataset:
    """10-class synthetic digits-like data (0..255 integer pixels, balanced)."""
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)
    for c in range(n_classes):
        mean = rng.integers(40, 215, size=n_features)
        block = rng.normal(mean, 30, size=(per_class, n_features)).clip(0, 255).astype(np.float32)
        X[c * per_class:(c + 1) * per_class] = block
        y[c * per_class:(c + 1) * per_class] = c
    half = n_samples // 2
    trials = {
        1: TrialSplit(np.arange(0, half), np.arange(half, n_samples)),
        2: TrialSplit(np.arange(half, n_samples), np.arange(0, half)),
    }
    return DigitDataset(X=X, y=y, trials=trials)


def test_experiment_a_smoke() -> None:
    """A few locked configs must produce valid metrics on synthetic data."""
    ds = make_synthetic_ds()
    result = run_experiment_a(ds, seed=42, n_jobs=1)
    assert result.trials_df.shape[0] == 5 * 2  # 5 models x 2 trials
    for col in ("test_accuracy", "test_macro_f1", "fit_time_sec", "predict_time_sec"):
        assert np.all(np.isfinite(result.trials_df[col]))
    assert np.all((result.trials_df["test_accuracy"] >= 0) & (result.trials_df["test_accuracy"] <= 1))
    assert not result.cv_df.empty  # CV rows for the three tuned models
    assert {"logreg", "svm_linear", "svm_rbf"} <= set(result.cv_df["config_id"])
    # summary table aggregates the two trials per model
    assert result.summary_df.shape[0] == 5
    assert np.all(np.isfinite(result.summary_df["mean_test_accuracy"]))
    # predictions stored for every model x trial (needed by Experiment C)
    assert len(result.predictions) == 10


def test_experiment_b_smoke() -> None:
    """PCA pipeline (dims adapted to synthetic data) must run and return metrics."""
    ds = make_synthetic_ds()
    # dims adapted to the synthetic 16-feature data (real protocol uses 50/100/200)
    result = run_experiment_b(ds, seed=42, n_jobs=1, dims=(None, 4, 8, 12))
    # 2 models x 4 dim settings x 2 trials
    assert result.pca_df.shape[0] == 16
    assert np.all(np.isfinite(result.pca_df["test_accuracy"]))
    # explained-variance ratio present for the three reduced settings
    evr = result.pca_df.loc[result.pca_df["n_components"] != "None (784)", "cum_explained_variance_ratio"]
    assert np.all(np.isfinite(evr))
    assert np.all((evr > 0) & (evr <= 1.0 + 1e-6))


def test_error_plot_handles_no_errors(tmp_path: Path) -> None:
    """Visualization must not crash when there are no misclassifications."""
    y = np.array([0, 1, 2, 3])
    out = plot_error_examples(
        X=np.zeros((4, 784), dtype=np.float32), y_true=y, y_pred=y, trial=1,
        out_path=tmp_path / "none.png",
    )
    assert out is None  # no figure produced, no crash


def test_error_plot_writes_file_with_errors(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([1, 1, 2, 3])  # one error
    out = plot_error_examples(
        X=np.random.default_rng(0).integers(0, 255, size=(4, 784)).astype(np.float32),
        y_true=y_true, y_pred=y_pred, trial=1,
        out_path=tmp_path / "with_errors.png",
    )
    assert out is not None and out.is_file()


def test_confusion_matrix_plot_writes_file(tmp_path: Path) -> None:
    cm = np.array([[3, 1], [0, 4]])
    out = plot_confusion_matrix(cm, trial=1, out_path=tmp_path / "cm.png")
    assert out.is_file()


def test_error_pairs_cover_both_trials() -> None:
    """Regression: the top-pair table must include both trials, not let one
    trial's long tail crowd out the other."""
    from digit_project.experiments import run_experiment_c

    ds = make_synthetic_ds()
    exp_a = run_experiment_a(ds, seed=42, n_jobs=1)
    exp_c = run_experiment_c(
        ds,
        {1: exp_a.predictions[(1, "svm_rbf")], 2: exp_a.predictions[(2, "svm_rbf")]},
    )
    assert {1, 2} <= set(exp_c.error_pairs["trial"])
    assert exp_c.error_pairs.groupby("trial").size().max() <= 8  # top_pairs cap


def test_rbf_cv_heatmap_handles_inmemory_dict_params(tmp_path: Path) -> None:
    """Regression: cv_df in memory holds raw dicts in the params column
    (they only become strings after a CSV round-trip); the heatmap must
    handle both."""
    from digit_project.plots import plot_rbf_cv_heatmap

    ds = make_synthetic_ds()
    exp_a = run_experiment_a(ds, seed=42, n_jobs=1)
    assert not exp_a.cv_df.empty
    out = plot_rbf_cv_heatmap(exp_a.cv_df, tmp_path / "heatmap.png")
    assert out.is_file()


def test_result_tables_have_required_columns() -> None:
    """The CSV columns consumed by the figure generators exist after a run."""
    ds = make_synthetic_ds()
    exp_a = run_experiment_a(ds, seed=42, n_jobs=1)
    exp_b = run_experiment_b(ds, seed=42, n_jobs=1, dims=(None, 4, 8, 12))
    required_a = {"trial", "config_id", "model_name", "test_accuracy"}
    required_b = {"trial", "config_id", "model_name", "n_components", "test_accuracy", "fit_time_sec"}
    assert required_a <= set(exp_a.trials_df.columns)
    assert required_b <= set(exp_b.pca_df.columns)
    # cv table rows are parseable back into param dicts
    assert not exp_a.cv_df.empty
    assert {"trial", "config_id", "params", "mean_cv_accuracy", "rank"} <= set(exp_a.cv_df.columns)
