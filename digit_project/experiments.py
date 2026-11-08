"""The three locked experiments: A (classifier benchmark), B (PCA), C (analysis).

Protocol (per the course project description):
- For each trial, only the trial's training indices are used for fitting and
  for inner CV; the trial's test indices are evaluated exactly once.
- Every data-dependent preprocessing step (PCA) lives inside a sklearn
  Pipeline, so each CV fold refits it on the fold's training data only.
- Pixel scaling ``X / 255`` is a deterministic transform of known range and
  is applied to the whole matrix before splitting.
- All hyper-parameters are selected by 3-fold stratified CV on the training
  set; the final model is then evaluated once on the test set.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

from .data import DigitDataset, TrialSplit
from .models import (
    make_grid_search,
    make_model_specs,
    make_pca_pipeline,
)

log = logging.getLogger(__name__)

# Experiment B: only these two classifiers are compared across PCA dims.
PCA_MODELS = ("logreg", "svm_rbf")
PCA_DIMS = (None, 50, 100, 200)


@dataclass
class ExperimentA:
    """Results of Experiment A: per-trial model rows, summary, CV details."""

    trials_df: pd.DataFrame  # one row per model x trial
    summary_df: pd.DataFrame  # per-model mean / difference over the two trials
    cv_df: pd.DataFrame  # training-set CV scores of every tuned parameter combo
    predictions: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)

    @property
    def predictions_by_model(self) -> dict[str, dict[int, np.ndarray]]:
        out: dict[str, dict[int, np.ndarray]] = {}
        for (trial, config_id), pred in self.predictions.items():
            out.setdefault(config_id, {})[trial] = pred
        return out


@dataclass
class ExperimentB:
    """Results of Experiment B: PCA-dimension comparison for LR and RBF SVM."""

    pca_df: pd.DataFrame  # one row per model x dims x trial


@dataclass
class ExperimentC:
    """Results of Experiment C: RBF-SVM confusion matrices and error samples."""

    confusion_matrices: dict[int, np.ndarray]  # trial -> (10, 10) counts
    misclassified_idx: dict[int, np.ndarray]  # trial -> 0-based test-set indices
    error_pairs: pd.DataFrame  # top true->pred confusion pairs per trial


def _fit_and_evaluate(
    search: GridSearchCV | None,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit (optionally via GridSearchCV) on train and evaluate once on test.

    Returns a dict with accuracy, macro_f1, best_params, fit/predict wall
    times, the test predictions, and the per-combo CV score table (empty list
    when no grid was used).  The predictions are reused later for Experiment C
    so nothing is recomputed.
    """
    t0 = time.perf_counter()
    if search is not None:
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = dict(search.best_params_)
        cv_mean = float(search.best_score_)
        cv_details = [
            {
                "params": p,
                "mean_cv_accuracy": float(m),
                "std_cv_accuracy": float(s),
                "rank": int(r),
            }
            for p, m, s, r in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                search.cv_results_["rank_test_score"],
            )
        ]
    else:
        estimator.fit(X_train, y_train)
        model = estimator
        best_params = None
        cv_mean = None
        cv_details = []
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "best_params": best_params,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "y_pred": y_pred,
        "cv_mean": cv_mean,
        "cv_details": cv_details,
    }


def _evaluate_model_on_trial(
    spec,
    X: np.ndarray,
    y: np.ndarray,
    split: TrialSplit,
    trial: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict, list[dict], np.ndarray]:
    """Evaluate one Experiment A configuration on one trial.

    Returns (result row, CV-detail rows, test predictions).
    """
    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_test, y_test = X[split.test_idx], y[split.test_idx]

    search = (
        make_grid_search(spec.estimator, spec.param_grid, seed=seed, n_jobs=n_jobs)
        if spec.param_grid is not None
        else None
    )
    res = _fit_and_evaluate(search, spec.estimator, X_train, y_train, X_test, y_test)
    row = {
        "trial": trial,
        "config_id": spec.config_id,
        "model_name": spec.model_name,
        "best_params": str(res["best_params"]) if res["best_params"] else "",
        "cv_mean_accuracy": res["cv_mean"],
        "test_accuracy": res["test_accuracy"],
        "test_macro_f1": res["test_macro_f1"],
        "fit_time_sec": res["fit_time_sec"],
        "predict_time_sec": res["predict_time_sec"],
    }
    cv_rows = [
        {
            "trial": trial,
            "config_id": spec.config_id,
            **detail,
        }
        for detail in res["cv_details"]
    ]
    return row, cv_rows, res["y_pred"]


def run_experiment_a(
    ds: DigitDataset, seed: int = 42, n_jobs: int = 1, trials: tuple[int, ...] = (1, 2)
) -> ExperimentA:
    """Experiment A: the five classifier configurations on both trials."""
    X = ds.X.astype(np.float32) / 255.0  # deterministic fixed-range scaling
    specs = make_model_specs(seed)
    rows: list[dict] = []
    cv_rows_all: list[dict] = []
    predictions: dict[tuple[int, str], np.ndarray] = {}

    for trial in trials:
        log.info("Experiment A: Trial %d", trial)
        for spec in specs:
            log.info("  - fitting %s", spec.config_id)
            row, cv_rows, y_pred = _evaluate_model_on_trial(
                spec, X, ds.y, ds.trials[trial], trial, seed, n_jobs
            )
            rows.append(row)
            cv_rows_all.extend(cv_rows)
            predictions[(trial, spec.config_id)] = y_pred
            log.info(
                "    test_acc=%.4f macro_f1=%.4f best_params=%s fit=%.1fs predict=%.2fs",
                row["test_accuracy"],
                row["test_macro_f1"],
                row["best_params"],
                row["fit_time_sec"],
                row["predict_time_sec"],
            )

    trials_df = pd.DataFrame(rows)
    summary = (
        trials_df.groupby(["config_id", "model_name"])
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_macro_f1=("test_macro_f1", "mean"),
            trial1_accuracy=("test_accuracy", lambda s: s.iloc[0]),
            trial2_accuracy=("test_accuracy", lambda s: s.iloc[1]),
        )
        .reset_index()
    )
    summary["acc_diff_t2_minus_t1"] = summary["trial2_accuracy"] - summary["trial1_accuracy"]
    summary = summary.sort_values("mean_test_accuracy", ascending=False).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows_all)
    return ExperimentA(trials_df=trials_df, summary_df=summary, cv_df=cv_df, predictions=predictions)


def run_experiment_b(
    ds: DigitDataset,
    seed: int = 42,
    n_jobs: int = 1,
    trials: tuple[int, ...] = (1, 2),
    dims: tuple = PCA_DIMS,
) -> ExperimentB:
    """Experiment B: LR and RBF SVM with PCA to {None, 50, 100, 200} dims.

    ``dims`` is a parameter so the smoke tests can pass dimensions adapted to
    their small synthetic data (the real protocol always uses the locked
    ``PCA_DIMS``).
    """
    X = ds.X.astype(np.float32) / 255.0
    y = ds.y
    specs = {s.config_id: s for s in make_model_specs(seed)}
    rows: list[dict] = []

    for trial in trials:
        split = ds.trials[trial]
        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_test, y_test = X[split.test_idx], y[split.test_idx]
        for config_id in PCA_MODELS:
            spec = specs[config_id]
            for n_components in dims:
                log.info(
                    "Experiment B: trial %d, %s, n_components=%s",
                    trial,
                    config_id,
                    "None" if n_components is None else n_components,
                )
                pipeline = make_pca_pipeline(spec.estimator, n_components, seed=seed)
                # Inside a Pipeline the grid keys need the step-name prefix.
                grid = {f"clf__{k}": v for k, v in spec.param_grid.items()}
                search = make_grid_search(pipeline, grid, seed=seed, n_jobs=n_jobs)
                res = _fit_and_evaluate(search, pipeline, X_train, y_train, X_test, y_test)
                cum_evr = None
                if n_components is not None:
                    cum_evr = float(
                        np.sum(search.best_estimator_.named_steps["pca"].explained_variance_ratio_)
                    )
                rows.append(
                    {
                        "trial": trial,
                        "config_id": config_id,
                        "model_name": spec.model_name,
                        "n_components": n_components if n_components is not None else "None (784)",
                        "cv_best_params": str(res["best_params"]) if res["best_params"] else "",
                        "cv_mean_accuracy": res["cv_mean"],
                        "test_accuracy": res["test_accuracy"],
                        "test_macro_f1": res["test_macro_f1"],
                        "cum_explained_variance_ratio": cum_evr,
                        "fit_time_sec": res["fit_time_sec"],
                        "predict_time_sec": res["predict_time_sec"],
                    }
                )
                log.info(
                    "    test_acc=%.4f macro_f1=%.4f cum_evr=%s fit=%.1fs predict=%.2fs",
                    res["test_accuracy"],
                    res["test_macro_f1"],
                    f"{cum_evr:.3f}" if cum_evr is not None else "n/a",
                    res["fit_time_sec"],
                    res["predict_time_sec"],
                )
    return ExperimentB(pca_df=pd.DataFrame(rows))


def run_experiment_c(
    ds: DigitDataset,
    predictions_by_trial: dict[int, np.ndarray],
    trials: tuple[int, ...] = (1, 2),
    top_pairs: int = 8,
) -> ExperimentC:
    """Experiment C: confusion matrices and error analysis for RBF SVM."""
    confusions: dict[int, np.ndarray] = {}
    misclassified: dict[int, np.ndarray] = {}
    pair_rows: list[dict] = []

    for trial in trials:
        split = ds.trials[trial]
        y_true = ds.y[split.test_idx]
        y_pred = predictions_by_trial[trial]
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
        confusions[trial] = cm
        wrong = np.nonzero(y_pred != y_true)[0]
        misclassified[trial] = wrong
        for a in range(10):
            for b in range(10):
                if a != b and cm[a, b] > 0:
                    pair_rows.append(
                        {
                            "trial": trial,
                            "true_label": a,
                            "pred_label": b,
                            "count": int(cm[a, b]),
                        }
                    )
        log.info("Experiment C: trial %d RBF SVM errors: %d / %d", trial, wrong.size, y_true.size)

    error_pairs = pd.DataFrame(pair_rows)
    if not error_pairs.empty:
        # top-N confusion pairs per trial (a single head() across the whole
        # table would let one trial's long tail crowd out the other trial)
        error_pairs = (
            error_pairs.sort_values(["trial", "count"], ascending=[True, False])
            .groupby("trial", sort=False)
            .head(top_pairs)
            .reset_index(drop=True)
        )
    return ExperimentC(
        confusion_matrices=confusions,
        misclassified_idx=misclassified,
        error_pairs=error_pairs,
    )
