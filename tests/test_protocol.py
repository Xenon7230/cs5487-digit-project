"""Protocol tests: locked model configs, PCA-inside-pipeline, CV leakage guard."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from digit_project.models import (
    make_grid_search,
    make_model_specs,
    make_pca_pipeline,
)

from .helpers import make_synthetic_arrays


def test_experiment_a_configs_are_locked() -> None:
    """The five configurations and their grids must match the course protocol."""
    specs = make_model_specs()
    assert [s.config_id for s in specs] == ["knn_1", "logreg", "lda", "svm_linear", "svm_rbf"]
    by_id = {s.config_id: s for s in specs}
    assert by_id["knn_1"].param_grid is None  # fixed baseline, no tuning
    assert by_id["lda"].param_grid is None  # fixed configuration
    assert by_id["logreg"].param_grid == {"C": [0.1, 1.0, 10.0]}
    assert by_id["svm_linear"].param_grid == {"C": [0.1, 1.0, 10.0]}
    assert by_id["svm_rbf"].param_grid == {"C": [0.1, 1.0, 10.0], "gamma": [0.001, 0.01, 0.1]}
    # 1-NN baseline matches the instructor's reference configuration
    assert by_id["knn_1"].estimator.n_neighbors == 1
    assert by_id["knn_1"].estimator.metric == "euclidean"
    # LDA is shrinkage with lsqr solver
    assert by_id["lda"].estimator.solver == "lsqr"
    assert by_id["lda"].estimator.shrinkage == "auto"


def test_pca_pipeline_structure() -> None:
    clf = LogisticRegression(max_iter=500)
    pipe = make_pca_pipeline(clf, n_components=50)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["pca", "clf"]
    assert isinstance(pipe.named_steps["pca"], PCA)
    # None means: no PCA step at all (full 784-dim representation)
    pipe_none = make_pca_pipeline(clf, n_components=None)
    assert [name for name, _ in pipe_none.steps] == ["clf"]


def test_grid_search_uses_stratified_3fold() -> None:
    search = make_grid_search(LogisticRegression(max_iter=500), {"C": [1.0]}, seed=42)
    assert search.scoring == "accuracy"
    assert search.refit is True
    cv = search.cv
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 3
    assert cv.shuffle is True
    assert cv.random_state == 42


class _RecordingPCA(PCA):
    """PCA that records fingerprints of every matrix it fits (class-level, so
    clones made by GridSearchCV share the same record).

    Both ``fit`` and ``fit_transform`` are overridden: sklearn's Pipeline
    calls ``fit_transform`` on transformers, and PCA's own ``fit_transform``
    bypasses ``fit`` via the private ``_fit``.
    """

    _records: list[str] = []

    def _record(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        self.__class__._records.append(
            hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()
        )

    def fit(self, X, y=None):  # noqa: D102
        self._record(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):  # noqa: D102
        self._record(X)
        return super().fit_transform(X, y)


def _fp(X: np.ndarray) -> str:
    X = np.asarray(X, dtype=np.float32)
    return hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()


def test_pca_inside_cv_fits_only_fold_training_data() -> None:
    """Leakage guard: during GridSearchCV, PCA must see exactly the fold's
    training subset (3 folds) and finally the full training set — never the
    test set, never the union of all folds."""
    X, y = make_synthetic_arrays(n_samples=90, n_features=12, n_classes=3, seed=3)
    X_train, y_train = X[:60], y[:60]
    X_test = X[60:]

    _RecordingPCA._records = []
    pipe = Pipeline(
        [("pca", _RecordingPCA(n_components=4, random_state=42)),
         ("clf", LogisticRegression(max_iter=500))]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, {"clf__C": [0.1, 1.0]}, scoring="accuracy", cv=cv, n_jobs=1)
    search.fit(X_train, y_train)

    fold_fps = {_fp(X_train[idx]) for idx, _ in cv.split(X_train, y_train)}
    records = _RecordingPCA._records
    # 2 params x 3 folds = 6 fold fits, then one refit on the full training set
    assert len(records) == 7
    assert set(records[:6]) == fold_fps
    assert records[-1] == _fp(X_train)
    test_fp = _fp(X_test)
    assert test_fp not in records
    assert search.predict(X_test).shape == (30,)
