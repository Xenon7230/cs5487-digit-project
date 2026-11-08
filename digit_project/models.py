"""Model definitions for the digit-classification experiments.

Defines the five fixed classifier configurations (Experiment A) and the
PCA pipeline builders (Experiment B).  All hyper-parameter selection happens
through ``GridSearchCV`` inside the training set only; the grids are fixed
here and never touched by test results.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Fixed, course-defined inner-CV protocol (see CS5487 project description):
# 3-fold stratified CV on the training set only, fixed seed.
N_SPLITS = 3
CV_RANDOM_STATE = 42


@dataclass
class ModelSpec:
    """One classifier configuration of Experiment A.

    ``param_grid is None`` means the configuration is fixed and no
    hyper-parameter search is performed (1-NN, Shrinkage LDA).
    """

    config_id: str
    model_name: str
    estimator: BaseEstimator
    param_grid: dict | None
    description: str


def make_cv_splitter(seed: int = CV_RANDOM_STATE) -> StratifiedKFold:
    """The course-defined inner-CV splitter used for all parameter tuning."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


def make_model_specs(seed: int = CV_RANDOM_STATE) -> list[ModelSpec]:
    """The five locked classifier configurations of Experiment A.

    - ``knn_1``: Euclidean 1-NN baseline, matches the instructor's reference
      numbers (91.35% / 91.85%).
    - ``logreg``: L2-regularized multinomial logistic regression, ``C`` tuned
      over {0.1, 1, 10} via training-set CV.
    - ``lda``: shrinkage LDA (fixed configuration, no grid search).
    - ``svm_linear``: linear SVM, ``C`` tuned over {0.1, 1, 10}.
    - ``svm_rbf``: RBF SVM, ``C`` x ``gamma`` tuned over a 3x3 grid.
    """
    return [
        ModelSpec(
            config_id="knn_1",
            model_name="1-NN (Euclidean)",
            estimator=KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
            param_grid=None,
            description="Euclidean 1-nearest-neighbor baseline (instructor reference).",
        ),
        ModelSpec(
            config_id="logreg",
            model_name="Logistic Regression (L2)",
            estimator=LogisticRegression(
                l1_ratio=0.0, C=1.0, max_iter=2000, random_state=seed
            ),
            param_grid={"C": [0.1, 1.0, 10.0]},
            description=(
                "Multinomial logistic regression with L2 penalty (l1_ratio=0 on "
                "modern scikit-learn, where the penalty argument is deprecated); "
                "C chosen by CV."
            ),
        ),
        ModelSpec(
            config_id="lda",
            model_name="Shrinkage LDA",
            estimator=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            param_grid=None,
            description="Linear discriminant analysis with automatic shrinkage; fixed config.",
        ),
        ModelSpec(
            config_id="svm_linear",
            model_name="Linear SVM",
            estimator=SVC(kernel="linear", C=1.0, random_state=seed),
            param_grid={"C": [0.1, 1.0, 10.0]},
            description="Linear-kernel SVM; C chosen by CV.",
        ),
        ModelSpec(
            config_id="svm_rbf",
            model_name="RBF SVM",
            estimator=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
            param_grid={"C": [0.1, 1.0, 10.0], "gamma": [0.001, 0.01, 0.1]},
            description="RBF-kernel SVM; C x gamma (3x3) chosen by CV.",
        ),
    ]


def make_pca_pipeline(
    classifier: BaseEstimator, n_components: int | None, seed: int = CV_RANDOM_STATE
) -> Pipeline:
    """Build ``PCA -> classifier`` as an sklearn Pipeline (CV-safe).

    ``n_components=None`` returns the classifier without a PCA step (the full
    784-dim representation), so the two arms of Experiment B share the same
    evaluation code.  The PCA is placed inside the pipeline so that every CV
    fold refits it on the fold's training data only (no leakage).
    """
    if n_components is None:
        return Pipeline([("clf", classifier)])
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=seed)
    return Pipeline([("pca", pca), ("clf", classifier)])


def make_grid_search(
    estimator: BaseEstimator, param_grid: dict, seed: int = CV_RANDOM_STATE, n_jobs: int = 1
) -> GridSearchCV:
    """GridSearchCV configured per the course protocol.

    ``refit=True`` leaves the best estimator trained on the full training set,
    ready for the single final test-set evaluation.
    """
    return GridSearchCV(
        estimator,
        param_grid,
        scoring="accuracy",
        cv=make_cv_splitter(seed),
        n_jobs=n_jobs,
        refit=True,
    )


def get_estimator(seed: int = CV_RANDOM_STATE, n_jobs: int = 1) -> dict[str, ModelSpec]:
    """Convenience lookup of Experiment A specs by config id."""
    return {s.config_id: s for s in make_model_specs(seed)}
