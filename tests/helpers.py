"""Shared helpers for the test suite: small synthetic datasets in TXT format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from digit_project.data import (
    LABELS_FILENAME,
    TESTSET_FILENAME,
    TRAINSET_FILENAME,
    VEC_FILENAME,
)


def make_synthetic_arrays(
    n_samples: int = 60, n_features: int = 10, n_classes: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-blob synthetic data with balanced classes.

    Values are rounded to integers: the TXT format stores integer pixels and
    the loader must round-trip them exactly.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)
    per_class = n_samples // n_classes
    for c in range(n_classes):
        mean = rng.normal(0, 3.0, size=n_features)
        X[c * per_class:(c + 1) * per_class] = np.rint(
            rng.normal(mean, 0.8, size=(per_class, n_features))
        ).astype(np.float32)
        y[c * per_class:(c + 1) * per_class] = c
    return X, y


def write_synthetic_txt(
    directory: Path,
    X: np.ndarray,
    y: np.ndarray,
    train_idx_1based: np.ndarray,
    test_idx_1based: np.ndarray,
) -> None:
    """Write the four TXT files in the course format (tab-delimited integers).

    ``train_idx_1based`` / ``test_idx_1based`` are (n_train, 2) arrays with one
    1-based column per trial, mirroring the real files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / VEC_FILENAME, X, fmt="%d")
    np.savetxt(directory / LABELS_FILENAME, y, fmt="%d")
    np.savetxt(directory / TRAINSET_FILENAME, train_idx_1based, fmt="%d")
    np.savetxt(directory / TESTSET_FILENAME, test_idx_1based, fmt="%d")


def make_synthetic_txt_dataset(
    directory: Path, n_samples: int = 60, n_features: int = 10, n_classes: int = 3, seed: int = 0
) -> None:
    """Create a full valid synthetic TXT dataset (balanced 50/50 splits)."""
    X, y = make_synthetic_arrays(n_samples, n_features, n_classes, seed)
    n = n_samples
    # trial 1: first half train, second half test; trial 2: the reverse.
    train = np.column_stack([np.arange(1, n // 2 + 1), np.arange(n // 2 + 1, n + 1)])
    test = np.column_stack([np.arange(n // 2 + 1, n + 1), np.arange(1, n // 2 + 1)])
    write_synthetic_txt(directory, X, y, train, test)
