"""Tests for data loading, index conversion, and split validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digit_project.data import (
    DataFormatError,
    DigitDataset,
    TrialSplit,
    load_txt,
    locate_data_dir,
    to_zero_based,
    validate_dataset,
)

from .helpers import make_synthetic_arrays, make_synthetic_txt_dataset, write_synthetic_txt


def test_locate_data_dir_finds_subdir(tmp_path: Path) -> None:
    make_synthetic_txt_dataset(tmp_path / "digits4000_txt")
    assert locate_data_dir(tmp_path) == (tmp_path / "digits4000_txt").resolve()
    assert locate_data_dir(tmp_path / "digits4000_txt") == (tmp_path / "digits4000_txt").resolve()
    with pytest.raises(DataFormatError):
        locate_data_dir(tmp_path / "nowhere")


def test_to_zero_based_conversion() -> None:
    idx = np.array([1, 2, 4000, 2000], dtype=np.int64)
    assert np.array_equal(to_zero_based(idx), np.array([0, 1, 3999, 1999]))


def test_to_zero_based_rejects_bad_input() -> None:
    with pytest.raises(DataFormatError, match="1\\.\\.4000"):
        to_zero_based(np.array([0, 5]))
    with pytest.raises(DataFormatError, match="1\\.\\.4000"):
        to_zero_based(np.array([4001]))
    with pytest.raises(DataFormatError, match="integers"):
        to_zero_based(np.array([1.0, 2.0]))
    with pytest.raises(DataFormatError, match="1-D"):
        to_zero_based(np.array([[1, 2]]))


def test_load_txt_roundtrip_orientation(tmp_path: Path) -> None:
    """Each row of the TXT is one sample (sample-major, no extra transpose)."""
    X, y = make_synthetic_arrays(n_samples=60, n_features=10, n_classes=3, seed=7)
    n = X.shape[0]
    train = np.column_stack([np.arange(1, n // 2 + 1), np.arange(n // 2 + 1, n + 1)])
    test = np.column_stack([np.arange(n // 2 + 1, n + 1), np.arange(1, n // 2 + 1)])
    write_synthetic_txt(tmp_path / "digits4000_txt", X, y, train, test)
    ds = load_txt(tmp_path, validate=False)
    assert ds.X.shape == (60, 10)
    assert ds.y.shape == (60,)
    assert np.array_equal(ds.X, X)
    assert np.array_equal(ds.y, y)
    # trial 1: first half trains, second half tests (1-based -> 0-based)
    assert np.array_equal(ds.trials[1].train_idx, np.arange(0, 30))
    assert np.array_equal(ds.trials[1].test_idx, np.arange(30, 60))
    assert np.array_equal(ds.trials[2].train_idx, np.arange(30, 60))
    assert np.array_equal(ds.trials[2].test_idx, np.arange(0, 30))


def test_load_txt_missing_file_raises(tmp_path: Path) -> None:
    make_synthetic_txt_dataset(tmp_path)
    (tmp_path / "digits4000_digits_labels.txt").unlink()
    with pytest.raises(DataFormatError, match="digits4000_digits_labels"):
        load_txt(tmp_path, validate=False)


def test_validation_overlapping_split_raises() -> None:
    ds = DigitDataset(
        X=np.zeros((40, 4), dtype=np.float32),
        y=np.repeat(np.arange(4), 10),
        trials={1: TrialSplit(np.arange(20), np.arange(5, 25))},
    )
    with pytest.raises(DataFormatError, match="overlap"):
        validate_dataset(ds, n_samples=40, n_features=4, n_classes=4,
                         samples_per_class=10, trial_size=20)


def test_validation_imbalanced_split_raises() -> None:
    ds = DigitDataset(
        X=np.zeros((40, 4), dtype=np.float32),
        y=np.repeat(np.arange(4), 10),
        trials={1: TrialSplit(np.arange(0, 20), np.arange(20, 40))},
    )
    # y is balanced but the split is not (train holds classes 0,1 only)
    with pytest.raises(DataFormatError, match="class-balanced"):
        validate_dataset(ds, n_samples=40, n_features=4, n_classes=4,
                         samples_per_class=10, trial_size=20)


def test_validation_wrong_shape_raises() -> None:
    ds = DigitDataset(
        X=np.zeros((40, 3), dtype=np.float32),
        y=np.repeat(np.arange(4), 10),
        trials={},
    )
    with pytest.raises(DataFormatError, match="Expected X shape"):
        validate_dataset(ds, n_samples=40, n_features=4, n_classes=4,
                         samples_per_class=10, trial_size=20)


def test_validation_pixel_range_raises() -> None:
    X = np.zeros((40, 4), dtype=np.float32)
    X[0, 0] = 300.0
    ds = DigitDataset(X=X, y=np.repeat(np.arange(4), 10), trials={})
    with pytest.raises(DataFormatError, match="\\[0, 255\\]"):
        validate_dataset(ds, n_samples=40, n_features=4, n_classes=4,
                         samples_per_class=10, trial_size=20)


# --- optional integration test on the real dataset -------------------------

@pytest.mark.skipif(
    not (Path("data/digits4000_txt/digits4000_digits_vec.txt").exists()),
    reason="real digits4000 TXT data not present",
)
def test_real_data_baseline_reproduction() -> None:
    """The course-defined sanity check: Euclidean 1-NN must reproduce the
    instructor's trial accuracies (91.35% / 91.85%) within a tiny tolerance."""
    from sklearn.neighbors import KNeighborsClassifier

    from digit_project.data import load_data

    ds = load_data("data/digits4000_txt")
    X = ds.X.astype(np.float32) / 255.0
    expected = {1: 0.9135, 2: 0.9185}
    for trial, ref in expected.items():
        split = ds.trials[trial]
        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn.fit(X[split.train_idx], ds.y[split.train_idx])
        pred = knn.predict(X[split.test_idx])
        acc = float(np.mean(pred == ds.y[split.test_idx]))
        assert abs(acc - ref) < 0.005, f"trial {trial}: got {acc:.4f}, expected ~{ref:.4f}"
