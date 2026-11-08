"""Data loading and validation for the digits4000 handwritten-digit dataset.

The original MATLAB matrices are column-major and the TXT export is their
transpose, so every row of a TXT file is one sample (sample-major layout).
The trainset/testset files contain 1-based MATLAB sample indices; this module
converts them to 0-based Python indices and validates the whole dataset
against the documented properties (shapes, label/range checks, disjoint
writer-separated splits, class balance).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_SAMPLES = 4000
N_FEATURES = 784
N_CLASSES = 10
SAMPLES_PER_CLASS = 400
TRIAL_SIZE = 2000

VEC_FILENAME = "digits4000_digits_vec.txt"
LABELS_FILENAME = "digits4000_digits_labels.txt"
TRAINSET_FILENAME = "digits4000_trainset.txt"
TESTSET_FILENAME = "digits4000_testset.txt"

_ALL_TXT_FILES = (VEC_FILENAME, LABELS_FILENAME, TRAINSET_FILENAME, TESTSET_FILENAME)


class DataFormatError(ValueError):
    """Raised when dataset files are missing or violate the expected format."""


@dataclass
class TrialSplit:
    """0-based sample indices of one trial of the course-defined split."""

    train_idx: np.ndarray  # (2000,) int64
    test_idx: np.ndarray  # (2000,) int64


@dataclass
class DigitDataset:
    """Sample-major view of the digits4000 dataset."""

    X: np.ndarray  # (4000, 784) raw pixel values, integers in [0, 255]
    y: np.ndarray  # (4000,) integer labels 0..9
    trials: dict[int, TrialSplit]  # trial number (1, 2) -> split
    source_files: dict[str, str] = field(default_factory=dict)  # filename -> sha256


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file (for reproducibility metadata)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_data_dir(data_dir: str | Path) -> Path:
    """Resolve a user-supplied path to the directory holding the four TXT files.

    Accepts either the directory itself or its parent (in which case a
    ``digits4000_txt`` subdirectory is looked up).
    """
    p = Path(data_dir).expanduser().resolve()
    if p.is_dir() and all((p / f).is_file() for f in _ALL_TXT_FILES):
        return p
    sub = p / "digits4000_txt"
    if sub.is_dir() and all((sub / f).is_file() for f in _ALL_TXT_FILES):
        return sub
    raise DataFormatError(
        f"Data directory {p} does not contain all expected files {_ALL_TXT_FILES} "
        "(neither directly nor in a 'digits4000_txt' subdirectory)."
    )


def to_zero_based(indices_1based: np.ndarray) -> np.ndarray:
    """Convert 1-based MATLAB sample indices to 0-based Python indices."""
    if indices_1based.ndim != 1:
        raise DataFormatError(f"Expected 1-D index array, got shape {indices_1based.shape}.")
    if not np.issubdtype(indices_1based.dtype, np.integer):
        raise DataFormatError(f"Indices must be integers, got dtype {indices_1based.dtype}.")
    if indices_1based.min() < 1 or indices_1based.max() > N_SAMPLES:
        raise DataFormatError(
            f"Indices out of the expected 1..{N_SAMPLES} range: "
            f"min={indices_1based.min()}, max={indices_1based.max()}."
        )
    return indices_1based - 1


def load_txt(data_dir: str | Path, validate: bool = True) -> DigitDataset:
    """Load the four TXT files from the given directory (tab-delimited integers).

    ``validate=False`` skips the shape/balance checks (used by tests that
    build small synthetic datasets in the same file format).
    """
    d = locate_data_dir(data_dir)
    try:
        X = np.loadtxt(d / VEC_FILENAME, dtype=np.float32)  # (4000, 784), row = sample
        y = np.loadtxt(d / LABELS_FILENAME, dtype=np.int64).ravel()  # (4000,)
        trainset = np.loadtxt(d / TRAINSET_FILENAME, dtype=np.int64)  # (2000, 2)
        testset = np.loadtxt(d / TESTSET_FILENAME, dtype=np.int64)  # (2000, 2)
    except FileNotFoundError as exc:
        raise DataFormatError(f"Missing data file: {exc.filename}") from exc
    except ValueError as exc:
        raise DataFormatError(f"Could not parse a data file in {d}: {exc}") from exc

    if trainset.ndim != 2 or testset.ndim != 2 or trainset.shape[1] != 2 or testset.shape[1] != 2:
        raise DataFormatError(
            f"trainset/testset must be (2000, 2), got {trainset.shape} / {testset.shape}."
        )

    trials: dict[int, TrialSplit] = {}
    for trial, col in ((1, 0), (2, 1)):
        trials[trial] = TrialSplit(
            train_idx=to_zero_based(trainset[:, col]),
            test_idx=to_zero_based(testset[:, col]),
        )

    ds = DigitDataset(
        X=X,
        y=y,
        trials=trials,
        source_files={f: sha256_file(d / f) for f in _ALL_TXT_FILES},
    )
    if validate:
        validate_dataset(ds)
    return ds


def load_mat(mat_path: str | Path) -> DigitDataset:
    """Optional loader for the original MATLAB .mat file.

    In the .mat the data is stored as MATLAB column-major matrices:
    ``digits_vec`` (784, 4000), ``digits_labels`` (1, 4000),
    ``trainset``/``testset`` (2, 2000) with one trial per row.
    """
    import scipy.io as sio  # imported lazily: .mat support is optional

    path = Path(mat_path).expanduser().resolve()
    if not path.is_file():
        raise DataFormatError(f"MAT file not found: {path}")
    try:
        mat = sio.loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - report a readable error
        raise DataFormatError(f"Could not read MAT file {path}: {exc}") from exc

    missing = [k for k in ("digits_vec", "digits_labels", "trainset", "testset") if k not in mat]
    if missing:
        raise DataFormatError(f"MAT file is missing variables: {missing}")

    X = np.asarray(mat["digits_vec"], dtype=np.float32).T  # (4000, 784), row = sample
    y = np.asarray(mat["digits_labels"], dtype=np.int64).ravel()  # (4000,)
    trials = {}
    for trial, row in ((1, 0), (2, 1)):
        trials[trial] = TrialSplit(
            train_idx=to_zero_based(np.asarray(mat["trainset"][row], dtype=np.int64)),
            test_idx=to_zero_based(np.asarray(mat["testset"][row], dtype=np.int64)),
        )
    ds = DigitDataset(X=X, y=y, trials=trials, source_files={path.name: sha256_file(path)})
    validate_dataset(ds)
    return ds


def validate_dataset(
    ds: DigitDataset,
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    samples_per_class: int = SAMPLES_PER_CLASS,
    trial_size: int = TRIAL_SIZE,
) -> None:
    """Check every documented property of the dataset; raise on any violation.

    This is intentionally strict: mismatches must surface as readable errors
    instead of silently producing wrong results.  The expected constants are
    parameters so tests can validate small synthetic datasets with the same
    rules.
    """
    if ds.X.shape != (n_samples, n_features):
        raise DataFormatError(f"Expected X shape ({n_samples}, {n_features}), got {ds.X.shape}.")
    if ds.y.shape != (n_samples,):
        raise DataFormatError(f"Expected y shape ({n_samples},), got {ds.y.shape}.")
    if not np.all(np.isfinite(ds.X)):
        raise DataFormatError("X contains non-finite values.")
    if not np.all(ds.X == np.round(ds.X)):
        raise DataFormatError("X is expected to contain integer pixel values.")
    if ds.X.min() < 0 or ds.X.max() > 255:
        raise DataFormatError(
            f"Pixel values out of [0, 255]: min={ds.X.min()}, max={ds.X.max()}."
        )
    if not np.all(np.isin(ds.y, np.arange(n_classes))):
        raise DataFormatError(f"Labels outside 0..{n_classes - 1} found.")
    counts = np.bincount(ds.y, minlength=n_classes)
    if not np.all(counts == samples_per_class):
        raise DataFormatError(f"Per-class counts are not all {samples_per_class}: {counts}.")

    for trial in (1, 2):
        if trial not in ds.trials:
            raise DataFormatError(f"Trial {trial} missing from the split files.")
        split = ds.trials[trial]
        tr, te = split.train_idx, split.test_idx
        if tr.shape != (trial_size,) or te.shape != (trial_size,):
            raise DataFormatError(
                f"Trial {trial}: expected ({trial_size},) indices, "
                f"got train {tr.shape} / test {te.shape}."
            )
        if len(np.unique(tr)) != trial_size or len(np.unique(te)) != trial_size:
            raise DataFormatError(f"Trial {trial}: duplicate indices inside a split.")
        if len(np.intersect1d(tr, te)) != 0:
            raise DataFormatError(f"Trial {trial}: train/test indices overlap.")
        union = np.union1d(tr, te)
        if union.shape != (n_samples,) or not np.array_equal(union, np.arange(n_samples)):
            raise DataFormatError(
                f"Trial {trial}: train ∪ test does not cover 0..{n_samples - 1} exactly."
            )
        for name, idx in (("train", tr), ("test", te)):
            counts = np.bincount(ds.y[idx], minlength=n_classes)
            if not np.all(counts == trial_size // n_classes):
                raise DataFormatError(
                    f"Trial {trial} {name} set is not class-balanced "
                    f"({trial_size // n_classes}/class): {counts}."
                )


def dataset_summary(ds: DigitDataset) -> str:
    """Human-readable summary of the loaded dataset (for logging)."""
    lines = [
        f"X shape: {ds.X.shape} (rows = samples), dtype {ds.X.dtype}, "
        f"pixel range [{ds.X.min():.0f}, {ds.X.max():.0f}], "
        f"non-zero ratio {np.count_nonzero(ds.X) / ds.X.size:.3f}",
        f"y shape: {ds.y.shape}, labels present: {sorted(np.unique(ds.y).tolist())}",
    ]
    for trial, split in ds.trials.items():
        lines.append(
            f"Trial {trial}: train {split.train_idx.shape[0]} / test {split.test_idx.shape[0]} "
            f"samples, disjoint={len(np.intersect1d(split.train_idx, split.test_idx)) == 0}, "
            f"union covers all={np.array_equal(np.union1d(split.train_idx, split.test_idx), np.arange(N_SAMPLES))}"
        )
    return "\n".join(lines)


def load_data(data_dir: str | Path) -> DigitDataset:
    """Load the dataset from ``data_dir``.

    Uses the four TXT files when present; otherwise falls back to a
    ``digits4000.mat`` file in the same directory (optional support).
    """
    p = Path(data_dir).expanduser().resolve()
    if all((p / f).is_file() for f in _ALL_TXT_FILES) or (p / "digits4000_txt").is_dir():
        return load_txt(p)
    mat_candidates = [p / "digits4000.mat", p / "digits4000_txt" / "digits4000.mat"]
    for mat_path in mat_candidates:
        if mat_path.is_file():
            return load_mat(mat_path)
    raise DataFormatError(
        f"No digits4000 data found under {p}: expected the TXT files "
        f"{_ALL_TXT_FILES} or digits4000.mat."
    )
