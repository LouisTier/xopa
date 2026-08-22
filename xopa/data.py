"""Dataset loading, the four-way split, and feature-category utilities.

The split policy is strict:

- ``train`` fits models,
- ``val`` selects features and parameters (and monitors early stopping),
- ``calib`` is consumed exclusively by :mod:`xopa.conformal`,
- ``test`` assesses final performance, once.

No training or selection function accepts the calibration split; outside
:mod:`xopa.conformal` it appears only in post-fit reporting (full-dataset
attributions, recorded split assignments).
"""

import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Inclusive ``data_NNN`` suffix bounds per feature category (domain knowledge).
#: The target ``data_164`` belongs to no category. On the reference industrial
#: dataset the per-category totals are CTX 4, PV 99, RMC 37, WC 22, PRS 154.
CATEGORY_RANGES: dict[str, tuple[int, int]] = {
    "CTX": (1, 4),
    "PV": (5, 103),
    "RMC": (104, 140),
    "WC": (141, 162),
    "PRS": (183, 609),
}

DEFAULT_TARGET = "data_164"
DEFAULT_RATIOS = (0.70, 0.075, 0.075, 0.15)

_SUFFIX_RE = re.compile(r"^data_(\d+)$")


@dataclass(frozen=True)
class DatasetIdentity:
    """Content identity and cleaning decisions for one source dataset."""

    source_path: Path
    sha256: str
    byte_size: int
    raw_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    target: str
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    dtypes: dict[str, str]
    dropped_index_columns: tuple[str, ...]
    dropped_subquality_columns: tuple[str, ...]
    dropped_constant_columns: tuple[str, ...]


@dataclass(frozen=True)
class LoadedDataset:
    """Clean feature/target data paired with its immutable source identity."""

    x: pd.DataFrame
    y: pd.Series
    identity: DatasetIdentity


@dataclass(frozen=True)
class DatasetContract:
    """Fail-closed identity a known dataset must match exactly."""

    sha256: str
    raw_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    target: str
    feature_schema_sha256: str | None
    dropped_index_columns: tuple[str, ...]
    dropped_subquality_count: int
    dropped_constant_count: int


CANDCE_2026_DATASET_CONTRACT = DatasetContract(
    sha256="8c2aaf58314872bc35535255cdb35b36624e68e7cedcc102b69565797201012e",
    raw_shape=(35125, 331),
    cleaned_shape=(35125, 316),
    target=DEFAULT_TARGET,
    feature_schema_sha256=(
        "0d06fb8925ed966c6ad515ab87f0291e24c4a88d12cbfb476d24557c53cc9ea9"
    ),
    dropped_index_columns=("Unnamed: 0",),
    dropped_subquality_count=13,
    dropped_constant_count=0,
)


def feature_schema_sha256(feature_names: tuple[str, ...] | list[str]) -> str:
    """Return a stable hash for an ordered feature-name sequence."""
    payload = "\0".join(str(name) for name in feature_names).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_dataset_identity(
    identity: DatasetIdentity, contract: DatasetContract
) -> None:
    """Reject a loaded dataset that differs from a recorded contract."""
    errors: list[str] = []
    observed = (
        ("source sha256", identity.sha256, contract.sha256),
        ("raw shape", identity.raw_shape, contract.raw_shape),
        ("cleaned shape", identity.cleaned_shape, contract.cleaned_shape),
        ("target", identity.target, contract.target),
        (
            "dropped index columns",
            identity.dropped_index_columns,
            contract.dropped_index_columns,
        ),
        (
            "dropped subquality count",
            len(identity.dropped_subquality_columns),
            contract.dropped_subquality_count,
        ),
        (
            "dropped constant count",
            len(identity.dropped_constant_columns),
            contract.dropped_constant_count,
        ),
    )
    for label, actual, expected in observed:
        if actual != expected:
            errors.append(f"{label}={actual!r}, expected {expected!r}")
    if (
        contract.feature_schema_sha256 is not None
        and identity.feature_schema_sha256 != contract.feature_schema_sha256
    ):
        errors.append(
            "feature schema sha256="
            f"{identity.feature_schema_sha256!r}, expected "
            f"{contract.feature_schema_sha256!r}"
        )
    forbidden = [
        name
        for name in identity.feature_names
        if name == identity.target
        or name.startswith("target_")
        or name.startswith("Unnamed")
        or name.startswith("__probe_")
    ]
    if forbidden:
        errors.append(f"forbidden accepted feature names={forbidden!r}")
    if errors:
        raise ValueError("dataset contract mismatch: " + "; ".join(errors))


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_bundle(
    path: str | Path, target: str = DEFAULT_TARGET
) -> LoadedDataset:
    """Load clean data and record input identity without copying source values."""
    path = Path(path).resolve(strict=True)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    raw_shape = tuple(df.shape)
    raw_dtypes = {str(name): str(dtype) for name, dtype in df.dtypes.items()}

    index_cols = [str(c) for c in df.columns if str(c).startswith("Unnamed")]
    subquality_cols = [str(c) for c in df.columns if str(c).startswith("target_")]
    df = df.drop(columns=index_cols + subquality_cols)

    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in {path}")
    y = df[target].astype(float)
    x = df.drop(columns=[target])

    constant = [str(name) for name in x.columns[x.nunique() <= 1].tolist()]
    if constant:
        logger.info("dropping %d constant columns: %s", len(constant), constant)
        x = x.drop(columns=constant)

    identity = DatasetIdentity(
        source_path=path,
        sha256=_sha256_path(path),
        byte_size=path.stat().st_size,
        raw_shape=(int(raw_shape[0]), int(raw_shape[1])),
        cleaned_shape=(int(x.shape[0]), int(x.shape[1])),
        target=target,
        feature_names=tuple(str(name) for name in x.columns),
        feature_schema_sha256=feature_schema_sha256(list(x.columns)),
        dtypes=raw_dtypes,
        dropped_index_columns=tuple(index_cols),
        dropped_subquality_columns=tuple(subquality_cols),
        dropped_constant_columns=tuple(constant),
    )
    logger.info("loaded %s: %d samples, %d features", path.name, len(x), x.shape[1])
    return LoadedDataset(x=x, y=y, identity=identity)


def load_dataset(
    path: str | Path, target: str = DEFAULT_TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a feature matrix and target from a CSV or Parquet file.

    Cleaning steps: drop any serialized index column (``Unnamed: *``), drop the
    sparse ``target_*`` sub-quality columns, drop constant feature columns
    (reported via logging), and split the target out of the features.

    Args:
        path: Path to the dataset file (``.csv`` or ``.parquet``).
        target: Name of the target column.

    Returns:
        Tuple ``(X, y)`` with the raw (unstandardized) features and target.
    """
    bundle = load_dataset_bundle(path, target=target)
    return bundle.x, bundle.y


@dataclass(frozen=True)
class Splits:
    """The fixed four-way partition and its train-fitted transformations."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_calib: pd.DataFrame
    y_calib: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler
    target_standardized: bool
    target_mean: float
    target_scale: float
    seed: int
    ratios: tuple[float, float, float, float]


def make_splits(
    x: pd.DataFrame,
    y: pd.Series,
    ratios: tuple[float, float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    standardize_target: bool = False,
) -> Splits:
    """Draw the fixed partition and fit requested transformations on training.

    A single random permutation (seeded) is sliced by the cumulative ratios.
    The feature scaler is fit on the training block only and applied to all
    four. Target standardization is optional and follows the same fit policy.

    Args:
        x: Raw feature matrix.
        y: Target series aligned with ``x``.
        ratios: Fractions for (train, val, calib, test); must sum to 1.
        seed: Seed of the single permutation defining the partition.
        standardize_target: Whether to center and scale every target split with
            the mean and population standard deviation of the training target.

    Returns:
        A :class:`Splits` object with standardized feature frames.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    if len(x) != len(y):
        raise ValueError("x and y have different lengths")

    n = len(x)
    order = np.random.default_rng(seed).permutation(n)
    bounds = np.cumsum([round(r * n) for r in ratios[:-1]])
    parts = np.split(order, bounds)

    frames: list[pd.DataFrame] = [x.iloc[p] for p in parts]
    targets: list[pd.Series] = [y.iloc[p] for p in parts]

    scaler = StandardScaler().fit(frames[0])
    standardized = [
        pd.DataFrame(scaler.transform(f), index=f.index, columns=f.columns)
        for f in frames
    ]

    target_mean = 0.0
    target_scale = 1.0
    transformed_targets = targets
    if standardize_target:
        target_mean = float(targets[0].mean())
        target_scale = float(targets[0].std(ddof=0))
        if not np.isfinite(target_scale) or target_scale <= 0.0:
            raise ValueError(
                "training target must have a finite, positive standard deviation"
            )
        transformed_targets = [
            (target - target_mean) / target_scale for target in targets
        ]

    return Splits(
        X_train=standardized[0],
        y_train=transformed_targets[0],
        X_val=standardized[1],
        y_val=transformed_targets[1],
        X_calib=standardized[2],
        y_calib=transformed_targets[2],
        X_test=standardized[3],
        y_test=transformed_targets[3],
        scaler=scaler,
        target_standardized=standardize_target,
        target_mean=target_mean,
        target_scale=target_scale,
        seed=seed,
        ratios=ratios,
    )


def feature_category(name: str) -> str:
    """Map a ``data_NNN`` feature name to its category.

    Raises:
        ValueError: If the name does not parse or falls in no category
            (notably the target ``data_164``).
    """
    match = _SUFFIX_RE.match(name)
    if match is None:
        raise ValueError(f"not a data_NNN feature name: {name!r}")
    suffix = int(match.group(1))
    for category, (low, high) in CATEGORY_RANGES.items():
        if low <= suffix <= high:
            return category
    raise ValueError(f"feature {name!r} falls in no category")


def category_counts(features: list[str]) -> dict[str, int]:
    """Count selected features per category (categories with zero included)."""
    counts = dict.fromkeys(CATEGORY_RANGES, 0)
    for name in features:
        counts[feature_category(name)] += 1
    return counts


def exclude_categories(
    x: pd.DataFrame, categories: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Drop every feature belonging to the given categories.

    Used for confound analyses (e.g. repeating the pipeline without the PRS
    features that encode recipes/grades). Exclusion happens
    after dataset-identity validation, so the loaded dataset is still the
    verified one; the dropped columns are recorded by the caller.

    Args:
        x: Feature matrix.
        categories: Category names from :data:`CATEGORY_RANGES`.

    Returns:
        Tuple of the reduced matrix and the dropped column names.

    Raises:
        ValueError: On unknown category names, when nothing matches, or when
            the exclusion would empty the feature matrix.
    """
    unknown = sorted(set(categories) - set(CATEGORY_RANGES))
    if unknown:
        raise ValueError(
            f"unknown categories {unknown}; known: {sorted(CATEGORY_RANGES)}"
        )
    selected = set(categories)
    dropped = [name for name in x.columns if feature_category(name) in selected]
    if not dropped:
        raise ValueError(f"no features belong to categories {sorted(selected)}")
    if len(dropped) == x.shape[1]:
        raise ValueError("excluding these categories would drop every feature")
    return x.drop(columns=dropped), dropped
