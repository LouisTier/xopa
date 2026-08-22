"""Configuration, seeds, run directories, and run metadata."""

import json
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def pool_map(
    worker: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    initializer: Callable[[dict[str, Any]], None],
    context: dict[str, Any],
    n_jobs: int,
) -> list[Any]:
    """Map ``worker`` over ``items``, sequentially or across worker processes.

    ``initializer(context)`` runs once in the parent (sequential path) or once
    per worker process, so ``worker`` reads its shared inputs from module
    state instead of pickling them per item. Processes are used instead of
    threads so a backbone drawing from a process-global random state (the
    torch MLP: weight initialization and dropout masks) reproduces the
    sequential streams exactly; results are identical for any ``n_jobs``.
    """
    initializer(context)
    if n_jobs <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(items)),
        initializer=initializer,
        initargs=(context,),
    ) as pool:
        return list(pool.map(worker, items))


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file into a plain mapping."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return config


def spawn_seeds(base_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent child seeds from one base seed.

    Uses :class:`numpy.random.SeedSequence` spawning, so the same child seeds
    can be regenerated from the same base seed.
    """
    children = np.random.SeedSequence(base_seed).spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]


def timestamp_dir(root: str | Path, label: str) -> Path:
    """Create and return ``<root>/<YYYY-MM-DD_HH-MM-SS>_<label>/``."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(root) / f"{stamp}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_common_artifacts(out_dir: str | Path, dataset, splits) -> None:
    """Record the shared data and split artifacts of one run.

    Writes ``split_assignments.csv`` (row id to split label),
    ``preprocessing.npz`` (feature names and train-fitted feature/target
    transformations), and ``dataset_identity.json`` (source hash, shapes,
    dropped columns).
    """
    out_dir = Path(out_dir)
    blocks = (
        ("train", splits.X_train),
        ("validation", splits.X_val),
        ("calibration", splits.X_calib),
        ("test", splits.X_test),
    )
    rows = [
        {"row_id": row_id, "split": label, "split_position": position}
        for label, frame in blocks
        for position, row_id in enumerate(frame.index)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "split_assignments.csv", index=False)

    np.savez_compressed(
        out_dir / "preprocessing.npz",
        feature_names=np.asarray(list(splits.X_train.columns)),
        mean=splits.scaler.mean_,
        scale=splits.scaler.scale_,
        target_name=np.asarray(dataset.identity.target),
        target_standardized=np.asarray(splits.target_standardized),
        target_mean=np.asarray(splits.target_mean),
        target_scale=np.asarray(splits.target_scale),
    )

    identity = asdict(dataset.identity)
    identity["source_path"] = str(identity["source_path"])
    (out_dir / "dataset_identity.json").write_text(
        json.dumps(identity, indent=2, default=str) + "\n"
    )


_MANIFEST_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "shap",
    "mrmr-selection",
    "scipy",
    "torch",
)


def _git_output(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_provenance(root: str | Path | None = None) -> dict:
    """Identify the exact source snapshot a run used.

    Records the git head and dirty state when available, plus a content hash
    of the package, script, and configuration files that works on deployments
    without git history.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = sha256()
    files = sorted(
        path
        for pattern in ("xopa/**/*.py", "scripts/*.py", "configs/*.yaml")
        for path in root.glob(pattern)
    )
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    status = _git_output(root, "status", "--porcelain")
    return {
        "git_head": _git_output(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "source_sha256": digest.hexdigest(),
        "n_source_files": len(files),
    }


def write_manifest(
    out_dir: str | Path,
    config: dict,
    extra: dict | None = None,
    *,
    stage: str | None = None,
    run_kind: str | None = None,
    seeds: dict | None = None,
) -> Path:
    """Write ``manifest.json`` with configuration and run metadata.

    The manifest records the resolved configuration, command, source and
    environment metadata, package versions, and any seed map supplied by the
    calling stage. Stage-specific artifacts may contain additional operation
    seeds and resampling details.

    Args:
        out_dir: Run directory the manifest is written into.
        config: Fully resolved configuration of the run.
        extra: Optional additional entries (e.g. resolved CLI values).
        stage: Semantic pipeline stage.
        run_kind: ``paper`` (authors' validated campaigns) or ``user``.
        seeds: Optional seed metadata supplied by the calling stage.

    Returns:
        Path of the written manifest file.
    """
    versions: dict[str, str] = {}
    for package in _MANIFEST_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover
            versions[package] = "not installed"

    out_dir = Path(out_dir)
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "package_versions": versions,
        "source_provenance": source_provenance(),
        "config": config,
        "seeds": seeds or {},
        **(extra or {}),
    }
    if stage is not None:
        manifest["stage"] = stage
    if run_kind is not None:
        manifest["run_kind"] = run_kind
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
