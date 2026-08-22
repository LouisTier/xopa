"""Synthetic probe features for noise-floor diagnostics.

A probe is a column known a priori to carry no information about the target:
either iid Gaussian noise or a permuted copy of a real feature (which keeps a
realistic marginal distribution while destroying any association with the
target, the Boruta/knockoff idea). Probes compete with real features inside
the ranking operators; how often and how highly they rank is the noise floor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PROBE_PREFIX = "__probe_"


@dataclass(frozen=True)
class ProbeConfig:
    """Number of synthetic probe columns to inject per draw."""

    n_gaussian: int = 1
    n_permuted: int = 1


def add_probes(
    x: pd.DataFrame, config: ProbeConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of ``x`` with fresh probe columns appended.

    Args:
        x: Feature matrix (real features only).
        config: How many probes of each kind to add.
        rng: Source of randomness (redrawn probes require a fresh state).

    Returns:
        Tuple of the augmented matrix and the probe column names.
    """
    columns: dict[str, np.ndarray] = {}
    for i in range(config.n_gaussian):
        columns[f"{PROBE_PREFIX}gauss{i}"] = rng.normal(size=len(x))
    real = list(x.columns)
    for i in range(config.n_permuted):
        source = real[int(rng.integers(len(real)))]
        columns[f"{PROBE_PREFIX}perm{i}"] = rng.permutation(x[source].to_numpy())

    augmented = x.copy()
    for name, values in columns.items():
        augmented[name] = values
    return augmented, list(columns)


def is_probe(name: str) -> bool:
    """Whether a column name denotes a synthetic probe."""
    return name.startswith(PROBE_PREFIX)
