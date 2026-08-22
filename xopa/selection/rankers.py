"""Ranking operators with a nested-prefix contract.

Every ranker returns an ordered feature list whose first K entries are exactly
the operator's selection at size K, for every K in ``checkpoints``. One
ranking per (subsample, operator) therefore serves an entire d-grid by prefix,
so changing the grid never requires re-running a simulation.

How each operator satisfies the contract:

- ``rfe``: recursive elimination along a canonical schedule (feature counts
  descend through multiples of ``step``), so the path does not depend on which
  checkpoints are requested. Ordering is the reverse elimination order.
- ``mrmr``: forward-greedy selection; prefixes are exact by construction.
- ``lasso``: features ranked by entry order along the regularization path.
- ``tree``: one model fit, features ranked by importance.

Rankers fit models plainly (no early stopping): the operator sees only the
data it is given, which inside FFS is a subsample of the training set.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from mrmr import mrmr_regression
from sklearn.linear_model import lasso_path

from xopa.models import fit_model, model_importances

ModelFactory = Callable[[], Any]


def _validate_checkpoints(checkpoints: list[int] | None, k_max: int) -> list[int]:
    checkpoints = sorted(set(checkpoints or [k_max]))
    if checkpoints[-1] > k_max:
        raise ValueError(f"checkpoint {checkpoints[-1]} exceeds k_max={k_max}")
    return checkpoints


def rank_rfe(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    k_max: int,
    seed: int,
    checkpoints: list[int] | None = None,
    step: int = 10,
) -> list[str]:
    """Recursive feature elimination driven by the backbone's importances.

    The elimination schedule is canonical: the feature count descends through
    the multiples of ``step`` (first drop lands on the largest multiple below
    the feature count), stopping at the smallest checkpoint. Every checkpoint
    must be a multiple of ``step`` so it lies on the schedule; this is what
    guarantees the nested-prefix property across different checkpoint lists.

    Args:
        x: Feature matrix (subsample of the training set inside FFS).
        y: Target aligned with ``x``.
        model_factory: Zero-argument callable returning a fresh unfitted model.
        k_max: Length of the returned ranking.
        seed: Seed for permutation importances (models seed via the factory).
        checkpoints: Sizes at which prefixes must be exact (default: [k_max]).
        step: Elimination granularity; all checkpoints must be multiples of it.

    Returns:
        Ordered feature list of length ``k_max``.
    """
    checkpoints = _validate_checkpoints(checkpoints, k_max)
    misaligned = [c for c in checkpoints if c % step != 0]
    if misaligned:
        raise ValueError(
            f"checkpoints {misaligned} are not multiples of step={step}; "
            "the canonical elimination schedule cannot land on them"
        )
    k_floor = checkpoints[0]

    current = list(x.columns)
    eliminated: list[str] = []  # least important first, in elimination order

    def importances(features: list[str]) -> np.ndarray:
        model = fit_model(model_factory(), x[features], y, early_stopping_rounds=None)
        return model_importances(model, x[features], y, kind="auto", seed=seed)

    while len(current) > k_floor:
        imp = importances(current)
        next_size = max(k_floor, (len(current) - 1) // step * step)
        n_drop = len(current) - next_size
        drop_positions = np.argsort(imp, kind="stable")[:n_drop]
        dropped = [current[i] for i in drop_positions]
        eliminated.extend(dropped)
        kept = set(current) - set(dropped)
        current = [f for f in current if f in kept]

    final_imp = importances(current)
    survivors = [current[i] for i in np.argsort(-final_imp, kind="stable")]
    ranking = survivors + eliminated[::-1]
    return ranking[:k_max]


def rank_mrmr(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    k_max: int,
    seed: int,
    checkpoints: list[int] | None = None,
) -> list[str]:
    """Minimum-redundancy maximum-relevance forward selection.

    F-test relevance, Pearson-correlation redundancy, mean denominator.
    Forward-greedy, hence nested by construction.
    """
    _validate_checkpoints(checkpoints, k_max)
    selected = mrmr_regression(
        X=x,
        y=y,
        K=k_max,
        relevance="f",
        redundancy="c",
        denominator="mean",
        return_scores=False,
        show_progress=False,
        n_jobs=1,
    )
    return list(selected)


def rank_lasso(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    k_max: int,
    seed: int,
    checkpoints: list[int] | None = None,
) -> list[str]:
    """LASSO-path ranking: order of entry along the regularization path.

    A feature's rank is set by the first (largest) alpha at which its
    coefficient becomes nonzero; features that never enter are ordered by
    their absolute coefficient at the smallest alpha. Backbone-free.
    """
    _validate_checkpoints(checkpoints, k_max)
    alphas, coefs, _ = lasso_path(x.to_numpy(), y.to_numpy())
    active = coefs != 0.0  # shape (n_features, n_alphas), alphas descending
    n_alphas = len(alphas)
    entry_index = np.where(active.any(axis=1), active.argmax(axis=1), n_alphas)
    final_abs = np.abs(coefs[:, -1])
    order = np.lexsort((-final_abs, entry_index))
    columns = list(x.columns)
    return [columns[i] for i in order][:k_max]


def rank_tree(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    k_max: int,
    seed: int,
    checkpoints: list[int] | None = None,
) -> list[str]:
    """One backbone fit, features ranked by importance (descending)."""
    _validate_checkpoints(checkpoints, k_max)
    model = fit_model(model_factory(), x, y, early_stopping_rounds=None)
    imp = model_importances(model, x, y, kind="auto", seed=seed)
    order = np.argsort(-imp, kind="stable")
    columns = list(x.columns)
    return [columns[i] for i in order][:k_max]


RANKERS: dict[str, Callable[..., list[str]]] = {
    "rfe": rank_rfe,
    "mrmr": rank_mrmr,
    "lasso": rank_lasso,
    "tree": rank_tree,
}
