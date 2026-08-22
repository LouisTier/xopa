"""Frequency Feature Selection (FFS).

Q subsamples of the fixed training set are drawn without replacement (ratio
``rho``); the ranking operator Psi runs once per subsample at the largest grid
size, and every grid value K is read off as a prefix (see
:mod:`xopa.selection.rankers` for the nested-prefix contract). A feature
survives at K when it appears in at least ``ceil(tau * Q)`` of the Q prefixes;
equality selects (the explicit tie-break rule).

Synthetic probes, when enabled, compete for prefix slots like real features.
Their selection frequency is the noise floor: ``tau="auto"`` derives the
threshold from it instead of asserting one.
"""

import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import pandas as pd

from xopa.data import Splits
from xopa.models import evaluate, fit_model, make_model
from xopa.selection.probes import ProbeConfig, add_probes, is_probe
from xopa.selection.rankers import RANKERS
from xopa.utils import pool_map, spawn_seeds

logger = logging.getLogger(__name__)

#: Shared inputs of the fit workers, set by :func:`xopa.utils.pool_map` once
#: per run in the parent and once per worker process.
_FFS_CONTEXT: dict[str, Any] = {}


def _set_ffs_context(context: dict[str, Any]) -> None:
    _FFS_CONTEXT.update(context)


def _run_iteration(item: tuple[int, int]) -> dict[str, Any]:
    q, q_seed = item
    c = _FFS_CONTEXT
    rng = np.random.default_rng(q_seed)
    subsample = rng.choice(len(c["x_train"]), size=c["n_sub"], replace=False)
    x_q, y_q = c["x_train"].iloc[subsample], c["y_train"].iloc[subsample]
    iteration_subsamples = [
        {
            "operator": c["operator"],
            "iteration": q + 1,
            "seed": q_seed,
            "row_id": row_id,
            "subsample_position": position,
        }
        for position, row_id in enumerate(x_q.index)
    ]
    if c["probes"] is not None:
        x_q, drawn = add_probes(x_q, c["probes"], rng)
        if sorted(drawn) != sorted(c["probe_names"]):  # pragma: no cover
            raise RuntimeError("probe naming drifted from configuration")

    factory = partial(make_model, c["backbone"], c["model_params"], q_seed)
    ranker_kwargs: dict[str, Any] = {}
    if c["operator"] == "rfe":
        ranker_kwargs["step"] = c["rfe_step"]
    ranking = RANKERS[c["operator"]](
        x_q,
        y_q,
        factory,
        k_max=c["k_max"],
        seed=q_seed,
        checkpoints=c["grid"],
        **ranker_kwargs,
    )
    return {
        "iteration": q,
        "seed": q_seed,
        "ranking": ranking,
        "subsamples": iteration_subsamples,
    }


def _evaluate_grid_cell(item: tuple[int, tuple[str, ...]]) -> dict[str, Any]:
    k, feature_tuple = item
    c = _FFS_CONTEXT
    kept = list(feature_tuple)
    model = fit_model(
        make_model(c["backbone"], c["model_params"], c["seed"]),
        c["x_train"][kept],
        c["y_train"],
        c["x_val"][kept],
        c["y_val"],
        early_stopping_rounds=c["early_stopping_rounds"],
    )
    metrics = evaluate(model, c["x_val"][kept], c["y_val"])
    prediction = np.asarray(model.predict(c["x_val"][kept]), dtype=float)
    return {
        "K": k,
        "features": kept,
        "metrics": metrics,
        "prediction": prediction,
    }


@dataclass
class FFSResult:
    """Everything one FFS run produces, per grid value K."""

    operator: str
    counts: pd.DataFrame  # real features x K: appearances in the top-K prefix
    probe_counts: pd.DataFrame  # probe names x K: probe appearances
    probe_ranks: pd.DataFrame  # probe names x q: rank position (NaN if unranked)
    survivors: dict[int, list[str]]  # K -> features with count >= ceil(tau * Q)
    tau_used: dict[int, float]  # K -> resolved threshold
    grid_eval: pd.DataFrame  # index K: d_star, val_mse, val_r2
    rankings: pd.DataFrame  # one row per ranked feature and iteration
    subsamples: pd.DataFrame  # source-row membership of every training draw
    validation_predictions: pd.DataFrame  # row-level predictions for each K
    params: dict[str, Any] = field(default_factory=dict)


def survivors_from_counts(counts: pd.Series, q: int, tau: float) -> list[str]:
    """Features appearing in at least ``ceil(tau * q)`` prefixes.

    Equality selects: a feature exactly at the threshold is kept. This is the
    explicit tie-break rule.
    """
    threshold = math.ceil(tau * q)
    return [name for name in counts.index if counts[name] >= threshold]


def auto_tau(probe_counts: pd.Series, q: int) -> float:
    """Data-determined threshold: just above the worst probe's frequency.

    Returns ``(max probe count + 1) / q``, or ``1 / q`` when no probe was ever
    ranked. May exceed 1 when a probe is selected in every iteration, in which
    case no feature is distinguishable from noise at this K and the survivor
    set is legitimately empty.
    """
    if len(probe_counts) == 0 or probe_counts.max() == 0:
        return 1.0 / q
    return (float(probe_counts.max()) + 1.0) / q


def run_ffs(
    splits: Splits,
    operator: str,
    grid: list[int],
    *,
    q_iterations: int = 10,
    rho: float = 0.7,
    tau: float | Literal["auto"] = 0.8,
    probes: ProbeConfig | None = None,
    backbone: str = "xgboost",
    model_params: dict[str, Any],
    seed: int = 42,
    rfe_step: int = 10,
    early_stopping_rounds: int | None = 20,
    n_jobs: int = 1,
) -> FFSResult:
    """Run FFS for one operator over a d-grid.

    Args:
        splits: The fixed four-way partition (only train and val are used).
        operator: Key into :data:`xopa.selection.rankers.RANKERS`.
        grid: Selection sizes K; every K is served by one ranking per
            subsample via the nested-prefix contract.
        q_iterations: Number of subsample draws Q.
        rho: Subsample ratio, drawn without replacement (0.5 < rho <= 1).
        tau: Frequency threshold, or ``"auto"`` to derive it per K from the
            probe frequencies.
        probes: Synthetic probe configuration; None disables probes
            (required for ``tau="auto"``).
        backbone: Model family driving model-based operators and the grid
            evaluation fits.
        model_params: Constructor parameters for the backbone.
        seed: Base seed; per-iteration seeds are spawned from it.
        rfe_step: Elimination granularity for the ``rfe`` operator.
        early_stopping_rounds: Patience for the evaluation fits (monitored on
            the validation split; ranking fits never early-stop).
        n_jobs: Worker processes for the iteration and evaluation fit loops.
            Intended for single-threaded backbones (the MLP); keep 1 for tree
            backbones, whose fits already use all cores. Results are identical
            for any value.

    Returns:
        An :class:`FFSResult` with counts, survivors, probe diagnostics, and
        the per-K validation evaluation.
    """
    if not 0.5 < rho <= 1.0:
        raise ValueError(f"rho must be in (0.5, 1], got {rho}")
    if tau == "auto" and probes is None:
        raise ValueError('tau="auto" requires probes')
    grid = sorted(set(grid), reverse=True)
    k_max = grid[0]

    x_train, y_train = splits.X_train, splits.y_train
    n_sub = math.floor(rho * len(x_train))
    iteration_seeds = spawn_seeds(seed, q_iterations)

    counts = pd.DataFrame(0, index=list(x_train.columns), columns=grid)
    probe_names: list[str] = []
    if probes is not None:
        probe_names = [f"__probe_gauss{i}" for i in range(probes.n_gaussian)] + [
            f"__probe_perm{i}" for i in range(probes.n_permuted)
        ]
    probe_counts = pd.DataFrame(0, index=probe_names, columns=grid)
    probe_ranks = pd.DataFrame(np.nan, index=probe_names, columns=range(q_iterations))
    ranking_rows: list[dict[str, Any]] = []
    subsample_rows: list[dict[str, Any]] = []
    validation_prediction_rows: list[dict[str, Any]] = []

    context = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": splits.X_val,
        "y_val": splits.y_val,
        "n_sub": n_sub,
        "probes": probes,
        "probe_names": probe_names,
        "operator": operator,
        "grid": grid,
        "k_max": k_max,
        "rfe_step": rfe_step,
        "backbone": backbone,
        "model_params": model_params,
        "seed": seed,
        "early_stopping_rounds": early_stopping_rounds,
    }
    iteration_inputs = list(enumerate(iteration_seeds))
    iteration_results = pool_map(
        _run_iteration,
        iteration_inputs,
        initializer=_set_ffs_context,
        context=context,
        n_jobs=n_jobs,
    )

    for iteration in iteration_results:
        q = int(iteration["iteration"])
        q_seed = int(iteration["seed"])
        ranking = list(iteration["ranking"])
        subsample_rows.extend(iteration["subsamples"])

        for position, name in enumerate(ranking):
            ranking_rows.append(
                {
                    "operator": operator,
                    "iteration": q + 1,
                    "seed": q_seed,
                    "rank": position + 1,
                    "feature": name,
                    "is_probe": is_probe(name),
                    "probe_kind": (
                        "gaussian"
                        if name.startswith("__probe_gauss")
                        else "permuted"
                        if name.startswith("__probe_perm")
                        else "none"
                    ),
                    **{f"in_top_{k}": position < k for k in grid},
                }
            )
            if is_probe(name):
                probe_ranks.loc[name, q] = position
                for k in grid:
                    if position < k:
                        probe_counts.loc[name, k] += 1
            else:
                for k in grid:
                    if position < k:
                        counts.loc[name, k] += 1
        logger.info("ffs[%s] iteration %d/%d done", operator, q + 1, q_iterations)

    survivors: dict[int, list[str]] = {}
    tau_used: dict[int, float] = {}
    rows_by_k: dict[int, dict[str, float | int]] = {}
    evaluation_inputs: list[tuple[int, tuple[str, ...]]] = []
    for k in grid:
        tau_k = auto_tau(probe_counts[k], q_iterations) if tau == "auto" else float(tau)
        kept = survivors_from_counts(counts[k], q_iterations, tau_k)
        survivors[k], tau_used[k] = kept, tau_k
        if kept:
            evaluation_inputs.append((k, tuple(kept)))
        else:
            logger.warning(
                "ffs[%s] K=%d: empty survivor set (tau=%.2f)", operator, k, tau_k
            )
            rows_by_k[k] = {
                "K": k,
                "d_star": 0,
                "val_mse": np.nan,
                "val_r2": np.nan,
            }

    evaluation_results = pool_map(
        _evaluate_grid_cell,
        evaluation_inputs,
        initializer=_set_ffs_context,
        context=context,
        n_jobs=n_jobs,
    )

    for result in evaluation_results:
        k = int(result["K"])
        kept = list(result["features"])
        metrics = result["metrics"]
        prediction = np.asarray(result["prediction"], dtype=float)
        y_true = splits.y_val.to_numpy(dtype=float)
        validation_prediction_rows.extend(
            {
                "operator": operator,
                "K": k,
                "d_star": len(kept),
                "model_seed": seed,
                "row_id": row_id,
                "y_true": truth,
                "prediction": pred,
                "residual": truth - pred,
            }
            for row_id, truth, pred in zip(
                splits.X_val.index, y_true, prediction, strict=True
            )
        )
        rows_by_k[k] = {
            "K": k,
            "d_star": len(kept),
            "val_mse": metrics["mse"],
            "val_r2": metrics["r2"],
        }

    rows = [rows_by_k[k] for k in grid]

    grid_eval = pd.DataFrame(rows).set_index("K")
    params = {
        "operator": operator,
        "grid": grid,
        "q_iterations": q_iterations,
        "rho": rho,
        "tau": tau,
        "probes": probes.__dict__ if probes else None,
        "backbone": backbone,
        "model_params": dict(model_params),
        "seed": seed,
        "iteration_seeds": iteration_seeds,
        "rfe_step": rfe_step,
        "n_jobs": n_jobs,
    }
    return FFSResult(
        operator=operator,
        counts=counts,
        probe_counts=probe_counts,
        probe_ranks=probe_ranks,
        survivors=survivors,
        tau_used=tau_used,
        grid_eval=grid_eval,
        rankings=pd.DataFrame(ranking_rows),
        subsamples=pd.DataFrame(subsample_rows),
        validation_predictions=pd.DataFrame(
            validation_prediction_rows,
            columns=(
                "operator",
                "K",
                "d_star",
                "model_seed",
                "row_id",
                "y_true",
                "prediction",
                "residual",
            ),
        ),
        params=params,
    )
