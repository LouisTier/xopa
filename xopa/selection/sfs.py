"""SHAP Feature Selection (SFS): ranked accumulation with a validation sweep.

The survivors of FFS are ranked once by global SHAP importance computed on the
training set, then a model is fitted on every top-k
prefix and evaluated on the validation split. ``d_plus`` is the validation
optimum; ``d_parsimonious`` is the smallest k within a relative tolerance of
it. Fresh synthetic probes ride along in the ranking model so their mean
absolute SHAP value provides the noise-floor line: features scoring below it
are not distinguishable from noise, which is the principled account of the
gap between the optimal and parsimonious subset sizes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from xopa.data import Splits
from xopa.explain.shap_analysis import shap_matrix
from xopa.models import fit_model, make_model
from xopa.selection.probes import ProbeConfig, add_probes, is_probe
from xopa.utils import pool_map, spawn_seeds

logger = logging.getLogger(__name__)

#: Shared inputs of the sweep workers, set by :func:`xopa.utils.pool_map` once
#: per run in the parent and once per worker process.
_SFS_CONTEXT: dict[str, Any] = {}


def _set_sfs_context(context: dict[str, Any]) -> None:
    _SFS_CONTEXT.update(context)


def _sweep_fit(item: tuple[int, int, int]) -> dict[str, Any]:
    k, repeat, r_seed = item
    c = _SFS_CONTEXT
    subset = c["order"][:k]
    model = fit_model(
        make_model(c["backbone"], c["model_params"], r_seed),
        c["x_train"][subset],
        c["y_train"],
        c["x_val"][subset],
        c["y_val"],
        early_stopping_rounds=c["early_stopping_rounds"],
    )
    prediction = np.asarray(model.predict(c["x_val"][subset]), dtype=float)
    y_true = c["y_val"].to_numpy(dtype=float)
    return {
        "k": k,
        "repeat": repeat,
        "seed": r_seed,
        "prediction": prediction,
        "mse": float(mean_squared_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


@dataclass
class SFSResult:
    """SHAP ordering, validation sweep curve, and the two selected sizes."""

    order: list[str]  # real features, by descending mean |SHAP|
    importance: pd.Series  # mean |SHAP| per feature (probes included)
    probe_floor: float | None  # max probe mean |SHAP|, None without probes
    curve: pd.DataFrame  # index k: mse_mean, mse_std, r2_mean, r2_std
    d_plus: int  # validation optimum (argmin mean MSE)
    d_parsimonious: int  # smallest k within (1 + epsilon) of the optimum
    repeat_metrics: pd.DataFrame  # one row per k and model seed
    validation_predictions: np.ndarray  # shape (k, repeat, validation row)
    validation_row_ids: np.ndarray
    validation_y_true: np.ndarray
    ranking_shap_values: pd.DataFrame
    ranking_feature_values: pd.DataFrame
    ranking_row_ids: np.ndarray
    ranking_predictions: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


def run_sfs(
    splits: Splits,
    features: list[str],
    *,
    backbone: str = "xgboost",
    model_params: dict[str, Any],
    probes: ProbeConfig | None = None,
    seed: int = 42,
    n_repeats: int = 5,
    epsilon: float = 0.02,
    shap_subsample: int | None = None,
    early_stopping_rounds: int | None = 20,
    n_jobs: int = 1,
) -> SFSResult:
    """Rank ``features`` by SHAP on the training set and sweep top-k on val.

    Args:
        splits: The fixed partition (train fits, val evaluates; never calib).
        features: The FFS survivor set to refine.
        backbone: Model family for both the ranking model and the sweep fits.
        model_params: Constructor parameters for the backbone.
        probes: Optional probe configuration for the noise-floor line.
        seed: Base seed (per-repeat seeds are spawned from it).
        n_repeats: Model seeds per k, for the variability band.
        epsilon: Relative MSE tolerance defining ``d_parsimonious``.
        shap_subsample: Rows used for the SHAP ranking (None: full train for
            tree backbones; strongly recommended for the MLP).
        early_stopping_rounds: Patience of the sweep fits (validation split).
        n_jobs: Worker processes for the sweep fits. Intended for
            single-threaded backbones (the MLP); keep 1 for tree backbones.
            Results are identical for any value.

    Returns:
        An :class:`SFSResult`.
    """
    rng = np.random.default_rng(seed)
    x_train = splits.X_train[features]
    x_ranking = x_train
    if probes is not None:
        x_ranking, _ = add_probes(x_train, probes, rng)

    x_shap = x_ranking
    if shap_subsample is not None and shap_subsample < len(x_ranking):
        keep = rng.choice(len(x_ranking), size=shap_subsample, replace=False)
        x_shap = x_ranking.iloc[keep]

    ranking_model = fit_model(
        make_model(backbone, model_params, seed),
        x_ranking,
        splits.y_train,
        early_stopping_rounds=None,
    )
    shap_values = shap_matrix(ranking_model, x_shap, backbone)
    ranking_predictions = np.asarray(ranking_model.predict(x_shap), dtype=float)
    importance = shap_values.abs().mean(axis=0)

    probe_floor = None
    if probes is not None:
        probe_floor = float(
            importance[[c for c in importance.index if is_probe(c)]].max()
        )
    real = importance[[c for c in importance.index if not is_probe(c)]]
    order = list(real.sort_values(ascending=False).index)

    repeat_seeds = spawn_seeds(seed, n_repeats)
    rows = []
    repeat_rows: list[dict[str, Any]] = []
    validation_predictions = np.empty(
        (len(order), n_repeats, len(splits.X_val)), dtype=float
    )
    validation_y_true = splits.y_val.to_numpy(dtype=float)
    sweep_inputs = [
        (k, repeat, r_seed)
        for k in range(1, len(order) + 1)
        for repeat, r_seed in enumerate(repeat_seeds)
    ]

    context = {
        "x_train": splits.X_train[order],
        "y_train": splits.y_train,
        "x_val": splits.X_val[order],
        "y_val": splits.y_val,
        "order": order,
        "backbone": backbone,
        "model_params": model_params,
        "early_stopping_rounds": early_stopping_rounds,
    }
    sweep_results = pool_map(
        _sweep_fit,
        sweep_inputs,
        initializer=_set_sfs_context,
        context=context,
        n_jobs=n_jobs,
    )

    for result in sweep_results:
        k = int(result["k"])
        repeat = int(result["repeat"])
        validation_predictions[k - 1, repeat] = result["prediction"]
        repeat_rows.append(
            {
                "k": k,
                "repeat": repeat + 1,
                "seed": result["seed"],
                "mse": result["mse"],
                "r2": result["r2"],
            }
        )

    for k in range(1, len(order) + 1):
        current = [row for row in repeat_rows if row["k"] == k]
        mse = [float(row["mse"]) for row in current]
        r2 = [float(row["r2"]) for row in current]
        rows.append(
            {
                "k": k,
                "mse_mean": float(np.mean(mse)),
                "mse_std": float(np.std(mse)),
                "r2_mean": float(np.mean(r2)),
                "r2_std": float(np.std(r2)),
            }
        )
        logger.info("sfs sweep k=%d/%d done", k, len(order))

    curve = pd.DataFrame(rows).set_index("k")
    d_plus = int(curve["mse_mean"].idxmin())
    tolerance = (1.0 + epsilon) * float(curve.loc[d_plus, "mse_mean"])
    d_parsimonious = int(curve.index[curve["mse_mean"] <= tolerance].min())

    params = {
        "features": list(features),
        "backbone": backbone,
        "model_params": dict(model_params),
        "probes": probes.__dict__ if probes else None,
        "seed": seed,
        "repeat_seeds": repeat_seeds,
        "n_repeats": n_repeats,
        "epsilon": epsilon,
        "shap_subsample": shap_subsample,
        "n_jobs": n_jobs,
    }
    return SFSResult(
        order=order,
        importance=importance,
        probe_floor=probe_floor,
        curve=curve,
        d_plus=d_plus,
        d_parsimonious=d_parsimonious,
        repeat_metrics=pd.DataFrame(repeat_rows),
        validation_predictions=validation_predictions,
        validation_row_ids=splits.X_val.index.to_numpy(),
        validation_y_true=validation_y_true,
        ranking_shap_values=shap_values,
        ranking_feature_values=x_shap,
        ranking_row_ids=x_shap.index.to_numpy(),
        ranking_predictions=ranking_predictions,
        params=params,
    )
