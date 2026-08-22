"""Bootstrap ensemble intervals.

Included for comparison-figure parity only: unlike SCP and CQR the ensemble
carries no distribution-free coverage guarantee, its variance is a heuristic
proxy for uncertainty. Members are selected on the validation split; the width
multiplier ``z`` is the one legitimate non-conformal use of the calibration
split, and it lives inside :mod:`xopa.conformal` for exactly that reason.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xopa.data import Splits
from xopa.models import evaluate, fit_model, make_model
from xopa.utils import spawn_seeds

logger = logging.getLogger(__name__)


@dataclass
class ENSResult:
    """Selected bootstrap members and the calibrated width multiplier."""

    models: list[Any]
    z: float
    target_coverage: float
    member_seeds: np.ndarray
    bootstrap_positions: np.ndarray
    validation_mse: np.ndarray
    selected_indices: np.ndarray
    validation_predictions: np.ndarray
    calibration_predictions: np.ndarray
    test_predictions: np.ndarray
    test_lower: np.ndarray
    test_upper: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


def fit_ens(
    splits: Splits,
    model_params: dict[str, Any],
    *,
    backbone: str = "xgboost",
    n_models: int = 15,
    n_best: int = 10,
    target_coverage: float = 0.9,
    seed: int = 0,
    early_stopping_rounds: int | None = 20,
) -> ENSResult:
    """Train bootstrap members and calibrate the interval multiplier.

    ``n_models`` members are trained on bootstrap resamples of the training
    split (early-stopped on validation); the ``n_best`` by validation MSE are
    kept. ``z`` is the smallest value on a fine grid whose ``mean +/- z * std``
    intervals reach ``target_coverage`` on the calibration split.

    Args:
        splits: The fixed partition.
        model_params: Constructor parameters for the backbone.
        backbone: Model family of the members.
        n_models: Number of bootstrap members to train.
        n_best: Members kept, ranked by validation MSE.
        target_coverage: Coverage the calibrated multiplier must reach.
        seed: Base seed (member seeds are spawned from it).
        early_stopping_rounds: Patience on the validation split.

    Returns:
        An :class:`ENSResult`.
    """
    member_seeds = spawn_seeds(seed, n_models)
    n_train = len(splits.X_train)
    bootstrap_positions = np.empty((n_models, n_train), dtype=np.int64)
    for member_index, member_seed in enumerate(member_seeds):
        rng = np.random.default_rng(member_seed)
        bootstrap_positions[member_index] = rng.integers(0, n_train, size=n_train)

    member_inputs = [
        (member_index, member_seed, bootstrap_positions[member_index])
        for member_index, member_seed in enumerate(member_seeds)
    ]

    def fit_member(item: tuple[int, int, np.ndarray]) -> dict[str, Any]:
        member_index, member_seed, resample = item
        model = fit_model(
            make_model(backbone, model_params, member_seed),
            splits.X_train.iloc[resample],
            splits.y_train.iloc[resample],
            splits.X_val,
            splits.y_val,
            early_stopping_rounds=early_stopping_rounds,
        )
        return {
            "member_index": member_index,
            "model": model,
            "validation_mse": evaluate(model, splits.X_val, splits.y_val)["mse"],
            "validation_prediction": model.predict(splits.X_val),
            "calibration_prediction": model.predict(splits.X_calib),
            "test_prediction": model.predict(splits.X_test),
        }

    member_results = [fit_member(item) for item in member_inputs]

    all_models = [result["model"] for result in member_results]
    validation_mse = [result["validation_mse"] for result in member_results]

    validation_mse_array = np.asarray(validation_mse, dtype=float)
    selected_indices = np.argsort(validation_mse_array, kind="stable")[:n_best]
    models = [all_models[index] for index in selected_indices]
    validation_predictions = np.stack(
        [result["validation_prediction"] for result in member_results]
    )
    calibration_predictions = np.stack(
        [result["calibration_prediction"] for result in member_results]
    )
    test_predictions = np.stack(
        [result["test_prediction"] for result in member_results]
    )

    selected_calibration = calibration_predictions[selected_indices]
    mean, std = selected_calibration.mean(axis=0), selected_calibration.std(axis=0)
    y = splits.y_calib.to_numpy()
    z_grid = np.arange(0.5, 10.0, 0.01)
    z = float(z_grid[-1])
    for candidate in z_grid:
        covered = np.mean((y >= mean - candidate * std) & (y <= mean + candidate * std))
        if covered >= target_coverage:
            z = float(candidate)
            break
    else:  # pragma: no cover
        logger.warning("z grid exhausted; using z=%.2f", z)

    selected_test = test_predictions[selected_indices]
    test_mean, test_std = selected_test.mean(axis=0), selected_test.std(axis=0)
    test_lower = test_mean - z * test_std
    test_upper = test_mean + z * test_std

    params = {
        "backbone": backbone,
        "n_models": n_models,
        "n_best": n_best,
        "member_seeds": member_seeds,
        "seed": seed,
    }
    return ENSResult(
        models=models,
        z=z,
        target_coverage=target_coverage,
        member_seeds=np.asarray(member_seeds, dtype=np.uint64),
        bootstrap_positions=bootstrap_positions,
        validation_mse=validation_mse_array,
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        validation_predictions=np.asarray(validation_predictions, dtype=float),
        calibration_predictions=np.asarray(calibration_predictions, dtype=float),
        test_predictions=np.asarray(test_predictions, dtype=float),
        test_lower=np.asarray(test_lower, dtype=float),
        test_upper=np.asarray(test_upper, dtype=float),
        params=params,
    )


def predict_ens(result: ENSResult, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Intervals ``mean +/- z * std`` over the selected members."""
    predictions = np.stack([m.predict(x) for m in result.models])
    mean, std = predictions.mean(axis=0), predictions.std(axis=0)
    return mean - result.z * std, mean + result.z * std
