"""M-split Conformalized Quantile Regression.

Two XGBoost quantile regressors (pinball loss) target the ``alpha/2`` and
``1 - alpha/2`` conditional quantiles; the levels are derived from ``alpha``,
never hard-coded. The signed score ``max(lo - y, y - hi)`` is calibrated with
the same m-split scheme as SCP (independent shuffle sequence via the seed),
and the deployed correction widens both quantile predictions.
"""

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from xopa.conformal.scp import ConformalResult, _multi_split
from xopa.data import Splits


def monotonicize_quantile_bounds(
    lower: Any, upper: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order independently fitted quantiles pointwise and flag crossings.

    Quantile regressors do not guarantee non-crossing predictions. Ordering
    the two fitted quantiles before nonconformity scoring preserves a valid
    lower/upper interval while retaining a crossing mask for diagnostics.
    """
    raw_lower = np.asarray(lower, dtype=float)
    raw_upper = np.asarray(upper, dtype=float)
    if raw_lower.ndim != 1 or raw_upper.ndim != 1:
        raise ValueError("CQR quantile predictions must be one-dimensional")
    if raw_lower.shape != raw_upper.shape:
        raise ValueError("CQR lower and upper predictions must have equal shapes")
    if not np.isfinite(raw_lower).all() or not np.isfinite(raw_upper).all():
        raise ValueError("CQR quantile predictions must be finite")
    crossed = raw_lower > raw_upper
    return (
        np.minimum(raw_lower, raw_upper),
        np.maximum(raw_lower, raw_upper),
        crossed,
    )


def fit_quantile_models(
    splits: Splits,
    model_params: dict[str, Any],
    alpha: float = 0.1,
    seed: int = 0,
    early_stopping_rounds: int | None = 20,
) -> tuple[XGBRegressor, XGBRegressor]:
    """Fit the lower and upper pinball-loss XGBoost quantile regressors.

    Both share the point model's hyperparameters, train on the training split,
    and early-stop on the validation split.

    Args:
        splits: The fixed partition.
        model_params: XGBoost constructor parameters (no objective inside).
        alpha: Miscoverage level; quantile levels are alpha/2 and 1 - alpha/2.
        seed: random_state of both regressors.
        early_stopping_rounds: Patience on the validation split.

    Returns:
        The fitted (lower, upper) quantile models.
    """
    models = []
    for level in (alpha / 2.0, 1.0 - alpha / 2.0):
        model = XGBRegressor(
            **model_params,
            objective="reg:quantileerror",
            quantile_alpha=level,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            splits.X_train,
            splits.y_train,
            eval_set=[(splits.X_val, splits.y_val)],
            verbose=False,
        )
        models.append(model)
    return models[0], models[1]


def fit_mcqr(
    lo_model,
    hi_model,
    splits: Splits,
    *,
    alpha: float = 0.1,
    m: int = 2200,
    calib_fraction: float = 0.8,
    seed: int = 0,
) -> ConformalResult:
    """Calibrate m-CQR for a fitted pair of quantile models.

    Args:
        lo_model: Fitted lower-quantile regressor.
        hi_model: Fitted upper-quantile regressor.
        splits: The fixed partition; only the calibration split is read here.
        alpha: Target miscoverage.
        m: Number of calibration shuffles.
        calib_fraction: Sub-calibration share of each shuffle.
        seed: Seed of the shuffle sequence (use a different one than SCP so
            the two methods see independent permutations).

    Returns:
        A :class:`ConformalResult` with ``method="cqr"``.
    """
    y = splits.y_calib.to_numpy()
    lo, hi, _ = monotonicize_quantile_bounds(
        lo_model.predict(splits.X_calib), hi_model.predict(splits.X_calib)
    )
    scores = np.maximum(lo - y, y - hi)
    thresholds, coverages, n_cal, split_seeds, masks = _multi_split(
        scores, alpha, m, calib_fraction, seed
    )
    return ConformalResult(
        method="cqr",
        alpha=alpha,
        m=m,
        calib_fraction=calib_fraction,
        q_hat=float(np.mean(thresholds)),
        thresholds=thresholds,
        coverages=coverages,
        n_calib=n_cal,
        seed=seed,
        scores=np.asarray(scores, dtype=float),
        calibration_row_ids=splits.X_calib.index.to_numpy(),
        split_seeds=split_seeds,
        calibration_masks=masks,
        params={"n_calib_total": len(scores)},
    )


def predict_mcqr(
    lo_model, hi_model, x: pd.DataFrame, result: ConformalResult
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive-width intervals: corrected quantile predictions."""
    lower, upper, _ = monotonicize_quantile_bounds(
        lo_model.predict(x), hi_model.predict(x)
    )
    return (
        lower - result.q_hat,
        upper + result.q_hat,
    )
