"""M-split Split Conformal Prediction.

The point model is trained upstream (never here). Nonconformity scores are the
absolute residuals on the calibration split, computed once; the calibration
step is repeated over ``m`` random 80/20 shuffles of the calibration set, and
the deployed threshold is the mean of the per-split thresholds.

Caveat: averaging thresholds across overlapping shuffles
forfeits the single-split finite-sample coverage guarantee. The M-split scheme
is a variance-reduction heuristic that empirically preserves near-nominal
coverage; the per-split evaluation coverages it records are the diagnostics.

This module and its siblings are the only consumers of the calibration split.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xopa.data import Splits
from xopa.utils import spawn_seeds


@dataclass
class ConformalResult:
    """Deployed threshold plus per-split diagnostics of one m-split run."""

    method: str  # "scp" or "cqr"
    alpha: float
    m: int
    calib_fraction: float
    q_hat: float  # deployed threshold: mean of per-split thresholds
    thresholds: np.ndarray  # shape (m,)
    coverages: np.ndarray  # per-split coverage on the evaluation folds
    n_calib: int  # sub-calibration fold size
    seed: int
    scores: np.ndarray  # fixed calibration nonconformity scores
    calibration_row_ids: np.ndarray  # fixed order used by scores and masks
    split_seeds: np.ndarray  # independent derived seed per shuffle
    calibration_masks: np.ndarray  # shape (m, n_calib_total), True=sub-calibration
    params: dict[str, Any] = field(default_factory=dict)


def _split_threshold(scores: np.ndarray, alpha: float, n_cal: int) -> float:
    """Finite-sample-corrected empirical quantile of one calibration fold.

    The ``ceil((1 - alpha) * (n_cal + 1))``-th order statistic; infinity when
    the fold is too small for that order statistic to exist.
    """
    k = math.ceil((1.0 - alpha) * (n_cal + 1))
    if k > n_cal:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def _multi_split(
    scores: np.ndarray,
    alpha: float,
    m: int,
    calib_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Run the m-split loop on precomputed scores.

    Returns per-split thresholds, per-split evaluation coverages, and the
    sub-calibration fold size. Coverage on the evaluation fold uses the score
    equivalence ``y in interval  <=>  score <= threshold``.
    """
    n = len(scores)
    n_cal = math.floor(calib_fraction * n)
    if not 0 < n_cal < n:
        raise ValueError(
            f"calib_fraction={calib_fraction} leaves no usable folds (n={n})"
        )

    thresholds = np.empty(m)
    coverages = np.empty(m)
    split_seeds = np.asarray(spawn_seeds(seed, m), dtype=np.uint64)
    calibration_masks = np.zeros((m, n), dtype=bool)
    for i, split_seed in enumerate(split_seeds):
        rng = np.random.default_rng(int(split_seed))
        order = rng.permutation(n)
        calibration_masks[i, order[:n_cal]] = True
        cal_scores = scores[order[:n_cal]]
        eval_scores = scores[order[n_cal:]]
        q = _split_threshold(cal_scores, alpha, n_cal)
        thresholds[i] = q
        coverages[i] = float(np.mean(eval_scores <= q))
    return thresholds, coverages, n_cal, split_seeds, calibration_masks


def fit_mscp(
    model,
    splits: Splits,
    *,
    alpha: float = 0.1,
    m: int = 2200,
    calib_fraction: float = 0.8,
    seed: int = 0,
) -> ConformalResult:
    """Calibrate m-SCP for a fitted point model.

    Args:
        model: Fitted point predictor (trained on train, early-stopped on val).
        splits: The fixed partition; only the calibration split is read here.
        alpha: Target miscoverage (0.1 gives 90% nominal coverage).
        m: Number of calibration shuffles.
        calib_fraction: Sub-calibration share of each shuffle.
        seed: Seed of the shuffle sequence.

    Returns:
        A :class:`ConformalResult` with ``method="scp"``.
    """
    scores = np.abs(splits.y_calib.to_numpy() - model.predict(splits.X_calib))
    thresholds, coverages, n_cal, split_seeds, masks = _multi_split(
        scores, alpha, m, calib_fraction, seed
    )
    return ConformalResult(
        method="scp",
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


def predict_mscp(
    model, x: pd.DataFrame, result: ConformalResult
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-width intervals: point prediction plus/minus the threshold."""
    pred = model.predict(x)
    return pred - result.q_hat, pred + result.q_hat
