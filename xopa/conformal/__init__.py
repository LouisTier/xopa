"""Uncertainty quantification: m-SCP, m-CQR, ensemble baseline, metrics.

- :mod:`xopa.conformal.scp`: M-split Split Conformal Prediction:
  absolute-residual scores calibrated over m random shuffles of the
  calibration split.
- :mod:`xopa.conformal.cqr`: M-split Conformalized Quantile Regression:
  two pinball-loss quantile regressors plus the same m-split calibration.
- :mod:`xopa.conformal.ensemble`: the bootstrap ensemble baseline (no coverage
  guarantee; width multiplier calibrated on the calibration split).
- :mod:`xopa.conformal.metrics`: PICP, MPIW, WCC, and conditional coverage.

No model is ever fitted, selected, or tuned on the calibration split; this
subpackage is its only methodological consumer (post-fit reporting, such as
full-dataset attributions, covers all splits).
"""

from xopa.conformal.cqr import (
    fit_mcqr,
    fit_quantile_models,
    monotonicize_quantile_bounds,
    predict_mcqr,
)
from xopa.conformal.ensemble import ENSResult, fit_ens, predict_ens
from xopa.conformal.metrics import mpiw, picp, picp_by_slice, running_average, wcc
from xopa.conformal.scp import ConformalResult, fit_mscp, predict_mscp

__all__ = [
    "ConformalResult",
    "ENSResult",
    "fit_ens",
    "fit_mcqr",
    "fit_mscp",
    "fit_quantile_models",
    "monotonicize_quantile_bounds",
    "mpiw",
    "picp",
    "picp_by_slice",
    "predict_ens",
    "predict_mcqr",
    "predict_mscp",
    "running_average",
    "wcc",
]
