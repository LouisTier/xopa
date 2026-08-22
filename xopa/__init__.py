"""xopa: explainable offline process analytics for tabular regression.

The package is three root modules and three method subpackages:

- :mod:`xopa.data`: dataset loading, the fixed train/val/calib/test split,
  feature categories, and dataset-identity validation.
- :mod:`xopa.models`: the four backbones (XGBoost, random forest, LightGBM,
  PyTorch MLP) behind one factory, fitting, and evaluation interface.
- :mod:`xopa.selection`: RFFS feature selection (FFS + SFS), the ranking
  operators, synthetic noise probes, and the standalone Boruta baseline.
- :mod:`xopa.conformal`: m-SCP and m-CQR multi-split conformal prediction,
  a bootstrap ensemble baseline, and interval-quality metrics.
- :mod:`xopa.explain`: SHAP attributions with drift, stability, and
  per-category importance analyses.
- :mod:`xopa.utils`: configuration, seeds, run directories, shared artifacts,
  and run metadata.

The split policy is strict throughout: train fits, validation selects and
monitors early stopping, no model is ever fitted, selected, or tuned on the
calibration split (only :mod:`xopa.conformal` consumes it; post-fit reporting
such as full-dataset attributions covers all splits), and test is touched
once.
"""

__version__ = "0.1.0"
