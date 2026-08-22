"""Explainability: SHAP attributions and their stability diagnostics.

- :mod:`xopa.explain.shap_analysis`: per-sample SHAP matrices (exact
  TreeExplainer for tree backbones, DeepExplainer for the MLP), global
  importance, record-index and output-bin drift, rank stability, top-k
  persistence, monotonicity, and the per-category importance summary.

Attributions are computed over the full dataset: with model parameters frozen
the model's reasoning is identical on training and test data, so restricting
explanations to the test set is unnecessarily conservative.
"""

from xopa.explain.shap_analysis import (
    category_summary,
    drift_by_index,
    drift_by_output,
    explanation_diagnostics,
    global_importance,
    monotonicity,
    rank_stability,
    shap_matrix,
    top_k_persistence,
)

__all__ = [
    "category_summary",
    "drift_by_index",
    "drift_by_output",
    "explanation_diagnostics",
    "global_importance",
    "monotonicity",
    "rank_stability",
    "shap_matrix",
    "top_k_persistence",
]
