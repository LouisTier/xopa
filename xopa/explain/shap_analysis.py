"""SHAP attribution utilities.

Attributions are safe to compute over the full dataset once model parameters
are frozen: the model's reasoning is identical on training and test data, so
restricting explanations to the test set is unnecessarily conservative.
"""

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from xopa.data import CATEGORY_RANGES, feature_category
from xopa.models import TREE_BACKBONES


def shap_matrix(
    model, x: pd.DataFrame, backbone: str, max_evals: int | None = None
) -> pd.DataFrame:
    """Per-sample, per-feature SHAP values for a fitted model.

    Tree backbones use the exact TreeExplainer. The PyTorch MLP uses
    DeepExplainer on its fitted network with a deterministic background.
    Unknown non-tree estimators fall back to permutation SHAP.

    Args:
        model: Fitted estimator.
        x: Data to attribute, columns matching the fitting features.
        backbone: Model family name, decides the explainer.
        max_evals: Budget for the permutation explainer (default: shap's).

    Returns:
        DataFrame of signed SHAP values, same index and columns as ``x``.
    """
    if backbone in TREE_BACKBONES:
        explanation = shap.TreeExplainer(model)(x, check_additivity=False)
        values = np.asarray(explanation.values)
    elif backbone == "mlp" and hasattr(model, "model_"):
        import torch

        model.model_.eval()
        background_size = min(100, len(x))
        background_positions = np.linspace(0, len(x) - 1, background_size, dtype=int)
        background = torch.as_tensor(
            x.iloc[background_positions].to_numpy(dtype=np.float32, copy=True)
        )
        explainer = shap.DeepExplainer(model.model_, background)
        chunks = []
        for start in range(0, len(x), 2048):
            batch = torch.as_tensor(
                x.iloc[start : start + 2048].to_numpy(dtype=np.float32, copy=True)
            )
            raw = explainer.shap_values(batch, check_additivity=False)
            if isinstance(raw, list):
                raw = raw[0]
            chunk = np.asarray(raw)
            if chunk.ndim == 3 and chunk.shape[-1] == 1:
                chunk = chunk[..., 0]
            chunks.append(chunk)
        values = np.concatenate(chunks, axis=0)
    else:
        masker = shap.maskers.Independent(x, max_samples=100)
        explainer = shap.PermutationExplainer(model.predict, masker)
        explanation = explainer(x, max_evals=max_evals) if max_evals else explainer(x)
        values = np.asarray(explanation.values)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def global_importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending."""
    return shap_values.abs().mean(axis=0).sort_values(ascending=False)


def drift_by_index(
    shap_values: pd.DataFrame, features: list[str], group_size: int = 1500
) -> pd.DataFrame:
    """Mean signed SHAP per consecutive record-index group (drift heatmap).

    Rows are ``features``; columns are consecutive groups of ``group_size``
    records in the order of ``shap_values.index``.
    """
    n = len(shap_values)
    groups = np.arange(n) // group_size
    labels = [
        f"{g * group_size}-{min((g + 1) * group_size, n) - 1}"
        for g in range(int(groups.max()) + 1)
    ]
    grouped = shap_values[features].groupby(groups).mean()
    grouped.index = labels
    return grouped.T


def drift_by_output(
    shap_values: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    bin_width: float = 0.5,
) -> pd.DataFrame:
    """Mean signed SHAP per output-value bin (quality-bin heatmap)."""
    edges = np.arange(
        np.floor(y.min() / bin_width) * bin_width,
        np.ceil(y.max() / bin_width) * bin_width + bin_width,
        bin_width,
    )
    bins = pd.cut(y.to_numpy(), edges, include_lowest=True)
    grouped = shap_values[features].groupby(bins, observed=True).mean()
    grouped.index = [f"{interval.mid:.2f}" for interval in grouped.index]
    return grouped.T


def rank_stability(
    shap_values: pd.DataFrame, features: list[str], group_size: int = 1500
) -> pd.Series:
    """Spearman correlation of per-group importance ranks vs the global ranks.

    One value per record-index group; high values mean the importance ordering
    of ``features`` is stable across the dataset.
    """
    global_rank = shap_values[features].abs().mean(axis=0).rank()
    n = len(shap_values)
    groups = np.arange(n) // group_size
    out = {}
    for g, block in shap_values[features].groupby(groups):
        local_rank = block.abs().mean(axis=0).rank()
        out[int(g)] = float(spearmanr(global_rank, local_rank).statistic)
    return pd.Series(out, name="spearman_vs_global")


def top_k_persistence(
    shap_values: pd.DataFrame,
    features: list[str],
    group_size: int = 1500,
    k: int = 5,
) -> pd.Series:
    """Fraction of the global top-k features retained in each record group."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if not features:
        return pd.Series(dtype=float, name=f"top_{k}_persistence")
    k_effective = min(k, len(features))
    global_importance = shap_values[features].abs().mean(axis=0)
    global_top = set(
        global_importance.sort_values(ascending=False, kind="mergesort")
        .index[:k_effective]
        .tolist()
    )
    groups = np.arange(len(shap_values)) // group_size
    persistence = {}
    for group, block in shap_values[features].groupby(groups):
        local_importance = block.abs().mean(axis=0)
        local_top = set(
            local_importance.sort_values(ascending=False, kind="mergesort")
            .index[:k_effective]
            .tolist()
        )
        persistence[int(group)] = len(global_top & local_top) / k_effective
    return pd.Series(persistence, name=f"top_{k}_persistence")


def monotonicity(drift_output: pd.DataFrame) -> pd.Series:
    """Spearman correlation of mean SHAP vs bin center, per feature.

    Values near +1 or -1 indicate a strong monotonic association between the
    target-bin center and the fitted contribution. Input is
    :func:`drift_by_output`.
    """
    centers = drift_output.columns.astype(float)
    return pd.Series(
        {
            feature: float(spearmanr(centers, row.to_numpy()).statistic)
            for feature, row in drift_output.iterrows()
        },
        name="monotonicity",
    )


def explanation_diagnostics(
    shap_values: pd.DataFrame,
    y: pd.Series,
    *,
    group_size: int = 1500,
    bin_width: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Compute rank stability and output monotonicity over every feature."""
    if shap_values.shape[1] == 0:
        raise ValueError("shap_values must contain at least one feature")
    if len(shap_values) != len(y):
        raise ValueError("shap_values and y must contain the same number of records")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    features = list(shap_values.columns)
    rank_agreement = rank_stability(shap_values, features, group_size)
    output_drift = drift_by_output(shap_values, features, y, bin_width)
    return rank_agreement, monotonicity(output_drift)


def category_summary(features: list[str], importance: pd.Series) -> pd.DataFrame:
    """Per-category selected-feature counts and share of total |SHAP|.

    Args:
        features: The selected feature set.
        importance: Mean |SHAP| per feature (e.g. :func:`global_importance`),
            covering at least ``features``.

    Returns:
        DataFrame indexed by category with columns ``n_selected`` and
        ``share_abs_shap`` (the latter sums to 1 over the selected set).
    """
    total = float(importance.loc[features].sum())
    rows = dict.fromkeys(CATEGORY_RANGES, None)
    for category in rows:
        members = [f for f in features if feature_category(f) == category]
        share = float(importance.loc[members].sum()) / total if total > 0 else 0.0
        rows[category] = {"n_selected": len(members), "share_abs_shap": share}
    return pd.DataFrame.from_dict(rows, orient="index")
