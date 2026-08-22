"""RFFS orchestration: FFS grid search, SFS refinement, consolidated report.

Ranked Frequency Feature Selection is the composition of FFS and SFS. This
module runs FFS for every requested operator over the shared d-grid, compares
cells validation MSE and retained dimension, refines the winner's survivors
with SFS, and only then touches the test split, once.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xopa.data import Splits, category_counts
from xopa.explain.shap_analysis import category_summary
from xopa.models import evaluate, fit_model, make_model
from xopa.selection.ffs import FFSResult, run_ffs
from xopa.selection.ffs_winner import ffs_winner_rule, select_ffs_winner
from xopa.selection.probes import ProbeConfig
from xopa.selection.sfs import SFSResult, run_sfs

logger = logging.getLogger(__name__)


@dataclass
class RFFSReport:
    """One backbone's full RFFS campaign result."""

    backbone: str
    ffs: dict[str, FFSResult]  # operator -> FFS result over the grid
    winner: tuple[str, int]  # selected preliminary (operator, K)
    sfs: SFSResult  # refinement of the winner's survivors
    final_features: list[str]  # top d_plus of the SFS order
    parsimonious_features: list[str]  # top d_parsimonious
    test_metrics: dict[str, dict[str, float]]  # per feature set, computed once
    category_table: pd.DataFrame  # counts + |SHAP| share for final_features
    overlap: pd.DataFrame  # Jaccard between all selected sets
    test_predictions: pd.DataFrame  # row-level evidence for final evaluations
    selected_feature_sets: dict[str, list[str]]  # every grid survivor and final set
    params: dict[str, Any] = field(default_factory=dict)


def jaccard_overlap(sets: dict[str, list[str]]) -> pd.DataFrame:
    """Pairwise Jaccard similarity between named feature sets."""
    names = list(sets)
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = set(sets[a]), set(sets[b])
            union = len(sa | sb)
            value = len(sa & sb) / union if union else 1.0
            matrix.loc[a, b] = matrix.loc[b, a] = value
    return matrix


def run_rffs(
    splits: Splits,
    operators: list[str],
    grid: list[int],
    *,
    backbone: str = "xgboost",
    model_params: dict[str, Any],
    q_iterations: int = 10,
    rho: float = 0.7,
    tau: float | str = 0.8,
    probes: ProbeConfig | None = None,
    seed: int = 42,
    rfe_step: int = 10,
    sfs_n_repeats: int = 5,
    sfs_epsilon: float = 0.02,
    shap_subsample: int | None = None,
    early_stopping_rounds: int | None = 20,
    n_jobs: int = 1,
) -> RFFSReport:
    """Run the full RFFS campaign for one backbone.

    Args:
        splits: The fixed partition (test is touched once, at the very end).
        operators: Psi operators to compare over the grid.
        grid: Selection sizes K, shared by every operator.
        backbone: Model family (drives selection, refinement, evaluation).
        model_params: Constructor parameters for the backbone.
        q_iterations: FFS subsample draws Q.
        rho: FFS subsample ratio.
        tau: Frequency threshold or ``"auto"``.
        probes: Synthetic probe configuration (None disables).
        seed: Base seed for all derived randomness.
        rfe_step: Elimination granularity of the ``rfe`` operator.
        sfs_n_repeats: Model seeds per k in the SFS sweep.
        sfs_epsilon: Relative tolerance defining the parsimonious size.
        shap_subsample: Rows used for the SFS SHAP ranking.
        early_stopping_rounds: Patience of evaluation fits (validation split).
        n_jobs: Worker processes for the FFS and SFS fit loops. Intended for
            single-threaded backbones (the MLP); keep 1 for tree backbones.
            Results are identical for any value.

    Returns:
        An :class:`RFFSReport`.
    """
    ffs_results: dict[str, FFSResult] = {}
    for operator in operators:
        logger.info("rffs[%s]: running FFS with operator %s", backbone, operator)
        ffs_results[operator] = run_ffs(
            splits,
            operator,
            grid,
            q_iterations=q_iterations,
            rho=rho,
            tau=tau,
            probes=probes,
            backbone=backbone,
            model_params=model_params,
            seed=seed,
            rfe_step=rfe_step,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
        )

    candidates = []
    for operator, result in ffs_results.items():
        for k, row in result.grid_eval.iterrows():
            if row["d_star"] > 0 and pd.notna(row["val_mse"]):
                candidates.append(
                    {
                        "operator": operator,
                        "K": int(k),
                        "d_star": int(row["d_star"]),
                        "val_mse": float(row["val_mse"]),
                    }
                )
    if not candidates:
        raise RuntimeError("every (operator, K) cell has an empty survivor set")
    best_operator, best_k = select_ffs_winner(pd.DataFrame(candidates))
    winner = (best_operator, best_k)
    winner_survivors = ffs_results[best_operator].survivors[best_k]
    logger.info(
        "rffs[%s]: winner %s at K=%d (d_star=%d)",
        backbone,
        best_operator,
        best_k,
        len(winner_survivors),
    )

    sfs = run_sfs(
        splits,
        winner_survivors,
        backbone=backbone,
        model_params=model_params,
        probes=probes,
        seed=seed,
        n_repeats=sfs_n_repeats,
        epsilon=sfs_epsilon,
        shap_subsample=shap_subsample,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
    )
    final_features = sfs.order[: sfs.d_plus]
    parsimonious_features = sfs.order[: sfs.d_parsimonious]

    test_metrics: dict[str, dict[str, float]] = {}
    test_prediction_rows: list[dict[str, Any]] = []
    for label, features in (
        (f"ffs_{best_operator}_K{best_k}", winner_survivors),
        (f"rffs_d_plus_{sfs.d_plus}", final_features),
        (f"rffs_d_parsimonious_{sfs.d_parsimonious}", parsimonious_features),
    ):
        model = fit_model(
            make_model(backbone, model_params, seed),
            splits.X_train[features],
            splits.y_train,
            splits.X_val[features],
            splits.y_val,
            early_stopping_rounds=early_stopping_rounds,
        )
        test_metrics[label] = evaluate(model, splits.X_test[features], splits.y_test)
        prediction = np.asarray(model.predict(splits.X_test[features]), dtype=float)
        test_prediction_rows.extend(
            {
                "feature_set": label,
                "n_features": len(features),
                "model_seed": seed,
                "row_id": row_id,
                "y_true": truth,
                "prediction": pred,
                "residual": truth - pred,
            }
            for row_id, truth, pred in zip(
                splits.X_test.index,
                splits.y_test.to_numpy(dtype=float),
                prediction,
                strict=True,
            )
        )

    category_table = category_summary(final_features, sfs.importance)
    category_table["n_selected_check"] = pd.Series(category_counts(final_features))

    sets = {
        f"{op}_K{best_k}": res.survivors.get(best_k, [])
        for op, res in ffs_results.items()
    }
    sets["rffs_final"] = final_features
    sets["rffs_parsimonious"] = parsimonious_features
    overlap = jaccard_overlap(sets)

    selected_feature_sets = {
        f"{operator}_K{k}": list(features)
        for operator, result in ffs_results.items()
        for k, features in result.survivors.items()
    }
    selected_feature_sets["rffs_final"] = list(final_features)
    selected_feature_sets["rffs_parsimonious"] = list(parsimonious_features)

    params = {
        "backbone": backbone,
        "operators": operators,
        "grid": grid,
        "q_iterations": q_iterations,
        "rho": rho,
        "tau": tau,
        "probes": probes.__dict__ if probes else None,
        "seed": seed,
        "rfe_step": rfe_step,
        "sfs_n_repeats": sfs_n_repeats,
        "sfs_epsilon": sfs_epsilon,
        "shap_subsample": shap_subsample,
        "n_jobs": n_jobs,
        "ffs_winner_rule": ffs_winner_rule(),
    }
    return RFFSReport(
        backbone=backbone,
        ffs=ffs_results,
        winner=winner,
        sfs=sfs,
        final_features=final_features,
        parsimonious_features=parsimonious_features,
        test_metrics=test_metrics,
        category_table=category_table,
        overlap=overlap,
        test_predictions=pd.DataFrame(test_prediction_rows),
        selected_feature_sets=selected_feature_sets,
        params=params,
    )
