"""Canonical Boruta feature selection (Kursa & Rudnicki, 2010).

A standalone baseline, deliberately not a Psi operator inside FFS: Boruta
already embeds its own resampling loop and statistical test, and its natural
output is a self-sized confirmed set rather than a size-K ranking.

Every active feature gets a shadow copy (its values shuffled across
samples, which preserves the marginal and destroys the association with the
target); a feature scores a hit when its importance exceeds a percentile
(default: the maximum) of the shadow importances; accumulated hits are tested
against Binomial(iterations, 0.5) with a multiplicity correction (the BorutaPy
two-step by default: Benjamini-Hochberg across features combined with a
Bonferroni over iterations; ``two_step=False`` gives the R package's plain
Bonferroni). Rejected features are removed permanently; features undecided at
the end are tentative, with the median rough fix flagging the supported ones.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, false_discovery_control

from xopa.models import fit_model, model_importances
from xopa.selection.rankers import ModelFactory

_SHADOW_MAX = "__shadow_max__"


@dataclass
class BorutaResult:
    """Three-state decision, ranking, and the importance history."""

    confirmed: list[str]
    tentative: list[str]  # undecided at stop (all of them)
    tentative_supported: list[str]  # tentative passing the median rough fix
    rejected: list[str]
    ranking: list[str]  # confirmed, then tentative, then rejected
    importance_history: pd.DataFrame  # index iteration; features + shadow max
    decision_history: pd.DataFrame  # long per-feature statistical state
    n_iter: int
    params: dict[str, Any] = field(default_factory=dict)


def run_boruta(
    x: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    *,
    max_iter: int = 100,
    alpha: float = 0.05,
    perc: float = 100.0,
    two_step: bool = True,
    seed: int = 42,
) -> BorutaResult:
    """Run canonical Boruta with any estimator exposing importances.

    Args:
        x: Feature matrix (training data).
        y: Target aligned with ``x``.
        model_factory: Zero-argument callable returning a fresh unfitted model.
        max_iter: Iteration cap; statistical tests start at iteration 5.
        alpha: Significance level of the binomial decisions.
        perc: Percentile of shadow importances used as the hit threshold
            (100 reproduces the original max-of-shadows rule).
        two_step: BorutaPy's Benjamini-Hochberg + Bonferroni-over-iterations
            correction; False gives the R package's plain Bonferroni.
        seed: Seed for the shadow shuffles.

    Returns:
        A :class:`BorutaResult`.
    """
    rng = np.random.default_rng(seed)
    features = list(x.columns)
    decision: dict[str, int] = dict.fromkeys(features, 0)  # 0 ?, 1 yes, -1 no
    hits: dict[str, int] = dict.fromkeys(features, 0)
    history: list[dict[str, float]] = []
    decision_rows: list[dict[str, Any]] = []

    iteration = 0
    while any(d == 0 for d in decision.values()) and iteration < max_iter:
        iteration += 1
        active = [f for f in features if decision[f] >= 0]

        x_active = x[active].to_numpy()
        x_shadow = x_active.copy()
        while x_shadow.shape[1] < 5:  # canonical padding to at least 5 shadows
            x_shadow = np.hstack([x_shadow, x_shadow])
        x_shadow = np.apply_along_axis(rng.permutation, 0, x_shadow)

        columns = active + [f"__shadow_{i}" for i in range(x_shadow.shape[1])]
        x_fit = pd.DataFrame(
            np.hstack([x_active, x_shadow]), index=x.index, columns=columns
        )
        model = fit_model(model_factory(), x_fit, y, early_stopping_rounds=None)
        importances = model_importances(model, x_fit, y, kind="auto", seed=seed)
        real_imp = importances[: len(active)]
        shadow_imp = importances[len(active) :]
        threshold = float(np.percentile(shadow_imp, perc))

        row: dict[str, float] = {_SHADOW_MAX: threshold}
        for name, value in zip(active, real_imp, strict=True):
            row[name] = float(value)
            if decision[name] == 0 and value > threshold:
                hits[name] += 1
        history.append(row)

        accept_probability = dict.fromkeys(active, np.nan)
        reject_probability = dict.fromkeys(active, np.nan)
        accepted_now: set[str] = set()
        rejected_now: set[str] = set()
        if iteration > 4:  # canonical: only test after the 5th round
            undecided = [f for f in features if decision[f] == 0]
            hit_counts = np.array([hits[f] for f in undecided])
            accept_ps = binom.sf(hit_counts - 1, iteration, 0.5)
            reject_ps = binom.cdf(hit_counts, iteration, 0.5)
            if two_step:
                accept = (false_discovery_control(accept_ps) <= alpha) & (
                    accept_ps <= alpha / iteration
                )
                reject = (false_discovery_control(reject_ps) <= alpha) & (
                    reject_ps <= alpha / iteration
                )
            else:
                bonferroni = alpha / len(features)
                accept = accept_ps <= bonferroni
                reject = reject_ps <= bonferroni
            for name, acc_p, rej_p, acc, rej in zip(
                undecided,
                accept_ps,
                reject_ps,
                accept,
                reject,
                strict=True,
            ):
                accept_probability[name] = float(acc_p)
                reject_probability[name] = float(rej_p)
                if acc:
                    decision[name] = 1
                    accepted_now.add(name)
                elif rej:
                    decision[name] = -1
                    rejected_now.add(name)

        status_names = {-1: "rejected", 0: "tentative", 1: "confirmed"}
        decision_rows.extend(
            {
                "iteration": iteration,
                "feature": name,
                "importance": float(real_imp[position]),
                "shadow_threshold": threshold,
                "hits": hits[name],
                "accept_probability": accept_probability[name],
                "reject_probability": reject_probability[name],
                "accepted_this_iteration": name in accepted_now,
                "rejected_this_iteration": name in rejected_now,
                "status": status_names[decision[name]],
            }
            for position, name in enumerate(active)
        )

    importance_history = pd.DataFrame(history, index=range(1, iteration + 1))

    confirmed = [f for f in features if decision[f] == 1]
    tentative = [f for f in features if decision[f] == 0]
    rejected = [f for f in features if decision[f] == -1]

    shadow_median = float(importance_history[_SHADOW_MAX].median())
    median_imp = importance_history.reindex(columns=features).median()
    tentative_supported = [f for f in tentative if float(median_imp[f]) > shadow_median]

    per_iter_ranks = (
        importance_history.reindex(columns=features)
        .rank(axis=1, ascending=False)
        .median()
    )
    by_median_importance = lambda f: -float(median_imp[f])  # noqa: E731
    ranking = (
        sorted(confirmed, key=by_median_importance)
        + sorted(tentative, key=by_median_importance)
        + sorted(rejected, key=lambda f: float(per_iter_ranks[f]))
    )

    params = {
        "max_iter": max_iter,
        "alpha": alpha,
        "perc": perc,
        "two_step": two_step,
        "seed": seed,
    }
    return BorutaResult(
        confirmed=confirmed,
        tentative=tentative,
        tentative_supported=tentative_supported,
        rejected=rejected,
        ranking=ranking,
        importance_history=importance_history,
        decision_history=pd.DataFrame(decision_rows),
        n_iter=iteration,
        params=params,
    )
