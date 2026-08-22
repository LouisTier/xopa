from conftest import SMALL_RF_PARAMS

from xopa.models import make_model
from xopa.selection.boruta import run_boruta


def rf_factory():
    return make_model("random_forest", SMALL_RF_PARAMS, seed=0)


def test_boruta_separates_signal_from_noise(synth):
    X, y = synth
    r = run_boruta(X, y, rf_factory, max_iter=30, seed=0)
    informative = {f"data_{i:03d}" for i in range(1, 9)}
    assert len(set(r.confirmed) & informative) >= 6
    assert len(set(r.confirmed) - informative) <= 3
    assert set(r.ranking[: len(r.confirmed)]) == set(r.confirmed)
    assert sorted(r.confirmed + r.tentative + r.rejected) == sorted(X.columns)


def test_boruta_decision_history_reproduces_final_status(synth):
    x, y = synth
    result = run_boruta(x, y, rf_factory, max_iter=8, seed=0)

    final = (
        result.decision_history.sort_values("iteration")
        .groupby("feature", sort=False)
        .tail(1)
    )

    assert set(final.loc[final["status"] == "confirmed", "feature"]) == set(
        result.confirmed
    )
    assert set(final.loc[final["status"] == "tentative", "feature"]) == set(
        result.tentative
    )
    assert set(final.loc[final["status"] == "rejected", "feature"]) == set(
        result.rejected
    )
    assert {
        "importance",
        "shadow_threshold",
        "hits",
        "accept_probability",
        "reject_probability",
    } <= set(result.decision_history)
