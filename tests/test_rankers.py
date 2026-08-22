import numpy as np
import pytest
from conftest import SMALL_RF_PARAMS

from xopa.models import make_model
from xopa.selection.probes import ProbeConfig, add_probes, is_probe
from xopa.selection.rankers import RANKERS


def rf_factory():
    return make_model("random_forest", SMALL_RF_PARAMS, seed=0)


@pytest.mark.parametrize("name", ["rfe", "mrmr", "lasso", "tree"])
def test_nested_prefix_property(synth, name):
    X, y = synth
    full = RANKERS[name](X, y, rf_factory, k_max=30, seed=0, checkpoints=[30, 20, 10])
    again = RANKERS[name](X, y, rf_factory, k_max=30, seed=0, checkpoints=[30, 20, 10])
    direct = RANKERS[name](X, y, rf_factory, k_max=10, seed=0, checkpoints=[10])
    assert full == again
    assert full[:10] == direct


def test_rankers_find_signal(synth):
    X, y = synth
    top8 = set(RANKERS["tree"](X, y, rf_factory, k_max=8, seed=0))
    informative = {f"data_{i:03d}" for i in range(1, 9)}
    assert len(top8 & informative) >= 6


def test_probes_are_tracked(synth):
    X, _ = synth
    Xp, names = add_probes(X, ProbeConfig(), np.random.default_rng(0))
    assert len(names) == 2 and all(is_probe(n) for n in names)
    assert Xp.shape[1] == X.shape[1] + 2
    assert not any(is_probe(c) for c in X.columns)
