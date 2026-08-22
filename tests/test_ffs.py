import inspect

import pandas as pd
from conftest import SMALL_RF_PARAMS

from xopa.data import make_splits
from xopa.selection import ffs, sfs
from xopa.selection.ffs import auto_tau, run_ffs, survivors_from_counts
from xopa.selection.probes import ProbeConfig


def test_survivor_tiebreak():
    counts = pd.Series({"a": 8, "b": 7, "c": 10})
    assert survivors_from_counts(counts, q=10, tau=0.8) == ["a", "c"]


def test_auto_tau():
    probe_counts = pd.Series({"__probe_gauss0": 3, "__probe_perm0": 5})
    assert auto_tau(probe_counts, q=10) == 0.6
    assert auto_tau(pd.Series(dtype=float), q=10) == 0.1


def test_ffs_recovers_informative(synth):
    X, y = synth
    s = make_splits(X, y, seed=3)
    r = run_ffs(
        s,
        operator="tree",
        grid=[16, 8],
        q_iterations=5,
        rho=0.7,
        tau=0.8,
        probes=ProbeConfig(),
        backbone="random_forest",
        model_params=SMALL_RF_PARAMS,
        seed=3,
    )
    informative = {f"data_{i:03d}" for i in range(1, 9)}
    assert len(set(r.survivors[8]) & informative) >= 6
    assert set(r.grid_eval.columns) >= {"d_star", "val_mse", "val_r2"}
    assert set(r.counts.columns) == {16, 8}
    assert r.grid_eval.loc[8, "d_star"] == len(r.survivors[8])


def test_selection_api_never_sees_calibration():
    for fn in (ffs.run_ffs, sfs.run_sfs):
        params = inspect.signature(fn).parameters
        assert not any("calib" in name for name in params)
