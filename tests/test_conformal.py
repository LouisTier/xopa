import numpy as np
from conftest import SMALL_RF_PARAMS

from xopa.conformal.cqr import fit_mcqr
from xopa.conformal.metrics import mpiw, picp, picp_by_slice, running_average, wcc
from xopa.conformal.scp import _split_threshold, fit_mscp, predict_mscp
from xopa.data import make_splits
from xopa.models import fit_model, make_model


def test_order_statistic_threshold_exact():
    scores = np.arange(1.0, 101.0)
    q = _split_threshold(scores, alpha=0.1, n_cal=100)
    assert q == 91.0


def test_mcqr_scores_use_monotonicized_calibration_quantiles(synth):
    x, y = synth
    splits = make_splits(x, y, seed=5)
    truth = splits.y_calib.to_numpy(dtype=float)

    class FixedPredictionModel:
        def __init__(self, values):
            self.values = values

        def predict(self, frame):
            assert len(frame) == len(self.values)
            return self.values

    result = fit_mcqr(
        FixedPredictionModel(truth + 1.0),
        FixedPredictionModel(truth - 1.0),
        splits,
        alpha=0.1,
        m=2,
        calib_fraction=0.8,
        seed=7,
    )

    np.testing.assert_allclose(result.scores, -1.0)


def test_mscp_coverage_synthetic(synth):
    X, y = synth
    s = make_splits(X, y, seed=5)
    model = fit_model(
        make_model("random_forest", SMALL_RF_PARAMS, 0), s.X_train, s.y_train
    )
    res = fit_mscp(model, s, alpha=0.1, m=200, seed=5)
    lb, ub = predict_mscp(model, s.X_test, res)
    assert 0.82 <= picp(s.y_test.to_numpy(), lb, ub) <= 0.98
    assert res.thresholds.shape == (200,)
    assert np.isfinite(res.q_hat)
    assert res.coverages.shape == (200,)
    assert wcc(res.coverages) <= res.coverages.mean()


def test_metrics_basics():
    y = np.array([0.0, 1.0, 2.0])
    lb, ub = np.array([-1.0, 0.5, 3.0]), np.array([1.0, 1.5, 4.0])
    assert picp(y, lb, ub) == 2 / 3
    assert np.isclose(mpiw(lb, ub), (2.0 + 1.0 + 1.0) / 3)
    ra = running_average(np.array([1.0, 0.0, 1.0]))
    assert np.allclose(ra, [1.0, 0.5, 2 / 3])


def test_picp_by_slice():
    y = np.array([0.0, 0.0, 10.0, 10.0])
    lb, ub = np.array([-1, -1, -1, -1.0]), np.array([1, 1, 1, 1.0])
    out = picp_by_slice(y, lb, ub, {"lo": np.array([0, 1]), "hi": np.array([2, 3])})
    assert out == {"lo": 1.0, "hi": 0.0}
