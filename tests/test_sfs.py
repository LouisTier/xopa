import numpy as np
import pandas as pd
from conftest import SMALL_RF_PARAMS

from xopa.data import make_splits
from xopa.explain.shap_analysis import shap_matrix
from xopa.models import make_model
from xopa.selection.probes import ProbeConfig
from xopa.selection.sfs import run_sfs


def test_sfs_retains_repeat_predictions_and_ranking_shap(synth):
    x, y = synth
    splits = make_splits(x.iloc[:, :4], y, seed=23)

    result = run_sfs(
        splits,
        list(x.columns[:4]),
        backbone="random_forest",
        model_params=SMALL_RF_PARAMS,
        probes=ProbeConfig(),
        seed=23,
        n_repeats=2,
    )

    assert result.validation_predictions.shape == (
        len(result.order),
        2,
        len(splits.X_val),
    )
    assert result.validation_row_ids.tolist() == splits.X_val.index.tolist()
    np.testing.assert_allclose(result.validation_y_true, splits.y_val.to_numpy())
    assert len(result.repeat_metrics) == len(result.order) * 2
    rebuilt = result.repeat_metrics.groupby("k").agg(
        mse_mean=("mse", "mean"),
        r2_mean=("r2", "mean"),
    )
    np.testing.assert_allclose(rebuilt["mse_mean"], result.curve["mse_mean"])
    np.testing.assert_allclose(rebuilt["r2_mean"], result.curve["r2_mean"])
    assert result.ranking_shap_values.shape == result.ranking_feature_values.shape
    assert result.ranking_shap_values.index.tolist() == result.ranking_row_ids.tolist()
    assert len(result.ranking_predictions) == len(result.ranking_row_ids)


def test_mlp_shap_uses_the_fitted_torch_network_without_permutation_predictions():
    rng = np.random.default_rng(71)
    x = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"])
    y = pd.Series(0.4 * x["a"] - 0.2 * x["b"])
    model = make_model(
        "mlp",
        {
            "n_epochs": 1,
            "n_layers": 2,
            "n_units": 4,
            "activation": "relu",
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 4,
            "optimizer": "adam",
            "loss": "mse",
            "device": "cpu",
            "n_threads": 1,
        },
        seed=71,
    ).fit(x, y)

    def reject_wrapper_prediction(values):
        raise AssertionError("MLP SHAP must use the fitted torch network")

    model.predict = reject_wrapper_prediction
    values = shap_matrix(model, x, "mlp")

    assert values.shape == x.shape
    assert values.index.tolist() == x.index.tolist()
    assert values.columns.tolist() == x.columns.tolist()
    assert np.isfinite(values.to_numpy()).all()
