from hashlib import sha256 as _sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xopa.data import (
    DatasetContract,
    feature_category,
    load_dataset_bundle,
    make_splits,
    validate_dataset_identity,
)


def test_split_sizes_and_disjointness(synth):
    X, y = synth
    s = make_splits(X, y, seed=1)
    idx = [
        set(s.X_train.index),
        set(s.X_val.index),
        set(s.X_calib.index),
        set(s.X_test.index),
    ]
    assert sum(len(i) for i in idx) == len(X)
    assert set.union(*idx) == set(X.index)
    assert abs(len(s.X_train) / len(X) - 0.70) < 0.01


def test_scaler_fit_on_train_only(synth):
    X, y = synth
    s = make_splits(X, y, seed=1)
    np.testing.assert_allclose(s.X_train.mean().to_numpy(), 0.0, atol=1e-9)
    assert abs(float(s.X_val.mean().mean())) > 1e-12


def test_target_scaler_is_fit_on_train_only(synth):
    X, y = synth

    s = make_splits(X, y, seed=1, standardize_target=True)
    train_index = s.X_train.index
    expected_mean = float(y.loc[train_index].mean())
    expected_scale = float(y.loc[train_index].std(ddof=0))

    assert s.target_standardized is True
    assert s.target_mean == pytest.approx(expected_mean)
    assert s.target_scale == pytest.approx(expected_scale)
    assert float(s.y_train.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(s.y_train.std(ddof=0)) == pytest.approx(1.0, abs=1e-12)
    for transformed in (s.y_val, s.y_calib, s.y_test):
        expected = (y.loc[transformed.index] - expected_mean) / expected_scale
        pd.testing.assert_series_equal(transformed, expected)


def test_split_is_deterministic(synth):
    X, y = synth
    a = make_splits(X, y, seed=9)
    b = make_splits(X, y, seed=9)
    assert list(a.X_test.index) == list(b.X_test.index)


def test_category_partition():
    assert feature_category("data_001") == "CTX"
    assert feature_category("data_005") == "PV"
    assert feature_category("data_104") == "RMC"
    assert feature_category("data_141") == "WC"
    assert feature_category("data_183") == "PRS"
    with pytest.raises(ValueError):
        feature_category("data_164")


def test_dataset_bundle_records_identity_without_copying_source(tmp_path):
    path = tmp_path / "obfuscated.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "data_001": [1.0, 2.0, 3.0],
            "data_002": [7.0, 7.0, 7.0],
            "target_aux": [np.nan, 1.0, np.nan],
            "data_164": [4.0, 5.0, 6.0],
        }
    ).to_csv(path, index=False)

    bundle = load_dataset_bundle(path, target="data_164")

    assert bundle.identity.sha256 == _sha256(Path(path).read_bytes()).hexdigest()
    assert bundle.identity.source_path.name == path.name
    assert bundle.identity.byte_size == path.stat().st_size
    assert bundle.identity.raw_shape == (3, 5)
    assert bundle.identity.cleaned_shape == (3, 1)
    assert bundle.identity.dropped_index_columns == ("Unnamed: 0",)
    assert bundle.identity.dropped_subquality_columns == ("target_aux",)
    assert bundle.identity.dropped_constant_columns == ("data_002",)
    assert bundle.x.columns.tolist() == ["data_001"]
    assert not hasattr(bundle.identity, "values")


def test_dataset_identity_contract_rejects_a_different_predictor_count(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "data_001": [1.0, 2.0, 3.0],
            "data_164": [7.0, 8.0, 9.0],
        }
    ).to_csv(path, index=False)
    identity = load_dataset_bundle(path).identity
    contract = DatasetContract(
        sha256=identity.sha256,
        raw_shape=identity.raw_shape,
        cleaned_shape=(3, 316),
        target=identity.target,
        feature_schema_sha256=identity.feature_schema_sha256,
        dropped_index_columns=(),
        dropped_subquality_count=0,
        dropped_constant_count=0,
    )

    with pytest.raises(ValueError, match="cleaned shape"):
        validate_dataset_identity(identity, contract)
