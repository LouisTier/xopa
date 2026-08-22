import numpy as np
import pandas as pd

from xopa.data import load_dataset_bundle, make_splits
from xopa.utils import write_common_artifacts


def test_preprocessing_artifact_records_train_fitted_target_transform(tmp_path):
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "data_001": np.arange(20, dtype=float),
            "data_002": np.arange(20, dtype=float) ** 2,
            "data_164": np.linspace(10.0, 50.0, 20),
        }
    ).to_csv(source, index=False)
    dataset = load_dataset_bundle(source)
    splits = make_splits(dataset.x, dataset.y, seed=4, standardize_target=True)

    write_common_artifacts(tmp_path, dataset, splits)

    with np.load(tmp_path / "preprocessing.npz") as artifact:
        assert bool(artifact["target_standardized"]) is True
        assert float(artifact["target_mean"]) == splits.target_mean
        assert float(artifact["target_scale"]) == splits.target_scale
        assert str(artifact["target_name"]) == "data_164"
