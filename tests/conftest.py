import numpy as np
import pandas as pd
import pytest

N, D_INFORMATIVE, D_NOISE = 600, 8, 32

SMALL_RF_PARAMS = {
    "n_estimators": 60,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_jobs": -1,
}


@pytest.fixture(scope="session")
def synth() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    x_inf = rng.normal(size=(N, D_INFORMATIVE))
    coefs = np.linspace(2.0, 0.5, D_INFORMATIVE)
    y = x_inf @ coefs + rng.normal(scale=0.5, size=N)
    x_noise = rng.normal(size=(N, D_NOISE))
    cols = [f"data_{i:03d}" for i in range(1, D_INFORMATIVE + D_NOISE + 1)]
    x = pd.DataFrame(np.hstack([x_inf, x_noise]), columns=cols)
    return x, pd.Series(y, name="data_164")
