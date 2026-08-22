import numpy as np
import pandas as pd

from xopa.data import exclude_categories


def frame(columns):
    return pd.DataFrame(np.zeros((3, len(columns))), columns=columns)


def test_exclude_drops_only_the_named_category():
    x = frame(["data_001", "data_050", "data_120", "data_183", "data_400"])
    reduced, dropped = exclude_categories(x, ["PRS"])
    assert dropped == ["data_183", "data_400"]
    assert list(reduced.columns) == ["data_001", "data_050", "data_120"]
