import importlib
import importlib.util

import pandas as pd


def test_truncated_mse_tie_prefers_smallest_survivor_set():
    module_name = "xopa.selection.ffs_winner"
    assert importlib.util.find_spec(module_name) is not None
    select_ffs_winner = importlib.import_module(module_name).select_ffs_winner
    candidates = pd.DataFrame(
        [
            {"operator": "mrmr", "K": 60, "d_star": 59, "val_mse": 0.067521},
            {"operator": "rfe", "K": 40, "d_star": 33, "val_mse": 0.067729},
            {"operator": "tree", "K": 60, "d_star": 53, "val_mse": 0.067802},
            {"operator": "lasso", "K": 50, "d_star": 46, "val_mse": 0.068171},
        ]
    )

    winner = select_ffs_winner(candidates)

    assert winner == ("rfe", 40)
