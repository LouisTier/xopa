"""Pure selection rule for the preliminary FFS winner."""

import math

import pandas as pd


def ffs_winner_rule() -> dict[str, object]:
    """Return the serializable preliminary-winner rule."""
    return {
        "metric": "validation_mse",
        "truncate_decimals": 2,
        "tie_breakers": ["d_star", "raw_validation_mse", "operator", "K"],
    }


def select_ffs_winner(candidates: pd.DataFrame) -> tuple[str, int]:
    """Select by validation MSE, then retained dimension.

    Cells in the same MSE bin are ordered by increasing ``d_star``.
    Raw validation MSE, operator name, and ``K`` provide deterministic final
    fallbacks without changing the compression preference.
    """
    if candidates.empty:
        raise ValueError("at least one FFS candidate is required")

    ranked = candidates.assign(
        truncated_val_mse=candidates["val_mse"].map(
            lambda value: math.trunc(float(value) * 100.0) / 100.0
        )
    ).sort_values(
        ["truncated_val_mse", "d_star", "val_mse", "operator", "K"],
        kind="stable",
    )
    winner = ranked.iloc[0]
    return str(winner["operator"]), int(winner["K"])
