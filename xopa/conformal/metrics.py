"""Interval-quality metrics: PICP, MPIW, WCC, and conditional coverage."""

import numpy as np


def picp(y: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """Prediction-interval coverage probability: fraction of y inside [lb, ub]."""
    return float(np.mean((y >= lb) & (y <= ub)))


def mpiw(lb: np.ndarray, ub: np.ndarray) -> float:
    """Mean prediction-interval width."""
    return float(np.mean(ub - lb))


def wcc(coverages: np.ndarray) -> float:
    """Worst-case coverage: the minimum per-split evaluation coverage."""
    return float(np.min(coverages))


def running_average(coverages: np.ndarray) -> np.ndarray:
    """Running mean of the per-split coverages (the convergence diagnostic)."""
    coverages = np.asarray(coverages, dtype=float)
    return np.cumsum(coverages) / np.arange(1, len(coverages) + 1)


def picp_by_slice(
    y: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    slices: dict[str, np.ndarray],
) -> dict[str, float]:
    """Coverage computed separately on named index sets (or boolean masks).

    Use for regime-conditional coverage, e.g. record ranges before and after a
    known change point.
    """
    return {name: picp(y[idx], lb[idx], ub[idx]) for name, idx in slices.items()}
