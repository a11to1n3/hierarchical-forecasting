"""Common evaluation metrics for hierarchical forecasting baselines."""

from __future__ import annotations

import numpy as np


def _to_numpy(array) -> np.ndarray:
    """Convert inputs to a numpy array without copying when possible."""
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def weighted_absolute_percentage_error(
    y_true,
    y_pred,
    *,
    percentage: bool = True,
    epsilon: float = 1e-12,
) -> float:
    """Compute Weighted Absolute Percentage Error (WAPE).

    WAPE = sum(|y - ŷ|) / sum(|y|)

    Args:
        y_true: Array-like of ground truth observations.
        y_pred: Array-like of predictions.
        percentage: If True, return percentage (×100).
        epsilon: Small constant to guard against zero division.

    Returns:
        Scalar WAPE (percentage if requested). Returns ``np.nan`` when the
        denominator is too small.
    """

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    denominator = np.sum(np.abs(y_true))
    if denominator <= epsilon:
        return float('nan')

    value = np.sum(np.abs(y_true - y_pred)) / denominator
    if percentage:
        value *= 100.0
    return float(value)


def weighted_absolute_squared_error(
    y_true,
    y_pred,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Compute Weighted Absolute Squared Error (WASE).

    WASE = sum(|y| * (y - ŷ)^2) / sum(|y|)

    Args:
        y_true: Array-like of ground truth observations.
        y_pred: Array-like of predictions.
        epsilon: Small constant to guard against zero division.

    Returns:
        Scalar WASE. Returns ``np.nan`` when the denominator is too small.
    """

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    weights = np.abs(y_true)
    denominator = np.sum(weights)
    if denominator <= epsilon:
        return float('nan')

    value = np.sum(weights * (y_true - y_pred) ** 2) / denominator
    return float(value)

