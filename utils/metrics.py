"""
Metrics used to evaluate forecasting performance.

The Autoformer paper mainly reports:
- MSE
- MAE
"""

import numpy as np


def mse(pred, true):
    """
    Mean Squared Error.

    It measures the average squared difference between predictions and true values.
    """

    return np.mean((pred - true) ** 2)


def mae(pred, true):
    """
    Mean Absolute Error.

    It measures the average absolute difference between predictions and true values.
    """

    return np.mean(np.abs(pred - true))


def metric(pred, true):
    """
    Compute the main forecasting metrics.

    Returns:
        mae_score, mse_score
    """

    mae_score = mae(pred, true)
    mse_score = mse(pred, true)

    return mae_score, mse_score