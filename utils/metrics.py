"""
Metrics used to evaluate forecasting performance:

- MSE
- MAE
"""

import numpy as np


def mse(pred, true):

    return np.mean((pred - true) ** 2)


def mae(pred, true):

    return np.mean(np.abs(pred - true))


def metric(pred, true):

    mae_score = mae(pred, true)
    mse_score = mse(pred, true)

    return mae_score, mse_score