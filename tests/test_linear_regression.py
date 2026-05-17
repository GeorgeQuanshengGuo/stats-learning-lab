import math

import numpy as np

from src.modeling.statistical.linear_regression import regression_metrics


def test_regression_metrics_perfect_predictions():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])

    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["R-squared"] == 1


def test_regression_metrics_with_errors():
    metrics = regression_metrics([1, 2, 3], [1, 2, 5])

    assert metrics["RMSE"] == np.sqrt(4 / 3)
    assert metrics["MAE"] == 2 / 3
    assert metrics["R-squared"] == -1


def test_regression_metrics_constant_y_returns_nan_r_squared():
    metrics = regression_metrics([2, 2, 2], [2, 2, 3])

    assert math.isnan(metrics["R-squared"])
