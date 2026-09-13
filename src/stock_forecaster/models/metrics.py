"""Common regression metrics for return forecasts."""

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    """Scalar evaluation metrics for one set of predictions."""

    mae: float
    rmse: float
    directional_accuracy: float
    correlation: float


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> RegressionMetrics:
    """Evaluate aligned, finite predictions without dropping observations."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")
    if y_true.empty:
        raise ValueError("y_true and y_pred must not be empty")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred indices must match exactly")
    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("y_true and y_pred must not contain null values")
    if not all(math.isfinite(value) for value in y_true) or not all(
        math.isfinite(value) for value in y_pred
    ):
        raise ValueError("y_true and y_pred must contain only finite values")

    errors = y_true - y_pred
    mae = float(errors.abs().mean())
    rmse = math.sqrt(float(errors.pow(2).mean()))

    true_direction = y_true.gt(0).astype(int) - y_true.lt(0).astype(int)
    predicted_direction = y_pred.gt(0).astype(int) - y_pred.lt(0).astype(int)
    directional_accuracy = float(true_direction.eq(predicted_direction).mean())

    if y_true.nunique() == 1 or y_pred.nunique() == 1:
        correlation = float("nan")
    else:
        correlation = float(y_true.corr(y_pred))

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        directional_accuracy=directional_accuracy,
        correlation=correlation,
    )
