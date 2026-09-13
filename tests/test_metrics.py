import math
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from stock_forecaster.models.metrics import RegressionMetrics, evaluate_predictions


def test_known_example_has_exact_mae_rmse_and_directional_accuracy():
    y_true = pd.Series([1.0, -1.0, 0.0, 2.0])
    y_pred = pd.Series([0.0, 1.0, 0.0, 4.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.mae == pytest.approx(1.25)
    assert metrics.rmse == pytest.approx(1.5)
    assert metrics.directional_accuracy == pytest.approx(0.5)


def test_zero_is_its_own_direction():
    y_true = pd.Series([0.0, 0.0, 0.01, -0.01])
    y_pred = pd.Series([0.0, 0.01, 0.0, -0.02])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.directional_accuracy == pytest.approx(0.5)


def test_perfect_nonconstant_predictions_have_perfect_metrics():
    y_true = pd.Series([-0.02, 0.0, 0.01, 0.04])

    metrics = evaluate_predictions(y_true, y_true.copy())

    assert metrics == RegressionMetrics(
        mae=0.0,
        rmse=0.0,
        directional_accuracy=1.0,
        correlation=1.0,
    )


def test_nonperfect_pearson_correlation_matches_hand_calculated_value():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 3.0, 2.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.correlation == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0])),
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([4.0, 4.0, 4.0])),
    ],
)
def test_correlation_is_nan_when_either_series_has_zero_variance(y_true, y_pred):
    metrics = evaluate_predictions(y_true, y_pred)

    assert math.isnan(metrics.correlation)


def test_regression_metrics_is_frozen():
    metrics = evaluate_predictions(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))

    with pytest.raises(FrozenInstanceError):
        metrics.mae = 1.0


def test_empty_inputs_are_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_predictions(pd.Series(dtype=float), pd.Series(dtype=float))


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions(pd.Series([1.0, 2.0]), pd.Series([1.0]))


def test_misaligned_indices_are_rejected_instead_of_silently_sorted():
    y_true = pd.Series([0.01, 0.02], index=[4, 9])
    y_pred = pd.Series([0.01, 0.02], index=[9, 4])

    with pytest.raises(ValueError, match="indices must match exactly"):
        evaluate_predictions(y_true, y_pred)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (pd.Series([0.01, float("nan")]), pd.Series([0.01, 0.02])),
        (pd.Series([0.01, 0.02]), pd.Series([float("nan"), 0.02])),
    ],
)
def test_null_values_are_rejected(y_true, y_pred):
    with pytest.raises(ValueError, match="must not contain null values"):
        evaluate_predictions(y_true, y_pred)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (pd.Series([0.01, float("inf")]), pd.Series([0.01, 0.02])),
        (pd.Series([0.01, 0.02]), pd.Series([0.01, float("-inf")])),
    ],
)
def test_positive_or_negative_infinity_is_rejected(y_true, y_pred):
    with pytest.raises(ValueError, match="must contain only finite values"):
        evaluate_predictions(y_true, y_pred)


def test_inputs_are_not_mutated_and_nonmonotonic_alignment_is_preserved():
    index = pd.Index([8, 2, 5], name="observation")
    y_true = pd.Series([0.02, -0.01, 0.03], index=index, name="actual")
    y_pred = pd.Series([0.01, -0.02, 0.04], index=index, name="prediction")
    original_true = y_true.copy(deep=True)
    original_pred = y_pred.copy(deep=True)

    metrics = evaluate_predictions(y_true, y_pred)

    pd.testing.assert_series_equal(y_true, original_true)
    pd.testing.assert_series_equal(y_pred, original_pred)
    assert metrics.mae == pytest.approx(0.01)
    assert metrics.directional_accuracy == 1.0
