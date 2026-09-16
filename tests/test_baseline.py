import pandas as pd
import pytest

import stock_forecaster.models.baseline as baseline_module
from stock_forecaster.models.baseline import MeanBaseline, MomentumBaseline


def test_mean_baseline_learns_the_global_training_target_mean():
    y_train = pd.Series([0.01, -0.02, 0.04], name="future_return_5d")

    model = MeanBaseline()
    returned = model.fit(y_train)

    assert returned is model
    assert model.mean_ == pytest.approx(0.01)


def test_mean_baseline_fit_does_not_mutate_training_targets():
    y_train = pd.Series([0.03, -0.01, 0.02], index=[8, 2, 5])
    original = y_train.copy(deep=True)

    MeanBaseline().fit(y_train)

    pd.testing.assert_series_equal(y_train, original)


def test_mean_baseline_predictions_equal_learned_mean_and_preserve_index():
    index = pd.Index([11, 4, 20], name="validation_row")
    model = MeanBaseline().fit(pd.Series([0.02, -0.01, 0.02]))

    predictions = model.predict(index)

    expected = pd.Series([0.01, 0.01, 0.01], index=index, name="prediction")
    pd.testing.assert_series_equal(predictions, expected)


def test_mean_baseline_predict_before_fit_is_rejected():
    with pytest.raises(RuntimeError, match="must be fitted before prediction"):
        MeanBaseline().predict(pd.Index([0, 1]))


def test_mean_baseline_rejects_empty_training_targets():
    with pytest.raises(ValueError, match="must not be empty"):
        MeanBaseline().fit(pd.Series(dtype=float))


def test_mean_baseline_rejects_null_training_targets():
    with pytest.raises(ValueError, match="must not contain null values"):
        MeanBaseline().fit(pd.Series([0.01, float("nan")]))


def test_momentum_baseline_returns_return_5d_with_original_row_alignment():
    index = pd.Index([9, 3, 12], name="validation_row")
    X = pd.DataFrame(
        {
            "return_1d": [0.01, 0.02, -0.01],
            "return_5d": [0.05, -0.02, 0.0],
            "future_return_5d": [-0.8, 0.9, 0.7],
        },
        index=index,
    )

    predictions = MomentumBaseline().predict(X)

    expected = pd.Series([0.05, -0.02, 0.0], index=index, name="prediction")
    pd.testing.assert_series_equal(predictions, expected)
    predictions.iloc[0] = 99.0
    assert X.loc[9, "return_5d"] == 0.05


def test_momentum_baseline_does_not_mutate_validation_features():
    X = pd.DataFrame(
        {
            "return_5d": [0.03, -0.01],
            "volume_ratio_20": [0.2, -0.1],
        },
        index=[7, 1],
    )
    original = X.copy(deep=True)

    MomentumBaseline().predict(X)

    pd.testing.assert_frame_equal(X, original)


def test_momentum_baseline_rejects_missing_return_5d():
    X = pd.DataFrame({"return_1d": [0.01, -0.02]})

    with pytest.raises(ValueError, match="Missing required feature: return_5d"):
        MomentumBaseline().predict(X)


def test_zero_baseline_returns_float_zeros_without_mutating_features():
    index = pd.Index([9, 3, 12], name="validation_row")
    X = pd.DataFrame(
        {
            "return_5d": [0.05, -0.02, 0.0],
            "unrelated": ["first", "second", "third"],
        },
        index=index,
    )
    original = X.copy(deep=True)

    predictions = baseline_module.ZeroBaseline().predict(X)

    expected = pd.Series([0.0, 0.0, 0.0], index=index, name="prediction", dtype=float)
    pd.testing.assert_series_equal(predictions, expected)
    pd.testing.assert_frame_equal(X, original)
