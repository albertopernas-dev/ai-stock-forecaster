import math

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_forecaster.features.engineering import FEATURE_COLUMNS
from stock_forecaster.models.linear import LinearRegressionForecaster


def _numeric_X(
    *,
    rows: int = 6,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    if index is None:
        index = pd.Index(range(rows), name="observation")
    return pd.DataFrame(
        {
            feature: [
                float((row + 1) * (feature_number + 2)) / 100.0
                for row in range(rows)
            ]
            for feature_number, feature in enumerate(FEATURE_COLUMNS)
        },
        index=index,
    )


def _numeric_y(index: pd.Index) -> pd.Series:
    return pd.Series(
        [float(row - 2) / 100.0 for row in range(len(index))],
        index=index,
        name="future_return_5d",
    )


def _fitted_model(rows: int = 6) -> tuple[
    LinearRegressionForecaster,
    pd.DataFrame,
    pd.Series,
]:
    X = _numeric_X(rows=rows)
    y = _numeric_y(X.index)
    return LinearRegressionForecaster().fit(X, y), X, y


def test_model_uses_the_approved_sklearn_pipeline():
    model = LinearRegressionForecaster()

    assert isinstance(model._pipeline, Pipeline)
    assert list(model._pipeline.named_steps) == ["scaler", "regressor"]
    assert isinstance(model._pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(
        model._pipeline.named_steps["regressor"], LinearRegression
    )


def test_fit_accepts_exact_feature_schema_and_returns_self():
    X = _numeric_X()
    y = _numeric_y(X.index)
    model = LinearRegressionForecaster()

    returned = model.fit(X, y)

    assert returned is model


def test_predict_before_fit_is_rejected_clearly():
    with pytest.raises(RuntimeError, match="must be fitted"):
        LinearRegressionForecaster().predict(_numeric_X())


def test_coefficients_before_fit_are_rejected_clearly():
    with pytest.raises(RuntimeError, match="must be fitted"):
        LinearRegressionForecaster().coefficients_


def test_intercept_before_fit_is_rejected_clearly():
    with pytest.raises(RuntimeError, match="must be fitted"):
        LinearRegressionForecaster().intercept_


def test_missing_feature_is_rejected():
    X = _numeric_X().drop(columns=[FEATURE_COLUMNS[-1]])

    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


def test_additional_feature_is_rejected():
    X = _numeric_X().assign(unapproved_feature=1.0)

    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


def test_reordered_features_are_rejected_without_silent_reordering():
    X = _numeric_X().loc[:, list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


def test_non_numeric_feature_is_rejected_before_sklearn():
    X = _numeric_X()
    X[FEATURE_COLUMNS[3]] = ["bad"] * len(X)

    with pytest.raises(ValueError, match="feature columns must be numeric"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


def test_non_numeric_target_is_rejected_before_sklearn():
    X = _numeric_X()
    y = pd.Series(["bad"] * len(X), index=X.index)

    with pytest.raises(ValueError, match="y must have a numeric"):
        LinearRegressionForecaster().fit(X, y)


def test_feature_nan_is_rejected():
    X = _numeric_X()
    X.loc[X.index[0], FEATURE_COLUMNS[0]] = float("nan")

    with pytest.raises(ValueError, match="X must not contain null"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


def test_target_nan_is_rejected():
    X = _numeric_X()
    y = _numeric_y(X.index)
    y.iloc[0] = float("nan")

    with pytest.raises(ValueError, match="y must not contain null"):
        LinearRegressionForecaster().fit(X, y)


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_feature_positive_or_negative_infinity_is_rejected(invalid):
    X = _numeric_X()
    X.loc[X.index[0], FEATURE_COLUMNS[0]] = invalid

    with pytest.raises(ValueError, match="X must contain only finite"):
        LinearRegressionForecaster().fit(X, _numeric_y(X.index))


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_target_positive_or_negative_infinity_is_rejected(invalid):
    X = _numeric_X()
    y = _numeric_y(X.index)
    y.iloc[0] = invalid

    with pytest.raises(ValueError, match="y must contain only finite"):
        LinearRegressionForecaster().fit(X, y)


def test_x_y_length_mismatch_is_rejected():
    X = _numeric_X()
    y = _numeric_y(X.index[:-1])

    with pytest.raises(ValueError, match="equal lengths"):
        LinearRegressionForecaster().fit(X, y)


def test_equal_length_x_y_index_mismatch_is_rejected():
    index = pd.Index([8, 2, 5, 9, 1, 7], name="observation")
    X = _numeric_X(index=index)
    y = _numeric_y(index[::-1])
    assert len(X) == len(y)
    assert not X.index.equals(y.index)

    with pytest.raises(ValueError, match="indices must match exactly"):
        LinearRegressionForecaster().fit(X, y)


def test_fit_does_not_mutate_x():
    X = _numeric_X()
    y = _numeric_y(X.index)
    original = X.copy(deep=True)

    LinearRegressionForecaster().fit(X, y)

    pd.testing.assert_frame_equal(X, original)


def test_fit_does_not_mutate_y():
    X = _numeric_X()
    y = _numeric_y(X.index)
    original = y.copy(deep=True)

    LinearRegressionForecaster().fit(X, y)

    pd.testing.assert_series_equal(y, original)


def test_prediction_contract_preserves_type_name_index_and_count():
    model, _, _ = _fitted_model()
    index = pd.Index([90, 4, 17], name="validation_row")
    validation_X = _numeric_X(rows=3, index=index)

    predictions = model.predict(validation_X)

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "prediction"
    assert predictions.index.equals(validation_X.index)
    assert len(predictions) == len(validation_X)


def test_predict_does_not_mutate_x():
    model, _, _ = _fitted_model()
    validation_X = _numeric_X(rows=3, index=pd.Index([5, 1, 8]))
    original = validation_X.copy(deep=True)

    model.predict(validation_X)

    pd.testing.assert_frame_equal(validation_X, original)


def test_predict_rejects_invalid_schema_and_values_before_sklearn():
    model, _, _ = _fitted_model()
    validation_X = _numeric_X().drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        model.predict(validation_X)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_predict_rejects_non_finite_features(invalid):
    model, _, _ = _fitted_model()
    validation_X = _numeric_X()
    validation_X.loc[validation_X.index[0], FEATURE_COLUMNS[0]] = invalid

    with pytest.raises(ValueError):
        model.predict(validation_X)


def test_coefficients_and_intercept_have_exact_public_contracts():
    model, _, _ = _fitted_model()

    coefficients = model.coefficients_
    intercept = model.intercept_

    assert isinstance(coefficients, pd.Series)
    assert coefficients.index.equals(pd.Index(FEATURE_COLUMNS))
    assert coefficients.name == "coefficient"
    assert isinstance(intercept, float)
    assert math.isfinite(intercept)


def test_coefficients_are_a_copy_not_mutable_estimator_state():
    model, _, _ = _fitted_model()
    original = model.coefficients_

    changed = model.coefficients_
    changed.iloc[0] = 999.0

    pd.testing.assert_series_equal(model.coefficients_, original)


def test_pipeline_recovers_a_deterministic_linear_relationship():
    rows = 40
    index = pd.Index(range(100, 100 + rows), name="observation")
    X = pd.DataFrame(
        {
            feature: [
                float(((row + 3) ** (feature_number + 1)) % 101) / 100.0
                for row in range(rows)
            ]
            for feature_number, feature in enumerate(FEATURE_COLUMNS)
        },
        index=index,
    )
    weights = pd.Series(
        [0.4, -0.3, 0.2, -0.1, 0.08, -0.06, 0.04, -0.03, 0.02, -0.01],
        index=FEATURE_COLUMNS,
    )
    y = X.mul(weights).sum(axis=1).add(0.007).rename("future_return_5d")

    predictions = LinearRegressionForecaster().fit(X, y).predict(X)

    pd.testing.assert_series_equal(
        predictions,
        y.rename("prediction"),
        check_exact=False,
        atol=1e-10,
        rtol=1e-10,
    )


def test_scaler_statistics_come_only_from_train_and_prediction_does_not_refit():
    train_index = pd.Index([10, 20, 30, 40, 50, 60], name="train_row")
    train_X = _numeric_X(index=train_index)
    train_y = _numeric_y(train_X.index)
    model = LinearRegressionForecaster().fit(train_X, train_y)
    scaler = model._pipeline.named_steps["scaler"]
    expected_mean = train_X.mean().to_numpy()
    expected_scale = train_X.std(ddof=0).to_numpy()
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    validation_X = _numeric_X(
        rows=4,
        index=pd.Index([900, 100, 700, 300], name="validation_row"),
    ).add(1_000_000.0)
    predictions = model.predict(validation_X)

    assert scaler.mean_ == pytest.approx(expected_mean)
    assert scaler.scale_ == pytest.approx(expected_scale)
    assert scaler.mean_ == pytest.approx(mean_before)
    assert scaler.scale_ == pytest.approx(scale_before)
    assert predictions.index.equals(validation_X.index)


def test_multiple_rows_produce_one_finite_prediction_per_row():
    model, _, _ = _fitted_model(rows=12)
    validation_X = _numeric_X(
        rows=5,
        index=pd.Index([42, 7, 81, 3, 19], name="validation_row"),
    )

    predictions = model.predict(validation_X)

    assert len(predictions) == 5
    assert all(math.isfinite(value) for value in predictions)
    assert predictions.index.equals(validation_X.index)
