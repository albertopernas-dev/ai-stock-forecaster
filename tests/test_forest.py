import math

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from stock_forecaster.features.engineering import FEATURE_COLUMNS
from stock_forecaster.models.forest import RandomForestForecaster


def _numeric_X(
    *, rows: int = 64, index: pd.Index | None = None
) -> pd.DataFrame:
    if index is None:
        index = pd.Index(range(rows), name="observation")
    return pd.DataFrame(
        {
            feature: [
                float(((row + 3) ** (number + 1)) % 101) / 100.0
                for row in range(rows)
            ]
            for number, feature in enumerate(FEATURE_COLUMNS)
        },
        index=index,
    )


def _numeric_y(X: pd.DataFrame) -> pd.Series:
    return (
        X[FEATURE_COLUMNS[0]].mul(0.4)
        .sub(X[FEATURE_COLUMNS[1]].mul(0.25))
        .add(X[FEATURE_COLUMNS[4]].mul(0.1))
        .rename("future_return_5d")
    )


def _fitted_model() -> tuple[RandomForestForecaster, pd.DataFrame, pd.Series]:
    X = _numeric_X()
    y = _numeric_y(X)
    return RandomForestForecaster().fit(X, y), X, y


def test_model_uses_exact_approved_estimator_without_preprocessing():
    model = RandomForestForecaster()

    assert type(model._estimator) is RandomForestRegressor
    assert not hasattr(model, "_pipeline")
    params = model._estimator.get_params()
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 8
    assert params["min_samples_leaf"] == 20
    assert params["max_features"] == 1.0
    assert params["random_state"] == 42
    assert params["n_jobs"] == -1


def test_fit_accepts_exact_schema_and_returns_self():
    X = _numeric_X()
    model = RandomForestForecaster()
    assert model.fit(X, _numeric_y(X)) is model


def test_predict_before_fit_raises_clear_runtime_error():
    with pytest.raises(RuntimeError, match="must be fitted"):
        RandomForestForecaster().predict(_numeric_X(rows=3))


def test_feature_importances_before_fit_raise_clear_runtime_error():
    with pytest.raises(RuntimeError, match="must be fitted"):
        RandomForestForecaster().feature_importances_


@pytest.mark.parametrize(
    "invalid_X",
    [
        lambda X: X.drop(columns=[FEATURE_COLUMNS[-1]]),
        lambda X: X.assign(extra=1.0),
        lambda X: X.loc[:, list(reversed(FEATURE_COLUMNS))],
    ],
)
def test_missing_extra_or_reordered_features_are_rejected(invalid_X):
    X = invalid_X(_numeric_X())
    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        RandomForestForecaster().fit(X, pd.Series(range(len(X)), index=X.index))


def test_non_numeric_feature_is_rejected_before_sklearn():
    X = _numeric_X()
    y = _numeric_y(X)
    X[FEATURE_COLUMNS[3]] = "bad"
    with pytest.raises(ValueError, match="feature columns must be numeric"):
        RandomForestForecaster().fit(X, y)


def test_non_numeric_y_is_rejected_before_sklearn():
    X = _numeric_X()
    y = pd.Series(["bad"] * len(X), index=X.index)
    with pytest.raises(ValueError, match="y must have a numeric"):
        RandomForestForecaster().fit(X, y)


def test_integer_and_float_numeric_dtypes_are_accepted():
    X = _numeric_X()
    X[FEATURE_COLUMNS[0]] = range(len(X))
    assert RandomForestForecaster().fit(X, _numeric_y(X))


@pytest.mark.parametrize("container", ["X", "y"])
def test_fit_rejects_nan(container):
    X = _numeric_X()
    y = _numeric_y(X)
    if container == "X":
        X.iloc[0, 0] = float("nan")
    else:
        y.iloc[0] = float("nan")
    with pytest.raises(ValueError, match="must not contain null"):
        RandomForestForecaster().fit(X, y)


@pytest.mark.parametrize("container", ["X", "y"])
@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_fit_rejects_positive_and_negative_infinity(container, invalid):
    X = _numeric_X()
    y = _numeric_y(X)
    if container == "X":
        X.iloc[0, 0] = invalid
    else:
        y.iloc[0] = invalid
    with pytest.raises(ValueError, match="only finite"):
        RandomForestForecaster().fit(X, y)


def test_x_y_length_mismatch_is_rejected():
    X = _numeric_X()
    with pytest.raises(ValueError, match="equal lengths"):
        RandomForestForecaster().fit(X, _numeric_y(X).iloc[:-1])


def test_equal_length_index_mismatch_is_rejected_without_alignment():
    X = _numeric_X(index=pd.Index(range(100, 164)))
    y = _numeric_y(X).set_axis(X.index[::-1])
    assert len(X) == len(y) and not X.index.equals(y.index)
    with pytest.raises(ValueError, match="indices must match exactly"):
        RandomForestForecaster().fit(X, y)


def test_fit_does_not_mutate_x_or_y():
    X = _numeric_X()
    y = _numeric_y(X)
    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    RandomForestForecaster().fit(X, y)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_prediction_preserves_series_name_index_count_and_multiple_rows():
    model, _, _ = _fitted_model()
    index = pd.Index([90, 4, 17, 2], name="validation_row")
    X = _numeric_X(rows=4, index=index)

    predictions = model.predict(X)

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "prediction"
    assert predictions.index.equals(X.index)
    assert len(predictions) == len(X)
    assert all(math.isfinite(value) for value in predictions)


def test_predict_does_not_mutate_x_or_fitted_estimator():
    model, _, _ = _fitted_model()
    X = _numeric_X(rows=4, index=pd.Index([7, 1, 9, 3]))
    original_X = X.copy(deep=True)
    importances_before = model._estimator.feature_importances_.copy()

    model.predict(X)

    pd.testing.assert_frame_equal(X, original_X)
    assert model._estimator.feature_importances_ == pytest.approx(
        importances_before
    )


def test_predict_rejects_non_numeric_or_wrong_schema():
    model, _, _ = _fitted_model()
    non_numeric = _numeric_X(rows=3)
    non_numeric[FEATURE_COLUMNS[0]] = "bad"
    with pytest.raises(ValueError, match="numeric"):
        model.predict(non_numeric)
    with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
        model.predict(_numeric_X(rows=3).drop(columns=[FEATURE_COLUMNS[-1]]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_predict_rejects_non_finite_features(invalid):
    model, _, _ = _fitted_model()
    X = _numeric_X(rows=3)
    X.iloc[0, 0] = invalid
    with pytest.raises(ValueError):
        model.predict(X)


def test_feature_importances_have_exact_public_contract():
    model, _, _ = _fitted_model()

    importances = model.feature_importances_

    assert isinstance(importances, pd.Series)
    assert importances.index.equals(pd.Index(FEATURE_COLUMNS))
    assert importances.to_numpy() == pytest.approx(
        model._estimator.feature_importances_
    )
    assert all(math.isfinite(value) for value in importances)
    assert importances.ge(0).all()
    assert importances.sum() == pytest.approx(1.0)


def test_returned_importance_series_cannot_mutate_estimator_state():
    model, _, _ = _fitted_model()
    expected = model.feature_importances_
    changed = model.feature_importances_

    changed.iloc[0] = 999.0

    pd.testing.assert_series_equal(model.feature_importances_, expected)
    assert model._estimator.feature_importances_ == pytest.approx(
        expected.to_numpy()
    )


@pytest.mark.parametrize(
    "invalid",
    [
        [float("nan"), *([0.0] * 9)],
        [-0.1, 1.1, *([0.0] * 8)],
        [0.05] * 10,
    ],
)
def test_invalid_fitted_importances_raise_instead_of_being_changed(
    monkeypatch, invalid
):
    model, _, _ = _fitted_model()
    monkeypatch.setattr(
        type(model._estimator),
        "feature_importances_",
        property(lambda self: invalid),
    )
    with pytest.raises(RuntimeError, match="diagnostics"):
        model.feature_importances_


def test_identical_data_and_seed_produce_matching_predictions():
    X = _numeric_X(index=pd.Index(range(500, 564)))
    y = _numeric_y(X)
    model_a = RandomForestForecaster().fit(X, y)
    model_b = RandomForestForecaster().fit(X, y)

    pd.testing.assert_series_equal(model_a.predict(X), model_b.predict(X))
