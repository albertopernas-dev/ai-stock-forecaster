"""Fixed global Random Forest return forecaster."""

import math

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor

from stock_forecaster.features.engineering import FEATURE_COLUMNS


def _validate_X(X: pd.DataFrame) -> None:
    if list(X.columns) != list(FEATURE_COLUMNS):
        raise ValueError(
            "X columns must be exactly FEATURE_COLUMNS in declared order"
        )
    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not is_numeric_dtype(X[column].dtype)
    ]
    if non_numeric:
        raise ValueError(
            "X feature columns must be numeric pandas dtypes: "
            + ", ".join(non_numeric)
        )
    if X.isna().any().any():
        raise ValueError("X must not contain null values")
    if not all(
        math.isfinite(value)
        for row in X.itertuples(index=False, name=None)
        for value in row
    ):
        raise ValueError("X must contain only finite values")


def _validate_y(y: pd.Series) -> None:
    if not is_numeric_dtype(y.dtype):
        raise ValueError("y must have a numeric pandas dtype")
    if y.isna().any():
        raise ValueError("y must not contain null values")
    if not all(math.isfinite(value) for value in y):
        raise ValueError("y must contain only finite values")


class RandomForestForecaster:
    """Fit one fixed Random Forest across the stock universe."""

    def __init__(self) -> None:
        self._estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self._is_fitted = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "RandomForestForecaster must be fitted before use"
            )

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "RandomForestForecaster":
        _validate_X(X)
        _validate_y(y)
        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths")
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match exactly")
        self._estimator.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        _validate_X(X)
        values = self._estimator.predict(X)
        return pd.Series(values, index=X.index, name="prediction")

    @property
    def feature_importances_(self) -> pd.Series:
        self._require_fitted()
        values = self._estimator.feature_importances_
        if (
            len(values) != len(FEATURE_COLUMNS)
            or not all(math.isfinite(value) for value in values)
            or any(value < 0 for value in values)
            or not math.isclose(sum(values), 1.0)
        ):
            raise RuntimeError(
                "Fitted Random Forest diagnostics must be finite, "
                "non-negative, and sum to one"
            )
        return pd.Series(
            values,
            index=FEATURE_COLUMNS,
            name="importance",
            dtype=float,
        ).copy()
