"""Global standardized linear regression forecaster."""

import math

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


class LinearRegressionForecaster:
    """Fit one standardized linear model across the stock universe."""

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        self._is_fitted = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LinearRegressionForecaster must be fitted before use"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "LinearRegressionForecaster":
        _validate_X(X)
        _validate_y(y)
        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths")
        if not X.index.equals(y.index):
            raise ValueError("X and y indices must match exactly")

        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        _validate_X(X)
        values = self._pipeline.predict(X)
        return pd.Series(values, index=X.index, name="prediction")

    @property
    def coefficients_(self) -> pd.Series:
        self._require_fitted()
        regressor = self._pipeline.named_steps["regressor"]
        if not all(math.isfinite(value) for value in regressor.coef_):
            raise RuntimeError(
                "Fitted linear regression diagnostics must be finite"
            )
        return pd.Series(
            regressor.coef_,
            index=FEATURE_COLUMNS,
            name="coefficient",
            dtype=float,
        ).copy()

    @property
    def intercept_(self) -> float:
        self._require_fitted()
        regressor = self._pipeline.named_steps["regressor"]
        value = float(regressor.intercept_)
        if not math.isfinite(value):
            raise RuntimeError(
                "Fitted linear regression diagnostics must be finite"
            )
        return value
