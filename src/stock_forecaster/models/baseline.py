"""Simple forecasting baselines for five-day stock returns."""

import pandas as pd


class MeanBaseline:
    """Predict the global mean learned from training targets."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y_train: pd.Series) -> "MeanBaseline":
        """Learn the mean of a complete, non-empty training target."""
        if y_train.empty:
            raise ValueError("y_train must not be empty")
        if y_train.isna().any():
            raise ValueError("y_train must not contain null values")

        self.mean_ = float(y_train.mean())
        return self

    def predict(self, index: pd.Index) -> pd.Series:
        """Return constant predictions aligned with the requested index."""
        if self.mean_ is None:
            raise RuntimeError("MeanBaseline must be fitted before prediction")
        return pd.Series(self.mean_, index=index, name="prediction", dtype=float)


class MomentumBaseline:
    """Use the current five-day return as the future-return prediction."""

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return the existing return_5d feature without changing row order."""
        if "return_5d" not in X.columns:
            raise ValueError("Missing required feature: return_5d")
        return X["return_5d"].copy().rename("prediction")
