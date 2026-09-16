"""Pure feature and forecast-target computation for normalized stock prices."""

import pandas as pd

FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "distance_sma_10",
    "distance_sma_20",
    "distance_sma_50",
    "volume_change_1d",
    "volume_ratio_20",
)
TARGET_COLUMN = "future_return_5d"
EXCESS_TARGET_COLUMN = "future_excess_return_5d"

_REQUIRED_COLUMNS = ("date", "ticker", "adjusted_close", "volume")
_OUTPUT_COLUMNS = (
    "date", "ticker", *FEATURE_COLUMNS, TARGET_COLUMN, EXCESS_TARGET_COLUMN
)


def _relative_change(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    safe_denominator = denominator.where(denominator.ne(0))
    return numerator / safe_denominator - 1


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return ticker-isolated predictors and the five-observation target."""
    missing_columns = [
        column for column in _REQUIRED_COLUMNS if column not in prices.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if prices.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate date and ticker rows")

    ordered = (
        prices.loc[:, _REQUIRED_COLUMNS]
        .copy()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    ticker = ordered["ticker"]
    price = ordered["adjusted_close"]
    volume = ordered["volume"]
    price_groups = price.groupby(ticker, sort=False)
    volume_groups = volume.groupby(ticker, sort=False)

    result = ordered.loc[:, ["date", "ticker"]].copy()
    for period in (1, 5, 20):
        result[f"return_{period}d"] = _relative_change(
            price,
            price_groups.shift(period),
        )

    daily_return_groups = result["return_1d"].groupby(ticker, sort=False)
    for window in (5, 20):
        result[f"volatility_{window}d"] = daily_return_groups.transform(
            lambda values: values.rolling(
                window=window,
                min_periods=window,
            ).std()
        )

    for window in (10, 20, 50):
        moving_average = price_groups.transform(
            lambda values: values.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
        result[f"distance_sma_{window}"] = _relative_change(
            price,
            moving_average,
        )

    result["volume_change_1d"] = _relative_change(
        volume,
        volume_groups.shift(1),
    )
    volume_average = volume_groups.transform(
        lambda values: values.rolling(window=20, min_periods=20).mean()
    )
    result["volume_ratio_20"] = _relative_change(volume, volume_average)
    result[TARGET_COLUMN] = _relative_change(price_groups.shift(-5), price)

    target_end_dates = ordered["date"].groupby(ticker, sort=False).shift(-5)
    spy_prices = (
        ordered.loc[ordered["ticker"].eq("SPY"), ["date", "adjusted_close"]]
        .set_index("date")["adjusted_close"]
    )
    spy_start = ordered["date"].map(spy_prices)
    spy_end = target_end_dates.map(spy_prices)
    spy_return = _relative_change(spy_end, spy_start)
    excess_target = result[TARGET_COLUMN] - spy_return
    result[EXCESS_TARGET_COLUMN] = excess_target.where(
        ~excess_target.isin([float("inf"), float("-inf")])
    )

    return (
        result.loc[:, _OUTPUT_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
