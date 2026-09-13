import importlib
import math
from statistics import stdev

import pandas as pd
import pytest

EXPECTED_FEATURES = [
    "return_1d", "return_5d", "return_20d", "volatility_5d",
    "volatility_20d", "distance_sma_10", "distance_sma_20",
    "distance_sma_50", "volume_change_1d", "volume_ratio_20",
]


def _engineering():
    return importlib.import_module("stock_forecaster.features.engineering")


def _prices(count=65, ticker="AAPL", scale=1.0):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=count),
            "ticker": ticker,
            "adjusted_close": [scale * (100 + i) for i in range(count)],
            "volume": [scale * 100 * (i + 1) for i in range(count)],
        }
    )


def test_missing_columns_are_listed_clearly():
    prices = _prices().drop(columns=["adjusted_close", "volume"])
    with pytest.raises(ValueError, match="adjusted_close.*volume"):
        _engineering().build_features(prices)


def test_duplicate_ticker_dates_are_rejected():
    prices = _prices()
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate.*date.*ticker"):
        _engineering().build_features(prices)


def test_input_is_unchanged_and_output_contract_is_exact():
    prices = _prices().iloc[::-1].copy()
    prices.index = range(100, 165)
    before = prices.copy(deep=True)
    module = _engineering()
    result = module.build_features(prices)
    pd.testing.assert_frame_equal(prices, before)
    assert list(module.FEATURE_COLUMNS) == EXPECTED_FEATURES
    assert module.TARGET_COLUMN == "future_return_5d"
    assert result.columns.tolist() == ["date", "ticker", *EXPECTED_FEATURES,
                                       "future_return_5d"]
    assert result.index.equals(pd.RangeIndex(65))
    assert len(result) == len(prices)


def test_unsorted_inputs_produce_deterministic_date_ticker_order():
    prices = pd.concat([_prices(10, "META"), _prices(10)], ignore_index=True)
    module = _engineering()
    result = module.build_features(prices.iloc[::-1])
    pd.testing.assert_frame_equal(result, module.build_features(prices))
    assert list(zip(result["date"], result["ticker"])) == sorted(
        zip(prices["date"], prices["ticker"])
    )


@pytest.mark.parametrize("column,denominator", [
    ("return_1d", 154), ("return_5d", 150), ("return_20d", 135),
])
def test_trailing_return_formulas(column, denominator):
    result = _engineering().build_features(_prices())
    assert result.loc[55, column] == pytest.approx(155 / denominator - 1)


@pytest.mark.parametrize("window", [5, 20])
def test_volatility_uses_sample_std_of_trailing_daily_returns(window):
    result = _engineering().build_features(_prices())
    trailing_returns = [(100 + i) / (99 + i) - 1
                        for i in range(56 - window, 56)]
    assert result.loc[55, f"volatility_{window}d"] == pytest.approx(
        stdev(trailing_returns)
    )


@pytest.mark.parametrize("window,mean_price", [
    (10, 150.5), (20, 145.5), (50, 130.5),
])
def test_sma_distance_formulas(window, mean_price):
    result = _engineering().build_features(_prices())
    assert result.loc[55, f"distance_sma_{window}"] == pytest.approx(
        155 / mean_price - 1
    )


def test_volume_feature_formulas():
    result = _engineering().build_features(_prices())
    assert result.loc[55, "volume_change_1d"] == pytest.approx(5600 / 5500 - 1)
    assert result.loc[55, "volume_ratio_20"] == pytest.approx(5600 / 4650 - 1)


def test_zero_volume_denominators_are_nan_and_never_infinite():
    prices = _prices(30)
    prices.loc[:19, "volume"] = 0
    result = _engineering().build_features(prices)
    assert pd.isna(result.loc[20, "volume_change_1d"])
    assert pd.isna(result.loc[19, "volume_ratio_20"])
    assert result.loc[20, "volume_ratio_20"] == pytest.approx(19.0)
    for column in ["volume_change_1d", "volume_ratio_20"]:
        assert not any(math.isinf(value) for value in result[column].dropna())


def test_target_uses_fifth_future_observation_and_keeps_tail_rows():
    result = _engineering().build_features(_prices())
    assert result.loc[55, "future_return_5d"] == pytest.approx(160 / 155 - 1)
    assert result["future_return_5d"].tail(5).isna().all()
    assert result["future_return_5d"].isna().sum() == 5
    assert len(result) == 65


def test_warmup_nans_are_preserved_without_dropping_rows():
    result = _engineering().build_features(_prices())
    warmups = {
        "return_1d": 1, "return_5d": 5, "return_20d": 20,
        "volatility_5d": 5, "volatility_20d": 20,
        "distance_sma_10": 9, "distance_sma_20": 19,
        "distance_sma_50": 49, "volume_change_1d": 1,
        "volume_ratio_20": 19,
    }
    for column, count in warmups.items():
        assert result[column].iloc[:count].isna().all(), column
        assert pd.notna(result.loc[count, column]), column
    assert len(result) == 65


def test_missing_prices_are_not_forward_filled():
    prices = _prices()
    prices.loc[30, "adjusted_close"] = float("nan")
    result = _engineering().build_features(prices)
    assert result.loc[30:31, "return_1d"].isna().all()
    assert pd.isna(result.loc[30, "distance_sma_10"])
    assert pd.isna(result.loc[35, "return_5d"])


def test_future_price_changes_only_target_not_current_predictors():
    prices = _prices()
    module = _engineering()
    before = module.build_features(prices)
    prices.loc[60, "adjusted_close"] = 320.0
    after = module.build_features(prices)
    pd.testing.assert_series_equal(
        before.loc[55, list(module.FEATURE_COLUMNS)],
        after.loc[55, list(module.FEATURE_COLUMNS)],
    )
    assert before.loc[55, module.TARGET_COLUMN] == pytest.approx(160 / 155 - 1)
    assert after.loc[55, module.TARGET_COLUMN] == pytest.approx(320 / 155 - 1)


def test_cross_ticker_windows_and_targets_restart_independently():
    aapl = _prices()
    meta = _prices(ticker="META", scale=100.0)
    meta["date"] += pd.Timedelta(days=30)
    prices = pd.concat([aapl, meta], ignore_index=True).iloc[::-1]
    module = _engineering()
    combined = module.build_features(prices)
    for ticker, single in [("AAPL", aapl), ("META", meta)]:
        actual = combined.loc[combined["ticker"] == ticker].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, module.build_features(single))
        assert pd.isna(actual.loc[0, "return_1d"])
        assert pd.isna(actual.loc[0, "volume_change_1d"])
        assert actual["distance_sma_50"].iloc[:49].isna().all()
        assert actual["future_return_5d"].tail(5).isna().all()
