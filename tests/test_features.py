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
EXPECTED_EXCESS_TARGET = "future_excess_return_5d"


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
    assert module.EXCESS_TARGET_COLUMN == EXPECTED_EXCESS_TARGET
    assert result.columns.tolist() == ["date", "ticker", *EXPECTED_FEATURES,
                                       "future_return_5d", EXPECTED_EXCESS_TARGET]
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


def _exact_window_prices(stock_dates, stock_prices, spy_dates, spy_prices):
    stock = pd.DataFrame({
        "date": pd.to_datetime(stock_dates),
        "ticker": "AAPL",
        "adjusted_close": stock_prices,
        "volume": [100] * len(stock_dates),
    })
    spy = pd.DataFrame({
        "date": pd.to_datetime(spy_dates),
        "ticker": "SPY",
        "adjusted_close": spy_prices,
        "volume": [200] * len(spy_dates),
    })
    return pd.concat([stock, spy], ignore_index=True)


def test_excess_uses_stock_fifth_future_date_and_exact_spy_window():
    prices = _exact_window_prices(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06",
         "2020-01-07", "2020-01-08", "2020-01-09"],
        [100, 101, 102, 103, 104, 105, 106],
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06",
         "2020-01-07", "2020-01-08", "2020-01-09"],
        [200, 202, 204, 206, 208, 210, 220],
    )
    result = _engineering().build_features(prices)
    row = result[(result.ticker == "AAPL") & (result.date == "2020-01-01")].iloc[0]
    expected_stock = 105 / 100 - 1
    expected_spy = 210 / 200 - 1
    assert row.future_return_5d == pytest.approx(expected_stock)
    assert row.future_excess_return_5d == pytest.approx(expected_stock - expected_spy)


def test_excess_is_zero_with_complete_spy_horizon_and_final_five_are_nan():
    dates = pd.bdate_range("2020-01-01", periods=12)
    prices = _exact_window_prices(
        dates,
        [100 + i for i in range(12)],
        dates,
        [200 + 2 * i for i in range(12)],
    )
    result = _engineering().build_features(prices)
    aapl = result[result.ticker == "AAPL"]
    assert aapl.future_excess_return_5d.iloc[:7].notna().all()
    assert aapl.future_excess_return_5d.iloc[:7].map(math.isfinite).all()
    assert aapl.future_excess_return_5d.iloc[0] == pytest.approx(
        (105 / 100 - 1) - (210 / 200 - 1)
    )
    spy = result[result.ticker == "SPY"]
    assert spy.future_excess_return_5d.iloc[:7].abs().max() == pytest.approx(0)
    for ticker in ["AAPL", "SPY"]:
        ticker_result = result[result.ticker == ticker]
        assert ticker_result.future_return_5d.tail(5).isna().all()
        assert ticker_result.future_excess_return_5d.tail(5).isna().all()


@pytest.mark.parametrize("missing_side", ["start", "end"])
def test_missing_exact_spy_endpoint_makes_only_excess_nan(missing_side):
    dates = pd.bdate_range("2020-01-01", periods=7)
    spy_dates = dates.delete(0 if missing_side == "start" else 5)
    prices = _exact_window_prices(
        dates,
        [100 + i for i in range(7)],
        spy_dates,
        [200 + 2 * i for i in range(6)],
    )
    result = _engineering().build_features(prices)
    row = result[(result.ticker == "AAPL") & (result.date == dates[0])].iloc[0]
    assert pd.notna(row.future_return_5d)
    assert pd.isna(row.future_excess_return_5d)


def test_missing_spy_values_are_not_forward_or_backward_filled():
    dates = pd.bdate_range("2020-01-01", periods=8)
    spy_dates = dates.delete(5)
    prices = _exact_window_prices(
        dates,
        [100 + i for i in range(8)],
        spy_dates,
        [200 + 2 * i for i in range(7)],
    )
    result = _engineering().build_features(prices)
    aapl = result[result.ticker == "AAPL"].reset_index(drop=True)
    assert pd.isna(aapl.loc[0, "future_excess_return_5d"])
    assert pd.notna(aapl.loc[1, "future_excess_return_5d"])


def test_spy_independent_fifth_observation_is_not_substituted():
    stock_dates = pd.bdate_range("2020-01-01", periods=7)
    spy_dates = stock_dates.delete(5).append(pd.DatetimeIndex(["2020-01-15"]))
    prices = _exact_window_prices(
        stock_dates,
        [100 + i for i in range(7)],
        spy_dates,
        [200 + 2 * i for i in range(7)],
    )
    result = _engineering().build_features(prices)
    row = result[(result.ticker == "AAPL") & (result.date == stock_dates[0])].iloc[0]
    assert pd.isna(row.future_excess_return_5d)


def test_excess_keeps_input_immutable_and_feature_columns_unchanged():
    prices = _exact_window_prices(
        pd.bdate_range("2020-01-01", periods=8),
        [100 + i for i in range(8)],
        pd.bdate_range("2020-01-01", periods=8),
        [200 + i for i in range(8)],
    )
    before = prices.copy(deep=True)
    result = _engineering().build_features(prices)
    pd.testing.assert_frame_equal(prices, before)
    assert list(_engineering().FEATURE_COLUMNS) == EXPECTED_FEATURES
    non_null = result[EXPECTED_EXCESS_TARGET].dropna()
    assert non_null.map(math.isfinite).all()
