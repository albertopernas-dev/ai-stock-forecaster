import pandas as pd
import pytest

from stock_forecaster.backtesting.simulation import (
    LONG_ONLY,
    LONG_SHORT,
    REBALANCE_STEP,
    build_period_returns,
    summarize_backtest,
)
from stock_forecaster.models.ranking import BUCKET_SIZE


def _rows(date, model, tickers, y_true, prediction, year=2016):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date] * len(tickers)),
            "ticker": list(tickers),
            "validation_year": year,
            "model": model,
            "y_true": list(y_true),
            "prediction": list(prediction),
        }
    )


def _ten(date, y_true, prediction, model="LinearRegression", year=2016):
    tickers = [f"T{index:02d}" for index in range(1, 11)]
    return _rows(date, model, tickers, y_true, prediction, year=year)


DESCENDING = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def test_long_short_gross_return_is_top_minus_bottom_bucket_mean():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=DESCENDING,
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert list(periods.columns) == [
        "model",
        "validation_year",
        "date",
        "n_tickers",
        "gross_return",
        "traded_notional",
    ]
    assert len(periods) == 1
    assert periods.loc[0, "n_tickers"] == 10
    assert periods.loc[0, "gross_return"] == pytest.approx(0.08 - 0.03)


def test_ties_resolve_by_ascending_ticker():
    frame = _ten(
        "2016-01-04",
        y_true=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        prediction=[10, 9, 8, 7, 6, 6, 4, 3, 2, 1],
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods.loc[0, "gross_return"] == pytest.approx(0.2)


def test_constant_predictions_give_an_undefined_period():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="ZeroBaseline",
    )

    periods = build_period_returns(frame)

    assert pd.isna(periods.loc[0, "gross_return"])
    assert pd.isna(periods.loc[0, "traded_notional"])


def test_fewer_tickers_than_two_buckets_give_an_undefined_period():
    count = 2 * BUCKET_SIZE - 1
    tickers = [f"T{index:02d}" for index in range(1, count + 1)]
    frame = _rows(
        "2016-01-04",
        "LinearRegression",
        tickers,
        [0.01 * index for index in range(count, 0, -1)],
        [float(index) for index in range(count, 0, -1)],
    )

    periods = build_period_returns(frame)

    assert periods.loc[0, "n_tickers"] == count
    assert pd.isna(periods.loc[0, "gross_return"])


def test_only_every_fifth_date_of_each_year_is_a_rebalance():
    first = pd.bdate_range("2016-01-04", periods=12)
    second = pd.bdate_range("2017-01-02", periods=12)
    frames = [
        _ten(date, [0.01] * 10, DESCENDING, year=year)
        for dates, year in ((first, 2016), (second, 2017))
        for date in dates
    ]

    periods = build_period_returns(pd.concat(frames, ignore_index=True))

    assert periods.loc[periods.validation_year.eq(2016), "date"].tolist() == [
        first[0],
        first[5],
        first[10],
    ]
    assert periods.loc[periods.validation_year.eq(2017), "date"].tolist() == [
        second[0],
        second[5],
        second[10],
    ]
    assert REBALANCE_STEP == 5


def test_models_are_emitted_in_known_presentation_order():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    shuffled = pd.concat(
        [
            _ten("2016-01-04", values, DESCENDING, model="RandomForest"),
            _ten("2016-01-04", values, DESCENDING, model="MomentumBaseline"),
            _ten("2016-01-04", values, DESCENDING, model="LinearRegression"),
        ],
        ignore_index=True,
    )

    periods = build_period_returns(shuffled)

    assert periods["model"].tolist() == [
        "MomentumBaseline",
        "LinearRegression",
        "RandomForest",
    ]


def test_rows_dated_on_or_after_the_test_boundary_are_rejected():
    frame = _ten("2024-01-02", [0.01] * 10, DESCENDING, year=2023)

    with pytest.raises(ValueError, match="2024"):
        build_period_returns(frame)


def test_unknown_strategy_is_rejected():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    with pytest.raises(ValueError, match="Unsupported strategy"):
        build_period_returns(frame, strategy="market_neutral_optimized")


@pytest.mark.parametrize("step", [0, -1])
def test_non_positive_step_is_rejected(step):
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    with pytest.raises(ValueError, match="step must be a positive integer"):
        build_period_returns(frame, step=step)


def test_duplicate_date_ticker_model_rows_are_rejected():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_period_returns(duplicated)


@pytest.mark.parametrize("missing", ["prediction", "y_true", "model", "date"])
def test_missing_required_columns_are_rejected(missing):
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING).drop(columns=[missing])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_period_returns(frame)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_period_returns(pd.DataFrame())


def test_build_period_returns_does_not_mutate_its_input():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)
    original = frame.copy(deep=True)

    build_period_returns(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_long_only_gross_return_is_top_bucket_minus_universe_mean():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    frame = _ten("2016-01-04", y_true=values, prediction=DESCENDING)

    periods = build_period_returns(frame, strategy=LONG_ONLY)

    universe_mean = sum(values) / len(values)
    assert periods.loc[0, "gross_return"] == pytest.approx(0.08 - universe_mean)


def test_long_only_differs_from_long_short_on_the_same_frame():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    frame = _ten("2016-01-04", y_true=values, prediction=DESCENDING)

    long_only = build_period_returns(frame, strategy=LONG_ONLY)
    long_short = build_period_returns(frame, strategy=LONG_SHORT)

    assert long_only.loc[0, "gross_return"] < long_short.loc[0, "gross_return"]


def test_long_only_is_undefined_for_constant_predictions():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="MeanBaseline",
    )

    periods = build_period_returns(frame, strategy=LONG_ONLY)

    assert pd.isna(periods.loc[0, "gross_return"])


ASCENDING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _sequence(dates, predictions_per_date, model="LinearRegression", year=2016):
    return pd.concat(
        [
            _ten(date, [0.01] * 10, prediction, model=model, year=year)
            for date, prediction in zip(dates, predictions_per_date, strict=True)
        ],
        ignore_index=True,
    )


def test_first_rebalance_trades_the_full_gross_exposure():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    long_short = build_period_returns(frame, strategy=LONG_SHORT)
    long_only = build_period_returns(frame, strategy=LONG_ONLY)

    assert long_short.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert long_only.loc[0, "traded_notional"] == pytest.approx(1.0)


def test_an_unchanged_book_trades_nothing():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = _sequence(dates, [DESCENDING] * 6)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert len(periods) == 2
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert periods.loc[1, "traded_notional"] == pytest.approx(0.0)


def test_a_fully_reversed_book_trades_twice_the_gross_exposure():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = _sequence(dates, [DESCENDING] + [ASCENDING] * 5)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods.loc[1, "traded_notional"] == pytest.approx(4.0)


def test_an_undefined_rebalance_carries_the_previous_book_forward():
    dates = pd.bdate_range("2016-01-04", periods=11)
    predictions = [DESCENDING] + [[0.0] * 10] * 5 + [DESCENDING] * 5
    frame = _sequence(dates, predictions)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert len(periods) == 3
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert pd.isna(periods.loc[1, "traded_notional"])
    assert periods.loc[2, "traded_notional"] == pytest.approx(0.0)


def test_turnover_sequencing_continues_across_a_year_boundary():
    first = _sequence(pd.bdate_range("2016-12-01", periods=1), [DESCENDING], year=2016)
    second = _sequence(pd.bdate_range("2017-01-03", periods=1), [DESCENDING], year=2017)
    frame = pd.concat([first, second], ignore_index=True)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods["validation_year"].tolist() == [2016, 2017]
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert periods.loc[1, "traded_notional"] == pytest.approx(0.0)


def test_each_model_keeps_its_own_book():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = pd.concat(
        [
            _sequence(dates, [DESCENDING] * 6, model="LinearRegression"),
            _sequence(dates, [DESCENDING] + [ASCENDING] * 5, model="RandomForest"),
        ],
        ignore_index=True,
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)
    linear = periods.loc[periods.model.eq("LinearRegression")].reset_index(drop=True)
    forest = periods.loc[periods.model.eq("RandomForest")].reset_index(drop=True)

    assert linear.loc[1, "traded_notional"] == pytest.approx(0.0)
    assert forest.loc[1, "traded_notional"] == pytest.approx(4.0)


def _periods(model, year, gross_values, traded_values):
    dates = pd.bdate_range("2016-01-04", periods=len(gross_values))
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": dates,
            "n_tickers": 15,
            "gross_return": list(gross_values),
            "traded_notional": list(traded_values),
        }
    )


def test_summary_reports_means_counts_and_break_even():
    periods = _periods(
        "LinearRegression",
        2016,
        gross_values=[0.004, -0.002, float("nan"), 0.001],
        traded_values=[2.0, 1.0, float("nan"), 1.0],
    )

    summary = summarize_backtest(periods)

    assert list(summary.columns) == [
        "model",
        "periods",
        "mean_gross_return",
        "mean_traded_notional",
        "breakeven_cost_bps",
        "share_positive_periods",
    ]
    mean_gross = (0.004 - 0.002 + 0.001) / 3
    mean_traded = (2.0 + 1.0 + 1.0) / 3
    assert summary.loc[0, "periods"] == 3
    assert summary.loc[0, "mean_gross_return"] == pytest.approx(mean_gross)
    assert summary.loc[0, "mean_traded_notional"] == pytest.approx(mean_traded)
    assert summary.loc[0, "breakeven_cost_bps"] == pytest.approx(
        10000 * mean_gross / mean_traded
    )
    assert summary.loc[0, "share_positive_periods"] == pytest.approx(2 / 3)


def test_break_even_is_negative_when_the_gross_edge_is_negative():
    periods = _periods("RandomForest", 2016, [-0.003, -0.001], [2.0, 2.0])

    assert summarize_backtest(periods).loc[0, "breakeven_cost_bps"] < 0


def test_break_even_is_undefined_without_trading():
    periods = _periods("LinearRegression", 2016, [0.004, 0.002], [0.0, 0.0])

    assert pd.isna(summarize_backtest(periods).loc[0, "breakeven_cost_bps"])


def test_exactly_zero_gross_return_is_not_a_positive_period():
    periods = _periods("LinearRegression", 2016, [0.0, 0.0], [2.0, 2.0])

    summary = summarize_backtest(periods)

    assert summary.loc[0, "share_positive_periods"] == pytest.approx(0.0)


def test_summary_rows_follow_known_model_order():
    periods = pd.concat(
        [
            _periods("RandomForest", 2016, [0.001, 0.002], [2.0, 2.0]),
            _periods("ZeroBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _periods("LinearRegression", 2016, [0.001, 0.002], [2.0, 2.0]),
        ],
        ignore_index=True,
    )

    summary = summarize_backtest(periods)

    assert summary["model"].tolist() == [
        "ZeroBaseline",
        "LinearRegression",
        "RandomForest",
    ]
    assert summary.loc[0, "periods"] == 0
    assert pd.isna(summary.loc[0, "breakeven_cost_bps"])


def test_annual_and_pooled_summaries_agree_on_counts():
    periods = pd.concat(
        [
            _periods("LinearRegression", 2016, [0.004, 0.002], [2.0, 2.0]),
            _periods("LinearRegression", 2017, [0.001, 0.003], [2.0, 2.0]),
        ],
        ignore_index=True,
    )

    annual = summarize_backtest(periods, by_year=True)
    pooled = summarize_backtest(periods)

    assert list(annual.columns)[:2] == ["model", "validation_year"]
    assert annual["validation_year"].tolist() == [2016, 2017]
    assert annual["periods"].sum() == pooled.loc[0, "periods"]


def test_summarize_backtest_does_not_mutate_its_input():
    periods = _periods("LinearRegression", 2016, [0.004, 0.002], [2.0, 2.0])
    original = periods.copy(deep=True)

    summarize_backtest(periods, by_year=True)

    pd.testing.assert_frame_equal(periods, original)


def test_summarize_backtest_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        summarize_backtest(pd.DataFrame())
