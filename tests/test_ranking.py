import pandas as pd
import pytest

from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES
from stock_forecaster.models.ranking import (
    BUCKET_SIZE,
    KNOWN_MODEL_ORDER,
    MIN_TICKERS_PER_DATE,
    build_daily_ranking,
    select_non_overlapping,
    summarize_ranking,
)


def _predictions(date, model, tickers, y_true, prediction, year=2016):
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


def test_known_model_order_matches_evaluation_presentation_order():
    assert KNOWN_MODEL_ORDER == ("ZeroBaseline", *DEFAULT_MODEL_NAMES)


def test_perfectly_ordered_predictions_give_ic_of_one():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )

    daily = build_daily_ranking(frame)

    assert list(daily.columns) == [
        "model",
        "validation_year",
        "date",
        "n_tickers",
        "ic",
        "top_mean",
        "bottom_mean",
        "spread",
    ]
    assert len(daily) == 1
    assert daily.loc[0, "n_tickers"] == 5
    assert daily.loc[0, "ic"] == pytest.approx(1.0)


def test_perfectly_inverted_predictions_give_ic_of_minus_one():
    frame = _predictions(
        "2016-03-01",
        "RandomForest",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )

    assert build_daily_ranking(frame).loc[0, "ic"] == pytest.approx(-1.0)


def test_one_swapped_pair_matches_independently_computed_spearman():
    # Ranks differ only in the last two positions: sum of squared rank
    # differences is 2, so rho = 1 - (6 * 2) / (5 * (25 - 1)) = 0.9
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.01, 0.02],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    )

    assert build_daily_ranking(frame).loc[0, "ic"] == pytest.approx(0.9)


def test_constant_predictions_give_undefined_ic_not_zero():
    frame = _predictions(
        "2016-03-01",
        "ZeroBaseline",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert pd.isna(build_daily_ranking(frame).loc[0, "ic"])


def test_constant_realized_values_give_undefined_ic():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.02, 0.02, 0.02, 0.02, 0.02],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )

    assert pd.isna(build_daily_ranking(frame).loc[0, "ic"])


def test_too_few_tickers_give_undefined_ic_but_record_the_count():
    tickers = [f"T{index}" for index in range(MIN_TICKERS_PER_DATE - 1)]
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        tickers,
        [0.04, 0.03, 0.02, 0.01],
        [0.4, 0.3, 0.2, 0.1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "n_tickers"] == MIN_TICKERS_PER_DATE - 1
    assert pd.isna(daily.loc[0, "ic"])


def test_rows_dated_on_or_after_the_test_boundary_are_rejected():
    frame = _predictions(
        "2024-01-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
        year=2023,
    )

    with pytest.raises(ValueError, match="2024"):
        build_daily_ranking(frame)


def test_duplicate_date_ticker_model_rows_are_rejected():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_daily_ranking(duplicated)


@pytest.mark.parametrize("missing", ["prediction", "y_true", "model", "date"])
def test_missing_required_columns_are_rejected(missing):
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    ).drop(columns=[missing])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_daily_ranking(frame)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_daily_ranking(pd.DataFrame())


def test_build_daily_ranking_does_not_mutate_its_input():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )
    original = frame.copy(deep=True)

    build_daily_ranking(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_models_are_emitted_in_known_presentation_order():
    tickers = ["A", "B", "C", "D", "E"]
    realized = [0.05, 0.04, 0.03, 0.02, 0.01]
    predicted = [0.9, 0.7, 0.5, 0.3, 0.1]
    shuffled = pd.concat(
        [
            _predictions("2016-03-01", "RandomForest", tickers, realized, predicted),
            _predictions("2016-03-01", "MeanBaseline", tickers, realized, predicted),
            _predictions(
                "2016-03-01", "LinearRegression", tickers, realized, predicted
            ),
        ],
        ignore_index=True,
    )

    daily = build_daily_ranking(shuffled)

    assert daily["model"].tolist() == [
        "MeanBaseline",
        "LinearRegression",
        "RandomForest",
    ]


def _ten_ticker_frame(y_true, prediction, model="LinearRegression"):
    tickers = [f"T{index:02d}" for index in range(1, 11)]
    return _predictions("2016-03-01", model, tickers, y_true, prediction)


def test_spread_uses_top_and_bottom_buckets_with_exact_arithmetic():
    frame = _ten_ticker_frame(
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "top_mean"] == pytest.approx(0.08)
    assert daily.loc[0, "bottom_mean"] == pytest.approx(0.03)
    assert daily.loc[0, "spread"] == pytest.approx(0.05)


def test_ties_at_the_bucket_boundary_resolve_by_ascending_ticker():
    # T05 and T06 tie on prediction; ascending ticker puts T05 in the top
    # bucket, so the top bucket collects the single 1.0 realized value.
    frame = _ten_ticker_frame(
        y_true=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        prediction=[10, 9, 8, 7, 6, 6, 4, 3, 2, 1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "top_mean"] == pytest.approx(0.2)
    assert daily.loc[0, "bottom_mean"] == pytest.approx(0.0)
    assert daily.loc[0, "spread"] == pytest.approx(0.2)


def test_fewer_tickers_than_two_buckets_give_an_undefined_spread():
    count = 2 * BUCKET_SIZE - 1
    tickers = [f"T{index:02d}" for index in range(1, count + 1)]
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        tickers,
        [0.01 * index for index in range(count, 0, -1)],
        [float(index) for index in range(count, 0, -1)],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "n_tickers"] == count
    assert pd.notna(daily.loc[0, "ic"])
    assert pd.isna(daily.loc[0, "spread"])


def test_constant_predictions_give_an_undefined_spread():
    frame = _ten_ticker_frame(
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="MeanBaseline",
    )

    daily = build_daily_ranking(frame)

    assert pd.isna(daily.loc[0, "spread"])
    assert pd.isna(daily.loc[0, "top_mean"])
    assert pd.isna(daily.loc[0, "bottom_mean"])


def _daily_frame(dates, year, model="LinearRegression"):
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": pd.to_datetime(dates),
            "n_tickers": 15,
            "ic": 0.1,
            "top_mean": 0.02,
            "bottom_mean": 0.01,
            "spread": 0.01,
        }
    )


def test_non_overlapping_selection_takes_every_fifth_date_within_each_year():
    first = pd.bdate_range("2016-01-04", periods=12)
    second = pd.bdate_range("2017-01-02", periods=12)
    daily = pd.concat(
        [_daily_frame(first, 2016), _daily_frame(second, 2017)], ignore_index=True
    )

    selected = select_non_overlapping(daily, step=5)

    assert selected.loc[selected.validation_year.eq(2016), "date"].tolist() == [
        first[0],
        first[5],
        first[10],
    ]
    assert selected.loc[selected.validation_year.eq(2017), "date"].tolist() == [
        second[0],
        second[5],
        second[10],
    ]
    assert list(selected.columns) == list(daily.columns)


def test_non_overlapping_selection_keeps_every_model_on_a_selected_date():
    dates = pd.bdate_range("2016-01-04", periods=6)
    daily = pd.concat(
        [
            _daily_frame(dates, 2016, model="LinearRegression"),
            _daily_frame(dates, 2016, model="RandomForest"),
        ],
        ignore_index=True,
    )

    selected = select_non_overlapping(daily, step=5)

    assert sorted(selected["model"].unique()) == ["LinearRegression", "RandomForest"]
    assert selected["date"].nunique() == 2


def test_non_overlapping_selection_does_not_mutate_its_input():
    daily = _daily_frame(pd.bdate_range("2016-01-04", periods=12), 2016)
    original = daily.copy(deep=True)

    select_non_overlapping(daily, step=5)

    pd.testing.assert_frame_equal(daily, original)


@pytest.mark.parametrize("step", [0, -1])
def test_non_positive_step_is_rejected(step):
    daily = _daily_frame(pd.bdate_range("2016-01-04", periods=12), 2016)

    with pytest.raises(ValueError, match="step must be a positive integer"):
        select_non_overlapping(daily, step=step)


def _daily_rows(model, year, ic_values, spread_values):
    dates = pd.bdate_range("2016-01-04", periods=len(ic_values))
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": dates,
            "n_tickers": 15,
            "ic": list(ic_values),
            "top_mean": 0.0,
            "bottom_mean": 0.0,
            "spread": list(spread_values),
        }
    )


def test_summary_aggregates_only_defined_dates_and_reports_both_counts():
    daily = _daily_rows(
        "LinearRegression",
        2016,
        ic_values=[0.2, -0.1, float("nan"), 0.3],
        spread_values=[0.01, -0.02, float("nan"), float("nan")],
    )

    summary = summarize_ranking(daily)

    assert list(summary.columns) == [
        "model",
        "ic_days",
        "ic_mean",
        "ic_std",
        "ic_share_positive",
        "ic_stability",
        "spread_days",
        "spread_mean",
        "spread_share_positive",
    ]
    assert summary.loc[0, "ic_days"] == 3
    assert summary.loc[0, "spread_days"] == 2
    assert summary.loc[0, "ic_mean"] == pytest.approx((0.2 - 0.1 + 0.3) / 3)
    assert summary.loc[0, "ic_share_positive"] == pytest.approx(2 / 3)
    assert summary.loc[0, "spread_mean"] == pytest.approx((0.01 - 0.02) / 2)
    assert summary.loc[0, "spread_share_positive"] == pytest.approx(0.5)


def test_exactly_zero_is_not_counted_as_positive():
    daily = _daily_rows(
        "LinearRegression", 2016, ic_values=[0.0, 0.0], spread_values=[0.0, 0.0]
    )

    summary = summarize_ranking(daily)

    assert summary.loc[0, "ic_share_positive"] == pytest.approx(0.0)
    assert summary.loc[0, "spread_share_positive"] == pytest.approx(0.0)


def test_stability_is_the_sample_dispersion_ratio():
    daily = _daily_rows(
        "LinearRegression",
        2016,
        ic_values=[0.2, -0.1, 0.3],
        spread_values=[0.0, 0.0, 0.0],
    )

    summary = summarize_ranking(daily)
    values = pd.Series([0.2, -0.1, 0.3])

    assert summary.loc[0, "ic_std"] == pytest.approx(values.std(ddof=1))
    assert summary.loc[0, "ic_stability"] == pytest.approx(
        values.mean() / values.std(ddof=1)
    )


@pytest.mark.parametrize(
    "ic_values", [[0.2], [0.2, 0.2]], ids=["single-date", "zero-dispersion"]
)
def test_stability_is_undefined_without_usable_dispersion(ic_values):
    daily = _daily_rows(
        "LinearRegression", 2016, ic_values, spread_values=[0.0] * len(ic_values)
    )

    assert pd.isna(summarize_ranking(daily).loc[0, "ic_stability"])


def test_summary_rows_follow_known_model_order():
    daily = pd.concat(
        [
            _daily_rows("RandomForest", 2016, [0.1, 0.2], [0.0, 0.0]),
            _daily_rows("ZeroBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _daily_rows("MeanBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _daily_rows("LinearRegression", 2016, [0.1, 0.2], [0.0, 0.0]),
        ],
        ignore_index=True,
    )

    summary = summarize_ranking(daily)

    assert summary["model"].tolist() == [
        "ZeroBaseline",
        "MeanBaseline",
        "LinearRegression",
        "RandomForest",
    ]
    assert summary.loc[0, "ic_days"] == 0
    assert pd.isna(summary.loc[0, "ic_mean"])


def test_annual_summary_splits_by_year_and_matches_the_pooled_totals():
    daily = pd.concat(
        [
            _daily_rows("LinearRegression", 2016, [0.1, 0.3], [0.01, 0.03]),
            _daily_rows("LinearRegression", 2017, [0.2, 0.4], [0.02, 0.04]),
        ],
        ignore_index=True,
    )

    annual = summarize_ranking(daily, by_year=True)
    pooled = summarize_ranking(daily)

    assert list(annual.columns)[:2] == ["model", "validation_year"]
    assert annual["validation_year"].tolist() == [2016, 2017]
    assert annual["ic_days"].sum() == pooled.loc[0, "ic_days"]
    assert pooled.loc[0, "ic_mean"] == pytest.approx((0.1 + 0.3 + 0.2 + 0.4) / 4)


def test_summarize_ranking_does_not_mutate_its_input():
    daily = _daily_rows("LinearRegression", 2016, [0.1, 0.2], [0.01, 0.02])
    original = daily.copy(deep=True)

    summarize_ranking(daily, by_year=True)

    pd.testing.assert_frame_equal(daily, original)
