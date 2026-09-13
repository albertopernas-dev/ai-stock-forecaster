from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pandas as pd
import pytest

from stock_forecaster.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN
from stock_forecaster.models.dataset import (
    DatasetPartition,
    TemporalDatasetSplit,
    prepare_supervised_data,
    temporal_split,
)


def _processed_rows(ticker_dates: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_number = 0
    for ticker, dates in ticker_dates.items():
        for value in dates:
            row: dict[str, object] = {
                "date": pd.Timestamp(value),
                "ticker": ticker,
                TARGET_COLUMN: float(10_000 + row_number),
            }
            row.update(
                {
                    feature: float(row_number * 100 + feature_number)
                    for feature_number, feature in enumerate(FEATURE_COLUMNS, start=1)
                }
            )
            rows.append(row)
            row_number += 1
    return pd.DataFrame(rows)


def _boundary_dates() -> list[str]:
    return [
        "2021-12-20",
        "2021-12-21",
        "2021-12-22",
        "2021-12-23",
        "2021-12-24",
        "2021-12-27",
        "2021-12-28",
        "2021-12-29",
        "2021-12-30",
        "2022-01-03",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
        "2022-01-07",
        "2023-12-18",
        "2023-12-19",
        "2023-12-20",
        "2023-12-21",
        "2023-12-22",
        "2023-12-26",
        "2023-12-27",
        "2023-12-28",
        "2023-12-29",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]


def _prepared_boundary_data() -> pd.DataFrame:
    return prepare_supervised_data(_processed_rows({"AAA": _boundary_dates()}))


def _split(supervised: pd.DataFrame) -> TemporalDatasetSplit:
    return temporal_split(
        supervised,
        train_end=date(2021, 12, 31),
        validation_start=datetime(2022, 1, 1),
        validation_end="2023-12-31",
        test_start="2024-01-01",
    )


def test_missing_required_columns_are_listed_clearly():
    frame = _processed_rows({"AAA": _boundary_dates()}).drop(
        columns=["ticker", "volatility_20d", TARGET_COLUMN]
    )

    with pytest.raises(ValueError) as error:
        prepare_supervised_data(frame)

    message = str(error.value)
    assert "ticker" in message
    assert "volatility_20d" in message
    assert TARGET_COLUMN in message


def test_duplicate_date_ticker_rows_are_rejected():
    frame = _processed_rows({"AAA": _boundary_dates()})
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
        prepare_supervised_data(frame)


def test_dates_are_normalized_before_duplicate_detection():
    frame = _processed_rows({"AAA": _boundary_dates()})
    frame["date"] = frame["date"].astype(object)
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, "date"] = "2021-12-20"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    assert isinstance(frame.loc[0, "date"], pd.Timestamp)
    assert isinstance(frame.loc[len(frame) - 1, "date"], str)

    with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
        prepare_supervised_data(frame)


def test_split_normalizes_dates_before_duplicate_detection():
    supervised = _prepared_boundary_data()
    supervised["date"] = supervised["date"].astype(object)
    duplicate = supervised.iloc[[0]].copy()
    duplicate.loc[:, "date"] = duplicate.iloc[0]["date"].strftime("%Y-%m-%d")
    supervised = pd.concat([supervised, duplicate], ignore_index=True)
    assert isinstance(supervised.loc[0, "date"], pd.Timestamp)
    assert isinstance(supervised.loc[len(supervised) - 1, "date"], str)

    with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
        _split(supervised)


def test_dates_are_normalized_before_ticker_local_sorting():
    frame = _processed_rows(
        {
            "AAA": [
                "2023-12-28",
                "2023-12-29",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        }
    )
    frame["date"] = frame["date"].dt.strftime("%m/%d/%Y")

    result = prepare_supervised_data(frame)

    target_dates = result.set_index("date")["target_end_date"]
    assert target_dates.loc[pd.Timestamp("2023-12-28")] == pd.Timestamp(
        "2024-01-05"
    )
    assert target_dates.loc[pd.Timestamp("2023-12-29")] == pd.Timestamp(
        "2024-01-08"
    )


def test_preparation_does_not_mutate_input_and_excludes_only_incomplete_rows():
    dates = pd.bdate_range("2024-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    frame = _processed_rows({"AAA": dates})
    frame.loc[1, FEATURE_COLUMNS[3]] = float("nan")
    frame.loc[2, TARGET_COLUMN] = float("nan")
    original = frame.copy(deep=True)

    result = prepare_supervised_data(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert result["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-09"),
        pd.Timestamp("2024-01-10"),
    ]
    assert result.loc[:, FEATURE_COLUMNS].notna().all().all()
    assert result[TARGET_COLUMN].notna().all()
    assert result["target_end_date"].notna().all()


def test_target_end_date_is_fifth_later_observation_within_each_ticker():
    frame = _processed_rows(
        {
            "AAA": [
                "2021-12-20",
                "2021-12-21",
                "2021-12-23",
                "2021-12-27",
                "2021-12-28",
                "2021-12-30",
                "2022-01-03",
            ],
            "BBB": [
                "2021-12-17",
                "2021-12-20",
                "2021-12-22",
                "2021-12-24",
                "2021-12-29",
                "2022-01-04",
                "2022-01-05",
                "2022-01-07",
            ],
        }
    ).iloc[::-1]

    result = prepare_supervised_data(frame)
    actual = {
        (row.ticker, row.date.strftime("%Y-%m-%d")): row.target_end_date.strftime(
            "%Y-%m-%d"
        )
        for row in result.itertuples()
    }

    assert actual == {
        ("BBB", "2021-12-17"): "2022-01-04",
        ("AAA", "2021-12-20"): "2021-12-30",
        ("BBB", "2021-12-20"): "2022-01-05",
        ("AAA", "2021-12-21"): "2022-01-03",
        ("BBB", "2021-12-22"): "2022-01-07",
    }


def test_prepared_output_has_exact_columns_and_deterministic_order():
    frame = _processed_rows(
        {
            "BBB": pd.bdate_range("2024-01-02", periods=7)
            .strftime("%Y-%m-%d")
            .tolist(),
            "AAA": pd.bdate_range("2024-01-02", periods=7)
            .strftime("%Y-%m-%d")
            .tolist(),
        }
    ).sample(frac=1, random_state=7)

    result = prepare_supervised_data(frame)

    assert list(result.columns) == [
        "date",
        "ticker",
        "target_end_date",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    assert list(result[["date", "ticker"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2024-01-02"), "AAA"),
        (pd.Timestamp("2024-01-02"), "BBB"),
        (pd.Timestamp("2024-01-03"), "AAA"),
        (pd.Timestamp("2024-01-03"), "BBB"),
    ]
    assert result.index.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("boundaries", "expected_message"),
    [
        (
            ("2022-01-01", "2022-01-01", "2023-12-31", "2024-01-01"),
            "train_end must be earlier than validation_start",
        ),
        (
            ("2021-12-31", "2023-01-01", "2022-12-31", "2024-01-01"),
            "validation_start must not be later than validation_end",
        ),
        (
            ("2021-12-31", "2022-01-01", "2024-01-01", "2024-01-01"),
            "validation_end must be earlier than test_start",
        ),
    ],
)
def test_invalid_split_boundary_ordering_is_rejected(boundaries, expected_message):
    supervised = _prepared_boundary_data()

    with pytest.raises(ValueError, match=expected_message):
        temporal_split(
            supervised,
            train_end=boundaries[0],
            validation_start=boundaries[1],
            validation_end=boundaries[2],
            test_start=boundaries[3],
        )


def test_split_date_ranges_and_target_horizons_respect_boundaries():
    split = _split(_prepared_boundary_data())

    assert split.train.metadata["date"].max() <= pd.Timestamp("2021-12-31")
    assert split.validation.metadata["date"].between(
        pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")
    ).all()
    assert split.test.metadata["date"].min() >= pd.Timestamp("2024-01-01")

    prepared = _prepared_boundary_data().set_index(["date", "ticker"])
    train_horizons = [
        prepared.loc[(row.date, row.ticker), "target_end_date"]
        for row in split.train.metadata.itertuples()
    ]
    validation_horizons = [
        prepared.loc[(row.date, row.ticker), "target_end_date"]
        for row in split.validation.metadata.itertuples()
    ]
    assert max(train_horizons) < pd.Timestamp("2022-01-01")
    assert max(validation_horizons) < pd.Timestamp("2024-01-01")


def test_train_purge_uses_observations_and_removes_crossing_horizon():
    supervised = _prepared_boundary_data()
    target_dates = supervised.set_index("date")["target_end_date"]
    assert target_dates.loc[pd.Timestamp("2021-12-20")] == pd.Timestamp(
        "2021-12-27"
    )
    assert target_dates.loc[pd.Timestamp("2021-12-24")] == pd.Timestamp(
        "2022-01-03"
    )
    assert pd.Timestamp("2021-12-24") + pd.Timedelta(days=5) < pd.Timestamp(
        "2022-01-01"
    )

    split = _split(supervised)
    train_dates = set(split.train.metadata["date"])

    assert pd.Timestamp("2021-12-20") in train_dates
    assert pd.Timestamp("2021-12-24") not in train_dates


def test_validation_purge_removes_horizon_that_enters_test():
    supervised = _prepared_boundary_data()
    target_dates = supervised.set_index("date")["target_end_date"]
    assert target_dates.loc[pd.Timestamp("2023-12-18")] == pd.Timestamp(
        "2023-12-26"
    )
    assert target_dates.loc[pd.Timestamp("2023-12-22")] == pd.Timestamp(
        "2024-01-02"
    )

    split = _split(supervised)
    validation_dates = set(split.validation.metadata["date"])

    assert pd.Timestamp("2023-12-18") in validation_dates
    assert pd.Timestamp("2023-12-22") not in validation_dates


def test_candidate_boundaries_are_inclusive_and_purge_boundaries_are_exclusive():
    supervised = _prepared_boundary_data()
    split = temporal_split(
        supervised,
        train_end="2021-12-20",
        validation_start="2022-01-03",
        validation_end="2023-12-18",
        test_start="2024-01-02",
    )

    assert pd.Timestamp("2021-12-20") in set(split.train.metadata["date"])
    assert pd.Timestamp("2022-01-03") in set(split.validation.metadata["date"])
    assert pd.Timestamp("2023-12-18") in set(split.validation.metadata["date"])
    assert pd.Timestamp("2024-01-02") in set(split.test.metadata["date"])

    equality_split = temporal_split(
        supervised,
        train_end="2021-12-31",
        validation_start="2022-01-03",
        validation_end="2023-12-31",
        test_start="2024-01-02",
    )
    assert pd.Timestamp("2021-12-24") not in set(
        equality_split.train.metadata["date"]
    )
    assert pd.Timestamp("2023-12-22") not in set(
        equality_split.validation.metadata["date"]
    )


def test_partition_contract_is_exact_aligned_complete_and_chronological():
    processed = _processed_rows({"AAA": _boundary_dates()})
    supervised = prepare_supervised_data(processed)
    original = supervised.copy(deep=True)
    split = _split(supervised)
    expected_targets = processed.set_index(["date", "ticker"])[TARGET_COLUMN]
    expected_features = processed.set_index(["date", "ticker"])

    pd.testing.assert_frame_equal(supervised, original)

    assert isinstance(split, TemporalDatasetSplit)
    for partition in (split.train, split.validation, split.test):
        assert isinstance(partition, DatasetPartition)
        assert list(partition.X.columns) == list(FEATURE_COLUMNS)
        assert partition.y.name == TARGET_COLUMN
        assert list(partition.metadata.columns) == ["date", "ticker"]
        assert not {
            "date",
            "ticker",
            "target_end_date",
            TARGET_COLUMN,
        }.intersection(partition.X.columns)
        assert len(partition.X) == len(partition.y) == len(partition.metadata)
        assert partition.X.index.equals(partition.y.index)
        assert partition.X.index.equals(partition.metadata.index)
        assert partition.X.notna().all().all()
        assert partition.y.notna().all()
        assert partition.metadata.notna().all().all()
        assert partition.metadata["date"].is_monotonic_increasing
        for index, row in partition.metadata.iterrows():
            key = (row["date"], row["ticker"])
            assert partition.y.iloc[index] == expected_targets.loc[key]
            assert partition.X.iloc[index].tolist() == expected_features.loc[
                key, FEATURE_COLUMNS
            ].tolist()


def test_partition_dataclasses_are_frozen():
    split = _split(_prepared_boundary_data())

    with pytest.raises(FrozenInstanceError):
        split.train = split.test
    with pytest.raises(FrozenInstanceError):
        split.train.X = split.test.X


def test_cross_ticker_purge_uses_each_tickers_own_observations():
    ticker_dates = {
        "AAA": [
            "2021-12-20",
            "2021-12-21",
            "2021-12-22",
            "2021-12-23",
            "2021-12-24",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-07",
            *_boundary_dates()[14:],
        ],
        "BBB": [
            "2021-12-20",
            "2021-12-21",
            "2021-12-23",
            "2021-12-24",
            "2021-12-27",
            "2021-12-28",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-07",
            *_boundary_dates()[14:],
        ],
    }
    supervised = prepare_supervised_data(_processed_rows(ticker_dates))
    horizons = supervised.set_index(["ticker", "date"])["target_end_date"]
    assert horizons.loc[("AAA", pd.Timestamp("2021-12-20"))] == pd.Timestamp(
        "2022-01-03"
    )
    assert horizons.loc[("BBB", pd.Timestamp("2021-12-20"))] == pd.Timestamp(
        "2021-12-28"
    )

    split = _split(supervised)
    train_keys = set(split.train.metadata.itertuples(index=False, name=None))

    assert (pd.Timestamp("2021-12-20"), "AAA") not in train_keys
    assert (pd.Timestamp("2021-12-20"), "BBB") in train_keys


@pytest.mark.parametrize(
    ("boundaries", "empty_name"),
    [
        (
            ("2020-12-31", "2021-01-01", "2023-12-31", "2024-01-01"),
            "TRAIN",
        ),
        (
            ("2021-12-31", "2022-02-01", "2022-11-30", "2024-01-01"),
            "VALIDATION",
        ),
        (
            ("2021-12-31", "2022-01-01", "2023-12-31", "2025-01-01"),
            "TEST",
        ),
    ],
)
def test_empty_partition_raises_clearly(boundaries, empty_name):
    with pytest.raises(ValueError, match=rf"{empty_name} partition is empty"):
        temporal_split(
            _prepared_boundary_data(),
            train_end=boundaries[0],
            validation_start=boundaries[1],
            validation_end=boundaries[2],
            test_start=boundaries[3],
        )
