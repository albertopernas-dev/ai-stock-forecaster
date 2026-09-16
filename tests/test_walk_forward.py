from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from stock_forecaster.features.engineering import (
    EXCESS_TARGET_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from stock_forecaster.models.dataset import (
    DatasetPartition,
    WalkForwardFold,
    walk_forward_splits,
)

VALIDATION_YEARS = tuple(range(2016, 2024))


def _supervised_rows(
    *,
    years=range(2015, 2024),
    tickers=("AAA", "BBB"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_number = 0
    for year in years:
        for month_day in ((1, 12), (6, 15)):
            for ticker in tickers:
                observed = pd.Timestamp(year=year, month=month_day[0], day=month_day[1])
                row = {
                    "date": observed,
                    "ticker": ticker,
                    "target_end_date": observed + pd.Timedelta(days=7),
                    TARGET_COLUMN: float(1000 + row_number),
                }
                row.update(
                    {
                        feature: float(row_number * 100 + index)
                        for index, feature in enumerate(FEATURE_COLUMNS, start=1)
                    }
                )
                rows.append(row)
                row_number += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=23).reset_index(drop=True)


def _target_ends(source: pd.DataFrame, partition: DatasetPartition) -> pd.Series:
    source_index = source.set_index(["date", "ticker"])
    keys = pd.MultiIndex.from_frame(partition.metadata[["date", "ticker"]])
    return pd.Series(source_index.loc[keys, "target_end_date"].to_numpy())


def test_walk_forward_splits_builds_chronological_annual_folds_and_partitions():
    supervised = _supervised_rows()
    years = tuple(range(2016, 2024))

    folds = walk_forward_splits(supervised, years)

    assert isinstance(folds, tuple)
    assert [fold.validation_year for fold in folds] == list(years)
    train_sizes = [len(fold.train.metadata) for fold in folds]
    assert train_sizes == sorted(train_sizes)
    assert all(left < right for left, right in zip(train_sizes, train_sizes[1:]))

    source = supervised.set_index(["date", "ticker"])
    for fold in folds:
        validation_start = pd.Timestamp(year=fold.validation_year, month=1, day=1)
        next_year_start = pd.Timestamp(year=fold.validation_year + 1, month=1, day=1)
        assert (fold.train.metadata["date"] < validation_start).all()
        assert fold.validation.metadata["date"].between(
            validation_start, next_year_start, inclusive="left"
        ).all()

        for partition in (fold.train, fold.validation):
            assert isinstance(partition, DatasetPartition)
            assert list(partition.X.columns) == list(FEATURE_COLUMNS)
            assert partition.y.name == TARGET_COLUMN
            assert list(partition.metadata.columns) == ["date", "ticker"]
            assert partition.X.index.equals(partition.y.index)
            assert partition.X.index.equals(partition.metadata.index)
            assert partition.metadata["date"].is_monotonic_increasing
            assert not partition.metadata.duplicated(["date", "ticker"]).any()
            for index, row in partition.metadata.iterrows():
                source_row = source.loc[(row["date"], row["ticker"])]
                assert partition.X.loc[index].tolist() == source_row.loc[
                    list(FEATURE_COLUMNS)
                ].tolist()
                assert partition.y.loc[index] == source_row[TARGET_COLUMN]


def test_walk_forward_splits_use_the_only_approved_target_for_every_y():
    supervised = _supervised_rows()
    supervised[EXCESS_TARGET_COLUMN] = supervised[TARGET_COLUMN] + 0.5
    supervised = supervised.drop(columns=[TARGET_COLUMN])
    folds = walk_forward_splits(supervised, (2016, 2017))

    for fold in folds:
        for partition in (fold.train, fold.validation):
            assert partition.y.name == EXCESS_TARGET_COLUMN
            source = supervised.set_index(["date", "ticker"])
            expected = [
                source.loc[(row.date, row.ticker), EXCESS_TARGET_COLUMN]
                for row in partition.metadata.itertuples()
            ]
            assert partition.y.tolist() == expected


def test_walk_forward_folds_are_frozen_and_do_not_mutate_supervised_input():
    supervised = _supervised_rows()
    original = supervised.copy(deep=True)

    folds = walk_forward_splits(supervised, [2016])

    assert isinstance(folds[0], WalkForwardFold)
    with pytest.raises(FrozenInstanceError):
        folds[0].validation_year = 2017
    pd.testing.assert_frame_equal(supervised, original)


@pytest.mark.parametrize("target_column", [TARGET_COLUMN, EXCESS_TARGET_COLUMN])
def test_train_and_validation_use_actual_target_end_date_for_purging(
    target_column,
):
    supervised = _supervised_rows(years=range(2015, 2017), tickers=("AAA",))
    if target_column == EXCESS_TARGET_COLUMN:
        supervised[EXCESS_TARGET_COLUMN] = supervised[TARGET_COLUMN] + 0.5
        supervised = supervised.drop(columns=[TARGET_COLUMN])
    crossing_train = supervised.iloc[[0]].copy()
    crossing_train.loc[:, "date"] = pd.Timestamp("2015-12-28")
    crossing_train.loc[:, "target_end_date"] = pd.Timestamp("2016-01-04")
    crossing_validation = supervised.iloc[[1]].copy()
    crossing_validation.loc[:, "date"] = pd.Timestamp("2016-12-28")
    crossing_validation.loc[:, "target_end_date"] = pd.Timestamp("2017-01-04")
    supervised = pd.concat(
        [supervised, crossing_train, crossing_validation], ignore_index=True
    )

    fold = walk_forward_splits(supervised, (2016,))[0]
    train_keys = set(fold.train.metadata.itertuples(index=False, name=None))
    validation_keys = set(fold.validation.metadata.itertuples(index=False, name=None))

    assert (pd.Timestamp("2015-12-28"), "AAA") not in train_keys
    assert (pd.Timestamp("2016-12-28"), "AAA") not in validation_keys
    assert _target_ends(supervised, fold.train).lt("2016-01-01").all()
    assert _target_ends(supervised, fold.validation).lt("2017-01-01").all()


def test_late_2023_target_using_2024_is_excluded():
    supervised = _supervised_rows(tickers=("AAA",))
    late_row = supervised.iloc[[0]].copy()
    late_row.loc[:, "date"] = pd.Timestamp("2023-12-28")
    late_row.loc[:, "target_end_date"] = pd.Timestamp("2024-01-05")
    supervised = pd.concat([supervised, late_row], ignore_index=True)

    fold = walk_forward_splits(supervised, (2023,))[0]

    assert pd.Timestamp("2023-12-28") not in set(fold.validation.metadata["date"])
    assert _target_ends(supervised, fold.validation).lt("2024-01-01").all()


def test_approved_development_folds_retain_no_2024_dates_or_target_ends():
    supervised = _supervised_rows(years=range(2015, 2025))
    development = supervised.loc[
        supervised["date"].lt("2024-01-01")
        & supervised["target_end_date"].lt("2024-01-01")
    ].copy()
    folds = walk_forward_splits(development, VALIDATION_YEARS)

    assert all(
        partition.metadata["date"].lt("2024-01-01").all()
        for fold in folds
        for partition in (fold.train, fold.validation)
    )
    assert all(
        _target_ends(development, partition).lt("2024-01-01").all()
        for fold in folds
        for partition in (fold.train, fold.validation)
    )


@pytest.mark.parametrize(
    ("years", "message"),
    [
        ((), "must not be empty"),
        ((2016, 2016), "must be unique"),
        ((2017, 2016), "strictly increasing"),
        ((2016, "2017"), "only integers"),
        ((2016, True), "only integers"),
    ],
)
def test_invalid_validation_years_are_rejected_without_repair(years, message):
    with pytest.raises(ValueError, match=message):
        walk_forward_splits(_supervised_rows(), years)


def test_missing_supervised_columns_are_rejected():
    supervised = _supervised_rows().drop(columns=[FEATURE_COLUMNS[-1]])
    with pytest.raises(ValueError, match=FEATURE_COLUMNS[-1]):
        walk_forward_splits(supervised, (2016,))


def test_duplicate_date_ticker_rows_are_rejected():
    supervised = _supervised_rows()
    supervised = pd.concat([supervised, supervised.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
        walk_forward_splits(supervised, (2016,))


def test_splitter_does_not_mutate_input():
    supervised = _supervised_rows()
    original = supervised.copy(deep=True)
    walk_forward_splits(supervised, VALIDATION_YEARS)
    pd.testing.assert_frame_equal(supervised, original)


def test_empty_train_and_validation_partitions_raise_clearly():
    with pytest.raises(ValueError, match="TRAIN 2015 partition is empty"):
        walk_forward_splits(_supervised_rows(years=range(2015, 2017)), (2015,))
    with pytest.raises(ValueError, match="VALIDATION 2014 partition is empty"):
        walk_forward_splits(_supervised_rows(years=range(2013, 2014)), (2014,))


def test_changing_2023_input_cannot_change_folds_2016_through_2022():
    original = _supervised_rows()
    changed = original.copy(deep=True)
    mask_2023 = changed["date"].dt.year.eq(2023)
    changed.loc[mask_2023, FEATURE_COLUMNS[0]] += 999.0
    changed.loc[mask_2023, TARGET_COLUMN] -= 999.0

    before = walk_forward_splits(original, VALIDATION_YEARS)
    after = walk_forward_splits(changed, VALIDATION_YEARS)

    for before_fold, after_fold in zip(before[:-1], after[:-1]):
        assert before_fold.validation_year == after_fold.validation_year
        for before_partition, after_partition in (
            (before_fold.train, after_fold.train),
            (before_fold.validation, after_fold.validation),
        ):
            pd.testing.assert_frame_equal(before_partition.X, after_partition.X)
            pd.testing.assert_series_equal(before_partition.y, after_partition.y)
            pd.testing.assert_frame_equal(
                before_partition.metadata, after_partition.metadata
            )
