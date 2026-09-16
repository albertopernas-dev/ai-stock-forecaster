"""Leakage-safe supervised dataset preparation and temporal splitting."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stock_forecaster.features.engineering import (
    EXCESS_TARGET_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

_APPROVED_TARGET_COLUMNS = (TARGET_COLUMN, EXCESS_TARGET_COLUMN)


@dataclass(frozen=True)
class DatasetPartition:
    """Aligned model inputs, target values, and observation identifiers."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class TemporalDatasetSplit:
    """Chronological train, validation, and test partitions."""

    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition


@dataclass(frozen=True)
class WalkForwardFold:
    """One annual validation fold with all prior observations as training data."""

    validation_year: int
    train: DatasetPartition
    validation: DatasetPartition


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _validate_target_column(target_column: str) -> None:
    if target_column not in _APPROVED_TARGET_COLUMNS:
        allowed = ", ".join(_APPROVED_TARGET_COLUMNS)
        raise ValueError(
            f"Unsupported target column {target_column!r}; expected one of: {allowed}"
        )


def _input_columns(target_column: str) -> tuple[str, ...]:
    return ("date", "ticker", *FEATURE_COLUMNS, target_column)


def _supervised_columns(target_column: str) -> tuple[str, ...]:
    return (
        "date",
        "ticker",
        "target_end_date",
        *FEATURE_COLUMNS,
        target_column,
    )


def _infer_target_column(frame: pd.DataFrame) -> str:
    present = tuple(
        target
        for target in _APPROVED_TARGET_COLUMNS
        if target in frame.columns
    )
    if len(present) != 1:
        raise ValueError(
            "Prepared data must contain exactly one approved target column"
        )
    return present[0]


def _reject_duplicate_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate date and ticker rows")


def prepare_supervised_data(
    features: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Return complete supervised rows with ticker-local target horizon dates."""
    _validate_target_column(target_column)
    input_columns = _input_columns(target_column)
    supervised_columns = _supervised_columns(target_column)
    _require_columns(features, input_columns)
    ordered = features.loc[:, input_columns].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    _reject_duplicate_keys(ordered)
    ordered = ordered.sort_values(["ticker", "date"]).reset_index(drop=True)
    ordered["target_end_date"] = ordered.groupby("ticker", sort=False)[
        "date"
    ].shift(-5)

    complete_columns = [*FEATURE_COLUMNS, target_column, "target_end_date"]
    complete = ordered.dropna(subset=complete_columns)
    return (
        complete.loc[:, supervised_columns]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def _partition(
    rows: pd.DataFrame,
    name: str,
    target_column: str,
) -> DatasetPartition:
    if rows.empty:
        raise ValueError(f"{name} partition is empty")

    ordered = rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    return DatasetPartition(
        X=ordered.loc[:, FEATURE_COLUMNS].copy(),
        y=ordered.loc[:, target_column].copy(),
        metadata=ordered.loc[:, ["date", "ticker"]].copy(),
    )


def temporal_split(
    supervised: pd.DataFrame,
    *,
    train_end: str | date | datetime,
    validation_start: str | date | datetime,
    validation_end: str | date | datetime,
    test_start: str | date | datetime,
) -> TemporalDatasetSplit:
    """Split supervised rows chronologically and purge crossing target horizons."""
    train_end_timestamp = pd.Timestamp(train_end)
    validation_start_timestamp = pd.Timestamp(validation_start)
    validation_end_timestamp = pd.Timestamp(validation_end)
    test_start_timestamp = pd.Timestamp(test_start)

    if train_end_timestamp >= validation_start_timestamp:
        raise ValueError("train_end must be earlier than validation_start")
    if validation_start_timestamp > validation_end_timestamp:
        raise ValueError(
            "validation_start must not be later than validation_end"
        )
    if validation_end_timestamp >= test_start_timestamp:
        raise ValueError("validation_end must be earlier than test_start")

    target_column = _infer_target_column(supervised)
    supervised_columns = _supervised_columns(target_column)
    _require_columns(supervised, supervised_columns)
    ordered = supervised.loc[:, supervised_columns].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered["target_end_date"] = pd.to_datetime(ordered["target_end_date"])
    _reject_duplicate_keys(ordered)

    train_rows = ordered.loc[
        (ordered["date"] <= train_end_timestamp)
        & (ordered["target_end_date"] < validation_start_timestamp)
    ]
    validation_rows = ordered.loc[
        ordered["date"].between(
            validation_start_timestamp,
            validation_end_timestamp,
        )
        & (ordered["target_end_date"] < test_start_timestamp)
    ]
    test_rows = ordered.loc[ordered["date"] >= test_start_timestamp]

    return TemporalDatasetSplit(
        train=_partition(train_rows, "TRAIN", target_column),
        validation=_partition(validation_rows, "VALIDATION", target_column),
        test=_partition(test_rows, "TEST", target_column),
    )


def walk_forward_splits(
    supervised_data: pd.DataFrame,
    validation_years: Sequence[int],
) -> tuple[WalkForwardFold, ...]:
    """Build annual walk-forward train and validation partitions."""
    years = tuple(validation_years)
    if not years:
        raise ValueError("validation_years must not be empty")
    if any(type(year) is not int for year in years):
        raise ValueError("validation_years must contain only integers")
    if len(set(years)) != len(years):
        raise ValueError("validation_years must be unique")
    if any(left >= right for left, right in zip(years, years[1:])):
        raise ValueError("validation_years must be strictly increasing")
    target_column = _infer_target_column(supervised_data)
    supervised_columns = _supervised_columns(target_column)
    _require_columns(supervised_data, supervised_columns)
    ordered = supervised_data.loc[:, supervised_columns].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered["target_end_date"] = pd.to_datetime(ordered["target_end_date"])
    _reject_duplicate_keys(ordered)

    folds = []
    for year in years:
        validation_start = pd.Timestamp(year=year, month=1, day=1)
        next_year_start = pd.Timestamp(year=year + 1, month=1, day=1)
        train_rows = ordered.loc[
            ordered["date"].lt(validation_start)
            & ordered["target_end_date"].lt(validation_start)
        ]
        validation_rows = ordered.loc[
            (ordered["date"] >= validation_start)
            & (ordered["date"] < next_year_start)
            & ordered["target_end_date"].lt(next_year_start)
        ]
        folds.append(
            WalkForwardFold(
                validation_year=year,
                train=_partition(train_rows, f"TRAIN {year}", target_column),
                validation=_partition(
                    validation_rows, f"VALIDATION {year}", target_column
                ),
            )
        )
    return tuple(folds)
