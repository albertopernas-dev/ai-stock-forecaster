"""Leakage-safe supervised dataset preparation and temporal splitting."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stock_forecaster.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN

_INPUT_COLUMNS = ("date", "ticker", *FEATURE_COLUMNS, TARGET_COLUMN)
_SUPERVISED_COLUMNS = (
    "date",
    "ticker",
    "target_end_date",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)


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


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _reject_duplicate_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate date and ticker rows")


def prepare_supervised_data(features: pd.DataFrame) -> pd.DataFrame:
    """Return complete supervised rows with ticker-local target horizon dates."""
    _require_columns(features, _INPUT_COLUMNS)
    ordered = features.loc[:, _INPUT_COLUMNS].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    _reject_duplicate_keys(ordered)
    ordered = ordered.sort_values(["ticker", "date"]).reset_index(drop=True)
    ordered["target_end_date"] = ordered.groupby("ticker", sort=False)[
        "date"
    ].shift(-5)

    complete_columns = [*FEATURE_COLUMNS, TARGET_COLUMN, "target_end_date"]
    complete = ordered.dropna(subset=complete_columns)
    return (
        complete.loc[:, _SUPERVISED_COLUMNS]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def _partition(rows: pd.DataFrame, name: str) -> DatasetPartition:
    if rows.empty:
        raise ValueError(f"{name} partition is empty")

    ordered = rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    return DatasetPartition(
        X=ordered.loc[:, FEATURE_COLUMNS].copy(),
        y=ordered.loc[:, TARGET_COLUMN].copy(),
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

    _require_columns(supervised, _SUPERVISED_COLUMNS)
    ordered = supervised.loc[:, _SUPERVISED_COLUMNS].copy()
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
        train=_partition(train_rows, "TRAIN"),
        validation=_partition(validation_rows, "VALIDATION"),
        test=_partition(test_rows, "TEST"),
    )
