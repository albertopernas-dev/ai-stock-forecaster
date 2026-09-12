from pathlib import Path

import pandas as pd
import pytest

from stock_forecaster.data.storage import ParquetPriceStorage


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-09-11", "2026-09-10", "2026-09-10"]
            ),
            "ticker": ["MSFT", "MSFT", "AAPL"],
            "open": [201.0, 200.0, 100.0],
            "high": [203.0, 202.0, 102.0],
            "low": [200.0, 199.0, 99.0],
            "close": [202.0, 201.0, 101.0],
            "adjusted_close": [201.5, 200.5, 100.5],
            "volume": [2_100, 2_000, 1_000],
        }
    )


def test_exists_is_false_before_saving(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")

    assert storage.exists() is False


def test_save_creates_parent_directory_and_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "prices.parquet"
    storage = ParquetPriceStorage(path)

    storage.save(_prices())

    assert path.is_file()
    assert storage.exists() is True


def test_save_and_load_round_trip_preserves_data(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    expected = _prices().sort_values(["date", "ticker"]).reset_index(drop=True)

    storage.save(_prices())
    result = storage.load()

    pd.testing.assert_frame_equal(result, expected)


def test_loaded_date_is_datetime_like(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(_prices())

    result = storage.load()

    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_load_returns_rows_sorted_by_date_and_ticker(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(_prices())

    result = storage.load()

    assert list(result[["date", "ticker"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2026-09-10"), "AAPL"),
        (pd.Timestamp("2026-09-10"), "MSFT"),
        (pd.Timestamp("2026-09-11"), "MSFT"),
    ]
    assert result.index.tolist() == [0, 1, 2]


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    storage = ParquetPriceStorage(path)

    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        storage.load()
