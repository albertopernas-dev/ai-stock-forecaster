from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from stock_forecaster.data.pipeline import MarketDataPipeline
from stock_forecaster.data.provider import MarketDataProvider
from stock_forecaster.data.storage import ParquetPriceStorage


def _prices(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(row_date),
                "ticker": ticker,
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "adjusted_close": close - 0.5,
                "volume": 1_000,
            }
            for row_date, ticker, close in rows
        ]
    )


class RecordingProvider(MarketDataProvider):
    def __init__(self, result: pd.DataFrame | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def download_prices(
        self,
        tickers: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> pd.DataFrame:
        self.calls.append({"tickers": list(tickers), "start": start, "end": end})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result.copy(deep=True)


def test_first_run_downloads_from_initial_start(tmp_path: Path) -> None:
    provider = RecordingProvider(_prices(("2026-09-10", "AAPL", 101.0)))
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    pipeline.update(
        tickers=["AAPL"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    assert provider.calls == [
        {
            "tickers": ["AAPL"],
            "start": "2010-01-01",
            "end": "2026-09-12",
        }
    ]


def test_first_run_validates_and_persists_data(tmp_path: Path) -> None:
    provider = RecordingProvider(
        _prices(
            ("2026-09-11", "MSFT", 201.0),
            ("2026-09-10", "AAPL", 101.0),
        )
    )
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    result = pipeline.update(
        tickers=["AAPL", "MSFT"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    assert storage.exists()
    pd.testing.assert_frame_equal(storage.load(), result)
    assert result["ticker"].tolist() == ["AAPL", "MSFT"]


def test_invalid_first_download_is_not_persisted(tmp_path: Path) -> None:
    invalid = _prices(("2026-09-10", "AAPL", 101.0))
    invalid.loc[0, "open"] = 0.0
    provider = RecordingProvider(invalid)
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    with pytest.raises(ValueError, match="open must be greater than zero"):
        pipeline.update(
            tickers=["AAPL"],
            initial_start="2010-01-01",
            end="2026-09-12",
        )

    assert storage.exists() is False


def test_incremental_download_starts_after_oldest_per_ticker_latest_date(
    tmp_path: Path,
) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(
        _prices(
            ("2026-09-10", "AAPL", 101.0),
            ("2026-09-08", "MSFT", 201.0),
        )
    )
    provider = RecordingProvider(
        _prices(
            ("2026-09-11", "AAPL", 102.0),
            ("2026-09-09", "MSFT", 202.0),
        )
    )
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    pipeline.update(
        tickers=["AAPL", "MSFT"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    assert pd.Timestamp(provider.calls[0]["start"]) == pd.Timestamp("2026-09-09")


def test_provider_is_not_called_when_stored_data_is_current(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(_prices(("2026-09-10", "AAPL", 101.0)))
    provider = RecordingProvider(RuntimeError("provider should not be called"))
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    result = pipeline.update(
        tickers=["AAPL"],
        initial_start="2010-01-01",
        end="2026-09-11",
    )

    assert provider.calls == []
    pd.testing.assert_frame_equal(result, storage.load())


def test_overlapping_downloaded_row_is_deduplicated_and_wins(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(_prices(("2026-09-10", "AAPL", 101.0)))
    provider = RecordingProvider(
        _prices(
            ("2026-09-10", "AAPL", 105.0),
            ("2026-09-11", "AAPL", 106.0),
        )
    )
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    result = pipeline.update(
        tickers=["AAPL"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    corrected = result.loc[result["date"] == pd.Timestamp("2026-09-10")]
    assert len(result) == 2
    assert corrected["close"].item() == 105.0


def test_incremental_result_is_sorted_by_date_and_ticker(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(
        _prices(
            ("2026-09-10", "MSFT", 201.0),
            ("2026-09-10", "AAPL", 101.0),
        )
    )
    provider = RecordingProvider(
        _prices(
            ("2026-09-11", "MSFT", 202.0),
            ("2026-09-11", "AAPL", 102.0),
        )
    )
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    result = pipeline.update(
        tickers=["AAPL", "MSFT"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    assert list(result[["date", "ticker"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2026-09-10"), "AAPL"),
        (pd.Timestamp("2026-09-10"), "MSFT"),
        (pd.Timestamp("2026-09-11"), "AAPL"),
        (pd.Timestamp("2026-09-11"), "MSFT"),
    ]
    assert result.index.tolist() == [0, 1, 2, 3]


def test_new_ticker_triggers_full_backfill_and_is_merged(tmp_path: Path) -> None:
    storage = ParquetPriceStorage(tmp_path / "prices.parquet")
    storage.save(_prices(("2026-09-10", "AAPL", 101.0)))
    provider = RecordingProvider(
        _prices(
            ("2026-09-11", "AAPL", 102.0),
            ("2010-01-04", "NVDA", 20.0),
        )
    )
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    result = pipeline.update(
        tickers=["AAPL", "NVDA"],
        initial_start="2010-01-01",
        end="2026-09-12",
    )

    assert pd.Timestamp(provider.calls[0]["start"]) == pd.Timestamp("2010-01-01")
    assert "NVDA" in set(result["ticker"])


def test_invalid_incremental_download_does_not_overwrite_existing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prices.parquet"
    storage = ParquetPriceStorage(path)
    storage.save(_prices(("2026-09-10", "AAPL", 101.0)))
    original_bytes = path.read_bytes()
    invalid = _prices(("2026-09-11", "AAPL", 102.0))
    invalid.loc[0, "high"] = 100.0
    provider = RecordingProvider(invalid)
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    with pytest.raises(ValueError, match="high must be at least open"):
        pipeline.update(
            tickers=["AAPL"],
            initial_start="2010-01-01",
            end="2026-09-12",
        )

    assert path.read_bytes() == original_bytes


def test_provider_failure_does_not_overwrite_existing(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    storage = ParquetPriceStorage(path)
    storage.save(_prices(("2026-09-10", "AAPL", 101.0)))
    original_bytes = path.read_bytes()
    provider = RecordingProvider(RuntimeError("Yahoo unavailable"))
    pipeline = MarketDataPipeline(provider=provider, storage=storage)

    with pytest.raises(RuntimeError, match="Yahoo unavailable"):
        pipeline.update(
            tickers=["AAPL"],
            initial_start="2010-01-01",
            end="2026-09-12",
        )

    assert path.read_bytes() == original_bytes
