"""Orchestration for persistent market price updates."""

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd

from stock_forecaster.data.provider import MarketDataProvider
from stock_forecaster.data.storage import ParquetPriceStorage
from stock_forecaster.data.validation import validate_prices


def _sort_prices(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


class MarketDataPipeline:
    """Download, validate, merge, and persist normalized market prices."""

    def __init__(
        self,
        provider: MarketDataProvider,
        storage: ParquetPriceStorage,
    ) -> None:
        self.provider = provider
        self.storage = storage

    def update(
        self,
        tickers: Sequence[str],
        initial_start: str | date | datetime,
        end: str | date | datetime,
    ) -> pd.DataFrame:
        """Load initial history or incrementally update stored prices."""
        if not self.storage.exists():
            downloaded = self.provider.download_prices(
                tickers=tickers,
                start=initial_start,
                end=end,
            )
            validate_prices(downloaded, expected_tickers=tickers)
            result = _sort_prices(downloaded)
            self.storage.save(result)
            return result

        existing = self.storage.load()
        validate_prices(existing)

        normalized_tickers = [ticker.upper() for ticker in tickers]
        initial_start_date = pd.Timestamp(initial_start)
        end_date = pd.Timestamp(end)
        next_required_dates: list[pd.Timestamp] = []
        for ticker in normalized_tickers:
            ticker_dates = existing.loc[existing["ticker"] == ticker, "date"]
            if ticker_dates.empty:
                next_required_dates.append(initial_start_date)
            else:
                next_required_dates.append(
                    pd.Timestamp(ticker_dates.max()) + pd.Timedelta(days=1)
                )

        download_start = min(next_required_dates)
        if download_start >= end_date:
            return _sort_prices(existing)

        downloaded = self.provider.download_prices(
            tickers=tickers,
            start=download_start,
            end=end,
        )
        validate_prices(downloaded, expected_tickers=tickers)

        merged = pd.concat([existing, downloaded], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "ticker"], keep="last")
        merged = _sort_prices(merged)
        validate_prices(merged, expected_tickers=tickers)
        self.storage.save(merged)
        return merged
