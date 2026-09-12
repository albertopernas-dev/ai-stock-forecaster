"""Yahoo Finance market data provider."""

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from stock_forecaster.data.provider import MarketDataProvider

_INTERNAL_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]
_YAHOO_COLUMN_NAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


def _normalize_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(ticker, axis="columns", level=-1)

    normalized = frame.rename_axis("date").reset_index()
    normalized = normalized.rename(columns=_YAHOO_COLUMN_NAMES)
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["ticker"] = ticker
    return normalized[_INTERNAL_COLUMNS].sort_values(["date", "ticker"]).reset_index(
        drop=True
    )


class YahooFinanceProvider(MarketDataProvider):
    """Download historical daily prices from Yahoo Finance."""

    def download_prices(
        self,
        tickers: Sequence[str],
        start: str | date | datetime,
        end: str | date | datetime,
    ) -> pd.DataFrame:
        """Download and normalize daily prices for one or more tickers."""
        if isinstance(tickers, str):
            raise TypeError("tickers must be a sequence of strings, not a string")

        normalized_tickers = [ticker.upper() for ticker in tickers]
        if not normalized_tickers:
            raise ValueError("At least one ticker is required")
        if len(normalized_tickers) != len(set(normalized_tickers)):
            raise ValueError("Duplicate tickers are not allowed")

        downloaded = yf.download(
            tickers=normalized_tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if downloaded.empty:
            raise ValueError("Yahoo Finance returned no market data")

        if len(normalized_tickers) > 1 and isinstance(
            downloaded.columns, pd.MultiIndex
        ):
            returned_tickers = {
                str(ticker).upper()
                for ticker in downloaded.columns.get_level_values(-1)
            }
            missing_tickers = [
                ticker
                for ticker in normalized_tickers
                if ticker not in returned_tickers
            ]
            if missing_tickers:
                missing_text = ", ".join(missing_tickers)
                raise ValueError(
                    f"Yahoo Finance returned no data for tickers: {missing_text}"
                )

        if len(normalized_tickers) == 1:
            return _normalize_ticker_frame(downloaded, normalized_tickers[0])

        ticker_frames = [
            _normalize_ticker_frame(downloaded, ticker)
            for ticker in normalized_tickers
        ]
        return (
            pd.concat(ticker_frames, ignore_index=True)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
