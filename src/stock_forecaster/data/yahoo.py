"""Yahoo Finance market data provider."""

import importlib
import sys
from collections.abc import Sequence
from datetime import date, datetime
from types import ModuleType

import pandas as pd

from stock_forecaster.data.http import configure_yahoo_http
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


def _curl_cffi_backend_status(yfinance_module: object) -> bool | None:
    http_module = getattr(yfinance_module, "_http", None)
    if http_module is None:
        http_module = sys.modules.get("yfinance._http")
    return getattr(http_module, "HAS_CURL_CFFI", None)


def _load_yfinance(use_system_trust: bool) -> ModuleType:
    if use_system_trust:
        loaded_yfinance = sys.modules.get("yfinance")
        if (
            loaded_yfinance is not None
            and _curl_cffi_backend_status(loaded_yfinance) is True
        ):
            raise RuntimeError(
                "System-trust mode must be configured before yfinance is imported "
                "with the curl_cffi backend."
            )
        configure_yahoo_http(use_system_trust=True)

    yfinance_module = importlib.import_module("yfinance")
    if (
        use_system_trust
        and _curl_cffi_backend_status(yfinance_module) is True
    ):
        raise RuntimeError(
            "System-trust mode requires yfinance's requests fallback, but "
            "curl_cffi is still active."
        )
    return yfinance_module


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

    def __init__(self, use_system_trust: bool = False) -> None:
        self._yfinance = _load_yfinance(use_system_trust)

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

        downloaded = self._yfinance.download(
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
