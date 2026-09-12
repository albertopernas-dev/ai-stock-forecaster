from unittest.mock import patch

import pandas as pd
import pytest

from stock_forecaster.data.yahoo import YahooFinanceProvider


def _single_ticker_yahoo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [101.0, 100.0],
            "High": [103.0, 102.0],
            "Low": [100.0, 99.0],
            "Close": [102.0, 101.0],
            "Adj Close": [101.5, 100.5],
            "Volume": [1_100, 1_000],
        },
        index=pd.to_datetime(["2026-09-11", "2026-09-10"]),
    )


def _multiple_ticker_yahoo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ("Open", "MSFT"): [201.0, 200.0],
            ("Open", "AAPL"): [101.0, 100.0],
            ("High", "MSFT"): [203.0, 202.0],
            ("High", "AAPL"): [103.0, 102.0],
            ("Low", "MSFT"): [200.0, 199.0],
            ("Low", "AAPL"): [100.0, 99.0],
            ("Close", "MSFT"): [202.0, 201.0],
            ("Close", "AAPL"): [102.0, 101.0],
            ("Adj Close", "MSFT"): [201.5, 200.5],
            ("Adj Close", "AAPL"): [101.5, 100.5],
            ("Volume", "MSFT"): [2_100, 2_000],
            ("Volume", "AAPL"): [1_100, 1_000],
        },
        index=pd.to_datetime(["2026-09-11", "2026-09-10"]),
    )


@patch("stock_forecaster.data.yahoo.yf.download")
def test_empty_ticker_list_is_rejected(download_mock) -> None:
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="At least one ticker is required"):
        provider.download_prices([], start="2026-09-01", end="2026-09-12")

    download_mock.assert_not_called()


@patch("stock_forecaster.data.yahoo.yf.download")
def test_single_ticker_response_is_normalized(download_mock) -> None:
    download_mock.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["aapl"],
        start="2026-09-01",
        end="2026-09-12",
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-10", "2026-09-11"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adjusted_close": [100.5, 101.5],
            "volume": [1_000, 1_100],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@patch("stock_forecaster.data.yahoo.yf.download")
def test_single_ticker_is_uppercased_before_download(download_mock) -> None:
    download_mock.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["aapl"],
        start="2026-09-01",
        end="2026-09-12",
    )

    assert result["ticker"].tolist() == ["AAPL", "AAPL"]
    assert download_mock.call_args.kwargs["tickers"] == ["AAPL"]


@patch("stock_forecaster.data.yahoo.yf.download")
def test_normalized_columns_are_exact(download_mock) -> None:
    download_mock.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["AAPL"],
        start="2026-09-01",
        end="2026-09-12",
    )

    assert result.columns.tolist() == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]


@patch("stock_forecaster.data.yahoo.yf.download")
def test_yfinance_download_uses_explicit_daily_price_options(download_mock) -> None:
    download_mock.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    provider.download_prices(
        ["AAPL"],
        start="2026-09-01",
        end="2026-09-12",
    )

    download_mock.assert_called_once_with(
        tickers=["AAPL"],
        start="2026-09-01",
        end="2026-09-12",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )


@patch("stock_forecaster.data.yahoo.yf.download")
def test_multiple_ticker_response_is_normalized_and_sorted(download_mock) -> None:
    download_mock.return_value = _multiple_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["msft", "aapl"],
        start="2026-09-01",
        end="2026-09-12",
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-09-10", "2026-09-10", "2026-09-11", "2026-09-11"]
            ),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "open": [100.0, 200.0, 101.0, 201.0],
            "high": [102.0, 202.0, 103.0, 203.0],
            "low": [99.0, 199.0, 100.0, 200.0],
            "close": [101.0, 201.0, 102.0, 202.0],
            "adjusted_close": [100.5, 200.5, 101.5, 201.5],
            "volume": [1_000, 2_000, 1_100, 2_100],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@patch("stock_forecaster.data.yahoo.yf.download")
def test_missing_prices_are_not_forward_filled(download_mock) -> None:
    yahoo_frame = _multiple_ticker_yahoo_frame()
    yahoo_frame.loc[pd.Timestamp("2026-09-11"), ("Adj Close", "AAPL")] = float(
        "nan"
    )
    download_mock.return_value = yahoo_frame
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["AAPL", "MSFT"],
        start="2026-09-01",
        end="2026-09-12",
    )

    aapl_missing_value = result.loc[
        (result["date"] == pd.Timestamp("2026-09-11"))
        & (result["ticker"] == "AAPL"),
        "adjusted_close",
    ].iloc[0]
    assert pd.isna(aapl_missing_value)


@patch("stock_forecaster.data.yahoo.yf.download")
def test_empty_yahoo_response_raises_clear_error(download_mock) -> None:
    download_mock.return_value = pd.DataFrame()
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="Yahoo Finance returned no market data"):
        provider.download_prices(
            ["AAPL"],
            start="2026-09-01",
            end="2026-09-12",
        )


@patch("stock_forecaster.data.yahoo.yf.download")
def test_duplicate_tickers_after_uppercase_normalization_are_rejected(
    download_mock,
) -> None:
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="Duplicate tickers are not allowed"):
        provider.download_prices(
            ["aapl", "AAPL"],
            start="2026-09-01",
            end="2026-09-12",
        )

    download_mock.assert_not_called()
