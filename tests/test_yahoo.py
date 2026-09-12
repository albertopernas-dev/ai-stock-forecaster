import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from stock_forecaster.data import yahoo
from stock_forecaster.data.yahoo import YahooFinanceProvider


@pytest.fixture
def yfinance_module(monkeypatch):
    module = SimpleNamespace(download=Mock())
    monkeypatch.setattr(yahoo.importlib, "import_module", lambda _: module)
    return module


def test_default_provider_uses_lazily_imported_yfinance(monkeypatch) -> None:
    download_mock = Mock(return_value=_single_ticker_yahoo_frame())
    imported_yfinance = SimpleNamespace(download=download_mock)
    monkeypatch.setattr(
        "stock_forecaster.data.yahoo.importlib.import_module",
        lambda module_name: imported_yfinance,
    )

    provider = YahooFinanceProvider()
    result = provider.download_prices(
        ["AAPL"], start="2026-09-01", end="2026-09-12"
    )

    assert not result.empty
    download_mock.assert_called_once()


def test_system_trust_is_configured_before_yfinance_import(monkeypatch) -> None:
    events = []
    imported_yfinance = SimpleNamespace(
        _http=SimpleNamespace(HAS_CURL_CFFI=False),
        download=Mock(),
    )
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)
    monkeypatch.setattr(
        "stock_forecaster.data.yahoo.configure_yahoo_http",
        lambda use_system_trust: events.append(("configure", use_system_trust)),
    )
    monkeypatch.setattr(
        "stock_forecaster.data.yahoo.importlib.import_module",
        lambda module_name: events.append(("import", module_name))
        or imported_yfinance,
    )

    YahooFinanceProvider(use_system_trust=True)

    assert events == [("configure", True), ("import", "yfinance")]


def test_system_trust_rejects_curl_cffi_after_yfinance_import(monkeypatch) -> None:
    imported_yfinance = SimpleNamespace(
        _http=SimpleNamespace(HAS_CURL_CFFI=True),
        download=Mock(),
    )
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)
    monkeypatch.setattr(
        "stock_forecaster.data.yahoo.configure_yahoo_http",
        lambda use_system_trust: None,
    )
    monkeypatch.setattr(
        "stock_forecaster.data.yahoo.importlib.import_module",
        lambda module_name: imported_yfinance,
    )

    with pytest.raises(RuntimeError, match="requests fallback"):
        YahooFinanceProvider(use_system_trust=True)


def test_system_trust_rejects_already_loaded_curl_cffi_yfinance(
    monkeypatch,
) -> None:
    loaded_yfinance = SimpleNamespace(
        _http=SimpleNamespace(HAS_CURL_CFFI=True),
        download=Mock(),
    )
    monkeypatch.setitem(sys.modules, "yfinance", loaded_yfinance)

    with pytest.raises(RuntimeError, match="before yfinance is imported"):
        YahooFinanceProvider(use_system_trust=True)


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


def test_empty_ticker_list_is_rejected(yfinance_module) -> None:
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="At least one ticker is required"):
        provider.download_prices([], start="2026-09-01", end="2026-09-12")

    yfinance_module.download.assert_not_called()


def test_plain_string_ticker_input_is_rejected(yfinance_module) -> None:
    provider = YahooFinanceProvider()

    with pytest.raises(TypeError, match="tickers must be a sequence of strings"):
        provider.download_prices(
            "AAPL",
            start="2026-09-01",
            end="2026-09-12",
        )

    yfinance_module.download.assert_not_called()


def test_single_ticker_response_is_normalized(yfinance_module) -> None:
    yfinance_module.download.return_value = _single_ticker_yahoo_frame()
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


def test_single_ticker_is_uppercased_before_download(yfinance_module) -> None:
    yfinance_module.download.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    result = provider.download_prices(
        ["aapl"],
        start="2026-09-01",
        end="2026-09-12",
    )

    assert result["ticker"].tolist() == ["AAPL", "AAPL"]
    assert yfinance_module.download.call_args.kwargs["tickers"] == ["AAPL"]


def test_normalized_columns_are_exact(yfinance_module) -> None:
    yfinance_module.download.return_value = _single_ticker_yahoo_frame()
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


def test_yfinance_download_uses_explicit_daily_price_options(
    yfinance_module,
) -> None:
    yfinance_module.download.return_value = _single_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    provider.download_prices(
        ["AAPL"],
        start="2026-09-01",
        end="2026-09-12",
    )

    yfinance_module.download.assert_called_once_with(
        tickers=["AAPL"],
        start="2026-09-01",
        end="2026-09-12",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )


def test_multiple_ticker_response_is_normalized_and_sorted(
    yfinance_module,
) -> None:
    yfinance_module.download.return_value = _multiple_ticker_yahoo_frame()
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


def test_missing_prices_are_not_forward_filled(yfinance_module) -> None:
    yahoo_frame = _multiple_ticker_yahoo_frame()
    yahoo_frame.loc[pd.Timestamp("2026-09-11"), ("Adj Close", "AAPL")] = float(
        "nan"
    )
    yfinance_module.download.return_value = yahoo_frame
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


def test_empty_yahoo_response_raises_clear_error(yfinance_module) -> None:
    yfinance_module.download.return_value = pd.DataFrame()
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="Yahoo Finance returned no market data"):
        provider.download_prices(
            ["AAPL"],
            start="2026-09-01",
            end="2026-09-12",
        )


def test_missing_requested_ticker_is_reported_clearly(yfinance_module) -> None:
    yfinance_module.download.return_value = _multiple_ticker_yahoo_frame()
    provider = YahooFinanceProvider()

    with pytest.raises(
        ValueError,
        match="Yahoo Finance returned no data for tickers: META",
    ):
        provider.download_prices(
            ["AAPL", "MSFT", "META"],
            start="2026-09-01",
            end="2026-09-12",
        )


def test_duplicate_tickers_after_uppercase_normalization_are_rejected(
    yfinance_module,
) -> None:
    provider = YahooFinanceProvider()

    with pytest.raises(ValueError, match="Duplicate tickers are not allowed"):
        provider.download_prices(
            ["aapl", "AAPL"],
            start="2026-09-01",
            end="2026-09-12",
        )

    yfinance_module.download.assert_not_called()
