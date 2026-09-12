import pandas as pd
import pytest

from stock_forecaster.data.validation import validate_prices


def _valid_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-10", "2026-09-11"]),
            "ticker": ["AAPL", "MSFT"],
            "open": [100.0, 200.0],
            "high": [102.0, 203.0],
            "low": [99.0, 198.0],
            "close": [101.0, 201.0],
            "adjusted_close": [100.5, 200.5],
            "volume": [1_000, 2_000],
        }
    )


def test_valid_frame_passes_without_modification() -> None:
    frame = _valid_prices()
    original = frame.copy(deep=True)

    assert validate_prices(frame) is None

    pd.testing.assert_frame_equal(frame, original)


def test_missing_required_column_fails() -> None:
    frame = _valid_prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        validate_prices(frame)


def test_missing_date_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "date"] = pd.NaT

    with pytest.raises(ValueError, match="date contains missing values"):
        validate_prices(frame)


def test_missing_ticker_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "ticker"] = None

    with pytest.raises(ValueError, match="ticker contains missing values"):
        validate_prices(frame)


@pytest.mark.parametrize("ticker", ["", "   ", 123])
def test_ticker_must_be_a_non_empty_string(ticker) -> None:
    frame = _valid_prices()
    frame["ticker"] = frame["ticker"].astype(object)
    frame.loc[0, "ticker"] = ticker

    with pytest.raises(ValueError, match="ticker values must be non-empty strings"):
        validate_prices(frame)


def test_duplicate_date_and_ticker_fails() -> None:
    frame = pd.concat([_valid_prices(), _valid_prices().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
        validate_prices(frame)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("open", 0.0),
        ("high", -1.0),
        ("low", 0.0),
        ("close", -1.0),
        ("adjusted_close", 0.0),
    ],
)
def test_zero_or_negative_price_fails(column: str, invalid_value: float) -> None:
    frame = _valid_prices()
    frame.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=rf"{column} must be greater than zero"):
        validate_prices(frame)


def test_negative_volume_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="volume must be non-negative"):
        validate_prices(frame)


def test_high_lower_than_open_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "high"] = 99.5

    with pytest.raises(ValueError, match="high must be at least open"):
        validate_prices(frame)


def test_high_lower_than_close_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "high"] = 100.5

    with pytest.raises(ValueError, match="high must be at least close"):
        validate_prices(frame)


def test_low_higher_than_open_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "low"] = 100.5

    with pytest.raises(ValueError, match="low must be at most open"):
        validate_prices(frame)


def test_low_higher_than_close_fails() -> None:
    frame = _valid_prices()
    frame.loc[0, "open"] = 102.0
    frame.loc[0, "low"] = 101.5

    with pytest.raises(ValueError, match="low must be at most close"):
        validate_prices(frame)


def test_numeric_nan_is_allowed() -> None:
    frame = _valid_prices()
    frame.loc[0, ["open", "high", "low", "close", "adjusted_close", "volume"]] = (
        float("nan")
    )

    assert validate_prices(frame) is None


def test_missing_expected_tickers_are_reported_clearly() -> None:
    frame = _valid_prices()

    with pytest.raises(ValueError, match="Missing expected tickers: META, NVDA"):
        validate_prices(frame, expected_tickers=["aapl", "meta", "nvda"])
