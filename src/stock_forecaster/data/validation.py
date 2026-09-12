"""Validation for normalized market price data."""

from collections.abc import Sequence

import pandas as pd

REQUIRED_PRICE_COLUMNS = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
_PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")


def validate_prices(
    frame: pd.DataFrame,
    expected_tickers: Sequence[str] | None = None,
) -> None:
    """Raise ``ValueError`` when normalized market prices are invalid."""
    missing_columns = [
        column for column in REQUIRED_PRICE_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if frame["date"].isna().any():
        raise ValueError("date contains missing values")
    if frame["ticker"].isna().any():
        raise ValueError("ticker contains missing values")
    if not frame["ticker"].map(
        lambda ticker: isinstance(ticker, str) and bool(ticker.strip())
    ).all():
        raise ValueError("ticker values must be non-empty strings")
    if frame.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate date and ticker rows")

    for column in _PRICE_COLUMNS:
        if (frame[column].notna() & frame[column].le(0)).any():
            raise ValueError(f"{column} must be greater than zero")
    if (frame["volume"].notna() & frame["volume"].lt(0)).any():
        raise ValueError("volume must be non-negative")

    _validate_price_relationship(frame, "high", "open", "at least")
    _validate_price_relationship(frame, "high", "close", "at least")
    _validate_price_relationship(frame, "low", "open", "at most")
    _validate_price_relationship(frame, "low", "close", "at most")

    if expected_tickers is not None:
        if isinstance(expected_tickers, str):
            raise TypeError("expected_tickers must be a sequence of strings")
        normalized_expected = [ticker.upper() for ticker in expected_tickers]
        actual_tickers = set(frame["ticker"])
        missing_tickers = [
            ticker
            for ticker in dict.fromkeys(normalized_expected)
            if ticker not in actual_tickers
        ]
        if missing_tickers:
            raise ValueError(f"Missing expected tickers: {', '.join(missing_tickers)}")


def _validate_price_relationship(
    frame: pd.DataFrame,
    left: str,
    right: str,
    relation: str,
) -> None:
    both_present = frame[left].notna() & frame[right].notna()
    if relation == "at least":
        invalid = both_present & frame[left].lt(frame[right])
    else:
        invalid = both_present & frame[left].gt(frame[right])
    if invalid.any():
        raise ValueError(f"{left} must be {relation} {right}")
