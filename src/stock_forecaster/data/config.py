"""Market data configuration loading."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MarketConfig:
    """Configuration for the market data universe."""

    start_date: date
    tickers: tuple[str, ...]


def load_market_config(path: str | Path) -> MarketConfig:
    """Load and validate market configuration from YAML."""
    config_path = Path(path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raw_start_date = raw_config.get("start_date")
    try:
        start_date = date.fromisoformat(str(raw_start_date))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid start_date: {raw_start_date}") from error

    raw_tickers = raw_config.get("tickers")
    if not isinstance(raw_tickers, list) or not all(
        isinstance(ticker, str) for ticker in raw_tickers
    ):
        raise ValueError("tickers must be a list of strings")
    if not raw_tickers:
        raise ValueError("At least one ticker is required")

    tickers = tuple(ticker.upper() for ticker in raw_tickers)
    if len(tickers) != len(set(tickers)):
        raise ValueError("Duplicate tickers are not allowed")

    return MarketConfig(start_date=start_date, tickers=tickers)
