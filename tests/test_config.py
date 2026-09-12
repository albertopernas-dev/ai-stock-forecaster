from datetime import date
from pathlib import Path

import pytest

from stock_forecaster.data.config import MarketConfig, load_market_config


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "market.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_market_config_loads_expected_structure(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers:\n  - AAPL\n  - MSFT\n',
    )

    config = load_market_config(path)

    assert config == MarketConfig(
        start_date=date(2010, 1, 1),
        tickers=("AAPL", "MSFT"),
    )


def test_start_date_becomes_date(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers:\n  - AAPL\n',
    )

    config = load_market_config(path)

    assert config.start_date == date(2010, 1, 1)
    assert isinstance(config.start_date, date)


def test_ticker_names_are_normalized_to_uppercase(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers:\n  - aapl\n  - MsFt\n',
    )

    config = load_market_config(path)

    assert config.tickers == ("AAPL", "MSFT")


def test_duplicate_ticker_fails(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers:\n  - aapl\n  - AAPL\n',
    )

    with pytest.raises(ValueError, match="Duplicate tickers are not allowed"):
        load_market_config(path)


def test_empty_ticker_list_fails(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers: []\n',
    )

    with pytest.raises(ValueError, match="At least one ticker is required"):
        load_market_config(path)


def test_plain_string_ticker_value_fails(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "2010-01-01"\ntickers: AAPL\n',
    )

    with pytest.raises(ValueError, match="tickers must be a list of strings"):
        load_market_config(path)


def test_invalid_start_date_fails(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        'start_date: "not-a-date"\ntickers:\n  - AAPL\n',
    )

    with pytest.raises(ValueError, match="Invalid start_date: not-a-date"):
        load_market_config(path)
