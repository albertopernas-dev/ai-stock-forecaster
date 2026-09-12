# AI Stock Forecaster

AI Stock Forecaster is an end-to-end machine learning project intended to
forecast five-day stock returns, use those forecasts for portfolio
optimization, backtest the resulting strategy, and expose predictions through
an API.

## High-level architecture

The project uses a `src` layout with separate packages for future data,
feature, model, portfolio, backtesting, and shared utility code. Configuration,
notebooks, datasets, trained artifacts, tests, and CI workflows are kept outside
the importable package.

The implemented market data flow is:

```text
configs/market.yaml
        ↓
MarketConfig
        ↓
YahooFinanceProvider
        ↓
normalized DataFrame
        ↓
validation
        ↓
MarketDataPipeline
        ↓
ParquetPriceStorage
        ↓
data/raw/prices.parquet
```

The first run downloads the configured history. Later runs start from the
oldest next-required date across requested tickers, merge corrected or new
rows, validate the result, and update Parquet storage. Numeric missing values
are preserved rather than filled.

## Current status

The project scaffold, market configuration, Yahoo Finance provider, validation,
Parquet persistence, and incremental update pipeline are in place. Feature
engineering and machine learning do not exist yet. Portfolio optimization,
backtesting, APIs, deployment, and CI/CD also remain out of scope.

## Development setup

Python 3.12 or newer is required.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows
PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```
