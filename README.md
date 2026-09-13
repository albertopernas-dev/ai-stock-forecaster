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

The next implemented transformation is:

```text
RAW prices
    ↓
Core feature engineering
    ↓
Features + future_return_5d
```

Predictor features use only the current and previous trading observations.
The `future_return_5d` target uses the price five trading observations ahead.
Processed-data persistence and model training are not implemented yet.

The first run downloads the configured history. Later runs start from the
oldest next-required date across requested tickers, merge corrected or new
rows, validate the result, and update Parquet storage. Numeric missing values
are preserved rather than filled.

## Current status

The project scaffold, market configuration, Yahoo Finance provider, validation,
Parquet persistence, incremental update pipeline, and pure core feature
engineering are in place. Processed feature persistence and machine learning do
not exist yet. Portfolio optimization, backtesting, APIs, deployment, and CI/CD
also remain out of scope.

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

## Yahoo Finance system trust

The provider uses yfinance's normal HTTP backend by default:

```python
YahooFinanceProvider(use_system_trust=False)
```

On systems where certifi does not contain a locally trusted TLS root, enable
the operating-system certificate store instead:

```python
YahooFinanceProvider(use_system_trust=True)
```

This configures the following verified TLS path before yfinance is imported:

```text
yfinance
    ↓
requests fallback
    ↓
truststore
    ↓
operating-system certificate store
```

TLS verification remains enabled. The opt-in default is retained because the
normal yfinance backend works correctly in many deployment environments.
