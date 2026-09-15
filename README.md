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
    ↓
Complete supervised rows
    ↓
Leakage-safe temporal split
    ↓
TRAIN / VALIDATION / TEST
```

Predictor features use only the current and previous trading observations.
The `future_return_5d` target uses the price five trading observations ahead.
Supervised rows keep predictors, the target, and identifying metadata separate.
The initial chronological boundaries are:

- TRAIN: through 2021-12-31
- VALIDATION: 2022-01-01 through 2023-12-31
- TEST: from 2024-01-01

Each supervised row records the trading date five ticker observations ahead as
internal split metadata. A train or validation row is removed when that target
horizon reaches into the next period, preventing future information from
crossing a split boundary. The horizon date is never included as a model
feature.

The model-evaluation progression is:

```text
Single temporal validation split
        ↓
Annual expanding walk-forward validation (2016-2023)
        ↓
Pooled out-of-sample model comparison
        ↓
Future modeling decisions
```

The global Mean Baseline always predicts the TRAIN target mean. The Momentum
Baseline predicts the future five-day return using the current `return_5d`
feature. Both use the same MAE, RMSE, directional accuracy, and Pearson
correlation metrics. TEST is held out and is not used for baseline comparison
or model selection.

The first trainable model is one global linear regression over 15 stocks.
SPY remains available as a benchmark but is excluded from model fitting and
validation comparison. The model consumes exactly the ten engineered
`FEATURE_COLUMNS` and does not encode ticker metadata. A single sklearn
pipeline fits `StandardScaler` and `LinearRegression` on TRAIN only. Model
comparison uses VALIDATION only; TEST remains untouched.

The first nonlinear benchmark is one fixed Random Forest across the same
15-stock universe. SPY remains the persisted benchmark and is excluded only
in modeling orchestration. The forest consumes the same ten engineered
features directly, without feature scaling, fits on TRAIN only, and is
compared with freshly fitted baselines and linear regression on VALIDATION.
TEST remains untouched.

Model stability is evaluated with annual 2016-2023 validation folds and an
expanding TRAIN window. TRAIN and each annual VALIDATION partition are purged
using the ticker-specific `target_end_date`. The four existing models produce
in-memory out-of-sample predictions; pooled metrics are recomputed from their
concatenated prediction rows rather than averaged across years. TEST remains
2024+ and untouched.

The first run downloads the configured history. Later runs start from the
oldest next-required date across requested tickers, merge corrected or new
rows, validate the result, and update Parquet storage. Numeric missing values
are preserved rather than filled.

## Current status

The project scaffold, market configuration, Yahoo Finance provider, validation,
Parquet persistence, incremental update pipeline, and pure core feature
engineering are in place. Leakage-safe supervised dataset preparation and
chronological train/validation/test splitting are also implemented, along with
two simple forecasting baselines, common regression metrics, and a standardized
linear regression forecaster, and a fixed Random Forest forecaster. Portfolio
optimization, backtesting, APIs, deployment, and CI/CD remain out of scope.
Expanding annual walk-forward validation and pooled out-of-sample evaluation
infrastructure are also in place; numerical Step 5E results are deferred to
the real evaluation.

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
