# AI Stock Forecaster

[![CI](https://github.com/albertopernas-dev/ai-stock-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/albertopernas-dev/ai-stock-forecaster/actions/workflows/ci.yml)

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
The original `future_return_5d` target uses the price five trading observations
ahead; it remains supported and is the default target. The optional
`future_excess_return_5d` target is the stock's `future_return_5d` minus SPY's
adjusted-close return over the exact same window: the stock row date to that
stock's own fifth future trading date. SPY values are looked up at those two
exact dates, not by SPY's fifth observation. Missing exact SPY start or end
points are never imputed, so the excess target is null for that row.

Target selection is explicit when preparing supervised data. Omitting the
selection preserves the absolute-return default, which keeps the established
Step 5E target and default model behavior reproducible. SPY remains available
to construct the excess target, but is excluded from `X` and from the modeled
stock universe.

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
correlation metrics. For the excess-return target, ZeroBaseline predicts zero
excess return and is the natural MAE/RMSE reference. Its constant predictions
make Pearson correlation expected to be `NaN`. The existing directional metric
treats zero as its own sign, so ZeroBaseline earns directional credit only on
rows whose target is exactly zero; when exact-zero targets are rare its
directional accuracy approaches zero. That number must therefore not be read
like the directional accuracy of a conventional variable predictor, and other
models beating it on direction is not by itself evidence of directional
forecasting skill. TEST is held out and is not used for baseline comparison or
model selection.

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
using the ticker-specific `target_end_date`. For the official Step 5F
evaluation, the model set is exactly ZeroBaseline, MeanBaseline,
LinearRegression, and RandomForest. Momentum remains supported but is excluded
from that official Step 5F set. The models produce in-memory out-of-sample
predictions; pooled metrics are recomputed from their concatenated prediction
rows rather than averaged across years. TEST remains 2024+ and untouched.

Point-error metrics answer how close a predicted number is to the realized
return. They do not answer whether a model orders stocks correctly on a given
date, which is the property portfolio construction actually consumes. The
ranking evaluation adds that lens over the same out-of-sample predictions.

For each model and trading date it computes the Spearman rank correlation
between predictions and realized returns across that date's tickers, and the
spread between the mean realized return of the five highest-ranked and five
lowest-ranked stocks. Both are undefined, and reported as null rather than
zero, when a date has too few tickers or when predictions do not vary across
stocks; ZeroBaseline and MeanBaseline are constant by construction and are
therefore structurally undefined here. MomentumBaseline varies across stocks
and serves as the naive ordering reference.

The two targets are rank-identical within a date. All 15 stocks share a trading
calendar, so on any date every stock's benchmark window is the same and the
excess target equals the absolute target minus one constant. Subtracting a
constant preserves order and shifts both bucket means equally, so the
information coefficient and the tercile spread are invariant to the choice
between the two targets. This was verified across all 4,193 dates: the
within-date range of the difference between the targets is exactly zero. The
excess target can therefore change a ranking result only through its effect on
model fitting, never through the evaluation itself.

Because the horizon is five sessions while observations are daily, consecutive
dates share most of their target window and overlapping statistics overstate
consistency. Every summary is therefore reported twice: over all dates, and
over a non-overlapping subsample taking every fifth trading date within each
validation year. The spread is a raw difference of realized returns: it applies
no transaction cost, position sizing, or compounding and is never a strategy
return. TEST remains 2024+ and the ranking module rejects any row dated on or
after 2024-01-01. Controlled evaluation remains pending; numerical results are
deferred to the real run.

Ordering ability is not the same as a tradable edge. The backtest layer answers
the remaining question: what per-unit trading cost would reduce that edge to
exactly zero. Rather than assuming a cost level, which would let the assumption
decide the conclusion, it reports the break-even cost in basis points as
`10000 * mean gross return / mean traded notional`. A non-positive value means
the edge is absent before any cost is applied.

Two fixed equal-weight strategies are simulated. The long/short book is
dollar-neutral: the five highest-ranked stocks at `+1/5` each and the five
lowest-ranked at `-1/5`, so its gross return is the tercile spread. The
long-only book holds the five highest-ranked and is measured against the
equal-weight universe. Rebalancing is weekly and non-overlapping, one date in
every five, which matches the five-session horizon so every position is held
for exactly its own target window. Daily rebalancing is deliberately excluded
because it trades strictly more and is therefore strictly worse on cost.

Traded notional is the sum of absolute weight changes between consecutive
rebalances, sequenced continuously across years, so the first rebalance carries
the real cost of establishing the book. Periods where predictions do not vary
across stocks are undefined rather than zero, and carry the previous book
forward. Short borrow cost and market impact are not modeled, so the reported
break-even is optimistic. TEST remains 2024+ and the module rejects any row
dated on or after 2024-01-01. Controlled evaluation remains pending; numerical
results are deferred to the real run.

The 2024+ hold-out was never read during development. Both evaluation modules
reject any row dated on or after 2024-01-01 and keep doing so by default; an
explicit `allow_holdout=True` at the call site is the only way to admit it, so
spending the hold-out is a visible, deliberate act rather than a silent default.

The confirmation was pre-registered before any hold-out value existed, in
`docs/superpowers/specs/2026-09-23-holdout-confirmation-design.md`. It fixes one
target, one strategy, two candidate models against a naive control, one primary
metric, and the comparison against the already published development figures. It
also fixes in advance that an ambiguous outcome is reported as inconclusive
rather than rescued by a favorable secondary statistic, and that the evaluation
runs exactly once with no refitting, reselection or repetition afterwards.

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
infrastructure are also in place. Controlled target migration and evaluation
are pending; numerical Step 5E and Step 5F results are deferred to the real
evaluation.

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
