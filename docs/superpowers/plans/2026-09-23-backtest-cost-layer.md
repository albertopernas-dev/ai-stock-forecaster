# Break-Even Cost Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the per-unit trading cost at which the Step 5G ordering edge reaches exactly zero, for two fixed equal-weight strategies, without assuming any cost level.

**Architecture:** One new additive module `backtesting/simulation.py` turns the existing out-of-sample prediction frames into one row per model and weekly rebalance date carrying a gross return and a traded notional, then aggregates them into a break-even cost. Bucket size, rebalance selection and model presentation order are imported from `ranking.py` so the two steps cannot drift. Nothing is persisted and no existing module changes.

**Tech Stack:** Python 3.12.14, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-23-backtest-cost-layer-design.md`

## Global Constraints

- **The specification is binding** and wins over any ambiguous wording here.
- **TEST is 2024-01-01 onward and is never read.** `build_period_returns` raises on any row dated on or after that boundary. Development filtering applies both `date < 2024-01-01` and `target_end_date < 2024-01-01`.
- **Costs are an output, never an input.** No cost level is assumed anywhere in code, tests, or reporting.
- **No tuning and no strategy selection.** Both strategies are fixed and equal-weight. A disappointing break-even figure is not a defect and must never trigger a re-run with different buckets, schedule, or weighting.
- **Protected and untouched:** `linear.py`, `forest.py`, `metrics.py`, `evaluation.py`, `dataset.py`, `baseline.py`, `engineering.py`, `ranking.py`, `pyproject.toml`, `configs/`.
- **No new dependency. Nothing is persisted.** Both data files are hash-verified around the official run.
- **Anchors:** raw `49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5`, processed `C717BFEE645C312CFD0086B6A1514D15CC47E0E4EED4D6557218B3EDC272DA57`.
- **One implementation commit.** Tasks 1-4 leave the work uncommitted; Task 5 creates the single tracked commit. Any later fix amends it.
- **Environment:** shared interpreter `C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe`, with `$env:PYTHONPATH` set to the worktree's `src`. Pytest no longer needs a `--basetemp` override.
- **No claim** of profitability, investability, expected return, statistical significance, causality, economic significance, TEST performance, Sharpe or other risk-adjusted statistics, compounding, or annualization.

## File Map

- Create: `src/stock_forecaster/backtesting/simulation.py` — the whole surface: strategy constants, validation, `build_period_returns`, `summarize_backtest`.
- Create: `tests/test_backtest.py` — all synthetic deterministic coverage.
- Modify: `README.md` — the break-even method, both strategies, the schedule choice, unmodeled costs, and the untouched TEST boundary.
- Read only during the official run: `data/processed/features.parquet`.

---

### Task 1: Validation and the long/short gross return

**Files:**
- Create: `src/stock_forecaster/backtesting/simulation.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `PREDICTION_COLUMNS` from `models.evaluation`; `BUCKET_SIZE`, `KNOWN_MODEL_ORDER`, `TEST_BOUNDARY`, `select_non_overlapping` from `models.ranking`.
- Produces: `LONG_SHORT`, `LONG_ONLY`, `STRATEGIES`, `REBALANCE_STEP`, `PERIOD_COLUMNS`, and `build_period_returns(predictions, strategy=LONG_SHORT, step=REBALANCE_STEP) -> pd.DataFrame` with columns `model, validation_year, date, n_tickers, gross_return, traded_notional`. Tasks 2-4 and 6 rely on exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
import pandas as pd
import pytest

from stock_forecaster.backtesting.simulation import (
    LONG_SHORT,
    REBALANCE_STEP,
    build_period_returns,
)
from stock_forecaster.models.ranking import BUCKET_SIZE


def _rows(date, model, tickers, y_true, prediction, year=2016):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date] * len(tickers)),
            "ticker": list(tickers),
            "validation_year": year,
            "model": model,
            "y_true": list(y_true),
            "prediction": list(prediction),
        }
    )


def _ten(date, y_true, prediction, model="LinearRegression", year=2016):
    tickers = [f"T{index:02d}" for index in range(1, 11)]
    return _rows(date, model, tickers, y_true, prediction, year=year)


DESCENDING = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def test_long_short_gross_return_is_top_minus_bottom_bucket_mean():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=DESCENDING,
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert list(periods.columns) == [
        "model",
        "validation_year",
        "date",
        "n_tickers",
        "gross_return",
        "traded_notional",
    ]
    assert len(periods) == 1
    assert periods.loc[0, "n_tickers"] == 10
    assert periods.loc[0, "gross_return"] == pytest.approx(0.08 - 0.03)


def test_ties_resolve_by_ascending_ticker():
    frame = _ten(
        "2016-01-04",
        y_true=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        prediction=[10, 9, 8, 7, 6, 6, 4, 3, 2, 1],
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods.loc[0, "gross_return"] == pytest.approx(0.2)


def test_constant_predictions_give_an_undefined_period():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="ZeroBaseline",
    )

    periods = build_period_returns(frame)

    assert pd.isna(periods.loc[0, "gross_return"])
    assert pd.isna(periods.loc[0, "traded_notional"])


def test_fewer_tickers_than_two_buckets_give_an_undefined_period():
    count = 2 * BUCKET_SIZE - 1
    tickers = [f"T{index:02d}" for index in range(1, count + 1)]
    frame = _rows(
        "2016-01-04",
        "LinearRegression",
        tickers,
        [0.01 * index for index in range(count, 0, -1)],
        [float(index) for index in range(count, 0, -1)],
    )

    periods = build_period_returns(frame)

    assert periods.loc[0, "n_tickers"] == count
    assert pd.isna(periods.loc[0, "gross_return"])


def test_only_every_fifth_date_of_each_year_is_a_rebalance():
    first = pd.bdate_range("2016-01-04", periods=12)
    second = pd.bdate_range("2017-01-02", periods=12)
    frames = [
        _ten(date, [0.01] * 10, DESCENDING, year=year)
        for dates, year in ((first, 2016), (second, 2017))
        for date in dates
    ]

    periods = build_period_returns(pd.concat(frames, ignore_index=True))

    assert periods.loc[periods.validation_year.eq(2016), "date"].tolist() == [
        first[0],
        first[5],
        first[10],
    ]
    assert periods.loc[periods.validation_year.eq(2017), "date"].tolist() == [
        second[0],
        second[5],
        second[10],
    ]
    assert REBALANCE_STEP == 5


def test_models_are_emitted_in_known_presentation_order():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    shuffled = pd.concat(
        [
            _ten("2016-01-04", values, DESCENDING, model="RandomForest"),
            _ten("2016-01-04", values, DESCENDING, model="MomentumBaseline"),
            _ten("2016-01-04", values, DESCENDING, model="LinearRegression"),
        ],
        ignore_index=True,
    )

    periods = build_period_returns(shuffled)

    assert periods["model"].tolist() == [
        "MomentumBaseline",
        "LinearRegression",
        "RandomForest",
    ]


def test_rows_dated_on_or_after_the_test_boundary_are_rejected():
    frame = _ten("2024-01-02", [0.01] * 10, DESCENDING, year=2023)

    with pytest.raises(ValueError, match="2024"):
        build_period_returns(frame)


def test_unknown_strategy_is_rejected():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    with pytest.raises(ValueError, match="Unsupported strategy"):
        build_period_returns(frame, strategy="market_neutral_optimized")


@pytest.mark.parametrize("step", [0, -1])
def test_non_positive_step_is_rejected(step):
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    with pytest.raises(ValueError, match="step must be a positive integer"):
        build_period_returns(frame, step=step)


def test_duplicate_date_ticker_model_rows_are_rejected():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_period_returns(duplicated)


@pytest.mark.parametrize("missing", ["prediction", "y_true", "model", "date"])
def test_missing_required_columns_are_rejected(missing):
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING).drop(columns=[missing])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_period_returns(frame)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_period_returns(pd.DataFrame())


def test_build_period_returns_does_not_mutate_its_input():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)
    original = frame.copy(deep=True)

    build_period_returns(frame)

    pd.testing.assert_frame_equal(frame, original)
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
$env:PYTHONPATH = 'C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.worktrees\backtest-cost-layer\src'
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'stock_forecaster.backtesting.simulation'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/stock_forecaster/backtesting/simulation.py`:

```python
"""Break-even cost backtest over out-of-sample return forecasts."""

import pandas as pd

from stock_forecaster.models.evaluation import PREDICTION_COLUMNS
from stock_forecaster.models.ranking import (
    BUCKET_SIZE,
    KNOWN_MODEL_ORDER,
    TEST_BOUNDARY,
    select_non_overlapping,
)

LONG_SHORT = "long_short"
LONG_ONLY = "long_only"
STRATEGIES = (LONG_SHORT, LONG_ONLY)
REBALANCE_STEP = 5

PERIOD_COLUMNS = (
    "model",
    "validation_year",
    "date",
    "n_tickers",
    "gross_return",
    "traded_notional",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _model_sort_key(model_name: str) -> int:
    if model_name in KNOWN_MODEL_ORDER:
        return KNOWN_MODEL_ORDER.index(model_name)
    return len(KNOWN_MODEL_ORDER)


def _validate(predictions: pd.DataFrame, strategy: str, step: int) -> pd.DataFrame:
    if strategy not in STRATEGIES:
        allowed = ", ".join(STRATEGIES)
        raise ValueError(
            f"Unsupported strategy {strategy!r}; expected one of: {allowed}"
        )
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty DataFrame")
    _require_columns(predictions, PREDICTION_COLUMNS)
    ordered = predictions.loc[:, PREDICTION_COLUMNS].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    if ordered.duplicated(subset=["date", "ticker", "model"]).any():
        raise ValueError("Duplicate date, ticker and model rows")
    if ordered["date"].ge(TEST_BOUNDARY).any():
        raise ValueError(
            "predictions must not contain rows dated 2024-01-01 or later; "
            "TEST is never evaluated"
        )
    return ordered


def _book(group: pd.DataFrame, strategy: str) -> tuple[dict[str, float] | None, float]:
    if len(group) < 2 * BUCKET_SIZE or group["prediction"].nunique(dropna=True) <= 1:
        return None, float("nan")
    ranked = group.sort_values(by=["prediction", "ticker"], ascending=[False, True])
    top = ranked.iloc[:BUCKET_SIZE]
    weights = {ticker: 1.0 / BUCKET_SIZE for ticker in top["ticker"]}
    if strategy == LONG_SHORT:
        bottom = ranked.iloc[-BUCKET_SIZE:]
        for ticker in bottom["ticker"]:
            weights[ticker] = -1.0 / BUCKET_SIZE
        gross = float(top["y_true"].mean()) - float(bottom["y_true"].mean())
    else:
        gross = float(top["y_true"].mean()) - float(ranked["y_true"].mean())
    return weights, gross


def build_period_returns(
    predictions: pd.DataFrame,
    strategy: str = LONG_SHORT,
    step: int = REBALANCE_STEP,
) -> pd.DataFrame:
    """Return one gross return and traded notional per model and rebalance."""
    ordered = _validate(predictions, strategy, step)
    rebalances = select_non_overlapping(ordered, step=step)

    rows: list[dict[str, object]] = []
    models = sorted(rebalances["model"].unique(), key=_model_sort_key)
    for model in models:
        model_rows = rebalances.loc[rebalances["model"].eq(model)]
        for date, group in model_rows.groupby("date", sort=True):
            weights, gross = _book(group, strategy)
            rows.append(
                {
                    "model": model,
                    "validation_year": int(group["validation_year"].iloc[0]),
                    "date": date,
                    "n_tickers": len(group),
                    "gross_return": gross,
                    "traded_notional": float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=list(PERIOD_COLUMNS))
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: every test from Step 1 passes. `traded_notional` is `NaN` everywhere, which Task 3 fills in.

- [ ] **Step 5: Do not commit**

Confirm with `git status --short` that only the two new files are untracked.

---

### Task 2: The long-only strategy

**Files:**
- Modify: `src/stock_forecaster/backtesting/simulation.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `_book` and `build_period_returns` from Task 1.
- Produces: `LONG_ONLY` accepted by `build_period_returns`, whose gross return is the top bucket mean minus the all-stock mean. No signature change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`, adding `LONG_ONLY` to the module import:

```python
def test_long_only_gross_return_is_top_bucket_minus_universe_mean():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    frame = _ten("2016-01-04", y_true=values, prediction=DESCENDING)

    periods = build_period_returns(frame, strategy=LONG_ONLY)

    universe_mean = sum(values) / len(values)
    assert periods.loc[0, "gross_return"] == pytest.approx(0.08 - universe_mean)


def test_long_only_differs_from_long_short_on_the_same_frame():
    values = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    frame = _ten("2016-01-04", y_true=values, prediction=DESCENDING)

    long_only = build_period_returns(frame, strategy=LONG_ONLY)
    long_short = build_period_returns(frame, strategy=LONG_SHORT)

    assert long_only.loc[0, "gross_return"] < long_short.loc[0, "gross_return"]


def test_long_only_is_undefined_for_constant_predictions():
    frame = _ten(
        "2016-01-04",
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="MeanBaseline",
    )

    assert pd.isna(build_period_returns(frame, strategy=LONG_ONLY).loc[0, "gross_return"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: collection fails with `ImportError: cannot import name 'LONG_ONLY'`.

- [ ] **Step 3: Write the minimal implementation**

No production change is needed: `_book` already branches on `strategy` and Task 1 defined `LONG_ONLY`. If the import fails, the constant is missing from Task 1's module and must be added exactly as specified there.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: every test in the file passes.

- [ ] **Step 5: Do not commit**

---

### Task 3: Traded notional and turnover sequencing

**Files:**
- Modify: `src/stock_forecaster/backtesting/simulation.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `_book` and the period loop from Tasks 1-2.
- Produces: `traded_notional` populated as `sum(abs(new_weight - previous_weight))`, sequenced per model in date order across years, with undefined dates carrying the previous book forward.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
ASCENDING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _sequence(dates, predictions_per_date, model="LinearRegression", year=2016):
    return pd.concat(
        [
            _ten(date, [0.01] * 10, prediction, model=model, year=year)
            for date, prediction in zip(dates, predictions_per_date, strict=True)
        ],
        ignore_index=True,
    )


def test_first_rebalance_trades_the_full_gross_exposure():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    long_short = build_period_returns(frame, strategy=LONG_SHORT)
    long_only = build_period_returns(frame, strategy=LONG_ONLY)

    assert long_short.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert long_only.loc[0, "traded_notional"] == pytest.approx(1.0)


def test_an_unchanged_book_trades_nothing():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = _sequence(dates, [DESCENDING] * 6)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert len(periods) == 2
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert periods.loc[1, "traded_notional"] == pytest.approx(0.0)


def test_a_fully_reversed_book_trades_twice_the_gross_exposure():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = _sequence(dates, [DESCENDING] + [ASCENDING] * 5)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods.loc[1, "traded_notional"] == pytest.approx(4.0)


def test_an_undefined_rebalance_carries_the_previous_book_forward():
    dates = pd.bdate_range("2016-01-04", periods=11)
    predictions = [DESCENDING] + [[0.0] * 10] * 5 + [DESCENDING] * 5
    frame = _sequence(dates, predictions)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert len(periods) == 3
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert pd.isna(periods.loc[1, "traded_notional"])
    assert periods.loc[2, "traded_notional"] == pytest.approx(0.0)


def test_turnover_sequencing_continues_across_a_year_boundary():
    first = _sequence(pd.bdate_range("2016-12-01", periods=1), [DESCENDING], year=2016)
    second = _sequence(pd.bdate_range("2017-01-03", periods=1), [DESCENDING], year=2017)
    frame = pd.concat([first, second], ignore_index=True)

    periods = build_period_returns(frame, strategy=LONG_SHORT)

    assert periods["validation_year"].tolist() == [2016, 2017]
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)
    assert periods.loc[1, "traded_notional"] == pytest.approx(0.0)


def test_each_model_keeps_its_own_book():
    dates = pd.bdate_range("2016-01-04", periods=6)
    frame = pd.concat(
        [
            _sequence(dates, [DESCENDING] * 6, model="LinearRegression"),
            _sequence(dates, [DESCENDING] + [ASCENDING] * 5, model="RandomForest"),
        ],
        ignore_index=True,
    )

    periods = build_period_returns(frame, strategy=LONG_SHORT)
    linear = periods.loc[periods.model.eq("LinearRegression")].reset_index(drop=True)
    forest = periods.loc[periods.model.eq("RandomForest")].reset_index(drop=True)

    assert linear.loc[1, "traded_notional"] == pytest.approx(0.0)
    assert forest.loc[1, "traded_notional"] == pytest.approx(4.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: the new turnover tests fail because `traded_notional` is `NaN`. The undefined-rebalance test may already pass on its `NaN` assertion; it stays as a regression guard.

- [ ] **Step 3: Write the minimal implementation**

Add the helper above `build_period_returns`:

```python
def _traded_notional(
    previous: dict[str, float],
    current: dict[str, float],
) -> float:
    tickers = set(previous) | set(current)
    return float(
        sum(abs(current.get(ticker, 0.0) - previous.get(ticker, 0.0)) for ticker in tickers)
    )
```

Then replace the per-model loop body in `build_period_returns` so each model
carries its own book forward:

```python
    for model in models:
        model_rows = rebalances.loc[rebalances["model"].eq(model)]
        previous: dict[str, float] = {}
        for date, group in model_rows.groupby("date", sort=True):
            weights, gross = _book(group, strategy)
            if weights is None:
                traded = float("nan")
            else:
                traded = _traded_notional(previous, weights)
                previous = weights
            rows.append(
                {
                    "model": model,
                    "validation_year": int(group["validation_year"].iloc[0]),
                    "date": date,
                    "n_tickers": len(group),
                    "gross_return": gross,
                    "traded_notional": traded,
                }
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: every test in the file passes.

- [ ] **Step 5: Do not commit**

---

### Task 4: Break-even summaries

**Files:**
- Modify: `src/stock_forecaster/backtesting/simulation.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Consumes: the period frame from Tasks 1-3 and `KNOWN_MODEL_ORDER`.
- Produces: `summarize_backtest(periods: pd.DataFrame, by_year: bool = False) -> pd.DataFrame` with columns `model[, validation_year], periods, mean_gross_return, mean_traded_notional, breakeven_cost_bps, share_positive_periods`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`, adding `summarize_backtest` to the module import:

```python
def _periods(model, year, gross_values, traded_values):
    dates = pd.bdate_range("2016-01-04", periods=len(gross_values))
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": dates,
            "n_tickers": 15,
            "gross_return": list(gross_values),
            "traded_notional": list(traded_values),
        }
    )


def test_summary_reports_means_counts_and_break_even():
    periods = _periods(
        "LinearRegression",
        2016,
        gross_values=[0.004, -0.002, float("nan"), 0.001],
        traded_values=[2.0, 1.0, float("nan"), 1.0],
    )

    summary = summarize_backtest(periods)

    assert list(summary.columns) == [
        "model",
        "periods",
        "mean_gross_return",
        "mean_traded_notional",
        "breakeven_cost_bps",
        "share_positive_periods",
    ]
    mean_gross = (0.004 - 0.002 + 0.001) / 3
    mean_traded = (2.0 + 1.0 + 1.0) / 3
    assert summary.loc[0, "periods"] == 3
    assert summary.loc[0, "mean_gross_return"] == pytest.approx(mean_gross)
    assert summary.loc[0, "mean_traded_notional"] == pytest.approx(mean_traded)
    assert summary.loc[0, "breakeven_cost_bps"] == pytest.approx(
        10000 * mean_gross / mean_traded
    )
    assert summary.loc[0, "share_positive_periods"] == pytest.approx(2 / 3)


def test_break_even_is_negative_when_the_gross_edge_is_negative():
    periods = _periods("RandomForest", 2016, [-0.003, -0.001], [2.0, 2.0])

    assert summarize_backtest(periods).loc[0, "breakeven_cost_bps"] < 0


def test_break_even_is_undefined_without_trading():
    periods = _periods("LinearRegression", 2016, [0.004, 0.002], [0.0, 0.0])

    assert pd.isna(summarize_backtest(periods).loc[0, "breakeven_cost_bps"])


def test_exactly_zero_gross_return_is_not_a_positive_period():
    periods = _periods("LinearRegression", 2016, [0.0, 0.0], [2.0, 2.0])

    assert summarize_backtest(periods).loc[0, "share_positive_periods"] == pytest.approx(0.0)


def test_summary_rows_follow_known_model_order():
    periods = pd.concat(
        [
            _periods("RandomForest", 2016, [0.001, 0.002], [2.0, 2.0]),
            _periods("ZeroBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _periods("LinearRegression", 2016, [0.001, 0.002], [2.0, 2.0]),
        ],
        ignore_index=True,
    )

    summary = summarize_backtest(periods)

    assert summary["model"].tolist() == [
        "ZeroBaseline",
        "LinearRegression",
        "RandomForest",
    ]
    assert summary.loc[0, "periods"] == 0
    assert pd.isna(summary.loc[0, "breakeven_cost_bps"])


def test_annual_and_pooled_summaries_agree_on_counts():
    periods = pd.concat(
        [
            _periods("LinearRegression", 2016, [0.004, 0.002], [2.0, 2.0]),
            _periods("LinearRegression", 2017, [0.001, 0.003], [2.0, 2.0]),
        ],
        ignore_index=True,
    )

    annual = summarize_backtest(periods, by_year=True)
    pooled = summarize_backtest(periods)

    assert list(annual.columns)[:2] == ["model", "validation_year"]
    assert annual["validation_year"].tolist() == [2016, 2017]
    assert annual["periods"].sum() == pooled.loc[0, "periods"]


def test_summarize_backtest_does_not_mutate_its_input():
    periods = _periods("LinearRegression", 2016, [0.004, 0.002], [2.0, 2.0])
    original = periods.copy(deep=True)

    summarize_backtest(periods, by_year=True)

    pd.testing.assert_frame_equal(periods, original)


def test_summarize_backtest_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        summarize_backtest(pd.DataFrame())
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: collection fails with `ImportError: cannot import name 'summarize_backtest'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/stock_forecaster/backtesting/simulation.py`:

```python
def summarize_backtest(periods: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Summarize gross edge, turnover and break-even cost per model."""
    if not isinstance(periods, pd.DataFrame) or periods.empty:
        raise ValueError("periods must be a non-empty DataFrame")
    _require_columns(
        periods, ("model", "validation_year", "gross_return", "traded_notional")
    )

    group_columns = ["model", "validation_year"] if by_year else ["model"]
    rows: list[dict[str, object]] = []
    for key, group in periods.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        gross = group["gross_return"].dropna()
        traded = group["traded_notional"].dropna()
        count = int(len(gross))
        mean_gross = float(gross.mean()) if count else float("nan")
        mean_traded = float(traded.mean()) if len(traded) else float("nan")
        breakeven = float("nan")
        if count and pd.notna(mean_traded) and mean_traded != 0:
            breakeven = 10000.0 * mean_gross / mean_traded
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "periods": count,
                "mean_gross_return": mean_gross,
                "mean_traded_notional": mean_traded,
                "breakeven_cost_bps": breakeven,
                "share_positive_periods": float(gross.gt(0).mean())
                if count
                else float("nan"),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            "periods",
            "mean_gross_return",
            "mean_traded_notional",
            "breakeven_cost_bps",
            "share_positive_periods",
        ],
    )
    summary = summary.sort_values(
        by=group_columns,
        key=lambda column: column.map(_model_sort_key)
        if column.name == "model"
        else column,
    )
    return summary.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: every test in `tests/test_backtest.py` passes.

- [ ] **Step 5: Do not commit**

---

### Task 5: Document, verify, review, and create the single implementation commit

**Files:**
- Modify: `README.md`
- Verify all files from Tasks 1-4

- [ ] **Step 1: Update README before any numerical result exists**

Append to the model-evaluation narrative, after the ranking paragraphs. Invent no Step 5H numbers.

```markdown
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
```

- [ ] **Step 2: Verify public imports and signatures**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -c "import inspect; from stock_forecaster.backtesting.simulation import LONG_ONLY, LONG_SHORT, REBALANCE_STEP, STRATEGIES, build_period_returns, summarize_backtest; assert STRATEGIES == ('long_short', 'long_only'); assert REBALANCE_STEP == 5; assert inspect.signature(build_period_returns).parameters['strategy'].default == LONG_SHORT; assert inspect.signature(summarize_backtest).parameters['by_year'].default is False; print('Step 5H imports/signatures: OK')"
```

- [ ] **Step 3: Run every focused and full verification command freshly**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -v
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
```

Expected: the focused file passes, the complete suite passes with the 290 pre-existing tests plus the new backtest tests and zero failures or errors, and Ruff reports `All checks passed!`. The official evaluation is prohibited until this gate is green.

- [ ] **Step 4: Review protected scope**

```powershell
git diff --check
git diff -- src/stock_forecaster/models src/stock_forecaster/features pyproject.toml configs
git status --short -- data models
git status --short
```

Expected: the protected, config and data commands print nothing. Only `simulation.py`, `tests/test_backtest.py` and `README.md` appear.

- [ ] **Step 5: Create the single implementation commit**

```powershell
git add src/stock_forecaster/backtesting/simulation.py tests/test_backtest.py README.md
git diff --cached --name-only
git commit -m "feat: add break-even cost backtest"
```

Expected: exactly three files committed. Do not push.

---

### Task 6: Run the one official Step 5H evaluation

**Files:**
- Read: `data/processed/features.parquet`
- Create/commit: no file. The evaluation script lives in the session scratchpad, outside the repository.

- [ ] **Step 1: Record hashes before the run**

```powershell
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

Expected: the two anchors from the Global Constraints. Stop and report if either differs.

- [ ] **Step 2: Write the evaluation script in the scratchpad**

```python
"""One official Step 5H break-even cost backtest. Read-only."""

import pandas as pd

from stock_forecaster.backtesting.simulation import (
    LONG_ONLY,
    LONG_SHORT,
    build_period_returns,
    summarize_backtest,
)
from stock_forecaster.features.engineering import EXCESS_TARGET_COLUMN
from stock_forecaster.models.dataset import prepare_supervised_data, walk_forward_splits
from stock_forecaster.models.evaluation import evaluate_walk_forward

PROCESSED = (
    r"C:\Users\alber\04_Desarrollo\ai-stock-forecaster"
    r"\data\processed\features.parquet"
)
STOCKS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
          "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG")
YEARS = tuple(range(2016, 2024))
CUTOFF = pd.Timestamp("2024-01-01")
EXCESS_MODELS = ("ZeroBaseline", "MeanBaseline", "MomentumBaseline",
                 "LinearRegression", "RandomForest")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def predictions_for(features, target_column, model_names):
    if target_column is None:
        prepared = prepare_supervised_data(features)
    else:
        prepared = prepare_supervised_data(features, target_column=target_column)
    prepared = prepared.loc[
        prepared["date"].lt(CUTOFF) & prepared["target_end_date"].lt(CUTOFF)
    ].reset_index(drop=True)
    folds = walk_forward_splits(prepared, YEARS)
    require(len(folds) == len(YEARS), "unexpected fold count")
    if model_names is None:
        results = evaluate_walk_forward(folds)
    else:
        results = evaluate_walk_forward(folds, model_names=model_names)
    frame = pd.concat([result.predictions for result in results], ignore_index=True)
    require(frame["date"].lt(CUTOFF).all(), "2024+ leakage in predictions")
    require(not frame["ticker"].eq("SPY").any(), "SPY entered the modeled universe")
    return frame


def report(label, predictions):
    for strategy in (LONG_SHORT, LONG_ONLY):
        periods = build_period_returns(predictions, strategy=strategy)
        print(f"\n\n===== {label} / {strategy} =====", flush=True)
        print(f"rebalance rows: {len(periods)}")
        print(f"\n--- pooled ---")
        print(summarize_backtest(periods).to_string(index=False))
        print(f"\n--- annual ---")
        print(summarize_backtest(periods, by_year=True).to_string(index=False))


def main():
    features = pd.read_parquet(PROCESSED)
    require(set(features["ticker"].unique()) == set(STOCKS) | {"SPY"},
            "processed universe mismatch")
    without_spy = features.loc[features["ticker"].ne("SPY")].copy()

    report("ABSOLUTE", predictions_for(without_spy, None, None))
    report("EXCESS", predictions_for(without_spy, EXCESS_TARGET_COLUMN, EXCESS_MODELS))

    print("\nBreak-even is an output, not an assumption. Short borrow and market "
          "impact are not modeled, so these figures are optimistic. No tuning, "
          "no TEST, no persisted artifact.", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it once**

Expected: four sections, each with a pooled and an annual table. Record every table verbatim. Do not re-run with different buckets, schedule, or weighting whatever the result.

- [ ] **Step 4: Record hashes after the run**

```powershell
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
git status --short
```

Expected: both hashes identical to Step 1 and a clean tracked tree.

---

### Task 7: Whole-branch review and final verification

**Files:**
- Review the committed implementation and README
- Modify only a proven defect, test-first; never tune on a Step 5H result

- [ ] **Step 1: Review every invariant**

Confirm against the branch diff: no existing module changed; `simulation.py` additive; no cost level assumed anywhere; both strategies fixed and equal-weight; undefined periods null rather than zero and carrying the book forward; establishment cost included; turnover sequenced across years and per model; tie-breaking deterministic; TEST guard present and tested; `BUCKET_SIZE`, `select_non_overlapping` and `KNOWN_MODEL_ORDER` reused from `ranking.py`; no tuning; no persistence; no new dependency. Classify findings as Critical, Important or Minor, and state explicitly if there are none.

- [ ] **Step 2: Repair any valid defect RED-GREEN, then amend**

Add the smallest failing test, observe the intended failure, apply the minimum fix, rerun focused and full checks, then `git commit --amend --no-edit`. Record the new SHA. A disappointing break-even figure is not a defect.

- [ ] **Step 3: Run fresh final verification**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
git status --short
git diff --exit-code
git diff --cached --exit-code
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

Expected: suite and Ruff green, tracked tree and index clean, both hashes equal to their anchors.

- [ ] **Step 4: Report and stop**

Report the implementation commit and SHA, the RED and GREEN evidence per task, the full verification output, the hashes before and after the official run, all four break-even sections, the comparison across targets and strategies, the review findings, and confirmation that no cost level was assumed and no tuning, TEST evaluation, merge or push occurred. State the three limitations from the specification. Then stop and wait for approval.

## Test Coverage Map

- `tests/test_backtest.py`: long/short and long-only gross return arithmetic; deterministic ticker tie-break; undefined periods for constant predictions and for too few tickers; rebalance date selection; known model order; establishment, zero, and full-reversal turnover; book carried forward across undefined dates and year boundaries; per-model book isolation; break-even formula, its negative case and its undefined case; strictly-positive share rule; annual and pooled consistency; input immutability; rejection of empty input, missing columns, duplicates, unknown strategy, non-positive step, and 2024+ rows.
- Existing suites remain regression coverage and must stay green.

## Controlled Execution and Failure Boundaries

1. No official evaluation before the full suite, Ruff, scope review and the single implementation commit are green.
2. No second run with a different bucket size, schedule, weighting or strategy, whatever the first shows.
3. No cost level is assumed at any point.
4. Any hash mismatch, tracked-tree surprise or protected-module change stops the phase with evidence intact.
5. No TEST read, no persisted output, no merge and no push without explicit authorization.

## Plan Self-Review

Every specification section maps to a task. Task 1 covers the input contract, validation, the TEST guard, the long/short gross return and presentation order. Task 2 covers the long-only strategy. Task 3 covers traded notional, establishment cost, book carry-forward and cross-year sequencing. Task 4 covers both summaries and the break-even formula with its negative and undefined cases. Task 5 covers documentation, verification, scope review and the single commit. Task 6 covers the one official run with hash-proven read-only behavior across both targets and both strategies. Task 7 covers review, defect repair by amendment, final verification and the bounded report.

Names are consistent across tasks: `LONG_SHORT`, `LONG_ONLY`, `STRATEGIES`, `REBALANCE_STEP`, `PERIOD_COLUMNS`, `_require_columns`, `_model_sort_key`, `_validate`, `_book`, `_traded_notional`, `build_period_returns` and `summarize_backtest` are each defined once and referenced identically afterwards. `_book` is introduced in Task 1 and used unchanged thereafter; the period loop written in Task 1 is replaced, not renamed, in Task 3. No placeholder or deferred decision remains.
