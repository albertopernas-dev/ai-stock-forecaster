# Cross-Sectional Ranking Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the existing forecasts order the 15 stocks correctly within each trading date, for both the absolute and excess targets, without changing any model or touching TEST.

**Architecture:** One new additive module `ranking.py` consumes the prediction frames `evaluate_walk_forward` already returns and reduces them to one row per model and date carrying a Spearman information coefficient and a tercile spread. Two further functions produce a deterministic non-overlapping subsample and aggregate summaries. No existing module changes, nothing is persisted, and predictions are regenerated in memory from the unchanged deterministic evaluation.

**Tech Stack:** Python 3.12.14, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-22-cross-sectional-ranking-design.md`

## Global Constraints

- **The specification is binding** and wins over any ambiguous wording in this plan.
- **TEST is 2024-01-01 onward and is never read.** `build_daily_ranking` raises on any row dated `2024-01-01` or later. Development filtering applies both `date < 2024-01-01` and `target_end_date < 2024-01-01`.
- **No tuning.** `LinearRegressionForecaster` and `RandomForestForecaster` keep their exact current configuration. A disappointing number is not a defect and must never trigger a re-run with different settings.
- **Protected and untouched:** `linear.py`, `forest.py`, `metrics.py`, `evaluation.py`, `dataset.py`, `engineering.py`, `baseline.py`, `pyproject.toml`, `configs/`. `ranking.py` is purely additive.
- **No new dependency.** Rank correlation comes from `pandas.Series.corr(method="spearman")`.
- **Nothing is persisted.** No prediction, ranking, summary, model artifact, or dataset is written. `data/raw/prices.parquet` and `data/processed/features.parquet` are read-only and hash-verified around the official run.
- **Anchors that must not change:** raw `49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5`, processed `C717BFEE645C312CFD0086B6A1514D15CC47E0E4EED4D6557218B3EDC272DA57`.
- **One implementation commit.** Tasks 1-4 build and review uncommitted increments; Task 5 creates the single tracked commit. This carries forward the project's established ruling and overrides the generic per-task commit template. Any later fix amends that commit rather than adding one.
- **Windows environment.** Run every Python command with the shared interpreter `C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe`, set `$env:PYTHONPATH` to the worktree's `src`, and pass `--basetemp=<writable dir>` to pytest because the inherited `.pytest_cache` is unreadable.
- **No claim** of profitability, investability, statistical significance, causality, economic significance, TEST or backtest performance, or any composite score. The tercile spread is never described as a strategy return.

## File Map

- Create: `src/stock_forecaster/models/ranking.py` — the whole ranking evaluation surface: constants, input validation, `build_daily_ranking`, `select_non_overlapping`, `summarize_ranking`. One responsibility, no dependency on anything but `pandas` and the public `DEFAULT_MODEL_NAMES`.
- Create: `tests/test_ranking.py` — all synthetic deterministic coverage for the above.
- Modify: `README.md` — document the ranking lens, both measurements, the overlap caveat, the breadth ceiling, and the untouched TEST boundary.
- Read only during the official run: `data/processed/features.parquet`.

---

### Task 1: Validated daily frame and cross-sectional information coefficient

**Files:**
- Create: `src/stock_forecaster/models/ranking.py`
- Create: `tests/test_ranking.py`

**Interfaces:**
- Consumes: `PREDICTION_COLUMNS` and `DEFAULT_MODEL_NAMES` from `stock_forecaster.models.evaluation` (both public, neither modified).
- Produces: `MIN_TICKERS_PER_DATE: int = 5`, `BUCKET_SIZE: int = 5`, `TEST_BOUNDARY: pd.Timestamp`, `KNOWN_MODEL_ORDER: tuple[str, ...]`, and `build_daily_ranking(predictions: pd.DataFrame) -> pd.DataFrame` returning columns `model, validation_year, date, n_tickers, ic, top_mean, bottom_mean, spread`. Tasks 2-4 and 6 rely on exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ranking.py`:

```python
import pandas as pd
import pytest

from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES
from stock_forecaster.models.ranking import (
    KNOWN_MODEL_ORDER,
    MIN_TICKERS_PER_DATE,
    build_daily_ranking,
)


def _predictions(date, model, tickers, y_true, prediction, year=2016):
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


def test_known_model_order_matches_evaluation_presentation_order():
    assert KNOWN_MODEL_ORDER == ("ZeroBaseline", *DEFAULT_MODEL_NAMES)


def test_perfectly_ordered_predictions_give_ic_of_one():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )

    daily = build_daily_ranking(frame)

    assert list(daily.columns) == [
        "model",
        "validation_year",
        "date",
        "n_tickers",
        "ic",
        "top_mean",
        "bottom_mean",
        "spread",
    ]
    assert len(daily) == 1
    assert daily.loc[0, "n_tickers"] == 5
    assert daily.loc[0, "ic"] == pytest.approx(1.0)


def test_perfectly_inverted_predictions_give_ic_of_minus_one():
    frame = _predictions(
        "2016-03-01",
        "RandomForest",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )

    assert build_daily_ranking(frame).loc[0, "ic"] == pytest.approx(-1.0)


def test_one_swapped_pair_matches_independently_computed_spearman():
    # Ranks differ only in the last two positions: sum of squared rank
    # differences is 2, so rho = 1 - (6 * 2) / (5 * (25 - 1)) = 0.9
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.01, 0.02],
        [0.5, 0.4, 0.3, 0.2, 0.1],
    )

    assert build_daily_ranking(frame).loc[0, "ic"] == pytest.approx(0.9)


def test_constant_predictions_give_undefined_ic_not_zero():
    frame = _predictions(
        "2016-03-01",
        "ZeroBaseline",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert pd.isna(build_daily_ranking(frame).loc[0, "ic"])


def test_constant_realized_values_give_undefined_ic():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.02, 0.02, 0.02, 0.02, 0.02],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )

    assert pd.isna(build_daily_ranking(frame).loc[0, "ic"])


def test_too_few_tickers_give_undefined_ic_but_record_the_count():
    tickers = [f"T{index}" for index in range(MIN_TICKERS_PER_DATE - 1)]
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        tickers,
        [0.04, 0.03, 0.02, 0.01],
        [0.4, 0.3, 0.2, 0.1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "n_tickers"] == MIN_TICKERS_PER_DATE - 1
    assert pd.isna(daily.loc[0, "ic"])


def test_rows_dated_on_or_after_the_test_boundary_are_rejected():
    frame = _predictions(
        "2024-01-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
        year=2023,
    )

    with pytest.raises(ValueError, match="2024"):
        build_daily_ranking(frame)


def test_duplicate_date_ticker_model_rows_are_rejected():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        build_daily_ranking(duplicated)


@pytest.mark.parametrize("missing", ["prediction", "y_true", "model", "date"])
def test_missing_required_columns_are_rejected(missing):
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    ).drop(columns=[missing])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_daily_ranking(frame)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_daily_ranking(pd.DataFrame())


def test_build_daily_ranking_does_not_mutate_its_input():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )
    original = frame.copy(deep=True)

    build_daily_ranking(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_models_are_emitted_in_known_presentation_order():
    tickers = ["A", "B", "C", "D", "E"]
    realized = [0.05, 0.04, 0.03, 0.02, 0.01]
    predicted = [0.9, 0.7, 0.5, 0.3, 0.1]
    shuffled = pd.concat(
        [
            _predictions("2016-03-01", "RandomForest", tickers, realized, predicted),
            _predictions("2016-03-01", "MeanBaseline", tickers, realized, predicted),
            _predictions("2016-03-01", "LinearRegression", tickers, realized, predicted),
        ],
        ignore_index=True,
    )

    daily = build_daily_ranking(shuffled)

    assert daily["model"].tolist() == [
        "MeanBaseline",
        "LinearRegression",
        "RandomForest",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
$env:PYTHONPATH = 'C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.worktrees\ranking-evaluation\src'
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: every test in the file errors during collection with `ModuleNotFoundError: No module named 'stock_forecaster.models.ranking'`. Record the exact count.

- [ ] **Step 3: Write the minimal implementation**

Create `src/stock_forecaster/models/ranking.py`:

```python
"""Cross-sectional ranking evaluation of out-of-sample return forecasts."""

import pandas as pd

from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES, PREDICTION_COLUMNS

MIN_TICKERS_PER_DATE = 5
BUCKET_SIZE = 5
TEST_BOUNDARY = pd.Timestamp("2024-01-01")
KNOWN_MODEL_ORDER = ("ZeroBaseline", *DEFAULT_MODEL_NAMES)

DAILY_COLUMNS = (
    "model",
    "validation_year",
    "date",
    "n_tickers",
    "ic",
    "top_mean",
    "bottom_mean",
    "spread",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
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


def _model_sort_key(model_name: str) -> int:
    if model_name in KNOWN_MODEL_ORDER:
        return KNOWN_MODEL_ORDER.index(model_name)
    return len(KNOWN_MODEL_ORDER)


def _daily_row(model: str, year: int, date: pd.Timestamp, group: pd.DataFrame) -> dict:
    prediction = group["prediction"]
    realized = group["y_true"]
    n_tickers = len(group)
    predictions_vary = prediction.nunique(dropna=True) > 1

    ic = float("nan")
    if (
        n_tickers >= MIN_TICKERS_PER_DATE
        and predictions_vary
        and realized.nunique(dropna=True) > 1
    ):
        ic = float(prediction.corr(realized, method="spearman"))

    return {
        "model": model,
        "validation_year": year,
        "date": date,
        "n_tickers": n_tickers,
        "ic": ic,
        "top_mean": float("nan"),
        "bottom_mean": float("nan"),
        "spread": float("nan"),
    }


def build_daily_ranking(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one cross-sectional ranking row per model and trading date."""
    ordered = _validate_predictions(predictions)
    rows = [
        _daily_row(model, year, date, group)
        for (model, year, date), group in ordered.groupby(
            ["model", "validation_year", "date"], sort=False
        )
    ]
    frame = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    frame = frame.sort_values(
        by=["model", "validation_year", "date"],
        key=lambda column: column.map(_model_sort_key)
        if column.name == "model"
        else column,
    )
    return frame.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: all tests written in Step 1 pass. The spread columns exist and are `NaN` everywhere, which Task 2 fills in.

- [ ] **Step 5: Do not commit**

Per the Global Constraints, Tasks 1-4 leave the change uncommitted. Confirm with `git status --short` that exactly `src/stock_forecaster/models/ranking.py` and `tests/test_ranking.py` are new and untracked.

---

### Task 2: Tercile spread

**Files:**
- Modify: `src/stock_forecaster/models/ranking.py`
- Modify: `tests/test_ranking.py`

**Interfaces:**
- Consumes: `BUCKET_SIZE`, `_daily_row` and `build_daily_ranking` from Task 1.
- Produces: `top_mean`, `bottom_mean` and `spread` populated in the frame `build_daily_ranking` already returns. No signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ranking.py`, and extend the import from `stock_forecaster.models.ranking` to also bring in `BUCKET_SIZE`:

```python
def _ten_ticker_frame(y_true, prediction, model="LinearRegression"):
    tickers = [f"T{index:02d}" for index in range(1, 11)]
    return _predictions("2016-03-01", model, tickers, y_true, prediction)


def test_spread_uses_top_and_bottom_buckets_with_exact_arithmetic():
    frame = _ten_ticker_frame(
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "top_mean"] == pytest.approx(0.08)
    assert daily.loc[0, "bottom_mean"] == pytest.approx(0.03)
    assert daily.loc[0, "spread"] == pytest.approx(0.05)


def test_ties_at_the_bucket_boundary_resolve_by_ascending_ticker():
    # T05 and T06 tie on prediction; ascending ticker puts T05 in the top
    # bucket, so the top bucket collects the single 1.0 realized value.
    frame = _ten_ticker_frame(
        y_true=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        prediction=[10, 9, 8, 7, 6, 6, 4, 3, 2, 1],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "top_mean"] == pytest.approx(0.2)
    assert daily.loc[0, "bottom_mean"] == pytest.approx(0.0)
    assert daily.loc[0, "spread"] == pytest.approx(0.2)


def test_fewer_tickers_than_two_buckets_give_an_undefined_spread():
    count = 2 * BUCKET_SIZE - 1
    tickers = [f"T{index:02d}" for index in range(1, count + 1)]
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        tickers,
        [0.01 * index for index in range(count, 0, -1)],
        [float(index) for index in range(count, 0, -1)],
    )

    daily = build_daily_ranking(frame)

    assert daily.loc[0, "n_tickers"] == count
    assert pd.notna(daily.loc[0, "ic"])
    assert pd.isna(daily.loc[0, "spread"])


def test_constant_predictions_give_an_undefined_spread():
    frame = _ten_ticker_frame(
        y_true=[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        prediction=[0.0] * 10,
        model="MeanBaseline",
    )

    daily = build_daily_ranking(frame)

    assert pd.isna(daily.loc[0, "spread"])
    assert pd.isna(daily.loc[0, "top_mean"])
    assert pd.isna(daily.loc[0, "bottom_mean"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: the four new tests fail on the `top_mean`, `bottom_mean` and `spread` assertions because Task 1 leaves them `NaN`. `test_constant_predictions_give_an_undefined_spread` may already pass; that is acceptable and it stays as a regression guard.

- [ ] **Step 3: Write the minimal implementation**

Replace the return block of `_daily_row` in `src/stock_forecaster/models/ranking.py` with bucket computation:

```python
    top_mean = float("nan")
    bottom_mean = float("nan")
    spread = float("nan")
    if n_tickers >= 2 * BUCKET_SIZE and predictions_vary:
        ranked = group.sort_values(
            by=["prediction", "ticker"], ascending=[False, True]
        )
        top_mean = float(ranked["y_true"].iloc[:BUCKET_SIZE].mean())
        bottom_mean = float(ranked["y_true"].iloc[-BUCKET_SIZE:].mean())
        spread = top_mean - bottom_mean

    return {
        "model": model,
        "validation_year": year,
        "date": date,
        "n_tickers": n_tickers,
        "ic": ic,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "spread": spread,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: every test in the file passes, Task 1 tests included.

- [ ] **Step 5: Do not commit**

---

### Task 3: Deterministic non-overlapping subsample

**Files:**
- Modify: `src/stock_forecaster/models/ranking.py`
- Modify: `tests/test_ranking.py`

**Interfaces:**
- Consumes: the daily frame produced by Tasks 1-2.
- Produces: `select_non_overlapping(daily: pd.DataFrame, step: int = 5) -> pd.DataFrame`, returning the same schema filtered to every `step`-th distinct date within each validation year.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ranking.py` and add `select_non_overlapping` to the module import:

```python
def _daily_frame(dates, year, model="LinearRegression"):
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": pd.to_datetime(dates),
            "n_tickers": 15,
            "ic": 0.1,
            "top_mean": 0.02,
            "bottom_mean": 0.01,
            "spread": 0.01,
        }
    )


def test_non_overlapping_selection_takes_every_fifth_date_within_each_year():
    first = pd.bdate_range("2016-01-04", periods=12)
    second = pd.bdate_range("2017-01-02", periods=12)
    daily = pd.concat(
        [_daily_frame(first, 2016), _daily_frame(second, 2017)], ignore_index=True
    )

    selected = select_non_overlapping(daily, step=5)

    assert selected.loc[selected.validation_year.eq(2016), "date"].tolist() == [
        first[0],
        first[5],
        first[10],
    ]
    assert selected.loc[selected.validation_year.eq(2017), "date"].tolist() == [
        second[0],
        second[5],
        second[10],
    ]
    assert list(selected.columns) == list(daily.columns)


def test_non_overlapping_selection_keeps_every_model_on_a_selected_date():
    dates = pd.bdate_range("2016-01-04", periods=6)
    daily = pd.concat(
        [
            _daily_frame(dates, 2016, model="LinearRegression"),
            _daily_frame(dates, 2016, model="RandomForest"),
        ],
        ignore_index=True,
    )

    selected = select_non_overlapping(daily, step=5)

    assert sorted(selected["model"].unique()) == ["LinearRegression", "RandomForest"]
    assert selected["date"].nunique() == 2


def test_non_overlapping_selection_does_not_mutate_its_input():
    daily = _daily_frame(pd.bdate_range("2016-01-04", periods=12), 2016)
    original = daily.copy(deep=True)

    select_non_overlapping(daily, step=5)

    pd.testing.assert_frame_equal(daily, original)


@pytest.mark.parametrize("step", [0, -1])
def test_non_positive_step_is_rejected(step):
    daily = _daily_frame(pd.bdate_range("2016-01-04", periods=12), 2016)

    with pytest.raises(ValueError, match="step must be a positive integer"):
        select_non_overlapping(daily, step=step)
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: collection fails with `ImportError: cannot import name 'select_non_overlapping'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/stock_forecaster/models/ranking.py`:

```python
def select_non_overlapping(daily: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Return every step-th distinct date within each validation year."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("daily must be a non-empty DataFrame")
    _require_columns(daily, ("validation_year", "date"))

    keep: set[tuple[object, pd.Timestamp]] = set()
    for year, group in daily.groupby("validation_year", sort=True):
        dates = sorted(pd.to_datetime(group["date"]).unique())
        keep.update((year, date) for date in dates[::step])

    selected = [
        (year, date) in keep
        for year, date in zip(
            daily["validation_year"], pd.to_datetime(daily["date"]), strict=True
        )
    ]
    return daily.loc[selected].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: all tests in the file pass.

- [ ] **Step 5: Do not commit**

---

### Task 4: Ranking summaries

**Files:**
- Modify: `src/stock_forecaster/models/ranking.py`
- Modify: `tests/test_ranking.py`

**Interfaces:**
- Consumes: the daily frame from Tasks 1-2 and `KNOWN_MODEL_ORDER`.
- Produces: `summarize_ranking(daily: pd.DataFrame, by_year: bool = False) -> pd.DataFrame` with columns `model[, validation_year], ic_days, ic_mean, ic_std, ic_share_positive, ic_stability, spread_days, spread_mean, spread_share_positive`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ranking.py` and add `summarize_ranking` to the module import:

```python
def _daily_rows(model, year, ic_values, spread_values):
    dates = pd.bdate_range("2016-01-04", periods=len(ic_values))
    return pd.DataFrame(
        {
            "model": model,
            "validation_year": year,
            "date": dates,
            "n_tickers": 15,
            "ic": list(ic_values),
            "top_mean": 0.0,
            "bottom_mean": 0.0,
            "spread": list(spread_values),
        }
    )


def test_summary_aggregates_only_defined_dates_and_reports_both_counts():
    daily = _daily_rows(
        "LinearRegression",
        2016,
        ic_values=[0.2, -0.1, float("nan"), 0.3],
        spread_values=[0.01, -0.02, float("nan"), float("nan")],
    )

    summary = summarize_ranking(daily)

    assert list(summary.columns) == [
        "model",
        "ic_days",
        "ic_mean",
        "ic_std",
        "ic_share_positive",
        "ic_stability",
        "spread_days",
        "spread_mean",
        "spread_share_positive",
    ]
    assert summary.loc[0, "ic_days"] == 3
    assert summary.loc[0, "spread_days"] == 2
    assert summary.loc[0, "ic_mean"] == pytest.approx((0.2 - 0.1 + 0.3) / 3)
    assert summary.loc[0, "ic_share_positive"] == pytest.approx(2 / 3)
    assert summary.loc[0, "spread_mean"] == pytest.approx((0.01 - 0.02) / 2)
    assert summary.loc[0, "spread_share_positive"] == pytest.approx(0.5)


def test_exactly_zero_is_not_counted_as_positive():
    daily = _daily_rows(
        "LinearRegression", 2016, ic_values=[0.0, 0.0], spread_values=[0.0, 0.0]
    )

    summary = summarize_ranking(daily)

    assert summary.loc[0, "ic_share_positive"] == pytest.approx(0.0)
    assert summary.loc[0, "spread_share_positive"] == pytest.approx(0.0)


def test_stability_is_the_sample_dispersion_ratio():
    daily = _daily_rows(
        "LinearRegression",
        2016,
        ic_values=[0.2, -0.1, 0.3],
        spread_values=[0.0, 0.0, 0.0],
    )

    summary = summarize_ranking(daily)
    values = pd.Series([0.2, -0.1, 0.3])

    assert summary.loc[0, "ic_std"] == pytest.approx(values.std(ddof=1))
    assert summary.loc[0, "ic_stability"] == pytest.approx(
        values.mean() / values.std(ddof=1)
    )


@pytest.mark.parametrize(
    "ic_values", [[0.2], [0.2, 0.2]], ids=["single-date", "zero-dispersion"]
)
def test_stability_is_undefined_without_usable_dispersion(ic_values):
    daily = _daily_rows(
        "LinearRegression", 2016, ic_values, spread_values=[0.0] * len(ic_values)
    )

    assert pd.isna(summarize_ranking(daily).loc[0, "ic_stability"])


def test_summary_rows_follow_known_model_order():
    daily = pd.concat(
        [
            _daily_rows("RandomForest", 2016, [0.1, 0.2], [0.0, 0.0]),
            _daily_rows("ZeroBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _daily_rows("MeanBaseline", 2016, [float("nan")] * 2, [float("nan")] * 2),
            _daily_rows("LinearRegression", 2016, [0.1, 0.2], [0.0, 0.0]),
        ],
        ignore_index=True,
    )

    summary = summarize_ranking(daily)

    assert summary["model"].tolist() == [
        "ZeroBaseline",
        "MeanBaseline",
        "LinearRegression",
        "RandomForest",
    ]
    assert summary.loc[0, "ic_days"] == 0
    assert pd.isna(summary.loc[0, "ic_mean"])


def test_annual_summary_splits_by_year_and_matches_the_pooled_totals():
    daily = pd.concat(
        [
            _daily_rows("LinearRegression", 2016, [0.1, 0.3], [0.01, 0.03]),
            _daily_rows("LinearRegression", 2017, [0.2, 0.4], [0.02, 0.04]),
        ],
        ignore_index=True,
    )

    annual = summarize_ranking(daily, by_year=True)
    pooled = summarize_ranking(daily)

    assert list(annual.columns)[:2] == ["model", "validation_year"]
    assert annual["validation_year"].tolist() == [2016, 2017]
    assert annual["ic_days"].sum() == pooled.loc[0, "ic_days"]
    assert pooled.loc[0, "ic_mean"] == pytest.approx((0.1 + 0.3 + 0.2 + 0.4) / 4)


def test_summarize_ranking_does_not_mutate_its_input():
    daily = _daily_rows("LinearRegression", 2016, [0.1, 0.2], [0.01, 0.02])
    original = daily.copy(deep=True)

    summarize_ranking(daily, by_year=True)

    pd.testing.assert_frame_equal(daily, original)
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: collection fails with `ImportError: cannot import name 'summarize_ranking'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/stock_forecaster/models/ranking.py`:

```python
def _describe(values: pd.Series) -> tuple[int, float, float, float]:
    defined = values.dropna()
    count = int(len(defined))
    if count == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(defined.mean())
    deviation = float(defined.std(ddof=1)) if count >= 2 else float("nan")
    share_positive = float(defined.gt(0).mean())
    return count, mean, deviation, share_positive


def summarize_ranking(daily: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Summarize daily ranking evidence per model, optionally per year."""
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("daily must be a non-empty DataFrame")
    _require_columns(daily, ("model", "validation_year", "ic", "spread"))

    group_columns = ["model", "validation_year"] if by_year else ["model"]
    rows: list[dict[str, object]] = []
    for key, group in daily.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        ic_days, ic_mean, ic_std, ic_share = _describe(group["ic"])
        spread_days, spread_mean, _, spread_share = _describe(group["spread"])
        stability = float("nan")
        if ic_days >= 2 and pd.notna(ic_std) and ic_std != 0:
            stability = ic_mean / ic_std
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "ic_days": ic_days,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_share_positive": ic_share,
                "ic_stability": stability,
                "spread_days": spread_days,
                "spread_mean": spread_mean,
                "spread_share_positive": spread_share,
            }
        )
        rows.append(row)

    summary = pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            "ic_days",
            "ic_mean",
            "ic_std",
            "ic_share_positive",
            "ic_stability",
            "spread_days",
            "spread_mean",
            "spread_share_positive",
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
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
```

Expected: every test in `tests/test_ranking.py` passes.

- [ ] **Step 5: Do not commit**

---

### Task 5: Document, verify, review, and create the single implementation commit

**Files:**
- Modify: `README.md`
- Verify all files from Tasks 1-4
- Do not touch data in this task

- [ ] **Step 1: Update README before any numerical result exists**

Add to the model-evaluation narrative, after the existing walk-forward paragraph. Do not invent Step 5G numbers.

```markdown
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

Because the horizon is five sessions while observations are daily, consecutive
dates share most of their target window and overlapping statistics overstate
consistency. Every summary is therefore reported twice: over all dates, and
over a non-overlapping subsample taking every fifth trading date within each
validation year. The spread is a raw difference of realized returns: it applies
no transaction cost, position sizing, or compounding and is never a strategy
return. TEST remains 2024+ and the ranking module rejects any row dated on or
after 2024-01-01. Controlled evaluation remains pending; numerical results are
deferred to the real run.
```

- [ ] **Step 2: Verify public imports and signatures**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -c "import inspect; from stock_forecaster.models.ranking import BUCKET_SIZE, KNOWN_MODEL_ORDER, MIN_TICKERS_PER_DATE, TEST_BOUNDARY, build_daily_ranking, select_non_overlapping, summarize_ranking; from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES; assert MIN_TICKERS_PER_DATE == 5 and BUCKET_SIZE == 5; assert str(TEST_BOUNDARY.date()) == '2024-01-01'; assert KNOWN_MODEL_ORDER == ('ZeroBaseline', *DEFAULT_MODEL_NAMES); assert inspect.signature(select_non_overlapping).parameters['step'].default == 5; assert inspect.signature(summarize_ranking).parameters['by_year'].default is False; print('Step 5G imports/signatures: OK')"
```

- [ ] **Step 3: Run every focused and full verification command freshly**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -v --basetemp=$env:TEMP\rank-tmp
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v --basetemp=$env:TEMP\rank-tmp
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
```

Expected: the focused file passes, the complete suite passes with 257 pre-existing tests plus the new ranking tests and zero failures or errors, and Ruff reports `All checks passed!`. The official evaluation is prohibited until this gate is green.

- [ ] **Step 4: Review protected scope**

```powershell
git diff --check
git diff -- src/stock_forecaster/models/linear.py src/stock_forecaster/models/forest.py src/stock_forecaster/models/metrics.py src/stock_forecaster/models/evaluation.py src/stock_forecaster/models/dataset.py src/stock_forecaster/models/baseline.py src/stock_forecaster/features/engineering.py pyproject.toml configs
git status --short -- data models
git status --short
```

Expected: the protected, config and data commands print nothing. Only `ranking.py`, `tests/test_ranking.py` and `README.md` appear.

- [ ] **Step 5: Create the single implementation commit**

```powershell
git add src/stock_forecaster/models/ranking.py tests/test_ranking.py README.md
git diff --cached --name-only
git commit -m "feat: add cross-sectional ranking evaluation"
```

Expected: exactly three files committed. Do not push.

---

### Task 6: Run the one official Step 5G evaluation

**Files:**
- Read: `data/processed/features.parquet`
- Create/commit: no file. The evaluation script lives in the ignored SDD workspace.

**Ordering invariant:** one Python process regenerates both prediction sets and produces every table. No result may trigger a second run with different settings.

- [ ] **Step 1: Record hashes before the run**

```powershell
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

Expected: raw `49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5`, processed `C717BFEE645C312CFD0086B6A1514D15CC47E0E4EED4D6557218B3EDC272DA57`. Stop and report if either differs.

- [ ] **Step 2: Write the evaluation script in the ignored workspace**

Save as `.superpowers/sdd/2026-09-22-cross-sectional-ranking/task-6-evaluate.py`:

```python
"""One official Step 5G cross-sectional ranking evaluation. Read-only."""

import pandas as pd

from stock_forecaster.features.engineering import EXCESS_TARGET_COLUMN
from stock_forecaster.models.dataset import prepare_supervised_data, walk_forward_splits
from stock_forecaster.models.evaluation import evaluate_walk_forward
from stock_forecaster.models.ranking import (
    build_daily_ranking,
    select_non_overlapping,
    summarize_ranking,
)

PROCESSED = r"C:\Users\alber\04_Desarrollo\ai-stock-forecaster\data\processed\features.parquet"
STOCKS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
          "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG")
YEARS = tuple(range(2016, 2024))
CUTOFF = pd.Timestamp("2024-01-01")
EXCESS_MODELS = ("ZeroBaseline", "MeanBaseline", "MomentumBaseline",
                 "LinearRegression", "RandomForest")


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
    require(not prepared.empty, "development frame is empty")
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
    daily = build_daily_ranking(predictions)
    sparse = select_non_overlapping(daily, step=5)
    print(f"\n===== {label} =====", flush=True)
    print(f"rows={len(predictions)} daily={len(daily)} non_overlapping={len(sparse)}")
    for name, frame in (("PRIMARY", daily), ("NON-OVERLAPPING", sparse)):
        print(f"\n--- {label} / {name} / pooled ---")
        print(summarize_ranking(frame).to_string(index=False))
        print(f"\n--- {label} / {name} / annual ---")
        print(summarize_ranking(frame, by_year=True).to_string(index=False))


def main():
    features = pd.read_parquet(PROCESSED)
    require(set(features["ticker"].unique()) == set(STOCKS) | {"SPY"},
            "processed universe mismatch")
    without_spy = features.loc[features["ticker"].ne("SPY")].copy()

    report("ABSOLUTE future_return_5d", predictions_for(without_spy, None, None))
    report("EXCESS future_excess_return_5d",
           predictions_for(without_spy, EXCESS_TARGET_COLUMN, EXCESS_MODELS))

    print("\nOne official configuration. No tuning, no TEST, no backtest, "
          "no persisted artifact.", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it once**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe .superpowers\sdd\2026-09-22-cross-sectional-ranking\task-6-evaluate.py
```

Expected: eight tables. The absolute run must contain exactly the four Step 5E default models and 29,580 prediction rows per model; the excess run exactly the five selected models. Record every table verbatim.

- [ ] **Step 4: Record hashes after the run and confirm nothing changed**

```powershell
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
git status --short
```

Expected: both hashes identical to Step 1 and a clean tracked tree, proving the run was read-only.

---

### Task 7: Whole-branch review and final verification

**Files:**
- Review the committed implementation and README
- Modify only a proven defect, test-first; never tune on a Step 5G result

- [ ] **Step 1: Review every invariant**

Confirm against the branch diff: no existing module changed; `ranking.py` additive; undefined cases return null rather than zero; tie-breaking deterministic; TEST guard present and tested; both counts reported; stability labelled descriptive; overlap handled by a second reported pass; no tuning; no persistence; no new dependency; no generated artifact committed. Classify findings as Critical, Important or Minor, and state explicitly if there are none.

- [ ] **Step 2: Repair any valid defect RED-GREEN, then amend**

Add the smallest failing test, observe the intended failure, apply the minimum fix, rerun focused and full checks, then `git commit --amend --no-edit` so the implementation stays one commit. Record the new SHA. A disappointing metric is not a defect.

- [ ] **Step 3: Run fresh final verification**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v --basetemp=$env:TEMP\rank-tmp
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
git status --short
git diff --exit-code
git diff --cached --exit-code
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

Expected: suite and Ruff green, tracked tree and index clean, both hashes equal to their anchors.

- [ ] **Step 4: Report and stop**

Report the implementation commit and SHA, the RED and GREEN evidence per task, the full verification output, the hashes before and after the official run, all eight ranking tables, the absolute-versus-excess comparison, the review findings and resolutions, and confirmation that no tuning, TEST evaluation, backtest, push or merge occurred. Then stop and wait for approval. Do not start portfolio construction.

## Test Coverage Map

- `tests/test_ranking.py`: known-order consistency with `evaluation.py`; perfect, inverted and independently computed Spearman values; undefined coefficient for constant predictions, constant realized values and too few tickers; exact bucket arithmetic; deterministic ticker tie-break; undefined spread below two buckets and for constant predictions; every fifth date per year; all models kept on a selected date; positive-step validation; aggregation over defined dates only with both counts; exactly zero not positive; sample dispersion ratio and its undefined cases; known model order in summaries; annual and pooled consistency; input immutability in all three functions; rejection of empty input, missing columns, duplicates and 2024+ rows.
- Existing suites remain regression coverage and must stay green unchanged.

## Controlled Execution and Failure Boundaries

1. No official evaluation before the full suite, Ruff, scope review and the single implementation commit are green.
2. No second evaluation run with a different model set, step, bucket size or threshold, whatever the first run shows.
3. Any hash mismatch, tracked-tree surprise or protected-module change stops the phase with the worktree and evidence intact.
4. No TEST read, no persisted output, no push and no merge without explicit authorization.

## Plan Self-Review

Every specification section maps to a task. Task 1 covers the input contract, validation, the TEST guard, the information coefficient and its undefined cases, and presentation order. Task 2 covers the tercile spread, its arithmetic, its tie-break and its own undefined cases. Task 3 covers the non-overlapping subsample. Task 4 covers both summaries, the separate `ic_days` and `spread_days` counts, the strictly-positive share rule and the descriptive stability ratio with its undefined cases. Task 5 covers documentation, verification, scope review and the single commit. Task 6 covers the one official run with hash-proven read-only behavior, both targets and the `MomentumBaseline` control. Task 7 covers whole-branch review, defect repair by amendment, final verification and the bounded report.

Names are consistent across tasks: `MIN_TICKERS_PER_DATE`, `BUCKET_SIZE`, `TEST_BOUNDARY`, `KNOWN_MODEL_ORDER`, `DAILY_COLUMNS`, `_require_columns`, `_model_sort_key`, `_daily_row`, `_describe`, `build_daily_ranking`, `select_non_overlapping` and `summarize_ranking` are defined once and referenced identically afterwards. `_daily_row` is introduced in Task 1 and its return block is replaced, not renamed, in Task 2. No placeholder, deferred decision or unspecified behavior remains.
