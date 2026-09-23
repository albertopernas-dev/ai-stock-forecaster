# Pre-Registered Hold-Out Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, default-off opt-in that admits 2024+ rows, then run the single pre-registered hold-out confirmation exactly once.

**Architecture:** `ranking.py` and `backtesting/simulation.py` each gain an `allow_holdout: bool = False` keyword. With the default, behavior is unchanged and every existing test passes untouched. Only an explicit `allow_holdout=True` at a call site admits the hold-out, keeping the act visible and greppable. No other module changes.

**Tech Stack:** Python 3.12.14, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-23-holdout-confirmation-design.md` — **this is a pre-registration, committed as `2beeea5` before any hold-out value was read.**

## Global Constraints

- **The pre-registration is binding.** Target `future_return_5d`, strategy `long_only`, models `LinearRegression` and `RandomForest` with `MomentumBaseline` as control, primary metric pooled break-even cost in basis points, compared against the published development figures `+12.39 / +11.62 / −1.59`. None of these may change.
- **One shot.** The evaluation runs exactly once. Afterwards no model, strategy, bucket size, schedule or threshold may be adjusted, and no second hold-out run is permitted under any justification. A disappointing result is the answer.
- **The guard is widened, never removed.** `allow_holdout` defaults to `False` permanently. Every existing test must pass unmodified.
- **No tuning, no new predictor, no new model, no new dependency, nothing persisted.**
- **Protected and untouched:** `linear.py`, `forest.py`, `metrics.py`, `evaluation.py`, `dataset.py`, `baseline.py`, `engineering.py`, `pyproject.toml`, `configs/`.
- **Anchors:** raw `49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5`, processed `C717BFEE645C312CFD0086B6A1514D15CC47E0E4EED4D6557218B3EDC272DA57`.
- **One implementation commit.** Tasks 1-2 leave work uncommitted; Task 3 commits.
- **Environment:** `C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe` with `$env:PYTHONPATH` set to the worktree `src`.
- **No claim** of profitability, investability, expected return, statistical significance, causality, economic significance, Sharpe, compounding, or annualization.

## File Map

- Modify: `src/stock_forecaster/models/ranking.py` — add `allow_holdout` to `build_daily_ranking` and its validator.
- Modify: `src/stock_forecaster/backtesting/simulation.py` — add `allow_holdout` to `build_period_returns` and its validator.
- Modify: `tests/test_ranking.py`, `tests/test_backtest.py` — prove the default still rejects and the opt-in admits.
- Modify: `README.md` — the opt-in, the pre-registration, and the one-shot rule.

---

### Task 1: Hold-out opt-in for the ranking module

**Files:**
- Modify: `src/stock_forecaster/models/ranking.py`
- Modify: `tests/test_ranking.py`

**Interfaces:**
- Produces: `build_daily_ranking(predictions, allow_holdout: bool = False)`. `_validate_predictions` gains the same keyword. No other signature changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ranking.py`:

```python
def _holdout_frame():
    return _predictions(
        "2024-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
        year=2024,
    )


def test_holdout_rows_are_admitted_only_by_the_explicit_opt_in():
    daily = build_daily_ranking(_holdout_frame(), allow_holdout=True)

    assert len(daily) == 1
    assert daily.loc[0, "validation_year"] == 2024
    assert daily.loc[0, "ic"] == pytest.approx(1.0)


def test_the_opt_in_defaults_to_refusing_the_holdout():
    import inspect

    from stock_forecaster.models.ranking import build_daily_ranking as builder

    assert inspect.signature(builder).parameters["allow_holdout"].default is False


def test_explicitly_disallowing_the_holdout_still_rejects_it():
    with pytest.raises(ValueError, match="2024"):
        build_daily_ranking(_holdout_frame(), allow_holdout=False)


def test_development_rows_are_unaffected_by_the_opt_in():
    frame = _predictions(
        "2016-03-01",
        "LinearRegression",
        ["A", "B", "C", "D", "E"],
        [0.05, 0.04, 0.03, 0.02, 0.01],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    )

    pd.testing.assert_frame_equal(
        build_daily_ranking(frame),
        build_daily_ranking(frame, allow_holdout=True),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
$env:PYTHONPATH = 'C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.worktrees\holdout-confirmation\src'
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -q
```

Expected: the opt-in tests fail with `TypeError: build_daily_ranking() got an unexpected keyword argument 'allow_holdout'`. The existing rejection test must still pass.

- [ ] **Step 3: Write the minimal implementation**

In `src/stock_forecaster/models/ranking.py`, change the validator signature and its guard:

```python
def _validate_predictions(
    predictions: pd.DataFrame,
    allow_holdout: bool = False,
) -> pd.DataFrame:
```

```python
    if not allow_holdout and ordered["date"].ge(TEST_BOUNDARY).any():
        raise ValueError(
            "predictions must not contain rows dated 2024-01-01 or later; "
            "pass allow_holdout=True only for the one pre-registered "
            "confirmation"
        )
```

and thread it through the public function:

```python
def build_daily_ranking(
    predictions: pd.DataFrame,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Return one cross-sectional ranking row per model and trading date."""
    ordered = _validate_predictions(predictions, allow_holdout=allow_holdout)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_ranking.py -q
```

Expected: every test passes, including the pre-existing ones unmodified.

- [ ] **Step 5: Do not commit**

---

### Task 2: Hold-out opt-in for the backtest module

**Files:**
- Modify: `src/stock_forecaster/backtesting/simulation.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Produces: `build_period_returns(predictions, strategy=LONG_SHORT, step=REBALANCE_STEP, allow_holdout: bool = False)`. `_validate` gains the same keyword.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
def _holdout_ten():
    return _ten("2024-03-01", [0.01] * 10, DESCENDING, year=2024)


def test_holdout_rows_are_admitted_only_by_the_explicit_opt_in():
    periods = build_period_returns(_holdout_ten(), allow_holdout=True)

    assert len(periods) == 1
    assert periods.loc[0, "validation_year"] == 2024
    assert periods.loc[0, "traded_notional"] == pytest.approx(2.0)


def test_the_backtest_opt_in_defaults_to_refusing_the_holdout():
    import inspect

    assert (
        inspect.signature(build_period_returns).parameters["allow_holdout"].default
        is False
    )


def test_explicitly_disallowing_the_holdout_still_rejects_it():
    with pytest.raises(ValueError, match="2024"):
        build_period_returns(_holdout_ten(), allow_holdout=False)


def test_development_periods_are_unaffected_by_the_opt_in():
    frame = _ten("2016-01-04", [0.01] * 10, DESCENDING)

    pd.testing.assert_frame_equal(
        build_period_returns(frame),
        build_period_returns(frame, allow_holdout=True),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: the opt-in tests fail with `TypeError: build_period_returns() got an unexpected keyword argument 'allow_holdout'`.

- [ ] **Step 3: Write the minimal implementation**

In `src/stock_forecaster/backtesting/simulation.py`:

```python
def _validate(
    predictions: pd.DataFrame,
    strategy: str,
    step: int,
    allow_holdout: bool = False,
) -> pd.DataFrame:
```

```python
    if not allow_holdout and ordered["date"].ge(TEST_BOUNDARY).any():
        raise ValueError(
            "predictions must not contain rows dated 2024-01-01 or later; "
            "pass allow_holdout=True only for the one pre-registered "
            "confirmation"
        )
```

```python
def build_period_returns(
    predictions: pd.DataFrame,
    strategy: str = LONG_SHORT,
    step: int = REBALANCE_STEP,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Return one gross return and traded notional per model and rebalance."""
    ordered = _validate(predictions, strategy, step, allow_holdout=allow_holdout)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest tests/test_backtest.py -q
```

Expected: every test passes.

- [ ] **Step 5: Do not commit**

---

### Task 3: Document, verify, and create the single implementation commit

**Files:**
- Modify: `README.md`
- Verify all files from Tasks 1-2

- [ ] **Step 1: Update README**

Append to the model-evaluation narrative:

```markdown
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
```

- [ ] **Step 2: Verify signatures and defaults**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -c "import inspect; from stock_forecaster.models.ranking import build_daily_ranking; from stock_forecaster.backtesting.simulation import build_period_returns; assert inspect.signature(build_daily_ranking).parameters['allow_holdout'].default is False; assert inspect.signature(build_period_returns).parameters['allow_holdout'].default is False; print('Step 5I opt-in defaults: OK')"
```

- [ ] **Step 3: Run the full verification**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
```

Expected: the complete suite passes with the 324 pre-existing tests plus the eight new opt-in tests, zero failures; Ruff clean. The evaluation is prohibited until this gate is green.

- [ ] **Step 4: Review protected scope**

```powershell
git diff -- src/stock_forecaster/models/linear.py src/stock_forecaster/models/forest.py src/stock_forecaster/models/metrics.py src/stock_forecaster/models/evaluation.py src/stock_forecaster/models/dataset.py src/stock_forecaster/models/baseline.py src/stock_forecaster/features/engineering.py pyproject.toml configs
git status --short -- data models
git status --short
```

Expected: protected and data commands print nothing. Only the two modules, the two test files and README appear.

- [ ] **Step 5: Create the single implementation commit**

```powershell
git add src/stock_forecaster/models/ranking.py src/stock_forecaster/backtesting/simulation.py tests/test_ranking.py tests/test_backtest.py README.md
git diff --cached --name-only
git commit -m "feat: add explicit default-off hold-out opt-in"
```

---

### Task 4: Run the one pre-registered hold-out confirmation

**Files:**
- Read: `data/processed/features.parquet`
- Create/commit: no file. The script lives in the session scratchpad.

- [ ] **Step 1: Record hashes before the run**

Expected: both anchors from the Global Constraints. Stop and report if either differs.

- [ ] **Step 2: Write the evaluation script**

```python
"""The one pre-registered Step 5I hold-out confirmation. Read-only, run once."""

import numpy as np
import pandas as pd

from stock_forecaster.backtesting.simulation import (
    LONG_ONLY,
    LONG_SHORT,
    build_period_returns,
    summarize_backtest,
)
from stock_forecaster.models.dataset import prepare_supervised_data, walk_forward_splits
from stock_forecaster.models.evaluation import evaluate_walk_forward
from stock_forecaster.models.ranking import build_daily_ranking, summarize_ranking

PROCESSED = (
    r"C:\Users\alber\04_Desarrollo\ai-stock-forecaster"
    r"\data\processed\features.parquet"
)
STOCKS = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
          "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG")
DEVELOPMENT_YEARS = tuple(range(2016, 2024))
HOLDOUT_YEARS = (2024, 2025, 2026)
ALL_YEARS = DEVELOPMENT_YEARS + HOLDOUT_YEARS
MODELS = ("MomentumBaseline", "LinearRegression", "RandomForest")
CUTOFF = pd.Timestamp("2024-01-01")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    features = pd.read_parquet(PROCESSED)
    require(set(features["ticker"].unique()) == set(STOCKS) | {"SPY"},
            "processed universe mismatch")
    without_spy = features.loc[features["ticker"].ne("SPY")].copy()

    prepared = prepare_supervised_data(without_spy)
    folds = walk_forward_splits(prepared, ALL_YEARS)
    require(len(folds) == len(ALL_YEARS), "unexpected fold count")
    results = evaluate_walk_forward(folds, model_names=MODELS)
    predictions = pd.concat(
        [result.predictions for result in results], ignore_index=True
    )

    # Gate: development years must reproduce exactly before any hold-out
    # number is read.
    development = predictions.loc[predictions["validation_year"].lt(2024)]
    reference_folds = walk_forward_splits(
        prepared.loc[
            prepared["date"].lt(CUTOFF) & prepared["target_end_date"].lt(CUTOFF)
        ].reset_index(drop=True),
        DEVELOPMENT_YEARS,
    )
    reference = pd.concat(
        [
            result.predictions
            for result in evaluate_walk_forward(reference_folds, model_names=MODELS)
        ],
        ignore_index=True,
    )
    require(len(development) == len(reference), "development row count changed")
    keys = ["date", "ticker", "model"]
    left = development.sort_values(keys).reset_index(drop=True)
    right = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys], right[keys], check_exact=True)
    np.testing.assert_allclose(
        left["prediction"].to_numpy(dtype=float),
        right["prediction"].to_numpy(dtype=float),
        rtol=1e-12, atol=1e-15,
    )
    print("development reproduction gate: PASSED", flush=True)

    holdout = predictions.loc[predictions["validation_year"].ge(2024)]
    print(f"\nhold-out prediction rows: {len(holdout)}")
    print(holdout.groupby(["validation_year", "model"]).size().to_string())

    print("\n\n########## PRIMARY: absolute target, long_only ##########")
    periods = build_period_returns(
        holdout, strategy=LONG_ONLY, allow_holdout=True
    )
    print("\n--- pooled hold-out ---")
    print(summarize_backtest(periods).to_string(index=False))
    print("\n--- annual hold-out (2026 is partial) ---")
    print(summarize_backtest(periods, by_year=True).to_string(index=False))

    print("\n\n########## SECONDARY: long_short ##########")
    short_periods = build_period_returns(
        holdout, strategy=LONG_SHORT, allow_holdout=True
    )
    print(summarize_backtest(short_periods).to_string(index=False))
    print(summarize_backtest(short_periods, by_year=True).to_string(index=False))

    print("\n\n########## SECONDARY: ranking ##########")
    daily = build_daily_ranking(holdout, allow_holdout=True)
    print(summarize_ranking(daily).to_string(index=False))
    print(summarize_ranking(daily, by_year=True).to_string(index=False))

    print("\nOne pre-registered run. No tuning, no reselection, no repetition.",
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it once**

The development reproduction gate must print `PASSED` before any hold-out table is read. If it fails, stop and report without reading further.

- [ ] **Step 4: Record hashes after the run**

Expected: both hashes identical and a clean tracked tree.

---

### Task 5: Whole-branch review, final verification and the closing report

- [ ] **Step 1: Review every invariant**

Confirm: the opt-in defaults to `False` in both modules; every pre-existing test passes unmodified; no other module changed; no tuning; no persistence; the pre-registration was committed before the run. Classify findings as Critical, Important or Minor.

- [ ] **Step 2: Run fresh final verification**

```powershell
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m pytest -v
C:\Users\alber\04_Desarrollo\ai-stock-forecaster\.venv\Scripts\python.exe -m ruff check .
git status --short
git diff --exit-code
git diff --cached --exit-code
Get-FileHash -Algorithm SHA256 data\raw\prices.parquet
Get-FileHash -Algorithm SHA256 data\processed\features.parquet
```

- [ ] **Step 3: Report against the pre-registration and stop**

Report the primary hold-out break-even for both learned models and the control, side by side with the published development figures; state plainly whether the pre-registered prediction survived, did not survive, or is inconclusive, using the criteria fixed in the specification and not others; give the annual breakdown flagging the partial 2026; report the secondary results labelled as secondary; restate the four limitations; and confirm the one-shot rule now binds. Do not propose refitting, reselection or a second run.

## Test Coverage Map

- `tests/test_ranking.py` and `tests/test_backtest.py`: the opt-in admits hold-out rows; the default and an explicit `False` both reject them; development results are byte-identical with and without the opt-in; all pre-existing coverage unchanged.

## Controlled Execution and Failure Boundaries

1. No evaluation before the full suite, Ruff, scope review and the single implementation commit are green.
2. No hold-out table may be read before the development reproduction gate prints `PASSED`.
3. Exactly one evaluation run, ever.
4. No refitting, reselection, tuning or repetition after the result is seen.
5. Any hash mismatch or protected-module change stops the phase with evidence intact.

## Plan Self-Review

Every specification section maps to a task. Task 1 and Task 2 add the default-off opt-in to the two modules that enforce the guard, each with tests proving the default still refuses. Task 3 documents, verifies and commits. Task 4 runs the single pre-registered evaluation behind a reproduction gate. Task 5 reviews, verifies and reports strictly against the pre-registered criteria. Names are consistent: `allow_holdout` is the only new identifier and it is spelled identically in both modules, both validators and all tests. No placeholder remains.
