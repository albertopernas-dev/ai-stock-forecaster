# Core Feature Engineering Implementation Plan

**Goal:** Implement `build_features(prices: pd.DataFrame) -> pd.DataFrame`
using only synthetic tests; do not process or persist real data.

**Architecture:** Copy and validate the four required columns, sort by ticker
and date, and compute shifts and rolling windows independently within each
ticker. Return identifiers, ten predictors, and the five-observation future
return target, sorted by date and ticker with a reset index.

**Tech stack:** Python >=3.12, existing pandas, pytest, and Ruff. No new dependencies.

**Specification:** User-approved Step 4A specification and design in this task.

## Files

- Create `src/stock_forecaster/features/engineering.py`: pure computation and
  public `FEATURE_COLUMNS`, `TARGET_COLUMN`, and `build_features`.
- Create `tests/test_features.py`: deterministic input, formulas, validation,
  ticker isolation, missing-value behavior, and anti-leakage tests.
- Modify `README.md`: describe the feature/target layer and remaining boundaries.

## Execution checklist

- [ ] Write synthetic tests before production code. Missing input columns and
  duplicate date/ticker pairs must raise clear `ValueError`s.
- [ ] Observe red with `python -m pytest tests/test_features.py -v`.
- [ ] Implement returns as `price / price.shift(n) - 1` for n=1,5,20;
  volatility as sample rolling std of daily returns for n=5,20;
  SMA distances as `price / rolling_mean(n) - 1` for n=10,20,50;
  volume change as `volume / volume.shift(1) - 1` and volume ratio as
  `volume / rolling_mean(20) - 1`, masking zero denominators to NaN.
  All rolling windows use `min_periods=window`.
- [ ] Compute only the target with a future shift:
  `price.shift(-5) / price - 1`. Preserve all warm-up and final-five target NaNs.
- [ ] Verify literal formula expectations, unchanged input, exact output
  columns/order, restarted ticker windows, and no volume infinities.
- [ ] Modify t+5 in a 65-observation synthetic series: all predictors at t
  must stay unchanged while the target changes. Compare two tickers with
  different price/volume scales and independent listing dates.
- [ ] Update README without introducing persistence or training behavior.
- [ ] Run feature tests, full pytest, Ruff, and a clean import check.
- [ ] Verify the raw file hash is unchanged and processed output is absent.
- [ ] Commit only the plan, module, tests, and README as
  `feat: add core stock features and forecast target`; do not push.

## Scope boundaries

No filesystem I/O in `build_features`, no raw OHLC requirement, no exposed raw
levels or SMA levels, no filling/dropping NaNs, no real-data processing,
no persistence, no extra indicators, and no model training.
