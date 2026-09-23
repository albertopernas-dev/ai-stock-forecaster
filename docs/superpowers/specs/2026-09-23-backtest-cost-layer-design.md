# Break-Even Cost Backtest Design

## Status and objective

The Step 5H functional design is approved. This document specifies future
implementation and one controlled evaluation. This specification task changes no
production code, tests, README content, dependencies, or datasets and performs
no evaluation.

Step 5G established that the learned models order stocks slightly better than
chance and better than the naive `MomentumBaseline` control: pooled information
coefficients of roughly 0.018 to 0.031 and a tercile spread near 0.23% per
five-session window, halving under the non-overlapping check and turning
negative in 2021 and 2022.

Step 5H asks the one remaining question that ordering evidence cannot answer:
**does that spread survive the cost of trading it?**

Everything else stays fixed. The same 15 stocks, the same ten `FEATURE_COLUMNS`,
the same unchanged models, the same 2016-2023 walk-forward folds, the same
untouched 2024+ TEST boundary. Step 5H introduces no predictor, no model, no
tuning, no dependency, and no persisted artifact.

## The central design decision: measure break-even, do not assume costs

The obvious approach is to assume a cost level and report whether the net result
is positive. That approach is rejected. The assumed number determines the
conclusion, and any assumption flattering enough to produce a positive result
would be indistinguishable from one chosen to produce it.

Step 5H instead computes the **break-even cost**: the per-unit trading cost at
which the net edge reaches exactly zero.

```text
breakeven_cost_bps = 10000 * mean(gross_return) / mean(traded_notional)
```

This is assumption-free. It converts the question into a single number that the
reader compares against whatever cost they actually pay. A non-positive value
means the strategy loses before any cost is applied at all, and is reported as
such rather than special-cased.

## Input contract

Step 5H consumes the same out-of-sample prediction frames that
`evaluate_walk_forward` produces, with the schema fixed by `PREDICTION_COLUMNS`:

```text
date  ticker  validation_year  model  y_true  prediction
```

No new data is required. `y_true` is already the realized five-session forward
return, so a portfolio's gross return over its holding window is the
weighted mean of `y_true` across its positions. Predictions are regenerated in
memory by the unchanged deterministic evaluation, exactly as in Step 5G, and
nothing is persisted.

## Rebalancing schedule

Rebalancing is **weekly and non-overlapping**: every fifth distinct trading date
within each validation year, selected deterministically from the sorted unique
dates at positions `0, 5, 10, ...`. This reuses `select_non_overlapping` from
`ranking.py` rather than reimplementing it.

This schedule matches the five-session forecast horizon exactly, so each
position is held for precisely its own target window and turnover is
unambiguous. Daily overlapping rebalancing is **deliberately excluded**: it
requires tracking five staggered sub-portfolios, it trades strictly more, and it
is therefore strictly worse on cost. A strategy that cannot survive the weekly
schedule cannot survive the daily one, so the simpler and more favorable case is
the one worth measuring.

## Strategies

Two fixed, unoptimized rules. Position sizing is equal-weight in both. There is
no leverage, no risk model, no optimizer, and no parameter to choose.

**`long_short`** — dollar-neutral. On each rebalance date, stocks are sorted by
`(prediction descending, ticker ascending)`. The top `BUCKET_SIZE = 5` receive
weight `+1/5` each and the bottom 5 receive `-1/5` each; every other stock
receives zero. Gross exposure is 2.0. The period gross return is
`mean(top y_true) - mean(bottom y_true)`, which is exactly the Step 5G tercile
spread, now carried through a cost model.

**`long_only`** — measured against the equal-weight universe. The top 5 receive
`+1/5` each. The period gross return is `mean(top y_true) - mean(all y_true)`:
the active return over holding all 15 stocks equally. The benchmark is a
reference, not a traded book, so only the active book contributes turnover.

`BUCKET_SIZE` is imported from `ranking.py` so the two steps cannot drift apart.

## Turnover and cost

Positions are sequenced per model in date order **across the whole 2016-2023
span**, not restarted each year, because a real strategy runs continuously while
only the model refits annually. Traded notional for a rebalance is

```text
traded_notional = sum over tickers of abs(new_weight - previous_weight)
```

The previous weights of the very first rebalance are all zero, so that period
carries the full cost of establishing the book. That establishment cost is real
and is included rather than discounted; it lands in 2016 and is visible in the
annual table.

Cost is charged as `traded_notional * c`, where `c` is the cost per unit of
notional traded. Reporting break-even makes `c` an output rather than an input.

**Two real costs are not modeled**, and both make the break-even figure
optimistic: the borrow cost of shorting in the `long_short` strategy, and market
impact. This is stated wherever results are reported.

## Undefined cases

A rebalance date produces `NaN` gross return and `NaN` traded notional, never
zero, when:

- fewer than `2 * BUCKET_SIZE = 10` tickers are available, so the buckets would
  overlap; or
- predictions do not vary across that date's stocks, which is the structural
  case for `ZeroBaseline` and `MeanBaseline`.

This matches the Step 5G rule exactly. A constant predictor carries no ordering
information, and an arbitrary tie-break must never be presented as a portfolio
decision. `NaN` periods are excluded from every aggregate and from turnover
sequencing: the previous weights carry forward unchanged across an undefined
date rather than being treated as a liquidation.

## Public interface

A new module `src/stock_forecaster/backtesting/simulation.py`, inside the
existing empty `backtesting` package. It is additive. `ranking.py`,
`evaluation.py`, `metrics.py`, `linear.py`, `forest.py`, `dataset.py`,
`baseline.py`, and `engineering.py` are all untouched.

```python
LONG_SHORT = "long_short"
LONG_ONLY = "long_only"
STRATEGIES = (LONG_SHORT, LONG_ONLY)
REBALANCE_STEP = 5

build_period_returns(
    predictions: pd.DataFrame,
    strategy: str = LONG_SHORT,
    step: int = REBALANCE_STEP,
) -> pd.DataFrame
```

Returns one row per model and rebalance date:

```text
model  validation_year  date  n_tickers  gross_return  traded_notional
```

```python
summarize_backtest(periods: pd.DataFrame, by_year: bool = False) -> pd.DataFrame
```

Returns, per model and optionally per validation year:

```text
model  [validation_year]  periods  mean_gross_return  mean_traded_notional
breakeven_cost_bps  share_positive_periods
```

- `periods` counts rebalance dates with a defined gross return;
- `share_positive_periods` is the share of those with a strictly positive gross
  return; exactly zero is not positive;
- `breakeven_cost_bps` is `10000 * mean_gross_return / mean_traded_notional`,
  and is `NaN` when `mean_traded_notional` is zero or no period is defined.

An unknown `strategy` raises `ValueError` before any computation. Model
presentation order reuses `KNOWN_MODEL_ORDER` from `ranking.py`, so tables stay
comparable with Steps 5E through 5G.

## Validation and invariants

`build_period_returns` rejects, before computing anything: a non-DataFrame or
empty input; a frame missing any of the six required columns; duplicate
`(date, ticker, model)` rows; an unknown strategy; a non-positive step; and any
row dated `2024-01-01` or later. The last is the same hard TEST guard as
`ranking.py` and has its own test.

For every function: the input frame is never mutated; no date, ticker, or model
appears in output that was absent from input; exclusions surface as `NaN`
alongside an `n_tickers` count that explains them; and nothing is written to
disk.

## Synthetic TDD coverage

Deterministic frames, no market data:

1. `long_short` gross return equals top-bucket mean minus bottom-bucket mean,
   matching hand-computed arithmetic.
2. `long_only` gross return equals top-bucket mean minus the all-stock mean.
3. Ties resolve deterministically by ascending ticker.
4. The first rebalance has traded notional equal to gross exposure: 2.0 for
   `long_short` and 1.0 for `long_only`.
5. An unchanged book between consecutive rebalances has zero traded notional.
6. A completely reversed book has traded notional equal to twice gross exposure.
7. Constant predictions give `NaN` gross return and `NaN` traded notional.
8. Fewer than `2 * BUCKET_SIZE` tickers give `NaN`.
9. An undefined date carries previous weights forward instead of liquidating.
10. Rebalance dates are positions `0, 5, 10, ...` of each validation year.
11. Turnover sequencing continues across a year boundary rather than restarting.
12. `breakeven_cost_bps` equals `10000 * mean_gross / mean_traded` on a known
    frame.
13. `breakeven_cost_bps` is negative when the mean gross return is negative.
14. `breakeven_cost_bps` is `NaN` when mean traded notional is zero.
15. `share_positive_periods` excludes `NaN` and treats exactly zero as not
    positive.
16. Summary rows follow `KNOWN_MODEL_ORDER`.
17. Annual and pooled summaries are mutually consistent on a known frame.
18. Neither function mutates its input.
19. Empty input, missing columns, duplicates, unknown strategy, non-positive
    step, and rows dated 2024-01-01 or later are all rejected.

Existing suites remain regression coverage and must stay green.

## Official Step 5H evaluation

One controlled run after implementation is complete, verified, reviewed, and
committed. It regenerates both prediction sets in memory, exactly as Step 5G
did, and produces for each target and each strategy a pooled and an annual
summary. Both data files are hashed before and after the run and must be
unchanged, since Step 5H reads and writes nothing.

## Interpretation and permitted conclusions

Permitted: the break-even cost of each model and strategy; how it compares to
plainly stated real-world trading costs; whether the gross edge exists at all
before costs; how turnover behaves; whether the picture is stable across years;
and whether the evidence justifies any further work on this formulation.

Forbidden, without exception: profitability, investability, or expected return;
statistical significance, p-values, or hypothesis tests; causality or economic
significance; any TEST or backtest-on-TEST statement; Sharpe or other
risk-adjusted statistics; compounded or annualized returns; and any composite
score.

Three limitations are stated whenever results are reported:

1. **Unmodeled costs.** Short borrow and market impact are excluded, so the
   break-even figure is optimistic.
2. **Universe breadth.** Fifteen large, mutually correlated stocks provide very
   few independent bets, capping capacity regardless of the result.
3. **The favorable schedule.** Weekly non-overlapping rebalancing is the
   lowest-turnover sensible choice; any more frequent schedule is worse.

## Execution contract

The established project protocol: an isolated worktree at
`.worktrees/backtest-cost-layer` on branch `codex/backtest-cost-layer`,
RED-GREEN TDD for every new behavior, full suite and Ruff green, one single
implementation commit, then the one official evaluation, then a report, then
stop for approval. Nothing is merged or pushed without explicit authorization.

## Explicit exclusions

Step 5H excludes daily or overlapping rebalancing, portfolio optimization,
non-equal position sizing, leverage, risk models, factor neutralization,
stop-losses, Sharpe and other risk-adjusted statistics, compounding,
annualization, TEST evaluation, model selection, tuning, new predictors, new
models, new dependencies, and any persisted dataset, prediction, or model
artifact.

## Design self-review

1. The measured question is cost survival, distinct from ordering ability.
2. Break-even is an output, so no cost assumption can bias the conclusion.
3. Predictions are regenerated deterministically and never persisted.
4. The rebalancing schedule matches the forecast horizon exactly.
5. The excluded daily schedule is strictly worse, so excluding it is safe.
6. Both strategies are fixed and unoptimized, with no parameter to select.
7. `BUCKET_SIZE`, `select_non_overlapping` and `KNOWN_MODEL_ORDER` are reused
   from `ranking.py`, so the two steps cannot drift apart.
8. Constant-prediction baselines yield `NaN`, never a misleading zero.
9. Turnover sequencing across undefined dates and year boundaries is specified.
10. Establishment cost is included rather than quietly discounted.
11. Unmodeled costs are named and their direction of bias is stated.
12. A hard TEST guard is enforced in code and covered by a test.
13. No existing module changes and no new dependency is required.
14. Every architectural decision is resolved; no placeholder remains.
