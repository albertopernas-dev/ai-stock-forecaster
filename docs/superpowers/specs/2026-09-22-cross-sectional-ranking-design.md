# Cross-Sectional Ranking Evaluation Design

## Status and objective

The Step 5G functional design is approved. This document specifies future
implementation and one controlled evaluation. This specification task changes no
production code, tests, README content, dependencies, or datasets and performs
no evaluation.

Steps 5A-5F measured point error: how close a predicted number is to the
realized return. Step 5G measures a different and, for this project's stated
purpose, more decisive property: **within a single trading date, does a model
order the 15 stocks correctly from best to worst?**

The distinction matters because the two can disagree. A model can be poor at
predicting magnitudes and still rank usefully, and a model with low MAE can rank
no better than chance. Pooled Pearson correlation, the statistic reported in
Steps 5E and 5F, mixes variation across time with variation across stocks and
answers neither question cleanly. No cross-sectional statistic has been computed
in this project so far.

Everything else stays fixed:

- the same 15 modeled stocks and the same `configs/market.yaml` universe;
- the same ten `FEATURE_COLUMNS`;
- the same `LinearRegressionForecaster` and `RandomForestForecaster` with
  unchanged hyperparameters;
- the same 2016-2023 expanding walk-forward folds;
- the same untouched 2024+ TEST boundary.

Step 5G adds an evaluation lens. It introduces no predictor, no model, no
tuning, no dependency, and no persisted artifact.

## Input contract

Step 5G consumes the out-of-sample prediction frames that
`evaluate_walk_forward` already produces, whose schema is fixed by
`PREDICTION_COLUMNS`:

```text
date  ticker  validation_year  model  y_true  prediction
```

Predictions are **regenerated in memory** on each run by calling the existing
evaluation with its existing deterministic configuration. `RandomForestForecaster`
carries `random_state=42`, and Step 5E's old/new gate already demonstrated
reproducibility to a maximum deviation of `1.3877787807814457e-17`. Regeneration
is therefore reproduction, not a new experiment, and it keeps the project's rule
that no generated prediction is persisted.

The new module accepts one concatenated `pd.DataFrame` obeying the contract
above. Assembling that frame from a `ModelFoldResult` sequence belongs to the
calling orchestration, not to the ranking module.

## Evaluated targets and model sets

Both already-evaluated targets are ranked and compared:

```text
absolute target: future_return_5d          (Step 5E defaults)
excess target:   future_excess_return_5d   (Step 5F selection)
```

The comparison is the point. The excess target was constructed precisely to
remove the common market movement that dominates all 15 stocks simultaneously,
which is the component that carries no ordering information. If any formulation
ranks better, it should be this one.

**`MomentumBaseline` is added to the excess-target model set.** This is a
deliberate change from the official Step 5F set and requires justification.
Ranking needs a reference that actually orders stocks. `ZeroBaseline` and
`MeanBaseline` predict one identical value for every stock on a given date, so
they carry zero ordering information by construction and their rank statistics
are undefined rather than poor. `MomentumBaseline` varies across stocks and is
therefore the only legitimate naive ordering reference available. It enters as a
**control**, not as a candidate hypothesis: its predictions are a deterministic
restatement of an existing feature, it is not fitted, and it is not eligible to
be selected as a model. Only its ordering is used; its predicted values are
past absolute returns and are not comparable in level to an excess target, which
is irrelevant here because every Step 5G measurement consumes ranks or realized
values, never predicted magnitudes. The two constant baselines are still
evaluated so that their undefined statistics are shown explicitly rather than
silently omitted.

The evaluated sets are therefore:

```text
absolute: MeanBaseline  MomentumBaseline  LinearRegression  RandomForest
excess:   ZeroBaseline  MeanBaseline  MomentumBaseline  LinearRegression  RandomForest
```

No model is tuned, refitted with alternative settings, or re-specified.

## Measurement one: daily cross-sectional information coefficient

For each `(model, date)` pair, the information coefficient is the Spearman rank
correlation between `prediction` and `y_true` across the tickers observed on
that date.

Spearman rather than Pearson because the quantity of interest is order, not
magnitude, and rank correlation is unaffected by the outliers that dominate
return distributions. `pandas.Series.corr(method="spearman")` provides this, so
no dependency is added.

The coefficient is `NaN`, not zero, when it is undefined:

- fewer than `MIN_TICKERS_PER_DATE = 5` usable pairs on that date;
- zero variance in `prediction` across that date's tickers, which is the
  structural case for `ZeroBaseline` and `MeanBaseline`;
- zero variance in `y_true` across that date's tickers.

`NaN` means "this date carries no ordering evidence for this model" and is
excluded from every aggregate rather than counted as a failure. A model whose
coefficient is always undefined must be visibly distinguishable from a model
whose coefficient is defined and averages zero.

## Measurement two: tercile spread

The information coefficient states whether ordering exists. The spread states
how much of the realized return separates the ends of that ordering, which is
the closest quantity to economic meaning that Step 5G is permitted to compute.

For each `(model, date)` pair, tickers are sorted by `prediction` descending;
the top `BUCKET_SIZE = 5` and bottom `BUCKET_SIZE = 5` are selected; the mean
realized `y_true` of each bucket is taken; and the spread is
`top_mean - bottom_mean`. With the 15-stock universe these buckets are terciles.

Rules:

- the date requires at least `2 * BUCKET_SIZE = 10` usable tickers so the two
  buckets cannot overlap; otherwise the spread is `NaN`;
- sorting is `(prediction descending, ticker ascending)` so bucket membership is
  deterministic when predictions tie;
- a date whose predictions have zero variance yields `NaN`, matching the
  information-coefficient rule, because an arbitrary tie-break must never be
  presented as an ordering result.

The spread is a raw difference of realized five-session returns. It is not a
strategy return: it applies no transaction cost, no position sizing, no
rebalancing schedule, and no compounding, and it must never be reported as one.

## Measurement three: non-overlapping robustness subsample

Observations are daily but the forecast horizon is five sessions, so the target
windows of consecutive dates overlap by roughly eighty percent. Consecutive
daily statistics are therefore heavily dependent, which **inflates the apparent
consistency** of any result: a share-of-positive-days figure computed on
overlapping windows looks far more stable than the underlying evidence
justifies.

Every summary is therefore produced twice:

- the **primary** pass over all dates, which uses all available evidence;
- a **robustness** pass over a non-overlapping subsample, taking every fifth
  distinct trading date within each validation year, selected deterministically
  from the sorted unique dates at positions `0, 5, 10, ...`.

Both are reported side by side. The robustness pass is never presented instead
of the primary pass, and the conservative reading is the one they agree on.

## Public interface

A new module `src/stock_forecaster/models/ranking.py`. It is additive: no
existing module changes, and `evaluation.py`, `metrics.py`, `linear.py`,
`forest.py`, `dataset.py`, and `engineering.py` are untouched.

```python
build_daily_ranking(predictions: pd.DataFrame) -> pd.DataFrame
```

Returns one row per `(model, date)`:

```text
model  validation_year  date  n_tickers  ic  top_mean  bottom_mean  spread
```

```python
select_non_overlapping(daily: pd.DataFrame, step: int = 5) -> pd.DataFrame
```

Returns the deterministic subsample described above, preserving the input
schema.

```python
summarize_ranking(daily: pd.DataFrame, by_year: bool = False) -> pd.DataFrame
```

Returns, per model and optionally per validation year:

```text
model  [validation_year]  ic_days  ic_mean  ic_std  ic_share_positive
ic_stability  spread_days  spread_mean  spread_share_positive
```

- `ic_days` counts dates with a defined information coefficient and `spread_days`
  counts dates with a defined spread. The two counts can differ, because a date
  with between `MIN_TICKERS_PER_DATE` and `2 * BUCKET_SIZE` tickers yields a
  defined coefficient but no spread. Each statistic is aggregated over its own
  defined dates, and both counts are always reported so the denominators are
  never implicit;
- `ic_share_positive` and `spread_share_positive` are shares of their own
  defined dates with a strictly positive value; ties at exactly zero are not
  positive;
- `ic_stability` is `ic_mean / ic_std`, reported as a **descriptive dispersion
  ratio only**. It is `NaN` when `ic_std` is zero or fewer than two dates are
  defined. It is not a test statistic, and no p-value, confidence interval, or
  significance claim is derived from it anywhere in Step 5G.

Model presentation order reuses the existing known-model order already defined
in `evaluation.py`, so tables stay comparable with Steps 5E and 5F. Only models
present in the input are emitted.

## Validation and invariants

`build_daily_ranking` rejects, before computing anything:

- a non-DataFrame or empty input;
- a frame missing any of the six required columns;
- duplicate `(date, ticker, model)` rows;
- any row whose `date` is `2024-01-01` or later.

The last rule is a hard TEST guard. Step 5G cannot be pointed at TEST data even
by accident, and the guard is covered by an explicit test.

Invariants that hold for every function:

- the input frame is never mutated;
- no date, ticker, or model appears in output that was absent from input;
- no row is dropped silently: exclusions surface as `NaN` with a `n_tickers`
  count that explains them;
- nothing is written to disk; `data/raw` and `data/processed` are not opened.

## Synthetic TDD coverage

Feature-free deterministic frames, no market data:

1. Perfectly ordered predictions give an information coefficient of exactly 1.
2. Perfectly inverted predictions give exactly -1.
3. A known mixed case matches an independently computed Spearman value.
4. Constant predictions give `NaN`, not zero, for both measurements.
5. Constant realized values give `NaN`.
6. A date below `MIN_TICKERS_PER_DATE` gives `NaN` and records `n_tickers`.
7. Bucket means and spread match exact hand-computed arithmetic.
8. A date with fewer than `2 * BUCKET_SIZE` tickers gives a `NaN` spread.
9. Tied predictions resolve deterministically by ascending ticker.
10. `select_non_overlapping` picks positions `0, 5, 10, ...` within each year.
11. Summary means, standard deviations, shares, and counts exclude `NaN` dates.
    A date with a defined coefficient but an undefined spread is counted in
    `ic_days` and not in `spread_days`.
12. `ic_stability` is `NaN` for zero dispersion and for a single defined date.
13. Shares treat exactly zero as not positive.
14. Per-year and pooled summaries are mutually consistent on a known frame.
15. Model presentation order is deterministic and matches `evaluation.py`.
16. Input frames are not mutated by any function.
17. Missing columns, duplicates, and empty input are rejected clearly.
18. A row dated 2024-01-01 or later is rejected.

Existing feature, dataset, walk-forward, baseline, evaluation, metric, linear,
and forest tests remain regression coverage and must stay green.

## Official Step 5G evaluation

One controlled run, after implementation is complete, verified, reviewed, and
committed. It regenerates both prediction sets in memory, builds the daily
ranking frames, and produces for each target:

```text
pooled ranking summary        (primary and non-overlapping)
annual ranking summary        (primary and non-overlapping)
```

Data integrity is proven around the run: `data/raw/prices.parquet` and
`data/processed/features.parquet` are hashed before and after and must be
unchanged, since Step 5G reads and writes nothing.

## Interpretation and permitted conclusions

Permitted:

- whether cross-sectional ordering ability exists at all;
- its magnitude and its stability across years;
- whether the excess target orders better than the absolute target;
- whether the learned models order better than the `MomentumBaseline` control;
- whether the primary and non-overlapping passes agree;
- whether the evidence justifies continuing toward portfolio construction.

Forbidden, without exception:

- profitability, investability, or expected return;
- statistical significance, p-values, confidence intervals, or hypothesis tests;
- causality or economic significance;
- any TEST or backtest statement;
- describing the tercile spread as a strategy return;
- any composite or combined score.

Two limitations are stated whenever results are reported:

1. **Overlapping windows.** The primary pass overstates consistency. Where the
   two passes disagree, the non-overlapping one is the conservative reading.
2. **Universe breadth.** Fifteen large, mutually correlated stocks provide very
   few independent bets. Ordering ability found here would have a low capacity
   ceiling regardless of its magnitude, and the number of stocks, not the model,
   is the binding constraint.

## Execution contract

Implementation follows the established project protocol: an isolated worktree at
`.worktrees/ranking-evaluation` on branch `codex/cross-sectional-ranking`,
RED-GREEN TDD for every new behavior, full suite and Ruff green, one single
implementation commit, then the one official evaluation, then a report, then
stop for approval. Nothing is pushed and nothing is merged without explicit
authorization.

## Explicit exclusions

Step 5G excludes transaction costs, position sizing, turnover, rebalancing
schedules, compounding, Sharpe or other risk-adjusted return statistics,
portfolio construction, optimization, backtesting, TEST evaluation, model
selection, hyperparameter tuning, new predictors, new models, new dependencies,
and any persisted dataset, prediction, or model artifact.

## Design self-review

1. The measured question is cross-sectional ordering, distinct from point error.
2. Predictions are regenerated deterministically, never persisted.
3. Both targets are evaluated and compared on identical machinery.
4. `MomentumBaseline` enters as a control, with its role stated explicitly.
5. Constant-prediction baselines yield `NaN`, never a misleading zero.
6. Undefined cases are enumerated and each has a test.
7. Tie-breaking is deterministic and specified.
8. Overlap is acknowledged and answered with a second reported pass.
9. `ic_stability` is explicitly descriptive, never inferential.
10. A hard TEST guard is enforced in code and covered by a test.
11. No existing module changes; the new module is purely additive.
12. No new dependency is required.
13. Interpretation limits are explicit, including the breadth ceiling.
14. Every architectural decision is resolved; no placeholder remains.
