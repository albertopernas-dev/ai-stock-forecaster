# Expanding-Window Walk-Forward Validation Design

## Status, objective, and motivation

The Step 5E functional design is approved. This document specifies future
implementation; this specification task creates no implementation, tests,
implementation plan, dependency changes, or evaluation results.

Step 5E replaces reliance on the fixed 2022-2023 validation period with annual
expanding-window walk-forward evaluation over 2016-2023. Its purpose is to
determine whether current model behavior is stable across multiple historical
regimes before adding model families or tuning. It adds temporal validation
infrastructure, fold-level out-of-sample (OOS) predictions, pooled OOS metrics,
and annual consistency diagnostics. It adds no forecasting model.

MeanBaseline, MomentumBaseline, LinearRegression, and RandomForest have all
informed decisions on the same fixed validation period. Repeatedly selecting
future work from that period risks overfitting the development process itself
to 2022-2023. Step 5E improves the evaluation methodology before further model
experimentation.

## Test boundary and modeling universe

TEST remains conceptually 2024-01-01 onward and stays closed. Before fold
construction, development supervised data must satisfy both:

```text
date < 2024-01-01
target_end_date < 2024-01-01
```

This defense prevents a late-2023 label from depending on a 2024 price. Step 5E
must not access or report TEST features, targets, metadata, counts, date ranges,
predictions, metrics, diagnostics, or model-selection evidence.

The modeling universe is exactly:

```text
AAPL  MSFT  GOOGL  AMZN  META  NVDA  JPM  V
JNJ   UNH   XOM    CVX   COST  WMT   PG
```

SPY remains persisted and configured as benchmark data. Orchestration excludes
SPY in memory before supervised preparation; it does not remove SPY from raw or
processed data, change market configuration, or add a hard-coded SPY rule to
generic dataset preparation.

## Annual expanding-window folds

Validation years are exactly the integer sequence 2016, 2017, 2018, 2019,
2020, 2021, 2022, and 2023. For validation year `Y`, define:

```text
validation_start = Y-01-01
next_year_start  = (Y + 1)-01-01
```

The exact row rules are:

```text
TRAIN candidate:             date < validation_start
Retained TRAIN:              date < validation_start
                             and target_end_date < validation_start

VALIDATION candidate:        validation_start <= date < next_year_start
Retained VALIDATION:         validation_start <= date < next_year_start
                             and target_end_date < next_year_start
```

Purging uses each ticker's actual `target_end_date`, never a calendar-day
approximation. In particular, a December 2023 observation whose five-session
target ends in 2024 is absent from the 2023 validation fold.

TRAIN expands chronologically: the 2016 fold trains through 2015, the 2017 fold
adds eligible 2016 history, and so on until the 2023 fold trains through 2022.
The real project's later TRAIN partitions should therefore have more rows than
their predecessors. This is not a rolling fixed-length window.

## Dataset architecture and contracts

Future work extends `src/stock_forecaster/models/dataset.py` with:

```python
@dataclass(frozen=True)
class WalkForwardFold:
    validation_year: int
    train: DatasetPartition
    validation: DatasetPartition

walk_forward_splits(
    supervised_data: pd.DataFrame,
    validation_years: Sequence[int],
) -> tuple[WalkForwardFold, ...]
```

The signature above is the interface design; its body is deferred to the
implementation phase. `DatasetPartition` remains the sole X/y/metadata
representation. `dataset.py` owns temporal fold rules, target-horizon purging,
and partition construction; it contains no model execution.

`walk_forward_splits` accepts already prepared supervised data and requires
`date`, `ticker`, `target_end_date`, every `FEATURE_COLUMNS` entry, and
`TARGET_COLUMN`. It must not mutate its input. `validation_years` must be
non-empty, contain integer years (not booleans), be unique, and be strictly
increasing. Invalid input raises clearly; the function never sorts,
deduplicates, coerces, or otherwise repairs the sequence.

Every returned fold must:

- have non-empty TRAIN and VALIDATION partitions;
- preserve chronological order and exact X/y/metadata row alignment;
- expose exactly `FEATURE_COLUMNS` in X and `TARGET_COLUMN` in y;
- expose metadata containing exactly `date` and `ticker`;
- contain no duplicate `date`+`ticker` observations; and
- appear in the requested validation-year order.

Step 5E real orchestration loads processed features, excludes SPY in memory,
prepares supervised data, applies both development-horizon filters, and only
then calls `walk_forward_splits`. The restricted frame remains in memory.

## Evaluation architecture

Future work creates `src/stock_forecaster/models/evaluation.py`. It owns model
execution, prediction records, fold metrics, pooled metrics, and consistency
summaries. It contains no temporal split or purge logic.

The result contract is:

```python
@dataclass(frozen=True)
class ModelFoldResult:
    model_name: str
    validation_year: int
    predictions: pd.DataFrame
    metrics: RegressionMetrics
```

`predictions` remains in memory and contains exactly these ordered columns:

```text
date  ticker  validation_year  model  y_true  prediction
```

Its rows correspond exactly, in the same order, to the fold's VALIDATION
partition. `date` and `ticker` equal VALIDATION metadata; `validation_year` is
the fold year on every row; `model` is the result's canonical model name;
`y_true` aligns exactly with VALIDATION y; and `prediction` aligns exactly with
the model output. Construction must not reset, sort, or realign data in a way
that breaks those associations. Across all OOS results, each
`date`+`ticker`+`model` combination is unique. No OOS prediction artifact is
persisted.

Canonical presentation names are identical everywhere:

```text
MeanBaseline
MomentumBaseline
LinearRegression
RandomForest
```

The orchestration interface is:

```python
evaluate_walk_forward(
    folds: Sequence[WalkForwardFold],
) -> tuple[ModelFoldResult, ...]
```

It explicitly evaluates only `MeanBaseline`, `MomentumBaseline`,
`LinearRegressionForecaster`, and `RandomForestForecaster`. A registry, generic
estimator protocol, plugin mechanism, dependency-injection framework, and
other speculative abstractions are unnecessary.

For every fold, create fresh MeanBaseline, LinearRegressionForecaster, and
RandomForestForecaster instances. MomentumBaseline is stateless, but its
prediction is generated independently for that fold. Never reuse fitted model
objects between years.

Per fold, fitting and prediction are exactly:

- MeanBaseline fits only `fold.train.y` and predicts the VALIDATION index.
- MomentumBaseline predicts only from `fold.validation.X`.
- LinearRegressionForecaster fits only `fold.train.X`/`fold.train.y` and
  predicts `fold.validation.X`.
- RandomForestForecaster fits only `fold.train.X`/`fold.train.y` and predicts
  `fold.validation.X`.

No VALIDATION target enters fitting. Every model/fold pair is evaluated with
`evaluate_predictions(fold.validation.y, predictions)`, retaining exactly MAE,
RMSE, directional accuracy, and Pearson correlation. Eight years times four
models yields exactly 32 `ModelFoldResult` objects, ordered by validation year
and then the canonical model order above.

## Metric and consistency outputs

The future functions accept `Sequence[ModelFoldResult]` and return deterministic
tables in canonical model order.

`build_fold_metrics_table(results)` returns exactly:

```text
validation_year  model  MAE  RMSE  directional_accuracy  correlation
```

There is one row per result, hence 32 rows for the approved experiment.

`build_pooled_metrics(results)` groups prediction records by model,
concatenates all 2016-2023 OOS rows for that model first, and then calls
`evaluate_predictions` on concatenated `y_true` and `prediction`. It returns
exactly:

```text
model  MAE  RMSE  directional_accuracy  correlation
```

It never derives pooled performance by averaging annual metrics; that would be
incorrect, especially for RMSE and correlation. This pooled OOS table is the
primary Step 5E model comparison.

Within a fold, MeanBaseline predictions are constant, so fold Pearson
correlation is normally NaN. Different folds may learn different TRAIN means,
making pooled MeanBaseline predictions nonconstant. A finite pooled correlation
is therefore mathematically valid and is not an error.

`build_consistency_summary(results)` compares every model with MeanBaseline in
the same fold and returns exactly:

```text
model
folds_better_mae_vs_mean
folds_better_rmse_vs_mean
folds_better_direction_vs_mean
folds_positive_correlation
```

Wins use strict comparisons: MAE and RMSE count only when lower than the
fold's MeanBaseline value; directional accuracy counts only when higher.
Positive correlation counts only when `correlation > 0`. Ties are not wins and
NaN correlation is not positive. No composite score is created.

Annual metrics diagnose temporal stability and regime dependence. They may
show consistent wins, isolated strong years, deterioration in specific years,
or pooled performance hiding annual instability. Fold counts alone do not
establish statistical significance.

## Future synthetic TDD coverage

Future `tests/test_walk_forward.py` covers these distinct behaviors:

1. Approved 2016-2023 input produces eight folds.
2. Folds are returned chronologically.
3. Each `validation_year` is correct.
4. TRAIN expands across consecutive folds.
5. Every TRAIN date precedes its validation year.
6. No TRAIN `target_end_date` crosses validation start.
7. VALIDATION dates belong only to the requested year.
8. No VALIDATION `target_end_date` crosses into the next year.
9. A late-2023 row whose target ends in 2024 is removed.
10. Approved development input retains no date on or after 2024-01-01.
11. Approved development input retains no target end on or after 2024-01-01.
12. X is exactly `FEATURE_COLUMNS`.
13. y remains aligned to the target.
14. metadata remains aligned and contains only date/ticker.
15. Duplicate date/ticker rows are rejected.
16. The input DataFrame is not mutated.
17. An empty year sequence is rejected.
18. Duplicate years are rejected.
19. Unordered years are rejected.
20. Non-integer years are rejected.
21. An empty TRAIN fold is rejected.
22. An empty VALIDATION fold is rejected.
23. Changing 2023 observations cannot change fold construction or results for
    2016-2022; the 2023 fold may change.

Future `tests/test_evaluation.py` covers:

1. Four model types are evaluated per fold.
2. Eight folds produce 32 results.
3. Model instances are fresh per fold.
4. Canonical model names are used everywhere.
5. Prediction columns are exact.
6. Every prediction row carries the correct validation year.
7. Every prediction row carries the correct model name.
8. Prediction date/ticker values equal fold metadata.
9. `y_true` equals fold y.
10. Prediction index and row alignment are preserved.
11. OOS date/ticker/model keys are unique.
12. Stored fold metrics equal direct `evaluate_predictions` recomputation.
13. The fold table schema is exact.
14. The fold table has one row per result.
15. Pooled metrics concatenate OOS predictions before evaluation.
16. Pooled metrics are not arithmetic means of annual metrics.
17. The pooled table has one row per model.
18. Finite pooled MeanBaseline correlation is accepted.
19. Strict MAE wins are counted correctly.
20. Strict RMSE wins are counted correctly.
21. Strict directional wins are counted correctly.
22. Ties do not count as wins.
23. NaN correlation does not count as positive.
24. Positive correlation is counted correctly.
25. Repeated evaluation is reproducible with the fixed RandomForest.

Tests use synthetic, nondegenerate data and avoid repeated assertions added
only to inflate counts. The future-data isolation regression test mutates 2023
input and proves that folds and evaluation outputs for 2016-2022 are unchanged.

## Scope boundaries

Step 5E does not modify MeanBaseline, MomentumBaseline,
LinearRegressionForecaster, or RandomForestForecaster. If implementation finds
a genuine existing model bug that blocks the approved evaluation, work stops
and reports the bug before changing model behavior.

No Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, Ridge,
Lasso, ElasticNet, Random Forest tuning, GridSearchCV, RandomizedSearchCV, new
features, feature selection, ticker encoding, or sector encoding enters this
step. There is no TEST evaluation, backtest, portfolio construction, or claim
of profitability, investability, causality, statistical significance, or
economic significance.

No new runtime dependency is required. Future implementation uses existing
Python, pandas, scikit-learn, and current test/lint tooling.

## Future real-data evaluation and safety

Only after implementation is committed, one read-only evaluation will:

1. Hash raw and processed Parquet datasets.
2. Load processed features and exclude SPY in memory.
3. Prepare supervised data.
4. Retain only rows with both date and target end before 2024-01-01.
5. Construct annual folds for 2016-2023.
6. Evaluate all four existing models.
7. Build the fold, pooled OOS, and consistency tables.
8. Report diagnostics.
9. Re-hash the datasets and require exact hash equality.

All processing is in memory; no restricted development frame, OOS prediction
record, evaluation dataset, or other dataset is written. Any unexpected hash
change is a stop condition.

The evaluation audits eight folds with years exactly 2016-2023; TRAIN and
VALIDATION row counts and earliest/latest dates; maximum TRAIN and VALIDATION
`target_end_date`; all 15 stocks where expected; SPY absence; no development
row touching 2024; and exactly 32 model/fold results. Because partition metadata
remains exactly date/ticker, target-end maxima are audited through an exact
one-to-one in-memory join of partition metadata back to the already prepared
development frame. They are not added to `DatasetPartition.metadata`. No TEST
statistics are read or reported.

The official result is the four-model pooled OOS table, accompanied by annual
fold metrics, the consistency summary, and optional model-by-year diagnostics.
Permitted conclusions cover best pooled MAE/RMSE, relative pooled direction and
correlation, folds beating MeanBaseline, temporal stability, regime dependence,
and whether current features/models show consistent predictive signal. They do
not extend to profitability, investability, significance, causality, economics,
TEST performance, or backtesting.

## Future implementation files and README

The implementation phase is expected to modify:

- `src/stock_forecaster/models/dataset.py`
- `README.md`

It is expected to create:

- `src/stock_forecaster/models/evaluation.py`
- `tests/test_walk_forward.py`
- `tests/test_evaluation.py`
- `docs/superpowers/plans/2026-09-15-walk-forward-validation.md`

No dependency file change is expected. The future README will show the
progression from a single temporal validation split, to annual expanding
walk-forward validation, to pooled OOS comparison. It will explain annual
2016-2023 validation, expanding TRAIN, ticker-specific target purging, pooled
OOS evaluation, and untouched 2024+ TEST data. Numerical Step 5E results appear
only after the evaluation occurs.

## Design self-review

The specification was reviewed against all approved constraints. It fixes the
years at 2016-2023; uses expanding TRAIN with explicit TRAIN and VALIDATION
target purges; protects the 2023/2024 boundary; filters TEST-era rows before
fold generation; and keeps SPY exclusion in orchestration. It reuses
`DatasetPartition`, keeps model execution out of `dataset.py`, and keeps split
logic out of `evaluation.py`. It requires fresh instances of exactly the four
existing models, concatenates predictions before pooled evaluation, never
averages annual metrics into pooled metrics, documents the MeanBaseline pooled
correlation nuance, uses strict consistency comparisons, and defines no
composite score. It includes future-data isolation coverage, preserves existing
model behavior, introduces no tuning, model, dependency, TEST access, or
incomplete architectural choice, and uses interface and type names consistently.
