# SPY-Relative Five-Session Excess Return Target Design

## Status and objective

The Step 5F functional design is approved. This document specifies future
implementation and controlled evaluation. This specification task changes no
production code, tests, README content, dependencies, or datasets and performs
no model evaluation.

Step 5F asks whether five-session stock excess return relative to SPY is more
predictable than absolute five-session stock return when all other important
experimental dimensions remain fixed:

- the same 15 modeled stocks;
- the same ten `FEATURE_COLUMNS`;
- the same `LinearRegressionForecaster` and `RandomForestForecaster`;
- the same 2016-2023 expanding walk-forward method; and
- the same MAE, RMSE, directional-accuracy, and Pearson-correlation metrics.

The experiment changes the target and its natural baseline. It does not add
predictors, tune models, or open TEST.

## Target definitions and processed schema

`src/stock_forecaster/features/engineering.py` will define both targets:

```python
TARGET_COLUMN = "future_return_5d"
EXCESS_TARGET_COLUMN = "future_excess_return_5d"
```

The existing target remains unchanged and the new target is additive:

```text
future_return_5d
    = stock_adjusted_close_at_target_end
      / stock_adjusted_close_at_date
      - 1

future_excess_return_5d
    = stock_future_return_same_window
      - SPY_future_return_same_window
```

The price convention is exactly the current convention: `adjusted_close` at
both endpoints, with return computed as `end / start - 1` through the existing
safe relative-change behavior. Step 5F does not substitute unadjusted close or
alter `future_return_5d`.

The processed output order becomes:

```text
date
ticker
FEATURE_COLUMNS
future_return_5d
future_excess_return_5d
```

The ten entries and declared order of `FEATURE_COLUMNS` do not change. Neither
SPY return nor any other market-derived value becomes an input feature.

## Exact benchmark-window construction

For each ticker row, that ticker's fifth future observation defines the
authoritative `target_end_date`. The stock return uses the row's `date` and
that exact end date. The SPY return must then use SPY adjusted-close values on
the same two dates:

```text
stock window: date -> stock target_end_date
SPY window:   date -> the same stock target_end_date
```

An independent SPY fifth-future observation is not used unless it happens to
land on the same end date. This prevents differing ticker calendars from
silently changing the benchmark horizon.

`build_features` may compute ticker-local fifth-future dates internally. A
single exact-date SPY adjusted-close lookup is then used for both window
endpoints. The internal end-date Series is calculation state only; it is not
added to the persisted processed schema. `prepare_supervised_data` remains the
owner of supervised `target_end_date` metadata.

SPY must exist on both exact endpoints. If either the start date or the stock's
target end date is absent from SPY history, the excess target is `NaN`. The
implementation performs no forward fill, backward fill, interpolation,
nearest-date lookup, or substitute-session selection. If no SPY history is
present, all excess-target values are consequently `NaN`; the absolute target
and predictors remain valid.

For a SPY row with a complete horizon and both endpoints, stock and benchmark
returns are identical, so `future_excess_return_5d` is zero within normal
floating-point tolerance. SPY is not special-cased to missing. Its zero target
is retained in processed data for auditing, while SPY remains excluded later
from the modeled universe.

The final five observations of every ticker lack a fifth future observation.
Both target columns remain `NaN` on those rows, and feature engineering keeps
the rows just as it does today.

The public `build_features(prices)` signature stays unchanged. The new logic
belongs in the existing feature-engineering module; one additional target does
not justify a target subsystem, registry, or new abstraction.

## Supervised target selection

The public preparation API becomes:

```python
prepare_supervised_data(
    features: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame
```

Only these two target names are valid:

```text
future_return_5d
future_excess_return_5d
```

An unknown name raises `ValueError` before row preparation. Arbitrary feature,
metadata, or future-looking columns cannot be selected as y.

The default call remains the Step 5A-5E behavior:

```python
prepare_supervised_data(features)
```

It selects `future_return_5d` and returns the same rows, values, ordering, and
column contract as before. An explicit Step 5F call selects the excess target:

```python
prepare_supervised_data(
    features_without_spy,
    target_column=EXCESS_TARGET_COLUMN,
)
```

Preparation selects exactly one target into the returned supervised frame:

```text
date
ticker
target_end_date
FEATURE_COLUMNS
selected target column
```

It does not carry the unselected target into the prepared frame. This makes the
target choice explicit and prevents the unselected target from entering X or
affecting completeness filtering. Rows are removed only for missing predictor
values, missing selected-target values, or missing `target_end_date`, matching
the current semantics for the selected target.

The internal input and supervised column contracts therefore become
target-dependent rather than fixed to `TARGET_COLUMN`. Existing
`temporal_split` and `walk_forward_splits` signatures remain unchanged. They
require their prepared input to contain exactly one approved target column,
reject an ambiguous frame containing both approved targets, and use that
column when constructing every `DatasetPartition.y`.

For either target:

- X contains exactly `FEATURE_COLUMNS` in declared order;
- y retains the selected target's pandas Series name;
- the prepared frame contains `date`, `ticker`, and `target_end_date`;
- partition metadata remains exactly `date` and `ticker`, as in Step 5E;
- `target_end_date` is used for purging before partition construction and is
  not exposed as a predictor;
- sorting, alignment, duplicate rejection, input immutability, null handling,
  and ticker-local target-end construction remain unchanged.

Adding the excess column to processed data must not change the absolute target,
target-end dates, default supervised rows, Step 5E fold membership, default
predictions, or Step 5E metric values. The only intentional Step 5E API change
is the consistency-summary column rename described below.

## ZeroBaseline and metric interpretation

`src/stock_forecaster/models/baseline.py` will add:

```python
class ZeroBaseline:
    def predict(self, X: pd.DataFrame) -> pd.Series: ...
```

It is stateless, has no `fit` method, and has no configurable constant. It
returns a floating-point `pd.Series` containing `0.0` for every input row, with
name `"prediction"` and index exactly equal to `X.index`. It reads no feature
value and does not mutate X.

The existing metric implementation is reused without modification. For the
excess target:

```text
target > 0  => stock outperformed SPY
target < 0  => stock underperformed SPY
target = 0  => equal return over the exact window
```

Zero predictions have zero variance, so ZeroBaseline Pearson correlation is
expected to be `NaN`. The current sign convention treats zero as its own sign,
so ZeroBaseline directional accuracy may be near zero when exact-zero targets
are rare. Neither metric is special-cased. ZeroBaseline is the primary natural
reference for excess-target MAE and RMSE; direction is interpreted mainly
among variable predictions, and correlation only for predictions that vary.

`MomentumBaseline` remains implemented and unchanged. It is excluded from the
official Step 5F comparison because absolute historical stock momentum is not
the intended natural baseline for a SPY-relative target.

## Parameterized walk-forward evaluation

`src/stock_forecaster/models/evaluation.py` will define:

```python
DEFAULT_MODEL_NAMES = (
    "MeanBaseline",
    "MomentumBaseline",
    "LinearRegression",
    "RandomForest",
)

evaluate_walk_forward(
    folds: Sequence[WalkForwardFold],
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> tuple[ModelFoldResult, ...]
```

Omitting `model_names` reproduces Step 5E's exact model set and fold-major model
order. Step 5F passes:

```text
ZeroBaseline
MeanBaseline
LinearRegression
RandomForest
```

The model-name sequence must be non-empty, contain only the five known names,
and contain no duplicate. A plain string is rejected rather than interpreted
as a sequence of characters. Unknown names raise `ValueError`. Validation
occurs before any fold is fitted, and returned `ModelFoldResult` objects
preserve the caller-provided model order within each fold.

Resolution remains a small explicit branch inside `evaluation.py` for:

```text
ZeroBaseline
MeanBaseline
MomentumBaseline
LinearRegression
RandomForest
```

There is no public registry, dynamic import, plugin framework, generic
estimator protocol, or dependency-injection layer.

Every selected model is independently constructed or invoked for every fold:

- ZeroBaseline predicts from VALIDATION X without fitting;
- MeanBaseline is newly created, fits only TRAIN y, and predicts the
  VALIDATION index;
- MomentumBaseline predicts independently from VALIDATION X when selected;
- LinearRegressionForecaster is newly created, fits only TRAIN X/y, and
  predicts VALIDATION X;
- RandomForestForecaster is newly created, fits only TRAIN X/y, and predicts
  VALIDATION X.

No fitted state crosses fold boundaries. No validation target enters fitting,
and no model hyperparameter changes.

The prediction record and `ModelFoldResult` contracts remain unchanged. OOS
records stay in memory with exactly:

```text
date  ticker  validation_year  model  y_true  prediction
```

Each fold/model pair continues to call `evaluate_predictions` directly.

## Aggregation with selectable model sets

The fold and pooled builders must operate on either official model set rather
than iterate only the Step 5E defaults. Internally, known model presentation
order is fixed as:

```text
ZeroBaseline
MeanBaseline
MomentumBaseline
LinearRegression
RandomForest
```

Only models present in the results are emitted. This leaves Step 5E table order
unchanged when ZeroBaseline is absent and gives the official Step 5F order when
MomentumBaseline is absent. Raw `evaluate_walk_forward` results still preserve
the caller-provided order; table builders normalize shuffled input results to
deterministic known-model presentation order.

`build_fold_metrics_table(results)` retains the exact schema:

```text
validation_year  model  MAE  RMSE  directional_accuracy  correlation
```

`build_pooled_metrics(results)` retains:

```text
model  MAE  RMSE  directional_accuracy  correlation
```

Pooled metrics continue to concatenate all OOS rows for a model before calling
`evaluate_predictions`; annual metrics are never averaged into pooled metrics.

## Parameterized consistency summary

The consistency API changes to consume the already built annual metrics table:

```python
build_consistency_summary(
    fold_metrics: pd.DataFrame,
    reference_model: str = "MeanBaseline",
) -> pd.DataFrame
```

The default reference preserves Step 5E numerical comparisons. Step 5F passes
`reference_model="ZeroBaseline"`.

The input must be a non-empty fold-metrics frame containing at least:

```text
validation_year
model
MAE
RMSE
directional_accuracy
correlation
```

The function rejects duplicate `(validation_year, model)` rows. It also
requires the named reference to exist and to appear exactly once for every
validation year represented by the table. Each non-reference row is joined
only to its same-year reference row. Missing or duplicated same-year reference
data raises clearly instead of being ignored or aggregated.

The exact output schema becomes:

```text
model
folds_better_mae_vs_reference
folds_better_rmse_vs_reference
folds_better_direction_vs_reference
folds_positive_correlation
```

The model order follows the deterministic order in `fold_metrics`. No duplicate
deprecated `*_vs_mean` columns remain because there is no external consumer.

Rules stay strict:

- lower MAE than the same-year reference is a win;
- lower RMSE than the same-year reference is a win;
- higher directional accuracy than the same-year reference is a win;
- `correlation > 0` counts as positive;
- ties are not wins;
- zero, negative, and `NaN` correlations are not positive;
- no composite score is calculated.

With `reference_model="MeanBaseline"`, Step 5E counts remain numerically
identical; only the three comparison column names change to
`*_vs_reference`. Fold and pooled Step 5E metrics remain numerically and
structurally unchanged.

## Controlled processed-data migration

Step 5F intentionally changes `data/processed/features.parquet` by adding one
column. The implementation/evaluation phase performs the migration once and
audits it. This specification phase does not modify the file.

Before regeneration, orchestration records SHA-256 for:

- authoritative raw `data/raw/prices.parquet`;
- existing `data/processed/features.parquet`.

The currently validated Step 5E hashes provide the starting anchors:

```text
raw:       49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5
processed: 56001B3748F349E0DF5A29F30C1884B74810FDD2F87D28A5064E71E02675631A
```

If the migration starts from different artifacts, execution stops for review
rather than silently changing the reference dataset.

The controlled sequence is:

1. Hash raw and old processed data.
2. Load and retain the old processed frame for comparison.
3. Regenerate a candidate processed frame from authoritative raw data through
   the updated `build_features`.
4. Align old and candidate frames deterministically by exact `(date, ticker)`.
5. Run the old-column equality gate.
6. Audit the new target.
7. Only after all gates pass, write the candidate as the new
   `features.parquet`.
8. Reload it, repeat schema/target checks, and record the new processed hash.

Before replacement, old and candidate frames must have:

- the same row count;
- the same unique `(date, ticker)` key set;
- identical deterministic ordering after sorting by `date`, then `ticker`;
- identical `date` and `ticker` values;
- matching `NaN` positions in every existing numeric column; and
- equal values for every existing numeric column.

The equality gate covers exactly:

```text
date
ticker
FEATURE_COLUMNS
future_return_5d
```

Keys and nonnumeric values compare exactly. Existing numeric values use
`rtol=1e-12`, `atol=1e-15`, and `equal_nan=True`, which is strict for the
project's float64 feature calculations while tolerating harmless floating-point
roundoff. Any unexpected old-column change stops execution before a write.

The new-target audit requires:

- `future_excess_return_5d` exists and has a numeric pandas dtype;
- every non-null value is finite;
- representative rows from multiple tickers and years match stock same-window
  return minus exact-date SPY same-window return;
- rows missing either exact SPY endpoint are `NaN`;
- complete-horizon SPY rows are zero within the same strict tolerance; and
- final incomplete SPY horizons may remain `NaN`.

Raw data is immutable throughout:

```text
raw_sha_before == raw_sha_after
```

The processed hash is expected to change once because the schema changes:

```text
old_processed_sha != new_processed_sha
```

The new hash is recorded before evaluation and recalculated afterward:

```text
new_processed_sha_before_evaluation
    == new_processed_sha_after_evaluation
```

Step 5F evaluation is read-only and persists no predictions, folds, reports,
or model artifacts.

## Mandatory Step 5E regression gate

Step 5E compatibility is proven before any Step 5F result is observed. The
migration process keeps the old processed frame in memory and establishes a
Step 5E reference from the validated old artifact. The new processed file is
then evaluated through the backward-compatible defaults:

```python
prepare_supervised_data(new_features)
evaluate_walk_forward(step_5e_folds)
build_consistency_summary(
    build_fold_metrics_table(step_5e_results),
    reference_model="MeanBaseline",
)
```

The regression run uses the original 15 stocks, excludes SPY before supervised
preparation, applies both `date < 2024-01-01` and
`target_end_date < 2024-01-01`, and builds the same 2016-2023 folds.

The old and new Step 5E outputs must have identical schemas, model/year keys,
and `NaN` positions. Every fold and pooled numeric metric is compared with
`rtol=1e-12` and `atol=1e-15`. Consistency counts compare exactly after mapping
the old semantic columns to their new `*_vs_reference` names. Default result
order and all predictions must remain reproducible.

This dynamic comparison is anchored to the previously validated old processed
hash, so it checks the complete 32-row fold table rather than only headline
values. If any default target row, fold membership, prediction, metric, pooled
value, or consistency count differs unexpectedly, execution stops and Step 5F
is not run.

## Official Step 5F evaluation

Only after migration and the Step 5E regression gate pass does the one official
Step 5F development evaluation run.

The modeled universe remains exactly:

```text
AAPL  MSFT  GOOGL  AMZN  META  NVDA  JPM  V
JNJ   UNH   XOM    CVX   COST  WMT   PG
```

SPY remains in processed data for target construction and audit, then is
excluded before Step 5F supervised preparation. The selected target is
`EXCESS_TARGET_COLUMN`.

Development rows must satisfy both:

```text
date < 2024-01-01
target_end_date < 2024-01-01
```

The validation years remain exactly 2016 through 2023 with expanding TRAIN,
ticker-specific target-end purging, fresh model instances, and in-memory OOS
predictions. TEST remains conceptually 2024+ and supplies no fitting,
evaluation, diagnostic, comparison, or selection evidence.

The official model set is exactly:

```text
ZeroBaseline
MeanBaseline
LinearRegression
RandomForest
```

MomentumBaseline is not part of the official Step 5F output. No model is tuned.

The three official output families are:

```text
Pooled OOS metrics:
model  MAE  RMSE  directional_accuracy  correlation

Annual fold metrics:
validation_year  model  MAE  RMSE  directional_accuracy  correlation

Consistency summary against ZeroBaseline:
model
folds_better_mae_vs_reference
folds_better_rmse_vs_reference
folds_better_direction_vs_reference
folds_positive_correlation
```

Pooled metrics are recomputed from concatenated OOS rows. Annual metrics show
regime stability. Consistency uses `reference_model="ZeroBaseline"` and strict
same-year comparisons.

## Interpretation and permitted conclusions

Step 5F is promising only when evidence improves coherently across multiple
dimensions, including:

- MAE or RMSE improvement over ZeroBaseline in a majority of annual folds;
- positive pooled correlation;
- positive correlation across many annual folds;
- useful and stable outperform/underperform direction among variable models;
- behavior that persists across years rather than one isolated regime.

There is no arbitrary single pass threshold and no composite score. Step 5E
absolute-return results may be shown for context, but raw MAE/RMSE from the two
different targets are not treated as a direct same-scale model victory. Useful
context includes correlation magnitude and stability, directional behavior,
and improvement relative to each target's natural baseline.

Permitted conclusions are limited to whether excess return appears more
learnable than absolute return, whether ML models reduce error relative to
ZeroBaseline, whether relative-return correlation is stronger or more stable,
whether signal persists across years, and whether future work should focus on
the excess-return formulation.

The experiment makes no claim about profitability, investability, causality,
statistical significance, economic significance, TEST performance, or
backtest performance.

## Future synthetic TDD coverage

Feature-engineering tests will use deterministic synthetic prices and cover:

1. Existing `future_return_5d` values remain unchanged.
2. The excess target exists.
3. The stock's own fifth-future date defines the benchmark window.
4. The stock return is correct.
5. The exact-date SPY return is correct.
6. Excess equals stock return minus SPY return.
7. Complete SPY horizons produce zero excess.
8. A missing exact SPY start produces `NaN`.
9. A missing exact SPY end produces `NaN`.
10. No forward fill occurs.
11. No backward fill occurs.
12. A different independent SPY fifth observation is not substituted.
13. The final five incomplete stock horizons remain `NaN`.
14. The ten `FEATURE_COLUMNS` and their values remain unchanged.
15. The input DataFrame is not mutated.
16. Every non-null excess value is finite.

Dataset tests will cover:

1. The default target remains `future_return_5d`.
2. Explicit absolute-target selection works.
3. Explicit excess-target selection works.
4. Unknown targets are rejected.
5. X remains exactly `FEATURE_COLUMNS`.
6. y retains the selected target name.
7. `target_end_date` is unchanged by target choice.
8. Prepared identifying and horizon data remain unchanged.
9. X/y/metadata row alignment is preserved.
10. Existing target-horizon purge behavior is preserved.
11. Input is not mutated.
12. Excess-target null rows are removed only when excess is selected.
13. The excess target never enters X.

Baseline and evaluation tests will cover:

1. ZeroBaseline predicts exactly zero.
2. It returns a pandas Series.
3. It preserves the input index.
4. Its output name is `prediction`.
5. It does not mutate X.
6. Default evaluation reproduces the Step 5E model set.
7. The Step 5F model set is accepted.
8. ZeroBaseline results are produced correctly.
9. Caller-provided evaluation model order is preserved.
10. Unknown model names are rejected.
11. Duplicate model names are rejected.
12. An empty model sequence is rejected.
13. Fresh fitted instances remain isolated per fold.
14. Default evaluation remains reproducible.
15. MeanBaseline is the default consistency reference.
16. ZeroBaseline is accepted as the consistency reference.
17. Consistency output uses only `*_vs_reference` columns.
18. Reference comparison is same-year only.
19. Strict MAE wins are correct.
20. Strict RMSE wins are correct.
21. Strict directional wins are correct.
22. Ties do not count.
23. `NaN` correlation does not count.
24. Positive correlation counts.
25. A missing reference is rejected.
26. A duplicate reference within a year is rejected.
27. A reference missing from one represented year is rejected.
28. Step 5E consistency counts remain unchanged with MeanBaseline reference.

Existing linear, forest, metric, walk-forward purge, alignment, reproducibility,
and future-isolation tests remain regression coverage. The implementation plan
must use RED-GREEN TDD for every new behavior.

## Future implementation files and documentation

Expected production/documentation changes are limited to:

- `src/stock_forecaster/features/engineering.py`
- `src/stock_forecaster/models/dataset.py`
- `src/stock_forecaster/models/baseline.py`
- `src/stock_forecaster/models/evaluation.py`
- `README.md`

Expected test changes are:

- `tests/test_features.py`
- `tests/test_dataset.py`
- `tests/test_baseline.py`
- `tests/test_evaluation.py`

`tests/test_walk_forward.py` may receive only compatibility coverage if needed;
the temporal fold algorithm itself does not change. No new dependency is
required. `linear.py`, `forest.py`, and `metrics.py` remain behaviorally and
textually unchanged unless an existing blocking defect is discovered, in which
case implementation stops before altering them.

The future README will explain both supported targets, exact same-window SPY
comparison, ZeroBaseline, the official Step 5F model set, the unchanged
2016-2023 walk-forward method, untouched 2024+ TEST boundary, and Step 5E
reproducibility through default target/model behavior. Numerical Step 5F
results appear only after the controlled evaluation.

The future implementation plan path is:

```text
docs/superpowers/plans/2026-09-16-excess-return-target.md
```

It is not created during this specification task.

## Explicit exclusions

Step 5F excludes new predictor features, SPY as X, market-return features,
beta adjustment, CAPM alpha, cross-sectional ranking, classification targets,
ticker encoding, sector encoding, and volatility-regime features. It also
excludes Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost,
Ridge, Lasso, ElasticNet, Random Forest tuning, GridSearchCV,
RandomizedSearchCV, portfolio construction, backtesting, TEST evaluation, and
model selection using TEST.

## Design self-review

The specification has been checked against the approved functional design:

1. Existing `future_return_5d` is unchanged.
2. The excess target is additive, not a replacement.
3. The stock target end defines the exact benchmark window.
4. A missing exact SPY endpoint produces `NaN`.
5. No SPY imputation or substitute-session logic exists.
6. Complete SPY horizons produce zero excess.
7. No new predictor is introduced.
8. Default supervised preparation remains the absolute target.
9. Only the two approved target names are accepted.
10. X remains exactly `FEATURE_COLUMNS`.
11. ZeroBaseline is stateless.
12. MomentumBaseline remains unchanged.
13. Default walk-forward evaluation reproduces Step 5E.
14. The Step 5F model set excludes MomentumBaseline.
15. Consistency reference is configurable.
16. MeanBaseline remains the default consistency reference.
17. The `*_vs_reference` rename is explicit and exclusive.
18. Step 5E numerical behavior has a mandatory regression gate.
19. Raw data is immutable.
20. Processed migration is controlled and auditable.
21. Every old processed column has an equality gate.
22. The processed hash change is intentional and recorded.
23. Evaluation after migration is read-only.
24. The Step 5E regression gate precedes Step 5F results.
25. Validation years remain exactly 2016-2023.
26. TEST remains excluded from fitting and evaluation.
27. No tuning or new trainable model family enters Step 5F.
28. No profitability or backtest claim is permitted.
29. No new dependency is required.
30. The document is complete and every architectural decision is resolved.
31. Public interface names and internal ownership boundaries are consistent.
