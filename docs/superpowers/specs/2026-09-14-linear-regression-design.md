# Global Standardized Linear Regression Design

## Goal

Build the first real machine-learning benchmark for `future_return_5d` using
the existing `FEATURE_COLUMNS` in this pipeline:

```text
StandardScaler
      ↓
LinearRegression
```

The model tests whether the engineered predictors contain useful linear
forecasting signal beyond the simple baselines. This step makes no claim about
profitability.

## Modeling universe

The modeling universe contains exactly these 15 stocks:

```text
AAPL  MSFT  GOOGL  AMZN  META  NVDA  JPM  V
JNJ   UNH   XOM    CVX   COST  WMT   PG
```

`benchmark_ticker = "SPY"`. SPY remains in raw data, processed data, and
`configs/market.yaml`, but is excluded in the Step 5C evaluation orchestration
before supervised rows and temporal partitions are rebuilt. The orchestration
must assert that SPY is absent and all 15 intended stocks are present in both
TRAIN and VALIDATION. The generic `prepare_supervised_data` and
`temporal_split` functions remain unchanged and contain no hard-coded universe
rule.

SPY remains available for later benchmark comparisons, portfolio backtesting,
and possible market-context features.

## Comparable validation experiment

The Step 5B results used all 16 tickers and are not direct comparison numbers
for this model. Step 5C must perform one new, internally comparable experiment:

1. Filter the processed frame to the 15-stock modeling universe.
2. Run `prepare_supervised_data` and `temporal_split` with the approved dates.
3. Fit `MeanBaseline` only on the resulting 15-stock TRAIN target.
4. Predict `MomentumBaseline` only from the same 15-stock VALIDATION `X`.
5. Fit linear regression only on the 15-stock TRAIN partition and predict the
   identical 15-stock VALIDATION rows.
6. Evaluate all three prediction Series against the same VALIDATION target.

Audit expectations are approximately 43,906 TRAIN rows and 7,440 VALIDATION
rows, calculated at runtime rather than hard-coded. The expected values follow
from removing SPY's 2,967 TRAIN and 496 VALIDATION rows from the current split.

## Global model and feature contract

One global model is fitted across all 15 stocks. There are no ticker-specific
or sector-specific models. `ticker` remains metadata and is not encoded.

The model imports `FEATURE_COLUMNS` from
`stock_forecaster.features.engineering`. Both fitting and prediction require
the ordered schema to be exactly `list(FEATURE_COLUMNS)`. Missing, extra, or
reordered columns raise `ValueError`; the implementation does not silently
select or reorder columns. This strict contract prevents accidental inclusion
of `date`, `ticker`, `target_end_date`, or `future_return_5d` in `X` and keeps
coefficient labels deterministic.

## Pipeline and leakage controls

The implementation uses one `sklearn.pipeline.Pipeline`:

```python
Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)
```

The scaler is never fitted separately. Calling `fit` with the 15-stock TRAIN
`X` and `y` is the only fitting operation, so both scaler statistics and
regression parameters learn from TRAIN only. VALIDATION is passed only to
`predict`; prediction must not call `fit`, `fit_transform`, or otherwise alter
the fitted scaler or regressor state.

TEST remains reserved. Step 5C does not access TEST features or targets,
generate TEST predictions, calculate TEST metrics or errors, compare TEST
against baselines, or use TEST for a modeling decision.

## `LinearRegressionForecaster` interface

The future module `stock_forecaster.models.linear` exposes one focused class:

```python
model = LinearRegressionForecaster()
model.fit(X_train, y_train)
predictions = model.predict(X_validation)
```

Contract:

- `fit` returns `self`.
- `predict` before a successful fit raises a clear error.
- `X` must have exactly `FEATURE_COLUMNS` in their declared order.
- Fit rejects unequal `X`/`y` lengths and non-identical indices.
- Fit rejects null or infinite values in either `X` or `y`.
- Predict rejects null or infinite values in `X`.
- Fit and predict do not mutate caller-owned DataFrames or Series.
- Predict returns a `pd.Series` named `prediction`, preserving the input row
  order and index exactly.
- The class wraps only the approved pipeline and does not introduce an abstract
  estimator hierarchy.

After fitting, the class exposes:

- `coefficients_`: a copied `pd.Series` indexed exactly by `FEATURE_COLUMNS`.
- `intercept_`: a finite `float`.

The implementation phase adds unpinned `scikit-learn` as a runtime dependency,
consistent with the project's current unpinned dependency policy. It adds no
other modeling library.

## Coefficient interpretation

Because inputs are standardized and the target remains a decimal return, a
coefficient is the fitted change in `future_return_5d` associated with a
one-standard-deviation increase in that feature, conditional on the other
linear terms.

Engineered predictors may be correlated. Coefficient sign and magnitude are
diagnostics, not causal effects, definitive feature importance, or independent
economic effects. Step 5C performs no statistical-significance testing.

The evaluation report sorts this table by `absolute_coefficient` descending:

```text
feature | coefficient | absolute_coefficient
```

It also reports the intercept without claiming that the largest coefficient is
the most economically important feature.

## Evaluation outputs

Reuse `evaluate_predictions` without adding a model-specific metric. The
15-stock VALIDATION comparison contains:

```text
model | MAE | RMSE | directional_accuracy | correlation
```

for `MeanBaseline`, `MomentumBaseline`, and `LinearRegression`. Each row also
has prediction mean, standard deviation, minimum, and maximum diagnostics.

Using VALIDATION metadata, report LinearRegression diagnostics by ticker:

```text
ticker | row_count | MAE | directional_accuracy | correlation
```

These results diagnose a single global model. They do not authorize separate
ticker models, baseline tuning, backtesting, portfolio returns, or profitability
claims.

## Testing and audit design

Implementation follows synthetic TDD. Tests must observe failure before
production code and cover:

- predict before fit;
- exact ordered `FEATURE_COLUMNS` acceptance;
- missing, extra, and reordered feature rejection;
- null and positive/negative infinity rejection in applicable `X` and `y`;
- `X`/`y` length and index mismatch rejection;
- no input mutation during fit or prediction;
- exact prediction index and row alignment;
- `coefficients_` indexed in `FEATURE_COLUMNS` order;
- a finite `intercept_`;
- recovery of predictions from a known deterministic linear relationship;
- scaler means learned from literal TRAIN values only;
- unchanged fitted scaler statistics after prediction on deliberately extreme
  VALIDATION values; and
- global multi-row, multi-feature prediction behavior.

The read-only real-data audit must verify:

- SPY is absent from modeling TRAIN and VALIDATION rows;
- all 15 intended stock tickers are present in both partitions;
- actual universe counts and date ranges are reported;
- all three models use identical VALIDATION indices and observations;
- MeanBaseline is refitted on only the 15-stock TRAIN target;
- MomentumBaseline consumes only the 15-stock VALIDATION `X`;
- LinearRegression fits only TRAIN and predicts only VALIDATION;
- comparison, prediction-distribution, per-ticker, and coefficient diagnostics
  are produced; and
- TEST is neither accessed nor evaluated.

## Expected implementation files

The later implementation phase is expected to create:

- `src/stock_forecaster/models/linear.py`
- `tests/test_linear.py`
- `docs/superpowers/plans/2026-09-14-linear-regression.md`

It is expected to modify only:

- `pyproject.toml`
- `README.md`

The dataset layer and market configuration are not implementation targets.

## Out of scope

Step 5C does not include Ridge, Lasso, ElasticNet, polynomial features,
interaction terms, per-ticker models, ticker encoding, feature selection,
hyperparameter tuning, walk-forward validation, cross-validation, TEST
evaluation, backtesting, portfolio construction, statistical-significance
testing, or profitability claims.
