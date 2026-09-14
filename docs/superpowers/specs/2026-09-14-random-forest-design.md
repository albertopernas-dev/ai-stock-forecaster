# Fixed Global Random Forest Regression Design

## Status and objective

The Step 5D functional design is approved. This document specifies future
implementation; this specification task creates no implementation, tests,
implementation plan, dependency changes, or real-model results.

The primary question is: can a moderately regularized nonlinear tree ensemble
extract useful predictive signal from the existing ten engineered features
where global standardized LinearRegression did not outperform MeanBaseline?

Random Forest is the project's first nonlinear forecasting model for
`future_return_5d`. It is a controlled nonlinear benchmark, not a search for the
best possible or optimal Random Forest and not a profitability experiment.

## Historical motivation

Step 5C reported approximately these results on the 15-stock, 2022–2023
VALIDATION universe:

| model | MAE | RMSE | directional_accuracy | correlation |
| --- | ---: | ---: | ---: | ---: |
| MeanBaseline | 0.032360 | 0.045708 | 0.537097 | NaN |
| LinearRegression | 0.032482 | 0.045850 | 0.536022 | 0.024382 |
| MomentumBaseline | 0.046808 | 0.065526 | 0.512097 | -0.025312 |

These are historical context only. Step 5D must recompute every comparison model
from scratch on the same in-memory universe; these numbers must not be
hard-coded or copied into future evaluation output.

## Modeling universe and predictor schema

Use exactly these 15 stocks:

```text
AAPL  MSFT  GOOGL  AMZN  META  NVDA  JPM  V
JNJ   UNH   XOM    CVX   COST  WMT   PG
```

`benchmark_ticker = "SPY"`. SPY remains in raw data, processed data, and
`configs/market.yaml`. Evaluation orchestration excludes SPY in memory before
supervised preparation and asserts the exact 15-stock universe in TRAIN and
VALIDATION. Neither `prepare_supervised_data` nor `temporal_split` acquires a
hard-coded SPY exclusion; the dataset layer and market configuration remain
unchanged.

Fit one global forest across all 15 stocks, never separate ticker or sector
models. Ticker remains identifying metadata only, with no ticker or sector
encoding. Import `FEATURE_COLUMNS` from
`stock_forecaster.features.engineering`; do not duplicate the ten-column
constant in model code.

Both fit and predict require `list(X.columns) == list(FEATURE_COLUMNS)` in the
declared order. Missing, extra, duplicate, or reordered columns raise a clear
`ValueError`; do not silently select or reorder columns. Thus `ticker`, `date`,
`target_end_date`, and `future_return_5d` are never predictors.

## Fixed estimator and no scaling

Use this estimator directly inside the forecaster:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)
```

The configuration is fixed before observing Step 5D VALIDATION results.

| parameter | rationale |
| --- | --- |
| `n_estimators=300` | Reduce ensemble variance while remaining computationally practical. |
| `max_depth=8` | Constrain individual tree complexity. |
| `min_samples_leaf=20` | Avoid highly specific tiny leaves and smooth regression predictions. |
| `max_features=1.0` | Allow all ten features to be considered at each split. |
| `random_state=42` | Make model behavior reproducible in the same supported environment. |
| `n_jobs=-1` | Allow local parallel tree construction. |

This is not claimed to be optimal. RandomForestForecaster uses no StandardScaler
or other feature scaling. Tree splits depend on ordered thresholds rather than
standardized magnitudes. Do not introduce a preprocessing pipeline merely for
symmetry with LinearRegressionForecaster.

## Public interface and input contracts

The future `src/stock_forecaster/models/forest.py` exposes one focused
`RandomForestForecaster` class, without an estimator hierarchy or unnecessary
abstractions:

```text
fit(X: pd.DataFrame, y: pd.Series) -> RandomForestForecaster
predict(X: pd.DataFrame) -> pd.Series
feature_importances_: pd.Series  # read-only post-fit diagnostic property
```

`fit` returns `self`. The caller supplies already aligned training data.
Before calling sklearn, validate the strict ordered predictor schema and:

- Every feature column and `y` must have numeric pandas dtypes. Integer and
  floating-point numeric dtypes are acceptable; no single dtype such as
  `float64` is required. Non-numeric input raises a clear `ValueError`, without
  silently coercing strings to numbers.
- Fit rejects nulls, positive infinity, and negative infinity in either `X` or
  `y`; predict rejects the same invalid values in `X`, with clear `ValueError`s.
- Fit requires both `len(X) == len(y)` and `X.index.equals(y.index)`.
  Equal-length inputs with different index values or ordering raise a clear
  `ValueError`. The model does not sort, reset, reindex, or automatically align
  either input.
- Fit does not mutate caller-owned `X` or `y`; predict does not mutate `X`.

Before a successful fit, predict and access to `feature_importances_` raise a
clear public `RuntimeError`, not an obscure internal sklearn fitted-state
exception. Predict must not fit, update the forest, or learn from its input.

Prediction returns a `pd.Series` with `name == "prediction"`, one value per
input row, and `predictions.index.equals(X.index)` exactly. Preserve index
values and ordering; never reset, sort, or otherwise change row alignment.

## Feature importance and reproducibility

After fitting, `feature_importances_` is a copied/safe `pd.Series`, indexed
exactly by `FEATURE_COLUMNS` in their approved order. Its values correspond
directly to `RandomForestRegressor.feature_importances_`, are finite and
non-negative, and sum approximately to 1.0. Consumers must not be able to mutate
estimator state through the returned Series. Do not fabricate or renormalize
values to satisfy the contract; an invalid diagnostic state raises a clear
`RuntimeError` rather than exposing misleading importances.

These impurity-based importances are model diagnostics only, not causality,
economic significance, independent effects, definitive feature importance, or
proof that a feature should be traded. Correlated predictors can divide or
distort apparent importance. Step 5D includes no SHAP, permutation importance,
or partial dependence; first establish validation performance.

With the fixed seed and configuration, identical training data must produce
matching predictions in the same supported environment. Test two separately
instantiated forecasters fitted on identical synthetic `X`/`y`. This is not a
promise of bitwise equality across different library versions or environments,
and tests should not inspect unrelated implementation details.

## Future synthetic TDD coverage

Create `tests/test_forest.py` only during implementation. Observe failing tests
before writing production behavior. Use informative, nondegenerate synthetic
training data with enough rows for the fixed leaf-size configuration when
testing splits and normalized importances. Cover all these distinct behaviors;
parameterization or grouped assertions are acceptable without count-padding:

1. The underlying estimator is exactly RandomForestRegressor.
2. `n_estimators` is exactly 300.
3. `max_depth` is exactly 8.
4. `min_samples_leaf` is exactly 20.
5. `max_features` is exactly 1.0.
6. `random_state` is exactly 42.
7. `n_jobs` is exactly -1.
8. Fit returns self.
9. Predict before fit raises RuntimeError.
10. Feature importance access before fit raises RuntimeError.
11. Exact ordered FEATURE_COLUMNS are accepted, including integer and floating-point numeric inputs.
12. A missing feature is rejected clearly.
13. An extra feature is rejected clearly.
14. Reordered features are rejected without silent reordering.
15. A non-numeric feature column is rejected before sklearn.
16. Non-numeric y is rejected before sklearn.
17. Feature NaN is rejected.
18. Target NaN is rejected.
19. Feature infinity is rejected for both signs.
20. Target infinity is rejected for both signs.
21. Unequal X/y lengths are rejected.
22. Equal-length X/y with different index values or index ordering are rejected.
23. Fit does not mutate X.
24. Fit does not mutate y.
25. Predict does not mutate X.
26. Prediction output is a pd.Series.
27. Prediction name is exactly "prediction".
28. Prediction index equals X.index, including a non-default, unsorted index.
29. Prediction count equals the input row count.
30. Feature importances are a pd.Series corresponding directly to estimator importances.
31. Feature importance index equals FEATURE_COLUMNS in order.
32. Importances are finite.
33. Importances are non-negative.
34. Importances sum approximately to 1.0 on informative synthetic data.
35. Separate instances trained on identical data produce matching predictions.
36. Multi-row predictions preserve each row's position and behave correctly.

Also verify copied-importance mutation cannot alter later diagnostics or
estimator state, and predict leaves fitted estimator state unchanged. Exercise
applicable schema, numeric, and finite checks on both fit and predict. These
are synthetic unit tests, not experiments on held-out real TEST data.

## Read-only real-data experiment

During the later implementation phase, hash `data/raw/prices.parquet` and
`data/processed/features.parquet` before evaluation. Load the processed frame,
filter to the exact 15-stock universe in memory before supervised preparation,
then use existing `prepare_supervised_data` and `temporal_split` with:

```text
train_end="2021-12-31"
validation_start="2022-01-01"
validation_end="2023-12-31"
test_start="2024-01-01"
```

Keep existing target-horizon purging intact. Approximate prior audit counts are
43,906 TRAIN and 7,440 VALIDATION rows; calculate actual counts and date ranges
at runtime, never hard-code them. Check SPY is absent and all 15 intended stocks
are present in both modeling partitions.

The generic split constructs its reserved TEST partition as before. After
splitting, orchestration accesses only TRAIN and VALIDATION: no TEST features,
targets, metadata, row counts, date ranges, predictions, metrics, diagnostics,
or model-selection use. TEST remains closed for a later final modeling decision.
Do not change generic dataset functions to implement this orchestration rule.

Recompute all four models from scratch on this one split:

- Fit MeanBaseline only on TRAIN y, then predict the VALIDATION index.
- MomentumBaseline predicts from VALIDATION X without fitting.
- Refit LinearRegressionForecaster and its StandardScaler + LinearRegression
  pipeline only on TRAIN X/y; predict VALIDATION X.
- Fit the single fixed RandomForestForecaster only on TRAIN X/y; predict
  VALIDATION X and, separately, TRAIN X for its overfitting diagnostic.

All comparison predictions must have exactly the same VALIDATION indices and
observations and be evaluated against the same VALIDATION y. Reuse
`stock_forecaster.models.metrics.evaluate_predictions` for every model and
partition diagnostic; no model-specific metric is introduced.

All filtering and evaluation remain in memory. Do not write either dataset or
a generated evaluation dataset. Hash both Parquet files afterward and require
unchanged hashes; no generated dataset is committed.

## Evaluation report and interpretation

VALIDATION is the official comparison partition. Produce this primary table:

```text
model | MAE | RMSE | directional_accuracy | correlation
```

Its rows are MeanBaseline, MomentumBaseline, LinearRegression, and RandomForest,
with freshly calculated results rather than historical Step 5C values.

For RandomForest only, additionally report the same four metrics for TRAIN and
VALIDATION predictions as an overfitting diagnostic. TRAIN metrics are not used
to compare against the other models. A large TRAIN-versus-VALIDATION gap may
indicate overfitting; it does not authorize parameter changes during this step.

For all four models report VALIDATION prediction mean, standard deviation,
minimum, and maximum to detect degenerate or implausibly compressed behavior.
For the forest's VALIDATION predictions report:

```text
ticker | row_count | MAE | directional_accuracy | correlation
```

Per-ticker results diagnose the global model only; do not fit separate forests,
change parameters by ticker, or remove stocks based on these results.

Report every FEATURE_COLUMN in a `feature | importance` table sorted by
importance descending, explicitly labeled as impurity-based model diagnostics.
Sorting this presentation does not change the public diagnostic Series order.

Interpret primarily MAE, then RMSE, then Pearson correlation, with directional
accuracy as supporting context. The forest need not win all four metrics to be
potentially interesting. Discuss whether it beats MeanBaseline on MAE/RMSE,
improves prediction/target correlation or directional accuracy, shows an
overfitting gap, and assigns higher impurity importance to particular features.
Make no profitability, investability, causality, statistical-significance,
economic-significance, or TEST-performance claims.

## Fixed experiment and scope boundaries

After observing Step 5D VALIDATION metrics, do not change n_estimators,
max_depth, min_samples_leaf, max_features, or random_state, or opportunistically
tune the model. The approved configuration remains fixed. Poor performance
still ends Step 5D; only a later explicitly designed step may decide to tune
Random Forest, try boosting, add features, or change validation methodology.

Step 5D explicitly excludes GridSearchCV, RandomizedSearchCV, manual parameter
search after seeing results, Ridge, Lasso, ElasticNet, GradientBoostingRegressor,
HistGradientBoostingRegressor, XGBoost, LightGBM, CatBoost, ticker encoding,
sector encoding, new features, feature selection, permutation importance, SHAP,
partial dependence, walk-forward validation, cross-validation, TEST evaluation,
backtesting, portfolio construction, and profitability claims.

Scikit-learn is already a runtime dependency from Step 5C. Add no new runtime
dependencies, including xgboost, lightgbm, statsmodels, shap, or an explicit
scipy dependency; sklearn's existing transitive requirements need no expansion.

## Future files and documentation

The later phases are expected to create:

- `src/stock_forecaster/models/forest.py`
- `tests/test_forest.py`
- `docs/superpowers/plans/2026-09-14-random-forest.md`

Only README.md is expected to need modification during model implementation;
no dependency file modification should be necessary unless a genuine existing
environment issue is discovered. Keep production dataset code and market
configuration unchanged.

The future README progression becomes Mean/Momentum baselines → Linear
Regression → Random Forest → future models. Document Random Forest only after
implementation, using actual results when available, not invented future
numbers. None of these future files or README changes belongs to this
specification-only task.

## Design self-review

The specification was reviewed against the approved design: estimator
parameters are identical throughout; the forest is unscaled and global;
FEATURE_COLUMNS is the sole ordered predictor schema; ticker is metadata and
SPY remains benchmark-only; TEST remains closed; TRAIN diagnostics do not
replace official VALIDATION comparison; no post-VALIDATION tuning is allowed;
importances are non-causal diagnostics; no dependency expansion or unnecessary
abstraction is introduced. Numeric/finite checks, exact X/y index alignment,
input immutability, prediction Series naming/index preservation, safe importance
exposure, and same-environment reproducibility are explicit. All 36 required
future test behaviors are documented, with no incomplete sections or unresolved
architectural choices.
