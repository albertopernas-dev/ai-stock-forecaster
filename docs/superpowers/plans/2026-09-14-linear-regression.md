# Standardized Linear Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first real ML forecaster using a global
`StandardScaler` -> `LinearRegression` pipeline on the 15-stock universe.

**Architecture:** Implement one focused `LinearRegressionForecaster` that
accepts exactly `FEATURE_COLUMNS`, keeps preprocessing and regression coupled
inside one sklearn `Pipeline`, and exposes aligned predictions plus standardized
coefficient diagnostics. Fit the model and both fitted pipeline stages on TRAIN
only, evaluate all three models on the same VALIDATION rows after orchestration
excludes SPY, and leave TEST untouched.

**Tech Stack:** Python 3.12, pandas, scikit-learn, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-14-linear-regression-design.md`

## Global Constraints

- Use exactly the ten predictors in `FEATURE_COLUMNS`, in their declared order:
  `return_1d`, `return_5d`, `return_20d`, `volatility_5d`,
  `volatility_20d`, `distance_sma_10`, `distance_sma_20`,
  `distance_sma_50`, `volume_change_1d`, and `volume_ratio_20`.
- Train one global model across exactly AAPL, MSFT, GOOGL, AMZN, META, NVDA,
  JPM, V, JNJ, UNH, XOM, CVX, COST, WMT, and PG.
- Exclude SPY only in the Step 5C modeling/evaluation orchestration. Keep SPY in
  RAW data, PROCESSED data, and `configs/market.yaml`.
- Keep `ticker` as metadata only. Do not encode ticker or include `date`,
  `ticker`, `target_end_date`, or `future_return_5d` in model inputs.
- Use one sklearn `Pipeline` containing `StandardScaler` followed by
  `LinearRegression`; do not manually scale data outside the pipeline.
- Fit preprocessing and regression on TRAIN only. Predict and evaluate on
  VALIDATION only. Never access or evaluate TEST after constructing the split.
- Reuse `evaluate_predictions`; do not add model-specific metrics.
- Recompute `MeanBaseline` and `MomentumBaseline` on the same 15-stock
  TRAIN/VALIDATION experiment rather than reusing Step 5B's 16-ticker results.
- Preserve exact X/y and prediction index alignment. Do not sort, reset,
  reindex, coerce, or silently realign model inputs.
- Reject missing, extra, reordered, non-numeric, null, infinite, length-
  mismatched, and index-mismatched inputs before sklearn receives them.
- Expose coefficients learned for standardized predictors and a finite
  intercept; describe coefficients as conditional linear diagnostics, not
  causal effects or economic significance.
- Make no profitability or investability claims. Perform no backtesting,
  portfolio construction, tuning, statistical-significance testing, or TEST
  analysis.
- Add only unpinned `scikit-learn`, matching the existing dependency-version
  policy. Do not add unrelated libraries.
- Do not introduce an abstract estimator base class, generic experiment
  framework, model registry, configuration system, or CLI framework.
- Do not modify generated datasets or market configuration, commit Parquet
  files, amend commits, or push.

## File Map

- Create `docs/superpowers/plans/2026-09-14-linear-regression.md`: this approved
  implementation plan, committed before any dependency, test, or production
  change.
- Create `src/stock_forecaster/models/linear.py`: the single global linear
  forecaster, validation helpers, prediction alignment, and fitted diagnostics.
- Create `tests/test_linear.py`: synthetic TDD coverage for all public contracts,
  pipeline structure, deterministic recovery, and scaling leakage protection.
- Modify `pyproject.toml`: add only unpinned `scikit-learn` to runtime
  dependencies.
- Modify `README.md`: describe the implemented model progression and Step 5C's
  universe and leakage boundaries without publishing results that do not yet
  exist.
- Create no persistent evaluation script or generated output. The read-only
  real-data evaluation runs as an inline Python command after the source commit.

---

### Task 1: Add the one approved runtime dependency

**Files:**
- Modify: `pyproject.toml` (`[project].dependencies`)

**Interfaces:**
- Consumes: the repository's existing unpinned runtime dependency list.
- Produces: importable `sklearn.pipeline.Pipeline`,
  `sklearn.preprocessing.StandardScaler`, and
  `sklearn.linear_model.LinearRegression` under Python 3.12.

- [ ] **Step 1: Add only scikit-learn to runtime dependencies**

  Keep the list unpinned and place the new dependency with the other runtime
  libraries:

  ```toml
  dependencies = [
      "pandas",
      "pyarrow",
      "PyYAML",
      "scikit-learn",
      "truststore",
      "yfinance",
  ]
  ```

- [ ] **Step 2: Install the updated project and development dependencies**

  Run:

  ```powershell
  python -m pip install -e ".[dev]"
  ```

  Expected: installation succeeds without adding another dependency entry to
  `pyproject.toml`.

- [ ] **Step 3: Verify the approved dependency imports**

  Run:

  ```powershell
  python -c "from sklearn.linear_model import LinearRegression; from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler; print('sklearn imports OK')"
  ```

  Expected: exit code 0 and `sklearn imports OK`.

### Task 2: Drive pipeline structure and fitted-state behavior with TDD

**Files:**
- Create: `tests/test_linear.py`
- Create: `src/stock_forecaster/models/linear.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, numeric `pd.DataFrame` inputs, and numeric aligned
  `pd.Series` targets.
- Produces: `LinearRegressionForecaster.fit(X, y) ->
  LinearRegressionForecaster`, `predict(X) -> pd.Series`,
  `coefficients_ -> pd.Series`, and `intercept_ -> float`.

- [ ] **Step 1: Create shared synthetic fixtures and structural tests**

  Start `tests/test_linear.py` with deterministic input builders and the initial
  structural and fitted-state checks:

  ```python
  import math

  import pandas as pd
  import pytest
  from sklearn.linear_model import LinearRegression
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler

  from stock_forecaster.features.engineering import FEATURE_COLUMNS
  from stock_forecaster.models.linear import LinearRegressionForecaster


  def _numeric_X(
      *,
      rows: int = 6,
      index: pd.Index | None = None,
  ) -> pd.DataFrame:
      if index is None:
          index = pd.Index(range(rows), name="observation")
      return pd.DataFrame(
          {
              feature: [
                  float((row + 1) * (feature_number + 2)) / 100.0
                  for row in range(rows)
              ]
              for feature_number, feature in enumerate(FEATURE_COLUMNS)
          },
          index=index,
      )


  def _numeric_y(index: pd.Index) -> pd.Series:
      return pd.Series(
          [float(row - 2) / 100.0 for row in range(len(index))],
          index=index,
          name="future_return_5d",
      )


  def _fitted_model(rows: int = 6) -> tuple[
      LinearRegressionForecaster,
      pd.DataFrame,
      pd.Series,
  ]:
      X = _numeric_X(rows=rows)
      y = _numeric_y(X.index)
      return LinearRegressionForecaster().fit(X, y), X, y


  def test_model_uses_the_approved_sklearn_pipeline():
      model = LinearRegressionForecaster()

      assert isinstance(model._pipeline, Pipeline)
      assert list(model._pipeline.named_steps) == ["scaler", "regressor"]
      assert isinstance(model._pipeline.named_steps["scaler"], StandardScaler)
      assert isinstance(
          model._pipeline.named_steps["regressor"], LinearRegression
      )


  def test_fit_accepts_exact_feature_schema_and_returns_self():
      X = _numeric_X()
      y = _numeric_y(X.index)
      model = LinearRegressionForecaster()

      returned = model.fit(X, y)

      assert returned is model


  def test_predict_before_fit_is_rejected_clearly():
      with pytest.raises(RuntimeError, match="must be fitted"):
          LinearRegressionForecaster().predict(_numeric_X())


  def test_coefficients_before_fit_are_rejected_clearly():
      with pytest.raises(RuntimeError, match="must be fitted"):
          LinearRegressionForecaster().coefficients_


  def test_intercept_before_fit_is_rejected_clearly():
      with pytest.raises(RuntimeError, match="must be fitted"):
          LinearRegressionForecaster().intercept_
  ```

  The `_pipeline` assertion intentionally verifies the approved internal
  estimator without adding another public API. The chosen pre-fit behavior for
  `predict`, `coefficients_`, and `intercept_` is a clear `RuntimeError` with a
  `must be fitted` message.

- [ ] **Step 2: Observe the first RED state**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: collection fails with `ModuleNotFoundError` for
  `stock_forecaster.models.linear`.

- [ ] **Step 3: Add the minimal pipeline shell and fitted-state guard**

  Create `src/stock_forecaster/models/linear.py` with this initial structure:

  ```python
  """Global standardized linear regression forecaster."""

  import math

  import pandas as pd
  from pandas.api.types import is_numeric_dtype
  from sklearn.linear_model import LinearRegression
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler

  from stock_forecaster.features.engineering import FEATURE_COLUMNS


  class LinearRegressionForecaster:
      """Fit one standardized linear model across the stock universe."""

      def __init__(self) -> None:
          self._pipeline = Pipeline(
              [
                  ("scaler", StandardScaler()),
                  ("regressor", LinearRegression()),
              ]
          )
          self._is_fitted = False

      def _require_fitted(self) -> None:
          if not self._is_fitted:
              raise RuntimeError(
                  "LinearRegressionForecaster must be fitted before use"
              )

      def fit(
          self,
          X: pd.DataFrame,
          y: pd.Series,
      ) -> "LinearRegressionForecaster":
          self._pipeline.fit(X, y)
          self._is_fitted = True
          return self

      def predict(self, X: pd.DataFrame) -> pd.Series:
          self._require_fitted()
          values = self._pipeline.predict(X)
          return pd.Series(values)

      @property
      def coefficients_(self) -> pd.Series:
          self._require_fitted()
          regressor = self._pipeline.named_steps["regressor"]
          return pd.Series(regressor.coef_)

      @property
      def intercept_(self) -> float:
          self._require_fitted()
          regressor = self._pipeline.named_steps["regressor"]
          return regressor.intercept_
  ```

  This is only the minimum GREEN for pipeline construction and fitted-state
  behavior. Prediction alignment and diagnostic labeling are deliberately not
  asserted yet; their tests precede their completed implementation in Task 4.

- [ ] **Step 4: Verify the first GREEN state**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: all tests currently present in `tests/test_linear.py` pass.

### Task 3: Enforce schema, numeric, finite, alignment, and immutability contracts

**Files:**
- Modify: `tests/test_linear.py`
- Modify: `src/stock_forecaster/models/linear.py`

**Interfaces:**
- Consumes: X with exactly `list(FEATURE_COLUMNS)` and y with the exact same
  index.
- Produces: validation failures as clear `ValueError` instances before
  `Pipeline.fit` or `Pipeline.predict` receives invalid input.

- [ ] **Step 1: Add exact-schema rejection tests**

  Append these tests to `tests/test_linear.py`:

  ```python
  def test_missing_feature_is_rejected():
      X = _numeric_X().drop(columns=[FEATURE_COLUMNS[-1]])

      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))


  def test_additional_feature_is_rejected():
      X = _numeric_X().assign(unapproved_feature=1.0)

      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))


  def test_reordered_features_are_rejected_without_silent_reordering():
      X = _numeric_X().loc[:, list(reversed(FEATURE_COLUMNS))]

      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))
  ```

- [ ] **Step 2: Add numeric and finite-value rejection tests**

  Append explicit tests for both signs of infinity and both model inputs:

  ```python
  def test_non_numeric_feature_is_rejected_before_sklearn():
      X = _numeric_X()
      X[FEATURE_COLUMNS[3]] = ["bad"] * len(X)

      with pytest.raises(ValueError, match="feature columns must be numeric"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))


  def test_non_numeric_target_is_rejected_before_sklearn():
      X = _numeric_X()
      y = pd.Series(["bad"] * len(X), index=X.index)

      with pytest.raises(ValueError, match="y must have a numeric"):
          LinearRegressionForecaster().fit(X, y)


  def test_feature_nan_is_rejected():
      X = _numeric_X()
      X.loc[X.index[0], FEATURE_COLUMNS[0]] = float("nan")

      with pytest.raises(ValueError, match="X must not contain null"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))


  def test_target_nan_is_rejected():
      X = _numeric_X()
      y = _numeric_y(X.index)
      y.iloc[0] = float("nan")

      with pytest.raises(ValueError, match="y must not contain null"):
          LinearRegressionForecaster().fit(X, y)


  @pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
  def test_feature_positive_or_negative_infinity_is_rejected(invalid):
      X = _numeric_X()
      X.loc[X.index[0], FEATURE_COLUMNS[0]] = invalid

      with pytest.raises(ValueError, match="X must contain only finite"):
          LinearRegressionForecaster().fit(X, _numeric_y(X.index))


  @pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
  def test_target_positive_or_negative_infinity_is_rejected(invalid):
      X = _numeric_X()
      y = _numeric_y(X.index)
      y.iloc[0] = invalid

      with pytest.raises(ValueError, match="y must contain only finite"):
          LinearRegressionForecaster().fit(X, y)
  ```

- [ ] **Step 3: Add length, exact-index, and fit immutability tests**

  Append:

  ```python
  def test_x_y_length_mismatch_is_rejected():
      X = _numeric_X()
      y = _numeric_y(X.index[:-1])

      with pytest.raises(ValueError, match="equal lengths"):
          LinearRegressionForecaster().fit(X, y)


  def test_equal_length_x_y_index_mismatch_is_rejected():
      index = pd.Index([8, 2, 5, 9, 1, 7], name="observation")
      X = _numeric_X(index=index)
      y = _numeric_y(index[::-1])
      assert len(X) == len(y)
      assert not X.index.equals(y.index)

      with pytest.raises(ValueError, match="indices must match exactly"):
          LinearRegressionForecaster().fit(X, y)


  def test_fit_does_not_mutate_x():
      X = _numeric_X()
      y = _numeric_y(X.index)
      original = X.copy(deep=True)

      LinearRegressionForecaster().fit(X, y)

      pd.testing.assert_frame_equal(X, original)


  def test_fit_does_not_mutate_y():
      X = _numeric_X()
      y = _numeric_y(X.index)
      original = y.copy(deep=True)

      LinearRegressionForecaster().fit(X, y)

      pd.testing.assert_series_equal(y, original)
  ```

- [ ] **Step 4: Observe validation RED**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: the new invalid-input tests fail because the pipeline shell does not
  yet enforce the model's contracts; immutability tests may already pass.

- [ ] **Step 5: Implement reusable validation before sklearn calls**

  Add these private helpers above the class in
  `src/stock_forecaster/models/linear.py`:

  ```python
  def _validate_X(X: pd.DataFrame) -> None:
      if list(X.columns) != list(FEATURE_COLUMNS):
          raise ValueError(
              "X columns must be exactly FEATURE_COLUMNS in declared order"
          )

      non_numeric = [
          column for column in FEATURE_COLUMNS
          if not is_numeric_dtype(X[column].dtype)
      ]
      if non_numeric:
          raise ValueError(
              "X feature columns must be numeric pandas dtypes: "
              + ", ".join(non_numeric)
          )
      if X.isna().any().any():
          raise ValueError("X must not contain null values")
      if not all(
          math.isfinite(value)
          for row in X.itertuples(index=False, name=None)
          for value in row
      ):
          raise ValueError("X must contain only finite values")


  def _validate_y(y: pd.Series) -> None:
      if not is_numeric_dtype(y.dtype):
          raise ValueError("y must have a numeric pandas dtype")
      if y.isna().any():
          raise ValueError("y must not contain null values")
      if not all(math.isfinite(value) for value in y):
          raise ValueError("y must contain only finite values")
  ```

  Then validate before fitting and predicting:

  ```python
      def fit(
          self,
          X: pd.DataFrame,
          y: pd.Series,
      ) -> "LinearRegressionForecaster":
          _validate_X(X)
          _validate_y(y)
          if len(X) != len(y):
              raise ValueError("X and y must have equal lengths")
          if not X.index.equals(y.index):
              raise ValueError("X and y indices must match exactly")

          self._pipeline.fit(X, y)
          self._is_fitted = True
          return self

      def predict(self, X: pd.DataFrame) -> pd.Series:
          self._require_fitted()
          _validate_X(X)
          values = self._pipeline.predict(X)
          return pd.Series(values)
  ```

  Do not copy, sort, reset, select, reindex, or coerce inputs. Exact schema and
  values are validated in place, while sklearn reads them without mutation.

- [ ] **Step 6: Verify validation GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: every structural, schema, dtype, finite-value, alignment, and
  immutability test added through Task 3 passes.

### Task 4: Complete prediction, diagnostics, recovery, and leakage tests

**Files:**
- Modify: `tests/test_linear.py`
- Modify: `src/stock_forecaster/models/linear.py`

**Interfaces:**
- Consumes: a successfully fitted pipeline and exact-schema validation X.
- Produces: a `prediction` Series with X's exact index, copied standardized
  coefficients in feature order, a finite float intercept, and stable TRAIN-only
  scaler state.

- [ ] **Step 1: Add prediction output and prediction immutability tests**

  Append:

  ```python
  def test_prediction_contract_preserves_type_name_index_and_count():
      model, _, _ = _fitted_model()
      index = pd.Index([90, 4, 17], name="validation_row")
      validation_X = _numeric_X(rows=3, index=index)

      predictions = model.predict(validation_X)

      assert isinstance(predictions, pd.Series)
      assert predictions.name == "prediction"
      assert predictions.index.equals(validation_X.index)
      assert len(predictions) == len(validation_X)


  def test_predict_does_not_mutate_x():
      model, _, _ = _fitted_model()
      validation_X = _numeric_X(rows=3, index=pd.Index([5, 1, 8]))
      original = validation_X.copy(deep=True)

      model.predict(validation_X)

      pd.testing.assert_frame_equal(validation_X, original)


  def test_predict_rejects_invalid_schema_and_values_before_sklearn():
      model, _, _ = _fitted_model()
      validation_X = _numeric_X().drop(columns=[FEATURE_COLUMNS[0]])

      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          model.predict(validation_X)
  ```

  The fit-side parameterized cases already cover both signs of infinity. Add a
  predict-side null/finite check so validation input cannot bypass the same X
  contract:

  ```python
  @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
  def test_predict_rejects_non_finite_features(invalid):
      model, _, _ = _fitted_model()
      validation_X = _numeric_X()
      validation_X.loc[validation_X.index[0], FEATURE_COLUMNS[0]] = invalid

      with pytest.raises(ValueError):
          model.predict(validation_X)
  ```

- [ ] **Step 2: Add coefficient and intercept contract tests**

  Append:

  ```python
  def test_coefficients_and_intercept_have_exact_public_contracts():
      model, _, _ = _fitted_model()

      coefficients = model.coefficients_
      intercept = model.intercept_

      assert isinstance(coefficients, pd.Series)
      assert coefficients.index.equals(pd.Index(FEATURE_COLUMNS))
      assert coefficients.name == "coefficient"
      assert isinstance(intercept, float)
      assert math.isfinite(intercept)


  def test_coefficients_are_a_copy_not_mutable_estimator_state():
      model, _, _ = _fitted_model()
      original = model.coefficients_

      changed = model.coefficients_
      changed.iloc[0] = 999.0

      pd.testing.assert_series_equal(model.coefficients_, original)
  ```

- [ ] **Step 3: Add deterministic whole-pipeline recovery test**

  Build a target from a literal linear combination in the original feature
  units, then verify predictions rather than comparing standardized coefficient
  values to the original-unit weights:

  ```python
  def test_pipeline_recovers_a_deterministic_linear_relationship():
      rows = 40
      index = pd.Index(range(100, 100 + rows), name="observation")
      X = pd.DataFrame(
          {
              feature: [
                  float(((row + 3) ** (feature_number + 1)) % 101) / 100.0
                  for row in range(rows)
              ]
              for feature_number, feature in enumerate(FEATURE_COLUMNS)
          },
          index=index,
      )
      weights = pd.Series(
          [0.4, -0.3, 0.2, -0.1, 0.08, -0.06, 0.04, -0.03, 0.02, -0.01],
          index=FEATURE_COLUMNS,
      )
      y = X.mul(weights).sum(axis=1).add(0.007).rename("future_return_5d")

      predictions = LinearRegressionForecaster().fit(X, y).predict(X)

      pd.testing.assert_series_equal(
          predictions,
          y.rename("prediction"),
          check_exact=False,
          atol=1e-10,
          rtol=1e-10,
      )
  ```

  This assertion remains conceptually correct with `StandardScaler`: scaling
  changes fitted coefficient units, but an unregularized linear regression with
  an intercept still reproduces a deterministic linear target through the full
  transform-and-predict flow.

- [ ] **Step 4: Add TRAIN-only scaler and no-refit-on-predict tests**

  Append one explicit leakage regression test that records both `mean_` and
  `scale_` before extreme VALIDATION prediction:

  ```python
  def test_scaler_statistics_come_only_from_train_and_prediction_does_not_refit():
      train_index = pd.Index([10, 20, 30, 40, 50, 60], name="train_row")
      train_X = _numeric_X(index=train_index)
      train_y = _numeric_y(train_X.index)
      model = LinearRegressionForecaster().fit(train_X, train_y)
      scaler = model._pipeline.named_steps["scaler"]
      expected_mean = train_X.mean().to_numpy()
      expected_scale = train_X.std(ddof=0).to_numpy()
      mean_before = scaler.mean_.copy()
      scale_before = scaler.scale_.copy()

      validation_X = _numeric_X(
          rows=4,
          index=pd.Index([900, 100, 700, 300], name="validation_row"),
      ).add(1_000_000.0)
      predictions = model.predict(validation_X)

      assert scaler.mean_ == pytest.approx(expected_mean)
      assert scaler.scale_ == pytest.approx(expected_scale)
      assert scaler.mean_ == pytest.approx(mean_before)
      assert scaler.scale_ == pytest.approx(scale_before)
      assert predictions.index.equals(validation_X.index)
  ```

- [ ] **Step 5: Add explicit multi-row prediction test**

  Append:

  ```python
  def test_multiple_rows_produce_one_finite_prediction_per_row():
      model, _, _ = _fitted_model(rows=12)
      validation_X = _numeric_X(
          rows=5,
          index=pd.Index([42, 7, 81, 3, 19], name="validation_row"),
      )

      predictions = model.predict(validation_X)

      assert len(predictions) == 5
      assert all(math.isfinite(value) for value in predictions)
      assert predictions.index.equals(validation_X.index)
  ```

- [ ] **Step 6: Observe completion-test RED where applicable**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: prediction name/index, coefficient index, and deterministic recovery
  assertions fail because the current minimal Series construction discards
  labels. The leakage-state assertions may already pass because sklearn
  `Pipeline.predict` does not fit; retain them as regression coverage. Do not
  weaken assertions to accommodate implementation details.

- [ ] **Step 7: Complete the minimal forecaster implementation**

  Make `src/stock_forecaster/models/linear.py` exactly this focused module:

  ```python
  """Global standardized linear regression forecaster."""

  import math

  import pandas as pd
  from pandas.api.types import is_numeric_dtype
  from sklearn.linear_model import LinearRegression
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler

  from stock_forecaster.features.engineering import FEATURE_COLUMNS


  def _validate_X(X: pd.DataFrame) -> None:
      if list(X.columns) != list(FEATURE_COLUMNS):
          raise ValueError(
              "X columns must be exactly FEATURE_COLUMNS in declared order"
          )

      non_numeric = [
          column
          for column in FEATURE_COLUMNS
          if not is_numeric_dtype(X[column].dtype)
      ]
      if non_numeric:
          raise ValueError(
              "X feature columns must be numeric pandas dtypes: "
              + ", ".join(non_numeric)
          )
      if X.isna().any().any():
          raise ValueError("X must not contain null values")
      if not all(
          math.isfinite(value)
          for row in X.itertuples(index=False, name=None)
          for value in row
      ):
          raise ValueError("X must contain only finite values")


  def _validate_y(y: pd.Series) -> None:
      if not is_numeric_dtype(y.dtype):
          raise ValueError("y must have a numeric pandas dtype")
      if y.isna().any():
          raise ValueError("y must not contain null values")
      if not all(math.isfinite(value) for value in y):
          raise ValueError("y must contain only finite values")


  class LinearRegressionForecaster:
      """Fit one standardized linear model across the stock universe."""

      def __init__(self) -> None:
          self._pipeline = Pipeline(
              [
                  ("scaler", StandardScaler()),
                  ("regressor", LinearRegression()),
              ]
          )
          self._is_fitted = False

      def _require_fitted(self) -> None:
          if not self._is_fitted:
              raise RuntimeError(
                  "LinearRegressionForecaster must be fitted before use"
              )

      def fit(
          self,
          X: pd.DataFrame,
          y: pd.Series,
      ) -> "LinearRegressionForecaster":
          _validate_X(X)
          _validate_y(y)
          if len(X) != len(y):
              raise ValueError("X and y must have equal lengths")
          if not X.index.equals(y.index):
              raise ValueError("X and y indices must match exactly")

          self._pipeline.fit(X, y)
          self._is_fitted = True
          return self

      def predict(self, X: pd.DataFrame) -> pd.Series:
          self._require_fitted()
          _validate_X(X)
          values = self._pipeline.predict(X)
          return pd.Series(values, index=X.index, name="prediction")

      @property
      def coefficients_(self) -> pd.Series:
          self._require_fitted()
          regressor = self._pipeline.named_steps["regressor"]
          if not all(math.isfinite(value) for value in regressor.coef_):
              raise RuntimeError(
                  "Fitted linear regression diagnostics must be finite"
              )
          return pd.Series(
              regressor.coef_,
              index=FEATURE_COLUMNS,
              name="coefficient",
              dtype=float,
          ).copy()

      @property
      def intercept_(self) -> float:
          self._require_fitted()
          regressor = self._pipeline.named_steps["regressor"]
          value = float(regressor.intercept_)
          if not math.isfinite(value):
              raise RuntimeError(
                  "Fitted linear regression diagnostics must be finite"
              )
          return value
  ```

  Return a newly constructed coefficient Series each time. Do not expose
  transformed arrays, add public tuning methods, or export metadata handling
  from this module.

- [ ] **Step 8: Verify the focused suite is GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_linear.py -v
  ```

  Expected: all 30 required behaviors are covered, including parameterized
  positive/negative infinity cases, and all collected cases pass.

  Coverage mapping:

  1. sklearn `Pipeline`: `test_model_uses_the_approved_sklearn_pipeline`
  2. scaler then regressor: same structural test and exact named-step order
  3. fit returns self: `test_fit_accepts_exact_feature_schema_and_returns_self`
  4-6. pre-fit behavior: the three explicit `RuntimeError` tests
  7-10. exact/missing/extra/reordered schema: fit acceptance plus three rejects
  11-16. non-numeric, NaN, and both signs of infinity: explicit and
  parameterized X/y tests
  17-18. length and exact index: two distinct rejection tests
  19-21. fit/predict immutability: three pandas equality tests
  22-24 and 30. Series type, name, index, count, and multi-row behavior:
  prediction-contract and multi-row tests
  25-26. coefficient index and finite intercept: diagnostics contract test
  27. known linear recovery: deterministic whole-pipeline test
  28-29. TRAIN-only statistics and no refit: leakage regression test

### Task 5: Document, verify, and create the single source commit

**Files:**
- Modify: `README.md`
- Verify: `pyproject.toml`
- Verify: `src/stock_forecaster/models/linear.py`
- Verify: `tests/test_linear.py`

**Interfaces:**
- Consumes: the completed forecaster and synthetic test evidence.
- Produces: accurate current-status documentation and one atomic implementation
  commit, `feat: add standardized linear regression forecaster`.

- [ ] **Step 1: Update the model progression in README**

  Replace the future-model end of the progression with:

  ```text
  Simple baselines
         ↓
  Global standardized linear regression
         ↓
  Future nonlinear models
  ```

  Add concise prose stating:

  ```markdown
  The first trainable model is one global linear regression over 15 stocks.
  SPY remains available as a benchmark but is excluded from model fitting and
  validation comparison. The model consumes exactly the ten engineered
  `FEATURE_COLUMNS` and does not encode ticker metadata. A single sklearn
  pipeline fits `StandardScaler` and `LinearRegression` on TRAIN only. Model
  comparison uses VALIDATION only; TEST remains untouched.
  ```

  Update Current Status to say the standardized linear forecaster is
  implemented, but do not publish evaluation values before the real-data run
  and do not describe the model as profitable.

- [ ] **Step 2: Verify the focused implementation and public import**

  Run in this order:

  ```powershell
  python -m pytest tests/test_linear.py -v
  python -c "from stock_forecaster.models.linear import LinearRegressionForecaster; print(LinearRegressionForecaster.__name__)"
  ```

  Expected: the focused suite passes and the import prints
  `LinearRegressionForecaster`.

- [ ] **Step 3: Verify the complete repository**

  Run:

  ```powershell
  python -m pytest -v
  python -m ruff check .
  ```

  Expected: the full suite reports zero failures and Ruff reports no errors.
  Stop and diagnose any failure before staging files.

- [ ] **Step 4: Confirm the implementation diff has exact scope**

  Run:

  ```powershell
  git status --short
  git diff --check
  git diff --name-only
  git check-ignore -v data/raw/prices.parquet data/processed/features.parquet
  ```

  Expected modified/untracked paths are exactly:

  ```text
  README.md
  pyproject.toml
  src/stock_forecaster/models/linear.py
  tests/test_linear.py
  ```

  Both Parquet paths must be reported as ignored. Neither dataset nor
  `configs/market.yaml` may appear in the diff.

- [ ] **Step 5: Stage only the four approved implementation files**

  Run:

  ```powershell
  git add -- pyproject.toml README.md src/stock_forecaster/models/linear.py tests/test_linear.py
  git diff --cached --check
  git diff --cached --name-only
  ```

  Expected: exactly those four paths are staged with no whitespace errors.

- [ ] **Step 6: Create the one approved source commit**

  Run:

  ```powershell
  git commit -m "feat: add standardized linear regression forecaster"
  ```

  Do not amend and do not push.

### Task 6: Run the read-only 15-stock VALIDATION evaluation and safety audit

**Files:**
- Read only: `data/processed/features.parquet`
- Read only: `data/raw/prices.parquet`
- Read only: `configs/market.yaml`
- Create: no files

**Interfaces:**
- Consumes: the persisted processed features, existing supervised split API,
  both existing baselines, `evaluate_predictions`, and the new forecaster.
- Produces: console-only TRAIN/VALIDATION universe counts, comparable validation
  metrics and distributions, standardized coefficient diagnostics, and
  per-ticker diagnostics. It produces no TEST output and no dataset changes.

- [ ] **Step 1: Record dataset hashes before evaluation**

  Run in PowerShell and retain both values for the final comparison:

  ```powershell
  Get-FileHash data/raw/prices.parquet -Algorithm SHA256
  Get-FileHash data/processed/features.parquet -Algorithm SHA256
  ```

- [ ] **Step 2: Run one read-only evaluation command**

  Run this code in an interactive Python session or as an inline Python command;
  do not save it as a repository file:

  ```python
  from dataclasses import asdict

  import pandas as pd

  from stock_forecaster.models.baseline import MeanBaseline, MomentumBaseline
  from stock_forecaster.models.dataset import (
      prepare_supervised_data,
      temporal_split,
  )
  from stock_forecaster.models.linear import LinearRegressionForecaster
  from stock_forecaster.models.metrics import evaluate_predictions

  MODEL_TICKERS = {
      "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
      "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG",
  }

  persisted = pd.read_parquet("data/processed/features.parquet")
  assert "SPY" in set(persisted["ticker"])
  modeling = persisted.loc[persisted["ticker"].ne("SPY")].copy()
  assert set(modeling["ticker"]) == MODEL_TICKERS

  supervised = prepare_supervised_data(modeling)
  split = temporal_split(
      supervised,
      train_end="2021-12-31",
      validation_start="2022-01-01",
      validation_end="2023-12-31",
      test_start="2024-01-01",
  )
  train = split.train
  validation = split.validation
  del split

  assert "SPY" not in set(train.metadata["ticker"])
  assert "SPY" not in set(validation.metadata["ticker"])
  assert set(train.metadata["ticker"]) == MODEL_TICKERS
  assert set(validation.metadata["ticker"]) == MODEL_TICKERS
  assert train.X.index.equals(train.y.index)
  assert validation.X.index.equals(validation.y.index)
  assert validation.X.index.equals(validation.metadata.index)

  print("TRAIN rows:", len(train.X))
  print("VALIDATION rows:", len(validation.X))
  print("TRAIN dates:", train.metadata["date"].min(), train.metadata["date"].max())
  print(
      "VALIDATION dates:",
      validation.metadata["date"].min(),
      validation.metadata["date"].max(),
  )
  print("TRAIN tickers:", sorted(train.metadata["ticker"].unique()))
  print(
      "VALIDATION tickers:",
      sorted(validation.metadata["ticker"].unique()),
  )

  mean_predictions = (
      MeanBaseline()
      .fit(train.y)
      .predict(validation.y.index)
  )
  momentum_predictions = MomentumBaseline().predict(validation.X)
  linear_model = LinearRegressionForecaster().fit(train.X, train.y)
  linear_predictions = linear_model.predict(validation.X)

  predictions_by_model = {
      "MeanBaseline": mean_predictions,
      "MomentumBaseline": momentum_predictions,
      "LinearRegression": linear_predictions,
  }
  for predictions in predictions_by_model.values():
      assert predictions.index.equals(validation.y.index)
      assert len(predictions) == len(validation.y)

  comparison_rows = []
  for model_name, predictions in predictions_by_model.items():
      metrics = evaluate_predictions(validation.y, predictions)
      comparison_rows.append({"model": model_name, **asdict(metrics)})
  comparison = pd.DataFrame(comparison_rows).rename(
      columns={"mae": "MAE", "rmse": "RMSE"}
  ).loc[:, ["model", "MAE", "RMSE", "directional_accuracy", "correlation"]]
  print("\nComparable VALIDATION metrics")
  print(comparison.to_string(index=False))

  distribution = pd.DataFrame(
      [
          {
              "model": model_name,
              "prediction_mean": float(predictions.mean()),
              "prediction_standard_deviation": float(predictions.std()),
              "prediction_minimum": float(predictions.min()),
              "prediction_maximum": float(predictions.max()),
          }
          for model_name, predictions in predictions_by_model.items()
      ]
  )
  print("\nPrediction distributions")
  print(distribution.to_string(index=False))

  coefficient_table = (
      linear_model.coefficients_
      .rename_axis("feature")
      .reset_index(name="coefficient")
  )
  coefficient_table["absolute_coefficient"] = coefficient_table[
      "coefficient"
  ].abs()
  coefficient_table = coefficient_table.sort_values(
      "absolute_coefficient", ascending=False
  )
  print("\nLinearRegression intercept:", linear_model.intercept_)
  print("\nStandardized conditional linear coefficients")
  print(coefficient_table.to_string(index=False))
  print(
      "Coefficient signs and magnitudes are conditional linear diagnostics, "
      "not causal effects or economic significance."
  )

  per_ticker_rows = []
  for ticker, metadata_rows in validation.metadata.groupby("ticker", sort=True):
      row_index = metadata_rows.index
      ticker_metrics = evaluate_predictions(
          validation.y.loc[row_index],
          linear_predictions.loc[row_index],
      )
      per_ticker_rows.append(
          {
              "ticker": ticker,
              "row_count": len(row_index),
              "MAE": ticker_metrics.mae,
              "directional_accuracy": ticker_metrics.directional_accuracy,
              "correlation": ticker_metrics.correlation,
          }
      )
  per_ticker = pd.DataFrame(per_ticker_rows)
  print("\nLinearRegression per-ticker VALIDATION diagnostics")
  print(per_ticker.to_string(index=False))
  ```

  Critical isolation rule: after `temporal_split` returns, this code binds only
  `split.train` and `split.validation`, deletes the split container, and never
  reads `split.test`, predicts TEST, calculates TEST metrics, or uses TEST for a
  decision.

- [ ] **Step 3: Audit actual universe and comparison evidence**

  Confirm the command reports runtime counts rather than enforcing them. The
  expected audit values are approximately 43,906 TRAIN rows and 7,440
  VALIDATION rows. Confirm all three models used the identical validation index
  and observations, the mean baseline fitted only `train.y`, momentum consumed
  only `validation.X`, and linear regression fitted only `train.X/train.y`.

  Interpret only whether LinearRegression beats MeanBaseline on VALIDATION MAE
  or RMSE, whether directional accuracy improves, whether target correlation is
  positive/negative/near zero, and which standardized coefficients have the
  largest absolute magnitude. Do not claim profitability, investability,
  causality, economic significance, backtest performance, or TEST performance.

- [ ] **Step 4: Prove dataset and repository safety after evaluation**

  Run:

  ```powershell
  Get-FileHash data/raw/prices.parquet -Algorithm SHA256
  Get-FileHash data/processed/features.parquet -Algorithm SHA256
  git check-ignore -v data/raw/prices.parquet data/processed/features.parquet
  git status --short
  git diff --exit-code
  git diff --cached --exit-code
  git log --oneline -5
  ```

  Expected: both hashes exactly match Step 1, both Parquet paths remain ignored,
  status is empty, both diff commands exit 0, and the latest commit is
  `feat: add standardized linear regression forecaster`. Do not push.

## Plan Self-Review Checklist

- [ ] Every section of the approved design maps to a concrete task above.
- [ ] Every step contains concrete executable guidance and every referenced
  production interface is defined.
- [ ] `fit`, `predict`, `coefficients_`, and `intercept_` names and types are
  consistent across tests, implementation guidance, documentation, and audit.
- [ ] Orchestration filters SPY before supervised preparation while preserving
  SPY in persisted datasets and market configuration.
- [ ] No step accesses `split.test` after split construction or produces TEST
  predictions, metrics, or decisions.
- [ ] X contains only exact ordered `FEATURE_COLUMNS`; metadata and target never
  enter the estimator.
- [ ] Exact X/y index equality, numeric dtype, null/infinity, length, schema,
  immutability, prediction name, and prediction index contracts are explicit.
- [ ] Synthetic leakage coverage proves scaler statistics come only from TRAIN
  and cannot change during prediction on shifted VALIDATION data.
- [ ] Baselines are recomputed on the same 15-stock experiment and all three
  models use identical VALIDATION observations.
- [ ] Coefficient reporting reflects standardized conditional linear terms and
  does not make causal, economic-significance, or profitability claims.
- [ ] No unnecessary abstraction, unrelated dependency, generated artifact,
  dataset mutation, backtest, tuning, or model variant is introduced.

## Future Implementation Commit

After all focused tests, the full suite, Ruff, imports, scope checks, and staged
diff checks pass, create exactly one source commit:

```powershell
git commit -m "feat: add standardized linear regression forecaster"
```

Do not amend and do not push. The read-only real-data evaluation follows that
commit and must leave the repository and both persisted Parquet files unchanged.

## Plan-Only Commit

Before executing any task above, commit this plan as the only changed path:

```powershell
git commit -m "docs: plan standardized linear regression implementation"
```

Do not include dependency, production, test, README, configuration, or dataset
changes in the plan commit. Do not amend and do not push.
