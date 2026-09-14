# Fixed Global Random Forest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed, global `RandomForestForecaster` to test whether
nonlinear relationships in the existing ten engineered features improve
validation performance beyond the current baselines and linear regression.

**Architecture:** Use one direct `RandomForestRegressor` across the 15-stock
universe with exactly `FEATURE_COLUMNS`, no StandardScaler or preprocessing
pipeline, strict contracts matching `LinearRegressionForecaster`, and the fixed
approved hyperparameters. Evaluation excludes SPY in orchestration, fits on
TRAIN only, compares models on VALIDATION, and never accesses TEST.

**Tech Stack:** Python 3.12, pandas, scikit-learn, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-14-random-forest-design.md`

## Global Constraints

- Train one global model across exactly AAPL, MSFT, GOOGL, AMZN, META, NVDA,
  JPM, V, JNJ, UNH, XOM, CVX, COST, WMT, and PG.
- Import and use exactly `FEATURE_COLUMNS` in declared order. Keep `ticker` as
  metadata; never include `ticker`, `date`, `target_end_date`, or
  `future_return_5d` in X.
- Keep SPY persisted in raw and processed data and configured as
  `benchmark_ticker = "SPY"`; exclude it only in evaluation orchestration before
  supervised preparation. Do not modify `configs/market.yaml`,
  `prepare_supervised_data`, or `temporal_split`.
- Instantiate `RandomForestRegressor` directly with `n_estimators=300`,
  `max_depth=8`, `min_samples_leaf=20`, `max_features=1.0`,
  `random_state=42`, and `n_jobs=-1`. Accept no runtime configuration.
- Use no StandardScaler, Pipeline, ColumnTransformer, feature scaling, or other
  preprocessing wrapper around the forest.
- Fit the forest and LinearRegressionForecaster on TRAIN only. VALIDATION is the
  sole official comparison partition. Never access TEST after constructing the
  split.
- Recompute MeanBaseline, MomentumBaseline, LinearRegressionForecaster, and
  RandomForestForecaster from scratch on identical VALIDATION rows, reusing
  `evaluate_predictions` for every reported metric.
- Require exact schema, numeric pandas dtypes, finite values, equal X/y lengths,
  and `X.index.equals(y.index)`. Do not coerce, sort, reset, reindex, reorder, or
  silently align caller data. Preserve caller inputs and prediction index.
- Expose a safe copied `feature_importances_` Series directly corresponding to
  sklearn diagnostics, indexed by `FEATURE_COLUMNS`, finite, non-negative, and
  summing approximately to 1.0. Raise RuntimeError for unavailable or invalid
  fitted diagnostics; never fabricate or renormalize them.
- Make no profitability, investability, causality, statistical-significance,
  economic-significance, or TEST-performance claims.
- After observing VALIDATION, do not alter any hyperparameter, run a second
  forest configuration, or tune manually. Poor performance still completes
  Step 5D.
- Perform no backtesting or portfolio construction and write no datasets or
  evaluation artifacts.
- Add no dependency. Scikit-learn is already present. Do not modify
  `pyproject.toml` unless execution is stopped by a genuine existing-environment
  fault and the design is revisited first.
- Introduce no estimator base class, model registry, experiment framework,
  configuration abstraction, CLI, or reusable tuning framework.
- Do not amend commits or push.

## File Map

- Create `src/stock_forecaster/models/forest.py`: focused estimator ownership,
  input validation, fitted-state guard, aligned predictions, and safe importance
  diagnostics.
- Create `tests/test_forest.py`: deterministic synthetic TDD coverage for all 40
  required behaviors.
- Modify `README.md`: document the implemented forest and model progression,
  without pre-announcing numerical results.
- Use this file, `docs/superpowers/plans/2026-09-14-random-forest.md`, as the
  execution checklist; it is committed before implementation.
- Create no evaluation script, generated table, model artifact, or dataset. Run
  the read-only evaluation as an inline Python command after the feature commit.

---

### Task 1: Drive the fixed estimator shell and fitted-state behavior

**Files:**
- Create: `tests/test_forest.py`
- Create: `src/stock_forecaster/models/forest.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, numeric `pd.DataFrame` X, and aligned numeric
  `pd.Series` y.
- Produces: `RandomForestForecaster.fit(X, y) -> RandomForestForecaster`,
  `predict(X) -> pd.Series`, and `feature_importances_ -> pd.Series`.

- [ ] **Step 1: Write shared synthetic builders and structural failing tests**

  Create `tests/test_forest.py` with deterministic, informative data. Sixty-four
  rows permit valid splits under `min_samples_leaf=20`:

  ```python
  import math

  import pandas as pd
  import pytest
  from sklearn.ensemble import RandomForestRegressor

  from stock_forecaster.features.engineering import FEATURE_COLUMNS
  from stock_forecaster.models.forest import RandomForestForecaster


  def _numeric_X(
      *, rows: int = 64, index: pd.Index | None = None
  ) -> pd.DataFrame:
      if index is None:
          index = pd.Index(range(rows), name="observation")
      return pd.DataFrame(
          {
              feature: [
                  float(((row + 3) ** (number + 1)) % 101) / 100.0
                  for row in range(rows)
              ]
              for number, feature in enumerate(FEATURE_COLUMNS)
          },
          index=index,
      )


  def _numeric_y(X: pd.DataFrame) -> pd.Series:
      return (
          X[FEATURE_COLUMNS[0]].mul(0.4)
          .sub(X[FEATURE_COLUMNS[1]].mul(0.25))
          .add(X[FEATURE_COLUMNS[4]].mul(0.1))
          .rename("future_return_5d")
      )


  def _fitted_model() -> tuple[RandomForestForecaster, pd.DataFrame, pd.Series]:
      X = _numeric_X()
      y = _numeric_y(X)
      return RandomForestForecaster().fit(X, y), X, y


  def test_model_uses_exact_approved_estimator_without_preprocessing():
      model = RandomForestForecaster()

      assert type(model._estimator) is RandomForestRegressor
      assert not hasattr(model, "_pipeline")
      params = model._estimator.get_params()
      assert params["n_estimators"] == 300
      assert params["max_depth"] == 8
      assert params["min_samples_leaf"] == 20
      assert params["max_features"] == 1.0
      assert params["random_state"] == 42
      assert params["n_jobs"] == -1


  def test_fit_accepts_exact_schema_and_returns_self():
      X = _numeric_X()
      model = RandomForestForecaster()
      assert model.fit(X, _numeric_y(X)) is model


  def test_predict_before_fit_raises_clear_runtime_error():
      with pytest.raises(RuntimeError, match="must be fitted"):
          RandomForestForecaster().predict(_numeric_X(rows=3))


  def test_feature_importances_before_fit_raise_clear_runtime_error():
      with pytest.raises(RuntimeError, match="must be fitted"):
          RandomForestForecaster().feature_importances_
  ```

- [ ] **Step 2: Run the initial tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: collection fails with `ModuleNotFoundError` for
  `stock_forecaster.models.forest`.

- [ ] **Step 3: Implement only the estimator shell and fitted guard**

  Create `src/stock_forecaster/models/forest.py`:

  ```python
  """Fixed global Random Forest return forecaster."""

  import pandas as pd
  from sklearn.ensemble import RandomForestRegressor

  from stock_forecaster.features.engineering import FEATURE_COLUMNS


  class RandomForestForecaster:
      """Fit one fixed Random Forest across the stock universe."""

      def __init__(self) -> None:
          self._estimator = RandomForestRegressor(
              n_estimators=300,
              max_depth=8,
              min_samples_leaf=20,
              max_features=1.0,
              random_state=42,
              n_jobs=-1,
          )
          self._is_fitted = False

      def _require_fitted(self) -> None:
          if not self._is_fitted:
              raise RuntimeError(
                  "RandomForestForecaster must be fitted before use"
              )

      def fit(
          self, X: pd.DataFrame, y: pd.Series
      ) -> "RandomForestForecaster":
          self._estimator.fit(X, y)
          self._is_fitted = True
          return self

      def predict(self, X: pd.DataFrame) -> pd.Series:
          self._require_fitted()
          return pd.Series(self._estimator.predict(X))

      @property
      def feature_importances_(self) -> pd.Series:
          self._require_fitted()
          return pd.Series(self._estimator.feature_importances_)
  ```

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: all tests currently present pass. The unfinished output contracts
  are not asserted until their RED cycles below.

### Task 2: Enforce schema, numeric, finite, alignment, and fit immutability

**Files:**
- Modify: `tests/test_forest.py`
- Modify: `src/stock_forecaster/models/forest.py`

**Interfaces:**
- Consumes: X with exactly `FEATURE_COLUMNS` and y with an identical index.
- Produces: clear pre-sklearn ValueErrors for invalid fit inputs while leaving X
  and y unchanged.

- [ ] **Step 1: Add failing schema and dtype tests**

  Append:

  ```python
  @pytest.mark.parametrize(
      "invalid_X",
      [
          lambda X: X.drop(columns=[FEATURE_COLUMNS[-1]]),
          lambda X: X.assign(extra=1.0),
          lambda X: X.loc[:, list(reversed(FEATURE_COLUMNS))],
      ],
  )
  def test_missing_extra_or_reordered_features_are_rejected(invalid_X):
      X = invalid_X(_numeric_X())
      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          RandomForestForecaster().fit(X, pd.Series(range(len(X)), index=X.index))


  def test_non_numeric_feature_is_rejected_before_sklearn():
      X = _numeric_X()
      y = _numeric_y(X)
      X[FEATURE_COLUMNS[3]] = "bad"
      with pytest.raises(ValueError, match="feature columns must be numeric"):
          RandomForestForecaster().fit(X, y)


  def test_non_numeric_y_is_rejected_before_sklearn():
      X = _numeric_X()
      y = pd.Series(["bad"] * len(X), index=X.index)
      with pytest.raises(ValueError, match="y must have a numeric"):
          RandomForestForecaster().fit(X, y)


  def test_integer_and_float_numeric_dtypes_are_accepted():
      X = _numeric_X()
      X[FEATURE_COLUMNS[0]] = range(len(X))
      assert RandomForestForecaster().fit(X, _numeric_y(X))
  ```

- [ ] **Step 2: Add failing null, infinity, and alignment tests**

  Append:

  ```python
  @pytest.mark.parametrize("container", ["X", "y"])
  def test_fit_rejects_nan(container):
      X = _numeric_X()
      y = _numeric_y(X)
      if container == "X":
          X.iloc[0, 0] = float("nan")
      else:
          y.iloc[0] = float("nan")
      with pytest.raises(ValueError, match="must not contain null"):
          RandomForestForecaster().fit(X, y)


  @pytest.mark.parametrize("container", ["X", "y"])
  @pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
  def test_fit_rejects_positive_and_negative_infinity(container, invalid):
      X = _numeric_X()
      y = _numeric_y(X)
      if container == "X":
          X.iloc[0, 0] = invalid
      else:
          y.iloc[0] = invalid
      with pytest.raises(ValueError, match="only finite"):
          RandomForestForecaster().fit(X, y)


  def test_x_y_length_mismatch_is_rejected():
      X = _numeric_X()
      with pytest.raises(ValueError, match="equal lengths"):
          RandomForestForecaster().fit(X, _numeric_y(X).iloc[:-1])


  def test_equal_length_index_mismatch_is_rejected_without_alignment():
      X = _numeric_X(index=pd.Index(range(100, 164)))
      y = _numeric_y(X).set_axis(X.index[::-1])
      assert len(X) == len(y) and not X.index.equals(y.index)
      with pytest.raises(ValueError, match="indices must match exactly"):
          RandomForestForecaster().fit(X, y)
  ```

- [ ] **Step 3: Add failing fit immutability tests**

  Append:

  ```python
  def test_fit_does_not_mutate_x_or_y():
      X = _numeric_X()
      y = _numeric_y(X)
      original_X = X.copy(deep=True)
      original_y = y.copy(deep=True)

      RandomForestForecaster().fit(X, y)

      pd.testing.assert_frame_equal(X, original_X)
      pd.testing.assert_series_equal(y, original_y)
  ```

  This single test contains separate equality assertions for X and y; retain
  both so failures identify either mutation contract.

- [ ] **Step 4: Run validation tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: invalid-input cases fail because the estimator shell does not yet
  enforce the public contracts; immutability may already pass.

- [ ] **Step 5: Add explicit validation before estimator calls**

  Add imports and private helpers above the class in `forest.py`:

  ```python
  import math

  from pandas.api.types import is_numeric_dtype


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
  ```

  Replace fit with:

  ```python
      def fit(
          self, X: pd.DataFrame, y: pd.Series
      ) -> "RandomForestForecaster":
          _validate_X(X)
          _validate_y(y)
          if len(X) != len(y):
              raise ValueError("X and y must have equal lengths")
          if not X.index.equals(y.index):
              raise ValueError("X and y indices must match exactly")
          self._estimator.fit(X, y)
          self._is_fitted = True
          return self
  ```

  Do not copy, coerce, sort, reset, select, reindex, or realign inputs.

- [ ] **Step 6: Run validation tests and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: every test currently present passes.

### Task 3: Complete prediction validation, alignment, and immutability

**Files:**
- Modify: `tests/test_forest.py`
- Modify: `src/stock_forecaster/models/forest.py`

**Interfaces:**
- Consumes: fitted forecaster and valid X with any exact pandas index.
- Produces: `pd.Series` named `prediction` with exactly X.index and no mutation
  of X or estimator state.

- [ ] **Step 1: Add failing prediction contract and immutability tests**

  Append:

  ```python
  def test_prediction_preserves_series_name_index_count_and_multiple_rows():
      model, _, _ = _fitted_model()
      index = pd.Index([90, 4, 17, 2], name="validation_row")
      X = _numeric_X(rows=4, index=index)

      predictions = model.predict(X)

      assert isinstance(predictions, pd.Series)
      assert predictions.name == "prediction"
      assert predictions.index.equals(X.index)
      assert len(predictions) == len(X)
      assert all(math.isfinite(value) for value in predictions)


  def test_predict_does_not_mutate_x_or_fitted_estimator():
      model, _, _ = _fitted_model()
      X = _numeric_X(rows=4, index=pd.Index([7, 1, 9, 3]))
      original_X = X.copy(deep=True)
      importances_before = model._estimator.feature_importances_.copy()

      model.predict(X)

      pd.testing.assert_frame_equal(X, original_X)
      assert model._estimator.feature_importances_ == pytest.approx(
          importances_before
      )
  ```

- [ ] **Step 2: Add failing predict-input validation tests**

  Append:

  ```python
  def test_predict_rejects_non_numeric_or_wrong_schema():
      model, _, _ = _fitted_model()
      non_numeric = _numeric_X(rows=3)
      non_numeric[FEATURE_COLUMNS[0]] = "bad"
      with pytest.raises(ValueError, match="numeric"):
          model.predict(non_numeric)
      with pytest.raises(ValueError, match="exactly FEATURE_COLUMNS"):
          model.predict(_numeric_X(rows=3).drop(columns=[FEATURE_COLUMNS[-1]]))


  @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
  def test_predict_rejects_non_finite_features(invalid):
      model, _, _ = _fitted_model()
      X = _numeric_X(rows=3)
      X.iloc[0, 0] = invalid
      with pytest.raises(ValueError):
          model.predict(X)
  ```

- [ ] **Step 3: Run prediction tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: the Series name/index contract and explicit predict validation fail
  against the minimal shell.

- [ ] **Step 4: Complete predict with validation and exact alignment**

  Replace predict in `forest.py`:

  ```python
      def predict(self, X: pd.DataFrame) -> pd.Series:
          self._require_fitted()
          _validate_X(X)
          values = self._estimator.predict(X)
          return pd.Series(values, index=X.index, name="prediction")
  ```

- [ ] **Step 5: Run prediction tests and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: every test currently present passes, including unsorted-index and
  multi-row behavior.

### Task 4: Drive feature-importance safety and seeded reproducibility

**Files:**
- Modify: `tests/test_forest.py`
- Modify: `src/stock_forecaster/models/forest.py`

**Interfaces:**
- Consumes: fitted `RandomForestRegressor.feature_importances_` and deterministic
  synthetic X/y.
- Produces: safe validated `pd.Series feature_importances_` and reproducible
  predictions for separate fixed-seed instances.

- [ ] **Step 1: Add failing importance contract and copy-safety tests**

  Append:

  ```python
  def test_feature_importances_have_exact_public_contract():
      model, _, _ = _fitted_model()

      importances = model.feature_importances_

      assert isinstance(importances, pd.Series)
      assert importances.index.equals(pd.Index(FEATURE_COLUMNS))
      assert importances.to_numpy() == pytest.approx(
          model._estimator.feature_importances_
      )
      assert all(math.isfinite(value) for value in importances)
      assert importances.ge(0).all()
      assert importances.sum() == pytest.approx(1.0)


  def test_returned_importance_series_cannot_mutate_estimator_state():
      model, _, _ = _fitted_model()
      expected = model.feature_importances_
      changed = model.feature_importances_

      changed.iloc[0] = 999.0

      pd.testing.assert_series_equal(model.feature_importances_, expected)
      assert model._estimator.feature_importances_ == pytest.approx(
          expected.to_numpy()
      )
  ```

- [ ] **Step 2: Add failing invalid-diagnostic and reproducibility tests**

  Append:

  ```python
  @pytest.mark.parametrize(
      "invalid",
      [
          [float("nan"), *([0.0] * 9)],
          [-0.1, 1.1, *([0.0] * 8)],
          [0.05] * 10,
      ],
  )
  def test_invalid_fitted_importances_raise_instead_of_being_changed(
      monkeypatch, invalid
  ):
      model, _, _ = _fitted_model()
      monkeypatch.setattr(
          type(model._estimator),
          "feature_importances_",
          property(lambda self: invalid),
      )
      with pytest.raises(RuntimeError, match="diagnostics"):
          model.feature_importances_


  def test_identical_data_and_seed_produce_matching_predictions():
      X = _numeric_X(index=pd.Index(range(500, 564)))
      y = _numeric_y(X)
      model_a = RandomForestForecaster().fit(X, y)
      model_b = RandomForestForecaster().fit(X, y)

      pd.testing.assert_series_equal(model_a.predict(X), model_b.predict(X))
  ```

  The class-level property patch is a focused diagnostic guard test; it does not
  inspect or compare tree identities.

- [ ] **Step 3: Run importance tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: ordered labels and invalid-diagnostic handling fail against the
  unlabeled shell; reproducibility may already pass because the seed is fixed.

- [ ] **Step 4: Implement safe importance diagnostics**

  Replace the property:

  ```python
      @property
      def feature_importances_(self) -> pd.Series:
          self._require_fitted()
          values = self._estimator.feature_importances_
          if (
              len(values) != len(FEATURE_COLUMNS)
              or not all(math.isfinite(value) for value in values)
              or any(value < 0 for value in values)
              or not math.isclose(sum(values), 1.0)
          ):
              raise RuntimeError(
                  "Fitted Random Forest diagnostics must be finite, "
                  "non-negative, and sum to one"
              )
          return pd.Series(
              values,
              index=FEATURE_COLUMNS,
              name="importance",
              dtype=float,
          ).copy()
  ```

  Do not normalize, sort, or cache the Series. The report may sort a separate
  presentation copy later; the public property remains in feature-schema order.

- [ ] **Step 5: Run the complete forest test file and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_forest.py -v
  ```

  Expected: all structural, validation, immutability, prediction, importance,
  reproducibility, and multi-row tests pass.

### Task 5: Document the implemented model, verify fully, and commit

**Files:**
- Modify: `README.md`
- Verify: `src/stock_forecaster/models/forest.py`
- Verify: `tests/test_forest.py`

**Interfaces:**
- Consumes: the completed public API from Tasks 1–4.
- Produces: accurate project documentation and one verified implementation
  commit, `feat: add fixed random forest forecaster`.

- [ ] **Step 1: Update README without numerical forest results**

  Replace the model-progression diagram with:

  ```text
  Supervised temporal split
          ↓
  Mean / Momentum baselines
          ↓
  Global standardized linear regression
          ↓
  Fixed global Random Forest
          ↓
  Future models
  ```

  Add this concise status paragraph after the existing linear-model paragraph:

  ```markdown
  The first nonlinear benchmark is one fixed Random Forest across the same
  15-stock universe. SPY remains the persisted benchmark and is excluded only
  in modeling orchestration. The forest consumes the same ten engineered
  features directly, without feature scaling, fits on TRAIN only, and is
  compared with freshly fitted baselines and linear regression on VALIDATION.
  TEST remains untouched.
  ```

  Update current status to include the fixed Random Forest forecaster. Do not
  add metrics until the read-only evaluation has actually run.

- [ ] **Step 2: Verify the public import**

  Run:

  ```powershell
  python -c "from stock_forecaster.models.forest import RandomForestForecaster; print(RandomForestForecaster.__name__)"
  ```

  Expected: exit code 0 and `RandomForestForecaster`.

- [ ] **Step 3: Run focused tests, the complete suite, and Ruff**

  Run each command freshly:

  ```powershell
  python -m pytest tests/test_forest.py -v
  python -m pytest -v
  python -m ruff check .
  ```

  Expected: all forest tests pass, the complete suite has zero failures, and
  Ruff reports success. Focused success alone is insufficient.

- [ ] **Step 4: Confirm scope and dataset cleanliness before staging**

  Run:

  ```powershell
  git status --short
  git diff --check
  git diff -- pyproject.toml configs/market.yaml src/stock_forecaster/models/dataset.py
  ```

  Expected: only `forest.py`, `test_forest.py`, and README.md are changed; the
  final command prints no diff.

- [ ] **Step 5: Create the one implementation commit**

  Run:

  ```powershell
  git add src/stock_forecaster/models/forest.py tests/test_forest.py README.md
  git diff --cached --check
  git diff --cached --name-only
  git commit -m "feat: add fixed random forest forecaster"
  ```

  Expected: exactly those three files are staged and committed. Do not stage
  Parquet files, model artifacts, configuration, dependencies, or this already
  committed plan. Do not push.

### Task 6: Run the fixed read-only TRAIN/VALIDATION evaluation

**Files:**
- Read: `data/raw/prices.parquet`
- Read: `data/processed/features.parquet`
- Create: no file
- Modify: no file

**Interfaces:**
- Consumes: the committed four model implementations, the persisted feature
  table, exact 15-stock universe, and approved temporal boundaries.
- Produces: console-only audit counts, official VALIDATION comparison,
  prediction distributions, forest TRAIN/VALIDATION diagnostics, per-ticker
  VALIDATION diagnostics, and impurity-based feature importances.

- [ ] **Step 1: Record both pre-evaluation SHA-256 hashes**

  Run:

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  ```

  Record both exact hashes outside the repository for comparison. Do not create
  an audit file in the workspace.

- [ ] **Step 2: Execute the single fixed experiment in memory**

  Run this from the repository root as an inline Python command (PowerShell here
  string piped to the interpreter). It deliberately never binds or reads
  `split.test`:

  ```powershell
  @'
  import pandas as pd

  from stock_forecaster.features.engineering import FEATURE_COLUMNS
  from stock_forecaster.models.baseline import MeanBaseline, MomentumBaseline
  from stock_forecaster.models.dataset import prepare_supervised_data, temporal_split
  from stock_forecaster.models.forest import RandomForestForecaster
  from stock_forecaster.models.linear import LinearRegressionForecaster
  from stock_forecaster.models.metrics import evaluate_predictions

  STOCKS = {
      "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
      "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG",
  }

  features = pd.read_parquet("data/processed/features.parquet")
  modeling = features.loc[features["ticker"].isin(STOCKS)].copy()
  assert set(modeling["ticker"].unique()) == STOCKS
  assert "SPY" not in set(modeling["ticker"])

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
  assert set(train.metadata["ticker"].unique()) == STOCKS
  assert set(validation.metadata["ticker"].unique()) == STOCKS
  assert "SPY" not in set(train.metadata["ticker"])
  assert "SPY" not in set(validation.metadata["ticker"])

  print("AUDIT")
  for name, partition in (("TRAIN", train), ("VALIDATION", validation)):
      print(
          name,
          len(partition.y),
          partition.metadata["date"].min(),
          partition.metadata["date"].max(),
          sorted(partition.metadata["ticker"].unique()),
      )

  predictions = {}
  predictions["MeanBaseline"] = MeanBaseline().fit(train.y).predict(validation.y.index)
  predictions["MomentumBaseline"] = MomentumBaseline().predict(validation.X)
  predictions["LinearRegression"] = (
      LinearRegressionForecaster().fit(train.X, train.y).predict(validation.X)
  )
  forest = RandomForestForecaster().fit(train.X, train.y)
  predictions["RandomForest"] = forest.predict(validation.X)
  for prediction in predictions.values():
      assert prediction.index.equals(validation.y.index)

  comparison_rows = []
  distribution_rows = []
  for name, prediction in predictions.items():
      metric = evaluate_predictions(validation.y, prediction)
      comparison_rows.append(
          {
              "model": name,
              "MAE": metric.mae,
              "RMSE": metric.rmse,
              "directional_accuracy": metric.directional_accuracy,
              "correlation": metric.correlation,
          }
      )
      distribution_rows.append(
          {
              "model": name,
              "mean": prediction.mean(),
              "standard_deviation": prediction.std(),
              "minimum": prediction.min(),
              "maximum": prediction.max(),
          }
      )
  print("VALIDATION COMPARISON")
  print(pd.DataFrame(comparison_rows).to_string(index=False))
  print("VALIDATION PREDICTION DISTRIBUTIONS")
  print(pd.DataFrame(distribution_rows).to_string(index=False))

  forest_partition_rows = []
  for name, actual, predicted in (
      ("TRAIN", train.y, forest.predict(train.X)),
      ("VALIDATION", validation.y, predictions["RandomForest"]),
  ):
      metric = evaluate_predictions(actual, predicted)
      forest_partition_rows.append(
          {
              "partition": name,
              "MAE": metric.mae,
              "RMSE": metric.rmse,
              "directional_accuracy": metric.directional_accuracy,
              "correlation": metric.correlation,
          }
      )
  print("RANDOM FOREST TRAIN VS VALIDATION")
  print(pd.DataFrame(forest_partition_rows).to_string(index=False))

  by_ticker_rows = []
  rf_validation = predictions["RandomForest"]
  for ticker in sorted(STOCKS):
      mask = validation.metadata["ticker"].eq(ticker)
      actual = validation.y.loc[mask]
      predicted = rf_validation.loc[mask]
      metric = evaluate_predictions(actual, predicted)
      by_ticker_rows.append(
          {
              "ticker": ticker,
              "row_count": len(actual),
              "MAE": metric.mae,
              "directional_accuracy": metric.directional_accuracy,
              "correlation": metric.correlation,
          }
      )
  print("RANDOM FOREST VALIDATION BY TICKER")
  print(pd.DataFrame(by_ticker_rows).to_string(index=False))

  importance_table = (
      forest.feature_importances_
      .rename_axis("feature")
      .rename("importance")
      .sort_values(ascending=False)
      .reset_index()
  )
  assert set(importance_table["feature"]) == set(FEATURE_COLUMNS)
  print("IMPURITY-BASED FEATURE IMPORTANCES")
  print(importance_table.to_string(index=False))
  '@ | python -
  ```

  Expected: reported counts are audited at runtime (approximately 43,906 TRAIN
  and 7,440 VALIDATION); all four comparison rows use identical VALIDATION
  observations; no TEST information appears. Do not rerun with different forest
  parameters regardless of results.

- [ ] **Step 3: Interpret once under the fixed-experiment rule**

  Report whether RandomForest improves MeanBaseline MAE/RMSE, correlation, and
  supporting directional accuracy; whether its TRAIN/VALIDATION gap suggests
  overfitting; and which predictors have larger impurity-based diagnostics.
  State that poor performance still completes Step 5D. Make no prohibited claim
  and do not modify or rerun the model configuration.

- [ ] **Step 4: Recalculate and compare both dataset hashes**

  Run:

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  ```

  Expected: each post-evaluation SHA-256 exactly equals its corresponding
  pre-evaluation value. If either differs, stop immediately and report the
  unexpected mutation without making further changes.

- [ ] **Step 5: Run fresh final verification and prove repository cleanliness**

  Run:

  ```powershell
  python -m pytest -v
  python -m ruff check .
  git status --short
  git diff --exit-code
  git diff --cached --exit-code
  ```

  Expected: the full suite passes, Ruff succeeds, status prints nothing, and
  both diff commands exit 0. No evaluation output, model artifact, or generated
  dataset is committed. Do not push.

## Explicitly Excluded Work

Do not use GridSearchCV, RandomizedSearchCV, manual parameter search, Ridge,
Lasso, ElasticNet, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM,
CatBoost, ticker encoding, sector encoding, new features, feature selection,
permutation importance, SHAP, partial dependence, cross-validation,
walk-forward validation, TEST evaluation, backtesting, or portfolio
construction.

## Plan Self-Review

Every approved specification requirement maps to Tasks 1–6. The exact estimator
type and all six hyperparameters are consistent; the plan uses no forest
scaling or preprocessing pipeline. Exact ordered feature schema, numeric and
finite inputs, X/y length and index equality, input immutability, prediction
Series naming/index/count, fitted-state errors, safe importance copying and
validation, and fixed-seed reproducibility all have explicit RED and GREEN
steps. SPY exclusion remains orchestration-only, all four models are recomputed,
VALIDATION is the comparison partition, forest TRAIN metrics are diagnostic
only, and TEST is never accessed. No dependency or unnecessary abstraction is
planned. The fixed-experiment rule prohibits a second configuration or tuning,
and dataset hashes plus Git checks prove read-only evaluation. Names and types
are consistent throughout, and all 40 requested test behaviors are covered
without unresolved implementation choices.
