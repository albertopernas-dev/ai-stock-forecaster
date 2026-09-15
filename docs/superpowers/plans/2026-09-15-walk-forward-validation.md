# Expanding-Window Walk-Forward Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build expanding-window annual walk-forward evaluation for 2016-2023 using the four existing forecasting models, producing fold-level and pooled out-of-sample metrics while keeping all 2024+ data completely outside the Step 5E development horizon.

**Architecture:** `dataset.py` owns annual temporal fold construction, target-horizon purging, and reuse of `DatasetPartition`; the new `evaluation.py` owns explicit four-model execution, aligned in-memory prediction records, and metric aggregation. Each fold receives fresh fitted model instances, pooled metrics are recomputed from concatenated OOS predictions, and orchestration filters the development horizon before the splitter sees any 2024+ row.

**Tech Stack:** Python 3.12, pandas, scikit-learn, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-15-walk-forward-validation-design.md`

## Global Constraints

- Use validation years exactly 2016 through 2023 with chronologically expanding TRAIN partitions, never rolling windows.
- Retain TRAIN rows only when `date < validation_start` and `target_end_date < validation_start`.
- Retain VALIDATION rows only when `validation_start <= date < next_year_start` and `target_end_date < next_year_start`.
- Use the actual ticker-specific `target_end_date`; do not approximate horizons with calendar days, business days, or splitter row offsets.
- Before fold construction, restrict development data to `date < 2024-01-01` and `target_end_date < 2024-01-01` so 2023 labels cannot consume 2024 observations.
- Exclude SPY only in orchestration and use exactly AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, V, JNJ, UNH, XOM, CVX, COST, WMT, and PG.
- Reuse `DatasetPartition`; keep model execution out of `dataset.py` and split/purge logic out of `evaluation.py`.
- Evaluate exactly MeanBaseline, MomentumBaseline, LinearRegressionForecaster, and RandomForestForecaster, using canonical output names `MeanBaseline`, `MomentumBaseline`, `LinearRegression`, and `RandomForest`.
- Create fresh fitted model objects for every fold and never let VALIDATION targets enter fitting.
- Reuse `evaluate_predictions`; calculate pooled metrics from concatenated OOS prediction rows, never from averages of annual metrics.
- Keep OOS predictions in memory and commit no generated predictions, datasets, reports, or model artifacts.
- Add no dependency, forecasting model, model registry, estimator protocol, experiment framework, plugin system, dependency-injection layer, configuration framework, CLI, or prediction persistence layer.
- Perform no tuning, feature changes, backtesting, portfolio work, or TEST evaluation. TEST remains 2024+ and completely invisible to Step 5E model comparison and decisions.
- Do not modify `baseline.py`, `linear.py`, `forest.py`, or `metrics.py`. If a genuine existing model bug blocks implementation, stop and report it before changing model behavior.
- The implementation phase ends with one feature commit, `feat: add expanding walk-forward validation`; do not push.

## File Map

- Modify: `src/stock_forecaster/models/dataset.py` — annual fold type, input validation, temporal construction, and target-horizon purging.
- Create: `src/stock_forecaster/models/evaluation.py` — explicit model execution, prediction records, fold metrics, pooled metrics, and consistency counts.
- Create: `tests/test_walk_forward.py` — synthetic splitter, purge, alignment, immutability, and future-isolation coverage.
- Create: `tests/test_evaluation.py` — synthetic orchestration, prediction, aggregation, reproducibility, and future-isolation coverage.
- Modify: `README.md` — validation-method progression and TEST boundary, without numerical Step 5E results.
- Create during this planning phase only: `docs/superpowers/plans/2026-09-15-walk-forward-validation.md`.
- Leave `pyproject.toml`, configuration, datasets, and all existing model modules unchanged.

---

### Task 1: Add `WalkForwardFold` and core annual fold construction

**Files:**
- Create: `tests/test_walk_forward.py`
- Modify: `src/stock_forecaster/models/dataset.py`

**Interfaces:**
- Consumes: an already prepared `pd.DataFrame` with `_SUPERVISED_COLUMNS` and a `Sequence[int]` of validation years.
- Produces: frozen `WalkForwardFold(validation_year, train, validation)` values and `walk_forward_splits(supervised_data, validation_years) -> tuple[WalkForwardFold, ...]`.

- [ ] **Step 1: Create deterministic supervised-data helpers**

  Start `tests/test_walk_forward.py` with imports and a builder that creates the actual prepared schema directly, so splitter tests do not retest feature preparation:

  ```python
  from dataclasses import FrozenInstanceError

  import pandas as pd
  import pytest

  from stock_forecaster.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN
  from stock_forecaster.models.dataset import (
      DatasetPartition,
      WalkForwardFold,
      walk_forward_splits,
  )

  VALIDATION_YEARS = tuple(range(2016, 2024))


  def _supervised_rows(
      years: range = range(2015, 2024),
      tickers: tuple[str, ...] = ("AAA", "BBB"),
  ) -> pd.DataFrame:
      rows: list[dict[str, object]] = []
      number = 0
      for year in years:
          for month, day in ((1, 12), (6, 15)):
              for ticker in tickers:
                  observation_date = pd.Timestamp(year=year, month=month, day=day)
                  row: dict[str, object] = {
                      "date": observation_date,
                      "ticker": ticker,
                      "target_end_date": observation_date + pd.Timedelta(days=7),
                      TARGET_COLUMN: (number - 12) / 1000.0,
                  }
                  row.update(
                      {
                          feature: (number + feature_number + 1) / 100.0
                          for feature_number, feature in enumerate(FEATURE_COLUMNS)
                      }
                  )
                  rows.append(row)
                  number += 1
      return pd.DataFrame(rows).sample(frac=1, random_state=17).reset_index(drop=True)


  def _target_ends(
      supervised: pd.DataFrame,
      partition: DatasetPartition,
  ) -> pd.Series:
      return partition.metadata.merge(
          supervised.loc[:, ["date", "ticker", "target_end_date"]],
          on=["date", "ticker"],
          how="left",
          validate="one_to_one",
          sort=False,
      )["target_end_date"]
  ```

- [ ] **Step 2: Add failing core fold and partition-contract tests**

  Append:

  ```python
  def test_approved_years_produce_eight_chronological_folds():
      folds = walk_forward_splits(_supervised_rows(), VALIDATION_YEARS)

      assert isinstance(folds, tuple)
      assert len(folds) == 8
      assert [fold.validation_year for fold in folds] == list(VALIDATION_YEARS)
      assert all(isinstance(fold, WalkForwardFold) for fold in folds)


  def test_train_expands_and_validation_is_confined_to_each_year():
      folds = walk_forward_splits(_supervised_rows(), VALIDATION_YEARS)

      train_counts = [len(fold.train.y) for fold in folds]
      assert all(left < right for left, right in zip(train_counts, train_counts[1:]))
      for fold in folds:
          boundary = pd.Timestamp(fold.validation_year, 1, 1)
          next_boundary = pd.Timestamp(fold.validation_year + 1, 1, 1)
          assert fold.train.metadata["date"].lt(boundary).all()
          assert fold.validation.metadata["date"].ge(boundary).all()
          assert fold.validation.metadata["date"].lt(next_boundary).all()


  def test_each_partition_has_exact_aligned_ordered_contract():
      supervised = _supervised_rows()
      expected = supervised.set_index(["date", "ticker"])
      folds = walk_forward_splits(supervised, VALIDATION_YEARS)

      for fold in folds:
          for partition in (fold.train, fold.validation):
              assert isinstance(partition, DatasetPartition)
              assert list(partition.X.columns) == list(FEATURE_COLUMNS)
              assert partition.y.name == TARGET_COLUMN
              assert list(partition.metadata.columns) == ["date", "ticker"]
              assert partition.X.index.equals(partition.y.index)
              assert partition.X.index.equals(partition.metadata.index)
              assert partition.metadata["date"].is_monotonic_increasing
              assert not partition.metadata.duplicated(["date", "ticker"]).any()
              for index, row in partition.metadata.iterrows():
                  key = (row["date"], row["ticker"])
                  assert partition.y.loc[index] == expected.loc[key, TARGET_COLUMN]
                  assert partition.X.loc[index].tolist() == expected.loc[
                      key, FEATURE_COLUMNS
                  ].tolist()


  def test_walk_forward_fold_is_frozen():
      fold = walk_forward_splits(_supervised_rows(), (2016,))[0]

      with pytest.raises(FrozenInstanceError):
          fold.validation_year = 2020
  ```

- [ ] **Step 3: Run the new test file and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_walk_forward.py -v
  ```

  Expected: collection fails because `WalkForwardFold` and `walk_forward_splits` do not exist.

- [ ] **Step 4: Implement the fold type and exact core splitter**

  In `dataset.py`, add `from collections.abc import Sequence`, place the frozen
  fold dataclass after `TemporalDatasetSplit`, and add this public function after
  `_partition`:

  ```python
  @dataclass(frozen=True)
  class WalkForwardFold:
      """One expanding TRAIN and annual VALIDATION partition."""

      validation_year: int
      train: DatasetPartition
      validation: DatasetPartition


  def walk_forward_splits(
      supervised_data: pd.DataFrame,
      validation_years: Sequence[int],
  ) -> tuple[WalkForwardFold, ...]:
      """Build expanding annual folds with ticker-specific horizon purging."""
      years = tuple(validation_years)
      _require_columns(supervised_data, _SUPERVISED_COLUMNS)
      ordered = supervised_data.loc[:, _SUPERVISED_COLUMNS].copy()
      ordered["date"] = pd.to_datetime(ordered["date"])
      ordered["target_end_date"] = pd.to_datetime(ordered["target_end_date"])
      _reject_duplicate_keys(ordered)

      folds: list[WalkForwardFold] = []
      for year in years:
          validation_start = pd.Timestamp(year, 1, 1)
          next_year_start = pd.Timestamp(year + 1, 1, 1)
          train_rows = ordered.loc[
              ordered["date"].lt(validation_start)
              & ordered["target_end_date"].lt(validation_start)
          ]
          validation_rows = ordered.loc[
              ordered["date"].ge(validation_start)
              & ordered["date"].lt(next_year_start)
              & ordered["target_end_date"].lt(next_year_start)
          ]
          folds.append(
              WalkForwardFold(
                  validation_year=year,
                  train=_partition(train_rows, f"TRAIN {year}"),
                  validation=_partition(validation_rows, f"VALIDATION {year}"),
              )
          )
      return tuple(folds)
  ```

  This first GREEN reuses the existing schema, date normalization, duplicate
  check, and `_partition` ordering/alignment behavior. Task 2 drives invalid
  year-sequence handling. Do not alter `temporal_split`.

- [ ] **Step 5: Run focused tests and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_walk_forward.py -v
  python -m pytest tests/test_dataset.py -v
  ```

  Expected: all new core fold tests and all existing dataset tests pass.

### Task 2: Complete splitter validation, purge safety, and future isolation

**Files:**
- Modify: `tests/test_walk_forward.py`
- Modify: `src/stock_forecaster/models/dataset.py`

**Interfaces:**
- Consumes: the Task 1 `walk_forward_splits` API and actual prepared schema.
- Produces: verified year validation, exact TRAIN/VALIDATION purge behavior,
  immutability, clear empty-partition failures, and earlier-fold isolation.

- [ ] **Step 1: Add failing target-horizon and 2024-boundary tests**

  Append:

  ```python
  def test_train_and_validation_use_actual_target_end_date_for_purging():
      supervised = _supervised_rows(years=range(2015, 2017), tickers=("AAA",))
      crossing_train = supervised.iloc[[0]].copy()
      crossing_train.loc[:, "date"] = pd.Timestamp("2015-12-28")
      crossing_train.loc[:, "target_end_date"] = pd.Timestamp("2016-01-04")
      crossing_validation = supervised.iloc[[1]].copy()
      crossing_validation.loc[:, "date"] = pd.Timestamp("2016-12-28")
      crossing_validation.loc[:, "target_end_date"] = pd.Timestamp("2017-01-04")
      supervised = pd.concat(
          [supervised, crossing_train, crossing_validation], ignore_index=True
      )

      fold = walk_forward_splits(supervised, (2016,))[0]
      train_keys = set(fold.train.metadata.itertuples(index=False, name=None))
      validation_keys = set(fold.validation.metadata.itertuples(index=False, name=None))

      assert (pd.Timestamp("2015-12-28"), "AAA") not in train_keys
      assert (pd.Timestamp("2016-12-28"), "AAA") not in validation_keys
      assert _target_ends(supervised, fold.train).lt("2016-01-01").all()
      assert _target_ends(supervised, fold.validation).lt("2017-01-01").all()


  def test_late_2023_target_using_2024_is_excluded():
      supervised = _supervised_rows(tickers=("AAA",))
      late_row = supervised.iloc[[0]].copy()
      late_row.loc[:, "date"] = pd.Timestamp("2023-12-28")
      late_row.loc[:, "target_end_date"] = pd.Timestamp("2024-01-05")
      supervised = pd.concat([supervised, late_row], ignore_index=True)

      fold = walk_forward_splits(supervised, (2023,))[0]

      assert pd.Timestamp("2023-12-28") not in set(fold.validation.metadata["date"])
      assert _target_ends(supervised, fold.validation).lt("2024-01-01").all()


  def test_approved_development_folds_retain_no_2024_dates_or_target_ends():
      supervised = _supervised_rows(years=range(2015, 2025))
      development = supervised.loc[
          supervised["date"].lt("2024-01-01")
          & supervised["target_end_date"].lt("2024-01-01")
      ].copy()
      folds = walk_forward_splits(development, VALIDATION_YEARS)

      assert all(
          partition.metadata["date"].lt("2024-01-01").all()
          for fold in folds
          for partition in (fold.train, fold.validation)
      )
      assert all(
          _target_ends(development, partition).lt("2024-01-01").all()
          for fold in folds
          for partition in (fold.train, fold.validation)
      )
  ```

- [ ] **Step 2: Add failing validation, uniqueness, and immutability tests**

  Append:

  ```python
  @pytest.mark.parametrize(
      ("years", "message"),
      [
          ((), "must not be empty"),
          ((2016, 2016), "must be unique"),
          ((2017, 2016), "strictly increasing"),
          ((2016, "2017"), "only integers"),
          ((2016, True), "only integers"),
      ],
  )
  def test_invalid_validation_years_are_rejected_without_repair(years, message):
      with pytest.raises(ValueError, match=message):
          walk_forward_splits(_supervised_rows(), years)


  def test_missing_supervised_columns_are_rejected():
      supervised = _supervised_rows().drop(columns=[FEATURE_COLUMNS[-1]])
      with pytest.raises(ValueError, match=FEATURE_COLUMNS[-1]):
          walk_forward_splits(supervised, (2016,))


  def test_duplicate_date_ticker_rows_are_rejected():
      supervised = _supervised_rows()
      supervised = pd.concat([supervised, supervised.iloc[[0]]], ignore_index=True)
      with pytest.raises(ValueError, match="Duplicate date and ticker rows"):
          walk_forward_splits(supervised, (2016,))


  def test_splitter_does_not_mutate_input():
      supervised = _supervised_rows()
      original = supervised.copy(deep=True)
      walk_forward_splits(supervised, VALIDATION_YEARS)
      pd.testing.assert_frame_equal(supervised, original)


  def test_empty_train_and_validation_partitions_raise_clearly():
      with pytest.raises(ValueError, match="TRAIN 2015 partition is empty"):
          walk_forward_splits(_supervised_rows(years=range(2015, 2017)), (2015,))
      with pytest.raises(ValueError, match="VALIDATION 2014 partition is empty"):
          walk_forward_splits(_supervised_rows(years=range(2013, 2014)), (2014,))
  ```

- [ ] **Step 3: Add the fold-construction future-isolation regression test**

  Append:

  ```python
  def test_changing_2023_input_cannot_change_folds_2016_through_2022():
      original = _supervised_rows()
      changed = original.copy(deep=True)
      mask_2023 = changed["date"].dt.year.eq(2023)
      changed.loc[mask_2023, FEATURE_COLUMNS[0]] += 999.0
      changed.loc[mask_2023, TARGET_COLUMN] -= 999.0

      before = walk_forward_splits(original, VALIDATION_YEARS)
      after = walk_forward_splits(changed, VALIDATION_YEARS)

      for before_fold, after_fold in zip(before[:-1], after[:-1]):
          assert before_fold.validation_year == after_fold.validation_year
          for before_partition, after_partition in (
              (before_fold.train, after_fold.train),
              (before_fold.validation, after_fold.validation),
          ):
              pd.testing.assert_frame_equal(before_partition.X, after_partition.X)
              pd.testing.assert_series_equal(before_partition.y, after_partition.y)
              pd.testing.assert_frame_equal(before_partition.metadata, after_partition.metadata)
  ```

  This calls the public splitter twice and compares complete observable
  partitions. The 2023 fold is intentionally excluded from equality.

- [ ] **Step 4: Run all splitter tests and observe RED behavior**

  Run:

  ```powershell
  python -m pytest tests/test_walk_forward.py -v
  ```

  Expected: any missing validation, boundary, immutability, or error-message
  behavior fails for its documented reason. Retain contract tests that Task 1
  already satisfies and continue to the remaining RED behavior.

- [ ] **Step 5: Add exact year-sequence validation for GREEN**

  Insert this block immediately after `years = tuple(validation_years)`:

  ```python
      if not years:
          raise ValueError("validation_years must not be empty")
      if any(type(year) is not int for year in years):
          raise ValueError("validation_years must contain only integers")
      if len(set(years)) != len(years):
          raise ValueError("validation_years must be unique")
      if any(left >= right for left, right in zip(years, years[1:])):
          raise ValueError("validation_years must be strictly increasing")
  ```

  Keep all behavior inside `walk_forward_splits`. Do not add SPY or 2024
  orchestration rules to generic dataset code.

- [ ] **Step 6: Run focused and regression tests**

  Run:

  ```powershell
  python -m pytest tests/test_walk_forward.py -v
  python -m pytest tests/test_dataset.py -v
  ```

  Expected: all walk-forward and existing dataset tests pass.

### Task 3: Add aligned fold evaluation with fresh model instances

**Files:**
- Create: `tests/test_evaluation.py`
- Create: `src/stock_forecaster/models/evaluation.py`

**Interfaces:**
- Consumes: `Sequence[WalkForwardFold]`, the four existing model classes, and
  `evaluate_predictions(pd.Series, pd.Series) -> RegressionMetrics`.
- Produces: frozen `ModelFoldResult` values and
  `evaluate_walk_forward(folds) -> tuple[ModelFoldResult, ...]`.

- [ ] **Step 1: Create focused evaluation fixtures**

  Start `tests/test_evaluation.py` with:

  ```python
  import math
  from dataclasses import FrozenInstanceError

  import pandas as pd
  import pytest

  import stock_forecaster.models.evaluation as evaluation_module
  from stock_forecaster.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN
  from stock_forecaster.models.baseline import MeanBaseline, MomentumBaseline
  from stock_forecaster.models.dataset import DatasetPartition, WalkForwardFold
  from stock_forecaster.models.evaluation import (
      ModelFoldResult,
      evaluate_walk_forward,
  )
  from stock_forecaster.models.forest import RandomForestForecaster
  from stock_forecaster.models.linear import LinearRegressionForecaster
  from stock_forecaster.models.metrics import RegressionMetrics, evaluate_predictions

  MODEL_NAMES = (
      "MeanBaseline",
      "MomentumBaseline",
      "LinearRegression",
      "RandomForest",
  )


  def _assert_metrics_equal(
      left: RegressionMetrics,
      right: RegressionMetrics,
  ) -> None:
      assert left.mae == pytest.approx(right.mae)
      assert left.rmse == pytest.approx(right.rmse)
      assert left.directional_accuracy == pytest.approx(right.directional_accuracy)
      if math.isnan(right.correlation):
          assert math.isnan(left.correlation)
      else:
          assert left.correlation == pytest.approx(right.correlation)


  def _partition(
      year: int,
      rows: int,
      *,
      start: int,
      target_shift: float = 0.0,
  ) -> DatasetPartition:
      index = pd.RangeIndex(rows)
      X = pd.DataFrame(
          {
              feature: [
                  ((start + row + feature_number) % 31) / 100.0
                  for row in range(rows)
              ]
              for feature_number, feature in enumerate(FEATURE_COLUMNS)
          },
          index=index,
      )
      X["return_5d"] = [((start + row) % 9 - 4) / 100.0 for row in range(rows)]
      y = pd.Series(
          [
              ((start + row * 3) % 13 - 6) / 100.0 + target_shift
              for row in range(rows)
          ],
          index=index,
          name=TARGET_COLUMN,
      )
      metadata = pd.DataFrame(
          {
              "date": pd.date_range(f"{year}-01-03", periods=rows, freq="B"),
              "ticker": ["AAA" if row % 2 == 0 else "BBB" for row in range(rows)],
          },
          index=index,
      )
      return DatasetPartition(X=X, y=y, metadata=metadata)


  def _fold(year: int, *, train_shift: float = 0.0) -> WalkForwardFold:
      return WalkForwardFold(
          validation_year=year,
          train=_partition(year - 1, 64, start=year, target_shift=train_shift),
          validation=_partition(year, 6, start=year + 100),
      )
  ```

  Keep feature columns in the imported order. Sixty-four training rows satisfy
  the fixed forest's leaf-size setting while remaining inexpensive.

- [ ] **Step 2: Add failing result, prediction, metric, and ordering tests**

  Append:

  ```python
  def test_two_folds_evaluate_four_models_each_in_canonical_order():
      folds = (_fold(2016), _fold(2017, train_shift=0.02))

      results = evaluate_walk_forward(folds)

      assert isinstance(results, tuple)
      assert [(result.validation_year, result.model_name) for result in results] == [
          (year, model) for year in (2016, 2017) for model in MODEL_NAMES
      ]
      assert all(isinstance(result, ModelFoldResult) for result in results)


  def test_prediction_records_preserve_exact_validation_association():
      fold = _fold(2016)

      results = evaluate_walk_forward((fold,))
      expected_predictions = {
          "MeanBaseline": MeanBaseline().fit(fold.train.y).predict(
              fold.validation.y.index
          ),
          "MomentumBaseline": MomentumBaseline().predict(fold.validation.X),
          "LinearRegression": LinearRegressionForecaster()
          .fit(fold.train.X, fold.train.y)
          .predict(fold.validation.X),
          "RandomForest": RandomForestForecaster()
          .fit(fold.train.X, fold.train.y)
          .predict(fold.validation.X),
      }

      for result in results:
          records = result.predictions
          assert list(records.columns) == [
              "date", "ticker", "validation_year", "model", "y_true", "prediction"
          ]
          assert len(records) == len(fold.validation.y)
          pd.testing.assert_frame_equal(
              records.loc[:, ["date", "ticker"]], fold.validation.metadata
          )
          pd.testing.assert_series_equal(
              records["y_true"], fold.validation.y, check_names=False
          )
          assert records["validation_year"].eq(2016).all()
          assert records["model"].eq(result.model_name).all()
          assert records.index.equals(fold.validation.y.index)
          pd.testing.assert_series_equal(
              records["prediction"],
              expected_predictions[result.model_name],
              check_names=False,
          )


  def test_stored_metrics_equal_direct_recomputation_and_keys_are_unique():
      results = evaluate_walk_forward((_fold(2016), _fold(2017)))

      for result in results:
          expected = evaluate_predictions(
              result.predictions["y_true"], result.predictions["prediction"]
          )
          _assert_metrics_equal(result.metrics, expected)
      combined = pd.concat([result.predictions for result in results])
      assert not combined.duplicated(["date", "ticker", "model"]).any()


  def test_model_fold_result_is_frozen():
      result = evaluate_walk_forward((_fold(2016),))[0]
      with pytest.raises(FrozenInstanceError):
          result.validation_year = 2020
  ```

- [ ] **Step 3: Add a meaningful fresh-instance test**

  Append:

  ```python
  def test_fitted_model_instances_are_fresh_for_every_fold(monkeypatch):
      created: dict[str, list[object]] = {"mean": [], "linear": [], "forest": []}

      class SpyMean:
          def __init__(self):
              created["mean"].append(self)

          def fit(self, y):
              self.fit_y = y
              self.value = float(y.mean())
              return self

          def predict(self, index):
              return pd.Series(self.value, index=index, name="prediction")

      def fitted_spy(kind):
          class SpyForecaster:
              def __init__(self):
                  created[kind].append(self)

              def fit(self, X, y):
                  self.fit_X = X
                  self.fit_y = y
                  self.value = float(y.mean())
                  return self

              def predict(self, X):
                  return pd.Series(self.value, index=X.index, name="prediction")

          return SpyForecaster

      monkeypatch.setattr(evaluation_module, "MeanBaseline", SpyMean)
      monkeypatch.setattr(
          evaluation_module, "LinearRegressionForecaster", fitted_spy("linear")
      )
      monkeypatch.setattr(
          evaluation_module, "RandomForestForecaster", fitted_spy("forest")
      )

      folds = (_fold(2016), _fold(2017, train_shift=0.03))
      evaluate_walk_forward(folds)

      assert all(len(instances) == 2 for instances in created.values())
      assert all(instances[0] is not instances[1] for instances in created.values())
      for index, fold in enumerate(folds):
          assert created["mean"][index].fit_y is fold.train.y
          for kind in ("linear", "forest"):
              assert created[kind][index].fit_X is fold.train.X
              assert created[kind][index].fit_y is fold.train.y
  ```

  Constructor identity is the clearest minimal observation: refitting a reused
  sklearn object could conceal state reuse. The test patches only module class
  references and adds no production injection interface.

- [ ] **Step 4: Run evaluation tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```

  Expected: collection fails because `evaluation.py`, `ModelFoldResult`, and
  `evaluate_walk_forward` do not exist.

- [ ] **Step 5: Implement the result contract and explicit four-model loop**

  Create `src/stock_forecaster/models/evaluation.py` with:

  ```python
  """Walk-forward model execution and out-of-sample metric aggregation."""

  from collections.abc import Sequence
  from dataclasses import dataclass

  import pandas as pd

  from stock_forecaster.models.baseline import MeanBaseline, MomentumBaseline
  from stock_forecaster.models.dataset import WalkForwardFold
  from stock_forecaster.models.forest import RandomForestForecaster
  from stock_forecaster.models.linear import LinearRegressionForecaster
  from stock_forecaster.models.metrics import RegressionMetrics, evaluate_predictions

  MODEL_NAMES = (
      "MeanBaseline",
      "MomentumBaseline",
      "LinearRegression",
      "RandomForest",
  )
  PREDICTION_COLUMNS = (
      "date",
      "ticker",
      "validation_year",
      "model",
      "y_true",
      "prediction",
  )


  @dataclass(frozen=True)
  class ModelFoldResult:
      """Predictions and metrics for one model in one validation year."""

      model_name: str
      validation_year: int
      predictions: pd.DataFrame
      metrics: RegressionMetrics


  def _result(
      fold: WalkForwardFold,
      model_name: str,
      prediction: pd.Series,
  ) -> ModelFoldResult:
      if not prediction.index.equals(fold.validation.y.index):
          raise ValueError("prediction index must match validation y exactly")
      records = fold.validation.metadata.copy()
      records["validation_year"] = fold.validation_year
      records["model"] = model_name
      records["y_true"] = fold.validation.y
      records["prediction"] = prediction
      records = records.loc[:, PREDICTION_COLUMNS]
      return ModelFoldResult(
          model_name=model_name,
          validation_year=fold.validation_year,
          predictions=records,
          metrics=evaluate_predictions(fold.validation.y, prediction),
      )


  def evaluate_walk_forward(
      folds: Sequence[WalkForwardFold],
  ) -> tuple[ModelFoldResult, ...]:
      """Evaluate the four approved models independently in every fold."""
      results: list[ModelFoldResult] = []
      for fold in folds:
          mean = MeanBaseline().fit(fold.train.y)
          linear = LinearRegressionForecaster().fit(fold.train.X, fold.train.y)
          forest = RandomForestForecaster().fit(fold.train.X, fold.train.y)
          predictions = (
              ("MeanBaseline", mean.predict(fold.validation.y.index)),
              ("MomentumBaseline", MomentumBaseline().predict(fold.validation.X)),
              ("LinearRegression", linear.predict(fold.validation.X)),
              ("RandomForest", forest.predict(fold.validation.X)),
          )
          results.extend(
              _result(fold, model_name, prediction)
              for model_name, prediction in predictions
          )

      if results:
          all_predictions = pd.concat(
              [result.predictions for result in results], ignore_index=True
          )
          if all_predictions.duplicated(["date", "ticker", "model"]).any():
              raise ValueError("OOS date, ticker, and model keys must be unique")
      return tuple(results)
  ```

  Do not export these through `models/__init__.py`; current project convention
  imports public model components from focused modules.

- [ ] **Step 6: Run focused evaluation tests and verify GREEN**

  Run:

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```

  Expected: all orchestration, fresh-instance, alignment, uniqueness, and metric
  tests pass.

### Task 4: Add fold, pooled, and consistency aggregations

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `src/stock_forecaster/models/evaluation.py`

**Interfaces:**
- Consumes: `Sequence[ModelFoldResult]` from Task 3.
- Produces: `build_fold_metrics_table`, `build_pooled_metrics`, and
  `build_consistency_summary` with exact deterministic DataFrame schemas.

- [ ] **Step 1: Add manual-result helpers and fold-table tests**

  First expand the `stock_forecaster.models.evaluation` import with
  `build_consistency_summary`, `build_fold_metrics_table`, and
  `build_pooled_metrics`. Then append:

  ```python
  def _manual_result(
      year: int,
      model: str,
      y_true: list[float],
      prediction: list[float],
  ) -> ModelFoldResult:
      index = pd.RangeIndex(len(y_true))
      actual = pd.Series(y_true, index=index, name="y_true")
      predicted = pd.Series(prediction, index=index, name="prediction")
      records = pd.DataFrame(
          {
              "date": pd.date_range(f"{year}-02-01", periods=len(index), freq="B"),
              "ticker": ["AAA"] * len(index),
              "validation_year": [year] * len(index),
              "model": [model] * len(index),
              "y_true": actual,
              "prediction": predicted,
          },
          index=index,
      )
      return ModelFoldResult(
          model, year, records, evaluate_predictions(actual, predicted)
      )


  def test_fold_metrics_table_has_exact_schema_count_and_order(monkeypatch):
      class FastForest:
          def fit(self, X, y):
              self.value = float(y.mean())
              return self

          def predict(self, X):
              return pd.Series(self.value, index=X.index, name="prediction")

      monkeypatch.setattr(evaluation_module, "RandomForestForecaster", FastForest)
      results = evaluate_walk_forward(
          tuple(_fold(year) for year in range(2016, 2024))
      )

      table = build_fold_metrics_table(tuple(reversed(results)))

      assert list(table.columns) == [
          "validation_year", "model", "MAE", "RMSE",
          "directional_accuracy", "correlation",
      ]
      assert len(table) == 32
      assert list(table[["validation_year", "model"]].itertuples(index=False, name=None)) == [
          (year, model) for year in range(2016, 2024) for model in MODEL_NAMES
      ]
  ```

- [ ] **Step 2: Add behavioral pooled-metric tests**

  Append:

  ```python
  def test_pooled_metrics_recompute_after_concatenation_not_annual_averaging():
      results: list[ModelFoldResult] = []
      for model in MODEL_NAMES:
          results.extend(
              [
                  _manual_result(2016, model, [0.0, 1.0], [0.0, 1.0]),
                  _manual_result(
                      2017, model, [10.0, 20.0, 30.0], [30.0, 10.0, 20.0]
                  ),
              ]
          )

      table = build_pooled_metrics(results)
      row = table.set_index("model").loc["RandomForest"]
      expected = evaluate_predictions(
          pd.Series([0.0, 1.0, 10.0, 20.0, 30.0]),
          pd.Series([0.0, 1.0, 30.0, 10.0, 20.0]),
      )
      annual = [
          result.metrics for result in results if result.model_name == "RandomForest"
      ]

      assert list(table.columns) == [
          "model", "MAE", "RMSE", "directional_accuracy", "correlation"
      ]
      assert table["model"].tolist() == list(MODEL_NAMES)
      assert row["RMSE"] == pytest.approx(expected.rmse)
      assert row["correlation"] == pytest.approx(expected.correlation)
      assert row["RMSE"] != pytest.approx(sum(item.rmse for item in annual) / 2)
      assert row["correlation"] != pytest.approx(
          sum(item.correlation for item in annual) / 2
      )


  def test_mean_baseline_fold_nan_can_become_finite_pooled_correlation():
      mean_results = (
          _manual_result(2016, "MeanBaseline", [-2.0, -1.0], [-1.5, -1.5]),
          _manual_result(2017, "MeanBaseline", [1.0, 2.0], [1.5, 1.5]),
      )
      results = tuple(
          result
          for model in MODEL_NAMES
          for result in (
              mean_results
              if model == "MeanBaseline"
              else (
                  _manual_result(2016, model, [-2.0, -1.0], [-1.0, -2.0]),
                  _manual_result(2017, model, [1.0, 2.0], [2.0, 1.0]),
              )
          )
      )

      assert all(math.isnan(result.metrics.correlation) for result in mean_results)
      pooled = build_pooled_metrics(results).set_index("model")
      assert math.isfinite(pooled.loc["MeanBaseline", "correlation"])
  ```

- [ ] **Step 3: Add strict consistency-summary tests**

  Append:

  ```python
  def _metric_result(
      year: int,
      model: str,
      *,
      mae: float,
      rmse: float,
      direction: float,
      correlation: float,
  ) -> ModelFoldResult:
      records = pd.DataFrame(
          columns=["date", "ticker", "validation_year", "model", "y_true", "prediction"]
      )
      return ModelFoldResult(
          model,
          year,
          records,
          RegressionMetrics(mae, rmse, direction, correlation),
      )


  def test_consistency_summary_uses_strict_fold_comparisons():
      results = (
          _metric_result(2016, "MeanBaseline", mae=2, rmse=3, direction=0.5, correlation=float("nan")),
          _metric_result(2016, "MomentumBaseline", mae=1, rmse=3, direction=0.6, correlation=0.1),
          _metric_result(2016, "LinearRegression", mae=2, rmse=2, direction=0.5, correlation=0.0),
          _metric_result(2016, "RandomForest", mae=3, rmse=4, direction=0.4, correlation=float("nan")),
          _metric_result(2017, "MeanBaseline", mae=2, rmse=3, direction=0.5, correlation=float("nan")),
          _metric_result(2017, "MomentumBaseline", mae=2, rmse=2, direction=0.5, correlation=-0.1),
          _metric_result(2017, "LinearRegression", mae=1, rmse=3, direction=0.6, correlation=0.2),
          _metric_result(2017, "RandomForest", mae=1, rmse=2, direction=0.7, correlation=0.3),
      )

      table = build_consistency_summary(results)

      assert list(table.columns) == [
          "model", "folds_better_mae_vs_mean", "folds_better_rmse_vs_mean",
          "folds_better_direction_vs_mean", "folds_positive_correlation",
      ]
      assert table["model"].tolist() == list(MODEL_NAMES)
      counts = table.set_index("model")
      assert counts.loc["MeanBaseline"].tolist() == [0, 0, 0, 0]
      assert counts.loc["MomentumBaseline"].tolist() == [1, 1, 1, 1]
      assert counts.loc["LinearRegression"].tolist() == [1, 1, 1, 1]
      assert counts.loc["RandomForest"].tolist() == [1, 1, 1, 1]
  ```

  Equal values prove ties are excluded; zero, negative, and NaN correlations
  prove only values strictly greater than zero count.

- [ ] **Step 4: Add fixed-forest reproducibility and evaluation-isolation tests**

  Append:

  ```python
  def test_repeated_evaluation_has_reproducible_random_forest_predictions():
      fold = _fold(2016)
      first = evaluate_walk_forward((fold,))
      second = evaluate_walk_forward((fold,))
      first_forest = next(
          result for result in first if result.model_name == "RandomForest"
      )
      second_forest = next(
          result for result in second if result.model_name == "RandomForest"
      )

      pd.testing.assert_series_equal(
          first_forest.predictions["prediction"],
          second_forest.predictions["prediction"],
      )


  def test_2023_changes_cannot_alter_2016_through_2022_evaluation(monkeypatch):
      class FastForest:
          def fit(self, X, y):
              self.value = float(y.mean())
              return self

          def predict(self, X):
              return pd.Series(self.value, index=X.index, name="prediction")

      monkeypatch.setattr(evaluation_module, "RandomForestForecaster", FastForest)
      before_folds = tuple(_fold(year) for year in range(2016, 2024))
      after_folds = list(before_folds)
      after_folds[-1] = WalkForwardFold(
          2023,
          before_folds[-1].train,
          _partition(2023, 6, start=999, target_shift=5.0),
      )

      before = evaluate_walk_forward(before_folds)
      after = evaluate_walk_forward(tuple(after_folds))
      before_early = [result for result in before if result.validation_year <= 2022]
      after_early = [result for result in after if result.validation_year <= 2022]

      assert len(before_early) == len(after_early) == 28
      for left, right in zip(before_early, after_early):
          assert left.model_name == right.model_name
          _assert_metrics_equal(left.metrics, right.metrics)
          pd.testing.assert_frame_equal(left.predictions, right.predictions)
  ```

  The first test uses the real fixed Random Forest. The isolation test uses a
  deterministic lightweight forest substitute to exercise public four-model
  orchestration across eight folds efficiently; Task 2 independently exercises
  public fold construction from changed source rows.

- [ ] **Step 5: Run aggregation tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```

  Expected: aggregation tests fail because the three builder functions do not
  exist; Task 3 orchestration tests remain green.

- [ ] **Step 6: Implement deterministic aggregation functions**

  Append to `evaluation.py`:

  ```python
  def _ordered_results(
      results: Sequence[ModelFoldResult],
  ) -> list[ModelFoldResult]:
      return sorted(
          results,
          key=lambda result: (
              result.validation_year,
              MODEL_NAMES.index(result.model_name),
          ),
      )


  def _metrics_row(model: str, metrics: RegressionMetrics) -> dict[str, object]:
      return {
          "model": model,
          "MAE": metrics.mae,
          "RMSE": metrics.rmse,
          "directional_accuracy": metrics.directional_accuracy,
          "correlation": metrics.correlation,
      }


  def build_fold_metrics_table(
      results: Sequence[ModelFoldResult],
  ) -> pd.DataFrame:
      """Return one deterministic metric row per model and validation year."""
      rows = [
          {
              "validation_year": result.validation_year,
              **_metrics_row(result.model_name, result.metrics),
          }
          for result in _ordered_results(results)
      ]
      return pd.DataFrame(
          rows,
          columns=[
              "validation_year", "model", "MAE", "RMSE",
              "directional_accuracy", "correlation",
          ],
      )


  def build_pooled_metrics(
      results: Sequence[ModelFoldResult],
  ) -> pd.DataFrame:
      """Recompute each model's metrics from concatenated OOS rows."""
      rows: list[dict[str, object]] = []
      for model_name in MODEL_NAMES:
          pooled = pd.concat(
              [
                  result.predictions
                  for result in _ordered_results(results)
                  if result.model_name == model_name
              ],
              ignore_index=True,
          )
          metrics = evaluate_predictions(pooled["y_true"], pooled["prediction"])
          rows.append(_metrics_row(model_name, metrics))
      return pd.DataFrame(
          rows,
          columns=["model", "MAE", "RMSE", "directional_accuracy", "correlation"],
      )


  def build_consistency_summary(
      results: Sequence[ModelFoldResult],
  ) -> pd.DataFrame:
      """Count strict annual comparisons with the same-year MeanBaseline."""
      metrics = build_fold_metrics_table(results)
      mean = metrics.loc[metrics["model"].eq("MeanBaseline")].set_index(
          "validation_year"
      )
      rows: list[dict[str, object]] = []
      for model_name in MODEL_NAMES:
          model = metrics.loc[metrics["model"].eq(model_name)].set_index(
              "validation_year"
          )
          comparison = model.join(mean, rsuffix="_mean", validate="one_to_one")
          rows.append(
              {
                  "model": model_name,
                  "folds_better_mae_vs_mean": int(
                      comparison["MAE"].lt(comparison["MAE_mean"]).sum()
                  ),
                  "folds_better_rmse_vs_mean": int(
                      comparison["RMSE"].lt(comparison["RMSE_mean"]).sum()
                  ),
                  "folds_better_direction_vs_mean": int(
                      comparison["directional_accuracy"].gt(
                          comparison["directional_accuracy_mean"]
                      ).sum()
                  ),
                  "folds_positive_correlation": int(
                      comparison["correlation"].gt(0).sum()
                  ),
              }
          )
      return pd.DataFrame(rows)
  ```

  `Series.gt(0)` treats NaN as false. Do not add a composite score or special
  handling for MeanBaseline pooled correlation.

- [ ] **Step 7: Run all evaluation and model-contract tests**

  Run:

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  python -m pytest tests/test_walk_forward.py -v
  python -m pytest tests/test_metrics.py tests/test_baseline.py tests/test_linear.py tests/test_forest.py -v
  ```

  Expected: all new and existing model-contract tests pass.

### Task 5: Document the methodology, verify fully, and create the feature commit

**Files:**
- Modify: `README.md`
- Verify: `src/stock_forecaster/models/dataset.py`
- Verify: `src/stock_forecaster/models/evaluation.py`
- Verify: `tests/test_walk_forward.py`
- Verify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: completed splitter, evaluation, and aggregation APIs.
- Produces: accurate documentation and one verified implementation commit,
  `feat: add expanding walk-forward validation`.

- [ ] **Step 1: Update README without real Step 5E numbers**

  Replace the model-evaluation progression with:

  ```text
  Single temporal validation split
          ↓
  Annual expanding walk-forward validation (2016-2023)
          ↓
  Pooled out-of-sample model comparison
          ↓
  Future modeling decisions
  ```

  Add this paragraph after the current Random Forest description:

  ```markdown
  Model stability is evaluated with annual 2016-2023 validation folds and an
  expanding TRAIN window. TRAIN and each annual VALIDATION partition are purged
  using the ticker-specific `target_end_date`. The four existing models produce
  in-memory out-of-sample predictions; pooled metrics are recomputed from their
  concatenated prediction rows rather than averaged across years. TEST remains
  2024+ and untouched.
  ```

  Update Current status to mention expanding walk-forward validation and pooled
  OOS evaluation infrastructure. Do not include numerical results before Task 6.

- [ ] **Step 2: Verify imports for all new public components**

  Run:

  ```powershell
  python -c "from stock_forecaster.models.dataset import WalkForwardFold, walk_forward_splits; from stock_forecaster.models.evaluation import ModelFoldResult, evaluate_walk_forward, build_fold_metrics_table, build_pooled_metrics, build_consistency_summary; print('walk-forward imports: OK')"
  ```

  Expected: exit code 0 and `walk-forward imports: OK`.

- [ ] **Step 3: Run focused tests, full suite, and Ruff freshly**

  Run:

  ```powershell
  python -m pytest tests/test_walk_forward.py -v
  python -m pytest tests/test_evaluation.py -v
  python -m pytest -v
  python -m ruff check .
  ```

  Expected: both focused files pass, the complete suite has zero failures, and
  Ruff reports success. Focused success alone is insufficient.

- [ ] **Step 4: Audit scope, generated files, and dataset safety before staging**

  Run:

  ```powershell
  git status --short
  git diff --check
  git diff -- pyproject.toml configs/market.yaml src/stock_forecaster/models/baseline.py src/stock_forecaster/models/linear.py src/stock_forecaster/models/forest.py src/stock_forecaster/models/metrics.py
  git status --short -- data/raw data/processed models
  ```

  Expected: only `dataset.py`, `evaluation.py`, the two new test files, and
  README.md are changed. The protected-file and dataset commands print no diff
  or status. The already committed plan is not part of the feature commit.

- [ ] **Step 5: Create the single implementation commit**

  Run:

  ```powershell
  git add src/stock_forecaster/models/dataset.py src/stock_forecaster/models/evaluation.py tests/test_walk_forward.py tests/test_evaluation.py README.md
  git diff --cached --check
  git diff --cached --name-only
  git commit -m "feat: add expanding walk-forward validation"
  ```

  Expected: exactly those five files are staged and committed. Do not stage
  dependency files, model artifacts, Parquet files, or prediction tables. Do
  not push.

### Task 6: Run one read-only real-data walk-forward evaluation

**Files:**
- Read: `data/raw/prices.parquet`
- Read: `data/processed/features.parquet`
- Create: no file
- Modify: no file

**Interfaces:**
- Consumes: committed Step 5E APIs, persisted features, the exact stock
  universe, and approved 2016-2023 folds.
- Produces: console-only fold audits, pooled OOS metrics, annual metrics,
  consistency counts, and bounded interpretation; repository state remains
  unchanged.

- [ ] **Step 1: Record pre-evaluation SHA-256 hashes**

  Run:

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  ```

  Record both exact values outside the repository. Do not create an audit file.

- [ ] **Step 2: Execute the approved evaluation once, entirely in memory**

  Run from the repository root:

  ```powershell
  @'
  import pandas as pd

  from stock_forecaster.models.dataset import prepare_supervised_data, walk_forward_splits
  from stock_forecaster.models.evaluation import (
      build_consistency_summary,
      build_fold_metrics_table,
      build_pooled_metrics,
      evaluate_walk_forward,
  )

  STOCKS = {
      "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
      "JNJ", "UNH", "XOM", "CVX", "COST", "WMT", "PG",
  }
  YEARS = tuple(range(2016, 2024))
  DEVELOPMENT_END = pd.Timestamp("2024-01-01")

  features = pd.read_parquet("data/processed/features.parquet")
  modeling = features.loc[features["ticker"].isin(STOCKS)].copy()
  assert set(modeling["ticker"].unique()) == STOCKS
  assert "SPY" not in set(modeling["ticker"])

  supervised = prepare_supervised_data(modeling)
  development = supervised.loc[
      supervised["date"].lt(DEVELOPMENT_END)
      & supervised["target_end_date"].lt(DEVELOPMENT_END)
  ].copy()
  assert development["date"].lt(DEVELOPMENT_END).all()
  assert development["target_end_date"].lt(DEVELOPMENT_END).all()
  assert "SPY" not in set(development["ticker"])

  folds = walk_forward_splits(development, YEARS)
  assert len(folds) == 8
  assert tuple(fold.validation_year for fold in folds) == YEARS

  audit_rows = []
  horizon_source = development.loc[:, ["date", "ticker", "target_end_date"]]
  for fold in folds:
      assert set(fold.train.metadata["ticker"].unique()) == STOCKS
      assert set(fold.validation.metadata["ticker"].unique()) == STOCKS
      assert "SPY" not in set(fold.train.metadata["ticker"])
      assert "SPY" not in set(fold.validation.metadata["ticker"])
      train_horizons = fold.train.metadata.merge(
          horizon_source,
          on=["date", "ticker"],
          how="left",
          validate="one_to_one",
          sort=False,
      )["target_end_date"]
      validation_horizons = fold.validation.metadata.merge(
          horizon_source,
          on=["date", "ticker"],
          how="left",
          validate="one_to_one",
          sort=False,
      )["target_end_date"]
      validation_start = pd.Timestamp(fold.validation_year, 1, 1)
      next_year_start = pd.Timestamp(fold.validation_year + 1, 1, 1)
      assert fold.train.metadata["date"].lt(validation_start).all()
      assert train_horizons.lt(validation_start).all()
      assert fold.validation.metadata["date"].ge(validation_start).all()
      assert fold.validation.metadata["date"].lt(next_year_start).all()
      assert validation_horizons.lt(next_year_start).all()
      audit_rows.append(
          {
              "validation_year": fold.validation_year,
              "train_rows": len(fold.train.y),
              "validation_rows": len(fold.validation.y),
              "earliest_train_date": fold.train.metadata["date"].min(),
              "latest_train_date": fold.train.metadata["date"].max(),
              "earliest_validation_date": fold.validation.metadata["date"].min(),
              "latest_validation_date": fold.validation.metadata["date"].max(),
              "max_train_target_end_date": train_horizons.max(),
              "max_validation_target_end_date": validation_horizons.max(),
          }
      )

  results = evaluate_walk_forward(folds)
  assert len(results) == 32
  pooled = build_pooled_metrics(results)
  annual = build_fold_metrics_table(results)
  consistency = build_consistency_summary(results)

  print("FOLD AUDIT")
  print(pd.DataFrame(audit_rows).to_string(index=False))
  print("PRIMARY POOLED OOS METRICS")
  print(pooled.to_string(index=False))
  print("ANNUAL FOLD METRICS")
  print(annual.to_string(index=False))
  print("CONSISTENCY SUMMARY")
  print(consistency.to_string(index=False))
  '@ | python -
  ```

  Expected: eight audited folds, 32 results, four pooled rows, 32 annual rows,
  and four consistency rows. No TEST row, statistic, date range, prediction,
  metric, or diagnostic is printed or used. Persist nothing and do not rerun
  with different model settings.

- [ ] **Step 3: Interpret only permitted development evidence**

  Report best pooled OOS MAE and RMSE, relative pooled directional accuracy and
  correlation, annual stability, fold wins over MeanBaseline, regime
  dependence, and whether current features/models show consistent signal. Do
  not claim profitability, investability, causality, statistical significance,
  economic significance, TEST performance, or backtest results. Create no
  combined winner score.

- [ ] **Step 4: Recalculate hashes and require exact equality**

  Run:

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  ```

  Expected: both post-evaluation values exactly equal their recorded
  pre-evaluation values. If either differs, stop immediately and report the
  unexpected mutation.

- [ ] **Step 5: Run fresh post-evaluation verification and prove cleanliness**

  Run:

  ```powershell
  python -m pytest -v
  python -m ruff check .
  git status --short
  git diff --exit-code
  git diff --cached --exit-code
  ```

  Expected: the complete suite passes, Ruff succeeds, status prints nothing,
  and both diff commands exit 0. No evaluation output or dataset is committed;
  do not push.

## Test Coverage Map

`tests/test_walk_forward.py` covers all 23 approved dataset behaviors: eight
2016-2023 folds; chronological order and correct year labels; observable TRAIN
growth; exact TRAIN and VALIDATION date boundaries and ticker-specific target
purges; the late-2023/2024 exclusion; no retained development date or target end
in 2024; exact X, y, and metadata contracts; alignment, uniqueness, and input
immutability; rejection of empty, duplicate, unordered, non-integer, and boolean
year inputs; empty TRAIN/VALIDATION rejection; and 2023-change isolation.

`tests/test_evaluation.py` covers all 27 requested evaluation behaviors: four
models per fold and 32 approved results; fresh fitted instances; canonical
names; exact prediction schema, constants, metadata/y/prediction alignment,
row counts, and OOS uniqueness; direct metric recomputation; deterministic fold
table schema/count/order; concatenation-first pooled RMSE and correlation; four
pooled rows; finite pooled MeanBaseline correlation; strict MAE, RMSE, and
directional comparisons; tie and NaN handling; positive-correlation counts;
fixed-forest reproducibility; and unchanged 2016-2022 outputs after a 2023
change. Assertions are grouped only where they describe one coherent contract.

## Plan Self-Review

Every approved specification requirement maps to Tasks 1-6. The plan fixes all
eight 2016-2023 folds, proves observable expanding TRAIN contents, applies exact
TRAIN and VALIDATION target-end purges, tests the 2023/2024 boundary, and filters
the development horizon before fold construction. It reuses `DatasetPartition`,
keeps models out of `dataset.py`, and keeps splitting, SPY filtering, Parquet
loading, and persistence out of `evaluation.py`.

The plan creates fresh instances for exactly the four current models, uses one
canonical name set, preserves the exact six-column prediction schema, and
computes every fold metric through `evaluate_predictions`. Pooled metrics are
recomputed after prediction concatenation and never average annual values; the
finite pooled MeanBaseline correlation case is tested without special handling.
Consistency comparisons are strict, ties and NaN do not count, and no composite
score exists. Public fold construction and public evaluation both have
future-data isolation regression coverage.

No existing model behavior, model configuration, feature, dependency, TEST
data, or dataset is changed. Execution hashes both persisted datasets, persists
no predictions, creates one implementation commit only after full verification,
and finishes with clean Git checks. All interfaces and type names match the
approved specification and current source. Each production change has a
concrete RED observation, minimal GREEN guidance, focused verification, and
full-suite verification; the document contains no incomplete instructions or
open architectural decisions.
