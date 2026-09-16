# SPY-Relative Excess Return Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SPY-relative five-session excess-return target while preserving the absolute-return workflow, then evaluate its learnability with the existing 2016-2023 expanding walk-forward method.

**Architecture:** `engineering.py` adds one target by looking up SPY adjusted close on each stock row's exact start and ticker-local fifth-future end dates; SPY never enters `FEATURE_COLUMNS`. `dataset.py` explicitly selects one of the two approved targets while retaining the existing split and purge contracts. `baseline.py` adds a stateless zero predictor, and `evaluation.py` accepts an explicit model subset plus a configurable same-year consistency reference while preserving Step 5E defaults. After TDD, full verification, and one implementation commit, a single in-memory controlled-migration/regression process protects every old processed value and all Step 5E outputs before any Step 5F metric is observed.

**Tech Stack:** Python 3.12, pandas, scikit-learn, pytest, Ruff, PyArrow through the existing pandas Parquet support

**Spec:** `docs/superpowers/specs/2026-09-16-excess-return-target-design.md`

## Global Constraints

- Keep `build_features(prices)` unchanged as a public signature. Define `EXCESS_TARGET_COLUMN = "future_excess_return_5d"`; append it after `future_return_5d` in processed output.
- Preserve all ten `FEATURE_COLUMNS`, their order, formulas, warm-up behavior, and values. SPY return, ticker, dates, targets, and target horizon never enter X.
- Compute each benchmark return on the stock row's exact `date -> ticker fifth-future date` window with `adjusted_close` at both endpoints. Never use SPY's independent fifth-future observation or impute/substitute an endpoint.
- Preserve `future_return_5d` exactly. Missing exact SPY start or end makes only the excess target null. Complete SPY horizons yield zero within floating-point tolerance; final five ticker horizons remain null.
- Permit only `TARGET_COLUMN` and `EXCESS_TARGET_COLUMN` in `prepare_supervised_data`; default to `TARGET_COLUMN`. Prepared frames carry exactly one approved target.
- Keep `temporal_split` and `walk_forward_splits` signatures unchanged. Infer the one approved target present, reject zero or two approved targets, and preserve X/y/metadata alignment and target-end purging.
- Add stateless `ZeroBaseline`; do not modify `MeanBaseline`, `MomentumBaseline`, linear regression, Random Forest, or metric behavior.
- Default `evaluate_walk_forward(folds)` to the original Step 5E models and order. Step 5F explicitly requests ZeroBaseline, MeanBaseline, LinearRegression, and RandomForest. Do not create a registry or framework.
- Change consistency input to the annual fold-metrics DataFrame and default reference to MeanBaseline. Rename the three comparison columns exclusively to `*_vs_reference`.
- Use RED-GREEN TDD for every behavior. Add no dependency, predictor, trainable model, tuning, model artifact, prediction artifact, CLI, experiment framework, or report file.
- TEST remains 2024+ and supplies no fitting, metric, diagnostic, comparison, or decision evidence. Do not backtest or do portfolio work.
- `src/stock_forecaster/models/linear.py`, `forest.py`, and `metrics.py` remain textually and behaviorally unchanged. Stop before modifying them if an existing defect blocks the approved design.
- Implementation occurs later in an isolated worktree/feature branch under the approved execution workflow. This planning task creates no worktree.
- The future implementation phase creates one code commit only: `feat: add SPY-relative excess return target`. The ignored processed-data migration is not committed. Never push.

## File Map

- Modify: `src/stock_forecaster/features/engineering.py` — excess-target constant, exact SPY lookups, additive output column.
- Modify: `src/stock_forecaster/models/dataset.py` — approved target selection and target-aware partitioning.
- Modify: `src/stock_forecaster/models/baseline.py` — stateless `ZeroBaseline`.
- Modify: `src/stock_forecaster/models/evaluation.py` — model-set validation/resolution, dynamic aggregation, fold-metrics consistency input/reference.
- Modify: `tests/test_features.py` — exact-window, missing-endpoint, zero-SPY, old-target/schema/immutability tests.
- Modify: `tests/test_dataset.py` — target selection, selected-target completeness, ambiguity, alignment, and purge tests.
- Modify: `tests/test_baseline.py` — ZeroBaseline contract.
- Modify: `tests/test_evaluation.py` — default/explicit model sets, validation, order, fresh instances, dynamic tables, consistency migration.
- Modify only if compatibility coverage is necessary: `tests/test_walk_forward.py` — both selected target names through unchanged folds; no algorithm change.
- Modify: `README.md` — both targets, exact SPY window, ZeroBaseline, official Step 5F models, unchanged walk-forward/TEST boundaries, and controlled-evaluation status without pre-emptive numerical results.
- Read only: `src/stock_forecaster/models/linear.py`, `forest.py`, `metrics.py`, `configs/market.yaml`.
- Controlled ignored-data replacement after the implementation commit: `data/processed/features.parquet`.
- Immutable/read only: `data/raw/prices.parquet`.
- Never create: persisted predictions, folds, result CSV/Parquet/JSON, model artifacts, or additional dependencies.

---

### Task 1: Add the exact-window excess target without changing existing features

**Files:**
- Modify: `tests/test_features.py`
- Modify: `src/stock_forecaster/features/engineering.py`

**Interfaces:**
- Consumes: normalized prices containing `date`, `ticker`, `adjusted_close`, and `volume`, including SPY when an excess target can be calculated.
- Produces: `date`, `ticker`, the unchanged ten `FEATURE_COLUMNS`, `future_return_5d`, then `future_excess_return_5d`.

- [ ] **Step 1: Add deterministic multi-calendar test data and schema protection**

  Import `numpy as np` and extend the existing tests rather than replacing them. Add a helper whose stock calendar intentionally differs from SPY, with at least six future stock observations and explicitly chosen prices. In `test_input_is_unchanged_and_output_contract_is_exact`, assert:

  ```python
  assert module.EXCESS_TARGET_COLUMN == "future_excess_return_5d"
  assert result.columns.tolist() == [
      "date", "ticker", *EXPECTED_FEATURES,
      "future_return_5d", "future_excess_return_5d",
  ]
  assert list(module.FEATURE_COLUMNS) == EXPECTED_FEATURES
  ```

  Preserve the deep input equality assertion. Snapshot the old feature/target columns from a single-ticker fixture and assert they are identical after SPY rows are added, so benchmark construction cannot change existing calculations.

- [ ] **Step 2: Write failing exact-window and subtraction tests**

  Add separate tests with descriptive names:

  ```python
  def test_excess_target_uses_stock_fifth_future_date_for_both_returns(): ...
  def test_spy_independent_fifth_observation_is_not_substituted(): ...
  def test_complete_spy_horizon_has_zero_excess(): ...
  ```

  For a chosen stock row, calculate expected values directly in the test:

  ```python
  stock_return = stock_end / stock_start - 1
  spy_return = spy_at_stock_end / spy_at_stock_start - 1
  assert row[TARGET_COLUMN] == pytest.approx(stock_return)
  assert row[EXCESS_TARGET_COLUMN] == pytest.approx(stock_return - spy_return)
  ```

  Include another SPY trading date between the start and stock end so `SPY.shift(-5)` points somewhere else. Assert the exact-date expectation and explicitly assert it differs from the independent-SPY-window result.

- [ ] **Step 3: Write failing missing-endpoint, no-fill, tail, and finiteness tests**

  Add:

  ```python
  def test_missing_exact_spy_start_is_nan_without_backward_fill(): ...
  def test_missing_exact_spy_end_is_nan_without_forward_fill(): ...
  def test_final_five_horizons_are_nan_for_both_targets(): ...
  def test_non_null_excess_targets_are_finite(): ...
  ```

  The start/end fixtures must have adjacent SPY dates with valid prices so a fill would produce a number; assert the excess target is nevertheless `NaN`. Retain the existing absolute-target assertion on those rows. Assert the last five rows per ticker are null for both targets and use `np.isfinite(result[EXCESS_TARGET_COLUMN].dropna()).all()`.

- [ ] **Step 4: Run the feature tests and observe RED**

  Run:

  ```powershell
  python -m pytest tests/test_features.py -v
  ```

  Expected RED: imports/attributes for `EXCESS_TARGET_COLUMN` fail and the additive output/exact-window assertions fail. Existing feature/absolute-target tests should remain green; if they do not, fix the test fixture before production code.

- [ ] **Step 5: Implement the minimal exact-date lookup**

  In `engineering.py`, retain the existing `_relative_change` and feature code, and make only these structural additions:

  ```python
  EXCESS_TARGET_COLUMN = "future_excess_return_5d"
  _OUTPUT_COLUMNS = (
      "date", "ticker", *FEATURE_COLUMNS, TARGET_COLUMN, EXCESS_TARGET_COLUMN,
  )

  target_end_date = ordered.groupby("ticker", sort=False)["date"].shift(-5)
  result[TARGET_COLUMN] = _relative_change(price_groups.shift(-5), price)

  spy_prices = (
      ordered.loc[ordered["ticker"].eq("SPY"), ["date", "adjusted_close"]]
      .set_index("date")["adjusted_close"]
  )
  spy_start = ordered["date"].map(spy_prices)
  spy_end = target_end_date.map(spy_prices)
  spy_return = _relative_change(spy_end, spy_start)
  result[EXCESS_TARGET_COLUMN] = result[TARGET_COLUMN] - spy_return
  ```

  Do not special-case SPY, fill values, add `target_end_date` to output, or calculate a benchmark feature. Existing duplicate `(date, ticker)` validation guarantees the lookup index is unique.

- [ ] **Step 6: Run focused GREEN and regression assertions**

  ```powershell
  python -m pytest tests/test_features.py -v
  ```

  Expected: all feature tests pass, including unchanged absolute target, unchanged `FEATURE_COLUMNS`, cross-ticker isolation, future-price leakage protection, warm-up NaNs, zero-volume handling, exact benchmark endpoints, missing endpoints, SPY zero, input immutability, and finite non-null excess values.

### Task 2: Make supervised preparation explicitly target-selectable

**Files:**
- Modify: `tests/test_dataset.py`
- Modify if needed only for compatibility: `tests/test_walk_forward.py`
- Modify: `src/stock_forecaster/models/dataset.py`

**Interfaces:**
- Produces `prepare_supervised_data(features, target_column=TARGET_COLUMN)` and unchanged temporal split signatures.
- Prepared schema is `date`, `ticker`, `target_end_date`, `FEATURE_COLUMNS`, and exactly the selected approved target.

- [ ] **Step 1: Extend synthetic processed rows with the excess target**

  Import `EXCESS_TARGET_COLUMN`. Make `_processed_rows` populate both targets with distinct values, allowing individual tests to drop either column. Existing default tests must continue to expect only `TARGET_COLUMN` in prepared output.

- [ ] **Step 2: Write failing selection and validation tests**

  Add explicit tests for:

  ```python
  default = prepare_supervised_data(frame)
  absolute = prepare_supervised_data(frame, target_column=TARGET_COLUMN)
  excess = prepare_supervised_data(frame, target_column=EXCESS_TARGET_COLUMN)
  pd.testing.assert_frame_equal(default, absolute)
  assert excess.columns.tolist() == [
      "date", "ticker", "target_end_date", *FEATURE_COLUMNS,
      EXCESS_TARGET_COLUMN,
  ]
  assert excess[EXCESS_TARGET_COLUMN].name == EXCESS_TARGET_COLUMN
  assert TARGET_COLUMN not in excess.columns
  ```

  Add `test_unknown_target_is_rejected_before_preparation`, passing a valid-looking feature name and asserting `ValueError` mentions approved target names. Add a selected-null test where absolute is null on one row and excess is null on another; prove each call removes only the row whose selected target is null. Deep-compare input before/after.

- [ ] **Step 3: Write failing target-aware partition tests**

  Feed separately prepared absolute and excess frames through `temporal_split` and `walk_forward_splits`. For every partition assert exact X schema, selected y name/value, `metadata == ["date", "ticker"]`, aligned indices, unchanged fifth-observation `target_end_date` purging, and absence of both target names from X. Add direct malformed prepared frames with both approved targets and with neither; both splitters must raise rather than guess.

- [ ] **Step 4: Run dataset tests and observe RED**

  ```powershell
  python -m pytest tests/test_dataset.py tests/test_walk_forward.py -v
  ```

  Expected RED: the new keyword argument is unsupported and split/partition code is fixed to `TARGET_COLUMN`.

- [ ] **Step 5: Implement small target-dependent contract helpers**

  In `dataset.py` import both constants and add no public abstraction:

  ```python
  _TARGET_COLUMNS = (TARGET_COLUMN, EXCESS_TARGET_COLUMN)

  def _validate_target_column(target_column: str) -> None:
      if target_column not in _TARGET_COLUMNS:
          raise ValueError(
              "target_column must be one of: " + ", ".join(_TARGET_COLUMNS)
          )

  def _prepared_target_column(frame: pd.DataFrame) -> str:
      present = [name for name in _TARGET_COLUMNS if name in frame.columns]
      if len(present) != 1:
          raise ValueError("Prepared data must contain exactly one approved target column")
      return present[0]
  ```

  Build input/supervised tuples from the selected name inside `prepare_supervised_data`; validate the name before required-column checks. Continue copying, normalizing, duplicate-checking, sorting, ticker-local shifting, completeness filtering, and final sorting exactly as today.

  Change `_partition(rows, name, target_column)` to select that y. At the start of each splitter, infer the target, build its required schema, then pass the same target name to every partition. Do not alter either public split signature, `DatasetPartition`, metadata, or purge predicates.

- [ ] **Step 6: Run focused GREEN**

  ```powershell
  python -m pytest tests/test_dataset.py tests/test_walk_forward.py -v
  ```

  Expected: both target paths pass; default rows/order/values match the old behavior; ambiguity, unknown targets, alignment, immutability, null selection, and purge contracts are enforced.

### Task 3: Add ZeroBaseline and configurable model selection

**Files:**
- Modify: `tests/test_baseline.py`
- Modify: `tests/test_evaluation.py`
- Modify: `src/stock_forecaster/models/baseline.py`
- Modify: `src/stock_forecaster/models/evaluation.py`

**Interfaces:**
- Adds `ZeroBaseline.predict(X) -> pd.Series`.
- Adds `DEFAULT_MODEL_NAMES` and `evaluate_walk_forward(folds, model_names=DEFAULT_MODEL_NAMES)`.
- Aggregators accept either approved model subset and emit only present models in known presentation order.

- [ ] **Step 1: Write failing ZeroBaseline contract tests**

  Add one coherent test plus an immutability test:

  ```python
  X = pd.DataFrame({"anything": [4.0, -2.0]}, index=pd.Index([9, 3]))
  before = X.copy(deep=True)
  prediction = ZeroBaseline().predict(X)
  expected = pd.Series([0.0, 0.0], index=X.index,
                       name="prediction", dtype=float)
  pd.testing.assert_series_equal(prediction, expected)
  pd.testing.assert_frame_equal(X, before)
  ```

  This proves exact zeros, Series type/dtype, index, name, and no feature read/mutation.

- [ ] **Step 2: Write failing default and explicit evaluation tests**

  Rename the test-local current tuple to match `DEFAULT_MODEL_NAMES`. Assert omitted `model_names` yields the original four models in fold-major order. Call with:

  ```python
  STEP_5F_MODEL_NAMES = (
      "ZeroBaseline", "MeanBaseline", "LinearRegression", "RandomForest",
  )
  ```

  Assert exactly those results and caller order, and compare ZeroBaseline records to explicit zero predictions. Also request a deliberately noncanonical valid order and prove raw results preserve it within each fold.

- [ ] **Step 3: Write failing model-name validation tests**

  Parameterize empty tuple, unknown name, duplicate name, and a plain string. Each must raise `ValueError` before any model is constructed/fitted. Patch constructors with call counters to prove prevalidation. Extend the existing fresh-instance spy to selected models and two folds; assert a new Mean/Linear/Forest object per fold and no cross-fold state. Zero and Momentum remain independent calls.

- [ ] **Step 4: Write failing dynamic aggregation-order tests**

  Build shuffled `ModelFoldResult` inputs for both official sets. Assert fold metrics are year-major and known-model ordered, pooled metrics contain only present models, Step 5E stays Mean/Momentum/Linear/Forest, and Step 5F is Zero/Mean/Linear/Forest. This prevents hard-coded concatenation of a missing model.

- [ ] **Step 5: Run baseline/evaluation tests and observe RED**

  ```powershell
  python -m pytest tests/test_baseline.py tests/test_evaluation.py -v
  ```

  Expected RED: `ZeroBaseline` and `DEFAULT_MODEL_NAMES` do not exist; evaluation rejects the new parameter; aggregators assume the four old models.

- [ ] **Step 6: Implement ZeroBaseline**

  ```python
  class ZeroBaseline:
      """Predict zero without fitting or reading feature values."""

      def predict(self, X: pd.DataFrame) -> pd.Series:
          return pd.Series(0.0, index=X.index, name="prediction", dtype=float)
  ```

  Add no `fit`, configuration, or shared base class.

- [ ] **Step 7: Implement explicit model-set validation and per-fold resolution**

  In `evaluation.py` define:

  ```python
  DEFAULT_MODEL_NAMES = (
      "MeanBaseline", "MomentumBaseline", "LinearRegression", "RandomForest",
  )
  _KNOWN_MODEL_NAMES = (
      "ZeroBaseline", *DEFAULT_MODEL_NAMES,
  )
  ```

  `_validate_model_names` must reject `str`, materialize the sequence once as a tuple, require non-empty, reject unknowns and duplicates, and return caller order. Validate before entering the fold loop. For each fold/name, use a small explicit branch: Zero predicts validation X; Mean is freshly fitted on TRAIN y; Momentum predicts validation X; Linear/Forest are freshly fitted on TRAIN X/y. Pass every prediction through existing `_result` and `evaluate_predictions` without metric changes.

  Replace hard-coded aggregation iteration with `_KNOWN_MODEL_NAMES` filtered to names present in results. Keep raw caller order separate from deterministic table presentation order. Rename/remove the old public `MODEL_NAMES` rather than leaving two competing defaults; update imports/tests accordingly.

- [ ] **Step 8: Run focused GREEN**

  ```powershell
  python -m pytest tests/test_baseline.py tests/test_evaluation.py -v
  ```

  Expected: default and explicit sets, validation, caller order, fresh instances, deterministic tables, direct metrics, OOS uniqueness, and Random Forest reproducibility all pass.

### Task 4: Parameterize the consistency reference and migrate its schema

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `src/stock_forecaster/models/evaluation.py`

**Interfaces:**
- Produces `build_consistency_summary(fold_metrics: pd.DataFrame, reference_model: str = "MeanBaseline")`.
- Output columns are `model`, three `*_vs_reference` counts, and `folds_positive_correlation`.

- [ ] **Step 1: Rewrite consistency tests against annual metrics input**

  Use a small explicit fold-metrics DataFrame with two years, ties, `NaN`, zero/negative/positive correlations, and values where a cross-year join would give a different answer. Assert exact same-year strict counts for default MeanBaseline and explicit ZeroBaseline. Assert reference rows have zero comparison wins.

- [ ] **Step 2: Add schema and malformed-reference RED cases**

  Assert the exact new columns and absence of every `*_vs_mean` name. Parameterize:

  - empty frame;
  - missing required metric column;
  - duplicate `(validation_year, model)` row, including duplicated reference;
  - missing named reference entirely;
  - reference missing in one year represented by another model.

  Each raises clearly; no year may be silently dropped. Also assert model order follows first deterministic appearance in the supplied fold table.

- [ ] **Step 3: Add the Step 5E default-count regression test**

  Supply annual metrics that encode the validated Step 5E win pattern and call without `reference_model`. Assert new-schema counts in Step 5E order:

  ```text
  MeanBaseline      0 / 0 / 0 / 0
  MomentumBaseline  0 / 0 / 1 / 0
  LinearRegression  3 / 5 / 2 / 8
  RandomForest      1 / 1 / 2 / 8
  ```

  This is a semantic regression, not a substitute for the later real-data gate.

- [ ] **Step 4: Run the consistency tests and observe RED**

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```

  Expected RED: the old function accepts results, fixes MeanBaseline internally, and emits `*_vs_mean` columns.

- [ ] **Step 5: Implement validation, same-year join, and strict counts**

  Validate a non-empty DataFrame and required columns, then reject duplicate year/model keys. Require `reference_model` to exist exactly once in every represented year. For each model in `fold_metrics["model"].drop_duplicates()` order, merge its rows with reference rows on `validation_year` using `validate="many_to_one"`; assert the merge retains every model row. Count only `<`, `<`, `>`, and correlation `> 0` respectively. Pandas comparisons naturally make ties and `NaN` false.

  Return exactly:

  ```python
  _CONSISTENCY_COLUMNS = (
      "model",
      "folds_better_mae_vs_reference",
      "folds_better_rmse_vs_reference",
      "folds_better_direction_vs_reference",
      "folds_positive_correlation",
  )
  ```

  Do not retain a compatibility wrapper or duplicate legacy columns; update all call sites to pass `build_fold_metrics_table(results)`.

- [ ] **Step 6: Run focused GREEN**

  ```powershell
  python -m pytest tests/test_evaluation.py -v
  ```

  Expected: both references, exact schema, same-year behavior, strict comparisons, malformed-input rejection, positive/NaN correlation behavior, and Step 5E default counts pass.

### Task 5: Document, verify, review, and commit the implementation

**Files:**
- Modify: `README.md`
- Verify all files listed in Tasks 1-4
- Do not modify data in this task

- [ ] **Step 1: Update README before numerical evaluation**

  Explain both target formulas; exact stock-defined SPY endpoint matching; missing-endpoint null behavior; SPY exclusion from X/model universe; explicit target selection with absolute default; ZeroBaseline as the excess-target error reference; default Step 5E and explicit Step 5F model sets; unchanged 2016-2023 expanding walk-forward; and untouched 2024+ TEST. At this point state that controlled migration/evaluation remains pending—do not invent Step 5F numbers.

- [ ] **Step 2: Verify public imports/signatures**

  ```powershell
  python -c "import inspect; from stock_forecaster.features.engineering import EXCESS_TARGET_COLUMN; from stock_forecaster.models.baseline import ZeroBaseline; from stock_forecaster.models.dataset import prepare_supervised_data; from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES, evaluate_walk_forward, build_consistency_summary; assert EXCESS_TARGET_COLUMN == 'future_excess_return_5d'; assert inspect.signature(prepare_supervised_data).parameters['target_column'].default == 'future_return_5d'; assert tuple(DEFAULT_MODEL_NAMES) == ('MeanBaseline','MomentumBaseline','LinearRegression','RandomForest'); assert 'model_names' in inspect.signature(evaluate_walk_forward).parameters; assert inspect.signature(build_consistency_summary).parameters['reference_model'].default == 'MeanBaseline'; print('Step 5F imports/signatures: OK')"
  ```

- [ ] **Step 3: Run every focused and full verification command freshly**

  ```powershell
  python -m pytest tests/test_features.py -v
  python -m pytest tests/test_dataset.py tests/test_walk_forward.py -v
  python -m pytest tests/test_baseline.py -v
  python -m pytest tests/test_evaluation.py -v
  python -m pytest -v
  python -m ruff check .
  ```

  Expected: all focused files and the complete suite pass; Ruff succeeds. Migration is prohibited until this gate is green.

- [ ] **Step 4: Review protected scope and generated artifacts**

  ```powershell
  git diff --check
  git diff -- src/stock_forecaster/models/linear.py src/stock_forecaster/models/forest.py src/stock_forecaster/models/metrics.py pyproject.toml configs
  git status --short -- data/raw data/processed models
  git status --short
  ```

  Expected: only the four production modules, four expected test files (plus `test_walk_forward.py` only if compatibility coverage was necessary), and README changed. Protected-module/config/dependency/data commands print nothing. Review the complete diff for no predictor, tuning, TEST, backtest, or persistence scope.

- [ ] **Step 5: Create the single implementation commit**

  Stage only reviewed source/tests/README, list the index, then commit:

  ```powershell
  git add src/stock_forecaster/features/engineering.py src/stock_forecaster/models/dataset.py src/stock_forecaster/models/baseline.py src/stock_forecaster/models/evaluation.py tests/test_features.py tests/test_dataset.py tests/test_baseline.py tests/test_evaluation.py README.md
  git add tests/test_walk_forward.py  # only when actually changed for compatibility coverage
  git diff --cached --check
  git diff --cached --name-only
  git commit -m "feat: add SPY-relative excess return target"
  ```

  Expected: exactly the reviewed implementation files are committed. The plan is already in its documentation commit. Do not push and do not stage Parquet or generated output.

### Task 6: Perform one controlled migration and the mandatory Step 5E gate

**Files:**
- Read: `data/raw/prices.parquet`
- Replace after all pre-write gates: ignored `data/processed/features.parquet`
- Create/commit: no file

**Ordering invariant:** This task uses one Python process from loading the old processed frame through old/new Step 5E prediction comparison. That is how the validated old artifact remains available without inventing a historical prediction file. No Step 5F target is prepared and no Step 5F metric is computed in this task.

- [ ] **Step 1: Verify starting hashes and ignored status**

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  git check-ignore -v data/raw/prices.parquet data/processed/features.parquet
  ```

  Require exactly:

  ```text
  raw_before = 49616FD5E3BA2D40E715085284663B13DC0FA529B2652AD2B000BE99814287F5
  old_processed_before = 56001B3748F349E0DF5A29F30C1884B74810FDD2F87D28A5064E71E02675631A
  ```

  Stop before loading/regenerating if either anchor differs or either artifact is not ignored.

- [ ] **Step 2: In one in-memory orchestration, capture old Step 5E references**

  Start a PowerShell here-string piped to `python -`. Import `hashlib`, `numpy`, `pandas`, the two target constants, `build_features`, dataset/evaluation APIs, and define the 15-stock set, years `tuple(range(2016, 2024))`, and `DEVELOPMENT_END = pd.Timestamp("2024-01-01")`.

  Define `step_5e_outputs(frame)` to exclude SPY, call `prepare_supervised_data(frame)` with no target argument, restrict both date columns below 2024, construct eight folds, evaluate with no model argument, then return supervised development rows, folds, results, fold metrics, pooled metrics, and `build_consistency_summary(fold_metrics)` with no reference argument. Call it first on the retained `old_processed` DataFrame. Do not print or compute any excess-target metrics.

- [ ] **Step 3: Build candidate and enforce old-column equality before write**

  Still in the same process:

  ```python
  raw = pd.read_parquet(RAW_PATH)
  old_processed = pd.read_parquet(PROCESSED_PATH)
  candidate = build_features(raw)
  old_columns = ["date", "ticker", *FEATURE_COLUMNS, TARGET_COLUMN]
  old_aligned = old_processed.loc[:, old_columns].sort_values(
      ["date", "ticker"]
  ).reset_index(drop=True)
  candidate_aligned = candidate.loc[:, old_columns].sort_values(
      ["date", "ticker"]
  ).reset_index(drop=True)
  assert len(candidate_aligned) == len(old_aligned)
  assert not old_aligned.duplicated(["date", "ticker"]).any()
  assert not candidate_aligned.duplicated(["date", "ticker"]).any()
  pd.testing.assert_frame_equal(
      candidate_aligned,
      old_aligned,
      check_exact=False,
      rtol=1e-12,
      atol=1e-15,
  )
  ```

  Before the frame assertion, explicitly compare ordered `(date, ticker)` keys; on failure calculate an outer merge indicator and print only differing/missing keys. On numeric failure, catch `AssertionError`, inspect each old numeric column with matching-NaN and `np.isclose(..., rtol=1e-12, atol=1e-15, equal_nan=True)`, print column/key/old/candidate for mismatches, then raise. STOP without writing on any mismatch.

- [ ] **Step 4: Audit the new target before write**

  Require exact processed column order, numeric excess dtype, and finite non-null values. Recompute `target_end_date` from raw independently of the candidate; build exact SPY `date -> adjusted_close`; recompute stock/benchmark returns directly. Across all rows assert candidate excess is null exactly where the absolute horizon or either SPY endpoint is unavailable, otherwise compare with `np.isclose(rtol=1e-12, atol=1e-15)`.

  Additionally print/check deterministic representative complete rows from at least three non-SPY tickers across 2016, 2020, and 2023, complete SPY rows near the beginning/middle/end, and every real missing-endpoint case. Require complete SPY values approximately zero and final five incomplete SPY horizons null. Synthetic tests remain the proof when the real artifact contains no missing exact endpoint case; report that count as zero rather than fabricating a case.

- [ ] **Step 5: Replace, reload, and verify hash transition**

  Only after Steps 3-4 pass:

  ```python
  candidate.to_parquet(PROCESSED_PATH, index=False)
  migrated = pd.read_parquet(PROCESSED_PATH)
  pd.testing.assert_frame_equal(migrated, candidate)
  ```

  Repeat schema, dtype, finite, SPY-zero, tail-null, and old-column equality checks on `migrated`. Recalculate SHA-256 in Python. Require raw hash still equals `raw_before`, new processed hash differs from `old_processed_before`, and record/print `new_processed_sha` outside the repository.

- [ ] **Step 6: Run the Step 5E regression before exposing any Step 5F result**

  In the same process, call `step_5e_outputs(migrated)`. Compare old and new default supervised development frames exactly; compare every fold's year, TRAIN/VALIDATION X, y, and metadata exactly. Require raw result order/model names unchanged. Compare every old/new `ModelFoldResult.predictions` frame, including all date/ticker keys, row counts, fold membership, y, and deterministic model predictions, with `rtol=1e-12`, `atol=1e-15`, and matching NaNs. This direct in-memory comparison satisfies full prediction equality without inventing a persisted historical prediction artifact.

  Compare the complete 32-row annual tables and four-row pooled tables by exact keys/order/schema, identical NaN positions, and the same strict numeric tolerance. Compare the four consistency rows/counts exactly after acknowledging the intentional new `*_vs_reference` schema. Require default models in original order and exactly eight validation years.

  As a secondary human-audit anchor (not a replacement for full-precision dynamic equality), require the old and migrated pooled results rounded to six decimals to equal:

  | model | MAE | RMSE | directional_accuracy | correlation |
  |---|---:|---:|---:|---:|
  | MeanBaseline | 0.027095 | 0.039366 | 0.569135 | -0.066013 |
  | MomentumBaseline | 0.039540 | 0.056663 | 0.507606 | -0.032182 |
  | LinearRegression | 0.027104 | 0.039375 | 0.566295 | 0.035659 |
  | RandomForest | 0.027247 | 0.039623 | 0.564638 | 0.023736 |

  Require exact consistency count tuples `[(0,0,0,0), (0,0,1,0), (3,5,2,8), (1,1,2,8)]` in Step 5E model order. The repository contains no separate full-precision historical prediction/report artifact, so the retained old frame is the authoritative full-precision source; the command must not compare all 32 annual rows to rounded/transcribed values.

- [ ] **Step 7: Enforce failure behavior and finish migration audit**

  If any old-column, hash, fold, prediction, annual, pooled, or consistency assertion fails, terminate nonzero, report the first exact mismatch, do not run Step 5F, and do not inspect excess-target metrics. If all pass, print only the migration hashes, target audit, and `STEP 5E REGRESSION: PASS`. Recheck in PowerShell:

  ```powershell
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  git status --short
  ```

  Expected: raw unchanged, processed equals the newly recorded hash, and Git remains clean because data is ignored.

### Task 7: Run the one official Step 5F development evaluation

**Files:**
- Read only: migrated `data/processed/features.parquet`
- Read only/hash only: `data/raw/prices.parquet`
- Persist: nothing

- [ ] **Step 1: Record immutable pre-evaluation hashes**

  Record `raw_before` and `new_processed_sha_before_evaluation`; require they match Task 6's values.

- [ ] **Step 2: Prepare only approved development data**

  In one console-only Python invocation, load migrated features, verify SPY exists for audit, then exclude SPY and require exactly the 15 approved stocks. Call:

  ```python
  supervised = prepare_supervised_data(
      features_without_spy,
      target_column=EXCESS_TARGET_COLUMN,
  )
  development = supervised.loc[
      supervised["date"].lt("2024-01-01")
      & supervised["target_end_date"].lt("2024-01-01")
  ].copy()
  folds = walk_forward_splits(development, tuple(range(2016, 2024)))
  ```

  Audit every TRAIN/VALIDATION date and exact target-end boundary as in Step 5E. Assert no SPY, no 2024+ date/end, exact X schema, excess y name, and eight folds.

- [ ] **Step 3: Evaluate the official set once with no tuning**

  ```python
  STEP_5F_MODELS = (
      "ZeroBaseline", "MeanBaseline", "LinearRegression", "RandomForest",
  )
  results = evaluate_walk_forward(folds, model_names=STEP_5F_MODELS)
  annual = build_fold_metrics_table(results)
  pooled = build_pooled_metrics(results)
  consistency = build_consistency_summary(
      annual, reference_model="ZeroBaseline"
  )
  ```

  Require 32 results/annual rows, four pooled rows, four consistency rows, exact model order/schema, and unique OOS `(date, ticker, model)` keys. Do not rerun with alternative settings after seeing results.

- [ ] **Step 4: Print exactly the three official result families**

  Print full precision sufficient for audit:

  1. pooled OOS `model, MAE, RMSE, directional_accuracy, correlation`;
  2. all 32 annual rows with `validation_year` plus those columns;
  3. consistency versus ZeroBaseline with the three `*_vs_reference` columns and `folds_positive_correlation`.

  Explicitly state that ZeroBaseline correlation is expected `NaN` and its directional accuracy uses zero as its own sign; treat it primarily as MAE/RMSE reference. Do not change metric behavior.

- [ ] **Step 5: Interpret Step 5E versus Step 5F within approved limits**

  Compare pooled correlation, annual positive-correlation consistency, directional behavior, cross-year stability, and each target's model improvement over its natural baseline (Mean for absolute; Zero for excess). Do not compare raw cross-target MAE/RMSE as a winner, construct a combined score, tune, claim profitability/significance, or mention TEST performance.

- [ ] **Step 6: Prove evaluation was read-only**

  Recalculate both hashes. Require raw equals the original anchor and processed equals `new_processed_sha_before_evaluation`. Run `git status --short`; persist no output.

### Task 8: Complete whole-branch review and final verification

**Files:**
- Review the committed implementation and README
- Modify only a proven implementation defect, test-first; never tune based on Step 5F results

- [ ] **Step 1: Review every architectural invariant**

  Check the diff from the feature branch base and confirm: old features/target unchanged; stock-defined exact SPY endpoints; no fills/substitutions; SPY zero behavior; no SPY value or target in X; absolute/default preparation preserved; Step 5E default models/order preserved; Step 5F explicit models; Mean default and Zero explicit consistency references; no legacy schema columns; no existing ML/metric changes; no tuning; no TEST; controlled migration/gates happened in order; no output artifact committed.

- [ ] **Step 2: If review finds a defect, repair it RED-GREEN**

  Add the smallest failing synthetic test, run the focused file to observe the intended failure, implement the minimum fix, rerun focused/full checks, and amend only under the later approved execution workflow so the implementation remains one commit. Do not interpret a disappointing metric as a defect and do not tune.

- [ ] **Step 3: Run fresh final verification**

  ```powershell
  python -m pytest -v
  python -m ruff check .
  git status --short
  git diff --exit-code
  git diff --cached --exit-code
  Get-FileHash -Algorithm SHA256 data/raw/prices.parquet
  Get-FileHash -Algorithm SHA256 data/processed/features.parquet
  ```

  Expected: suite and Ruff green; tracked tree/index clean; raw still equals its anchor; processed equals the recorded migrated hash. The ignored processed file may differ from another checkout's pre-Step-5F copy and must never be staged.

- [ ] **Step 4: Prepare the execution report and stop**

  Report implementation commit, TDD commands/results, migration old/new hashes, old-column audit, target audit, complete Step 5E regression gate, official Step 5F tables, bounded comparison, post-evaluation hashes, final verification, clean Git state, and confirmation nothing was pushed. Do not start Step 5G, TEST evaluation, portfolio optimization, or backtesting.

## Test Coverage Map

- `tests/test_features.py`: old target and all ten features unchanged; excess column; stock fifth-future date; exact stock and SPY returns; subtraction; complete SPY zero; missing exact start/end; no forward/backward fill; no independent SPY horizon; final five null; schema/order; immutability; finite non-null values; existing leakage, cross-ticker, warm-up, and zero-volume protection.
- `tests/test_dataset.py` (and compatibility-only `test_walk_forward.py` if needed): default/explicit absolute and explicit excess; unknown target; selected-target-only frame; exact X/y/metadata; y name; target end; alignment; purge; immutability; selected null behavior; no excess in X; ambiguous/no prepared target rejection.
- `tests/test_baseline.py`: ZeroBaseline exact zero float Series, name/index preservation, and immutability.
- `tests/test_evaluation.py`: Step 5E default and Step 5F explicit sets; Zero results; caller order; empty/string/unknown/duplicate rejection before fitting; fresh per-fold instances; deterministic existing behavior; dynamic fold/pooled ordering; Mean default and Zero reference; exclusive `*_vs_reference` schema; same-year strict wins/ties; NaN/positive correlation; missing/duplicate/incomplete reference rejection; unchanged Step 5E counts.
- Existing tests remain regression coverage for linear/forest contracts, metrics including zero sign and constant correlation, target-end purging, future isolation, and reproducibility.

## Controlled Execution and Failure Boundaries

1. No migration before focused tests, full suite, Ruff, review, and implementation commit pass.
2. No processed write before old-column and new-target pre-write audits pass.
3. No Step 5F metric may be calculated or observed before the full Step 5E default prediction/metric regression passes.
4. No result-driven rerun, tuning, TEST, persisted output, or second implementation commit is allowed.
5. Any hash/equality/regression failure stops the phase with the worktree/branch and evidence intact for diagnosis.

## Plan Self-Review

The approved specification maps completely to Tasks 1-8. Exact stock-defined benchmark windows, adjusted-close convention, missing endpoint semantics, no imputation/substitute date, SPY zero behavior, unchanged old target/features, and final incomplete horizons all have explicit RED-GREEN coverage. The dataset accepts only two targets, returns exactly one, preserves default behavior, X, metadata, alignment, and purge semantics, and rejects ambiguity.

ZeroBaseline, configurable validated model selection, fresh fold instances, default Step 5E order, explicit Step 5F order, dynamic deterministic aggregations, configurable same-year consistency, strict comparison rules, malformed reference rejection, and exclusive schema rename are all specified with concrete tests and implementation seams matching current source. No registry or unnecessary abstraction is introduced.

Migration is gated before write by anchored hashes, exact keys/row counts/NaN patterns, and strict equality of every old column. The new target has numeric/finite, representative exact-window, missing-endpoint, SPY-zero, and tail audits. Raw immutability and the processed hash transition are recorded. The retained old frame supplies a full-precision deterministic Step 5E comparison of supervised rows, folds, all model predictions, all 32 annual rows, pooled rows, and consistency counts before Step 5F is observed; published rounded pooled values are only a secondary audit.

Step 5F uses only pre-2024 dates and target ends, exactly 2016-2023 folds, and exactly ZeroBaseline, MeanBaseline, LinearRegression, and RandomForest. Its three official tables and ZeroBaseline metric nuance are explicit. Contextual comparison avoids cross-target error ranking and unsupported claims. Evaluation is hash-proven read-only. The plan adds no dependency, predictor, tuning, TEST, portfolio work, backtest, incomplete marker, or unresolved decision; all names and signatures match the approved design and current repository.
