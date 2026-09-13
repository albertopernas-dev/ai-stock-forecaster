# Supervised Dataset Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, leakage-safe supervised dataset preparation and
chronological train/validation/test splitting layer without training a model.

**Architecture:** Validate and copy processed features, derive each ticker's
five-observation `target_end_date`, and retain only complete supervised rows.
Split those rows chronologically while purging train and validation targets
whose horizons cross the next period, then expose aligned `X`, `y`, and
metadata containers.

**Tech Stack:** Python 3.12, pandas, pytest, and Ruff; no new dependencies.

**Spec:** User-approved Step 5A specification attached to this task.

## Global Constraints

- Import `FEATURE_COLUMNS` and `TARGET_COLUMN` from
  `stock_forecaster.features.engineering`; never duplicate the feature list.
- Do not mutate inputs, fill missing values, silently deduplicate, shuffle, or
  expose identifiers, target dates, or targets as model features.
- Derive target horizons using five ticker-local trading observations, never
  calendar-day arithmetic or a global shift.
- Do not modify raw or processed Parquet files, add dependencies, train models,
  or push commits.

---

### Task 1: Specify preparation and split behavior with synthetic tests

**Files:**
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `TARGET_COLUMN`, and the planned public dataset
  API.
- Produces: deterministic contract tests for validation, horizon calculation,
  purging, partition structure, ordering, alignment, and empty-partition errors.

- [ ] **Step 1: Build deterministic fixtures**

  Construct complete feature frames with literal values for one and two
  tickers, including different trading calendars around 2021/2022 and
  2023/2024 boundaries. Make feature and target values uniquely identify their
  source row so alignment and ordering assertions do not reuse implementation
  logic.

- [ ] **Step 2: Write validation and preparation tests**

  Assert clear failures for missing required columns and duplicate
  `date+ticker` rows; assert the caller frame is unchanged; assert feature,
  target, and unavailable-horizon rows are excluded; and assert output columns
  are exactly `date`, `ticker`, `target_end_date`, `*FEATURE_COLUMNS`, and
  `TARGET_COLUMN`, sorted by date then ticker.

- [ ] **Step 3: Write horizon-isolation tests**

  Use literal expected dates to prove `target_end_date` is the fifth later
  observation within each ticker, including weekends, missing trading dates,
  and ticker boundaries.

- [ ] **Step 4: Write split and purge tests**

  Assert boundary-order validation, exact train/validation/test candidate date
  rules, train and validation horizon purges, retention of a late-December row
  whose fifth later observation remains in December, removal of a late-December
  row whose fifth later observation is in January, ticker-local purge behavior,
  and clear errors for each empty partition.

- [ ] **Step 5: Write partition contract tests**

  Assert `X` has `FEATURE_COLUMNS` in exact order and no identifiers or target;
  `y` is a named `TARGET_COLUMN` Series; metadata is exactly `date,ticker`; all
  three have reset, aligned indices, no nulls, and chronological order without
  shuffling.

- [ ] **Step 6: Observe the red phase**

  Run `python -m pytest tests/test_dataset.py -v`. Expected result: collection
  fails because `stock_forecaster.models.dataset` does not exist.

### Task 2: Implement the focused dataset module

**Files:**
- Create: `src/stock_forecaster/models/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: a processed `pd.DataFrame`, imported feature/target constants, and
  string, `date`, or `datetime` split boundaries.
- Produces: `DatasetPartition`, `TemporalDatasetSplit`,
  `prepare_supervised_data(features)`, and `temporal_split(...)`.

- [ ] **Step 1: Implement preparation minimally**

  Validate required columns and duplicate keys, copy only required columns,
  sort by ticker/date, derive `target_end_date` with ticker-local `shift(-5)`,
  retain only rows complete across features, target, and horizon date, then sort
  date/ticker and reset the index.

- [ ] **Step 2: Implement immutable containers and partition construction**

  Add frozen dataclasses containing only the specified frames/Series. Build
  each partition with copied and reset `X`, `y`, and metadata objects in their
  exact required column order.

- [ ] **Step 3: Implement temporal splitting**

  Convert boundaries with `pd.Timestamp`; require
  `train_end < validation_start <= validation_end < test_start`; select train,
  validation, and test candidates; purge train at `target_end_date <
  validation_start` and validation at `target_end_date < test_start`; reject an
  empty named partition; never purge test beyond preparation's natural target
  availability.

- [ ] **Step 4: Reach and preserve green**

  Run `python -m pytest tests/test_dataset.py -v`, correct only behavior covered
  by the approved contract, and rerun until all dataset tests pass.

### Task 3: Document and verify the completed layer

**Files:**
- Modify: `README.md`
- Verify: `src/stock_forecaster/models/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: the completed public API and the existing ignored processed file.
- Produces: user documentation, verification evidence, one implementation
  commit, and a read-only real-data audit.

- [ ] **Step 1: Update README**

  Add the processed-features → supervised-rows → leakage-safe temporal split →
  train/validation/test flow, initial period boundaries, and the plain-language
  five-session boundary-purge rule. Do not document model behavior.

- [ ] **Step 2: Run implementation verification**

  Run the focused dataset tests, full pytest suite, Ruff, and an import command
  for all four public names. Stop before the real-data audit if any command
  fails.

- [ ] **Step 3: Commit tracked implementation files**

  Confirm Parquet files are absent from staged changes and create exactly one
  commit named `feat: add leakage-safe temporal dataset split`. Do not push.

- [ ] **Step 4: Audit the real processed dataset read-only**

  Prepare and split `data/processed/features.parquet` at the approved dates.
  Report overall and per-ticker counts, ranges, purge counts, maximum retained
  target-end dates, exact schemas, null counts, row alignment, duplicate keys,
  and non-overlapping periods.

- [ ] **Step 5: Verify repository and artifact safety**

  Run `git status --short`, `git diff --exit-code`, and
  `git diff --cached --exit-code`; compare before/after Parquet hashes; and show
  `git log --oneline -5`. The repository must be clean and datasets unchanged.
