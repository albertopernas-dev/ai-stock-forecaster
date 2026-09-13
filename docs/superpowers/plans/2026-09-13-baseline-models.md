# Baseline Models and Regression Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a global training-mean baseline, a stateless five-day
momentum baseline, and common regression metrics using validation data only.

**Architecture:** Keep metric calculation in one pure function that validates
two aligned Series and returns an immutable value object. Keep baselines as
small independent classes: one learns only the training target mean, while the
other copies the existing `return_5d` feature from validation `X`.

**Tech Stack:** Python 3.12, pandas, standard-library `dataclasses` and `math`,
pytest, and Ruff; no new dependencies.

**Spec:** User-approved Step 5B specification attached to this task.

## Global Constraints

- Do not use TEST predictions, targets, or metrics for implementation or
  evaluation.
- Do not implement linear regression, estimator frameworks, tuning,
  backtesting, profitability analysis, or portfolio logic.
- Do not mutate inputs, silently drop invalid metric observations, recompute
  momentum from prices, or calculate per-ticker fitted baselines.
- Do not modify raw or processed Parquet files, add dependencies, or push.

---

### Task 1: Specify common regression metrics with synthetic tests

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/stock_forecaster/models/metrics.py`

**Interfaces:**
- Consumes: aligned, finite, non-empty `pd.Series` objects named `y_true` and
  `y_pred`.
- Produces: frozen `RegressionMetrics(mae, rmse, directional_accuracy,
  correlation)` and `evaluate_predictions(y_true, y_pred)`.

- [ ] **Step 1: Write literal metric and validation tests**

  Use hand-calculated examples for MAE, RMSE, sign agreement including exact
  zero, perfect predictions, perfect non-constant correlation, and undefined
  constant-series correlation. Assert clear rejection of empty, unequal,
  null, infinite, or misaligned inputs and unchanged input Series.

- [ ] **Step 2: Observe metric RED**

  Run `python -m pytest tests/test_metrics.py -v`. Expected: collection fails
  because `stock_forecaster.models.metrics` does not exist.

- [ ] **Step 3: Implement the minimal metric module**

  Validate length, emptiness, index alignment, nulls, and infinities without
  sorting or dropping rows. Calculate decimal-return MAE, RMSE, exact sign
  agreement, and pandas Pearson correlation; explicitly return `float("nan")`
  when either input has zero variance.

- [ ] **Step 4: Verify metric GREEN**

  Run `python -m pytest tests/test_metrics.py -v` and keep the implementation
  focused on the approved metric contract.

### Task 2: Specify and implement the two baselines

**Files:**
- Create: `tests/test_baseline.py`
- Create: `src/stock_forecaster/models/baseline.py`

**Interfaces:**
- Consumes: training `y` for `MeanBaseline.fit`, a requested prediction index
  for `MeanBaseline.predict`, and validation `X` for
  `MomentumBaseline.predict`.
- Produces: index-aligned prediction Series and a clearly named `mean_` fitted
  attribute.

- [ ] **Step 1: Write literal baseline behavior tests**

  Assert the global mean is learned from the supplied target Series, constant
  predictions equal that mean and preserve the requested index, predict before
  fit fails, and empty or null training targets fail. Assert momentum output is
  an unchanged copy of `X["return_5d"]`, preserves ordering/index, does not
  mutate `X`, and clearly rejects a missing feature.

- [ ] **Step 2: Observe baseline RED**

  Run `python -m pytest tests/test_baseline.py -v`. Expected: collection fails
  because `stock_forecaster.models.baseline` does not exist.

- [ ] **Step 3: Implement minimal baseline classes**

  `MeanBaseline.fit(y_train)` stores `float(y_train.mean())` in `mean_` and
  returns `self`; `predict(index)` requires fitted state and returns a constant
  Series aligned to that index. `MomentumBaseline.predict(X)` returns a copied
  Series from the existing `return_5d` column and never accepts or inspects a
  target.

- [ ] **Step 4: Verify baseline GREEN**

  Run `python -m pytest tests/test_baseline.py -v`, then run both new test files
  together.

### Task 3: Document, verify, commit, and evaluate on validation only

**Files:**
- Modify: `README.md`
- Verify: `src/stock_forecaster/models/metrics.py`,
  `src/stock_forecaster/models/baseline.py`, `tests/test_metrics.py`, and
  `tests/test_baseline.py`

**Interfaces:**
- Consumes: the existing supervised split API and ignored processed features.
- Produces: concise documentation, one implementation commit, and read-only
  TRAIN/VALIDATION benchmark evidence.

- [ ] **Step 1: Update README**

  Document supervised split → baselines → common metrics → future models; list
  both baseline definitions and all four metrics; explicitly state TEST is not
  used for baseline selection.

- [ ] **Step 2: Run implementation verification**

  Run both focused test files, full pytest, Ruff, and imports for all four new
  public names. Stop before real-data evaluation if any command fails.

- [ ] **Step 3: Commit only approved tracked files**

  Confirm no Parquet is staged and commit as
  `feat: add forecasting baselines and evaluation metrics`. Do not push.

- [ ] **Step 4: Evaluate TRAIN and VALIDATION read-only**

  Recreate the approved split, fit the mean baseline only on `split.train.y`,
  predict both baselines on validation rows, and report common metrics,
  prediction distributions, and per-ticker diagnostics. Do not access
  `split.test` after split construction.

- [ ] **Step 5: Verify repository and dataset safety**

  Run Git status and unstaged/staged diff checks, compare both Parquet hashes,
  and show five commits. Confirm a clean repository and unchanged datasets.
