# Pre-Registered Hold-Out Confirmation Design

## Status and objective

The Step 5I functional design is approved. This document specifies future
implementation and **the single confirmatory evaluation of the 2024+ hold-out**.
This specification task changes no production code, tests, README content,
dependencies, or datasets and reads no hold-out outcome.

Steps 5A through 5H built and characterized the system entirely on development
data through 2023. The hold-out from 2024-01-01 onward has never been read: not
for fitting, not for evaluation, not for diagnosis, not for model choice. Step
5I spends it, once.

**This document is a pre-registration.** Its purpose is to fix the question, the
configuration, the metric, and the comparison *before* any hold-out number
exists, so that the result cannot be reinterpreted after the fact. It is
committed to version control before the evaluation runs, and the commit
timestamp is the evidence that it was written first.

## Hold-out extent

Calendar inspection only, performed during specification. No target, feature, or
return value was read.

```text
period          2024-01-02 -> 2026-09-11
trading dates   676
modeled stocks  15
rows            10140
dates per year  2024: 252   2025: 250   2026: 174 (partial)
```

2026 is a partial year ending 2026-09-11. Its final five observations per ticker
have no five-session target and are dropped by the existing completeness rules,
exactly as every other period's tail is.

## Pre-registered primary configuration

One target, one strategy, two candidate models, one control, one metric. The
configuration is chosen from development evidence, which is the legitimate
purpose of development data. Choosing it after seeing hold-out results would not
be legitimate, which is why it is fixed here.

**Target: `future_return_5d` (absolute).** Step 5G proved the excess target is
rank-identical to the absolute target within a date, so it can only differ
through model fitting, and Step 5H showed it fitted worse. The absolute target
is therefore the pre-registered choice.

**Strategy: `long_only`.** Step 5H showed it dominated `long_short` on break-even
in every cell, because the short leg added cost faster than return. It is also
the realistic case for an investor who does not short.

**Models: `LinearRegression` and `RandomForest`, with `MomentumBaseline` as the
naive control.** Both learned models are carried rather than picking the
development winner, because development evidence did not separate them cleanly:
LinearRegression had the higher pooled break-even (12.39 bps) while RandomForest
was far more stable across years (positive in 7 of 8 versus 6 of 8, with a worst
year of −2.9 bps versus −66.2 bps). Carrying both costs one extra comparison and
avoids pretending to a confidence the development data does not support.

**Primary metric: pooled break-even cost in basis points**, over the whole
hold-out period, computed exactly as in Step 5H.

## Pre-registered comparison

There is no arbitrary pass threshold. The comparison is against the already
published development figures, which are fixed and cannot be revised:

```text
development pooled break-even, absolute target, long_only strategy
    LinearRegression   +12.39 bps
    RandomForest       +11.62 bps
    MomentumBaseline    −1.59 bps  (control)
```

The reading is committed in advance:

- A hold-out break-even of **similar sign and magnitude** to development, with
  the learned models above the control, means the development finding survived
  one honest out-of-sample test.
- A hold-out break-even **near zero or negative**, or learned models at or below
  the control, means it did not survive.
- Anything between is **inconclusive and will be reported as inconclusive**,
  not resolved by choosing a favorable secondary statistic.

The annual breakdown is reported alongside and answers stability, not
confirmation. The 2026 partial year is flagged wherever it appears.

**Even the most favorable outcome is not a claim of profitability or
investability.** It would mean one pre-registered prediction was not falsified,
on one universe, over one 2.7-year window, before unmodeled costs.

## Secondary reported results

Reported for completeness, explicitly labelled secondary, and never substituted
for the primary metric: the `long_short` strategy on the same target; the
information coefficient and tercile spread from Step 5G's machinery; and the
Step 5E point-error metrics. These are descriptive context. No secondary result
can convert an inconclusive or negative primary result into a positive one.

## Deliberate widening of the hold-out guard

`ranking.py` and `backtesting/simulation.py` both raise on any row dated
2024-01-01 or later. That guard is intentional and stays on by default forever.

Step 5I adds an explicit opt-in rather than removing it:

```python
build_daily_ranking(predictions, allow_holdout: bool = False)
select_non_overlapping(daily, step=5)              # unchanged
build_period_returns(predictions, strategy=..., step=..., allow_holdout: bool = False)
summarize_ranking(daily, by_year=False)            # unchanged
summarize_backtest(periods, by_year=False)         # unchanged
```

With `allow_holdout=False`, behavior is byte-for-byte what it is today and every
existing test continues to pass unchanged. Only an explicit `allow_holdout=True`
at the call site admits 2024+ rows. Spending the hold-out therefore remains a
visible, deliberate, greppable act in the calling code rather than a silent
default. The same opt-in is added to nothing else.

`walk_forward_splits` needs no change: it already accepts arbitrary validation
years.

## Evaluation procedure

One run, in one process:

1. Hash `data/raw/prices.parquet` and `data/processed/features.parquet` and
   confirm both anchors.
2. Prepare supervised rows with the default absolute target, excluding SPY.
3. Apply **no** 2024 cutoff. Build expanding walk-forward folds for validation
   years 2016 through 2026. Folds 2016-2023 reproduce development exactly; folds
   2024, 2025 and 2026 are the hold-out. Each hold-out fold trains only on
   strictly prior data, exactly as every development fold did.
4. Evaluate `MomentumBaseline`, `LinearRegression` and `RandomForest`.
5. Confirm the 2016-2023 predictions are **numerically identical** to the
   development run at `rtol=1e-12, atol=1e-15`. If they are not, stop and report
   before reading any hold-out number: an unexplained change invalidates the
   comparison.
6. Split results into development years and hold-out years.
7. Produce the primary table, then the secondary tables.
8. Re-hash both data files and confirm they are unchanged.

Model configurations are unchanged. No hyperparameter, feature, target,
strategy, bucket size or rebalancing schedule is altered.

## The one-shot rule

This evaluation runs **exactly once**. After it runs:

- no model may be refitted, retuned, reconfigured or reselected;
- no strategy, bucket size, schedule or threshold may be adjusted;
- no second hold-out evaluation may be run under any justification;
- a disappointing result is the answer, not a defect.

Any future modeling work must treat 2024-2026 as contaminated and would require
genuinely new data for a further honest test. This is stated so that the
temptation, when it arrives, meets a written commitment.

## Interpretation and permitted conclusions

Permitted: whether the pre-registered prediction survived; the hold-out
break-even beside its development counterpart; whether the learned models beat
the naive control out of sample; stability across 2024, 2025 and the partial
2026; and what the result implies for whether this line of work deserves more
effort.

Forbidden, without exception: profitability, investability, or expected return;
statistical significance, p-values, or hypothesis tests; causality or economic
significance; Sharpe or other risk-adjusted statistics; compounding or
annualization; any composite score; and any claim derived from a secondary
result that the primary metric does not support.

Four limitations are restated wherever results appear:

1. **Unmodeled costs.** Short borrow, market impact and taxes are excluded, so
   break-even is optimistic.
2. **Universe breadth.** Fifteen large, correlated stocks cap capacity.
3. **One window.** 2024-2026 is a single regime, and a favorable outcome is not
   a general result.
4. **Survivorship.** The universe was chosen in 2026 from companies that are
   large today, which biases a historical study of that universe upward. This
   affects the development period far more than the hold-out but is real
   throughout.

## Execution contract

The established protocol: specification committed **before** implementation;
isolated worktree at `.worktrees/holdout-confirmation` on branch
`codex/holdout-confirmation`; RED-GREEN TDD for the `allow_holdout` opt-in with
tests proving the default still rejects 2024+; full suite and Ruff green; one
single implementation commit; then the one evaluation; then a report; then stop.
Nothing merged or pushed without explicit authorization.

## Explicit exclusions

Step 5I excludes new predictors, new models, new targets, new strategies,
tuning, model selection after the fact, portfolio optimization, a second
hold-out run, live trading, deployment, and any persisted dataset, prediction or
model artifact.

## Design self-review

1. The configuration, metric and comparison are fixed before any hold-out value
   is read, and the commit timestamp evidences it.
2. The configuration was chosen from development data, which is its legitimate
   use.
3. Both learned models are carried, avoiding a false claim of separation.
4. A naive control is included so the result is interpretable.
5. The comparison is against published development figures, not an arbitrary
   threshold.
6. An inconclusive outcome is named in advance as a permitted verdict.
7. Secondary results cannot override the primary one, and this is stated.
8. The hold-out guard is widened by explicit opt-in, never removed, and stays
   off by default.
9. The 2016-2023 reproduction check must pass before any hold-out number is
   read.
10. The one-shot rule is written down before the temptation to break it exists.
11. Survivorship bias is named, having gone unstated in earlier steps.
12. The partial 2026 year is flagged rather than silently averaged.
13. No new dependency and no persisted artifact.
14. Every architectural decision is resolved; no placeholder remains.
