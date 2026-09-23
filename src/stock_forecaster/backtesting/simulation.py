"""Break-even cost backtest over out-of-sample return forecasts."""

import pandas as pd

from stock_forecaster.models.evaluation import PREDICTION_COLUMNS
from stock_forecaster.models.ranking import (
    BUCKET_SIZE,
    KNOWN_MODEL_ORDER,
    TEST_BOUNDARY,
    select_non_overlapping,
)

LONG_SHORT = "long_short"
LONG_ONLY = "long_only"
STRATEGIES = (LONG_SHORT, LONG_ONLY)
REBALANCE_STEP = 5

PERIOD_COLUMNS = (
    "model",
    "validation_year",
    "date",
    "n_tickers",
    "gross_return",
    "traded_notional",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _model_sort_key(model_name: str) -> int:
    if model_name in KNOWN_MODEL_ORDER:
        return KNOWN_MODEL_ORDER.index(model_name)
    return len(KNOWN_MODEL_ORDER)


def _validate(predictions: pd.DataFrame, strategy: str, step: int) -> pd.DataFrame:
    if strategy not in STRATEGIES:
        allowed = ", ".join(STRATEGIES)
        raise ValueError(
            f"Unsupported strategy {strategy!r}; expected one of: {allowed}"
        )
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty DataFrame")
    _require_columns(predictions, PREDICTION_COLUMNS)
    ordered = predictions.loc[:, PREDICTION_COLUMNS].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    if ordered.duplicated(subset=["date", "ticker", "model"]).any():
        raise ValueError("Duplicate date, ticker and model rows")
    if ordered["date"].ge(TEST_BOUNDARY).any():
        raise ValueError(
            "predictions must not contain rows dated 2024-01-01 or later; "
            "TEST is never evaluated"
        )
    return ordered


def _book(
    group: pd.DataFrame,
    strategy: str,
) -> tuple[dict[str, float] | None, float]:
    if len(group) < 2 * BUCKET_SIZE or group["prediction"].nunique(dropna=True) <= 1:
        return None, float("nan")
    ranked = group.sort_values(by=["prediction", "ticker"], ascending=[False, True])
    top = ranked.iloc[:BUCKET_SIZE]
    weights = {ticker: 1.0 / BUCKET_SIZE for ticker in top["ticker"]}
    if strategy == LONG_SHORT:
        bottom = ranked.iloc[-BUCKET_SIZE:]
        for ticker in bottom["ticker"]:
            weights[ticker] = -1.0 / BUCKET_SIZE
        gross = float(top["y_true"].mean()) - float(bottom["y_true"].mean())
    else:
        gross = float(top["y_true"].mean()) - float(ranked["y_true"].mean())
    return weights, gross


def _traded_notional(
    previous: dict[str, float],
    current: dict[str, float],
) -> float:
    tickers = set(previous) | set(current)
    return float(
        sum(
            abs(current.get(ticker, 0.0) - previous.get(ticker, 0.0))
            for ticker in tickers
        )
    )


def build_period_returns(
    predictions: pd.DataFrame,
    strategy: str = LONG_SHORT,
    step: int = REBALANCE_STEP,
) -> pd.DataFrame:
    """Return one gross return and traded notional per model and rebalance."""
    ordered = _validate(predictions, strategy, step)
    rebalances = select_non_overlapping(ordered, step=step)

    rows: list[dict[str, object]] = []
    models = sorted(rebalances["model"].unique(), key=_model_sort_key)
    for model in models:
        model_rows = rebalances.loc[rebalances["model"].eq(model)]
        previous: dict[str, float] = {}
        for date, group in model_rows.groupby("date", sort=True):
            weights, gross = _book(group, strategy)
            if weights is None:
                traded = float("nan")
            else:
                traded = _traded_notional(previous, weights)
                previous = weights
            rows.append(
                {
                    "model": model,
                    "validation_year": int(group["validation_year"].iloc[0]),
                    "date": date,
                    "n_tickers": len(group),
                    "gross_return": gross,
                    "traded_notional": traded,
                }
            )
    return pd.DataFrame(rows, columns=list(PERIOD_COLUMNS))


def summarize_backtest(periods: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Summarize gross edge, turnover and break-even cost per model."""
    if not isinstance(periods, pd.DataFrame) or periods.empty:
        raise ValueError("periods must be a non-empty DataFrame")
    _require_columns(
        periods, ("model", "validation_year", "gross_return", "traded_notional")
    )

    group_columns = ["model", "validation_year"] if by_year else ["model"]
    rows: list[dict[str, object]] = []
    for key, group in periods.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        gross = group["gross_return"].dropna()
        traded = group["traded_notional"].dropna()
        count = int(len(gross))
        mean_gross = float(gross.mean()) if count else float("nan")
        mean_traded = float(traded.mean()) if len(traded) else float("nan")
        breakeven = float("nan")
        if count and pd.notna(mean_traded) and mean_traded != 0:
            breakeven = 10000.0 * mean_gross / mean_traded
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "periods": count,
                "mean_gross_return": mean_gross,
                "mean_traded_notional": mean_traded,
                "breakeven_cost_bps": breakeven,
                "share_positive_periods": float(gross.gt(0).mean())
                if count
                else float("nan"),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            "periods",
            "mean_gross_return",
            "mean_traded_notional",
            "breakeven_cost_bps",
            "share_positive_periods",
        ],
    )
    summary = summary.sort_values(
        by=group_columns,
        key=lambda column: column.map(_model_sort_key)
        if column.name == "model"
        else column,
    )
    return summary.reset_index(drop=True)
