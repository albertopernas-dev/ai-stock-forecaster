"""Cross-sectional ranking evaluation of out-of-sample return forecasts."""

import pandas as pd

from stock_forecaster.models.evaluation import DEFAULT_MODEL_NAMES, PREDICTION_COLUMNS

MIN_TICKERS_PER_DATE = 5
BUCKET_SIZE = 5
TEST_BOUNDARY = pd.Timestamp("2024-01-01")
KNOWN_MODEL_ORDER = ("ZeroBaseline", *DEFAULT_MODEL_NAMES)

DAILY_COLUMNS = (
    "model",
    "validation_year",
    "date",
    "n_tickers",
    "ic",
    "top_mean",
    "bottom_mean",
    "spread",
)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _validate_predictions(
    predictions: pd.DataFrame,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a non-empty DataFrame")
    _require_columns(predictions, PREDICTION_COLUMNS)
    ordered = predictions.loc[:, PREDICTION_COLUMNS].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    if ordered.duplicated(subset=["date", "ticker", "model"]).any():
        raise ValueError("Duplicate date, ticker and model rows")
    if not allow_holdout and ordered["date"].ge(TEST_BOUNDARY).any():
        raise ValueError(
            "predictions must not contain rows dated 2024-01-01 or later; "
            "pass allow_holdout=True only for the one pre-registered "
            "confirmation"
        )
    return ordered


def _model_sort_key(model_name: str) -> int:
    if model_name in KNOWN_MODEL_ORDER:
        return KNOWN_MODEL_ORDER.index(model_name)
    return len(KNOWN_MODEL_ORDER)


def _daily_row(
    model: str,
    year: int,
    date: pd.Timestamp,
    group: pd.DataFrame,
) -> dict[str, object]:
    prediction = group["prediction"]
    realized = group["y_true"]
    n_tickers = len(group)
    predictions_vary = prediction.nunique(dropna=True) > 1

    ic = float("nan")
    if (
        n_tickers >= MIN_TICKERS_PER_DATE
        and predictions_vary
        and realized.nunique(dropna=True) > 1
    ):
        ic = float(prediction.corr(realized, method="spearman"))

    top_mean = float("nan")
    bottom_mean = float("nan")
    spread = float("nan")
    if n_tickers >= 2 * BUCKET_SIZE and predictions_vary:
        ranked = group.sort_values(
            by=["prediction", "ticker"], ascending=[False, True]
        )
        top_mean = float(ranked["y_true"].iloc[:BUCKET_SIZE].mean())
        bottom_mean = float(ranked["y_true"].iloc[-BUCKET_SIZE:].mean())
        spread = top_mean - bottom_mean

    return {
        "model": model,
        "validation_year": year,
        "date": date,
        "n_tickers": n_tickers,
        "ic": ic,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "spread": spread,
    }


def build_daily_ranking(
    predictions: pd.DataFrame,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Return one cross-sectional ranking row per model and trading date."""
    ordered = _validate_predictions(predictions, allow_holdout=allow_holdout)
    rows = [
        _daily_row(model, year, date, group)
        for (model, year, date), group in ordered.groupby(
            ["model", "validation_year", "date"], sort=False
        )
    ]
    frame = pd.DataFrame(rows, columns=list(DAILY_COLUMNS))
    frame = frame.sort_values(
        by=["model", "validation_year", "date"],
        key=lambda column: column.map(_model_sort_key)
        if column.name == "model"
        else column,
    )
    return frame.reset_index(drop=True)


def select_non_overlapping(daily: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Return every step-th distinct date within each validation year."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("daily must be a non-empty DataFrame")
    _require_columns(daily, ("validation_year", "date"))

    keep: set[tuple[object, pd.Timestamp]] = set()
    for year, group in daily.groupby("validation_year", sort=True):
        dates = sorted(pd.to_datetime(group["date"]).unique())
        keep.update((year, date) for date in dates[::step])

    selected = [
        (year, date) in keep
        for year, date in zip(
            daily["validation_year"], pd.to_datetime(daily["date"]), strict=True
        )
    ]
    return daily.loc[selected].reset_index(drop=True)


def _describe(values: pd.Series) -> tuple[int, float, float, float]:
    defined = values.dropna()
    count = int(len(defined))
    if count == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(defined.mean())
    deviation = float(defined.std(ddof=1)) if count >= 2 else float("nan")
    share_positive = float(defined.gt(0).mean())
    return count, mean, deviation, share_positive


def summarize_ranking(daily: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Summarize daily ranking evidence per model, optionally per year."""
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        raise ValueError("daily must be a non-empty DataFrame")
    _require_columns(daily, ("model", "validation_year", "ic", "spread"))

    group_columns = ["model", "validation_year"] if by_year else ["model"]
    rows: list[dict[str, object]] = []
    for key, group in daily.groupby(group_columns, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        ic_days, ic_mean, ic_std, ic_share = _describe(group["ic"])
        spread_days, spread_mean, _, spread_share = _describe(group["spread"])
        stability = float("nan")
        if ic_days >= 2 and pd.notna(ic_std) and ic_std != 0:
            stability = ic_mean / ic_std
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            {
                "ic_days": ic_days,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_share_positive": ic_share,
                "ic_stability": stability,
                "spread_days": spread_days,
                "spread_mean": spread_mean,
                "spread_share_positive": spread_share,
            }
        )
        rows.append(row)

    summary = pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            "ic_days",
            "ic_mean",
            "ic_std",
            "ic_share_positive",
            "ic_stability",
            "spread_days",
            "spread_mean",
            "spread_share_positive",
        ],
    )
    summary = summary.sort_values(
        by=group_columns,
        key=lambda column: column.map(_model_sort_key)
        if column.name == "model"
        else column,
    )
    return summary.reset_index(drop=True)
