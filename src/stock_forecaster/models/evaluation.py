"""Walk-forward model execution and out-of-sample metric evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from stock_forecaster.models.baseline import (
    MeanBaseline,
    MomentumBaseline,
    ZeroBaseline,
)
from stock_forecaster.models.dataset import WalkForwardFold
from stock_forecaster.models.forest import RandomForestForecaster
from stock_forecaster.models.linear import LinearRegressionForecaster
from stock_forecaster.models.metrics import RegressionMetrics, evaluate_predictions

DEFAULT_MODEL_NAMES = (
    "MeanBaseline",
    "MomentumBaseline",
    "LinearRegression",
    "RandomForest",
)
_KNOWN_MODEL_NAMES = ("ZeroBaseline", *DEFAULT_MODEL_NAMES)
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
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> tuple[ModelFoldResult, ...]:
    """Evaluate selected models independently in every walk-forward fold."""
    selected_model_names = _validate_model_names(model_names)
    results: list[ModelFoldResult] = []
    for fold in folds:
        for model_name in selected_model_names:
            if model_name == "ZeroBaseline":
                prediction = ZeroBaseline().predict(fold.validation.X)
            elif model_name == "MeanBaseline":
                prediction = MeanBaseline().fit(fold.train.y).predict(
                    fold.validation.y.index
                )
            elif model_name == "MomentumBaseline":
                prediction = MomentumBaseline().predict(fold.validation.X)
            elif model_name == "LinearRegression":
                prediction = LinearRegressionForecaster().fit(
                    fold.train.X, fold.train.y
                ).predict(fold.validation.X)
            else:
                prediction = RandomForestForecaster().fit(
                    fold.train.X, fold.train.y
                ).predict(fold.validation.X)
            results.append(_result(fold, model_name, prediction))

    if results:
        all_predictions = pd.concat(
            [result.predictions for result in results], ignore_index=True
        )
        if all_predictions.duplicated(["date", "ticker", "model"]).any():
            raise ValueError("OOS date, ticker, and model keys must be unique")
    return tuple(results)


def _validate_model_names(model_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(model_names, str):
        raise ValueError("model_names must be a sequence of model names, not a string")
    names = tuple(model_names)
    if not names:
        raise ValueError("model_names must not be empty")
    if any(model_name not in _KNOWN_MODEL_NAMES for model_name in names):
        raise ValueError("model_names contains an unknown model")
    if len(names) != len(set(names)):
        raise ValueError("model_names must not contain duplicates")
    return names


def _ordered_results(
    results: Sequence[ModelFoldResult],
) -> list[ModelFoldResult]:
    return sorted(
        results,
        key=lambda result: (
            result.validation_year,
            _KNOWN_MODEL_NAMES.index(result.model_name),
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
    ordered_results = _ordered_results(results)
    present_models = {result.model_name for result in ordered_results}
    for model_name in _KNOWN_MODEL_NAMES:
        if model_name not in present_models:
            continue
        pooled = pd.concat(
            [
                result.predictions
                for result in ordered_results
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
    fold_metrics: pd.DataFrame,
    reference_model: str = "MeanBaseline",
) -> pd.DataFrame:
    """Count strict annual metric comparisons against a named same-year model."""
    required_columns = {
        "validation_year",
        "model",
        "MAE",
        "RMSE",
        "directional_accuracy",
        "correlation",
    }
    if not isinstance(fold_metrics, pd.DataFrame) or fold_metrics.empty:
        raise ValueError("fold_metrics must be a non-empty DataFrame")
    if missing_columns := required_columns.difference(fold_metrics.columns):
        raise ValueError(f"fold_metrics is missing required columns: {missing_columns}")
    if fold_metrics.duplicated(["validation_year", "model"]).any():
        raise ValueError("fold_metrics must have unique validation_year and model keys")

    years = fold_metrics["validation_year"].drop_duplicates()
    reference = fold_metrics.loc[
        fold_metrics["model"].eq(reference_model)
    ]
    if reference.empty:
        raise ValueError(f"reference model {reference_model!r} is absent")
    if not reference["validation_year"].isin(years).all() or set(
        reference["validation_year"]
    ) != set(years):
        raise ValueError("reference model must appear exactly once in every year")
    reference = reference.loc[
        :, ["validation_year", "MAE", "RMSE", "directional_accuracy"]
    ].rename(
        columns={
            "MAE": "reference_MAE",
            "RMSE": "reference_RMSE",
            "directional_accuracy": "reference_directional_accuracy",
        }
    )

    rows: list[dict[str, object]] = []
    for model_name in fold_metrics["model"].drop_duplicates():
        model = fold_metrics.loc[fold_metrics["model"].eq(model_name)]
        comparison = model.merge(
            reference,
            on="validation_year",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if len(comparison) != len(model) or not comparison["_merge"].eq("both").all():
            raise ValueError("every model row must have a same-year reference row")
        rows.append(
            {
                "model": model_name,
                "folds_better_mae_vs_reference": int(
                    comparison["MAE"].lt(comparison["reference_MAE"]).sum()
                ),
                "folds_better_rmse_vs_reference": int(
                    comparison["RMSE"].lt(comparison["reference_RMSE"]).sum()
                ),
                "folds_better_direction_vs_reference": int(
                    comparison["directional_accuracy"].gt(
                        comparison["reference_directional_accuracy"]
                    ).sum()
                ),
                "folds_positive_correlation": int(
                    comparison["correlation"].gt(0).sum()
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "folds_better_mae_vs_reference",
            "folds_better_rmse_vs_reference",
            "folds_better_direction_vs_reference",
            "folds_positive_correlation",
        ],
    )
