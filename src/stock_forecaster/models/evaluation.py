"""Walk-forward model execution and out-of-sample metric evaluation."""

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
