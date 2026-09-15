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
    build_consistency_summary,
    build_fold_metrics_table,
    build_pooled_metrics,
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


def _assert_metrics_equal(left: RegressionMetrics, right: RegressionMetrics) -> None:
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
        "MeanBaseline": MeanBaseline().fit(fold.train.y).predict(fold.validation.y.index),
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
        pd.testing.assert_frame_equal(records.loc[:, ["date", "ticker"]], fold.validation.metadata)
        pd.testing.assert_series_equal(records["y_true"], fold.validation.y, check_names=False)
        assert records["validation_year"].eq(2016).all()
        assert records["model"].eq(result.model_name).all()
        assert records.index.equals(fold.validation.y.index)
        pd.testing.assert_series_equal(
            records["prediction"], expected_predictions[result.model_name], check_names=False
        )


def test_stored_metrics_equal_direct_recomputation_and_keys_are_unique():
    results = evaluate_walk_forward((_fold(2016), _fold(2017)))

    for result in results:
        expected = evaluate_predictions(result.predictions["y_true"], result.predictions["prediction"])
        _assert_metrics_equal(result.metrics, expected)
    combined = pd.concat([result.predictions for result in results])
    assert not combined.duplicated(["date", "ticker", "model"]).any()


def test_model_fold_result_is_frozen():
    result = evaluate_walk_forward((_fold(2016),))[0]
    with pytest.raises(FrozenInstanceError):
        result.validation_year = 2020


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
    monkeypatch.setattr(evaluation_module, "LinearRegressionForecaster", fitted_spy("linear"))
    monkeypatch.setattr(evaluation_module, "RandomForestForecaster", fitted_spy("forest"))

    folds = (_fold(2016), _fold(2017, train_shift=0.03))
    evaluate_walk_forward(folds)

    assert all(len(instances) == 2 for instances in created.values())
    assert all(instances[0] is not instances[1] for instances in created.values())
    for index, fold in enumerate(folds):
        assert created["mean"][index].fit_y is fold.train.y
        for kind in ("linear", "forest"):
            assert created[kind][index].fit_X is fold.train.X
            assert created[kind][index].fit_y is fold.train.y


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
