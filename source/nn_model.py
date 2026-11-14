"""Type A / Software 2.0 model for hourly near-accident risk."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_pipeline import (
    ILLUMINANCE_MILLILUX,
    SPEED_LIMIT_KMH,
    TEMPERATURE_C,
    WATER_LEVEL_UM,
    NEAR_ACCIDENTS,
)
from settings import NNSettings, SpeedRules, WeatherRules
from weather_router import WeatherRouter

LOG = logging.getLogger("speedlimit.nn")


class ModelArtifactStore(Protocol):
    def save(self, bundle: dict[str, Any], history: dict[str, list]) -> None: ...

    def load(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EvaluationReport:
    n_hours: int
    n_triggered_hours: int
    observed_triggered_mean: float | None
    triggered_ci95: tuple[float, float] | None
    predicted_triggered_mean: float | None
    observed_all_hours_mean: float
    predicted_all_hours_mean: float
    mae: float
    rmse: float
    baseline_mae: float


@dataclass(frozen=True)
class TrainingReport:
    strategy: str
    n_train: int
    n_validation: int
    n_test: int
    epochs_run: int
    best_epoch: int
    validation_mae: float
    test: EvaluationReport
    candidate_support: dict[int, int]


class SpeedLimitNN:
    """Estimate observed near-accident counts from weather, light, and speed.

    Candidate predictions are observational estimates, not evidence that
    changing the speed limit causes fewer accidents.
    """

    STRATEGY = "near_accident_rate"

    def __init__(
        self,
        artifacts: ModelArtifactStore,
        weather_rules: WeatherRules,
        speed_rules: SpeedRules,
        nn_settings: NNSettings,
    ) -> None:
        self._artifacts = artifacts
        self._speed = speed_rules
        self._cfg = nn_settings
        self._router = WeatherRouter(weather_rules)
        self._model: MLPRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._columns: list[str] = []
        self._candidate_speeds: tuple[int, ...] = ()
        self._training_mean = 0.0

    @property
    def candidate_speeds(self) -> tuple[int, ...]:
        if not self._candidate_speeds:
            raise RuntimeError("model has no supported candidate speeds")
        return self._candidate_speeds

    def train(self, features: pd.DataFrame, target: pd.Series) -> TrainingReport:
        X, y = self._validate_training_data(features, target)
        n = len(X)
        test_start = int(n * (1.0 - self._cfg.test_fraction))
        validation_start = int(n * (1.0 - self._cfg.test_fraction - self._cfg.validation_fraction))
        if (validation_start < self._cfg.min_training_rows
                or test_start <= validation_start or test_start >= n):
            raise ValueError("not enough chronological data for train, validation, and test splits")

        X_train, X_validation, X_test = (
            X.iloc[:validation_start],
            X.iloc[validation_start:test_start],
            X.iloc[test_start:],
        )
        y_train, y_validation, y_test = (
            y.iloc[:validation_start],
            y.iloc[validation_start:test_start],
            y.iloc[test_start:],
        )
        self._columns = list(X.columns)
        self._candidate_speeds = tuple(
            sorted(int(value) for value in X_train[SPEED_LIMIT_KMH].unique())
        )
        if any(speed < self._speed.min_kmh or speed > self._speed.max_kmh
               for speed in self._candidate_speeds):
            raise ValueError("observed candidate speed is outside configured output bounds")
        self._training_mean = float(y_train.mean())

        self._scaler = StandardScaler().fit(X_train)
        x_train = self._scaler.transform(X_train)
        x_validation = self._scaler.transform(X_validation)
        x_train_validation = self._scaler.transform(pd.concat((X_train, X_validation)))
        y_train_validation = pd.concat((y_train, y_validation)).to_numpy()

        selector = self._new_model()
        history: dict[str, list] = {"train_mae": [], "validation_mae": [], "best_epoch": [0]}
        best_model: MLPRegressor | None = None
        best_epoch = 0
        best_validation_mae = float("inf")
        stale_epochs = 0

        for epoch in range(1, self._cfg.max_epochs + 1):
            selector.partial_fit(x_train, y_train.to_numpy())
            train_mae = mean_absolute_error(y_train, selector.predict(x_train))
            validation_mae = mean_absolute_error(y_validation, selector.predict(x_validation))
            history["train_mae"].append(float(train_mae))
            history["validation_mae"].append(float(validation_mae))
            if validation_mae < best_validation_mae - 1e-8:
                best_model = copy.deepcopy(selector)
                best_validation_mae = float(validation_mae)
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self._cfg.patience:
                    break

        if best_model is None:
            raise RuntimeError("training did not produce a validation model")

        # Refit for the validation-selected epoch count; the test set stays untouched.
        final_model = self._new_model()
        for _ in range(best_epoch):
            final_model.partial_fit(x_train_validation, y_train_validation)
        self._model = final_model
        history["best_epoch"] = [best_epoch]

        test_report = self.evaluate(X_test, y_test)
        report = TrainingReport(
            strategy=self.STRATEGY,
            n_train=len(X_train),
            n_validation=len(X_validation),
            n_test=len(X_test),
            epochs_run=len(history["train_mae"]),
            best_epoch=best_epoch,
            validation_mae=best_validation_mae,
            test=test_report,
            candidate_support={
                speed: int((X[SPEED_LIMIT_KMH] == speed).sum())
                for speed in self._candidate_speeds
            },
        )
        LOG.info("Risk-model training report: %s", report)
        self._artifacts.save(
            {
                "strategy": self.STRATEGY,
                "model": self._model,
                "scaler": self._scaler,
                "columns": self._columns,
                "candidate_speeds": self._candidate_speeds,
                "training_mean": self._training_mean,
                "report": asdict(report),
            },
            history,
        )
        return report

    def predict_risk(self, features: pd.DataFrame) -> np.ndarray:
        """Return non-negative predicted near-accidents/hour for each feature row."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("model not trained or loaded")
        scaled = self._scaler.transform(features[self._columns])
        values = np.asarray(self._model.predict(scaled), dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("risk model produced a non-finite prediction")
        return np.maximum(values, 0.0)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> EvaluationReport:
        X, y = self._validate_training_data(features, target)
        prediction = self.predict_risk(X)
        triggered = self._triggered_mask(X)
        observed_triggered = y.to_numpy()[triggered]
        predicted_triggered = prediction[triggered]
        return EvaluationReport(
            n_hours=len(X),
            n_triggered_hours=int(triggered.sum()),
            observed_triggered_mean=self._mean_or_none(observed_triggered),
            triggered_ci95=self._bootstrap_mean_ci(observed_triggered),
            predicted_triggered_mean=self._mean_or_none(predicted_triggered),
            observed_all_hours_mean=float(y.mean()),
            predicted_all_hours_mean=float(prediction.mean()),
            mae=float(mean_absolute_error(y, prediction)),
            rmse=float(np.sqrt(mean_squared_error(y, prediction))),
            baseline_mae=float(mean_absolute_error(y, np.full(len(y), self._training_mean))),
        )

    def load(self) -> None:
        bundle = self._artifacts.load()
        if bundle.get("strategy") != self.STRATEGY:
            raise ValueError("saved model is not a near-accident risk model; retrain it")
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._columns = bundle["columns"]
        self._candidate_speeds = tuple(bundle["candidate_speeds"])
        self._training_mean = float(bundle["training_mean"])

    def _new_model(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self._cfg.hidden_layer_sizes,
            learning_rate_init=self._cfg.learning_rate,
            random_state=self._cfg.random_seed,
            shuffle=False,
        )

    def _validate_training_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        if SPEED_LIMIT_KMH not in features:
            raise ValueError(f"training features must include {SPEED_LIMIT_KMH}")
        if len(features) != len(target) or features.empty:
            raise ValueError("features and target must have the same non-zero row count")
        X = features.reset_index(drop=True).copy()
        y = pd.to_numeric(target, errors="coerce").reset_index(drop=True).astype(float)
        if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y).all():
            raise ValueError("training features and target must be finite")
        if (y < 0).any():
            raise ValueError("near-accident counts cannot be negative")
        return X, y

    def _triggered_mask(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([
            self._router.should_invoke_nn(illum, water, temp)
            for illum, water, temp in zip(
                features[ILLUMINANCE_MILLILUX],
                features[WATER_LEVEL_UM],
                features[TEMPERATURE_C],
            )
        ])

    @staticmethod
    def _mean_or_none(values: np.ndarray) -> float | None:
        return float(values.mean()) if len(values) else None

    def _bootstrap_mean_ci(self, values: np.ndarray) -> tuple[float, float] | None:
        if not len(values):
            return None
        generator = np.random.default_rng(self._cfg.random_seed)
        samples = generator.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, (0.025, 0.975))
        return float(low), float(high)
