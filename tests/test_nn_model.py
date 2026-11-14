"""Tests for the trained near-accident risk model."""

import builtins
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

import nn_model as nn_module
from model_artifacts import FilesystemModelArtifactStore, HISTORY_FILE, MODEL_FILE, PLOT_FILE
from nn_model import SpeedLimitNN
from tests._support import pytest


class FixedRegressor:
    def __init__(self, **_):
        self.fits = 0

    def partial_fit(self, X, y):
        self.fits += 1
        return self

    def predict(self, X):
        return np.zeros(len(X))


def training_data(rows=30, *, dark=True):
    hours = pd.date_range("2024-01-01", periods=rows, freq="h")
    return pd.DataFrame({
        "illuminance_millilux": 100 if dark else 50000,
        "water_level_um": 0.0,
        "temperature_c": 10.0,
        "speed_limit_kmh": np.resize([40.0, 50.0, 60.0], rows),
        "feature": np.arange(rows, dtype=float),
    }), pd.Series(np.arange(rows, dtype=float) / 3)


def create_model(S, model_dir, *, min_rows=1, epochs=3, patience=1):
    config = S.technical.nn
    config = nn_module.NNSettings(
        hidden_layer_sizes=(2,),
        learning_rate=config.learning_rate,
        max_epochs=epochs,
        patience=patience,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_seed=42,
        min_training_rows=min_rows,
    )
    return SpeedLimitNN(
        FilesystemModelArtifactStore(model_dir),
        S.business.weather,
        S.business.speed,
        config,
    )


def test_training_persists_report_model_history_and_prediction(S, tmp_path):
    model = create_model(S, tmp_path, patience=2)
    X, y = training_data()
    with patch.object(nn_module, "MLPRegressor", FixedRegressor):
        report = model.train(X, y)

    assert (report.n_train, report.n_validation, report.n_test) == (18, 6, 6)
    assert report.epochs_run == 3
    assert report.candidate_support == {40: 10, 50: 10, 60: 10}
    assert model.candidate_speeds == (40, 50, 60)
    assert (tmp_path / MODEL_FILE).exists()
    assert (tmp_path / HISTORY_FILE).exists()
    assert (tmp_path / PLOT_FILE).exists()
    history = json.loads((tmp_path / HISTORY_FILE).read_text())
    assert len(history["train_mae"]) == len(history["validation_mae"]) == 3

    loaded = create_model(S, tmp_path)
    loaded.load()
    np.testing.assert_array_equal(loaded.predict_risk(X.iloc[[0]]), [0.0])


def test_training_rejects_too_few_training_rows(S, tmp_path):
    model = create_model(S, tmp_path, min_rows=20)
    X, y = training_data(rows=19, dark=False)
    with pytest.raises(ValueError, match="not enough chronological data"):
        model.train(X, y)


def test_training_rejects_candidate_speeds_outside_bounds(S, tmp_path):
    model = create_model(S, tmp_path)
    X, y = training_data()
    X["speed_limit_kmh"] = 100
    with pytest.raises(ValueError, match="outside configured output bounds"):
        model.train(X, y)


def test_training_rejects_nonfinite_validation_scores(S, tmp_path, monkeypatch):
    model = create_model(S, tmp_path)
    X, y = training_data()
    monkeypatch.setattr(nn_module, "MLPRegressor", FixedRegressor)
    monkeypatch.setattr(nn_module, "mean_absolute_error", lambda *_: float("nan"))
    with pytest.raises(RuntimeError, match="did not produce a validation model"):
        model.train(X, y)


@pytest.mark.parametrize(
    "features,target,message",
    [
        (pd.DataFrame({"feature": [1.0]}), pd.Series([0.0]), "must include speed_limit_kmh"),
        (pd.DataFrame({"speed_limit_kmh": [40.0]}), pd.Series(dtype=float), "same non-zero"),
        (pd.DataFrame({"speed_limit_kmh": [float("nan")]}), pd.Series([0.0]), "must be finite"),
        (pd.DataFrame({"speed_limit_kmh": [40.0]}), pd.Series([-1.0]), "cannot be negative"),
    ],
)
def test_training_validation_errors(S, tmp_path, features, target, message):
    model = create_model(S, tmp_path)
    with pytest.raises(ValueError, match=message):
        model._validate_training_data(features, target)


def test_prediction_requires_loaded_or_trained_model(S, tmp_path):
    model = create_model(S, tmp_path)
    X, _ = training_data()
    with pytest.raises(RuntimeError, match="not trained or loaded"):
        model.predict_risk(X.iloc[[0]])


def test_prediction_clamps_negative_risk_to_zero(S, tmp_path):
    model = create_model(S, tmp_path)
    X, _ = training_data()
    model._model = FixedRegressor()
    model._scaler = nn_module.StandardScaler().fit(X)
    model._columns = list(X.columns)
    np.testing.assert_array_equal(model.predict_risk(X.iloc[[0]]), [0.0])


def test_prediction_rejects_nonfinite_model_output(S, tmp_path):
    class NonfiniteRegressor:
        def predict(self, X):
            return np.full(len(X), np.nan)

    model = create_model(S, tmp_path)
    X, _ = training_data()
    model._model = NonfiniteRegressor()
    model._scaler = nn_module.StandardScaler().fit(X)
    model._columns = list(X.columns)
    with pytest.raises(ValueError, match="non-finite prediction"):
        model.predict_risk(X.iloc[[0]])


def test_candidate_speeds_unavailable_before_training(S, tmp_path):
    with pytest.raises(RuntimeError, match="no supported candidate speeds"):
        _ = create_model(S, tmp_path).candidate_speeds


def test_empty_triggered_data_has_no_mean_or_interval(S, tmp_path, monkeypatch):
    model = create_model(S, tmp_path)
    X, y = training_data(dark=False)
    monkeypatch.setattr(nn_module, "MLPRegressor", FixedRegressor)
    report = model.train(X, y)
    assert report.test.n_triggered_hours == 0
    assert report.test.observed_triggered_mean is None
    assert report.test.triggered_ci95 is None
    assert model._bootstrap_mean_ci(np.array([])) is None


def test_load_rejects_other_strategy_and_missing_bundle(S, tmp_path):
    model = create_model(S, tmp_path)
    with pytest.raises(FileNotFoundError):
        model.load()
    import joblib

    joblib.dump({"strategy": "other"}, tmp_path / MODEL_FILE)
    with pytest.raises(ValueError, match="retrain"):
        model.load()


def test_plot_skips_when_matplotlib_is_missing(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    FilesystemModelArtifactStore._plot_loss_curves(
        {"train_mae": [1], "validation_mae": [2], "best_epoch": [0]},
        tmp_path / "mae.png",
    )
    assert not (tmp_path / "mae.png").exists()


def test_report_is_persisted_as_json(tmp_path):
    store = FilesystemModelArtifactStore(tmp_path)
    path = store.save_report("evaluation.json", {"mae": 0.25})
    assert json.loads(path.read_text()) == {"mae": 0.25}


def test_training_uses_validation_mae_for_early_stopping(S, tmp_path, monkeypatch):
    model = create_model(S, tmp_path, epochs=5, patience=1)
    X, y = training_data()
    monkeypatch.setattr(nn_module, "MLPRegressor", FixedRegressor)
    report = model.train(X, y)
    assert report.best_epoch == 1
    assert report.epochs_run == 2


def test_training_can_finish_at_epoch_limit_without_early_stopping(S, tmp_path, monkeypatch):
    model = create_model(S, tmp_path, epochs=1, patience=2)
    X, y = training_data()
    monkeypatch.setattr(nn_module, "MLPRegressor", FixedRegressor)
    report = model.train(X, y)
    assert report.epochs_run == report.best_epoch == 1
