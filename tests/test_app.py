"""Tests for application wiring and command-line orchestration."""

import logging
import queue
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

import app as app_module
import air_quality_api
import air_quality_llm
import data_io
import data_pipeline
import gui_app
import guardrails
import llm_clients
import nn_model
import model_artifacts
import sensor_simulator
import speed_controller
import weather_router
from nn_model import EvaluationReport, TrainingReport
from settings import ConfigError
from tests._support import pytest


def test_configure_logging_creates_folder_and_selects_level(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(app_module.logging, "basicConfig", lambda **kwargs: seen.append(kwargs))
    monkeypatch.setattr(app_module.logging, "FileHandler", lambda *_args, **_kwargs: object())
    app_module.configure_logging(tmp_path / "logs", verbose=True)
    assert (tmp_path / "logs").is_dir()
    assert seen[-1]["level"] == logging.DEBUG
    app_module.configure_logging(tmp_path / "logs", verbose=False)
    assert seen[-1]["level"] == logging.INFO


def test_runtime_uses_annual_artifacts_when_available(tmp_path):
    assert app_module._runtime_model_dir(tmp_path) == tmp_path / "year_n"
    annual = tmp_path / "annual"
    annual.mkdir()
    (annual / model_artifacts.MODEL_FILE).touch()
    assert app_module._runtime_model_dir(tmp_path) == annual


def test_apply_overrides_preserves_defaults_and_applies_flags(S):
    args = SimpleNamespace(ui=None, seconds_per_hour=None)
    unchanged = app_module.apply_overrides(S, args)
    assert unchanged.technical.runtime.ui == S.technical.runtime.ui
    changed = app_module.apply_overrides(
        S, SimpleNamespace(ui="headless", seconds_per_hour=0.25))
    assert changed.technical.runtime.ui == "headless"
    assert changed.technical.simulator.seconds_per_sim_hour == 0.25
    with pytest.raises(ConfigError, match="seconds_per_sim_hour"):
        app_module.apply_overrides(S, SimpleNamespace(ui=None, seconds_per_hour=0))


def test_parse_args_defaults_and_explicit_options():
    defaults = app_module.parse_args(["run"])
    assert defaults.config == Path("configuration.toml")
    assert defaults.ui is None and not defaults.verbose
    args = app_module.parse_args(["train", "--ui", "headless", "--seconds-per-hour", "0.5", "-v"])
    assert args.command == "train" and args.verbose and args.seconds_per_hour == 0.5


def test_factory_runs_training_pipeline(S, tmp_path, monkeypatch):
    paths = S.technical.paths
    settings = S.__class__(
        S.business,
        S.technical.__class__(
            paths=paths.__class__(tmp_path, tmp_path, tmp_path, paths.sensor_readings_file,
                                  paths.accidents_file, paths.sensor_readings_year_plus_one_file,
                                  paths.accidents_year_plus_one_file),
            nn=S.technical.nn, llm=S.technical.llm, simulator=S.technical.simulator,
            air_quality_api=S.technical.air_quality_api, runtime=S.technical.runtime,
        ),
    )
    calls = []

    class Cleaner:
        def __init__(self, _bounds, **_kwargs):
            pass

        def clean_sensor_readings(self, path):
            calls.append(("sensors", path))
            return "sensors"

        def clean_accidents(self, path):
            calls.append(("accidents", path))
            return "accidents"

    class Reader:
        def read_sensor_readings(self, path, *, headerless=False):
            calls.append(("read_sensors", path, headerless))
            return path

        def read_accidents(self, path):
            calls.append(("read_accidents", path))
            return path

    class Engineer:
        def __init__(self, *_):
            pass

        def build_training_set(self, sensors, accidents):
            calls.append(("features", sensors, accidents))
            return "X", "y"

    class Model:
        def __init__(self, *args):
            calls.append(("model", args))

        def train(self, features, labels):
            calls.append(("train", features, labels))
            return TrainingReport(
                strategy="test",
                n_train=1,
                n_validation=1,
                n_test=1,
                epochs_run=1,
                best_epoch=1,
                validation_mae=0.1,
                test=EvaluationReport(
                    n_hours=1, n_triggered_hours=1, observed_triggered_mean=0.1,
                    triggered_ci95=(0.0, 0.2), predicted_triggered_mean=0.1,
                    observed_all_hours_mean=0.1, predicted_all_hours_mean=0.1,
                    mae=0.1, rmse=0.1, baseline_mae=0.2,
                ),
                candidate_support={40: 1},
            )

    monkeypatch.setattr(data_pipeline, "DataCleaner", Cleaner)
    monkeypatch.setattr(data_pipeline, "FeatureEngineer", Engineer)
    monkeypatch.setattr(nn_model, "SpeedLimitNN", Model)
    monkeypatch.setattr(data_io, "CSVDataReader", Reader)
    monkeypatch.setattr(model_artifacts, "FilesystemModelArtifactStore", lambda path: path)
    app_module.ApplicationFactory(settings).run_training()
    assert calls[-1] == ("train", "X", "y")
    model_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "model")
    assert model_call[1][0] == tmp_path / "year_n"
    assert (tmp_path / "models").is_dir() or tmp_path.is_dir()


def test_factory_builds_runtime_and_connects_listener(S, monkeypatch):
    made = {}

    class FakeNN:
        def __init__(self, *_):
            made["nn"] = self

        def load(self):
            made["loaded"] = True

    class FakeCleaner:
        def __init__(self, *_, **__):
            pass

        def clean_sensor_readings(self, path):
            made["readings"] = path
            return "readings"

        def clean_accidents(self, path):
            made["accidents"] = path
            return "accidents"

    class FakeReader:
        def read_sensor_readings(self, path, *, headerless=False):
            made.setdefault("sensor_paths", []).append((path, headerless))
            return path

        def read_accidents(self, path):
            made.setdefault("accident_paths", []).append(path)
            return path

    class FakeEngineer:
        def __init__(self, *_):
            pass

    class FakeController:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.process = Mock()

        def add_listener(self, listener):
            made["listener"] = listener

    class FakeSimulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(nn_model, "SpeedLimitNN", FakeNN)
    monkeypatch.setattr(data_io, "CSVDataReader", FakeReader)
    monkeypatch.setattr(model_artifacts, "FilesystemModelArtifactStore", lambda path: path)
    monkeypatch.setattr(data_pipeline, "DataCleaner", FakeCleaner)
    monkeypatch.setattr(data_pipeline, "FeatureEngineer", FakeEngineer)
    monkeypatch.setattr(speed_controller, "SpeedLimitController", FakeController)
    monkeypatch.setattr(sensor_simulator, "SensorSimulator", FakeSimulator)
    monkeypatch.setattr(air_quality_api, "AirQualityAPIClient", lambda *_: "aq")
    monkeypatch.setattr(air_quality_llm, "AirQualityLLMEvaluator", lambda *args: ("llm", args))
    monkeypatch.setattr(llm_clients, "OllamaClient", lambda *args: ("ollama", args))
    monkeypatch.setattr(guardrails, "InputGuardrail", lambda *args: ("input", args))
    monkeypatch.setattr(guardrails, "OutputGuardrail", lambda *args: ("output", args))
    monkeypatch.setattr(weather_router, "WeatherRouter", lambda *args: ("router", args))

    app = app_module.ApplicationFactory(S).build_runtime()
    paths = S.technical.paths
    assert made["loaded"]
    assert made["sensor_paths"] == [(paths.data_dir / paths.sensor_readings_file, False)]
    assert made["readings"] == paths.data_dir / paths.sensor_readings_file
    assert "accident_paths" not in made                 # runtime never reads accidents
    assert app.simulator.kwargs["on_reading"] == app.controller.process
    assert app.simulator.kwargs["stop_on_callback_error"] is True
    assert app.telemetry_queue is not None


def test_year_plus_one_evaluation_is_saved_without_safety_claim(S, monkeypatch, tmp_path):
    calls = []
    evaluation = EvaluationReport(
        n_hours=1, n_triggered_hours=1, observed_triggered_mean=0.5,
        triggered_ci95=(0.0, 1.0), predicted_triggered_mean=0.4,
        observed_all_hours_mean=0.5, predicted_all_hours_mean=0.4,
        mae=0.1, rmse=0.1, baseline_mae=0.2,
    )

    class Model:
        def __init__(self, *_):
            pass

        def load(self):
            calls.append("load")

        def evaluate(self, features, target):
            calls.append(("evaluate", features, target))
            return evaluation

    class Artifacts:
        def __init__(self, path):
            calls.append(("artifacts", path))

        def save_report(self, filename, report):
            calls.append(("save", filename, report))
            return tmp_path / filename

    monkeypatch.setattr(nn_model, "SpeedLimitNN", Model)
    monkeypatch.setattr(model_artifacts, "FilesystemModelArtifactStore", Artifacts)
    factory = app_module.ApplicationFactory(S)
    periods = []
    factory._training_set = lambda *, year_plus_one=False: (periods.append(year_plus_one)
                                                             or ("features", "target"))
    report = factory.evaluate_year_plus_one()

    assert periods == [True]
    assert ("evaluate", "features", "target") in calls
    assert calls.index("load") < next(i for i, call in enumerate(calls)
                                       if isinstance(call, tuple) and call[0] == "evaluate")
    assert ("artifacts", S.technical.paths.model_dir / "year_n") in calls
    assert calls[-1][0:2] == ("save", "year_plus_one_evaluation.json")
    assert report["causal_safety_claim"] is False
    assert "observational" in report["limitation"].lower()
    assert "temporal autocorrelation" in report["uncertainty_limitation"]


def test_annual_retraining_evaluates_before_training(S, monkeypatch):
    calls = []

    class Model:
        def __init__(self, *args):
            calls.append(("model_artifacts", args[0]))

        def train(self, features, target):
            calls.append(("train", features, target))
            return TrainingReport(
                strategy="test", n_train=1, n_validation=1, n_test=1,
                epochs_run=1, best_epoch=1, validation_mae=0.1,
                test=EvaluationReport(
                    n_hours=1, n_triggered_hours=0, observed_triggered_mean=None,
                    triggered_ci95=None, predicted_triggered_mean=None,
                    observed_all_hours_mean=0.0, predicted_all_hours_mean=0.0,
                    mae=0.0, rmse=0.0, baseline_mae=0.0,
                ),
                candidate_support={40: 1},
            )

    monkeypatch.setattr(nn_model, "SpeedLimitNN", Model)
    monkeypatch.setattr(model_artifacts, "FilesystemModelArtifactStore", lambda path: path)
    factory = app_module.ApplicationFactory(S)
    factory.evaluate_year_plus_one = lambda: calls.append("evaluate")
    periods = {False: (pd.DataFrame({"n": [1]}), pd.Series([1])),
               True: (pd.DataFrame({"n": [2]}), pd.Series([2]))}

    def training_set(*, year_plus_one=False):
        calls.append(("period", year_plus_one))
        return periods[year_plus_one]

    factory._training_set = training_set

    factory.run_annual_retraining()
    assert calls[:3] == ["evaluate", ("period", False), ("period", True)]
    assert calls[-1][0] == "train"
    assert calls[-1][1]["n"].tolist() == [1, 2]
    assert calls[-1][2].tolist() == [1, 2]
    assert ("model_artifacts", S.technical.paths.model_dir / "annual") in calls


def test_headless_runner_finishes_and_propagates_callback_error(monkeypatch):
    monkeypatch.setattr(app_module.signal, "signal", lambda *_: None)

    class Simulator:
        callback_error = None
        finished = SimpleNamespace(is_set=lambda: True)

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    app = SimpleNamespace(simulator=Simulator(), telemetry_queue=queue.Queue())
    app_module.run_headless(app)
    app.simulator.callback_error = ValueError("boom")
    with pytest.raises(RuntimeError, match="could not be evaluated") as error:
        app_module.run_headless(app)
    assert isinstance(error.value.__cause__, ValueError)


def test_headless_runner_logs_decisions_and_can_stop_from_signal(monkeypatch):
    handlers = {}
    monkeypatch.setattr(app_module.signal, "signal", lambda signum, callback: handlers.update(
        {signum: callback}))

    class Simulator:
        callback_error = None
        finished = SimpleNamespace(is_set=lambda: True)

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    app = SimpleNamespace(simulator=Simulator(), telemetry_queue=queue.Queue())
    app.telemetry_queue.put("decision")
    app_module.run_headless(app)

    app.simulator.finished = SimpleNamespace(is_set=lambda: False)

    class InterruptingQueue:
        def get(self, timeout):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            raise queue.Empty

    app.telemetry_queue = InterruptingQueue()
    app_module.run_headless(app)


def test_gui_runner_stops_simulator_on_success_and_failure(monkeypatch):
    class Simulator:
        callback_error = None
        calls = 0

        def start(self):
            self.calls += 1

        def stop(self):
            self.calls += 1

        def join(self):
            self.calls += 1

    sim = Simulator()
    app = SimpleNamespace(
        settings=SimpleNamespace(business=SimpleNamespace(speed=SimpleNamespace(
            min_kmh=20, max_kmh=80, default_kmh=80))),
        controller=object(), simulator=sim, telemetry_queue=queue.Queue(),
    )
    dashboard = Mock()
    monkeypatch.setattr(gui_app, "OperatorDashboard", Mock(return_value=dashboard), raising=False)
    app_module.run_gui(app)
    dashboard.run.assert_called_once()
    assert sim.calls == 3

    sim.callback_error = ValueError("failed")
    with pytest.raises(RuntimeError, match="could not be evaluated"):
        app_module.run_gui(app)


def test_main_handles_configuration_and_runtime_errors(S, monkeypatch, capsys):
    monkeypatch.setattr(app_module, "configure_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "load_settings", lambda *_: S)
    monkeypatch.setattr(app_module, "ApplicationFactory", lambda *_: SimpleNamespace(
        run_training=lambda: None, build_runtime=lambda: object()))
    monkeypatch.setattr(app_module, "run_gui", lambda _app: None)
    monkeypatch.setattr(app_module, "run_headless", lambda _app: None)
    assert app_module.main(["train"]) == 0
    assert app_module.main(["run", "--ui", "headless"]) == 0

    monkeypatch.setattr(app_module, "load_settings", lambda *_: (_ for _ in ()).throw(ConfigError("bad")))
    assert app_module.main(["run"]) == 1
    assert "Configuration error: bad" in capsys.readouterr().err

    monkeypatch.setattr(app_module, "load_settings", lambda *_: S)
    for exception, status in [
        (ConfigError("bad runtime"), 1),
        (ModuleNotFoundError("missing", name="not_ready"), 2),
        (FileNotFoundError("model"), 3),
        (RuntimeError("stopped"), 4),
    ]:
        monkeypatch.setattr(app_module, "ApplicationFactory", lambda *_args, e=exception:
                            SimpleNamespace(run_training=lambda: None,
                                            build_runtime=lambda: (_ for _ in ()).throw(e)))
        assert app_module.main(["run"]) == status


def test_main_routes_evaluate_and_retrain_commands(S, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "configure_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "load_settings", lambda *_: S)
    monkeypatch.setattr(app_module, "ApplicationFactory", lambda *_: SimpleNamespace(
        evaluate_year_plus_one=lambda: calls.append("evaluate"),
        run_annual_retraining=lambda: calls.append("retrain"),
    ))
    assert app_module.main(["evaluate"]) == 0
    assert app_module.main(["retrain"]) == 0
    assert calls == ["evaluate", "retrain"]


def test_executed_module_entrypoint_handles_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("app", run_name="__main__")
    assert exc.value.code == 0
