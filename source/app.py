"""Composition root for configuration, dependency injection, and runtime."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import queue
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from settings import ConfigError, Settings, load_settings, validate_settings

LOG = logging.getLogger("speedlimit.app")


def _runtime_model_dir(model_dir: Path) -> Path:
    from model_artifacts import MODEL_FILE

    annual = model_dir / "annual"
    return annual if (annual / MODEL_FILE).is_file() else model_dir / "year_n"


def configure_logging(log_dir: Path, verbose: bool = False) -> None:
    """Console + file logging. Component execution logs (NN vs. LLM vs. Default)
    are emitted by the controller via `logging`."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "speedlimit.log", encoding="utf-8"),
        ],
    )


@dataclass
class Application:
    """Plain holder of wired-up components."""
    settings: Settings
    controller: object
    simulator: object
    telemetry_queue: "queue.Queue" = field(default_factory=queue.Queue)


class ApplicationFactory:
    """
    Dependency-injection / object wiring only.

    Imports are lazy so `train` mode does not need the GUI/LLM modules and a
    missing module yields a clear error message.
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def _clean_sensors(self, *, year_plus_one: bool = False) -> pd.DataFrame:
        from data_io import CSVDataReader

        paths = self._s.technical.paths
        file = paths.sensor_readings_year_plus_one_file if year_plus_one else paths.sensor_readings_file
        raw = CSVDataReader().read_sensor_readings(paths.data_dir / file, headerless=year_plus_one)
        return self._cleaner().clean_sensor_readings(raw)

    def _training_set(self, *, year_plus_one: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        """Clean one period and join its labels within that period."""
        from data_io import CSVDataReader

        paths = self._s.technical.paths
        file = paths.accidents_year_plus_one_file if year_plus_one else paths.accidents_file
        accidents = self._cleaner().clean_accidents(
            CSVDataReader().read_accidents(paths.data_dir / file))
        sensors = self._clean_sensors(year_plus_one=year_plus_one)
        return self._feature_engineer().build_training_set(sensors, accidents)

    def _cleaner(self):
        from data_pipeline import DataCleaner

        business = self._s.business
        return DataCleaner(business.sensor_bounds,
                           temperature_units_by_sensor=business.data.temperature_units_by_sensor)

    def _feature_engineer(self):
        from data_pipeline import FeatureEngineer

        biz = self._s.business
        return FeatureEngineer(biz.weather, biz.sensor_bounds, biz.danger_index,
                               biz.data.near_accident_types)

    def _model(self, directory: Path):
        from model_artifacts import FilesystemModelArtifactStore
        from nn_model import SpeedLimitNN

        biz = self._s.business
        return SpeedLimitNN(FilesystemModelArtifactStore(directory), biz.weather, biz.speed,
                            self._s.technical.nn)

    def run_training(self) -> None:
        """Train on Year N and reserve chronological validation/test periods."""
        model_dir = self._s.technical.paths.model_dir / "year_n"
        LOG.info("Training pipeline started")
        report = self._model(model_dir).train(*self._training_set())
        LOG.info("Training report: %s", json.dumps(dataclasses.asdict(report), indent=2))
        LOG.info("Training pipeline finished; weights in %s", model_dir)

    def evaluate_year_plus_one(self) -> dict[str, Any]:
        """Evaluate the frozen Year N model on untouched Year N+1 observations."""
        from model_artifacts import FilesystemModelArtifactStore

        model_dir = self._s.technical.paths.model_dir
        model = self._model(model_dir / "year_n")
        model.load()
        report = dataclasses.asdict(model.evaluate(*self._training_set(year_plus_one=True)))
        report["evaluation_period"] = "Year N+1; model trained on Year N"
        report["causal_safety_claim"] = False
        report["limitation"] = (
            "Observational historical data estimates association, not the counterfactual "
            "effect of applying a candidate speed."
        )
        report["uncertainty_limitation"] = (
            "The percentile bootstrap resamples individual triggered hours and does not "
            "account for temporal autocorrelation."
        )
        report_path = FilesystemModelArtifactStore(model_dir).save_report(
            "year_plus_one_evaluation.json", report
        )
        LOG.info("Out-of-time evaluation written to %s:\n%s",
                 report_path, json.dumps(report, indent=2))
        return report

    def run_annual_retraining(self) -> None:
        """Evaluate Year N+1 first, then retrain on both supplied periods."""
        self.evaluate_year_plus_one()
        features_n, target_n = self._training_set()
        features_n1, target_n1 = self._training_set(year_plus_one=True)
        features = pd.concat((features_n, features_n1), ignore_index=True)
        target = pd.concat((target_n, target_n1), ignore_index=True)
        report = self._model(self._s.technical.paths.model_dir / "annual").train(features, target)
        LOG.info("Annual retraining report: %s", json.dumps(dataclasses.asdict(report), indent=2))

    def build_runtime(self) -> Application:
        """Instantiate and connect all runtime components."""
        from air_quality_api import AirQualityAPIClient
        from air_quality_llm import AirQualityLLMEvaluator
        from llm_clients import OllamaClient
        from controller_state import ThreadSafeControllerState
        from guardrails import InputGuardrail, OutputGuardrail
        from sensor_simulator import SensorSimulator
        from speed_controller import SpeedLimitController
        from weather_router import WeatherRouter

        tech, biz = self._s.technical, self._s.business

        nn = self._model(_runtime_model_dir(tech.paths.model_dir))
        nn.load()

        controller = SpeedLimitController(
            router=WeatherRouter(biz.weather),
            nn=nn,
            feature_engineer=self._feature_engineer(),
            aq_client=AirQualityAPIClient(tech.air_quality_api),
            aq_llm=AirQualityLLMEvaluator(
                OllamaClient(tech.llm.ollama_url, tech.llm.ollama_model, tech.llm.timeout_s),
                biz.air_quality),
            input_guardrail=InputGuardrail(biz.sensor_bounds),
            output_guardrail=OutputGuardrail(biz.speed),
            speed_rules=biz.speed,
            air_quality_rules=biz.air_quality,
            risk_target_per_hour=biz.weather.target_near_accidents_per_hour,
            state=ThreadSafeControllerState(),
        )

        # Bridge simulator callbacks to Tk's UI thread.
        telemetry_queue: "queue.Queue" = queue.Queue()
        controller.add_listener(telemetry_queue.put)

        simulator = SensorSimulator(
            readings=self._clean_sensors(),
            seconds_per_sim_hour=tech.simulator.seconds_per_sim_hour,
            on_reading=controller.process,
            stop_on_callback_error=True,
        )
        return Application(self._s, controller, simulator, telemetry_queue)


def _stop_simulation(app: Application) -> None:
    """Shared shutdown: stop the replay and surface a failed reading as an error."""
    app.simulator.stop()
    app.simulator.join()
    LOG.info("Shutdown complete")
    if app.simulator.callback_error is not None:
        raise RuntimeError("simulation stopped because a reading could not be evaluated") \
            from app.simulator.callback_error


def run_headless(app: Application) -> None:
    """No UI: let the simulator run and log decisions until it is stopped."""
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    app.simulator.start()
    LOG.info("Headless mode running. Press Ctrl+C to stop.")
    try:
        while not stop.is_set():
            try:
                LOG.info("DECISION %s", app.telemetry_queue.get(timeout=0.5))
            except queue.Empty:
                if app.simulator.finished.is_set():   # replay finished or stopped on error
                    break
    finally:
        _stop_simulation(app)


def run_gui(app: Application) -> None:
    """Start the simulator, then hand the main thread to the dashboard."""
    from gui_app import OperatorDashboard

    speed = app.settings.business.speed
    dashboard = OperatorDashboard(
        controller=app.controller,
        simulator=app.simulator,
        telemetry_queue=app.telemetry_queue,
        speed_range=(speed.min_kmh, speed.max_kmh),
        default_kmh=speed.default_kmh,
    )
    app.simulator.start()
    try:
        dashboard.run()  # blocks until the operator closes the window
    finally:
        _stop_simulation(app)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpeedLimit Highway MVP")
    p.add_argument(
        "command",
        choices=["train", "evaluate", "retrain", "run"],
        help="train on Year N, evaluate Year N+1, retrain annually, or run the system",
    )
    p.add_argument("--config", type=Path, default=Path("configuration.toml"))
    p.add_argument("--ui", choices=["gui", "headless"])
    p.add_argument("--seconds-per-hour", type=float)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """CLI flags override technical settings only; business rules stay in the file."""
    tech = settings.technical
    runtime, sim = tech.runtime, tech.simulator
    if args.ui:
        runtime = dataclasses.replace(runtime, ui=args.ui)
    if args.seconds_per_hour is not None:
        sim = dataclasses.replace(sim, seconds_per_sim_hour=args.seconds_per_hour)
    tech = dataclasses.replace(tech, runtime=runtime, simulator=sim)
    overridden = dataclasses.replace(settings, technical=tech)
    validate_settings(overridden)
    return overridden


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = apply_overrides(load_settings(args.config), args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(settings.technical.paths.log_dir, args.verbose)
    factory = ApplicationFactory(settings)

    try:
        if args.command == "train":
            factory.run_training()
            return 0
        if args.command == "evaluate":
            factory.evaluate_year_plus_one()
            return 0
        if args.command == "retrain":
            factory.run_annual_retraining()
            return 0
        app = factory.build_runtime()
        (run_gui if settings.technical.runtime.ui == "gui" else run_headless)(app)
        return 0
    except ConfigError as exc:
        LOG.error("Configuration error: %s", exc)
        return 1
    except ModuleNotFoundError as exc:
        LOG.error("Missing Python module: %s (run `pip install -r requirements.txt`)", exc.name)
        return 2
    except FileNotFoundError as exc:
        LOG.error("Missing file: %s (did you run `python -m app train`?)", exc)
        return 3
    except RuntimeError as exc:
        LOG.error("%s: %s", exc, exc.__cause__ or exc)
        return 4


if __name__ == "__main__":
    sys.exit(main())
