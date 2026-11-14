"""Shared fixtures and small fakes for the module-level tests."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import settings as cfg
from air_quality_api import AirQualityAPIClient, AirQualityAPIError
from air_quality_llm import AirQualityLLMEvaluator
from controller_state import ThreadSafeControllerState
from data_pipeline import DataCleaner, FeatureEngineer
from guardrails import InputGuardrail, OutputGuardrail
from llm_clients import LLMClientError
from speed_controller import SpeedLimitController
from weather_router import WeatherRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configuration.toml"


@pytest.fixture(scope="module")
def S() -> cfg.Settings:
    return cfg.load_settings(CONFIG_PATH)


@pytest.fixture(scope="module")
def spec_weather() -> cfg.WeatherRules:
    """Specification thresholds, independent of the loaded configuration."""
    return cfg.WeatherRules(
        darkness_illuminance_below_millilux=500.0,
        black_ice_water_level_above_um=1000.0,
        black_ice_temperature_below_c=0.0,
        target_near_accidents_per_hour=1.0,
    )


class FakeNN:
    def __init__(self, value):
        self.value, self.calls = value, 0
        self.candidate_speeds = (40, 50, 60, 70, 80)

    def predict_risk(self, features):
        assert isinstance(features, pd.DataFrame) and len(features) == len(self.candidate_speeds)
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        if self.value == "none":
            return np.full(len(self.candidate_speeds), 1.5)
        return np.array([
            0.5 if speed <= self.value else 1.5 for speed in self.candidate_speeds
        ])


class FakeAQ:
    def __init__(self, aqi):
        self.aqi = aqi

    def fetch(self):
        if isinstance(self.aqi, Exception):
            raise self.aqi
        return {
            "source": "fake",
            "location": "Vienna",
            "time": "t",
            "units": {},
            "european_aqi": self.aqi,
            "pm10": 50.0,
        }


class ScriptedClient:
    """LLM transport that returns canned replies (the last one repeats)."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, system, user):
        self.calls += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def wait_until(condition, timeout=3.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(step)
    return condition()


SAFE = {
    "illuminance_millilux": 5e7,
    "water_level_um": 0.0,
    "temperature_c": 20.0,
}


def route(router, **overrides):
    return router.should_invoke_nn(**{**SAFE, **overrides})


DAY = {
    "timestamp": pd.Timestamp("2026-01-15 12:00"),
    "temperature_c": 10.0,
    "water_level_um": 0.0,
    "illuminance_millilux": 5e7,
}
NIGHT = {**DAY, "illuminance_millilux": 100.0}


@pytest.fixture()
def make_controller(S):
    business = S.business

    def build(nn=60, aqi=20.0, llm_client=None):
        evaluator = AirQualityLLMEvaluator(
            llm_client or ScriptedClient('{"speed_reduction_kmh": 13, "reason": "test response"}'),
            business.air_quality,
        )
        nn_fake = FakeNN(nn)
        controller = SpeedLimitController(
            WeatherRouter(business.weather),
            nn_fake,
            FeatureEngineer(business.weather, business.sensor_bounds, business.danger_index),
            FakeAQ(aqi),
            evaluator,
            InputGuardrail(business.sensor_bounds),
            OutputGuardrail(business.speed),
            business.speed,
            business.air_quality,
            business.weather.target_near_accidents_per_hour,
            ThreadSafeControllerState(),
        )
        controller.nn_fake = nn_fake
        return controller

    return build


_LONG_SENSOR_IDS = {"temperature_c": 15, "water_level_um": 1, "illuminance_millilux": 2,
                    "speed_limit_kmh": -1}


def long_readings(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly test table -> case-study rows (Month, Day, Hour, Minute, Second, Sensor, Value)."""
    rows = []
    for _, row in hourly.iterrows():
        ts = row["timestamp"]
        for column, sensor in _LONG_SENSOR_IDS.items():
            if column in row:
                value = row[column]
                if column == "illuminance_millilux":
                    value = value / 1000           # the light sensors report lux
                rows.append({"Month": ts.month, "Day": ts.day, "Hour": ts.hour, "Minute": 0,
                             "Second": 0, "Sensor": sensor, "Value": value})
    return pd.DataFrame(rows)


@pytest.fixture()
def raw_sensors():
    timestamps = pd.date_range("2000-01-01", periods=48, freq="h")
    hourly = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": [-2 + 0.1 * index for index in range(48)],
        "water_level_um": 1500.0,
        "illuminance_millilux": [300.0 if ts.hour < 7 else 50_000.0 for ts in timestamps],
        "speed_limit_kmh": 60,
    })
    rows = long_readings(hourly).astype({"Value": object, "Day": object})
    hour = rows["Day"].sub(1).mul(24).add(rows["Hour"])
    rows.loc[(hour == 5) & (rows["Sensor"] == 15), "Value"] = 999            # impossible
    rows.loc[hour.between(10, 12) & (rows["Sensor"] == 1), "Value"] = "oops"  # not numeric
    rows.loc[hour == 20, "Day"] = "garbage"                                  # unparseable time
    no_sensor = {**rows.iloc[0].to_dict(), "Sensor": None, "Value": 50}
    return pd.concat([rows, rows.iloc[[3]], pd.DataFrame([no_sensor])], ignore_index=True)


@pytest.fixture()
def raw_accidents():
    def event(hour, second=0, **attributes):
        return {"Month": 1, "Day": 1 + hour // 24, "Hour": hour % 24, "Second": second,
                "LicencePlate": "W 123AB", **attributes}

    return pd.DataFrame([
        event(2, 10, SkidAngle=40),
        event(2, 40, CloseCarCm=15),
        event(9, Damage=1000),
        {**event(5, SkidAngle=10), "Day": None},                             # unparseable time
        event(30, CloseGuardrailCm=20),
    ])


@pytest.fixture()
def fake_air_quality_server(S):
    state = {"mode": "ok", "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            state["hits"] += 1
            mode = state["mode"]
            if mode == "ok":
                code, body = 200, {
                    "current_units": {"pm10": "μg/m³"},
                    "current": {
                        "time": "2026-01-15T14:00",
                        "european_aqi": 42,
                        "pm10": 20.1,
                    },
                }
            elif mode == "400":
                code, body = 400, {"error": True, "reason": "bad variable"}
            else:
                code, body = 200, {"current": {"pm10": 1}}
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = dataclasses.replace(
        S.technical.air_quality_api,
        timeout_s=2.0,
        base_url=f"http://127.0.0.1:{server.server_port}/v1/air-quality",
    )
    yield api, state
    server.shutdown()
    server.server_close()
