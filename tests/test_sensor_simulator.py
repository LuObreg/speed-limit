"""Tests for sensor simulator."""

import time

import pandas as pd
import pytest

from llm_clients import LLMClientError
from sensor_simulator import SensorSimulator
from tests._support import wait_until

class TestSensorSimulator:
    @pytest.fixture()
    def table(self):
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
                             "temperature_c": [1, 2, 3, 4], "water_level_um": [10, 20, 30, 40],
                             "illuminance_millilux": [100, 200, 300, 400]})

    def test_loops_and_keeps_original_timestamps(self, table):
        got = []
        sim = SensorSimulator(table, 0.01, got.append, loop=True)
        sim.start()
        assert wait_until(lambda: len(got) >= 9)
        sim.stop(); sim.join(1)
        assert [r["row_index"] for r in got[:5]] == [0, 1, 2, 3, 0]
        assert got[4]["cycle"] == 1 and got[4]["timestamp"] == got[0]["timestamp"]
        assert not sim.is_running

    def test_without_loop_it_finishes(self, table):
        got = []
        sim = SensorSimulator(table, 0.01, got.append, loop=False)
        sim.start()
        assert wait_until(sim.finished.is_set) and len(got) == 4

    def test_pause_and_resume(self, table):
        got = []
        sim = SensorSimulator(table, 0.01, got.append, loop=True)
        sim.start()
        assert wait_until(lambda: len(got) >= 3)
        sim.pause(); time.sleep(0.05); n = len(got); time.sleep(0.1)
        assert len(got) <= n + 1 and sim.is_paused                            # at most the tick in flight
        sim.resume()
        assert wait_until(lambda: len(got) > n + 3)
        sim.stop(); sim.join(1)

    def test_failing_callback_does_not_stop_the_feed(self, table):
        calls = []

        def boom(reading):
            calls.append(reading)
            raise RuntimeError("consumer bug")
        sim = SensorSimulator(table, 0.01, boom, loop=True)
        sim.start()
        assert wait_until(lambda: len(calls) >= 6)
        sim.stop(); sim.join(1)

    def test_strict_callback_failure_stops_the_feed(self, table):
        calls = []

        def boom(reading):
            calls.append(reading)
            raise LLMClientError("LLM unavailable")

        sim = SensorSimulator(table, 0.01, boom, loop=True, stop_on_callback_error=True)
        sim.start()
        assert wait_until(sim.finished.is_set)
        sim.join(1)
        assert len(calls) == 1
        assert isinstance(sim.callback_error, LLMClientError)

    def test_invalid_tables_are_rejected(self, table):
        with pytest.raises(ValueError):
            SensorSimulator(table.iloc[0:0], 1, print)
        with pytest.raises(ValueError):
            SensorSimulator(table.drop(columns=["temperature_c"]), 1, print)
        with pytest.raises(ValueError):
            SensorSimulator(table, 0, print)

    def test_clock_property_and_running_start_are_safe(self, table):
        sim = SensorSimulator(table, 0.01, lambda _: None, loop=False)
        assert sim.seconds_per_sim_hour == 0.01
        sim.start()
        thread = sim._thread
        sim.start()
        assert sim._thread is thread
        sim.stop()
        sim.join(1)

    @pytest.mark.parametrize("interval", [0, -0.0, -1, float("nan"), float("inf"), float("-inf")])
    def test_invalid_clock_constructor_is_rejected(self, table, interval):
        with pytest.raises(ValueError, match="seconds_per_sim_hour"):
            SensorSimulator(table, interval, lambda _: None)

    @pytest.mark.parametrize("interval", [5e-324, 0.01, 1, 1e308])
    def test_positive_finite_clock_intervals_are_accepted(self, table, interval):
        sim = SensorSimulator(table, interval, lambda _: None)
        assert sim.seconds_per_sim_hour == interval

    def test_observation_properties_and_join_before_start(self, table):
        sim = SensorSimulator(table, 1, lambda _: None)
        assert sim.total_rows == 4 and sim.position == 0 and sim.cycle == 0
        assert not sim.is_running
        sim.join()

    def test_slow_callback_resets_schedule_and_stops_cleanly(self, table):
        calls = []

        def slow_callback(reading):
            calls.append(reading)
            time.sleep(0.02)

        sim = SensorSimulator(table, 0.001, slow_callback, loop=True)
        sim.start()
        assert wait_until(lambda: len(calls) >= 2)
        sim.stop()
        sim.join(1)
        assert not sim.is_running

    def test_stopping_while_paused_releases_the_waiting_thread(self, table):
        sim = SensorSimulator(table, 0.01, lambda _: None)
        sim.pause()
        sim.start()
        sim.stop()
        sim.join(1)
        assert not sim.is_running
