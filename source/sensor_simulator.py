"""Type T / Software 1.0 replay of hourly readings on a drift-free clock.

The simulator owns timing and delivery only; control decisions stay in Type A.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

import pandas as pd

from data_pipeline import ILLUMINANCE_MILLILUX, TEMPERATURE_C, TIMESTAMP, WATER_LEVEL_UM

LOG = logging.getLogger("speedlimit.simulator")

_REQUIRED = (TIMESTAMP, TEMPERATURE_C, WATER_LEVEL_UM, ILLUMINANCE_MILLILUX)


class SensorSimulator:
    """Type T / Software 1.0 - replays hourly sensor readings on an accelerated clock."""

    def __init__(
        self,
        readings: pd.DataFrame,
        seconds_per_sim_hour: float,
        on_reading: Callable[[dict[str, Any]], Any],
        loop: bool = True,
        stop_on_callback_error: bool = False,
    ) -> None:
        missing = [c for c in _REQUIRED if c not in readings.columns]
        if missing:
            raise ValueError(f"readings lack columns {missing}")
        if readings.empty:
            raise ValueError("readings are empty - nothing to simulate")
        if not math.isfinite(seconds_per_sim_hour) or seconds_per_sim_hour <= 0:
            raise ValueError("seconds_per_sim_hour must be finite and > 0")

        self._rows = readings.sort_values(TIMESTAMP)[list(_REQUIRED)].to_dict("records")
        self._interval = float(seconds_per_sim_hour)
        self._on_reading = on_reading
        self._loop = loop
        self._stop_on_callback_error = stop_on_callback_error
        self.callback_error: Exception | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._gate = threading.Event()          # set = running, cleared = paused
        self._gate.set()
        self._resync = False
        self.finished = threading.Event()       # set when a non-looping replay is complete
        self._position = 0
        self._cycle = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.finished.clear()
        self.callback_error = None
        self._thread = threading.Thread(target=self._run, name="SensorSimulator", daemon=True)
        self._thread.start()
        LOG.info("Simulator started: %d hourly rows, %.2f s per simulated hour, loop=%s",
                 len(self._rows), self._interval, self._loop)

    def stop(self) -> None:
        self._stop.set()
        self._gate.set()                         # release a paused thread so it can exit

    def stop_with_error(self, error: Exception) -> None:
        self.callback_error = error
        self.finished.set()
        self.stop()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def pause(self) -> None:
        self._gate.clear()
        LOG.info("Simulator paused")

    def resume(self) -> None:
        self._resync = True                      # restart the schedule from "now"
        self._gate.set()
        LOG.info("Simulator resumed")

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    @property
    def is_paused(self) -> bool:
        return not self._gate.is_set()

    @property
    def seconds_per_sim_hour(self) -> float:
        return self._interval

    @property
    def total_rows(self) -> int:
        return len(self._rows)

    @property
    def position(self) -> int:
        """Index of the NEXT row to be delivered (0 .. total_rows-1)."""
        return self._position

    @property
    def cycle(self) -> int:
        """Number of completed replays (0 during the first pass)."""
        return self._cycle

    def _run(self) -> None:
        next_due = time.monotonic()
        while not self._stop.is_set():
            self._gate.wait()
            if self._stop.is_set():
                break
            if self._resync:
                self._resync, next_due = False, time.monotonic()

            self._emit(self._position)
            if self._stop.is_set():
                break
            self._position += 1
            if self._position >= len(self._rows):
                if not self._loop:
                    self.finished.set()
                    LOG.info("Replay finished (loop disabled)")
                    break
                self._position, self._cycle = 0, self._cycle + 1
                LOG.info("Replay restarted (cycle %d)", self._cycle)

            next_due += self._interval
            delay = next_due - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)          # wakes immediately on stop()
            else:
                next_due = time.monotonic()      # fell behind: do not burst to catch up

    def _emit(self, index: int) -> None:
        row = self._rows[index]
        reading = {
            TIMESTAMP: row[TIMESTAMP],
            TEMPERATURE_C: float(row[TEMPERATURE_C]),
            WATER_LEVEL_UM: float(row[WATER_LEVEL_UM]),
            ILLUMINANCE_MILLILUX: float(row[ILLUMINANCE_MILLILUX]),
            "row_index": index,
            "cycle": self._cycle,
        }
        try:
            self._on_reading(reading)
        except Exception as exc:  # noqa: BLE001 - callback policy is selected by the caller
            LOG.exception("on_reading callback failed for row %d", index)
            if self._stop_on_callback_error:
                self.stop_with_error(exc)
