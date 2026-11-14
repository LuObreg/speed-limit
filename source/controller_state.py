"""Type T thread-safe storage for controller state shared with the dashboard."""

from __future__ import annotations

import threading


class ThreadSafeControllerState:
    """Protect the operator's override and fallback speed across UI/simulator threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._override: int | None = None
        self._operator_speed: int | None = None

    def get_override(self) -> int | None:
        with self._lock:
            return self._override

    def set_override(self, speed_kmh: int | None) -> None:
        with self._lock:
            self._override = speed_kmh

    def get_operator_speed(self) -> int | None:
        with self._lock:
            return self._operator_speed

    def set_operator_speed(self, speed_kmh: int | None) -> None:
        with self._lock:
            self._operator_speed = speed_kmh
