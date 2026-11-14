"""Type A / Software 1.0 input and output validation policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from data_pipeline import physical_bounds
from settings import SensorBounds, SpeedRules


@dataclass(frozen=True)
class InputCheckResult:
    """Outcome of the input filter.

    reading     copy of the input where every invalid field is replaced by None
    violations  human-readable reasons (for the dashboard / component log)
    """
    is_valid: bool
    reading: dict[str, Any]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class OutputCheckResult:
    """Outcome of the output sanity check."""
    requested: Any
    speed_kmh: int
    was_modified: bool
    reason: str


class InputGuardrail:
    """Type A / Software 1.0 - filters sensor readings against physical bounds."""

    def __init__(self, bounds: SensorBounds) -> None:
        self._bounds = physical_bounds(bounds)

    def validate(self, reading: Mapping[str, Any]) -> InputCheckResult:
        """Check every guarded field; invalid/missing ones are set to None."""
        filtered = dict(reading)
        violations: list[str] = []

        for field, (low, high) in self._bounds.items():
            value = reading.get(field)
            problem = self._check_value(field, value, low, high)
            if problem:
                violations.append(problem)
                filtered[field] = None

        return InputCheckResult(not violations, filtered, tuple(violations))

    @staticmethod
    def _check_value(field: str, value: Any, low: float, high: float) -> str | None:
        if value is None:
            return f"{field}: missing"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{field}: not numeric ({value!r})"
        if not math.isfinite(value):
            return f"{field}: not finite ({value!r})"
        if not low <= value <= high:
            return f"{field}: {value} outside physical range [{low}, {high}]"
        return None


class OutputGuardrail:
    """Type A / Software 1.0 - guarantees min_kmh <= published speed <= max_kmh."""

    def __init__(self, rules: SpeedRules) -> None:
        self._rules = rules

    def enforce(self, requested_kmh: Any) -> OutputCheckResult:
        """
        Return a safe integer speed limit.

        - out-of-range numbers are clamped into [min_kmh, max_kmh]
        - floats are rounded to the nearest integer
        - None / NaN / non-numeric values FAIL SAFE to min_kmh (the most
          conservative limit), because an unknown model output must never
          raise the limit.
        """
        lo, hi = self._rules.min_kmh, self._rules.max_kmh

        if (
            requested_kmh is None
            or isinstance(requested_kmh, bool)
            or not isinstance(requested_kmh, (int, float))
            or not math.isfinite(requested_kmh)
        ):
            return OutputCheckResult(requested_kmh, lo, True,
                                     f"invalid value -> fail-safe {lo} km/h")

        value = int(round(requested_kmh))
        if value < lo:
            return OutputCheckResult(requested_kmh, lo, True, f"{value} below minimum {lo}")
        if value > hi:
            return OutputCheckResult(requested_kmh, hi, True, f"{value} above maximum {hi}")
        return OutputCheckResult(requested_kmh, value, value != requested_kmh,
                                 "ok" if value == requested_kmh else "rounded")
