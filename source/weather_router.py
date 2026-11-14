"""Type A / Software 1.0 rules for deciding whether REQ1 invokes the neural network."""

from __future__ import annotations

import math

from settings import WeatherRules


class WeatherRouter:
    """Type A / Software 1.0 - decides whether REQ1 requires the neural network."""

    def __init__(self, rules: WeatherRules) -> None:
        self._rules = rules

    def is_dark(self, illuminance_millilux: float) -> bool:
        """SPEC1: illuminance strictly below the darkness threshold."""
        self._require_finite(illuminance_millilux=illuminance_millilux)
        return illuminance_millilux < self._rules.darkness_illuminance_below_millilux

    def is_black_ice_danger(self, water_level_um: float, temperature_c: float) -> bool:
        """SPEC2: water level strictly above AND temperature strictly below threshold."""
        self._require_finite(water_level_um=water_level_um, temperature_c=temperature_c)
        return (
            water_level_um > self._rules.black_ice_water_level_above_um
            and temperature_c < self._rules.black_ice_temperature_below_c
        )

    def should_invoke_nn(
        self,
        illuminance_millilux: float,
        water_level_um: float,
        temperature_c: float,
    ) -> bool:
        """
        True  -> trigger the neural network (SPEC1 OR SPEC2).
        False -> pass; the controller continues with REQ2 / REQ3.

        Inputs are expected to be already validated by the InputGuardrail.
        Non-finite values (None/NaN/inf) are programming errors upstream and
        raise ValueError instead of silently evaluating to False.
        """
        return self.is_dark(illuminance_millilux) or self.is_black_ice_danger(
            water_level_um, temperature_c
        )

    @staticmethod
    def _require_finite(**values: float) -> None:
        for name, value in values.items():
            if value is None or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")