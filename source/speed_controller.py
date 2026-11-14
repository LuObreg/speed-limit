"""Type A / Software 1.0 decision policy, coordinating REQ1-REQ3 and HOTL overrides."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from air_quality_llm import POLLUTION_INDEX_KEY
from data_pipeline import ILLUMINANCE_MILLILUX, TEMPERATURE_C, TIMESTAMP, WATER_LEVEL_UM
from settings import AirQualityRules, SpeedRules

if TYPE_CHECKING:  # type hints only - keeps this Type A module free of Type T imports
    from air_quality_api import AirQualityAPIClient
    from air_quality_llm import AirQualityLLMEvaluator
    from data_pipeline import FeatureEngineer
    from guardrails import InputGuardrail, OutputGuardrail
    from nn_model import SpeedLimitNN
    from weather_router import WeatherRouter

LOG = logging.getLogger("speedlimit.controller")

DECIDED_NN, DECIDED_LLM, DECIDED_DEFAULT, DECIDED_MANUAL = "NN", "LLM", "DEFAULT", "MANUAL_OVERRIDE"
DECIDED_OPERATOR, AWAITING_OPERATOR = "OPERATOR", "AWAITING_OPERATOR"


@dataclass(frozen=True)
class Decision:
    """Everything the operator dashboard needs about one hourly update."""
    timestamp: Any                         # simulated time of the reading (as delivered)
    origin: str                            # "simulator", "scenario", or "manual_test"
    reading: dict[str, Any]                # reading after the input guardrail
    input_violations: tuple[str, ...]
    weather_triggered: bool | None         # REQ1 router result (None = could not assess)
    nn_speed_kmh: int | None
    aq_metrics: dict[str, Any] | None
    aq_high: bool | None
    llm_reduction_kmh: int | None
    llm_source: str | None                 # set only when the LLM returned a valid answer
    llm_speed_kmh: int | None
    automatic_speed_kmh: int | None        # what the system would publish on its own (None = operator needed)
    speed_kmh: int | None                  # what is actually published (None = awaiting operator)
    decided_by: str                        # NN | LLM | NN+LLM | DEFAULT | OPERATOR | AWAITING_OPERATOR | MANUAL_OVERRIDE
    override_active: bool
    degraded: bool                         # a component failed or input was invalid
    log: tuple[str, ...] = field(default_factory=tuple)
    nn_risk_estimates: dict[int, float] = field(default_factory=dict)
    scenario_id: int | None = None
    nn_target_met: bool | None = None      # None when the NN made no proposal

    def __str__(self) -> str:
        flags = (" [OVERRIDE]" if self.override_active else "") + (" [OPERATOR NEEDED]" if self.degraded else "")
        speed = "awaiting operator" if self.speed_kmh is None else f"{self.speed_kmh} km/h"
        return f"{self.timestamp} -> {speed} by {self.decided_by}{flags}"


class ControllerState(Protocol):
    def get_override(self) -> int | None: ...

    def set_override(self, speed_kmh: int | None) -> None: ...

    def get_operator_speed(self) -> int | None: ...

    def set_operator_speed(self, speed_kmh: int | None) -> None: ...


class SpeedLimitController:
    """Type A / Software 1.0 - orchestrates REQ1, REQ2, REQ3, guardrails and override."""

    def __init__(
        self,
        router: "WeatherRouter",
        nn: "SpeedLimitNN",
        feature_engineer: "FeatureEngineer",
        aq_client: "AirQualityAPIClient",
        aq_llm: "AirQualityLLMEvaluator",
        input_guardrail: "InputGuardrail",
        output_guardrail: "OutputGuardrail",
        speed_rules: SpeedRules,
        air_quality_rules: AirQualityRules,
        risk_target_per_hour: float,
        state: "ControllerState",
    ) -> None:
        self._router, self._nn, self._features = router, nn, feature_engineer
        self._aq_client, self._aq_llm = aq_client, aq_llm
        self._input, self._output = input_guardrail, output_guardrail
        self._speed, self._aq_rules = speed_rules, air_quality_rules
        self._risk_target = risk_target_per_hour
        self._state = state
        self._listeners: list[Callable[[Decision], None]] = []
        self._process_lock = threading.Lock()

    def set_manual_override(self, speed_kmh: int | None) -> int | None:
        """Set (or clear with None) the operator override. Returns the effective value,
        which is clamped into [min_kmh, max_kmh] by the OutputGuardrail."""
        effective = None if speed_kmh is None else self._output.enforce(speed_kmh).speed_kmh
        self._state.set_override(effective)
        LOG.warning("Manual override %s", f"SET to {effective} km/h" if effective is not None else "CLEARED")
        return effective

    def set_operator_speed(self, speed_kmh: int | None) -> int | None:
        """Operator's answer while the system cannot decide; it is cleared automatically
        by the next committed decision the system can make on its own."""
        effective = None if speed_kmh is None else self._output.enforce(speed_kmh).speed_kmh
        self._state.set_operator_speed(effective)
        LOG.warning("Operator decision %s", f"SET to {effective} km/h" if effective is not None else "CLEARED")
        return effective

    def add_listener(self, callback: Callable[[Decision], None]) -> None:
        """Called after every decision (app.py registers the telemetry queue here)."""
        self._listeners.append(callback)

    def process(self, reading: Mapping[str, Any], origin: str = "simulator",
                air_quality: Mapping[str, Any] | None = None, *,
                scenario_id: int | None = None, commit_state: bool = True) -> Decision:
        """Serialize decisions so concurrent callers cannot interleave state updates."""
        with self._process_lock:
            return self._process(reading, origin, air_quality, scenario_id, commit_state)

    def _process(self, reading: Mapping[str, Any], origin: str,
                 air_quality: Mapping[str, Any] | None, scenario_id: int | None,
                 commit_state: bool) -> Decision:
        """Evaluate one sensor reading. A non-simulator `origin` marks dashboard scenario
        readings; `air_quality` replaces the live API fetch for such scenarios, and
        `commit_state=False` keeps them from clearing the operator's speed."""
        log: list[str] = []
        degraded = False

        # Input guardrail
        check = self._input.validate(reading)
        filtered = check.reading
        for v in check.violations:
            log.append(f"INPUT GUARDRAIL: {v}")
        degraded |= not check.is_valid

        # REQ1
        weather_triggered: bool | None = None
        nn_speed: int | None = None
        nn_risks: dict[int, float] = {}
        nn_target_met: bool | None = None
        nn_ok = True
        weather = (filtered.get(ILLUMINANCE_MILLILUX), filtered.get(WATER_LEVEL_UM),
                   filtered.get(TEMPERATURE_C))
        if all(v is not None for v in weather):
            weather_triggered = self._router.should_invoke_nn(*weather)
            if weather_triggered:
                nn_speed, nn_risks, nn_ok = self._run_nn(filtered, log)
                if nn_speed is not None:
                    nn_target_met = nn_risks[nn_speed] <= self._risk_target
                degraded |= not nn_ok
            else:
                log.append("REQ1: not dark, no black-ice danger -> NN not needed")
        else:
            log.append("REQ1: SKIPPED - weather inputs invalid (cannot assess darkness/black ice)")

        # REQ2
        try:
            aq, aq_high, reduction, llm_source, llm_speed, aq_ok = self._run_air_quality(log, air_quality)
        except (RuntimeError, ValueError, KeyError) as exc:
            LOG.error("Air-quality decision failed: %s", exc)
            log.append(f"REQ2: FAILED ({exc})")
            aq, aq_high, reduction, llm_source, llm_speed, aq_ok = None, None, None, None, None, False
        degraded |= not aq_ok

        if degraded:
            automatic = self._state.get_operator_speed()
            if automatic is None:
                decided_by = AWAITING_OPERATOR
                log.append("OPERATOR DECISION REQUIRED: the system cannot decide safely on its own")
            else:
                decided_by = DECIDED_OPERATOR
                log.append(f"OPERATOR DECISION: {automatic} km/h while the system cannot decide")
        else:
            proposals = [(source, speed) for source, speed in
                         ((DECIDED_NN, nn_speed), (DECIDED_LLM, llm_speed)) if speed is not None]
            if proposals:
                lowest = min(speed for _, speed in proposals)
                decided_by = "+".join(source for source, speed in proposals if speed == lowest)
                automatic = lowest
            else:
                decided_by, automatic = DECIDED_DEFAULT, self._speed.default_kmh
                log.append(f"REQ3: no reduction required -> default {automatic} km/h")

        # Output guardrail
        if automatic is not None:
            guarded = self._output.enforce(automatic)
            if guarded.was_modified:
                log.append(f"OUTPUT GUARDRAIL: {guarded.requested} -> {guarded.speed_kmh} ({guarded.reason})")
            automatic = guarded.speed_kmh

        # HOTL override
        override = self._state.get_override()
        published, override_active = automatic, override is not None
        if override_active:
            published, decided_by = override, DECIDED_MANUAL
            log.append(f"MANUAL OVERRIDE: operator holds {override} km/h (automatic would be {automatic})")

        decision = Decision(
            timestamp=filtered.get(TIMESTAMP), origin=origin, reading=dict(filtered),
            input_violations=check.violations, weather_triggered=weather_triggered,
            nn_speed_kmh=nn_speed, aq_metrics=aq, aq_high=aq_high, llm_reduction_kmh=reduction,
            llm_source=llm_source, llm_speed_kmh=llm_speed, automatic_speed_kmh=automatic,
            speed_kmh=published, decided_by=decided_by, override_active=override_active,
            degraded=degraded, log=tuple(log), nn_risk_estimates=nn_risks,
            scenario_id=scenario_id, nn_target_met=nn_target_met,
        )
        if commit_state and not degraded:
            self._state.set_operator_speed(None)
        LOG.info("%s | %s", decision, " ; ".join(log))
        self._notify(decision)
        return decision

    def _run_nn(
        self, reading: Mapping[str, Any], log: list[str]
    ) -> tuple[int | None, dict[int, float], bool]:
        try:
            candidates = self._nn.candidate_speeds
            features = self._features.build_candidate_features(reading, candidates)
            estimates = self._nn.predict_risk(features)
            if len(estimates) != len(candidates) or not all(math.isfinite(float(v)) for v in estimates):
                raise ValueError("risk model returned invalid candidate estimates")
            risks = {speed: float(risk) for speed, risk in zip(candidates, estimates)}
            log.append("REQ1: NN estimated near-accidents/hour per speed: "
                       + ", ".join(f"{s} km/h = {r:.2f}" for s, r in risks.items()))
            reductions = {s: r for s, r in risks.items() if s < self._speed.default_kmh}
            if not reductions:
                log.append(f"REQ1: NN has no candidate speed below {self._speed.default_kmh} km/h")
                return None, risks, False
            eligible = [speed for speed, risk in reductions.items() if risk <= self._risk_target]
            if eligible:
                speed = max(eligible)
                log.append(f"REQ1: NN selects {speed} km/h (estimated risk {risks[speed]:.2f} <= target "
                           f"{self._risk_target:g} near-accidents/hour)")
                return speed, risks, True
            # Ties go to the lower speed because candidates are ordered ascending.
            speed = min(reductions, key=reductions.get)
            log.append(
                f"REQ1: TARGET NOT MET - no speed below {self._speed.default_kmh} km/h reaches "
                f"{self._risk_target:g} near-accidents/hour; NN selects the lowest estimated risk: "
                f"{speed} km/h ({risks[speed]:.2f})"
            )
            return speed, risks, True
        except (ValueError, RuntimeError, KeyError, AttributeError) as exc:
            LOG.error("NN failed: %s", exc)
            log.append(f"REQ1: NN FAILED ({exc})")
            return None, {}, False

    def _run_air_quality(self, log: list[str], supplied: Mapping[str, Any] | None = None):
        """Returns (metrics, is_high, reduction, llm_source, llm_speed, ok)."""
        metrics = self._aq_client.fetch() if supplied is None else dict(supplied)
        aqi = metrics[POLLUTION_INDEX_KEY]
        if isinstance(aqi, bool) or not isinstance(aqi, (int, float)) or not math.isfinite(aqi):
            raise ValueError(f"{POLLUTION_INDEX_KEY} is not a finite number: {aqi!r}")

        threshold = self._aq_rules.high_pollution_threshold
        if aqi < threshold:
            log.append(f"REQ2: AQI {aqi:g} < {threshold:g} -> pollution not high, LLM not needed")
            return metrics, False, None, None, None, True

        d = self._aq_llm.evaluate(metrics)
        reduction, source = d.reduction_kmh, d.source
        speed = self._speed.default_kmh - reduction
        log.append(f"REQ2: AQI {aqi:g} >= {threshold:g} -> LLM ({source}) reduces by "
                   f"{reduction} km/h -> {speed} km/h")
        return metrics, True, reduction, source, speed, True

    def _notify(self, decision: Decision) -> None:
        for callback in list(self._listeners):
            try:
                callback(decision)
            except Exception:  # noqa: BLE001 - a broken observer must not stop the system
                LOG.exception("decision listener failed")
