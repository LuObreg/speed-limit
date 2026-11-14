"""Type A / Software 3.0 air-quality evaluator with a deterministic harness."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from llm_clients import LLMClient, LLMClientError
from settings import AirQualityRules

LOG = logging.getLogger("speedlimit.airquality.llm")

POLLUTION_INDEX_KEY = "european_aqi"
_RETRY_HINT = ("\n\nYour previous reply could not be used. Reply with ONLY a JSON object like "
               '{"speed_reduction_kmh": 15, "reason": "short text"} and nothing else.')


@dataclass(frozen=True)
class LLMDecision:
    """Result of one evaluation (for the dashboard's component log)."""
    reduction_kmh: int
    reason: str
    source: str
    attempts: int
    raw_response: str | None


class AirQualityLLMEvaluator:
    """Type A / Software 3.0 - LLM evaluator with a deterministic software harness."""

    def __init__(self, client: LLMClient, rules: AirQualityRules) -> None:
        self._client = client
        self._rules = rules

    def evaluate(self, metrics: Mapping[str, Any]) -> LLMDecision:
        """Ask for a reduction; the result is always inside [min_reduction, max_reduction]."""
        system, user = self.build_prompt(metrics)
        raw: str | None = None
        for attempt in (1, 2):
            raw = self._client.complete(system, user if attempt == 1 else user + _RETRY_HINT)
            parsed = self._parse(raw)
            if parsed is not None:
                value, reason = parsed
                clamped = self._clamp(value)
                if clamped != value:
                    LOG.warning("LLM answered %s km/h, outside range; clamped to %s", value, clamped)
                    reason = f"{reason} (clamped from {value})"
                return LLMDecision(clamped, reason, "llm", attempt, raw)
            LOG.warning("LLM reply not parseable (attempt %d): %r", attempt, (raw or "")[:120])
        raise LLMClientError(f"LLM reply was unusable after two attempts: {(raw or '')[:120]}")

    def build_prompt(self, metrics: Mapping[str, Any]) -> tuple[str, str]:
        r = self._rules
        system = (
            "You are a safety assistant for an automatic highway speed-limit system. "
            "Air pollution is currently high. Recommend by how many km/h the speed limit "
            f"should be reduced, as a whole number between {r.min_reduction_kmh} and "
            f"{r.max_reduction_kmh}. Stronger pollution means a larger reduction. "
            'Reply with ONLY a JSON object: {"speed_reduction_kmh": <integer>, "reason": "<max 15 words>"}'
        )
        units = metrics.get("units") or {}
        lines = [f"Current air quality near {metrics.get('location', 'the highway')} "
                 f"at {metrics.get('time', 'unknown time')}:"]
        for key, value in metrics.items():
            if key in ("units", "location", "time", "source", "attribution") or value is None:
                continue
            note = " (European AQI: 0-20 good, 20-40 fair, 40-60 moderate, 60-80 poor, " \
                   "80-100 very poor, >100 extremely poor)" if key == POLLUTION_INDEX_KEY else ""
            lines.append(f"- {key}: {value} {units.get(key, '') if key != POLLUTION_INDEX_KEY else ''}".rstrip()
                         + note)
        return system, "\n".join(lines)

    @staticmethod
    def _parse(raw: str | None) -> tuple[int, str] | None:
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                value = obj.get("speed_reduction_kmh")
                if isinstance(value, str):
                    value = float(value.strip().split()[0])
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value):
                    return None
                return int(round(value)), str(obj.get("reason", ""))[:120]
            except (json.JSONDecodeError, ValueError, AttributeError, IndexError):
                return None
        bare = re.fullmatch(r"\s*(\d{1,3})\s*(?:km/?h)?\s*\.?\s*", raw)   # reply was only "15"
        return (int(bare.group(1)), "") if bare else None

    def _clamp(self, value: int) -> int:
        return max(self._rules.min_reduction_kmh, min(self._rules.max_reduction_kmh, value))


@dataclass(frozen=True)
class GoldenCase:
    name: str
    metrics: dict[str, Any]
    min_kmh: int       # lowest acceptable answer
    max_kmh: int       # highest acceptable answer


def _payload(aqi: float, pm10: float, pm25: float, no2: float, o3: float) -> dict[str, Any]:
    return {"source": "golden", "location": "Vienna", "time": "2026-01-15T08:00",
            "units": {"pm10": "μg/m³", "pm2_5": "μg/m³", "nitrogen_dioxide": "μg/m³", "ozone": "μg/m³",
                      "sulphur_dioxide": "μg/m³", "carbon_monoxide": "μg/m³"},
            "european_aqi": aqi, "pm10": pm10, "pm2_5": pm25, "nitrogen_dioxide": no2, "ozone": o3,
            "sulphur_dioxide": 5.0, "carbon_monoxide": 300.0}


# Expected answer ranges for the default 10-20 km/h reduction.
GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase("just above threshold (AQI 62)",  _payload(62, 45, 28, 55, 100),   10, 13),
    GoldenCase("poor (AQI 72)",                  _payload(72, 58, 36, 70, 130),   11, 16),
    GoldenCase("very poor (AQI 95)",             _payload(95, 95, 62, 110, 170),  16, 20),
    GoldenCase("extremely poor (AQI 130)",       _payload(130, 150, 90, 160, 220), 18, 20),
)


@dataclass(frozen=True)
class GoldenResult:
    case: GoldenCase
    answer: int
    source: str
    passed: bool


class GoldenAnswerHarness:
    """Type A / Software 1.0 - checks the Software 3.0 evaluator against golden ranges."""

    def __init__(self, evaluator: AirQualityLLMEvaluator,
                 cases: tuple[GoldenCase, ...] = GOLDEN_CASES) -> None:
        self._evaluator, self._cases = evaluator, cases

    def run(self) -> list[GoldenResult]:
        results = []
        for case in self._cases:
            d = self._evaluator.evaluate(case.metrics)
            results.append(GoldenResult(case, d.reduction_kmh, d.source,
                                        case.min_kmh <= d.reduction_kmh <= case.max_kmh))
        return results

    @staticmethod
    def pass_rate(results: list[GoldenResult]) -> float:
        return sum(r.passed for r in results) / len(results) if results else 0.0

    @staticmethod
    def format_report(results: list[GoldenResult]) -> str:
        rows = [f"{'PASS' if r.passed else 'FAIL'}  {r.case.name:<32} answer={r.answer:>2} "
                f"expected {r.case.min_kmh}-{r.case.max_kmh} ({r.source})" for r in results]
        return "\n".join(rows)
