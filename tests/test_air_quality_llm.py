"""Tests for air quality llm."""

import pytest

from tests._support import ScriptedClient
from air_quality_llm import GOLDEN_CASES, AirQualityLLMEvaluator, GoldenAnswerHarness
from llm_clients import LLMClientError


class TestAirQualityLLMHarness:
    CASE = GOLDEN_CASES[1].metrics

    @pytest.mark.parametrize("reply,expected", [
        ('{"speed_reduction_kmh": 14, "reason": "poor air"}', 14),
        ('Sure!\n```json\n{"speed_reduction_kmh": "15 km/h"}\n```', 15),
        ("16", 16),
        ('{"speed_reduction_kmh": 13.6}', 14),
        ('{"speed_reduction_kmh": 50}', 20),         # clamped to max
        ('{"speed_reduction_kmh": 1}', 10),          # clamped to min
    ])
    def test_parsing_and_range_check(self, S, reply, expected):
        d = AirQualityLLMEvaluator(ScriptedClient(reply), S.business.air_quality).evaluate(self.CASE)
        assert d.reduction_kmh == expected and d.source == "llm"

    def test_unparseable_reply_is_retried_once(self, S):
        client = ScriptedClient("I would say about fifteen", '{"speed_reduction_kmh": 12}')
        d = AirQualityLLMEvaluator(client, S.business.air_quality).evaluate(self.CASE)
        assert (d.reduction_kmh, d.source, client.calls) == (12, "llm", 2)

    @pytest.mark.parametrize("reply", ["no idea", '{"speed_reduction_kmh": true}'])
    def test_unusable_llm_reply_raises_instead_of_using_a_rule(self, S, reply):
        client = ScriptedClient(reply)
        with pytest.raises(LLMClientError, match="unusable after two attempts"):
            AirQualityLLMEvaluator(client, S.business.air_quality).evaluate(self.CASE)
        assert client.calls == 2

    def test_llm_transport_error_propagates(self, S):
        client = ScriptedClient(LLMClientError("provider unavailable"))
        with pytest.raises(LLMClientError, match="provider unavailable"):
            AirQualityLLMEvaluator(client, S.business.air_quality).evaluate(self.CASE)

    def test_prompt_contains_readings_and_allowed_range(self, S):
        system, user = AirQualityLLMEvaluator(ScriptedClient("x"), S.business.air_quality).build_prompt(self.CASE)
        assert "10" in system and "20" in system and "JSON" in system
        assert "european_aqi: 72" in user and "pm10" in user

    @pytest.mark.parametrize("reply", [
        None, "", "{broken}", '{"speed_reduction_kmh": ""}',
        '{"speed_reduction_kmh": "bad"}', '{"speed_reduction_kmh": true}',
        '{"speed_reduction_kmh": Infinity}', '{"speed_reduction_kmh": []}',
        "15 km/h extra",
    ])
    def test_parser_rejects_malformed_values(self, reply):
        assert AirQualityLLMEvaluator._parse(reply) is None

    def test_prompt_handles_missing_location_units_and_optional_values(self, S):
        _, user = AirQualityLLMEvaluator(ScriptedClient("12"), S.business.air_quality).build_prompt(
            {"european_aqi": 60, "pm10": None, "pm2_5": 3, "units": None}
        )
        assert "near the highway" in user
        assert "unknown time" in user
        assert "- pm2_5: 3" in user
        assert "pm10" not in user

    def test_golden_harness_reports_pass_and_empty_result(self, S):
        evaluator = AirQualityLLMEvaluator(
            ScriptedClient('{"speed_reduction_kmh": 12}'), S.business.air_quality)
        harness = GoldenAnswerHarness(evaluator, (GOLDEN_CASES[1],))
        results = harness.run()
        assert results[0].passed
        assert GoldenAnswerHarness.pass_rate(results) == 1
        assert "PASS" in GoldenAnswerHarness.format_report(results)
        assert GoldenAnswerHarness.pass_rate([]) == 0
        assert GoldenAnswerHarness.format_report([]) == ""
