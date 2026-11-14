"""Tests for guardrails."""

import dataclasses

import pytest

from guardrails import InputGuardrail, OutputGuardrail

class TestGuardrails:
    GOOD = dict(temperature_c=10.0, water_level_um=500.0, illuminance_millilux=50_000.0)

    @pytest.fixture()
    def inp(self, S):
        return InputGuardrail(S.business.sensor_bounds)

    @pytest.fixture()
    def out(self, S):
        return OutputGuardrail(S.business.speed)

    @pytest.mark.parametrize("field,value,valid", [
        ("temperature_c", -30, True), ("temperature_c", -30.1, False),
        ("temperature_c", 70, True), ("temperature_c", 70.1, False),
        ("water_level_um", 0, True), ("water_level_um", -1, False),
        ("water_level_um", 10_000, True), ("water_level_um", 10_001, False),
        ("illuminance_millilux", 0, True), ("illuminance_millilux", -1, False),
        ("illuminance_millilux", 1_000_000_000, True), ("illuminance_millilux", 1_000_000_001, False)])
    def test_input_bounds_are_inclusive(self, inp, field, value, valid):
        result = inp.validate({**self.GOOD, field: value})
        assert result.is_valid is valid
        assert (result.reading[field] is not None) is valid

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "warm", True])
    def test_input_rejects_missing_nonfinite_and_nonnumeric(self, inp, bad):
        result = inp.validate({**self.GOOD, "temperature_c": bad})
        assert not result.is_valid and result.reading["temperature_c"] is None
        assert result.violations and result.reading["water_level_um"] == 500.0

    @pytest.mark.parametrize("requested,expected", [
        (19, 20), (20, 20), (55, 55), (80, 80), (81, 80), (-5, 20), (1000, 80), (55.6, 56),
        (None, 20), (float("nan"), 20), ("fast", 20), (True, 20)])
    def test_output_guardrail_enforces_range(self, out, requested, expected):
        assert out.enforce(requested).speed_kmh == expected

    def test_output_guardrail_reports_modifications(self, out):
        assert not out.enforce(60).was_modified and out.enforce(90).was_modified

    def test_input_bounds_come_from_domain_configuration(self, S):
        guard = InputGuardrail(dataclasses.replace(S.business.sensor_bounds, temperature_c_max=5.0))
        assert guard.validate({**self.GOOD, "temperature_c": 5.0}).is_valid
        assert not guard.validate({**self.GOOD, "temperature_c": 5.1}).is_valid

    def test_missing_input_field_is_reported_and_filtered(self, inp):
        result = inp.validate({"water_level_um": 0, "illuminance_millilux": 0})
        assert not result.is_valid
        assert result.reading["temperature_c"] is None
        assert result.violations == ("temperature_c: missing",)
