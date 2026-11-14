"""Tests for weather router."""

import dataclasses
import itertools

import pytest

from tests._support import SAFE, route
from settings import WeatherRules
from weather_router import WeatherRouter

class TestWeatherRouter:
    @pytest.fixture()
    def router(self, spec_weather):
        return WeatherRouter(spec_weather)

    @pytest.mark.parametrize(
        "illuminance,water,temperature,expected",
        [(499, 1000, 0, True), (500, 1001, -1, True),
         (500, 1000, -1, False), (500, 1001, 0, False)],
        ids=["R-01", "R-02", "R-03", "R-04"],
    )
    def test_course_router_cases(self, router, illuminance, water, temperature, expected):
        assert router.should_invoke_nn(illuminance, water, temperature) is expected

    # -- Equivalence classes: every combination of (dark?, wet?, freezing?) ----------
    @pytest.mark.parametrize("dark,wet,freezing", list(itertools.product([False, True], repeat=3)))
    def test_equivalence_classes_truth_table(self, router, dark, wet, freezing):
        reading = dict(illuminance_millilux=100.0 if dark else 5e7,
                       water_level_um=1500.0 if wet else 500.0,
                       temperature_c=-5.0 if freezing else 5.0)
        expected = dark or (wet and freezing)                    # REQ1 = SPEC1 OR SPEC2
        assert router.should_invoke_nn(**reading) is expected

    # -- BVA SPEC1: darkness boundary 500 millilux (strictly below) ----------------------
    @pytest.mark.parametrize("illuminance,expected", [
        (0, True), (1, True), (499, True), (499.999, True),
        (500, False), (500.001, False), (501, False), (1_000_000_000, False)])
    def test_bva_illuminance(self, router, illuminance, expected):
        assert route(router, illuminance_millilux=illuminance) is expected

    # -- BVA SPEC2 water: boundary 1000 um (strictly above), freezing held constant -----------
    @pytest.mark.parametrize("water,expected", [
        (0, False), (999, False), (1000, False), (1000.001, True), (1001, True), (10_000, True)])
    def test_bva_water_level(self, router, water, expected):
        assert route(router, water_level_um=water, temperature_c=-1.0) is expected

    # -- BVA SPEC2 temperature: boundary 0 C (strictly below), wetness held constant ------------
    @pytest.mark.parametrize("temperature,expected", [
        (-30, True), (-1, True), (-0.001, True), (0, False), (0.001, False), (1, False), (70, False)])
    def test_bva_temperature(self, router, temperature, expected):
        assert route(router, water_level_um=1500.0, temperature_c=temperature) is expected

    def test_individual_specs(self, router):
        assert router.is_dark(499) and not router.is_dark(500)
        assert router.is_black_ice_danger(1001, -1)
        assert not router.is_black_ice_danger(1000, -1)
        assert not router.is_black_ice_danger(1001, 0)

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True])
    def test_invalid_input_is_rejected_not_ignored(self, router, bad):
        with pytest.raises(ValueError):
            route(router, illuminance_millilux=bad)

    def test_thresholds_come_from_injected_rules_not_from_code(self, spec_weather):
        custom = dataclasses.replace(spec_weather, darkness_illuminance_below_millilux=300.0)
        assert route(WeatherRouter(custom), illuminance_millilux=400) is False
        assert route(WeatherRouter(spec_weather), illuminance_millilux=400) is True
