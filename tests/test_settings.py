"""Tests for settings."""

import dataclasses

import pytest

import settings as cfg
from tests._support import CONFIG_PATH, PROJECT_ROOT

class TestConfiguration:
    def test_default_config_path_loads_shipped_configuration(self, monkeypatch):
        monkeypatch.chdir(PROJECT_ROOT)
        settings = cfg.load_settings()
        assert settings.technical.llm.ollama_model == "llama3.2:3b"

    def test_shipped_config_is_valid_and_matches_the_specification(self, S):
        w, sp = S.business.weather, S.business.speed
        assert (w.darkness_illuminance_below_millilux, w.black_ice_water_level_above_um,
                w.black_ice_temperature_below_c) == (500.0, 1000.0, 0.0)
        assert (sp.default_kmh, sp.min_kmh, sp.max_kmh) == (80, 20, 80)
        b = S.business.sensor_bounds
        assert (b.temperature_c_min, b.temperature_c_max) == (-30.0, 70.0)
        assert (b.water_level_um_min, b.water_level_um_max) == (0.0, 10_000.0)
        assert (b.illuminance_lux_min, b.illuminance_lux_max) == (0.0, 1_000_000.0)
        llm_settings = S.technical.llm
        assert llm_settings.ollama_url == "http://localhost:11434"
        assert llm_settings.ollama_model == "llama3.2:3b"

    def _write(self, tmp_path, old, new):
        path = tmp_path / "bad.toml"
        path.write_text(CONFIG_PATH.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
        return path

    @pytest.mark.parametrize("old,new", [
        ("min_kmh = 20", "min_kph = 20"),                                       # typo -> missing + unknown key
        ("default_kmh = 80", "default_kmh = 95"),                               # default above max
        ('ollama_model = "llama3.2:3b"', 'ollama_model = "llama3.2:3b"\nunsupported = true'),
        ("seconds_per_sim_hour = 2.0", "seconds_per_sim_hour = 0")])
    def test_invalid_configuration_is_rejected(self, tmp_path, old, new):
        with pytest.raises(cfg.ConfigError):
            cfg.load_settings(self._write(tmp_path, old, new))

    def test_missing_file(self, tmp_path):
        with pytest.raises(cfg.ConfigError):
            cfg.load_settings(tmp_path / "nope.toml")

    def test_invalid_toml_and_missing_section_are_rejected(self, tmp_path):
        broken = tmp_path / "broken.toml"
        broken.write_text("[business\n", encoding="utf-8")
        with pytest.raises(cfg.ConfigError, match="invalid TOML"):
            cfg.load_settings(broken)

        missing_section = self._write(tmp_path, "[technical.nn]", "[technical.no_nn]")
        with pytest.raises(cfg.ConfigError, match="missing section"):
            cfg.load_settings(missing_section)

        missing_path = self._write(tmp_path, 'data_dir = "data"\n', "")
        with pytest.raises(cfg.ConfigError, match="data_dir"):
            cfg.load_settings(missing_path)

    def test_path_settings_name_both_periods(self, S):
        paths = S.technical.paths
        assert paths.sensor_readings_year_plus_one_file.endswith("YearNPlus1_V100.csv")
        assert paths.accidents_year_plus_one_file.endswith("YearNPlus1_V100.csv")

    @pytest.mark.parametrize("changes,message", [
        ({"speed": {"min_kmh": 0}}, "min_kmh"),
        ({"air_quality": {"min_reduction_kmh": -1}}, "min_reduction"),
        ({"air_quality": {"max_reduction_kmh": 81}}, "max_reduction"),
        ({"sensor_bounds": {"temperature_c_min": 70.0}}, "temperature_c_min"),
        ({"sensor_bounds": {"water_level_um_min": 10000.0}}, "water_level_um_min"),
        ({"sensor_bounds": {"illuminance_lux_min": 1000000.0}}, "illuminance_lux_min"),
        ({"weather": {"target_near_accidents_per_hour": 0}}, "target_near"),
        ({"danger_index": {"darkness_weight": -1}}, "weights"),
        ({"danger_index": {"freezing_saturation_c": 0}}, "weights"),
        ({"air_quality_api": {"latitude": 91}}, "latitude"),
        ({"air_quality_api": {"longitude": 181}}, "longitude"),
        ({"air_quality_api": {"timeout_s": 0}}, "timeout_s"),
        ({"air_quality_api": {"cache_ttl_s": -1}}, "cache_ttl_s"),
        ({"nn": {"hidden_layer_sizes": ()}}, "hidden_layer_sizes"),
        ({"nn": {"hidden_layer_sizes": (0,)}}, "hidden_layer_sizes"),
        ({"nn": {"learning_rate": 0}}, "learning_rate"),
        ({"nn": {"learning_rate": -0.01}}, "learning_rate"),
        ({"llm": {"timeout_s": 0}}, "timeout_s"),
        ({"llm": {"timeout_s": -1}}, "timeout_s"),
        ({"nn": {"test_fraction": 0}}, "validation/test fractions"),
        ({"nn": {"validation_fraction": 0.8}}, "validation/test fractions"),
        ({"nn": {"max_epochs": 0}}, "validation/test fractions"),
        ({"nn": {"patience": 0}}, "validation/test fractions"),
        ({"nn": {"min_training_rows": 0}}, "min_training_rows"),
        ({"data": {"near_accident_types": ()}}, "near_accident_types"),
        ({"data": {"near_accident_types": ("",)}}, "near_accident_types"),
        ({"data": {"near_accident_types": (" ",)}}, "near_accident_types"),
        ({"data": {"temperature_units_by_sensor": {"15": "K"}}}, "temperature sensor units"),
        ({"simulator": {"seconds_per_sim_hour": 0}}, "seconds_per_sim_hour"),
        ({"runtime": {"ui": "terminal"}}, "runtime"),
    ])
    def test_validation_rejects_invalid_settings(self, S, changes, message):
        business = S.business
        technical = S.technical
        for field, updates in changes.items():
            if hasattr(business, field):
                business = dataclasses.replace(business, **{field: dataclasses.replace(
                    getattr(business, field), **updates)})
            else:
                technical = dataclasses.replace(technical, **{field: dataclasses.replace(
                    getattr(technical, field), **updates)})
        invalid = dataclasses.replace(S, business=business, technical=technical)
        with pytest.raises(cfg.ConfigError, match=message):
            cfg.validate_settings(invalid)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_all_numeric_fields_reject_nonfinite_values(self, S, value):
        for group_field in dataclasses.fields(S):
            group = getattr(S, group_field.name)
            for section_field in dataclasses.fields(group):
                section = getattr(group, section_field.name)
                for field in dataclasses.fields(section):
                    original = getattr(section, field.name)
                    if isinstance(original, (int, float)):
                        replacement = value
                    elif isinstance(original, tuple) and original and isinstance(original[0], int):
                        replacement = (value,)
                    else:
                        continue
                    changed_section = dataclasses.replace(section, **{field.name: replacement})
                    changed_group = dataclasses.replace(group, **{section_field.name: changed_section})
                    invalid = dataclasses.replace(S, **{group_field.name: changed_group})
                    with pytest.raises(cfg.ConfigError, match=rf"{field.name}.*finite"):
                        cfg.validate_settings(invalid)

    @pytest.mark.parametrize("latitude,longitude", [(-90, -180), (90, 180), (0, 0)])
    def test_valid_boundaries_and_zero_values_are_preserved(self, S, latitude, longitude):
        business = dataclasses.replace(
            S.business,
            air_quality=dataclasses.replace(S.business.air_quality, min_reduction_kmh=0),
            danger_index=dataclasses.replace(
                S.business.danger_index, darkness_weight=0, black_ice_weight=0),
        )
        technical = dataclasses.replace(
            S.technical,
            air_quality_api=dataclasses.replace(
                S.technical.air_quality_api, latitude=latitude, longitude=longitude, cache_ttl_s=0),
            nn=dataclasses.replace(S.technical.nn, learning_rate=0.000001, random_seed=0),
            llm=dataclasses.replace(S.technical.llm, timeout_s=0.000001),
        )
        cfg.validate_settings(dataclasses.replace(S, business=business, technical=technical))

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_loader_rejects_nonfinite_toml(self, tmp_path, value):
        path = self._write(tmp_path, "seconds_per_sim_hour = 2.0",
                           f"seconds_per_sim_hour = {value}")
        with pytest.raises(cfg.ConfigError, match="seconds_per_sim_hour.*finite"):
            cfg.load_settings(path)
