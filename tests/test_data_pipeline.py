"""Tests for data pipeline."""

import pandas as pd
import pytest

from tests._support import DAY, long_readings, raw_accidents, raw_sensors
from data_io import CSVDataReader
from data_pipeline import DataCleaner, DataSchemaError, FeatureEngineer

NEAR_ACCIDENT_TYPES = ("skidding", "close_cars", "close_guardrails")


class TestDataPipeline:
    def test_sensor_cleaning_rules(self, S, raw_sensors):
        clean = DataCleaner(S.business.sensor_bounds).clean_sensor_readings(raw_sensors)
        assert len(clean) == 48 and not clean[[
            "temperature_c", "water_level_um", "illuminance_millilux",
        ]].isna().any().any()
        assert clean["timestamp"].is_monotonic_increasing and clean["timestamp"].is_unique
        assert clean["temperature_c"].between(-30, 70).all()           # 999 was interpolated away

    def test_accident_cleaning_rules(self, S, raw_accidents):
        clean = DataCleaner(S.business.sensor_bounds).clean_accidents(raw_accidents)
        assert list(clean.columns) == ["timestamp", "event_type"]       # no licence plates
        assert list(clean["event_type"]) == ["skidding", "close_cars", "crash", "close_guardrails"]

    def test_missing_columns_raise_a_clear_error(self, S):
        cleaner = DataCleaner(S.business.sensor_bounds)
        with pytest.raises(DataSchemaError, match="missing columns"):
            cleaner.clean_sensor_readings(pd.DataFrame({"x": [1]}))
        with pytest.raises(DataSchemaError, match="missing columns"):
            cleaner.clean_accidents(pd.DataFrame({"Month": [1], "Day": [1], "Hour": [1]}))

    def test_temperature_configuration_rejects_unknown_units(self, S):
        with pytest.raises(ValueError, match="unsupported temperature units"):
            DataCleaner(
                S.business.sensor_bounds,
                temperature_units_by_sensor={"15": "K"},
            )

    def test_training_set_counts_only_near_accidents_with_valid_time(self, S, raw_sensors, raw_accidents):
        cleaner = DataCleaner(S.business.sensor_bounds)
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index,
                             NEAR_ACCIDENT_TYPES)
        X, y = fe.build_training_set(cleaner.clean_sensor_readings(raw_sensors),
                                     cleaner.clean_accidents(raw_accidents))
        assert list(X.columns) == list(fe.FEATURE_COLUMNS) and len(X) == len(y) == 47
        assert y.sum() == 3 and y.iloc[2] == 2 and y.iloc[29] == 1     # crash and NaT rows ignored

    def test_event_policy_can_count_only_skidding(self, S, raw_sensors, raw_accidents):
        cleaner = DataCleaner(S.business.sensor_bounds)
        engineer = FeatureEngineer(
            S.business.weather,
            S.business.sensor_bounds,
            S.business.danger_index,
            (" SKIDDING ",),
        )
        _, target = engineer.build_training_set(
            cleaner.clean_sensor_readings(raw_sensors),
            cleaner.clean_accidents(raw_accidents),
        )
        assert target.sum() == 1 and target.iloc[2] == 1

    def test_fahrenheit_sensor_is_normalized_before_hourly_aggregation(self, S):
        rows = pd.DataFrame([
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 15, "Value": 41},
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 18, "Value": 7},
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 1, "Value": 100},
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 2, "Value": 5},
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": -1, "Value": 60},
        ])
        cleaner = DataCleaner(
            S.business.sensor_bounds,
            temperature_units_by_sensor={"15": "F", "18": "C"},
        )
        clean = cleaner.clean_sensor_readings(rows)
        assert clean.iloc[0]["temperature_c"] == 6

    def test_training_requires_recorded_speed_sensor(self, S, raw_sensors, raw_accidents):
        cleaner = DataCleaner(S.business.sensor_bounds)
        sensors = cleaner.clean_sensor_readings(raw_sensors).drop(columns="speed_limit_kmh")
        engineer = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        with pytest.raises(DataSchemaError, match="no speed-limit sensor"):
            engineer.build_training_set(sensors, cleaner.clean_accidents(raw_accidents))

    def test_training_rejects_rows_without_valid_recorded_speed(self, S, raw_sensors, raw_accidents):
        cleaner = DataCleaner(S.business.sensor_bounds)
        sensors = cleaner.clean_sensor_readings(raw_sensors)
        sensors["speed_limit_kmh"] = float("nan")
        engineer = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        with pytest.raises(DataSchemaError, match="no hourly rows"):
            engineer.build_training_set(sensors, cleaner.clean_accidents(raw_accidents))

    def test_hourly_speed_discards_values_outside_supported_range(self):
        readings = pd.DataFrame({
            "timestamp": pd.to_datetime(["2000-01-01 00:00", "2000-01-01 01:00"]),
            "value": [0, 201],
        })
        assert DataCleaner._hourly_speed(readings).empty

    def test_training_and_inference_features_are_identical(self, S, raw_sensors, raw_accidents):
        cleaner = DataCleaner(S.business.sensor_bounds)
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        sensors = cleaner.clean_sensor_readings(raw_sensors)
        X, _ = fe.build_training_set(sensors, cleaner.clean_accidents(raw_accidents))
        for i in (0, 7, 18):
            live = fe.build_inference_features(sensors.iloc[i].to_dict())
            pd.testing.assert_frame_equal(live, X.iloc[[i]].reset_index(drop=True))

    def test_danger_index_behaviour(self, S):
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)

        def tdi(**kw):
            r = dict(timestamp="2026-01-15 12:00", temperature_c=10.0, water_level_um=0.0,
                     illuminance_millilux=5e7, speed_limit_kmh=60)
            return float(fe.build_inference_features({**r, **kw})["total_danger_index"].iloc[0])
        assert tdi() == 0.0                                                   # day, dry, warm
        assert tdi(illuminance_millilux=0.0) > tdi(illuminance_millilux=250.0) > 0
        assert tdi(temperature_c=-10.0, water_level_um=5000.0) > tdi(temperature_c=-2.0, water_level_um=5000.0)
        assert tdi(temperature_c=5.0, water_level_um=9000.0) == 0.0           # wet but not freezing

    @pytest.mark.parametrize("patch", [{"temperature_c": None}, {"timestamp": None},
                                       {"water_level_um": float("nan")}])
    def test_inference_rejects_incomplete_readings(self, S, patch):
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        reading = {**DAY, **patch}
        with pytest.raises(ValueError):
            fe.build_inference_features(reading)

    def test_long_sensor_csv_format_is_parsed_and_aggregated(self, S, tmp_path):
        rows = []
        for hour in (0, 1):
            for sensor, value in ((15, 5), (18, 7), (19, 9), (-1, 60), (1, 100), (8, 200),
                                  (11, 300), (2, 1), (9, 2), (12, 3)):
                rows.append({"Month": 1, "Day": 2, "Hour": hour, "Minute": 30,
                             "Second": 0, "Sensor": sensor, "Value": value})
        rows.extend([
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 45, "Second": 0, "Sensor": "bad", "Value": 2},
            {"Month": 13, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 15, "Value": 10},
        ])
        path = tmp_path / "long.csv"
        pd.DataFrame(rows).to_csv(path, sep=";", index=False)
        clean = DataCleaner(S.business.sensor_bounds).clean_sensor_readings(CSVDataReader().read(path))
        assert len(clean) == 2
        assert clean.iloc[0]["temperature_c"] == 6
        assert clean.iloc[0]["water_level_um"] == 200
        assert clean.iloc[0]["illuminance_millilux"] == 2000

    def test_long_sensor_data_without_each_measurement_type_fails(self, S):
        rows = pd.DataFrame([{"Month": 1, "Day": 2, "Hour": 0, "Minute": 0,
                              "Second": 0, "Sensor": 15, "Value": 5}])
        with pytest.raises(DataSchemaError, match="no usable"):
            DataCleaner(S.business.sensor_bounds).clean_sensor_readings(rows)

    @pytest.mark.parametrize("gap_hours", [6, 7])
    @pytest.mark.parametrize("column", [
        "temperature_c", "water_level_um", "illuminance_millilux",
    ])
    def test_interpolation_requires_complete_gap_within_limit(self, S, gap_hours, column):
        timestamps = pd.date_range("2000-01-01", periods=12, freq="h")
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_c": range(12),
            "water_level_um": range(100, 112),
            "illuminance_millilux": range(1000, 1012),
            "speed_limit_kmh": [60, *([float("nan")] * 10), 80],
        }).astype({"temperature_c": float, "water_level_um": float, "illuminance_millilux": float})
        measurements = ["temperature_c", "water_level_um", "illuminance_millilux"]
        short_gap_column = next(name for name in measurements if name != column)
        frame.loc[1:gap_hours, column] = float("nan")
        frame.loc[2:3, short_gap_column] = float("nan")

        clean = DataCleaner(S.business.sensor_bounds).clean_sensor_readings(long_readings(frame))

        expected = timestamps if gap_hours == 6 else timestamps.delete(range(1, 8))
        assert clean["timestamp"].tolist() == expected.tolist()
        for name, offset in zip(measurements, (0, 100, 1000)):
            assert clean[name].tolist() == pytest.approx([offset + ts.hour for ts in expected])
        assert clean["speed_limit_kmh"].dropna().tolist() == [60, 80]
        assert clean["speed_limit_kmh"].isna().sum() == len(clean) - 2

    def test_interpolation_does_not_fill_unbounded_measurement_gaps(self, S):
        timestamps = pd.date_range("2000-01-01", periods=10, freq="h")
        frame = pd.DataFrame({
            "timestamp": timestamps,
            "temperature_c": [None, None, *range(2, 10)],
            "water_level_um": [*range(100, 108), None, None],
            "illuminance_millilux": range(1000, 1010),
        }).astype({"temperature_c": float, "water_level_um": float, "illuminance_millilux": float})

        clean = DataCleaner(S.business.sensor_bounds).clean_sensor_readings(long_readings(frame))

        assert clean["timestamp"].tolist() == timestamps[2:8].tolist()

    def test_sensors_with_no_usable_rows_raise(self, S):
        rows = pd.DataFrame([{"Month": 1, "Day": "bad", "Hour": 0, "Minute": 0,
                              "Second": 0, "Sensor": 15, "Value": 5}])
        with pytest.raises(DataSchemaError, match="no usable"):
            DataCleaner(S.business.sensor_bounds).clean_sensor_readings(rows)

    def test_case_study_accidents_map_sparse_event_fields(self, S):
        frame = pd.DataFrame({
            "Month": [1, 1, 1, 1], "Day": [2, 2, 2, 2], "Hour": [3, 4, 5, 6],
            "Second": [0, 0, 0, 0], "CloseCarCm": [5, None, None, None],
            "CloseGuardrailCm": [None, 7, None, None], "SkidAngle": [None, None, 45, None],
        })
        cleaned = DataCleaner(S.business.sensor_bounds).clean_accidents(frame)
        assert list(cleaned["event_type"]) == ["close_cars", "close_guardrails", "skidding", "crash"]

    def test_case_study_accidents_without_optional_event_columns(self, S):
        frame = pd.DataFrame({"Month": [1], "Day": [2], "Hour": [3], "Second": [0]})
        cleaned = DataCleaner(S.business.sensor_bounds).clean_accidents(frame)
        assert cleaned.iloc[0]["event_type"] == "crash"

    def test_long_readings_with_no_complete_hours_raise(self, S):
        rows = [
            {"Month": 1, "Day": 2, "Hour": 0, "Minute": 0, "Second": 0, "Sensor": 15, "Value": 5},
            {"Month": 1, "Day": 2, "Hour": 1, "Minute": 0, "Second": 0, "Sensor": 1, "Value": 100},
            {"Month": 1, "Day": 2, "Hour": 2, "Minute": 0, "Second": 0, "Sensor": 2, "Value": 5},
        ]
        with pytest.raises(DataSchemaError, match="no usable hourly"):
            DataCleaner(S.business.sensor_bounds, max_interpolation_gap_hours=1).clean_sensor_readings(
                pd.DataFrame(rows))

    @pytest.mark.parametrize("value", [True, "warm", float("inf")])
    def test_inference_rejects_non_numeric_values(self, S, value):
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        with pytest.raises(ValueError, match="must be a finite number"):
            fe.build_inference_features({**DAY, "speed_limit_kmh": 60, "temperature_c": value})

    def test_feature_transform_accounts_for_partial_hours(self, S):
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        features = fe.build_inference_features({
            **DAY, "speed_limit_kmh": 60, "timestamp": "2026-01-15 12:30",
        })
        assert features.loc[0, "hour_sin"] != 0

    def test_candidate_features_reject_empty_candidates(self, S):
        fe = FeatureEngineer(S.business.weather, S.business.sensor_bounds, S.business.danger_index)
        with pytest.raises(ValueError, match="at least one candidate"):
            fe.build_candidate_features(DAY, ())
