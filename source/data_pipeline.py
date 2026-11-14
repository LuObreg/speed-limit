"""Type A / Software 1.0 data cleaning and feature engineering.

The cleaner interprets the case-study CSV layout (read by data_io). Training
and inference share one feature transform to avoid training-serving skew.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from settings import DangerIndexRules, SensorBounds, WeatherRules

LOG = logging.getLogger("speedlimit.data")

TIMESTAMP = "timestamp"
SENSOR_ID = "sensor_id"
TEMPERATURE_C = "temperature_c"
WATER_LEVEL_UM = "water_level_um"
ILLUMINANCE_MILLILUX = "illuminance_millilux"
SPEED_LIMIT_KMH = "speed_limit_kmh"
EVENT_TYPE = "event_type"
NEAR_ACCIDENTS = "near_accidents_per_hour"

MEASUREMENTS = (TEMPERATURE_C, WATER_LEVEL_UM, ILLUMINANCE_MILLILUX)
SENSOR_COLUMNS = ("Month", "Day", "Hour", "Minute", "Second", "Sensor", "Value")
ACCIDENT_COLUMNS = ("Month", "Day", "Hour", "Second")
_MILLILUX_PER_LUX = 1000.0

# Sensor 16 is assumed to be traffic; sensor 19 remains unclassified.
_WATER_SENSOR_IDS = (1, 8, 11)
_LIGHT_SENSOR_IDS = (2, 9, 12)
_TEMPERATURE_SENSOR_IDS = (15, 18)
_SPEED_SENSOR_ID = -1


class DataSchemaError(ValueError):
    """Raised when required columns are missing or no usable rows remain."""


def normalize_temperature(value: float, unit: str) -> float:
    """Convert a sensor temperature to Celsius before applying domain rules."""
    return (float(value) - 32.0) * (5.0 / 9.0) if unit.upper() == "F" else float(value)


def physical_bounds(bounds: SensorBounds) -> dict[str, tuple[float, float]]:
    """Plausible range per canonical measurement (shared by cleaning and the input guardrail)."""
    return {
        TEMPERATURE_C: (bounds.temperature_c_min, bounds.temperature_c_max),
        WATER_LEVEL_UM: (bounds.water_level_um_min, bounds.water_level_um_max),
        ILLUMINANCE_MILLILUX: (bounds.illuminance_lux_min * _MILLILUX_PER_LUX,
                               bounds.illuminance_lux_max * _MILLILUX_PER_LUX),
    }


class DataCleaner:
    """Type A / Software 1.0 - cleaning and imputation rules for the raw data."""

    def __init__(
        self,
        bounds: SensorBounds,
        max_interpolation_gap_hours: int = 6,
        temperature_units_by_sensor: Mapping[str, str] | None = None,
    ) -> None:
        self._bounds = physical_bounds(bounds)
        self._max_gap = max_interpolation_gap_hours
        self._temperature_units = {
            str(sensor): unit.upper()
            for sensor, unit in (temperature_units_by_sensor or {}).items()
        }
        invalid_units = {unit for unit in self._temperature_units.values()} - {"C", "F"}
        if invalid_units:
            raise ValueError(f"unsupported temperature units: {sorted(invalid_units)}")

    def clean_sensor_readings(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Case-study rows (date parts, sensor ID, value) -> ONE row per hour:
        [timestamp, temperature_c, water_level_um, illuminance_millilux,
        speed_limit_kmh], sorted, measurements without NaN.

        Rules:
          1. unparseable timestamps or sensor IDs -> row dropped
          2. configured Fahrenheit sensors -> Celsius; lux -> millilux
          3. non-numeric / physically impossible values -> NaN
          4. duplicate (hour, sensor) -> last one wins
          5. several sensors per hour -> averaged (NaN ignored)
          6. gaps up to `max_interpolation_gap_hours` -> time interpolation;
             longer gaps are NOT invented, those hours are dropped
        """
        self._require(frame, SENSOR_COLUMNS)
        n0 = len(frame)
        df = pd.DataFrame({
            TIMESTAMP: self._timestamp_from_parts(frame, include_minute=True).dt.floor("h"),
            SENSOR_ID: pd.to_numeric(frame["Sensor"], errors="coerce"),
            "value": pd.to_numeric(frame["Value"], errors="coerce"),
        }).dropna(subset=[TIMESTAMP, SENSOR_ID])
        df = df.sort_values(TIMESTAMP).drop_duplicates([TIMESTAMP, SENSOR_ID], keep="last")

        sensor_groups = {
            TEMPERATURE_C: _TEMPERATURE_SENSOR_IDS,
            WATER_LEVEL_UM: _WATER_SENSOR_IDS,
            ILLUMINANCE_MILLILUX: _LIGHT_SENSOR_IDS,
        }
        measurements: list[pd.Series] = []
        for column, sensor_ids in sensor_groups.items():
            selected = df[df[SENSOR_ID].isin(sensor_ids)].copy()
            if column == ILLUMINANCE_MILLILUX:
                selected["value"] *= _MILLILUX_PER_LUX
            elif column == TEMPERATURE_C:
                selected["value"] = [
                    normalize_temperature(value, self._temperature_units.get(str(int(sensor)), "C"))
                    for sensor, value in zip(selected[SENSOR_ID], selected["value"])
                ]
            low, high = self._bounds[column]
            selected.loc[~selected["value"].between(low, high), "value"] = np.nan
            measurements.append(
                selected.groupby(TIMESTAMP)["value"].mean().rename(column)
            )

        hourly = pd.concat(measurements, axis=1).dropna(how="all")
        if hourly.empty or hourly[list(MEASUREMENTS)].isna().all().any():
            raise DataSchemaError("sensor data has no usable temperature, water, or light readings")

        hourly = self._interpolate_hourly(hourly, n0)
        if hourly.empty:
            raise DataSchemaError("no usable hourly sensor rows after cleaning")
        hourly[SPEED_LIMIT_KMH] = self._hourly_speed(df[df[SENSOR_ID] == _SPEED_SENSOR_ID]).reindex(
            hourly.index)
        return hourly.rename_axis(TIMESTAMP).reset_index()

    @staticmethod
    def _hourly_speed(rows: pd.DataFrame) -> pd.Series:
        """Speed-limit readings (sensor ID -1) by hour; implausible values are discarded."""
        valid = rows[rows["value"].between(1, 200)]
        return valid.set_index(TIMESTAMP)["value"].rename(SPEED_LIMIT_KMH)

    def _interpolate_hourly(self, hourly: pd.DataFrame, raw_rows: int) -> pd.DataFrame:
        full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
        hourly = hourly.reindex(full_index)
        for column in MEASUREMENTS:
            missing = hourly[column].isna()
            gap_lengths = missing.groupby((~missing).cumsum()).transform("sum")
            interpolated = hourly[column].interpolate(method="time", limit_area="inside")
            hourly[column] = interpolated.where(~missing | gap_lengths.le(self._max_gap))
        before = len(hourly)
        hourly = hourly.dropna(subset=list(MEASUREMENTS))
        LOG.info("Sensor cleaning: %d raw rows -> %d hourly rows (%d hours dropped, unfillable)",
                 raw_rows, len(hourly), before - len(hourly))
        return hourly

    def clean_accidents(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Case-study event rows -> one row per event: [timestamp, event_type].

        The event type follows the filled attribute column (skidding wins over
        close_guardrails over close_cars); rows with none of them are crashes.
        Rows with an unparseable timestamp cannot be aligned and are dropped.
        Licence plates are never read (see data_io).
        """
        self._require(frame, ACCIDENT_COLUMNS)
        event_type = pd.Series("crash", index=frame.index, dtype="string")
        for column, name in (("CloseCarCm", "close_cars"),
                             ("CloseGuardrailCm", "close_guardrails"),
                             ("SkidAngle", "skidding")):
            if column in frame:
                event_type.loc[frame[column].notna()] = name
        events = pd.DataFrame({
            TIMESTAMP: self._timestamp_from_parts(frame, include_minute=False),
            EVENT_TYPE: event_type,
        })
        return events.dropna(subset=[TIMESTAMP]).reset_index(drop=True)

    @staticmethod
    def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise DataSchemaError(f"missing columns {missing}")

    @staticmethod
    def _timestamp_from_parts(df: pd.DataFrame, *, include_minute: bool) -> pd.Series:
        # The files have no year; a neutral year keeps both periods comparable.
        parts: dict[str, Any] = {
            "year": 2000,
            "month": pd.to_numeric(df["Month"], errors="coerce"),
            "day": pd.to_numeric(df["Day"], errors="coerce"),
            "hour": pd.to_numeric(df["Hour"], errors="coerce"),
            "minute": (pd.to_numeric(df["Minute"], errors="coerce")
                       if include_minute else 0),
            "second": pd.to_numeric(df["Second"], errors="coerce"),
        }
        return pd.to_datetime(parts, errors="coerce")


class FeatureEngineer:
    """Type A / Software 1.0 - builds model features (training AND live inference)."""

    FEATURE_COLUMNS: tuple[str, ...] = (
        ILLUMINANCE_MILLILUX, WATER_LEVEL_UM, TEMPERATURE_C, SPEED_LIMIT_KMH,
        "darkness_score", "ice_score", "total_danger_index",
        "hour_sin", "hour_cos",
    )

    def __init__(
        self,
        weather_rules: WeatherRules,
        bounds: SensorBounds,
        danger_rules: DangerIndexRules,
        near_accident_types: tuple[str, ...] = (),
    ) -> None:
        """`near_accident_types` (SPEC4 label policy) is needed for training only."""
        self._weather = weather_rules
        self._bounds = bounds
        self._danger = danger_rules
        self._near_types = {event.strip().casefold() for event in near_accident_types}

    def build_training_set(
        self, sensors_df: pd.DataFrame, accidents_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        X: features per hour with a recorded speed limit (see FEATURE_COLUMNS)
        y: number of near-accidents in that hour (SPEC4), 0 if none recorded
        """
        if SPEED_LIMIT_KMH not in sensors_df.columns:
            raise DataSchemaError("sensor data has no speed-limit sensor (ID -1)")
        near = accidents_df[accidents_df[EVENT_TYPE].isin(self._near_types)]
        counts = (
            near.assign(**{TIMESTAMP: near[TIMESTAMP].dt.floor("h")})
            .groupby(TIMESTAMP).size().rename(NEAR_ACCIDENTS)
        )
        merged = sensors_df.merge(counts, how="left", left_on=TIMESTAMP, right_index=True)
        merged[NEAR_ACCIDENTS] = merged[NEAR_ACCIDENTS].fillna(0).astype(int)
        merged = merged.dropna(subset=[SPEED_LIMIT_KMH])
        if merged.empty:
            raise DataSchemaError("no hourly rows have both weather data and a valid speed limit")

        LOG.info("Training set: %d hours, %d near-accidents, %.2f per hour on average",
                 len(merged), int(merged[NEAR_ACCIDENTS].sum()), merged[NEAR_ACCIDENTS].mean())
        return self.transform(merged), merged[NEAR_ACCIDENTS]

    def build_inference_features(
        self, reading: Mapping[str, Any], candidate_speed_kmh: int | None = None
    ) -> pd.DataFrame:
        """One validated live reading -> 1-row feature frame (same transform as training).

        `reading` needs the canonical keys: timestamp, temperature_c,
        water_level_um, illuminance_millilux (finite numbers, already passed
        through the InputGuardrail).
        Supply candidate_speed_kmh or a speed_limit_kmh reading. Missing
        fields and non-finite numeric values raise ValueError; this path
        never imputes live sensor readings or uses accident outcomes.
        """
        speed = candidate_speed_kmh
        if speed is None:
            speed = reading.get(SPEED_LIMIT_KMH)
        missing = [k for k in (TIMESTAMP, *MEASUREMENTS) if reading.get(k) is None]
        if speed is None:
            missing.append(SPEED_LIMIT_KMH)
        if missing:
            raise ValueError(f"reading lacks required fields: {missing}")
        for k in (*MEASUREMENTS, SPEED_LIMIT_KMH):
            v = speed if k == SPEED_LIMIT_KMH else reading[k]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) \
                    or not math.isfinite(float(v)):
                raise ValueError(f"{k} must be a finite number, got {v!r}")
        frame = pd.DataFrame([{
            **{k: reading[k] for k in (TIMESTAMP, *MEASUREMENTS)},
            SPEED_LIMIT_KMH: speed,
        }])
        return self.transform(frame)

    def build_candidate_features(
        self, reading: Mapping[str, Any], candidates: tuple[int, ...]
    ) -> pd.DataFrame:
        if not candidates:
            raise ValueError("at least one candidate speed is required")
        return pd.concat(
            [self.build_inference_features(reading, speed) for speed in candidates],
            ignore_index=True,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform timestamp, three measurements, and speed into model features."""
        ts = pd.to_datetime(df[TIMESTAMP], errors="raise")
        illum = df[ILLUMINANCE_MILLILUX].astype(float)
        water = df[WATER_LEVEL_UM].astype(float)
        temp = df[TEMPERATURE_C].astype(float)
        speed = df[SPEED_LIMIT_KMH].astype(float)

        dark_thr = self._weather.darkness_illuminance_below_millilux
        darkness = ((dark_thr - illum) / dark_thr).clip(0.0, 1.0)

        wet = (water / self._bounds.water_level_um_max).clip(0.0, 1.0)
        cold = (-temp / self._danger.freezing_saturation_c).clip(0.0, 1.0)
        ice = wet * cold

        total = (self._danger.darkness_weight * darkness
                 + self._danger.black_ice_weight * ice)

        angle = 2.0 * np.pi * (ts.dt.hour + ts.dt.minute / 60.0) / 24.0
        out = pd.DataFrame({
            ILLUMINANCE_MILLILUX: illum,
            WATER_LEVEL_UM: water,
            TEMPERATURE_C: temp,
            SPEED_LIMIT_KMH: speed,
            "darkness_score": darkness,
            "ice_score": ice,
            "total_danger_index": total,
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
        })
        return out[list(self.FEATURE_COLUMNS)].reset_index(drop=True)
