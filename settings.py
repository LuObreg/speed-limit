"""
settings.py - Configuration loading, parsing and validation
===========================================================

Blood type : Type T (Technik) - file I/O (TOML), parsing and validation.
             The dataclasses below are passive value objects: they carry the
             business constants but contain NO business logic. Type A classes
             receive them through their constructors (dependency injection), so
             Type A code never touches files.
Generation : Software 1.0.

"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(ValueError):
    """Raised when the configuration file is missing, incomplete or inconsistent."""


# --------------------------------------------------------------------------
# Business rule value objects (passive data, no logic)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WeatherRules:
    """Thresholds for SPEC1 (darkness) and SPEC2 (black ice) + SPEC4 target."""
    darkness_illuminance_below_millilux: float
    black_ice_water_level_above_um: float
    black_ice_temperature_below_c: float
    target_near_accidents_per_hour: float


@dataclass(frozen=True)
class SpeedRules:
    """REQ3 default speed and output-guardrail bounds."""
    default_kmh: int
    min_kmh: int
    max_kmh: int


@dataclass(frozen=True)
class AirQualityRules:
    """REQ2 pollution trigger and plausible LLM reduction range."""
    high_pollution_threshold: float
    min_reduction_kmh: int
    max_reduction_kmh: int


@dataclass(frozen=True)
class DangerIndexRules:
    """Weights/scales of the Total Danger Index feature (data_pipeline.py)."""
    darkness_weight: float
    black_ice_weight: float
    freezing_saturation_c: float


@dataclass(frozen=True)
class DataRules:
    near_accident_types: tuple[str, ...]
    temperature_units_by_sensor: dict[str, str]


@dataclass(frozen=True)
class SensorBounds:
    """Physical plausibility bounds for the input guardrail."""
    temperature_c_min: float
    temperature_c_max: float
    water_level_um_min: float
    water_level_um_max: float
    illuminance_lux_min: float
    illuminance_lux_max: float


@dataclass(frozen=True)
class BusinessRules:
    weather: WeatherRules
    speed: SpeedRules
    air_quality: AirQualityRules
    danger_index: DangerIndexRules
    data: DataRules
    sensor_bounds: SensorBounds


# --------------------------------------------------------------------------
# Technical settings value objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PathSettings:
    data_dir: Path
    model_dir: Path
    log_dir: Path
    sensor_readings_file: str
    accidents_file: str
    sensor_readings_year_plus_one_file: str
    accidents_year_plus_one_file: str


@dataclass(frozen=True)
class NNSettings:
    hidden_layer_sizes: tuple[int, ...]
    learning_rate: float
    max_epochs: int
    patience: int
    validation_fraction: float
    test_fraction: float
    random_seed: int
    min_training_rows: int


@dataclass(frozen=True)
class LLMSettings:
    timeout_s: float
    ollama_url: str
    ollama_model: str


@dataclass(frozen=True)
class SimulatorSettings:
    seconds_per_sim_hour: float


@dataclass(frozen=True)
class AirQualityApiSettings:
    base_url: str
    location_name: str
    latitude: float
    longitude: float
    timezone: str
    timeout_s: float
    cache_ttl_s: float


@dataclass(frozen=True)
class RuntimeSettings:
    ui: str


@dataclass(frozen=True)
class TechnicalSettings:
    paths: PathSettings
    nn: NNSettings
    llm: LLMSettings
    simulator: SimulatorSettings
    air_quality_api: AirQualityApiSettings
    runtime: RuntimeSettings


@dataclass(frozen=True)
class Settings:
    business: BusinessRules
    technical: TechnicalSettings


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
T = TypeVar("T")


def _build(cls: type[T], table: dict[str, Any], where: str) -> T:
    """Build a dataclass from a TOML table; reject missing AND unknown keys
    (so a typo in configuration.toml fails loudly instead of silently using a default)."""
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    missing = names - table.keys()
    unknown = table.keys() - names
    if missing:
        raise ConfigError(f"[{where}] missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"[{where}] unknown keys: {sorted(unknown)}")
    return cls(**table)  # type: ignore[call-arg]


def _section(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: Any = root
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing section [{'.'.join(keys)}]")
        node = node[key]
    return dict(node)


def validate_settings(s: Settings) -> None:
    """Validate loaded settings or settings replaced by runtime overrides."""
    pending = [(s, "")]
    while pending:
        node, prefix = pending.pop()
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            name = f"{prefix}.{field.name}" if prefix else field.name
            if dataclasses.is_dataclass(value):
                pending.append((value, name))
            else:
                values = value if isinstance(value, tuple) else (value,)
                if any(isinstance(item, (int, float)) and not math.isfinite(item)
                       for item in values):
                    raise ConfigError(f"{name} must contain only finite numbers")
    b, t = s.business, s.technical
    sp, aq, sb = b.speed, b.air_quality, b.sensor_bounds
    if not sp.min_kmh <= sp.default_kmh <= sp.max_kmh:
        raise ConfigError("speed: require min_kmh <= default_kmh <= max_kmh")
    if sp.min_kmh <= 0:
        raise ConfigError("speed: min_kmh must be > 0")
    if not 0 <= aq.min_reduction_kmh <= aq.max_reduction_kmh:
        raise ConfigError("air_quality: require 0 <= min_reduction_kmh <= max_reduction_kmh")
    if sp.max_kmh - aq.max_reduction_kmh < 0:
        raise ConfigError("air_quality: max_reduction_kmh exceeds max speed")
    for name in ("temperature_c", "water_level_um", "illuminance_lux"):
        if getattr(sb, f"{name}_min") >= getattr(sb, f"{name}_max"):
            raise ConfigError(f"sensor_bounds: {name}_min must be < {name}_max")
    if b.weather.target_near_accidents_per_hour <= 0:
        raise ConfigError("weather: target_near_accidents_per_hour must be > 0")
    di = b.danger_index
    if di.darkness_weight < 0 or di.black_ice_weight < 0 or di.freezing_saturation_c <= 0:
        raise ConfigError("danger_index: weights must be >= 0 and freezing_saturation_c > 0")
    aqa = t.air_quality_api
    if not (-90 <= aqa.latitude <= 90 and -180 <= aqa.longitude <= 180):
        raise ConfigError("air_quality_api: latitude/longitude out of range")
    if aqa.timeout_s <= 0 or aqa.cache_ttl_s < 0:
        raise ConfigError("air_quality_api: timeout_s must be > 0 and cache_ttl_s >= 0")
    nn = t.nn
    if nn.learning_rate <= 0:
        raise ConfigError("nn: learning_rate must be > 0")
    if t.llm.timeout_s <= 0:
        raise ConfigError("llm: timeout_s must be > 0")
    if not nn.hidden_layer_sizes or any(int(n) <= 0 for n in nn.hidden_layer_sizes):
        raise ConfigError("nn: hidden_layer_sizes must be a non-empty list of positive ints")
    if (nn.validation_fraction <= 0 or nn.test_fraction <= 0
            or nn.validation_fraction + nn.test_fraction >= 1
            or nn.max_epochs <= 0 or nn.patience <= 0 or nn.min_training_rows <= 0):
        raise ConfigError(
            "nn: require positive validation/test fractions with sum < 1, "
            "max_epochs > 0, patience > 0, min_training_rows > 0"
        )
    if not b.data.near_accident_types or any(not item.strip() for item in b.data.near_accident_types):
        raise ConfigError("data: near_accident_types must contain at least one event type")
    if any(unit.upper() not in ("C", "F") for unit in b.data.temperature_units_by_sensor.values()):
        raise ConfigError("data: temperature sensor units must be C or F")
    if t.simulator.seconds_per_sim_hour <= 0:
        raise ConfigError("simulator: seconds_per_sim_hour must be > 0")
    if t.runtime.ui not in ("gui", "headless"):
        raise ConfigError('runtime: ui must be "gui" or "headless"')


def load_settings(path: str | Path = "configuration.toml") -> Settings:
    """Read, parse and validate the configuration file."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    business = BusinessRules(
        weather=_build(WeatherRules, _section(raw, "business", "weather"), "business.weather"),
        speed=_build(SpeedRules, _section(raw, "business", "speed"), "business.speed"),
        air_quality=_build(AirQualityRules, _section(raw, "business", "air_quality"),
                           "business.air_quality"),
        danger_index=_build(DangerIndexRules, _section(raw, "business", "danger_index"),
                            "business.danger_index"),
        data=_build(DataRules, {
            **_section(raw, "business", "data"),
            "near_accident_types": tuple(_section(raw, "business", "data")
                                         .get("near_accident_types", ())),
        }, "business.data"),
        sensor_bounds=_build(SensorBounds, _section(raw, "business", "sensor_bounds"),
                             "business.sensor_bounds"),
    )

    paths_tbl = _section(raw, "technical", "paths")
    for key in ("data_dir", "model_dir", "log_dir"):
        if key in paths_tbl:
            paths_tbl[key] = Path(paths_tbl[key])

    technical = TechnicalSettings(
        paths=_build(PathSettings, paths_tbl, "technical.paths"),
        nn=_build(NNSettings, {**_section(raw, "technical", "nn"),
                               "hidden_layer_sizes": tuple(_section(raw, "technical", "nn")
                                                           .get("hidden_layer_sizes", ()))},
                  "technical.nn"),
        llm=_build(LLMSettings, _section(raw, "technical", "llm"), "technical.llm"),
        simulator=_build(SimulatorSettings, _section(raw, "technical", "simulator"),
                         "technical.simulator"),
        air_quality_api=_build(AirQualityApiSettings,
                               _section(raw, "technical", "air_quality_api"),
                               "technical.air_quality_api"),
        runtime=_build(RuntimeSettings, _section(raw, "technical", "runtime"),
                       "technical.runtime"),
    )

    settings = Settings(business=business, technical=technical)
    validate_settings(settings)
    return settings
