"""Type T / Software 1.0 client for the Open-Meteo air-quality API."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from air_quality_llm import POLLUTION_INDEX_KEY
from settings import AirQualityApiSettings

LOG = logging.getLogger("speedlimit.airquality")

ATTRIBUTION = "Air quality data by Open-Meteo.com (CC BY 4.0), based on CAMS ENSEMBLE data (Copernicus)."

CURRENT_VARIABLES = (
    "european_aqi", "pm10", "pm2_5",
    "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide",
)

class AirQualityAPIError(RuntimeError):
    """Network failure, HTTP error or unusable response from Open-Meteo."""


class AirQualityAPIClient:
    """Type T / Software 1.0 - fetches and normalises current air quality."""

    _ATTEMPTS = 2  # one retry on network errors

    def __init__(self, settings: AirQualityApiSettings) -> None:
        self._s = settings
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def fetch(self) -> dict[str, Any]:
        """Return normalized metrics; missing pollutants are None, but AQI is required."""
        with self._lock:
            now = time.monotonic()
            if self._cache and now - self._cache[0] < self._s.cache_ttl_s:
                data = self._cache[1]
                LOG.info(
                    "Using cached air quality: source=%s location=%s "
                    "coordinates=%.4f,%.4f time=%s european_aqi=%s",
                    data["source"], data["location"], self._s.latitude,
                    self._s.longitude, data["time"], data[POLLUTION_INDEX_KEY],
                )
                return dict(data)
            data = self._normalise(self._http_get(), source="open-meteo")
            self._cache = (now, data)
            LOG.info(
                "Fetched air quality: source=%s location=%s coordinates=%.4f,%.4f "
                "time=%s european_aqi=%s",
                data["source"], data["location"], self._s.latitude,
                self._s.longitude, data["time"], data[POLLUTION_INDEX_KEY],
            )
            return dict(data)

    def _build_url(self) -> str:
        query = urllib.parse.urlencode({
            "latitude": self._s.latitude,
            "longitude": self._s.longitude,
            "current": ",".join(CURRENT_VARIABLES),
            "timezone": self._s.timezone,
        })
        return f"{self._s.base_url}?{query}"

    def _http_get(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self._build_url(),
            headers={"Accept": "application/json", "User-Agent": "speedlimit-highway-mvp/1.0"},
        )
        last_error: Exception | None = None
        for attempt in range(1, self._ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._s.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:          # 4xx / 5xx
                raise AirQualityAPIError(f"HTTP {exc.code}: {self._error_reason(exc)}") from exc
            except json.JSONDecodeError as exc:
                raise AirQualityAPIError(f"response is not valid JSON: {exc}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:  # network down, DNS, timeout
                last_error = exc
                LOG.warning("Air quality request failed (attempt %d/%d): %s",
                            attempt, self._ATTEMPTS, exc)
        raise AirQualityAPIError(f"network error: {last_error}") from last_error

    @staticmethod
    def _error_reason(exc: urllib.error.HTTPError) -> str:
        try:
            return str(json.loads(exc.read().decode("utf-8")).get("reason", exc.reason))
        except Exception:  # noqa: BLE001 - best effort only
            return str(exc.reason)

    def _normalise(self, raw: dict[str, Any], source: str) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise AirQualityAPIError("unexpected payload type")
        if raw.get("error"):
            raise AirQualityAPIError(f"API error: {raw.get('reason', 'unknown reason')}")
        current = raw.get("current")
        if not isinstance(current, dict):
            raise AirQualityAPIError("payload has no 'current' section")

        result: dict[str, Any] = {
            "source": source,
            "location": self._s.location_name,
            "time": current.get("time"),
            "units": raw.get("current_units", {}),
            "attribution": ATTRIBUTION,
        }
        for name in CURRENT_VARIABLES:
            value = current.get(name)
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) and value >= 0)
            result[name] = float(value) if ok else None
        if result[POLLUTION_INDEX_KEY] is None:
            raise AirQualityAPIError(f"payload lacks '{POLLUTION_INDEX_KEY}'")
        return result
