"""Tests for air quality api."""

import dataclasses
import io
import urllib.error
from unittest.mock import patch

import pytest

import air_quality_api as api_module
from air_quality_api import AirQualityAPIClient, AirQualityAPIError

class TestAirQualityClient:
    def test_online_parsing_and_cache(self, fake_air_quality_server, caplog):
        caplog.set_level("INFO")
        api, state = fake_air_quality_server
        client = AirQualityAPIClient(dataclasses.replace(api, cache_ttl_s=60))
        first = client.fetch()
        assert first["source"] == "open-meteo" and first["european_aqi"] == 42.0 and first["ozone"] is None
        client.fetch()
        assert state["hits"] == 1                                              # second call served from cache
        assert "Fetched air quality: source=open-meteo" in caplog.text
        assert f"location={api.location_name}" in caplog.text
        assert "european_aqi=42.0" in caplog.text
        assert "Using cached air quality: source=open-meteo" in caplog.text

    @pytest.mark.parametrize("mode", ["400", "noaqi"])
    def test_errors_become_AirQualityAPIError(self, fake_air_quality_server, mode):
        api, state = fake_air_quality_server
        state["mode"] = mode
        with pytest.raises(AirQualityAPIError):
            AirQualityAPIClient(dataclasses.replace(api, cache_ttl_s=0)).fetch()

    def test_unreachable_server_raises(self, S):
        api = dataclasses.replace(S.technical.air_quality_api, base_url="http://127.0.0.1:1", timeout_s=0.5)
        with pytest.raises(AirQualityAPIError):
            AirQualityAPIClient(api).fetch()

    def test_normalize_rejects_malformed_payloads(self, S):
        client = AirQualityAPIClient(S.technical.air_quality_api)
        for payload, message in [
            ([], "payload type"),
            ({"error": True}, "API error"),
            ({}, "current"),
            ({"current": []}, "current"),
            ({"current": {"european_aqi": True}}, "european_aqi"),
        ]:
            with pytest.raises(AirQualityAPIError, match=message):
                client._normalise(payload, "test")

    def test_normalize_sanitizes_nonfinite_and_non_numeric_pollutants(self, S):
        client = AirQualityAPIClient(S.technical.air_quality_api)
        result = client._normalise(
            {"current": {"european_aqi": 30, "pm10": True, "pm2_5": float("inf"),
                         "nitrogen_dioxide": "bad"}},
            "test",
        )
        assert result["european_aqi"] == 30.0
        assert result["pm10"] is result["pm2_5"] is result["nitrogen_dioxide"] is None

    @pytest.mark.parametrize("value", [-1, -0.001, float("nan"), float("inf"), float("-inf")])
    def test_normalize_rejects_negative_or_nonfinite_aqi(self, S, value):
        client = AirQualityAPIClient(S.technical.air_quality_api)
        with pytest.raises(AirQualityAPIError, match="european_aqi"):
            client._normalise({"current": {"european_aqi": value}}, "test")

    @pytest.mark.parametrize("name", api_module.CURRENT_VARIABLES[1:])
    @pytest.mark.parametrize("value", [-1, -0.001, float("nan"), float("inf"), float("-inf")])
    def test_normalize_discards_invalid_optional_pollutants(self, S, name, value):
        client = AirQualityAPIClient(S.technical.air_quality_api)
        result = client._normalise({"current": {"european_aqi": 30, name: value}}, "test")
        assert result[name] is None
        assert result["european_aqi"] == 30.0

    @pytest.mark.parametrize("value", [0, -0.0, 0.001, 1e308])
    def test_normalize_preserves_nonnegative_finite_metrics(self, S, value):
        client = AirQualityAPIClient(S.technical.air_quality_api)
        current = dict.fromkeys(api_module.CURRENT_VARIABLES, value)
        result = client._normalise({"current": current}, "test")
        for name in api_module.CURRENT_VARIABLES:
            assert result[name] == float(value)

    def test_network_error_retries_then_succeeds(self, S):
        response = io.BytesIO(b'{"current":{"european_aqi":42}}')
        response.__enter__ = lambda: response
        response.__exit__ = lambda *_: False
        error = urllib.error.URLError("temporary")
        with patch.object(api_module.urllib.request, "urlopen", side_effect=[error, response]) as request:
            result = AirQualityAPIClient(S.technical.air_quality_api)._http_get()
        assert result["current"]["european_aqi"] == 42
        assert request.call_count == 2

    def test_invalid_json_and_http_error_are_wrapped(self, S):
        settings = S.technical.air_quality_api
        with patch.object(api_module.urllib.request, "urlopen", return_value=_Response(b"{")):
            with pytest.raises(AirQualityAPIError, match="valid JSON"):
                AirQualityAPIClient(settings)._http_get()

        error = urllib.error.HTTPError("url", 400, "bad request", {}, io.BytesIO(b"not-json"))
        with patch.object(api_module.urllib.request, "urlopen", side_effect=error):
            with pytest.raises(AirQualityAPIError, match="HTTP 400: bad request"):
                AirQualityAPIClient(settings)._http_get()


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body
