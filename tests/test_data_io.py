"""Tests for safe, explicit CSV input."""

import csv
import pytest

from data_io import CSVDataReader, SENSOR_COLUMNS


@pytest.mark.parametrize(
    "content,expected",
    [
        ("timestamp,value\n2024-01-01,1\n", 1.0),
        ("timestamp;value\n2024-01-01;1,5\n", 1.5),
    ],
)
def test_reader_detects_delimiter_and_decimal_separator(tmp_path, content, expected):
    path = tmp_path / "readings.csv"
    path.write_text(content, encoding="utf-8")

    result = CSVDataReader().read(path)

    assert list(result.columns) == ["timestamp", "value"]
    assert result.iloc[0]["timestamp"] == "2024-01-01"
    assert result.iloc[0]["value"] == expected


def test_reader_surfaces_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVDataReader().read(tmp_path / "missing.csv")


@pytest.mark.parametrize("headerless", [None, False])
def test_headered_sensor_reading_delegates_to_shared_reader(tmp_path, monkeypatch, headerless):
    reader = CSVDataReader()
    path = tmp_path / "headered.csv"
    expected = object()
    calls = []

    def read(requested_path):
        calls.append(requested_path)
        return expected

    monkeypatch.setattr(reader, "read", read)
    result = (reader.read_sensor_readings(path) if headerless is None
              else reader.read_sensor_readings(path, headerless=headerless))

    assert result is expected
    assert calls == [path]


def test_headerless_sensor_reading_uses_known_columns(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text("1;2;3;4;5;-1;60,5\n", encoding="utf-8")
    frame = CSVDataReader().read_sensor_readings(path, headerless=True)
    assert tuple(frame.columns) == SENSOR_COLUMNS
    assert frame.iloc[0]["Sensor"] == -1
    assert frame.iloc[0]["Value"] == 60.5


def test_accident_reader_excludes_licence_plate_case_insensitively(tmp_path):
    path = tmp_path / "accidents.csv"
    path.write_text("Month;LicencePlate;Event\n1;secret;skidding\n", encoding="utf-8")
    frame = CSVDataReader().read_accidents(path)
    assert frame.to_dict(orient="records") == [{"Month": 1, "Event": "skidding"}]


def test_delimiter_fallback_detects_present_separator(tmp_path, monkeypatch):
    path = tmp_path / "fallback.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")

    class NoSniff:
        def sniff(self, *_args, **_kwargs):
            raise csv.Error("ambiguous")

    monkeypatch.setattr("data_io.csv.Sniffer", NoSniff)
    assert CSVDataReader._delimiter(path) == ";"


def test_delimiter_detection_fails_for_unseparated_input(tmp_path, monkeypatch):
    path = tmp_path / "invalid.csv"
    path.write_text("header\nvalue\n", encoding="utf-8")

    class NoSniff:
        def sniff(self, *_args, **_kwargs):
            raise csv.Error("ambiguous")

    monkeypatch.setattr("data_io.csv.Sniffer", NoSniff)
    with pytest.raises(ValueError, match="could not detect CSV delimiter"):
        CSVDataReader._delimiter(path)
