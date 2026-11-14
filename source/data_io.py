"""Type T CSV input for training and sensor replay."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from data_pipeline import SENSOR_COLUMNS


class CSVDataReader:
    """Read project CSV files without embedding file access in domain transforms."""

    def read(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        delimiter = self._delimiter(path)
        return pd.read_csv(path, sep=delimiter, decimal=",", encoding="utf-8-sig")

    def read_sensor_readings(
        self, path: Path | str, *, headerless: bool = False
    ) -> pd.DataFrame:
        if not headerless:
            return self.read(path)
        return pd.read_csv(
            Path(path), sep=";", decimal=",", encoding="utf-8-sig",
            names=SENSOR_COLUMNS, header=None,
        )

    def read_accidents(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        return pd.read_csv(
            path,
            sep=self._delimiter(path),
            decimal=",",
            encoding="utf-8-sig",
            usecols=lambda name: name.strip().casefold() != "licenceplate",
        )

    @staticmethod
    def _delimiter(path: Path) -> str:
        with path.open(encoding="utf-8-sig", newline="") as file:
            sample = file.read(4096)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            counts = {delimiter: sample.count(delimiter) for delimiter in ",;\t"}
            delimiter = max(counts, key=counts.get)
            if counts[delimiter] == 0:
                raise ValueError(f"could not detect CSV delimiter in {path}")
            return delimiter
