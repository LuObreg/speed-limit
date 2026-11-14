"""Type T persistence for trained model bundles and training diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

LOG = logging.getLogger("speedlimit.model_artifacts")

MODEL_FILE = "speedlimit_nn.joblib"
HISTORY_FILE = "loss_history.json"
PLOT_FILE = "loss_curves.png"


class FilesystemModelArtifactStore:
    """Persist model artifacts under one configured directory."""

    def __init__(self, model_dir: Path) -> None:
        self._directory = Path(model_dir)

    def save(self, bundle: dict[str, Any], history: dict[str, list]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self._directory / MODEL_FILE)
        history_path = self._directory / HISTORY_FILE
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self._plot_loss_curves(history, self._directory / PLOT_FILE)

    def load(self) -> dict[str, Any]:
        path = self._directory / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(path)
        return joblib.load(path)

    def save_report(self, filename: str, report: dict[str, Any]) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / filename
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _plot_loss_curves(history: dict[str, list], path: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            LOG.warning("matplotlib not installed - loss curves only written as JSON")
            return
        epochs = range(1, len(history["train_mae"]) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["train_mae"], label="training MAE")
        ax.plot(epochs, history["validation_mae"], label="validation MAE")
        ax.axvline(history["best_epoch"][0], color="grey", linestyle="--", label="best epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Near-accidents/hour MAE")
        ax.set_title("SpeedLimitNN - training vs. validation MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
