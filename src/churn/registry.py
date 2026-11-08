"""A simple file-based model registry.

    models/
      registry.json          which versions exist and which one is the champion
      v1/model.joblib        the pipeline (SMOTE + scaler + model)
      v1/metadata.json       parameters, threshold, metrics, training data
      v1/reference.csv       the training features, used to detect data drift

The champion is the model currently used for scoring and by the API.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


class NoChampionError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "registry.json"

    # -- internal ---------------------------------------------------------
    def _read(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"champion": None, "versions": []}

    def _write(self, index: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(index, indent=2))

    def _dir(self, version: int) -> Path:
        return self.root / f"v{version}"

    # -- public -----------------------------------------------------------
    def register(self, model, metadata: dict, reference: pd.DataFrame) -> int:
        index = self._read()
        version = len(index["versions"]) + 1
        folder = self._dir(version)
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {**metadata, "version": version}
        joblib.dump(model, folder / "model.joblib")
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        reference.to_csv(folder / "reference.csv", index=False)
        index["versions"].append({
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_pr_auc": metadata.get("test", {}).get("pr_auc"),
            "reason": metadata.get("reason", ""),
        })
        self._write(index)
        return version

    def promote(self, version: int) -> None:
        index = self._read()
        if version not in [v["version"] for v in index["versions"]]:
            raise ValueError(f"Version {version} is not registered")
        index["champion"] = version
        self._write(index)

    def champion_version(self) -> int | None:
        return self._read()["champion"]

    def versions(self) -> list[dict]:
        index = self._read()
        return [{**v, "champion": v["version"] == index["champion"]} for v in index["versions"]]

    def load(self, version: int | None = None):
        """Return (model, metadata). Defaults to the champion."""
        version = version or self.champion_version()
        if version is None:
            raise NoChampionError("No champion model yet. Run `make simulate` or `make train` first.")
        folder = self._dir(version)
        model = joblib.load(folder / "model.joblib")
        metadata = json.loads((folder / "metadata.json").read_text())
        return model, metadata

    def load_reference(self, version: int | None = None) -> pd.DataFrame:
        version = version or self.champion_version()
        return pd.read_csv(self._dir(version) / "reference.csv")
