"""Load settings from config.yaml."""

import os
from pathlib import Path

import yaml

# Project folder. Can be overridden (used inside Docker).
PROJECT_ROOT = Path(os.environ.get("CHURN_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from the config into an absolute path inside the project."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
