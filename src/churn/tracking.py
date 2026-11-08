"""Optional experiment tracking with MLflow.

If MLflow is installed and tracking is enabled in config.yaml, every training
run and every monthly cycle is logged. If not, the project still works.
View runs with `make mlflow`, then open http://localhost:5000
"""

from pathlib import Path

from churn.config import resolve

_warned = False


def _tracking_uri(uri: str) -> str:
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + str(resolve(uri[len(prefix):]))
    return uri


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_run(config: dict, run_name: str, params: dict | None = None, metrics: dict | None = None,
            artifacts: list | None = None, tags: dict | None = None) -> str | None:
    """Log one run. Returns the MLflow run id, or None if tracking is off/unavailable."""
    global _warned
    cfg = config.get("tracking", {})
    if not cfg.get("enabled", False):
        return None
    try:
        import mlflow
    except ImportError:
        if not _warned:
            print("(MLflow not installed, so runs are not tracked. Install it with `make install-mlflow`.)")
            _warned = True
        return None

    try:
        mlflow.set_tracking_uri(_tracking_uri(cfg.get("uri", "sqlite:///mlflow.db")))
        mlflow.set_experiment(cfg.get("experiment", "churn"))
        numeric = {k: float(v) for k, v in _flatten(metrics or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None}
        with mlflow.start_run(run_name=run_name) as run:
            if params:
                mlflow.log_params({k: str(v) for k, v in _flatten(params).items()})
            if numeric:
                mlflow.log_metrics(numeric)
            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            for path in artifacts or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:  # tracking must never break the pipeline
        print(f"(MLflow logging skipped: {exc})")
        return None
