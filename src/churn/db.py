"""SQLite storage for customers, predictions, monitoring results and model events.

SQLite needs no server, so the project runs anywhere. The SQL is standard,
so switching to MySQL later means changing the connection, not the logic.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churn.data import FEATURES


def _col_type(col: str) -> str:
    return "REAL" if col == "customer_value" else "INTEGER"


SCHEMA = (
    "CREATE TABLE IF NOT EXISTS customers (\n"
    "  customer_id TEXT PRIMARY KEY,\n"
    "  month INTEGER NOT NULL,\n"
    + "".join(f"  {c} {_col_type(c)} NOT NULL,\n" for c in FEATURES)
    + "  churn INTEGER  -- NULL until we learn whether the customer left\n);\n"
    """
CREATE TABLE IF NOT EXISTS predictions (
  customer_id TEXT NOT NULL,
  month INTEGER NOT NULL,
  model_version INTEGER NOT NULL,
  churn_probability REAL NOT NULL,
  targeted INTEGER NOT NULL,
  scored_at TEXT NOT NULL,
  PRIMARY KEY (customer_id, model_version)
);

CREATE TABLE IF NOT EXISTS monitoring (
  month INTEGER PRIMARY KEY,
  model_version INTEGER,
  n_customers INTEGER,
  churn_rate REAL,
  drift_share REAL,
  data_drift INTEGER,
  roc_auc REAL,
  pr_auc REAL,
  precision REAL,
  recall REAL,
  n_targeted INTEGER,
  campaign_profit REAL,
  random_profit REAL,
  action TEXT,
  report_path TEXT
);

CREATE TABLE IF NOT EXISTS model_events (
  happened_at TEXT NOT NULL,
  event TEXT NOT NULL,
  version INTEGER,
  details TEXT
);
""")

CUSTOMER_COLUMNS = ["customer_id", "month", *FEATURES, "churn"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _py(value):
    """Convert numpy/pandas scalars to plain Python values for sqlite3."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _rows(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    return [tuple(_py(v) for v in row) for row in df[cols].itertuples(index=False)]


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def insert_customers(conn, df: pd.DataFrame, month: int, include_labels: bool) -> int:
    """Store a batch of customers. Without labels, churn is stored as NULL."""
    df = df.copy()
    df["month"] = month
    if not include_labels or "churn" not in df.columns:
        df["churn"] = None
    placeholders = ", ".join("?" * len(CUSTOMER_COLUMNS))
    conn.executemany(
        f"INSERT OR REPLACE INTO customers ({', '.join(CUSTOMER_COLUMNS)}) VALUES ({placeholders})",
        _rows(df, CUSTOMER_COLUMNS))
    conn.commit()
    return len(df)


def add_labels(conn, labels: pd.DataFrame) -> None:
    """Record whether customers churned, once that is known."""
    conn.executemany("UPDATE customers SET churn = ? WHERE customer_id = ?",
                     _rows(labels, ["churn", "customer_id"]))
    conn.commit()


def load_customers(conn, months: list[int] | None = None, labelled_only: bool = False) -> pd.DataFrame:
    query, params, where = "SELECT * FROM customers", [], []
    if months is not None:
        where.append(f"month IN ({', '.join('?' * len(months))})")
        params += list(months)
    if labelled_only:
        where.append("churn IS NOT NULL")
    if where:
        query += " WHERE " + " AND ".join(where)
    df = pd.read_sql_query(query + " ORDER BY customer_id", conn, params=params)
    if labelled_only:
        df["churn"] = df["churn"].astype(int)
    return df


def save_predictions(conn, df: pd.DataFrame) -> None:
    df = df.assign(scored_at=_now())
    cols = ["customer_id", "month", "model_version", "churn_probability", "targeted", "scored_at"]
    conn.executemany(f"INSERT OR REPLACE INTO predictions ({', '.join(cols)}) VALUES (?, ?, ?, ?, ?, ?)",
                     _rows(df, cols))
    conn.commit()


def save_monitoring(conn, row: dict) -> None:
    cols = list(row)
    conn.execute(f"INSERT OR REPLACE INTO monitoring ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                 [_py(row[c]) for c in cols])
    conn.commit()


def log_event(conn, event: str, version: int | None = None, details: dict | None = None) -> None:
    conn.execute("INSERT INTO model_events VALUES (?, ?, ?, ?)",
                 (_now(), event, version, json.dumps(details or {})))
    conn.commit()


def read_table(conn, name: str) -> pd.DataFrame:
    if name not in {"customers", "predictions", "monitoring", "model_events"}:
        raise ValueError(f"Unknown table: {name}")
    return pd.read_sql_query(f"SELECT * FROM {name}", conn)
