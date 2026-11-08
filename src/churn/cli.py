"""Command line entry point.

    python -m churn.cli simulate                  full end-to-end run (setup, train, 6 months)
    python -m churn.cli setup                     fresh database with historical data only
    python -m churn.cli train [--promote]         train on all labelled data in the database
    python -m churn.cli run-month 3               run one monthly cycle
    python -m churn.cli score --input f.csv --output out.csv
    python -m churn.cli status                    registry and monitoring overview

The Makefile wraps these: `make simulate`, `make status`, ...
"""

import argparse

import pandas as pd

from churn import db
from churn.config import DEFAULT_CONFIG, load_config, resolve
from churn.cycle import run_month, train_and_register
from churn.predict import score
from churn.registry import ModelRegistry
from churn.simulation import run_simulation, setup


def main() -> None:
    parser = argparse.ArgumentParser(prog="churn", description="Customer churn MLOps pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("simulate", help="run the whole pipeline end to end")
    sub.add_parser("setup", help="fresh database with historical data")
    p_train = sub.add_parser("train", help="train on all labelled data")
    p_train.add_argument("--promote", action="store_true", help="make the new model the champion")
    p_month = sub.add_parser("run-month", help="run one monthly cycle")
    p_month.add_argument("month", type=int)
    p_score = sub.add_parser("score", help="score a CSV of customers with the champion")
    p_score.add_argument("--input", required=True)
    p_score.add_argument("--output", required=True)
    sub.add_parser("status", help="show models and monitoring results")
    args = parser.parse_args()

    config = load_config(args.config)
    registry = ModelRegistry(resolve(config["registry"]["dir"]))

    if args.command == "simulate":
        run_simulation(config)

    elif args.command == "setup":
        setup(config).close()

    elif args.command == "train":
        conn = db.connect(resolve(config["database"]["path"]))
        data = db.load_customers(conn, labelled_only=True)
        if data.empty:
            raise SystemExit("No labelled data in the database. Run `make setup` first.")
        print(f"Training on {len(data)} labelled customers (with hyperparameter tuning)...")
        _, meta = train_and_register(conn, registry, config, data, reason="manual training")
        if args.promote or registry.champion_version() is None:
            registry.promote(meta["version"])
            db.log_event(conn, "promoted", meta["version"], {"reason": "manual"})
        t = meta["test"]
        print(f"v{meta['version']}: test ROC-AUC {t['roc_auc']} | PR-AUC {t['pr_auc']} | "
              f"precision {t['precision']} | recall {t['recall']} | threshold {meta['threshold']}")
        print(f"Champion is v{registry.champion_version()}")
        conn.close()

    elif args.command == "run-month":
        conn = db.connect(resolve(config["database"]["path"]))
        row = run_month(args.month, conn, registry, config)
        print(pd.Series(row).to_string())
        conn.close()

    elif args.command == "score":
        model, meta = registry.load()
        scores = score(pd.read_csv(args.input), model, meta["threshold"])
        scores.to_csv(args.output, index=False)
        print(f"Scored {len(scores)} customers with v{meta['version']}; "
              f"{scores['targeted'].sum()} targeted for a retention offer. Saved to {args.output}")

    elif args.command == "status":
        versions = pd.DataFrame(registry.versions())
        print("Models:\n" + (versions.to_string(index=False) if not versions.empty else "  none yet"))
        path = resolve(config["database"]["path"])
        if path.exists():
            conn = db.connect(path)
            mon = db.read_table(conn, "monitoring")
            cols = ["month", "model_version", "pr_auc", "drift_share", "campaign_profit", "action"]
            print("\nMonitoring:\n" + (mon[cols].to_string(index=False) if not mon.empty else "  no months run yet"))
            conn.close()


if __name__ == "__main__":
    main()
