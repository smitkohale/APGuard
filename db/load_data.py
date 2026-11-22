"""Load the generated CSVs into PostgreSQL in foreign-key order.

    python db/load_data.py [--data data/generated]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard.db import SOURCE_TABLES, get_engine  # noqa: E402

DATE_COLUMNS = {"vendors": ["created_at"], "contracts": ["start_date", "end_date"], "purchase_orders": ["po_date"],
                "goods_receipts": ["received_date"], "invoices": ["invoice_date"], "payments": ["payment_date"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    data_dir = Path(ap.parse_args().data)

    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE flag_events, flags, rule_runs, " + ", ".join(reversed(SOURCE_TABLES))
                             + " RESTART IDENTITY CASCADE")
        for table in SOURCE_TABLES:
            df = pd.read_csv(data_dir / f"{table}.csv")
            for col in df.columns:
                if col.endswith("_id") and df[col].isna().any():
                    df[col] = df[col].astype("Int64")
            for col in DATE_COLUMNS.get(table, []):
                df[col] = pd.to_datetime(df[col]).dt.date
            df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=2000)
            print(f"loaded {len(df):>7,} rows into {table}")
    print("ETL complete.")


if __name__ == "__main__":
    main()
