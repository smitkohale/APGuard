"""Loads the generated CSVs into Postgres, in FK-safe order."""
import os
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

DB_URL = os.getenv("DATABASE_URL")
DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

engine = create_engine(DB_URL)

TABLES_IN_ORDER = [
    ("vendors.csv", "vendors"),
    ("contracts.csv", "contracts"),
    ("purchase_orders.csv", "purchase_orders"),
    ("goods_receipts.csv", "goods_receipts"),
    ("invoices.csv", "invoices"),
    ("payments.csv", "payments"),
]

with engine.begin() as conn:
    # clear existing rows, respecting FK order (children first)
    for _, table in reversed(TABLES_IN_ORDER):
        conn.exec_driver_sql(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

for csv_name, table in TABLES_IN_ORDER:
    df = pd.read_csv(DATA_DIR / csv_name)
    # nullable integer FK columns come back as float when there are NaNs; fix before insert
    for col in df.columns:
        if col.endswith("_id") and df[col].isna().any():
            df[col] = df[col].astype("Int64")
    df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=2000)
    print(f"loaded {len(df):,} rows into {table}")

print("ETL complete.")
