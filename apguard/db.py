"""Database engine and table loading."""
from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

SOURCE_TABLES = ["vendors", "contracts", "purchase_orders", "goods_receipts", "invoices", "payments"]


def database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and point it at PostgreSQL.")
    return normalize_url(url)


def normalize_url(url: str) -> str:
    """Hosting providers (Render, Neon, Supabase, Heroku) hand out postgres:// or
    postgresql:// URLs; SQLAlchemy needs the driver named explicitly."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True)


def read_sql(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    from sqlalchemy import text
    with (engine or get_engine()).connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def load_source_tables(engine: Engine | None = None) -> dict[str, pd.DataFrame]:
    """Read every source table. NUMERIC columns arrive as floats, dates as datetimes."""
    eng = engine or get_engine()
    tables = {t: pd.read_sql(f"SELECT * FROM {t}", eng, coerce_float=True) for t in SOURCE_TABLES}
    tables["invoices"]["po_id"] = tables["invoices"]["po_id"].astype("Int64")
    tables["purchase_orders"]["contract_id"] = tables["purchase_orders"]["contract_id"].astype("Int64")
    return tables
