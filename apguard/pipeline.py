"""A detection run: read source data, run every detector, upsert flags.

Flags are identified by a stable key (rule|vendor|invoice|payment), so re-running
the engine never wipes reviewer work: existing flags keep their status, assignee
and history, new ones are inserted, and flags the latest run no longer
reproduces simply stop appearing in v_active_flags.
"""
from __future__ import annotations

import json
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from apguard import __version__
from apguard.db import get_engine, load_source_tables
from rules.config import load_config
from rules.engine import run_all_rules


def flag_key(df: pd.DataFrame) -> pd.Series:
    def part(c):
        return df[c].map(lambda v: "" if pd.isna(v) else str(int(v))).astype(object)
    return (df["rule_code"].astype(object) + "|" + part("vendor_id") + "|" + part("invoice_id")
            + "|" + part("payment_id"))


def run_detection(engine: Engine | None = None, config: dict | None = None, actor: str = "system") -> dict:
    eng = engine or get_engine()
    cfg = load_config(overrides=config)
    t = load_source_tables(eng)

    with eng.begin() as conn:
        run_id = conn.execute(
            text("INSERT INTO rule_runs (engine_version, config) VALUES (:v, CAST(:c AS JSONB)) RETURNING run_id"),
            {"v": __version__, "c": json.dumps(cfg)},
        ).scalar_one()

    t0 = time.perf_counter()
    flags = run_all_rules(t["vendors"], t["contracts"], t["purchase_orders"], t["goods_receipts"],
                          t["invoices"], t["payments"], config=cfg)
    runtime = time.perf_counter() - t0

    flags = flags.copy()
    flags["flag_key"] = flag_key(flags)
    flags = flags.drop_duplicates("flag_key")
    flags["run_id"] = run_id

    with eng.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS flags_staging")
        flags.to_sql("flags_staging", conn, index=False, method="multi", chunksize=2000)
        new = conn.execute(text("""
            INSERT INTO flags (flag_key, rule_code, rule_name, category, vendor_id, invoice_id, po_id,
                               payment_id, severity, amount_at_risk, details, first_seen_run_id, last_seen_run_id)
            SELECT flag_key, rule_code, rule_name, category, vendor_id, invoice_id, po_id, payment_id,
                   severity, amount_at_risk, details, run_id, run_id
            FROM flags_staging
            ON CONFLICT (flag_key) DO UPDATE SET
                severity = EXCLUDED.severity, amount_at_risk = EXCLUDED.amount_at_risk,
                details = EXCLUDED.details, last_seen_run_id = EXCLUDED.last_seen_run_id
            RETURNING (xmax = 0) AS inserted, flag_id
        """)).fetchall()
        inserted = [r.flag_id for r in new if r.inserted]
        if inserted:
            conn.execute(text("""
                INSERT INTO flag_events (flag_id, event_type, to_status, actor, note)
                SELECT flag_id, 'created', 'open', :actor, 'Raised by run ' || :run
                FROM flags WHERE flag_id = ANY(:ids)
            """), {"actor": actor, "run": str(run_id), "ids": inserted})
        retired = conn.execute(text("SELECT count(*) FROM flags WHERE last_seen_run_id < :r"), {"r": run_id}).scalar_one()
        conn.exec_driver_sql("DROP TABLE flags_staging")
        conn.execute(text("""
            UPDATE rule_runs SET finished_at = now(), flags_raised = :n, flags_new = :new,
                   flags_retired = :ret, runtime_seconds = :rt WHERE run_id = :r
        """), {"n": len(flags), "new": len(inserted), "ret": retired, "rt": round(runtime, 3), "r": run_id})

    return {"run_id": run_id, "flags_raised": len(flags), "flags_new": len(inserted),
            "flags_retired": retired, "runtime_seconds": round(runtime, 3),
            "by_rule": flags["rule_code"].value_counts().sort_index().to_dict()}
