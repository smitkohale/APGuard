import os
from dotenv import load_dotenv

load_dotenv()
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
from sqlalchemy import create_engine
import io

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

app = FastAPI(title="APGuard API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _clean(records):
    """Convert NaN/NaT to None so FastAPI's JSON encoder doesn't choke.
    Must build plain python dicts first -- reassigning a NaN-replaced column
    back onto a DataFrame lets pandas re-infer dtype and silently flips None
    back to NaN, which is a real, easy-to-hit pandas gotcha here."""
    clean_records = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                clean[k] = None
            else:
                clean[k] = v
        clean_records.append(clean)
    return clean_records


@app.get("/")
def root():
    return {"service": "APGuard API", "status": "ok"}


@app.get("/flags")
def get_flags(
    rule_code: Optional[str] = None,
    vendor_id: Optional[int] = None,
    severity: Optional[str] = None,
    limit: int = Query(100, le=5000),
    offset: int = 0,
):
    query = "SELECT * FROM flags WHERE 1=1"
    params = {}
    if rule_code:
        query += " AND rule_code = %(rule_code)s"
        params["rule_code"] = rule_code
    if vendor_id:
        query += " AND vendor_id = %(vendor_id)s"
        params["vendor_id"] = vendor_id
    if severity:
        query += " AND severity = %(severity)s"
        params["severity"] = severity
    query += " ORDER BY amount_at_risk DESC LIMIT %(limit)s OFFSET %(offset)s"
    params["limit"] = limit
    params["offset"] = offset

    df = pd.read_sql(query, engine, params=params)
    return {"count": len(df), "flags": _clean(df.to_dict(orient="records"))}


@app.get("/flags/summary")
def flags_summary():
    by_rule = pd.read_sql(
        "SELECT rule_code, rule_name, count(*) AS flag_count, "
        "COALESCE(sum(amount_at_risk),0) AS total_at_risk "
        "FROM flags GROUP BY rule_code, rule_name ORDER BY rule_code", engine,
    )
    by_severity = pd.read_sql(
        "SELECT severity, count(*) AS flag_count, COALESCE(sum(amount_at_risk),0) AS total_at_risk "
        "FROM flags GROUP BY severity ORDER BY total_at_risk DESC", engine,
    )
    totals = pd.read_sql(
        "SELECT count(*) AS total_flags, COALESCE(sum(amount_at_risk),0) AS total_at_risk FROM flags", engine,
    )
    return {
        "totals": _clean(totals.to_dict(orient="records"))[0],
        "by_rule": _clean(by_rule.to_dict(orient="records")),
        "by_severity": _clean(by_severity.to_dict(orient="records")),
    }


@app.get("/vendors/risk-ranking")
def vendor_risk_ranking(limit: int = Query(25, le=200)):
    df = pd.read_sql(
        """
        SELECT v.vendor_id, v.vendor_name, v.risk_category,
               count(f.flag_id) AS flag_count,
               COALESCE(sum(f.amount_at_risk), 0) AS total_at_risk
        FROM vendors v
        LEFT JOIN flags f ON f.vendor_id = v.vendor_id
        GROUP BY v.vendor_id, v.vendor_name, v.risk_category
        ORDER BY total_at_risk DESC
        LIMIT %(limit)s
        """,
        engine, params={"limit": limit},
    )
    return {"count": len(df), "vendors": _clean(df.to_dict(orient="records"))}


@app.get("/vendors/{vendor_id}")
def vendor_drilldown(vendor_id: int):
    vendor = pd.read_sql("SELECT * FROM vendors WHERE vendor_id = %(vid)s", engine, params={"vid": vendor_id})
    if vendor.empty:
        raise HTTPException(status_code=404, detail="Vendor not found")

    flags = pd.read_sql(
        "SELECT * FROM flags WHERE vendor_id = %(vid)s ORDER BY amount_at_risk DESC", engine, params={"vid": vendor_id},
    )
    invoices = pd.read_sql(
        "SELECT * FROM invoices WHERE vendor_id = %(vid)s ORDER BY invoice_date DESC LIMIT 100",
        engine, params={"vid": vendor_id},
    )
    return {
        "vendor": _clean(vendor.to_dict(orient="records"))[0],
        "flag_count": len(flags),
        "total_at_risk": float(flags["amount_at_risk"].sum()) if not flags.empty else 0.0,
        "flags": _clean(flags.to_dict(orient="records")),
        "recent_invoices": _clean(invoices.to_dict(orient="records")),
    }


@app.get("/flags/export.csv")
def export_flags_csv(rule_code: Optional[str] = None, vendor_id: Optional[int] = None):
    query = "SELECT * FROM flags WHERE 1=1"
    params = {}
    if rule_code:
        query += " AND rule_code = %(rule_code)s"
        params["rule_code"] = rule_code
    if vendor_id:
        query += " AND vendor_id = %(vendor_id)s"
        params["vendor_id"] = vendor_id
    df = pd.read_sql(query, engine, params=params)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=apguard_flags.csv"},
    )
