"""APGuard REST API.

Read endpoints are open. Write endpoints (review actions, triggering a run)
require an `X-API-Key` header matching APGUARD_API_KEY; if that variable is not
set, writes are disabled entirely rather than left unauthenticated.
"""
from __future__ import annotations

import contextlib
import hmac
import io
import json
import math
import os
import sys
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard import __version__, metrics  # noqa: E402
from apguard.cases import STATUSES, WorkflowError, update_flag  # noqa: E402
from apguard.db import get_engine, read_sql  # noqa: E402
from apguard.evidence import evidence_for_flag  # noqa: E402
from rules.registry import RULES, SEVERITIES  # noqa: E402

app = FastAPI(
    title="APGuard API",
    version=__version__,
    description="Accounts payable leakage detection: flags, evidence, review workflow and vendor risk.",
)
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "PATCH", "POST"],
                   allow_headers=["Content-Type", "X-API-Key"])

RuleCode = Enum("RuleCode", {c: c for c in RULES}, type=str)
Severity = Enum("Severity", {s: s for s in SEVERITIES}, type=str)
Status = Enum("Status", {s: s for s in STATUSES}, type=str)


class Category(StrEnum):
    control = "control"
    analytic = "analytic"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class Flag(BaseModel):
    flag_id: int
    rule_code: str
    rule_name: str
    category: str
    vendor_id: Optional[int] = None
    vendor_name: Optional[str] = None
    invoice_id: Optional[int] = None
    po_id: Optional[int] = None
    payment_id: Optional[int] = None
    severity: str
    amount_at_risk: float
    details: str
    status: str
    assignee: Optional[str] = None
    recovered_amount: float = 0.0
    first_seen_run_id: int
    last_seen_run_id: int
    updated_at: datetime


class FlagPage(BaseModel):
    total: int = Field(description="Flags matching the filters, across all pages")
    limit: int
    offset: int
    items: list[Flag]


class FlagUpdate(BaseModel):
    status: Optional[Status] = None
    assignee: Optional[str] = Field(None, max_length=80)
    note: Optional[str] = Field(None, max_length=2000)
    recovered_amount: Optional[float] = Field(None, gt=0)
    actor: str = Field("api", max_length=80)


class RuleSummary(BaseModel):
    rule_code: str
    rule_name: str
    category: str
    flag_count: int
    total_at_risk: float
    confirmed: int
    false_positive: int
    recovered: float


class Summary(BaseModel):
    active_flags: int
    control_flags: int
    vendors_flagged: int
    gross_at_risk: float = Field(description="Sum of every control flag; double-counts invoices hit by several rules")
    net_exposure: float = Field(description="Largest control flag per invoice, excluding dismissed flags")
    vendor_level_exposure: float = Field(description="Exposure from vendor-level controls (R10)")
    recovered: float
    open: int
    in_review: int
    confirmed: int
    false_positive: int
    recovered_count: int
    reviewed: int
    observed_precision: Optional[float] = Field(description="Confirmed / reviewed, from reviewer decisions")
    by_rule: list[RuleSummary]


class VendorRisk(BaseModel):
    vendor_id: int
    vendor_name: str
    risk_category: str
    flag_count: int
    rules: str
    invoiced: float
    net_exposure: float
    exposure_share: float
    max_severity: str
    integrity_flags: int
    risk_score: float


class Run(BaseModel):
    run_id: int
    started_at: datetime
    finished_at: Optional[datetime]
    engine_version: str
    flags_raised: Optional[int]
    flags_new: Optional[int]
    flags_retired: Optional[int]
    runtime_seconds: Optional[float]
    config: dict[str, Any]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean(value):
    """Make pandas/numpy values JSON-safe (NaN/NaT -> None, numpy scalars -> python)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        with contextlib.suppress(ValueError, AttributeError):
            value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value.__class__.__name__ in ("NaTType", "NAType"):
        return None
    if isinstance(value, (datetime, date)):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _records(df) -> list[dict]:
    return [_clean(r) for r in df.to_dict(orient="records")]


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected = os.getenv("APGUARD_API_KEY")
    if not expected:
        raise HTTPException(503, "Write endpoints are disabled: APGUARD_API_KEY is not configured.")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


def _filters(rule_code, severity, status, category, vendor_id, min_amount):
    where, params = ["1=1"], {}
    if rule_code:
        where.append("rule_code = :rule_code")
        params["rule_code"] = rule_code.value
    if severity:
        where.append("severity = :severity")
        params["severity"] = severity.value
    if status:
        where.append("status = :status")
        params["status"] = status.value
    if category:
        where.append("category = :category")
        params["category"] = category.value
    if vendor_id is not None:
        where.append("vendor_id = :vendor_id")
        params["vendor_id"] = vendor_id
    if min_amount:
        where.append("amount_at_risk >= :min_amount")
        params["min_amount"] = min_amount
    return " AND ".join(where), params


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root():
    return {"service": "APGuard API", "version": __version__, "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
            run = conn.execute(text("SELECT max(run_id) FROM rule_runs")).scalar()
        return {"status": "ok", "database": "ok", "latest_run": run}
    except Exception as exc:  # pragma: no cover - exercised manually
        raise HTTPException(503, f"database unavailable: {exc.__class__.__name__}") from None


@app.get("/rules", tags=["rules"])
def list_rules():
    """Every detector, the control it tests and the assertion at risk."""
    return [s.to_dict() for s in RULES.values()]


@app.get("/flags", response_model=FlagPage, tags=["flags"])
def list_flags(
    rule_code: Optional[RuleCode] = None,
    severity: Optional[Severity] = None,
    status: Optional[Status] = None,
    category: Optional[Category] = None,
    vendor_id: Optional[int] = Query(None, ge=1),
    min_amount: float = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Active flags (raised by the latest run), largest exposure first."""
    where, params = _filters(rule_code, severity, status, category, vendor_id, min_amount)
    total = int(read_sql(f"SELECT count(*) AS n FROM v_active_flags WHERE {where}", params).iloc[0]["n"])
    df = read_sql(f"SELECT * FROM v_active_flags WHERE {where} ORDER BY amount_at_risk DESC, flag_id "
                  f"LIMIT :limit OFFSET :offset", {**params, "limit": limit, "offset": offset})
    return {"total": total, "limit": limit, "offset": offset, "items": _records(df)}


@app.get("/flags/summary", response_model=Summary, tags=["flags"])
def flags_summary():
    s = metrics.summary()
    s.pop("last_run", None)
    return {**s, "by_rule": _records(metrics.by_rule())}


@app.get("/flags/export.csv", tags=["flags"])
def export_flags_csv(
    rule_code: Optional[RuleCode] = None,
    severity: Optional[Severity] = None,
    status: Optional[Status] = None,
    category: Optional[Category] = None,
    vendor_id: Optional[int] = Query(None, ge=1),
    min_amount: float = Query(0, ge=0),
):
    where, params = _filters(rule_code, severity, status, category, vendor_id, min_amount)
    df = read_sql(f"SELECT * FROM v_active_flags WHERE {where} ORDER BY amount_at_risk DESC", params)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=apguard_flags.csv"})


@app.get("/flags/{flag_id}", tags=["flags"])
def get_flag(flag_id: int):
    """A flag with its full evidence chain and audit trail."""
    ev = evidence_for_flag(flag_id)
    if ev is None:
        raise HTTPException(404, "Flag not found")
    return _clean(ev)


@app.patch("/flags/{flag_id}", response_model=Flag, tags=["review"], dependencies=[Depends(require_api_key)])
def review_flag(flag_id: int, body: FlagUpdate):
    """Move a flag through the review workflow:
    open -> in_review -> confirmed -> recovered, or -> false_positive (note required)."""
    try:
        update_flag(flag_id, status=body.status.value if body.status else None, assignee=body.assignee,
                    note=body.note, recovered_amount=body.recovered_amount, actor=body.actor)
    except LookupError:
        raise HTTPException(404, "Flag not found") from None
    except WorkflowError as exc:
        raise HTTPException(422, str(exc)) from None
    df = read_sql("SELECT f.*, v.vendor_name FROM flags f LEFT JOIN vendors v USING (vendor_id) "
                  "WHERE flag_id = :id", {"id": flag_id})
    return _records(df)[0]


@app.get("/vendors/risk-ranking", response_model=list[VendorRisk], tags=["vendors"])
def vendor_risk_ranking(limit: int = Query(25, ge=1, le=500)):
    """Composite 0-100 risk score: 40% exposure as a share of spend, 30% net exposure,
    30% vendor-integrity signal (R10 / R11 / A1)."""
    return _records(metrics.vendor_risk(limit=limit))


@app.get("/vendors/{vendor_id}", tags=["vendors"])
def vendor_drilldown(vendor_id: int):
    vendor = read_sql("SELECT * FROM vendors WHERE vendor_id = :v", {"v": vendor_id})
    if vendor.empty:
        raise HTTPException(404, "Vendor not found")
    flags = read_sql("SELECT * FROM v_active_flags WHERE vendor_id = :v ORDER BY amount_at_risk DESC", {"v": vendor_id})
    spend = read_sql("""SELECT count(*) AS invoices, COALESCE(sum(total_amount), 0) AS invoiced
                        FROM invoices WHERE vendor_id = :v""", {"v": vendor_id}).iloc[0]
    exposure = read_sql("SELECT COALESCE(sum(exposure), 0) AS e FROM v_invoice_exposure WHERE vendor_id = :v",
                        {"v": vendor_id}).iloc[0]["e"]
    return {
        "vendor": _records(vendor)[0],
        "invoices": int(spend["invoices"]),
        "invoiced": float(spend["invoiced"]),
        "flag_count": len(flags),
        "net_exposure": float(exposure),
        "flags": _records(flags),
    }


@app.get("/runs", response_model=list[Run], tags=["runs"])
def list_runs(limit: int = Query(20, ge=1, le=200)):
    df = read_sql("SELECT * FROM rule_runs ORDER BY run_id DESC LIMIT :l", {"l": limit})
    return _records(df)


@app.post("/runs", response_model=Run, tags=["runs"], dependencies=[Depends(require_api_key)])
def trigger_run():
    """Re-run every detector. Reviewer decisions on existing flags are preserved."""
    from apguard.pipeline import run_detection
    r = run_detection(actor="api")
    return _records(read_sql("SELECT * FROM rule_runs WHERE run_id = :r", {"r": r["run_id"]}))[0]


@app.get("/evaluation", tags=["rules"])
def evaluation():
    """Latest offline evaluation (precision/recall against injected ground truth)."""
    p = ROOT / "reports" / "evaluation.json"
    if not p.exists():
        raise HTTPException(404, "No evaluation report. Run scripts/evaluate.py.")
    data = json.loads(p.read_text())
    return {k: data[k] for k in ("generated_at", "dataset", "scores", "dollar_recall", "baseline_comparison")}
