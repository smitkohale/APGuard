"""Portfolio-level metrics shared by the API and the dashboard."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from apguard.db import read_sql

SEVERITY_WEIGHT = {"critical": 8, "high": 4, "medium": 2, "low": 1}
INTEGRITY_RULES = ["R10", "R11", "A1"]


def summary(engine: Engine | None = None) -> dict:
    s = read_sql("""
        SELECT count(*) AS active_flags,
               count(*) FILTER (WHERE category = 'control') AS control_flags,
               COALESCE(sum(amount_at_risk) FILTER (WHERE category = 'control'), 0) AS gross_at_risk,
               COALESCE(sum(recovered_amount), 0) AS recovered,
               count(*) FILTER (WHERE status = 'open') AS open,
               count(*) FILTER (WHERE status = 'in_review') AS in_review,
               count(*) FILTER (WHERE status = 'confirmed') AS confirmed,
               count(*) FILTER (WHERE status = 'false_positive') AS false_positive,
               count(*) FILTER (WHERE status = 'recovered') AS recovered_count,
               count(DISTINCT vendor_id) AS vendors_flagged
        FROM v_active_flags
    """, engine=engine).iloc[0].to_dict()
    net = read_sql("SELECT COALESCE(sum(exposure), 0) AS net FROM v_invoice_exposure", engine=engine).iloc[0]["net"]
    vendor_level = read_sql("""SELECT COALESCE(sum(amount_at_risk), 0) AS v FROM v_active_flags
                               WHERE category = 'control' AND invoice_id IS NULL AND status <> 'false_positive'""",
                            engine=engine).iloc[0]["v"]
    run = read_sql("SELECT * FROM rule_runs ORDER BY run_id DESC LIMIT 1", engine=engine)
    money = {"gross_at_risk", "recovered"}
    out = {k: (float(v) if k in money else int(v)) for k, v in s.items()}
    reviewed = out["confirmed"] + out["recovered_count"] + out["false_positive"]
    out.update({
        "net_exposure": float(net),
        "vendor_level_exposure": float(vendor_level),
        "reviewed": reviewed,
        "observed_precision": (out["confirmed"] + out["recovered_count"]) / reviewed if reviewed else None,
        "last_run": run.iloc[0].to_dict() if not run.empty else None,
    })
    return out


def by_rule(engine: Engine | None = None) -> pd.DataFrame:
    """Per-rule counts, exposure and reviewer-observed precision."""
    return read_sql("""
        SELECT rule_code, rule_name, category,
               count(*) AS flag_count,
               COALESCE(sum(amount_at_risk), 0) AS total_at_risk,
               count(*) FILTER (WHERE status IN ('confirmed', 'recovered')) AS confirmed,
               count(*) FILTER (WHERE status = 'false_positive') AS false_positive,
               COALESCE(sum(recovered_amount), 0) AS recovered
        FROM v_active_flags
        GROUP BY rule_code, rule_name, category
        ORDER BY length(rule_code), rule_code
    """, engine=engine)


def vendor_risk(engine: Engine | None = None, limit: int | None = None) -> pd.DataFrame:
    """Composite vendor risk score, 0-100:
        40%  exposure as a share of the vendor's invoiced spend (percentile rank)
        30%  net exposure in dollars (percentile rank)
        30%  vendor-integrity signal: the vendor itself looks wrong (R10 cloned
             record, R11 paid days after creation, A1 Benford nonconformity), as
             opposed to individual transactions going wrong
    Counting flags or taking the worst severity just ranks the biggest suppliers
    first, because transaction errors scale with volume. Concentration and
    integrity signals surface the small vendor whose every payment is suspect."""
    df = read_sql("""
        SELECT v.vendor_id, v.vendor_name, v.risk_category, f.rule_code, f.severity, f.amount_at_risk,
               f.invoice_id, f.category
        FROM v_active_flags f JOIN vendors v ON v.vendor_id = f.vendor_id
        WHERE f.status <> 'false_positive'
    """, engine=engine)
    cols = ["vendor_id", "vendor_name", "risk_category", "flag_count", "rules", "invoiced", "net_exposure",
            "exposure_share", "max_severity", "integrity_flags", "risk_score"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    spend = read_sql("SELECT vendor_id, sum(total_amount) AS invoiced FROM invoices GROUP BY vendor_id",
                     engine=engine).set_index("vendor_id")["invoiced"]
    df["points"] = df["severity"].map(SEVERITY_WEIGHT).fillna(1)
    ctrl = df[df["category"] == "control"]
    inv_exp = ctrl.dropna(subset=["invoice_id"]).groupby(["vendor_id", "invoice_id"])["amount_at_risk"].max() \
        .groupby("vendor_id").sum()
    ven_exp = ctrl[ctrl["invoice_id"].isna()].groupby("vendor_id")["amount_at_risk"].sum()
    g = df.groupby(["vendor_id", "vendor_name", "risk_category"]).agg(
        flag_count=("rule_code", "size"),
        rules=("rule_code", lambda s: ", ".join(sorted(set(s), key=lambda c: (c[0], int(c[1:]))))),
        n_rules=("rule_code", "nunique"), max_points=("points", "max")).reset_index()
    g["invoiced"] = g["vendor_id"].map(spend).fillna(0.0)
    # vendor-level exposure (everything paid to a cloned record) can't exceed spend
    g["net_exposure"] = g["vendor_id"].map(inv_exp).fillna(0.0)
    g["net_exposure"] = (g["net_exposure"] + g["vendor_id"].map(ven_exp).fillna(0.0)).clip(upper=g["invoiced"])
    g["exposure_share"] = (g["net_exposure"] / g["invoiced"].where(g["invoiced"] > 0)).fillna(0.0).clip(upper=1)
    inv_sev = {v: k for k, v in SEVERITY_WEIGHT.items()}
    g["max_severity"] = g["max_points"].map(inv_sev)
    integrity = df[df["rule_code"].isin(INTEGRITY_RULES)].groupby("vendor_id")["rule_code"].nunique()
    g["integrity_flags"] = g["vendor_id"].map(integrity).fillna(0).astype(int)
    g["risk_score"] = (40 * g["exposure_share"].rank(pct=True)
                       + 30 * g["net_exposure"].rank(pct=True)
                       + 30 * (g["integrity_flags"] > 0)).round(1)
    g = g.sort_values(["risk_score", "net_exposure"], ascending=False)
    return (g.head(limit) if limit else g)[cols]
