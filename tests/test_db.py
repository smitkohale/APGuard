"""Integration tests against a real PostgreSQL database.

They build their own database from scratch, so they need a disposable one:

    APGUARD_TEST_DATABASE_URL=postgresql+psycopg2://postgres:pw@localhost:5432/apguard_test pytest -m db

Skipped when the variable is not set. Never point it at a database you care about.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_URL = os.getenv("APGUARD_TEST_DATABASE_URL")
pytestmark = [pytest.mark.db, pytest.mark.skipif(not TEST_URL, reason="APGUARD_TEST_DATABASE_URL not set")]


@pytest.fixture(scope="module")
def engine():
    if not (ROOT / "data" / "generated" / "invoices.csv").exists():
        pytest.skip("generated data not present: run python data/generate_data.py")
    os.environ["DATABASE_URL"] = TEST_URL
    from apguard import db
    db.get_engine.cache_clear()
    eng = db.get_engine()
    with eng.begin() as conn:
        conn.exec_driver_sql((ROOT / "db" / "schema.sql").read_text())
    sys.argv = ["load_data.py"]
    sys.path.insert(0, str(ROOT / "db"))
    import load_data
    load_data.main()
    from apguard.pipeline import run_detection
    run_detection(eng)
    yield eng
    db.get_engine.cache_clear()


@pytest.fixture(scope="module")
def engine_flags(engine):
    from apguard.db import load_source_tables
    from rules.engine import run_all_rules
    t = load_source_tables(engine)
    return run_all_rules(t["vendors"], t["contracts"], t["purchase_orders"], t["goods_receipts"],
                         t["invoices"], t["payments"])


# --------------------------------------------------------------- SQL vs pandas parity
@pytest.mark.parametrize("view,rule,key", [("v_rule_r4_missing_po", "R4", "invoice_id"),
                                           ("v_rule_r5_duplicate_payment", "R5", "payment_id"),
                                           ("v_rule_r8_overpayment", "R8", "payment_id")])
def test_sql_view_matches_pandas_engine(engine, engine_flags, view, rule, key):
    sql = pd.read_sql(f"SELECT {key}, amount_at_risk FROM {view}", engine).set_index(key)["amount_at_risk"]
    py = engine_flags[engine_flags["rule_code"] == rule].set_index(key)["amount_at_risk"]
    assert set(sql.index) == set(py.index.astype(int))
    assert (sql.sort_index().round(2).values == py.sort_index().astype(float).round(2).values).all()


def test_db_flags_match_offline_engine(engine, engine_flags):
    """Postgres types (NUMERIC, DATE) must not change what the rules find."""
    from apguard.evaluation import load_dataset
    from rules.engine import run_all_rules
    ds = load_dataset(ROOT / "data" / "generated")
    offline = run_all_rules(*ds.tables)
    assert engine_flags.groupby("rule_code").size().to_dict() == offline.groupby("rule_code").size().to_dict()


# --------------------------------------------------------------- pipeline
def test_rerun_is_idempotent_and_preserves_reviews(engine):
    from apguard.cases import update_flag
    from apguard.pipeline import run_detection
    fid = int(pd.read_sql("SELECT min(flag_id) AS f FROM flags WHERE status = 'open'", engine).iloc[0]["f"])
    update_flag(fid, status="confirmed", note="checked", actor="test", engine=engine)
    before = pd.read_sql("SELECT count(*) AS n FROM flags", engine).iloc[0]["n"]
    r = run_detection(engine)
    after = pd.read_sql("SELECT count(*) AS n FROM flags", engine).iloc[0]["n"]
    assert r["flags_new"] == 0 and before == after
    assert pd.read_sql(f"SELECT status FROM flags WHERE flag_id = {fid}", engine).iloc[0]["status"] == "confirmed"


def test_run_records_config_snapshot(engine):
    run = pd.read_sql("SELECT * FROM rule_runs ORDER BY run_id DESC LIMIT 1", engine).iloc[0]
    assert run["config"]["approval_threshold"] == 5000.0
    assert run["flags_raised"] > 0 and run["finished_at"] is not None


# --------------------------------------------------------------- workflow
def _open_flag(engine) -> int:
    return int(pd.read_sql("SELECT max(flag_id) AS f FROM flags WHERE status = 'open' AND amount_at_risk > 0",
                           engine).iloc[0]["f"])


def test_workflow_happy_path_and_audit_trail(engine):
    from apguard.cases import update_flag
    fid = _open_flag(engine)
    update_flag(fid, status="in_review", assignee="a.reviewer", actor="lead", engine=engine)
    update_flag(fid, status="confirmed", actor="a.reviewer", engine=engine)
    f = update_flag(fid, status="recovered", recovered_amount=100.0, actor="a.reviewer", engine=engine)
    assert f["status"] == "recovered" and float(f["recovered_amount"]) == 100.0
    ev = pd.read_sql(f"SELECT event_type, to_status FROM flag_events WHERE flag_id = {fid} ORDER BY event_id", engine)
    assert ev["to_status"].dropna().tolist() == ["open", "in_review", "confirmed", "recovered"]
    assert "assigned" in ev["event_type"].tolist()


@pytest.mark.parametrize("kwargs,msg", [
    ({"status": "recovered", "recovered_amount": 10.0}, "cannot move"),
    ({"status": "false_positive"}, "note"),
    ({"status": "bogus"}, "unknown status"),
    ({}, "nothing to update"),
])
def test_workflow_rejects_invalid_actions(engine, kwargs, msg):
    from apguard.cases import WorkflowError, update_flag
    with pytest.raises(WorkflowError, match=msg):
        update_flag(_open_flag(engine), engine=engine, **kwargs)


def test_dismissed_flags_leave_net_exposure(engine):
    from apguard import metrics
    from apguard.cases import update_flag
    before = metrics.summary(engine)["net_exposure"]
    fid = int(pd.read_sql("""SELECT f.flag_id FROM v_active_flags f JOIN v_invoice_exposure e USING (invoice_id)
                             WHERE f.status = 'open' AND e.flag_count = 1 AND f.amount_at_risk > 0
                             LIMIT 1""", engine).iloc[0]["flag_id"])
    amt = float(pd.read_sql(f"SELECT amount_at_risk FROM flags WHERE flag_id = {fid}", engine).iloc[0, 0])
    update_flag(fid, status="false_positive", note="legitimate", engine=engine)
    assert metrics.summary(engine)["net_exposure"] == pytest.approx(before - amt, abs=0.01)


# --------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(engine):
    os.environ["APGUARD_API_KEY"] = "test-key"
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_api_flags_pagination_reports_total(client):
    r = client.get("/flags", params={"rule_code": "R5", "limit": 10})
    body = r.json()
    assert r.status_code == 200 and body["total"] > 10 and len(body["items"]) == 10


def test_api_validates_enums(client):
    assert client.get("/flags", params={"severity": "urgent"}).status_code == 422


def test_api_flag_evidence(client):
    fid = client.get("/flags", params={"rule_code": "R2", "limit": 1}).json()["items"][0]["flag_id"]
    body = client.get(f"/flags/{fid}").json()
    assert body["purchase_order"] and body["invoice"] and body["events"]


def test_api_review_requires_key(client):
    fid = client.get("/flags", params={"status": "open", "limit": 1}).json()["items"][0]["flag_id"]
    assert client.patch(f"/flags/{fid}", json={"status": "in_review"}).status_code == 401
    r = client.patch(f"/flags/{fid}", json={"status": "in_review", "assignee": "x"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200 and r.json()["status"] == "in_review"
    bad = client.patch(f"/flags/{fid}", json={"status": "recovered"}, headers={"X-API-Key": "test-key"})
    assert bad.status_code == 422


def test_api_summary_and_ranking(client):
    s = client.get("/flags/summary").json()
    assert s["net_exposure"] <= s["gross_at_risk"]
    ranking = client.get("/vendors/risk-ranking", params={"limit": 10}).json()
    assert len(ranking) == 10 and ranking[0]["risk_score"] >= ranking[-1]["risk_score"]


def test_api_health(client):
    assert client.get("/health").json()["status"] == "ok"
