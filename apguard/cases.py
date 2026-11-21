"""Review workflow for flags: status transitions, assignment, notes and recoveries,
each recorded in flag_events as an append-only audit trail."""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from apguard.db import get_engine

STATUSES = ["open", "in_review", "confirmed", "false_positive", "recovered"]
STATUS_LABELS = {"open": "Open", "in_review": "In review", "confirmed": "Confirmed",
                 "false_positive": "False positive", "recovered": "Recovered"}

TRANSITIONS: dict[str, set[str]] = {
    "open": {"in_review", "confirmed", "false_positive"},
    "in_review": {"open", "confirmed", "false_positive"},
    "confirmed": {"in_review", "recovered"},
    "false_positive": {"open"},
    "recovered": set(),  # terminal: a recovery is booked
}


class WorkflowError(ValueError):
    pass


def allowed_transitions(status: str) -> list[str]:
    return [s for s in STATUSES if s in TRANSITIONS.get(status, set())]


def update_flag(flag_id: int, *, status: str | None = None, assignee: str | None = None,
                note: str | None = None, recovered_amount: float | None = None,
                actor: str = "analyst", engine: Engine | None = None) -> dict:
    """Apply a review action atomically. Raises WorkflowError on an invalid request
    and LookupError if the flag does not exist."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("SELECT flag_id, status, assignee, amount_at_risk FROM flags "
                                "WHERE flag_id = :id FOR UPDATE"), {"id": flag_id}).mappings().first()
        if row is None:
            raise LookupError(f"flag {flag_id} not found")
        current = row["status"]
        events = []

        if status is not None and status != current:
            if status not in STATUSES:
                raise WorkflowError(f"unknown status '{status}'")
            if status not in TRANSITIONS[current]:
                raise WorkflowError(f"cannot move a flag from '{current}' to '{status}'; "
                                    f"allowed: {', '.join(allowed_transitions(current)) or 'none'}")
            if status == "recovered" and (recovered_amount is None or recovered_amount <= 0):
                raise WorkflowError("recovered_amount must be positive when marking a flag recovered")
            if status == "false_positive" and not note:
                raise WorkflowError("a note explaining why is required to dismiss a flag as a false positive")
            events.append(("status_change", current, status))
        elif recovered_amount is not None and status != "recovered":
            raise WorkflowError("recovered_amount can only be set together with status 'recovered'")

        if assignee is not None and assignee != row["assignee"]:
            events.append(("assigned", None, None))

        if not events and not note:
            raise WorkflowError("nothing to update")

        conn.execute(text("""
            UPDATE flags SET
                status = COALESCE(:status, status),
                assignee = COALESCE(:assignee, assignee),
                recovered_amount = COALESCE(:recovered, recovered_amount),
                updated_at = now()
            WHERE flag_id = :id
        """), {"status": status, "assignee": assignee, "recovered": recovered_amount, "id": flag_id})

        for etype, frm, to in events:
            msg = note if etype == "status_change" else (f"Assigned to {assignee}" if etype == "assigned" else None)
            if etype == "status_change" and to == "recovered":
                msg = f"Recovered {recovered_amount:,.2f}" + (f": {note}" if note else "")
            conn.execute(text("INSERT INTO flag_events (flag_id, event_type, from_status, to_status, actor, note) "
                              "VALUES (:f, :t, :fr, :to, :a, :n)"),
                         {"f": flag_id, "t": etype, "fr": frm, "to": to, "a": actor, "n": msg})
        if note and not any(e[0] == "status_change" for e in events):
            conn.execute(text("INSERT INTO flag_events (flag_id, event_type, actor, note) VALUES (:f, 'note', :a, :n)"),
                         {"f": flag_id, "a": actor, "n": note})

        updated = conn.execute(text("SELECT * FROM flags WHERE flag_id = :id"), {"id": flag_id}).mappings().first()
    return dict(updated)
