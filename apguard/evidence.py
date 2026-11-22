"""The document chain behind a flag: vendor, contract, PO, receipts, invoice(s), payments."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from apguard.db import read_sql


def _one(df: pd.DataFrame):
    return df.iloc[0].to_dict() if not df.empty else None


def evidence_for_flag(flag_id: int, engine: Engine | None = None) -> dict | None:
    flag = read_sql("SELECT f.*, v.vendor_name FROM flags f LEFT JOIN vendors v USING (vendor_id) "
                    "WHERE f.flag_id = :id", {"id": flag_id}, engine)
    if flag.empty:
        return None
    f = flag.iloc[0]
    out: dict = {"flag": f.to_dict()}

    invoice = read_sql("SELECT * FROM invoices WHERE invoice_id = :i", {"i": int(f["invoice_id"])}, engine) \
        if pd.notna(f["invoice_id"]) else pd.DataFrame()
    out["invoice"] = _one(invoice)
    po_id = f["po_id"] if pd.notna(f["po_id"]) else (invoice.iloc[0]["po_id"] if not invoice.empty else None)
    po_id = int(po_id) if po_id is not None and pd.notna(po_id) else None

    out["purchase_order"] = (_one(read_sql("SELECT * FROM purchase_orders WHERE po_id = :p", {"p": po_id}, engine))
                             if po_id else None)
    out["contract"] = None
    if out["purchase_order"] and pd.notna(out["purchase_order"].get("contract_id")):
        out["contract"] = _one(read_sql("SELECT * FROM contracts WHERE contract_id = :c",
                                        {"c": int(out["purchase_order"]["contract_id"])}, engine))
    out["goods_receipts"] = read_sql("SELECT * FROM goods_receipts WHERE po_id = :p ORDER BY received_date",
                                     {"p": po_id}, engine).to_dict(orient="records") if po_id else []
    out["invoices_on_po"] = read_sql("SELECT * FROM invoices WHERE po_id = :p ORDER BY invoice_date",
                                     {"p": po_id}, engine).to_dict(orient="records") if po_id else []
    out["payments"] = read_sql("SELECT * FROM payments WHERE invoice_id = :i ORDER BY payment_date",
                               {"i": int(f["invoice_id"])}, engine).to_dict(orient="records") \
        if pd.notna(f["invoice_id"]) else []
    out["vendor"] = _one(read_sql("SELECT * FROM vendors WHERE vendor_id = :v", {"v": int(f["vendor_id"])}, engine)) \
        if pd.notna(f["vendor_id"]) else None
    # For duplicate-invoice flags, surface the earlier invoices it may duplicate:
    # same vendor, SKU and quantity, dated before it.
    out["similar_invoices"] = []
    if f["rule_code"] == "R1" and out["invoice"]:
        inv = out["invoice"]
        out["similar_invoices"] = read_sql(
            """SELECT * FROM invoices WHERE vendor_id = :v AND item_sku = :s AND quantity = :q
               AND invoice_id <> :i AND invoice_date <= :d ORDER BY invoice_date DESC LIMIT 5""",
            {"v": int(inv["vendor_id"]), "s": inv["item_sku"], "q": int(inv["quantity"]),
             "i": int(inv["invoice_id"]), "d": inv["invoice_date"]}, engine).to_dict(orient="records")
    out["events"] = read_sql("SELECT * FROM flag_events WHERE flag_id = :id ORDER BY created_at, event_id",
                             {"id": flag_id}, engine).to_dict(orient="records")
    return out
