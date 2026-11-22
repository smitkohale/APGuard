"""Shared fixture builders. Each test assembles only the rows it needs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def inv(**kw) -> dict:
    row = dict(invoice_id=1, vendor_id=1, po_id=10, invoice_number="INV-0001-000001", invoice_date="2026-01-01",
               item_sku="SKU-0001", quantity=10, unit_price=100.0, subtotal=1000.0, tax_amount=70.0,
               total_amount=1070.0)
    row.update(kw)
    return row


def po(**kw) -> dict:
    row = dict(po_id=10, vendor_id=1, contract_id=None, item_sku="SKU-0001", quantity=10, unit_price=100.0,
               po_date="2025-12-20", po_amount=1000.0)
    row.update(kw)
    return row


def gr(**kw) -> dict:
    row = dict(gr_id=1, po_id=10, received_qty=10, received_date="2025-12-28")
    row.update(kw)
    return row


def pay(**kw) -> dict:
    row = dict(payment_id=1, invoice_id=1, payment_date="2026-01-20", payment_amount=1070.0, payment_ref="PMT-1")
    row.update(kw)
    return row


def contract(**kw) -> dict:
    row = dict(contract_id=1, vendor_id=1, item_sku="SKU-0001", contract_price=100.0, tax_rate=0.07,
               start_date="2024-01-01", end_date="2026-12-31")
    row.update(kw)
    return row


def vendor(**kw) -> dict:
    row = dict(vendor_id=1, vendor_name="Acme Supplies", tax_id="AB-1", bank_account="GB00ACME1",
               country="UK", risk_category="low", created_at="2022-01-01")
    row.update(kw)
    return row


def df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def invoices(*rows) -> pd.DataFrame:
    d = pd.DataFrame(list(rows))
    d["po_id"] = d["po_id"].astype("Int64")
    return d
