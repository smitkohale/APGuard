"""Naive reference implementations, used only by the evaluation.

These mirror the first version of APGuard (and what most tutorials do):
exact-match duplicates that flag both invoices, per-invoice three-way matching,
the first contract found regardless of date, purely relative tolerances, and
invoice splitting with no awareness of duplicates or partial deliveries.
Scoring them on the same data shows what each refinement in rules/engine.py
actually buys, in precision and recall, instead of just asserting it.
"""
from __future__ import annotations

import pandas as pd

from rules.engine import _make_flags, _with_po


def naive_duplicate_invoice(invoices: pd.DataFrame) -> pd.DataFrame:
    inv = _with_po(invoices)
    key = ["vendor_id", "po_id", "item_sku", "quantity", "total_amount"]
    df = inv[inv.duplicated(key, keep=False)]
    return _make_flags("R1", df, details=[""] * len(df), amount="total_amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


def naive_three_way_match(invoices, pos, receipts) -> pd.DataFrame:
    inv = _with_po(invoices).merge(pos[["po_id", "quantity"]], on="po_id", suffixes=("", "_po"))
    recv = receipts.groupby("po_id")["received_qty"].sum()
    inv["recv"] = inv["po_id"].map(recv)
    hit = (inv["quantity"] != inv["quantity_po"]) | (inv["recv"].notna() & (inv["quantity"] != inv["recv"]))
    df = inv[hit]
    return _make_flags("R2", df, details=[""] * len(df), amount=(df["quantity"] - df["quantity_po"]).abs() * df["unit_price"],
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


def naive_contract_price(invoices, contracts) -> pd.DataFrame:
    first = contracts.sort_values("contract_id").drop_duplicates(["vendor_id", "item_sku"])
    df = invoices.merge(first[["vendor_id", "item_sku", "contract_price"]], on=["vendor_id", "item_sku"])
    df = df[df["unit_price"] > df["contract_price"] * 1.001]
    return _make_flags("R3", df, details=[""] * len(df), amount=(df["unit_price"] - df["contract_price"]) * df["quantity"],
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


def naive_tax_validation(invoices, contracts, pos) -> pd.DataFrame:
    df = _with_po(invoices).merge(pos[["po_id", "contract_id"]], on="po_id").dropna(subset=["contract_id"])
    df = df.merge(contracts[["contract_id", "tax_rate"]], on="contract_id")
    exp = (df["subtotal"] * df["tax_rate"]).round(2)
    df = df[(exp > 0) & ((df["tax_amount"] - exp).abs() / exp > 0.01)]
    return _make_flags("R6", df, details=[""] * len(df), amount=0.0,
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


def naive_invoice_splitting(invoices, threshold=5000.0, window=10) -> pd.DataFrame:
    inv = _with_po(invoices)
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])
    g = inv.groupby("po_id").agg(n=("invoice_id", "size"), mx=("total_amount", "max"), s=("total_amount", "sum"),
                                 a=("invoice_date", "min"), b=("invoice_date", "max"))
    hit = g[(g["n"] >= 2) & (g["mx"] < threshold) & (g["s"] >= threshold) & ((g["b"] - g["a"]).dt.days <= window)]
    df = inv[inv["po_id"].isin(hit.index)]
    return _make_flags("R7", df, details=[""] * len(df), amount="total_amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


def naive_overpayment(payments, invoices) -> pd.DataFrame:
    df = payments.merge(invoices[["invoice_id", "vendor_id", "total_amount"]], on="invoice_id")
    df = df[df["payment_amount"] > df["total_amount"] * 1.001]
    return _make_flags("R8", df, details=[""] * len(df), amount=df["payment_amount"] - df["total_amount"],
                       vendor_id="vendor_id", invoice_id="invoice_id", payment_id="payment_id")


def run_baseline(vendors, contracts, pos, receipts, invoices, payments) -> pd.DataFrame:
    return pd.concat([
        naive_duplicate_invoice(invoices),
        naive_three_way_match(invoices, pos, receipts),
        naive_contract_price(invoices, contracts),
        naive_tax_validation(invoices, contracts, pos),
        naive_invoice_splitting(invoices),
        naive_overpayment(payments, invoices),
    ], ignore_index=True)


BASELINE_RULES = ["R1", "R2", "R3", "R6", "R7", "R8"]
