"""
APGuard rules engine.

Each rule is a pure function: takes the relevant DataFrames, returns a DataFrame
of flags with columns: rule_code, rule_name, vendor_id, invoice_id, po_id,
payment_id, severity, amount_at_risk, details.

Keeping these as pure functions (no DB access inside) is what makes them
independently pytest-able against small, hand-built fixtures.
"""
import pandas as pd

TAX_TOLERANCE = 0.01       # 1% relative tolerance on tax_amount vs contract tax_rate
QTY_MISMATCH_TOLERANCE = 0  # units; any mismatch beyond this is flagged
SPLIT_WINDOW_DAYS = 10
APPROVAL_THRESHOLD = 5000.00


def _empty_flags():
    return pd.DataFrame(columns=[
        "rule_code", "rule_name", "vendor_id", "invoice_id", "po_id",
        "payment_id", "severity", "amount_at_risk", "details",
    ])


def rule_duplicate_invoice(invoices: pd.DataFrame) -> pd.DataFrame:
    """R1: Two or more invoices against the same PO with the same vendor,
    SKU, quantity and amount -> likely duplicate billing."""
    if invoices.empty:
        return _empty_flags()
    subset = invoices.dropna(subset=["po_id"]).copy()
    key_cols = ["vendor_id", "po_id", "item_sku", "quantity", "total_amount"]
    dupe_mask = subset.duplicated(subset=key_cols, keep=False)
    dupes = subset[dupe_mask].sort_values(key_cols)
    rows = []
    for _, r in dupes.iterrows():
        rows.append({
            "rule_code": "R1", "rule_name": "Duplicate Invoice",
            "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": r["po_id"],
            "payment_id": None, "severity": "high", "amount_at_risk": r["total_amount"],
            "details": f"Invoice {r['invoice_number']} matches another invoice on PO {int(r['po_id'])} "
                       f"(same SKU, qty, amount).",
        })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_three_way_match(invoices: pd.DataFrame, pos: pd.DataFrame, receipts: pd.DataFrame) -> pd.DataFrame:
    """R2: Invoice quantity must reconcile with PO quantity and goods receipt
    quantity — but only checked against the receipt when one actually exists.
    A PO with NO goods receipt at all isn't a quantity mismatch, it's a
    different problem (see R9, Payment Without Goods Receipt) — treating a
    missing receipt as "received_qty=0" here would double-flag every R9 case
    as a match failure too, which conflates two distinct root causes."""
    if invoices.empty:
        return _empty_flags()
    inv = invoices.dropna(subset=["po_id"]).copy()
    inv["po_id"] = inv["po_id"].astype(int)
    po_lookup = pos.set_index("po_id")
    gr_by_po = receipts.groupby("po_id")["received_qty"].sum()

    rows = []
    for _, r in inv.iterrows():
        po_id = r["po_id"]
        if po_id not in po_lookup.index:
            continue
        po_qty = po_lookup.loc[po_id, "quantity"]
        inv_qty = r["quantity"]
        has_receipt = po_id in gr_by_po.index
        received_qty = gr_by_po.get(po_id) if has_receipt else None

        mismatch_vs_po = abs(inv_qty - po_qty) > QTY_MISMATCH_TOLERANCE
        mismatch_vs_gr = has_receipt and abs(inv_qty - received_qty) > QTY_MISMATCH_TOLERANCE
        if mismatch_vs_po or mismatch_vs_gr:
            unit_price = r["unit_price"]
            variance_amt = round(abs(inv_qty - po_qty) * unit_price, 2)
            received_desc = received_qty if has_receipt else "no receipt on file"
            rows.append({
                "rule_code": "R2", "rule_name": "Three-Way Match Failure",
                "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": po_id,
                "payment_id": None, "severity": "high", "amount_at_risk": variance_amt,
                "details": f"Invoice qty {inv_qty} vs PO qty {po_qty} vs received qty {received_desc}.",
            })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_contract_price(invoices: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """R3: Invoiced unit price must not exceed the contracted price for that vendor/SKU."""
    if invoices.empty or contracts.empty:
        return _empty_flags()
    contract_price = contracts.set_index(["vendor_id", "item_sku"])["contract_price"]
    rows = []
    for _, r in invoices.iterrows():
        key = (r["vendor_id"], r["item_sku"])
        if key not in contract_price.index:
            continue
        c_price = contract_price.loc[key]
        if isinstance(c_price, pd.Series):  # duplicate contract rows edge case
            c_price = c_price.iloc[0]
        if r["unit_price"] > c_price * 1.001:  # small float tolerance
            variance = round((r["unit_price"] - c_price) * r["quantity"], 2)
            rows.append({
                "rule_code": "R3", "rule_name": "Contract Price Violation",
                "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": r.get("po_id"),
                "payment_id": None, "severity": "medium", "amount_at_risk": variance,
                "details": f"Invoiced at {r['unit_price']:.2f} vs contracted {c_price:.2f} for {r['item_sku']}.",
            })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_missing_po(invoices: pd.DataFrame) -> pd.DataFrame:
    """R4: Invoice has no associated purchase order (no upstream approval trail)."""
    if invoices.empty:
        return _empty_flags()
    missing = invoices[invoices["po_id"].isna()]
    rows = []
    for _, r in missing.iterrows():
        rows.append({
            "rule_code": "R4", "rule_name": "Missing Purchase Order",
            "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": None,
            "payment_id": None, "severity": "high", "amount_at_risk": r["total_amount"],
            "details": f"Invoice {r['invoice_number']} has no linked PO.",
        })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_duplicate_payment(payments: pd.DataFrame) -> pd.DataFrame:
    """R5: Same invoice paid more than once."""
    if payments.empty:
        return _empty_flags()
    dupe_mask = payments.duplicated(subset=["invoice_id"], keep=False)
    # Sort by payment_id explicitly (not just invoice_id) — SQL row order is not
    # guaranteed, so without this the "first" row in a group can be arbitrary,
    # which sometimes flagged the original payment instead of the later duplicate.
    dupes = payments[dupe_mask].sort_values(["invoice_id", "payment_id"])
    rows = []
    for invoice_id, grp in dupes.groupby("invoice_id"):
        for _, r in grp.iloc[1:].iterrows():  # flag every payment after the earliest
            rows.append({
                "rule_code": "R5", "rule_name": "Duplicate Payment",
                "vendor_id": None, "invoice_id": invoice_id, "po_id": None,
                "payment_id": r["payment_id"], "severity": "critical", "amount_at_risk": r["payment_amount"],
                "details": f"Invoice {invoice_id} has {len(grp)} payments; ref {r['payment_ref']} is extra.",
            })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_tax_validation(invoices: pd.DataFrame, contracts: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """R6: Tax amount should equal subtotal * contracted tax rate (within tolerance).
    Only checked where the invoice's PO is linked to a contract, since that's the
    only place we have an authoritative tax rate to check against."""
    if invoices.empty or contracts.empty:
        return _empty_flags()
    po_to_contract = pos.set_index("po_id")["contract_id"]
    contract_rate = contracts.set_index("contract_id")["tax_rate"]

    rows = []
    inv = invoices.dropna(subset=["po_id"]).copy()
    inv["po_id"] = inv["po_id"].astype(int)
    for _, r in inv.iterrows():
        po_id = r["po_id"]
        if po_id not in po_to_contract.index:
            continue
        contract_id = po_to_contract.loc[po_id]
        if pd.isna(contract_id) or contract_id not in contract_rate.index:
            continue
        expected_tax = round(r["subtotal"] * contract_rate.loc[contract_id], 2)
        actual_tax = r["tax_amount"]
        if expected_tax == 0:
            continue
        if abs(actual_tax - expected_tax) / max(expected_tax, 0.01) > TAX_TOLERANCE:
            rows.append({
                "rule_code": "R6", "rule_name": "Tax Validation Failure",
                "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": po_id,
                "payment_id": None, "severity": "low", "amount_at_risk": round(abs(actual_tax - expected_tax), 2),
                "details": f"Tax charged {actual_tax:.2f} vs expected {expected_tax:.2f} "
                           f"(contract rate {contract_rate.loc[contract_id]*100:.2f}%).",
            })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_invoice_splitting(invoices: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """R7: Multiple invoices against the same PO, each individually under the
    approval threshold, that together exceed it and were filed within a short window
    -> possible deliberate splitting to avoid approval controls."""
    if invoices.empty:
        return _empty_flags()
    inv = invoices.dropna(subset=["po_id"]).copy()
    inv["po_id"] = inv["po_id"].astype(int)
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"])

    rows = []
    for po_id, grp in inv.groupby("po_id"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("invoice_date")
        window_days = (grp["invoice_date"].max() - grp["invoice_date"].min()).days
        all_under_threshold = (grp["total_amount"] < APPROVAL_THRESHOLD).all()
        combined_total = grp["total_amount"].sum()
        if all_under_threshold and combined_total >= APPROVAL_THRESHOLD and window_days <= SPLIT_WINDOW_DAYS:
            for _, r in grp.iterrows():
                rows.append({
                    "rule_code": "R7", "rule_name": "Invoice Splitting",
                    "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": po_id,
                    "payment_id": None, "severity": "medium", "amount_at_risk": r["total_amount"],
                    "details": f"1 of {len(grp)} invoices on PO {po_id} totalling {combined_total:.2f} "
                               f"filed within {window_days} days, each under the {APPROVAL_THRESHOLD:.0f} threshold.",
                })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_overpayment(payments: pd.DataFrame, invoices: pd.DataFrame) -> pd.DataFrame:
    """R8: Payment amount exceeds the invoice total it's paying."""
    if payments.empty:
        return _empty_flags()
    inv_totals = invoices.set_index("invoice_id")["total_amount"]
    rows = []
    for _, r in payments.iterrows():
        invoice_id = r["invoice_id"]
        if invoice_id not in inv_totals.index:
            continue
        expected = inv_totals.loc[invoice_id]
        if r["payment_amount"] > expected * 1.001:
            rows.append({
                "rule_code": "R8", "rule_name": "Overpayment",
                "vendor_id": None, "invoice_id": invoice_id, "po_id": None,
                "payment_id": r["payment_id"], "severity": "high",
                "amount_at_risk": round(r["payment_amount"] - expected, 2),
                "details": f"Paid {r['payment_amount']:.2f} vs invoice total {expected:.2f}.",
            })
    return pd.DataFrame(rows) if rows else _empty_flags()


def rule_payment_without_gr(invoices: pd.DataFrame, payments: pd.DataFrame,
                             receipts: pd.DataFrame) -> pd.DataFrame:
    """R9: Payment made against an invoice whose PO has no goods receipt on
    file at all — paying for something never confirmed as received. Distinct
    from R2 (three-way match failure), which is about a receipt that EXISTS
    but disagrees with the invoiced quantity; R9 is about no receipt existing
    at all, which R2 deliberately does not treat as a quantity mismatch."""
    if payments.empty or invoices.empty:
        return _empty_flags()
    inv = invoices.dropna(subset=["po_id"]).copy()
    inv["po_id"] = inv["po_id"].astype(int)
    received_po_ids = set(receipts["po_id"].unique())
    unreceived_invoices = inv[~inv["po_id"].isin(received_po_ids)]
    if unreceived_invoices.empty:
        return _empty_flags()

    paid = payments.merge(unreceived_invoices[["invoice_id", "po_id", "vendor_id"]], on="invoice_id")
    rows = []
    for _, r in paid.iterrows():
        rows.append({
            "rule_code": "R9", "rule_name": "Payment Without Goods Receipt",
            "vendor_id": r["vendor_id"], "invoice_id": r["invoice_id"], "po_id": r["po_id"],
            "payment_id": r["payment_id"], "severity": "high", "amount_at_risk": r["payment_amount"],
            "details": f"Payment {r.get('payment_ref', r['payment_id'])} made against PO {int(r['po_id'])} "
                       f"with no goods receipt on file.",
        })
    return pd.DataFrame(rows) if rows else _empty_flags()


RULE_REGISTRY = {
    "R1": "Duplicate Invoice",
    "R2": "Three-Way Match Failure",
    "R3": "Contract Price Violation",
    "R4": "Missing Purchase Order",
    "R5": "Duplicate Payment",
    "R6": "Tax Validation Failure",
    "R7": "Invoice Splitting",
    "R8": "Overpayment",
    "R9": "Payment Without Goods Receipt",
}


def run_all_rules(vendors, contracts, pos, receipts, invoices, payments) -> pd.DataFrame:
    """Run all 9 rules and return one concatenated flags DataFrame."""
    results = [
        rule_duplicate_invoice(invoices),
        rule_three_way_match(invoices, pos, receipts),
        rule_contract_price(invoices, contracts),
        rule_missing_po(invoices),
        rule_duplicate_payment(payments),
        rule_tax_validation(invoices, contracts, pos),
        rule_invoice_splitting(invoices, pos),
        rule_overpayment(payments, invoices),
        rule_payment_without_gr(invoices, payments, receipts),
    ]
    non_empty = [r for r in results if not r.empty]
    if not non_empty:
        return _empty_flags()
    combined = pd.concat(non_empty, ignore_index=True)
    return combined
