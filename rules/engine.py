"""
APGuard rules engine.

Every detector is a pure function: it takes DataFrames (and a config dict) and
returns a DataFrame of flags with the columns in FLAG_COLUMNS. Nothing in here
touches the database, so every rule can be tested against small hand-built
fixtures and benchmarked on synthetic data of any size.

All rules are vectorised (merges and group-bys, no row-by-row loops). The only
Python-level iteration left is building the human-readable `details` strings
for rows that have already been flagged.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from rules.config import load_config
from rules.registry import RULES

FLAG_COLUMNS = [
    "rule_code", "rule_name", "category", "vendor_id", "invoice_id", "po_id",
    "payment_id", "severity", "amount_at_risk", "details",
]
ID_COLUMNS = ["vendor_id", "invoice_id", "po_id", "payment_id"]

# Kept as module constants for backwards compatibility and readable tests.
APPROVAL_THRESHOLD = 5000.00


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _empty_flags() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in FLAG_COLUMNS})
    for c in ID_COLUMNS:
        df[c] = df[c].astype("Int64")
    df["amount_at_risk"] = df["amount_at_risk"].astype(float)
    return df


def _make_flags(code: str, df: pd.DataFrame, *, details, amount, severity=None,
                vendor_id=None, invoice_id=None, po_id=None, payment_id=None) -> pd.DataFrame:
    """Assemble a flags frame for rule `code` from column names / arrays on `df`."""
    if df.empty:
        return _empty_flags()
    spec = RULES[code]

    def col(value):
        if value is None:
            return pd.Series([pd.NA] * len(df), index=df.index)
        if isinstance(value, str):
            return df[value]
        return pd.Series(value, index=df.index)

    out = pd.DataFrame({
        "rule_code": code,
        "rule_name": spec.name,
        "category": spec.category,
        "vendor_id": col(vendor_id),
        "invoice_id": col(invoice_id),
        "po_id": col(po_id),
        "payment_id": col(payment_id),
        "severity": col(severity) if severity is not None else spec.severity,
        "amount_at_risk": col(amount).astype(float).round(2),
        "details": list(details),
    }, index=df.index)
    for c in ID_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)[FLAG_COLUMNS]


def _dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def _with_po(invoices: pd.DataFrame) -> pd.DataFrame:
    inv = invoices[invoices["po_id"].notna()].copy()
    inv["po_id"] = inv["po_id"].astype("int64")
    return inv


def normalize_invoice_number(number: pd.Series) -> pd.Series:
    """'inv-0001/000123 ' -> 'INV0001000123'. Catches resubmissions that were
    retyped with different separators, case or whitespace."""
    return number.astype(str).str.upper().map(lambda s: re.sub(r"[^0-9A-Z]", "", s))


def _invoice_vendor(payments: pd.DataFrame, invoices: pd.DataFrame) -> pd.DataFrame:
    """Attach vendor_id / po_id / total to payments via their invoice."""
    cols = ["invoice_id", "vendor_id", "po_id", "total_amount"]
    cols = [c for c in cols if c in invoices.columns]
    return payments.merge(invoices[cols], on="invoice_id", how="left")


# --------------------------------------------------------------------------- #
# R1 duplicate invoice
# --------------------------------------------------------------------------- #
def find_duplicate_invoices(invoices: pd.DataFrame, pos: pd.DataFrame | None = None,
                            config: dict | None = None) -> pd.DataFrame:
    """Return (invoice_id, original_invoice_id, original_number, match_type) for every
    invoice that duplicates an earlier one. Only the *later* invoice is returned.

    Two independent matchers:
      * invoice number: same vendor + same invoice number (normalised when fuzzy)
      * content: same vendor/PO/SKU/qty, amount within tolerance, within N days,
        AND total invoiced quantity on the PO exceeds the PO quantity. That last
        condition is what stops legitimate partial deliveries (two equal invoices
        that together make up the order) from being flagged.
    """
    cfg = load_config(overrides=config)["duplicate_invoice"]
    cols = ["invoice_id", "original_invoice_id", "original_number", "match_type", "matcher"]
    if invoices.empty:
        return pd.DataFrame(columns=cols)

    fuzzy = bool(cfg["fuzzy"])
    inv = _dates(invoices, "invoice_date")
    inv = inv.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
    inv["_order"] = np.arange(len(inv))
    inv["_num"] = normalize_invoice_number(inv["invoice_number"]) if fuzzy \
        else inv["invoice_number"].astype(str)

    keep = ["invoice_id", "invoice_number", "_order", "invoice_date", "total_amount"]

    # --- matcher A: invoice number -------------------------------------------
    a = inv[["vendor_id", "_num"] + keep].merge(
        inv[["vendor_id", "_num"] + keep], on=["vendor_id", "_num"], suffixes=("_o", "_d"))
    a = a[a["_order_o"] < a["_order_d"]]
    a = a.assign(match_type="invoice number" + (" (normalised)" if fuzzy else ""), _matcher="A")

    # --- matcher B: content ---------------------------------------------------
    inv_po = _with_po(inv)
    key = ["vendor_id", "po_id", "item_sku", "quantity"]
    b = inv_po[key + keep].merge(inv_po[key + keep], on=key, suffixes=("_o", "_d"))
    b = b[b["_order_o"] < b["_order_d"]]
    if fuzzy:
        tol = np.maximum(cfg["amount_tolerance_abs"], cfg["amount_tolerance_pct"] * b["total_amount_o"].abs())
    else:
        tol = 0.005  # float noise only
    b = b[((b["total_amount_d"] - b["total_amount_o"]).abs() <= tol)
          & ((b["invoice_date_d"] - b["invoice_date_o"]).dt.days <= cfg["window_days"])]
    if not b.empty and pos is not None and not pos.empty:
        billed = inv_po.groupby("po_id")["quantity"].sum()
        ordered = pos.set_index("po_id")["quantity"]
        over_billed = billed[billed > ordered.reindex(billed.index).fillna(np.inf)].index
        b = b[b["po_id"].isin(over_billed)]
    b = b.assign(match_type="same PO, SKU, quantity and amount" + (" (within tolerance)" if fuzzy else ""),
                 _matcher="B")

    # --- matcher C: same content across different / missing PO references -----
    # Catches resubmissions keyed without the PO. Same vendor, SKU, quantity and
    # amount, but with a much tighter date window, because recurring monthly
    # billing looks identical on every field except the date.
    frames = [a, b]
    if fuzzy and cfg.get("cross_po_window_days", 0) > 0:
        key_c = ["vendor_id", "item_sku", "quantity"]
        cols_c = key_c + keep + ["po_id"]
        c = inv[cols_c].merge(inv[cols_c], on=key_c, suffixes=("_o", "_d"))
        missing_po = c["po_id_o"].isna() | c["po_id_d"].isna()
        if cfg.get("cross_po_scope", "missing_po_only") == "any_po":
            different_po = missing_po | (c["po_id_o"] != c["po_id_d"]).fillna(True).astype(bool)
        else:
            different_po = missing_po
        c = c[(c["_order_o"] < c["_order_d"]) & different_po]
        tol_c = np.maximum(cfg["amount_tolerance_abs"], cfg["amount_tolerance_pct"] * c["total_amount_o"].abs())
        c = c[((c["total_amount_d"] - c["total_amount_o"]).abs() <= tol_c)
              & ((c["invoice_date_d"] - c["invoice_date_o"]).dt.days <= cfg["cross_po_window_days"])]
        frames.append(c.assign(match_type="same vendor, SKU, quantity and amount across PO references",
                               _matcher="C"))

    pairs = pd.concat(frames, ignore_index=True)
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    pairs = pairs.sort_values(["_order_d", "_order_o"]).drop_duplicates("invoice_id_d")
    return pd.DataFrame({
        "invoice_id": pairs["invoice_id_d"].astype("int64").values,
        "original_invoice_id": pairs["invoice_id_o"].astype("int64").values,
        "original_number": pairs["invoice_number_o"].values,
        "match_type": pairs["match_type"].values,
        "matcher": pairs["_matcher"].values,
    })


def _same_po_duplicate_ids(invoices, pos, config) -> set:
    """Duplicates that sit on the same PO as their original. R2 and R7 reason about
    billing per PO, so only these should be removed before they aggregate."""
    d = find_duplicate_invoices(invoices, pos, config)
    if d.empty:
        return set()
    inv_po = invoices.set_index("invoice_id")["po_id"]
    same = inv_po.reindex(d["invoice_id"]).values == inv_po.reindex(d["original_invoice_id"]).values
    return set(d.loc[same, "invoice_id"])


def rule_duplicate_invoice(invoices: pd.DataFrame, pos: pd.DataFrame | None = None,
                           config: dict | None = None) -> pd.DataFrame:
    """R1: flag the later invoice of each duplicate pair."""
    dups = find_duplicate_invoices(invoices, pos, config)
    if dups.empty:
        return _empty_flags()
    df = dups.merge(invoices, on="invoice_id", how="left")
    details = [f"Invoice {n} duplicates invoice {o} ({orig}); matched on {m}."
               for n, o, orig, m in zip(df["invoice_number"], df["original_number"],
                                        df["original_invoice_id"], df["match_type"], strict=True)]
    return _make_flags("R1", df, details=details, amount="total_amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# R2 three-way match
# --------------------------------------------------------------------------- #
def rule_three_way_match(invoices: pd.DataFrame, pos: pd.DataFrame, receipts: pd.DataFrame,
                         config: dict | None = None, duplicate_ids: set | None = None) -> pd.DataFrame:
    """R2: reconcile *total* invoiced quantity per PO against ordered and received
    quantity. Aggregating per PO is what lets partial deliveries (several receipts,
    several invoices) reconcile cleanly. Invoices already identified as duplicates
    are excluded so a duplicate is reported once, by R1, not twice.

    A PO with no receipt at all is only compared against the PO quantity; the
    missing receipt itself is R9's concern."""
    cfg = load_config(overrides=config)
    tol = cfg["three_way_match"]["qty_tolerance"]
    if invoices.empty or pos.empty:
        return _empty_flags()
    if duplicate_ids is None:
        duplicate_ids = _same_po_duplicate_ids(invoices, pos, config)

    inv = _dates(_with_po(invoices), "invoice_date")
    inv = inv[~inv["invoice_id"].isin(duplicate_ids)]
    inv = inv.sort_values(["invoice_date", "invoice_id"])
    per_po = inv.groupby("po_id").agg(
        invoiced_qty=("quantity", "sum"), n_invoices=("invoice_id", "size"),
        invoice_id=("invoice_id", "last"), vendor_id=("vendor_id", "last"),
        unit_price=("unit_price", "last"),
    ).reset_index()
    per_po = per_po.merge(pos[["po_id", "quantity"]].rename(columns={"quantity": "po_qty"}), on="po_id")
    received = receipts.groupby("po_id")["received_qty"].sum().rename("received_qty") \
        if not receipts.empty else pd.Series(dtype=float, name="received_qty")
    per_po = per_po.merge(received, left_on="po_id", right_index=True, how="left")

    has_gr = per_po["received_qty"].notna()
    vs_po = (per_po["invoiced_qty"] - per_po["po_qty"]).abs() > tol
    vs_gr = has_gr & ((per_po["invoiced_qty"] - per_po["received_qty"]).abs() > tol)
    hit = per_po[vs_po | vs_gr].copy()
    if hit.empty:
        return _empty_flags()

    billable = np.where(hit["received_qty"].notna(),
                        np.minimum(hit["po_qty"], hit["received_qty"].fillna(np.inf)), hit["po_qty"])
    over_qty = np.maximum(hit["invoiced_qty"] - billable, 0)
    hit["amount"] = over_qty * hit["unit_price"]
    hit["severity"] = np.where(over_qty > tol, "high", "low")
    details = []
    for iq, pq, rq, n, ov in zip(hit["invoiced_qty"], hit["po_qty"], hit["received_qty"],
                                 hit["n_invoices"], over_qty, strict=True):
        recv = "no receipt on file" if pd.isna(rq) else f"{int(rq)}"
        kind = f"over-billed by {int(ov)} units" if ov > tol else "under-billed (no exposure)"
        details.append(f"Invoiced qty {int(iq)} across {n} invoice(s) vs PO qty {int(pq)} "
                       f"vs received {recv}: {kind}.")
    return _make_flags("R2", hit, details=details, amount="amount", severity="severity",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# R3 contract price
# --------------------------------------------------------------------------- #
def _effective_contract(invoices: pd.DataFrame, pos: pd.DataFrame | None,
                        contracts: pd.DataFrame) -> pd.DataFrame:
    """Attach the contract that governs each invoice: the PO's contract when the PO
    references one, otherwise the vendor/SKU contract in force on the invoice date."""
    inv = _dates(invoices, "invoice_date")
    con = _dates(contracts, "start_date", "end_date")
    if pos is not None and not pos.empty and "contract_id" in pos.columns:
        inv = inv.merge(pos[["po_id", "contract_id"]], on="po_id", how="left")
    else:
        inv["contract_id"] = np.nan

    linked = inv[inv["contract_id"].notna()].merge(
        con[["contract_id", "contract_price", "tax_rate"]], on="contract_id", how="inner")

    unlinked = inv[inv["contract_id"].isna()].drop(columns="contract_id")
    by_date = unlinked.merge(con, on=["vendor_id", "item_sku"], how="inner")
    if not by_date.empty:
        in_force = (by_date["invoice_date"] >= by_date["start_date"]) & \
                   (by_date["invoice_date"] <= by_date["end_date"])
        by_date = by_date[in_force].sort_values("start_date").drop_duplicates("invoice_id", keep="last")
        by_date = by_date.drop(columns=["start_date", "end_date"])
    return pd.concat([linked, by_date], ignore_index=True)


def rule_contract_price(invoices: pd.DataFrame, contracts: pd.DataFrame, pos: pd.DataFrame | None = None,
                        config: dict | None = None) -> pd.DataFrame:
    """R3: unit price above the contract price in force (amendments respected)."""
    cfg = load_config(overrides=config)["contract_price"]
    if invoices.empty or contracts.empty:
        return _empty_flags()
    df = _effective_contract(invoices, pos, contracts)
    if df.empty:
        return _empty_flags()
    df = df[df["unit_price"] > df["contract_price"] * (1 + cfg["price_tolerance_pct"])].copy()
    df["amount"] = (df["unit_price"] - df["contract_price"]) * df["quantity"]
    details = [f"Invoiced at {p:,.2f} vs contracted {c:,.2f} for {s} (contract {int(k)})."
               for p, c, s, k in zip(df["unit_price"], df["contract_price"], df["item_sku"], df["contract_id"], strict=True)]
    return _make_flags("R3", df, details=details, amount="amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# R4 missing PO
# --------------------------------------------------------------------------- #
def rule_missing_po(invoices: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """R4: invoice with no purchase order."""
    if invoices.empty:
        return _empty_flags()
    df = invoices[invoices["po_id"].isna()]
    details = [f"Invoice {n} has no linked purchase order." for n in df["invoice_number"]]
    return _make_flags("R4", df, details=details, amount="total_amount",
                       vendor_id="vendor_id", invoice_id="invoice_id")


# --------------------------------------------------------------------------- #
# R5 duplicate payment
# --------------------------------------------------------------------------- #
def rule_duplicate_payment(payments: pd.DataFrame, invoices: pd.DataFrame | None = None,
                           config: dict | None = None) -> pd.DataFrame:
    """R5: every payment after the earliest one on the same invoice.

    Sorted explicitly by (payment_date, payment_id): SQL gives no row-order
    guarantee, and relying on arrival order once flagged the original payment
    instead of the duplicate."""
    if payments.empty:
        return _empty_flags()
    pay = _dates(payments, "payment_date").sort_values(["invoice_id", "payment_date", "payment_id"])
    counts = pay.groupby("invoice_id")["payment_id"].transform("size")
    extra = pay[pay.duplicated("invoice_id", keep="first")].copy()
    extra["n"] = counts[extra.index]
    if invoices is not None and not invoices.empty:
        extra = extra.merge(invoices[["invoice_id", "vendor_id", "po_id"]], on="invoice_id", how="left")
    else:
        extra["vendor_id"] = pd.NA
        extra["po_id"] = pd.NA
    details = [f"Invoice {i} has {n} payments; {ref} is a repeat payment."
               for i, n, ref in zip(extra["invoice_id"], extra["n"], extra["payment_ref"], strict=True)]
    return _make_flags("R5", extra, details=details, amount="payment_amount", vendor_id="vendor_id",
                       invoice_id="invoice_id", po_id="po_id", payment_id="payment_id")


# --------------------------------------------------------------------------- #
# R6 tax validation
# --------------------------------------------------------------------------- #
def rule_tax_validation(invoices: pd.DataFrame, contracts: pd.DataFrame, pos: pd.DataFrame,
                        config: dict | None = None) -> pd.DataFrame:
    """R6: tax should equal subtotal x contracted rate. Only checked where the PO
    references a contract, the one place with an authoritative rate."""
    cfg = load_config(overrides=config)["tax_validation"]
    if invoices.empty or contracts.empty or pos.empty:
        return _empty_flags()
    df = _with_po(invoices).merge(pos[["po_id", "contract_id"]], on="po_id")
    df = df[df["contract_id"].notna()].merge(contracts[["contract_id", "tax_rate"]], on="contract_id")
    df["expected"] = (df["subtotal"] * df["tax_rate"]).round(2)
    df = df[df["expected"] != 0]
    diff = (df["tax_amount"] - df["expected"]).abs()
    tol = np.maximum(cfg["tolerance_pct"] * df["expected"].abs(), cfg["tolerance_abs"])
    df = df[diff > tol].copy()
    df["amount"] = (df["tax_amount"] - df["expected"]).abs()
    details = [f"Tax charged {t:,.2f} vs expected {e:,.2f} (contract rate {r * 100:.2f}%)."
               for t, e, r in zip(df["tax_amount"], df["expected"], df["tax_rate"], strict=True)]
    return _make_flags("R6", df, details=details, amount="amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# R7 invoice splitting
# --------------------------------------------------------------------------- #
def rule_invoice_splitting(invoices: pd.DataFrame, pos: pd.DataFrame | None = None,
                           receipts: pd.DataFrame | None = None, config: dict | None = None,
                           duplicate_ids: set | None = None) -> pd.DataFrame:
    """R7: invoices on one PO, each under the approval limit, together over it, filed
    within a short window. Two exclusions keep this precise:
      * duplicates (R1) are removed first: a resubmitted invoice is not a split;
      * POs with at least as many goods receipts as invoices are skipped: billing
        that follows separate deliveries is partial billing, not splitting."""
    cfg = load_config(overrides=config)
    threshold = cfg["approval_threshold"]
    window = cfg["invoice_splitting"]["window_days"]
    if invoices.empty:
        return _empty_flags()
    if duplicate_ids is None:
        duplicate_ids = _same_po_duplicate_ids(invoices, pos, config)

    inv = _dates(_with_po(invoices), "invoice_date")
    inv = inv[~inv["invoice_id"].isin(duplicate_ids)]
    g = inv.groupby("po_id").agg(n=("invoice_id", "size"), max_total=("total_amount", "max"),
                                 sum_total=("total_amount", "sum"), first=("invoice_date", "min"),
                                 last=("invoice_date", "max"))
    g["window"] = (g["last"] - g["first"]).dt.days
    if receipts is not None and not receipts.empty:
        g["n_receipts"] = receipts.groupby("po_id").size().reindex(g.index).fillna(0)
    else:
        g["n_receipts"] = 0
    hit = g[(g["n"] >= 2) & (g["max_total"] < threshold) & (g["sum_total"] >= threshold)
            & (g["window"] <= window) & (g["n_receipts"] < g["n"])]
    if hit.empty:
        return _empty_flags()
    df = inv[inv["po_id"].isin(hit.index)].merge(hit, left_on="po_id", right_index=True)
    details = [f"1 of {n} invoices on PO {p} totalling {s:,.2f} filed within {w} days, "
               f"each under the {threshold:,.0f} approval limit."
               for n, p, s, w in zip(df["n"], df["po_id"], df["sum_total"], df["window"], strict=True)]
    return _make_flags("R7", df, details=details, amount="total_amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# R8 overpayment
# --------------------------------------------------------------------------- #
def rule_overpayment(payments: pd.DataFrame, invoices: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """R8: payment exceeds the invoice total beyond tolerance."""
    cfg = load_config(overrides=config)["overpayment"]
    if payments.empty or invoices.empty:
        return _empty_flags()
    df = payments.merge(invoices[["invoice_id", "vendor_id", "po_id", "total_amount"]], on="invoice_id")
    diff = df["payment_amount"] - df["total_amount"]
    tol = np.maximum(cfg["tolerance_abs"], cfg["tolerance_pct"] * df["total_amount"].abs())
    df = df[diff > tol].copy()
    df["amount"] = df["payment_amount"] - df["total_amount"]
    details = [f"Paid {p:,.2f} against invoice total {t:,.2f}."
               for p, t in zip(df["payment_amount"], df["total_amount"], strict=True)]
    return _make_flags("R8", df, details=details, amount="amount", vendor_id="vendor_id",
                       invoice_id="invoice_id", po_id="po_id", payment_id="payment_id")


# --------------------------------------------------------------------------- #
# R9 payment without goods receipt
# --------------------------------------------------------------------------- #
def rule_payment_without_gr(invoices: pd.DataFrame, payments: pd.DataFrame, receipts: pd.DataFrame,
                            config: dict | None = None) -> pd.DataFrame:
    """R9: payment against a PO with no goods receipt at all. Distinct from R2, which
    covers receipts that exist but disagree with the invoice."""
    if payments.empty or invoices.empty:
        return _empty_flags()
    inv = _with_po(invoices)
    received = set(receipts["po_id"].unique()) if not receipts.empty else set()
    inv = inv[~inv["po_id"].isin(received)]
    df = payments.merge(inv[["invoice_id", "po_id", "vendor_id"]], on="invoice_id")
    details = [f"Payment {ref} made against PO {p} with no goods receipt on file."
               for ref, p in zip(df["payment_ref"], df["po_id"], strict=True)]
    return _make_flags("R9", df, details=details, amount="payment_amount", vendor_id="vendor_id",
                       invoice_id="invoice_id", po_id="po_id", payment_id="payment_id")


# --------------------------------------------------------------------------- #
# R10 duplicate vendor record
# --------------------------------------------------------------------------- #
def rule_duplicate_vendor(vendors: pd.DataFrame, invoices: pd.DataFrame | None = None,
                          payments: pd.DataFrame | None = None, config: dict | None = None) -> pd.DataFrame:
    """R10: vendor shares a bank account or tax ID with an earlier vendor record.
    The later record is flagged; exposure is everything paid to it."""
    if vendors.empty:
        return _empty_flags()
    v = _dates(vendors, "created_at")
    sort_cols = ["created_at", "vendor_id"] if "created_at" in v.columns else ["vendor_id"]
    v = v.sort_values(sort_cols)
    found = []
    for key, label in (("bank_account", "bank account"), ("tax_id", "tax ID")):
        if key not in v.columns:
            continue
        sub = v[v[key].notna() & (v[key].astype(str).str.strip() != "")]
        first = sub.drop_duplicates(key, keep="first")[[key, "vendor_id", "vendor_name"]]
        later = sub[sub.duplicated(key, keep="first")]
        m = later.merge(first, on=key, suffixes=("", "_orig"))
        m["shared"] = label
        found.append(m)
    if not found:
        return _empty_flags()
    df = pd.concat(found, ignore_index=True).drop_duplicates("vendor_id")
    if df.empty:
        return _empty_flags()

    paid = pd.Series(dtype=float)
    if invoices is not None and payments is not None and not payments.empty:
        paid = _invoice_vendor(payments, invoices).groupby("vendor_id")["payment_amount"].sum()
    df["amount"] = df["vendor_id"].map(paid).fillna(0.0)
    details = []
    for name, oid, oname, shared in zip(df["vendor_name"], df["vendor_id_orig"],
                                        df["vendor_name_orig"], df["shared"], strict=True):
        sim = SequenceMatcher(None, str(name).lower(), str(oname).lower()).ratio()
        details.append(f"'{name}' shares a {shared} with vendor {oid} '{oname}' "
                       f"(name similarity {sim:.0%}).")
    return _make_flags("R10", df, details=details, amount="amount", vendor_id="vendor_id")


# --------------------------------------------------------------------------- #
# R11 new vendor large payment
# --------------------------------------------------------------------------- #
def rule_new_vendor_payment(vendors: pd.DataFrame, invoices: pd.DataFrame, payments: pd.DataFrame,
                            config: dict | None = None) -> pd.DataFrame:
    """R11: large payment within N days of the vendor record being created."""
    cfg = load_config(overrides=config)["new_vendor_payment"]
    if vendors.empty or payments.empty or "created_at" not in vendors.columns:
        return _empty_flags()
    df = _invoice_vendor(_dates(payments, "payment_date"), invoices)
    df = df.merge(_dates(vendors, "created_at")[["vendor_id", "created_at"]], on="vendor_id")
    df["age_days"] = (df["payment_date"] - df["created_at"]).dt.days
    df = df[(df["age_days"] >= 0) & (df["age_days"] <= cfg["window_days"])
            & (df["payment_amount"] >= cfg["min_payment"])]
    details = [f"Paid {a:,.2f} ({ref}) {d} days after the vendor record was created."
               for a, ref, d in zip(df["payment_amount"], df["payment_ref"], df["age_days"], strict=True)]
    return _make_flags("R11", df, details=details, amount="payment_amount", vendor_id="vendor_id",
                       invoice_id="invoice_id", po_id="po_id", payment_id="payment_id")


# --------------------------------------------------------------------------- #
# A1 Benford's Law
# --------------------------------------------------------------------------- #
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def first_digit(amounts: pd.Series) -> pd.Series:
    x = pd.to_numeric(amounts, errors="coerce").abs()
    x = x[x >= 1]
    return (x // 10 ** np.floor(np.log10(x))).astype(int)


def benford_test(amounts: pd.Series) -> dict:
    """First-digit test: chi-square p-value plus Nigrini's mean absolute deviation."""
    from scipy.stats import chi2

    d = first_digit(amounts)
    n = len(d)
    observed = np.bincount(d, minlength=10)[1:10]
    expected = BENFORD * n
    stat = float(((observed - expected) ** 2 / expected).sum()) if n else 0.0
    p_obs = observed / n if n else np.zeros(9)
    return {
        "n": n,
        "chi2": stat,
        "p_value": float(chi2.sf(stat, 8)) if n else 1.0,
        "mad": float(np.abs(p_obs - BENFORD).mean()),
        "observed": p_obs.tolist(),
        "top_digit": int(np.argmax(p_obs - BENFORD) + 1),
    }


def rule_benford(invoices: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """A1: vendors whose invoice amounts do not follow Benford's Law."""
    cfg = load_config(overrides=config)["benford"]
    if invoices.empty:
        return _empty_flags()
    counts = invoices.groupby("vendor_id").size()
    rows = []
    for vid in counts[counts >= cfg["min_invoices"]].index:
        res = benford_test(invoices.loc[invoices["vendor_id"] == vid, "total_amount"])
        if res["p_value"] < cfg["p_value"] and res["mad"] > cfg["mad_threshold"]:
            d = res["top_digit"]
            rows.append({"vendor_id": vid, "details":
                         f"First-digit MAD {res['mad']:.3f} (> {cfg['mad_threshold']}), chi-square "
                         f"p = {res['p_value']:.1e} over {res['n']} invoices; digit {d} appears "
                         f"{res['observed'][d - 1]:.0%} vs {BENFORD[d - 1]:.0%} expected."})
    if not rows:
        return _empty_flags()
    df = pd.DataFrame(rows)
    return _make_flags("A1", df, details=df["details"], amount=np.zeros(len(df)), vendor_id="vendor_id")


# --------------------------------------------------------------------------- #
# A2 unit price outlier
# --------------------------------------------------------------------------- #
def rule_price_outlier(invoices: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """A2: unit price far above the organisation's typical price for the SKU,
    using a robust z-score on log price (median / MAD, so outliers can't mask
    themselves by inflating the mean and standard deviation)."""
    cfg = load_config(overrides=config)["price_outlier"]
    if invoices.empty:
        return _empty_flags()
    df = invoices[invoices["unit_price"] > 0].copy()
    df["lp"] = np.log(df["unit_price"])
    grp = df.groupby("item_sku")["lp"]
    df["n"] = grp.transform("size")
    df["med"] = grp.transform("median")
    df["mad"] = (df["lp"] - df["med"]).abs().groupby(df["item_sku"]).transform("median")
    df = df[df["n"] >= cfg["min_observations"]]
    # Floor the spread: a SKU bought mostly under one tight contract has a tiny MAD,
    # which would turn an ordinary market-rate purchase into a "10-sigma" outlier.
    mad = np.maximum(df["mad"], cfg.get("min_log_mad", 0.05))
    df["z"] = 0.6745 * (df["lp"] - df["med"]) / mad
    df = df[df["z"] > cfg["robust_z_threshold"]].copy()
    df["typical"] = np.exp(df["med"])
    df["amount"] = (df["unit_price"] - df["typical"]) * df["quantity"]
    details = [f"Unit price {p:,.2f} for {s} is {p / t - 1:.0%} above the typical {t:,.2f} "
               f"(robust z = {z:.1f})."
               for p, s, t, z in zip(df["unit_price"], df["item_sku"], df["typical"], df["z"], strict=True)]
    return _make_flags("A2", df, details=details, amount="amount",
                       vendor_id="vendor_id", invoice_id="invoice_id", po_id="po_id")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_all_rules(vendors, contracts, pos, receipts, invoices, payments,
                  config: dict | None = None, rules: list[str] | None = None) -> pd.DataFrame:
    """Run every detector (or the subset in `rules`) and return one flags frame."""
    cfg = load_config(overrides=config)
    selected = set(rules) if rules else set(RULES)
    needs_dups = selected & {"R2", "R7"}
    dup_ids = _same_po_duplicate_ids(invoices, pos, cfg) if needs_dups and not invoices.empty else set()

    runners = {
        "R1": lambda: rule_duplicate_invoice(invoices, pos, cfg),
        "R2": lambda: rule_three_way_match(invoices, pos, receipts, cfg, dup_ids),
        "R3": lambda: rule_contract_price(invoices, contracts, pos, cfg),
        "R4": lambda: rule_missing_po(invoices, cfg),
        "R5": lambda: rule_duplicate_payment(payments, invoices, cfg),
        "R6": lambda: rule_tax_validation(invoices, contracts, pos, cfg),
        "R7": lambda: rule_invoice_splitting(invoices, pos, receipts, cfg, dup_ids),
        "R8": lambda: rule_overpayment(payments, invoices, cfg),
        "R9": lambda: rule_payment_without_gr(invoices, payments, receipts, cfg),
        "R10": lambda: rule_duplicate_vendor(vendors, invoices, payments, cfg),
        "R11": lambda: rule_new_vendor_payment(vendors, invoices, payments, cfg),
        "A1": lambda: rule_benford(invoices, cfg),
        "A2": lambda: rule_price_outlier(invoices, cfg),
    }
    results = [runners[c]() for c in RULES if c in selected]
    results = [r for r in results if not r.empty]
    if not results:
        return _empty_flags()
    return pd.concat(results, ignore_index=True)[FLAG_COLUMNS]
