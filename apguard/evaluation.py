"""Scores detectors against the generator's ground truth.

Matching is done at the level each detector reports on: invoice-level rules are
matched on invoice_id, payment-level rules on payment_id (R9 on invoice_id, since
the defect is the unreceived invoice), vendor-level detectors on vendor_id.
Precision and recall are computed over *unique entities*, so a rule that flags the
same invoice twice is not rewarded twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rules.registry import RULES

# detector -> (ground-truth exception type, id column used for matching)
TARGETS: dict[str, tuple[str, str]] = {
    "R1": ("duplicate_invoice", "invoice_id"),
    "R2": ("three_way_mismatch", "invoice_id"),
    "R3": ("contract_price_violation", "invoice_id"),
    "R4": ("missing_po", "invoice_id"),
    "R5": ("duplicate_payment", "payment_id"),
    "R6": ("tax_mismatch", "invoice_id"),
    "R7": ("invoice_splitting", "invoice_id"),
    "R8": ("overpayment", "payment_id"),
    "R9": ("payment_without_gr", "invoice_id"),
    "R10": ("duplicate_vendor", "vendor_id"),
    "R11": ("new_vendor_large_payment", "payment_id"),
    "A1": ("threshold_hugging", "vendor_id"),
    "A2": ("price_outlier", "invoice_id"),
}

TABLES = ["vendors", "contracts", "purchase_orders", "goods_receipts", "invoices", "payments"]


@dataclass
class Dataset:
    vendors: pd.DataFrame
    contracts: pd.DataFrame
    pos: pd.DataFrame
    receipts: pd.DataFrame
    invoices: pd.DataFrame
    payments: pd.DataFrame
    truth: pd.DataFrame
    benign: pd.DataFrame

    @property
    def tables(self):
        return (self.vendors, self.contracts, self.pos, self.receipts, self.invoices, self.payments)


def load_dataset(data_dir: Path | str) -> Dataset:
    d = Path(data_dir)
    t = {n: pd.read_csv(d / f"{n}.csv") for n in TABLES}
    for col in ("po_id",):
        t["invoices"][col] = t["invoices"][col].astype("Int64")
    benign_path = d / "benign_lookalikes.csv"
    return Dataset(t["vendors"], t["contracts"], t["purchase_orders"], t["goods_receipts"],
                   t["invoices"], t["payments"], pd.read_csv(d / "injected_exceptions.csv"),
                   pd.read_csv(benign_path) if benign_path.exists() else pd.DataFrame())


def _ids(series: pd.Series) -> set[int]:
    return set(pd.to_numeric(series, errors="coerce").dropna().astype(int))


def score_rule(code: str, flags: pd.DataFrame, truth: pd.DataFrame) -> dict:
    exc_type, key = TARGETS[code]
    flagged = _ids(flags.loc[flags["rule_code"] == code, key])
    expected = _ids(truth.loc[truth["exception_type"] == exc_type, key])
    any_truth = _ids(truth[key])
    tp = flagged & expected
    p = len(tp) / len(flagged) if flagged else float("nan")
    r = len(tp) / len(expected) if expected else float("nan")
    f1 = 2 * p * r / (p + r) if flagged and expected and (p + r) else float("nan")
    return {
        "rule": code, "name": RULES[code].name, "category": RULES[code].category,
        "exception_type": exc_type, "level": key.replace("_id", ""),
        "injected": len(expected), "flagged": len(flagged), "true_positives": len(tp),
        "false_positives": len(flagged - expected),
        # flagged entities carrying a *different* injected exception: real problems,
        # caught by the "wrong" rule. Not meaningful for vendor-level detectors.
        "fp_other_exception": len((flagged - expected) & any_truth) if key != "vendor_id" else 0,
        "missed": len(expected - flagged),
        "precision": p, "recall": r, "f1": f1,
    }


def score_all(flags: pd.DataFrame, truth: pd.DataFrame, codes=None) -> pd.DataFrame:
    return pd.DataFrame([score_rule(c, flags, truth) for c in (codes or TARGETS)])


def dollar_recall(flags: pd.DataFrame, ds: Dataset) -> dict:
    """Share of injected *leakage dollars* that sit on an entity some control flagged.

    Leakage per exception: duplicate invoice/payment = full amount; overpayment =
    excess paid; contract price / tax = variance; mismatch = over-billed value; missing
    PO / no receipt / new vendor = full amount at risk. Splitting is a control bypass
    rather than a loss, and vendor-level patterns have no per-invoice amount, so
    neither contributes dollars here."""
    inv = ds.invoices.set_index("invoice_id")
    pay = ds.payments.set_index("payment_id")
    truth = ds.truth.copy()
    po_price = ds.pos.set_index("po_id")["unit_price"]
    po_qty = ds.pos.set_index("po_id")["quantity"]
    contract_price = ds.contracts.set_index("contract_id")["contract_price"]
    contract_tax = ds.contracts.set_index("contract_id")["tax_rate"]
    po_contract = ds.pos.set_index("po_id")["contract_id"]

    def leak(row) -> float:
        t, i, p = row["exception_type"], row["invoice_id"], row["payment_id"]
        try:
            if t in ("duplicate_invoice", "missing_po", "payment_without_gr"):
                return float(inv.loc[i, "total_amount"])
            if t in ("duplicate_payment", "new_vendor_large_payment"):
                return float(pay.loc[p, "payment_amount"])
            if t == "overpayment":
                return float(pay.loc[p, "payment_amount"] - inv.loc[i, "total_amount"])
            if t == "contract_price_violation":
                c = contract_price[po_contract[row["po_id"]]]
                return float((inv.loc[i, "unit_price"] - c) * inv.loc[i, "quantity"])
            if t == "tax_mismatch":
                rate = contract_tax[po_contract[row["po_id"]]]
                return float(abs(inv.loc[i, "tax_amount"] - round(inv.loc[i, "subtotal"] * rate, 2)))
            if t == "three_way_mismatch":
                over = inv.loc[i, "quantity"] - po_qty[row["po_id"]]
                return float(max(over, 0) * po_price[row["po_id"]])
        except KeyError:
            return 0.0
        return 0.0

    truth["leak"] = truth.apply(leak, axis=1)
    truth = truth[truth["leak"] > 0]
    controls = flags[flags["category"] == "control"]
    flagged_inv = _ids(controls["invoice_id"])
    flagged_pay = _ids(controls["payment_id"])
    payment_level = truth["exception_type"].isin(["duplicate_payment", "overpayment", "new_vendor_large_payment"])
    caught = np.where(payment_level, truth["payment_id"].isin(flagged_pay), truth["invoice_id"].isin(flagged_inv))
    total = float(truth["leak"].sum())
    return {"injected_leakage": round(total, 2), "leakage_on_flagged_entities": round(float(truth.loc[caught, "leak"].sum()), 2),
            "dollar_recall": float(truth.loc[caught, "leak"].sum() / total) if total else float("nan")}


def benign_false_positives(flags: pd.DataFrame, benign: pd.DataFrame, pos: pd.DataFrame,
                           invoices: pd.DataFrame, truth: pd.DataFrame | None = None) -> pd.DataFrame:
    """How often each legitimate look-alike pattern was wrongly flagged, by detector.
    Entities that also carry an injected exception are excluded: a flag there is
    deserved, whatever else is true of the entity."""
    if benign.empty:
        return pd.DataFrame(columns=["pattern", "entities", "flagged", "flagged_by"])
    if truth is not None and not flags.empty:
        bad_inv, bad_pay, bad_ven = _ids(truth["invoice_id"]), _ids(truth["payment_id"]), \
            _ids(truth.loc[truth["exception_type"].isin(["duplicate_vendor", "threshold_hugging"]), "vendor_id"])
        lvl = flags["rule_code"].map(lambda c: RULES[c].level)
        deserved = ((lvl == "invoice") & flags["invoice_id"].isin(bad_inv)) | \
                   ((lvl == "payment") & (flags["payment_id"].isin(bad_pay) | flags["invoice_id"].isin(bad_inv))) | \
                   ((lvl == "vendor") & flags["vendor_id"].isin(bad_ven))
        flags = flags[~deserved]
    rows = []
    inv_by_po = invoices.dropna(subset=["po_id"]).groupby("po_id")["invoice_id"].apply(set)
    for pattern, grp in benign.groupby("pattern"):
        if pattern in ("partial_delivery", "recurring_billing"):
            ids = set().union(*[inv_by_po.get(p, set()) for p in grp["po_id"].dropna().astype(int)])
            hit = flags[flags["invoice_id"].isin(ids)]
            n = grp["po_id"].nunique()
            flagged = hit["po_id"].nunique()
        elif pattern in ("tax_rounding", "payment_rounding"):
            ids = _ids(grp["invoice_id"])
            hit = flags[flags["invoice_id"].isin(ids)]
            n, flagged = len(ids), hit["invoice_id"].nunique()
        elif pattern == "contract_amendment":
            cids = _ids(grp["contract_id"])
            po_ids = set(pos.loc[pos["contract_id"].isin(cids), "po_id"])
            ids = set().union(*[inv_by_po.get(p, set()) for p in po_ids]) if po_ids else set()
            hit = flags[flags["invoice_id"].isin(ids)]
            n, flagged = len(ids), hit["invoice_id"].nunique()
        else:  # vendor-level
            ids = _ids(grp["vendor_id"])
            hit = flags[flags["vendor_id"].isin(ids)]
            n, flagged = len(ids), hit["vendor_id"].nunique()
        rows.append({"pattern": pattern, "entities": n, "flagged": int(flagged),
                     "flagged_by": ", ".join(sorted(hit["rule_code"].unique())) or "-"})
    return pd.DataFrame(rows)


def overlap_matrix(flags: pd.DataFrame, codes=None) -> pd.DataFrame:
    """Invoices flagged by both rule i and rule j (diagonal = invoices flagged by i)."""
    codes = codes or [c for c in RULES if RULES[c].level != "vendor"]
    sets = {c: _ids(flags.loc[flags["rule_code"] == c, "invoice_id"]) for c in codes}
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in codes] for a in codes], index=codes, columns=codes)
