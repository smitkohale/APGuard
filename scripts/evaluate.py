"""Offline evaluation: runs every detector on the generated CSVs and writes
reports/evaluation.json and reports/EVALUATION.md. No database needed, so CI
runs it on every push.

    python scripts/evaluate.py [--data data/generated] [--out reports]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard.evaluation import (  # noqa: E402
    TARGETS,
    benign_false_positives,
    dollar_recall,
    load_dataset,
    overlap_matrix,
    score_all,
    score_rule,
)
from rules.baseline import BASELINE_RULES, run_baseline  # noqa: E402
from rules.config import load_config  # noqa: E402
from rules.engine import run_all_rules  # noqa: E402
from rules.registry import RULES  # noqa: E402


def sweep(ds, code, label, values, make_override):
    rows = []
    for v in values:
        flags = run_all_rules(*ds.tables, config=make_override(v), rules=[code])
        s = score_rule(code, flags, ds.truth)
        rows.append({"parameter": label, "value": v, "flagged": s["flagged"],
                     "precision": s["precision"], "recall": s["recall"], "f1": s["f1"]})
    return rows


def isolation_forest_benchmark(ds, control_flags) -> dict:
    """Unsupervised baseline: score every invoice with an Isolation Forest on generic
    features, flag the top-k (k = number of injected invoice-level exceptions) and
    compare recall per exception type with the rules."""
    from sklearn.ensemble import IsolationForest

    inv = ds.invoices.copy()
    pos, pay, gr = ds.pos, ds.payments, ds.receipts
    inv["log_total"] = np.log1p(inv["total_amount"].clip(lower=0))
    inv["price_ratio"] = inv["unit_price"] / inv.groupby("item_sku")["unit_price"].transform("median")
    po_qty = pos.set_index("po_id")["quantity"]
    inv["qty_ratio"] = (inv["quantity"] / inv["po_id"].map(po_qty)).fillna(1.0).astype(float)
    inv["has_po"] = inv["po_id"].notna().astype(int)
    inv["tax_ratio"] = (inv["tax_amount"] / inv["subtotal"].replace(0, np.nan)).fillna(0)
    inv["invoices_on_po"] = inv["po_id"].map(inv.groupby("po_id").size()).fillna(1).astype(float)
    inv["has_receipt"] = inv["po_id"].isin(gr["po_id"]).astype(int)
    p = pay.groupby("invoice_id").agg(n_pay=("payment_id", "size"), paid=("payment_amount", "sum"))
    inv = inv.merge(p, left_on="invoice_id", right_index=True, how="left").fillna({"n_pay": 0, "paid": 0})
    inv["paid_ratio"] = inv["paid"] / inv["total_amount"].replace(0, np.nan)
    created = pd.to_datetime(ds.vendors.set_index("vendor_id")["created_at"])
    inv["vendor_age"] = (pd.to_datetime(inv["invoice_date"]) - inv["vendor_id"].map(created)).dt.days.clip(lower=0)
    feats = ["log_total", "price_ratio", "qty_ratio", "has_po", "tax_ratio", "invoices_on_po",
             "has_receipt", "n_pay", "paid_ratio", "vendor_age"]
    X = inv[feats].fillna(0).to_numpy()

    model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1).fit(X)
    inv["score"] = -model.score_samples(X)

    invoice_types = [t for t, k in TARGETS.values() if k in ("invoice_id", "payment_id")]
    truth = ds.truth[ds.truth["exception_type"].isin(invoice_types)]
    bad = set(truth["invoice_id"].dropna().astype(int))
    k = len(bad)
    top = set(inv.nlargest(k, "score")["invoice_id"])
    rules_hit = set(control_flags["invoice_id"].dropna().astype(int))
    per_type = []
    for t, grp in truth.groupby("exception_type"):
        ids = set(grp["invoice_id"].dropna().astype(int))
        per_type.append({"exception_type": t, "injected": len(ids),
                         "isolation_forest_recall": len(ids & top) / len(ids),
                         "rules_recall": len(ids & rules_hit) / len(ids)})
    return {"features": feats, "k": k,
            "isolation_forest_precision": len(top & bad) / k,
            "rules_precision": len(rules_hit & bad) / len(rules_hit) if rules_hit else float("nan"),
            "per_type": per_type}


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:,}"


def md_table(df: pd.DataFrame, pct_cols=(), money_cols=()) -> str:
    cols = list(df.columns)
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if numeric[c] else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append(fmt(v))
            elif c in money_cols:
                cells.append(f"${v:,.0f}")
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    ds = load_dataset(args.data)
    cfg = load_config()

    t0 = time.perf_counter()
    flags = run_all_rules(*ds.tables)
    runtime = time.perf_counter() - t0
    scores = score_all(flags, ds.truth)
    controls = flags[flags["category"] == "control"]

    # exact vs fuzzy duplicate matching, per variant
    exact = run_all_rules(*ds.tables, config={"duplicate_invoice": {"fuzzy": False}}, rules=["R1"])
    dup_truth = ds.truth[ds.truth["exception_type"] == "duplicate_invoice"]
    variants = []
    exact_ids = set(exact["invoice_id"].dropna().astype(int))
    fuzzy_ids = set(flags.loc[flags.rule_code == "R1", "invoice_id"].dropna().astype(int))
    for v, grp in dup_truth.groupby("variant"):
        ids = set(grp["invoice_id"].astype(int))
        variants.append({"variant": v, "injected": len(ids),
                         "exact_recall": len(ids & exact_ids) / len(ids),
                         "apguard_recall": len(ids & fuzzy_ids) / len(ids)})

    # naive baseline
    base = run_baseline(*ds.tables)
    base_scores = score_all(base, ds.truth, BASELINE_RULES)
    comparison = base_scores[["rule", "name", "flagged", "precision", "recall", "f1"]].merge(
        scores[["rule", "flagged", "precision", "recall", "f1"]], on="rule", suffixes=("_naive", "_apguard"))

    benign = benign_false_positives(flags, ds.benign, ds.pos, ds.invoices, ds.truth)
    benign_naive = benign_false_positives(base, ds.benign, ds.pos, ds.invoices, ds.truth)
    benign = benign.merge(benign_naive[["pattern", "flagged", "flagged_by"]], on="pattern", suffixes=("", "_naive"))

    sensitivity = []
    sensitivity += sweep(ds, "R1", "cross_po_window_days", [0, 7, 14, 21, 30, 45, 60],
                         lambda v: {"duplicate_invoice": {"cross_po_window_days": v}})
    sensitivity += sweep(ds, "R1", "cross_po_scope=any_po, window", [14, 30, 45],
                         lambda v: {"duplicate_invoice": {"cross_po_window_days": v, "cross_po_scope": "any_po"}})
    sensitivity += sweep(ds, "R7", "split window_days", [1, 3, 5, 10, 20, 30],
                         lambda v: {"invoice_splitting": {"window_days": v}})
    sensitivity += sweep(ds, "A2", "robust_z_threshold", [2.5, 3.0, 3.5, 4.5, 5.5],
                         lambda v: {"price_outlier": {"robust_z_threshold": v}})
    sensitivity += sweep(ds, "A2", "min_log_mad (z=3.5)", [0.0, 0.03, 0.05, 0.08],
                         lambda v: {"price_outlier": {"min_log_mad": v}})
    sensitivity += sweep(ds, "A1", "benford mad_threshold", [0.006, 0.010, 0.015, 0.025],
                         lambda v: {"benford": {"mad_threshold": v}})

    iforest = isolation_forest_benchmark(ds, controls)
    dollars = dollar_recall(flags, ds)
    overlap = overlap_matrix(flags, [c for c in RULES if RULES[c].level != "vendor"])

    gross = float(controls["amount_at_risk"].sum())
    net = float(controls.dropna(subset=["invoice_id"]).groupby("invoice_id")["amount_at_risk"].max().sum())

    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "config": cfg,
        "dataset": {n: int(len(t)) for n, t in zip(["vendors", "contracts", "purchase_orders", "goods_receipts",
                                                    "invoices", "payments"], ds.tables, strict=True)},
        "injected_exceptions": int(len(ds.truth)),
        "benign_lookalikes": int(len(ds.benign)),
        "runtime_seconds": round(runtime, 3),
        "flags": int(len(flags)),
        "gross_at_risk": round(gross, 2),
        "net_exposure": round(net, 2),
        "scores": scores.to_dict(orient="records"),
        "duplicate_variants": variants,
        "baseline_comparison": comparison.to_dict(orient="records"),
        "benign_false_positives": benign.to_dict(orient="records"),
        "sensitivity": sensitivity,
        "isolation_forest": iforest,
        "dollar_recall": dollars,
        "overlap": {"codes": list(overlap.index), "matrix": overlap.values.tolist()},
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2, default=float))

    # ---------------- markdown report ----------------
    s = scores.copy()
    s["detector"] = s["rule"] + " " + s["name"]
    md = [
        "# APGuard evaluation report",
        "",
        f"_Generated by `scripts/evaluate.py` on {result['generated_at']}. Synthetic data, seed 42._",
        "",
        f"**Dataset:** {sum(result['dataset'].values()):,} records, {result['injected_exceptions']:,} injected exceptions, "
        f"{result['benign_lookalikes']:,} legitimate look-alikes.  ",
        f"**Run:** {result['flags']:,} flags in {runtime:.2f}s. Gross flagged ${gross:,.0f}; "
        f"net exposure after de-duplicating overlapping flags ${net:,.0f}.  ",
        f"**Dollar recall:** {dollars['dollar_recall']:.1%} of ${dollars['injected_leakage']:,.0f} injected leakage "
        "sits on an entity a control flagged.",
        "",
        "## Precision and recall per detector",
        "",
        "Matched on unique entities. `fp_other_exception` counts false positives that carry a *different* "
        "injected exception (a real problem, caught by another rule).",
        "",
        md_table(s[["detector", "level", "injected", "flagged", "true_positives", "false_positives",
                    "fp_other_exception", "precision", "recall", "f1"]], pct_cols=("precision", "recall", "f1")),
        "",
        "## What the refinements buy: naive baseline vs APGuard",
        "",
        "The naive versions (`rules/baseline.py`) are how the first version of this project worked.",
        "",
        md_table(comparison, pct_cols=[c for c in comparison.columns if c.startswith(("precision", "recall", "f1"))]),
        "",
        "## Legitimate look-alikes wrongly flagged",
        "",
        md_table(benign.rename(columns={"flagged": "flagged_apguard", "flagged_by": "by_apguard"})),
        "",
        "## Duplicate invoices: exact matching vs fuzzy matching",
        "",
        md_table(pd.DataFrame(variants), pct_cols=("exact_recall", "apguard_recall")),
        "",
        "## Threshold sensitivity",
        "",
        md_table(pd.DataFrame(sensitivity), pct_cols=("precision", "recall", "f1")),
        "",
        "## Unsupervised baseline: Isolation Forest",
        "",
        f"Top {iforest['k']:,} invoices by anomaly score (k = number of injected invoice-level exceptions). "
        f"Precision {iforest['isolation_forest_precision']:.1%} vs {iforest['rules_precision']:.1%} for the control rules.",
        "",
        md_table(pd.DataFrame(iforest["per_type"]), pct_cols=("isolation_forest_recall", "rules_recall")),
        "",
        "## Rule overlap (invoices flagged by both)",
        "",
        md_table(overlap.reset_index().rename(columns={"index": "rule"})),
        "",
    ]
    (out / "EVALUATION.md").write_text("\n".join(md))
    print(s[["detector", "flagged", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(f"\nnet exposure ${net:,.0f}  gross ${gross:,.0f}  runtime {runtime:.2f}s")
    print(f"wrote {out / 'EVALUATION.md'} and {out / 'evaluation.json'}")


if __name__ == "__main__":
    main()
