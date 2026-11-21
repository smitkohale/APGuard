"""Simulate an analyst team working the case queue, for the demo only.

Takes a sample of active flags and resolves them the way a reviewer with perfect
information would: flags on injected exceptions are confirmed (and some recovered),
everything else is dismissed as a false positive with a note. This populates the
workflow views (observed precision, recoveries, audit trail). The dashboard labels
these decisions as simulated.

    python scripts/simulate_reviews.py [--fraction 0.15]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard.cases import update_flag  # noqa: E402
from apguard.db import read_sql  # noqa: E402
from apguard.evaluation import TARGETS  # noqa: E402

ANALYSTS = ["a.chen", "m.okafor", "s.patel", "j.romero"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    truth = pd.read_csv(ROOT / "data" / "generated" / "injected_exceptions.csv")
    flags = read_sql("SELECT flag_id, rule_code, vendor_id, invoice_id, payment_id, amount_at_risk "
                     "FROM v_active_flags WHERE status = 'open' ORDER BY flag_id")
    sample = flags.sample(frac=args.fraction, random_state=args.seed)

    real = {}
    for code, (exc, key) in TARGETS.items():
        real[code] = (key, set(truth.loc[truth["exception_type"] == exc, key].dropna().astype(int)))
    any_inv = set(truth["invoice_id"].dropna().astype(int))

    counts = {"confirmed": 0, "recovered": 0, "false_positive": 0, "in_review": 0}
    for f in sample.itertuples(index=False):
        key, ids = real[f.rule_code]
        val = getattr(f, key)
        is_real = (pd.notna(val) and int(val) in ids) or (pd.notna(f.invoice_id) and int(f.invoice_id) in any_inv)
        who = rnd.choice(ANALYSTS)
        roll = rnd.random()
        if roll < 0.15:
            update_flag(int(f.flag_id), status="in_review", assignee=who, actor=who, note="Requested vendor statement.")
            counts["in_review"] += 1
        elif is_real:
            update_flag(int(f.flag_id), status="confirmed", assignee=who, actor=who,
                        note="Verified against source documents. (simulated)")
            counts["confirmed"] += 1
            if f.amount_at_risk > 0 and rnd.random() < 0.55:
                amt = round(float(f.amount_at_risk) * rnd.uniform(0.6, 1.0), 2)
                update_flag(int(f.flag_id), status="recovered", recovered_amount=amt, actor=who,
                            note="Credit note received. (simulated)")
                counts["recovered"] += 1
                counts["confirmed"] -= 1
        else:
            update_flag(int(f.flag_id), status="false_positive", assignee=who, actor=who,
                        note="Legitimate transaction on review. (simulated)")
            counts["false_positive"] += 1
    print(f"simulated review of {len(sample):,} flags: {counts}")


if __name__ == "__main__":
    main()
