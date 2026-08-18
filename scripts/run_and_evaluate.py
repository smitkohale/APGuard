import os
from dotenv import load_dotenv

load_dotenv()
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from rules.engine import run_all_rules

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

vendors = pd.read_sql("SELECT * FROM vendors", engine)
contracts = pd.read_sql("SELECT * FROM contracts", engine)
pos = pd.read_sql("SELECT * FROM purchase_orders", engine)
receipts = pd.read_sql("SELECT * FROM goods_receipts", engine)
invoices = pd.read_sql("SELECT * FROM invoices", engine)
payments = pd.read_sql("SELECT * FROM payments", engine)

flags = run_all_rules(vendors, contracts, pos, receipts, invoices, payments)
print(f"Total flags raised: {len(flags):,}")
print(flags["rule_code"].value_counts().sort_index())

# write flags to DB
flags_out = flags.copy()
flags_out = flags_out.where(pd.notnull(flags_out), None)
with engine.begin() as conn:
    conn.exec_driver_sql("TRUNCATE TABLE flags RESTART IDENTITY;")
flags_out.to_sql("flags", engine, if_exists="append", index=False, method="multi", chunksize=2000)
print(f"\nWrote {len(flags_out):,} flags to the flags table.")

# ---- recall against injected ground truth ----
gt = pd.read_csv(Path(__file__).parent.parent / "data" / "generated" / "injected_exceptions.csv")

rule_for_exception = {
    "duplicate_invoice": "R1",
    "three_way_mismatch": "R2",
    "contract_price_violation": "R3",
    "missing_po": "R4",
    "duplicate_payment": "R5",
    "tax_mismatch": "R6",
    "invoice_splitting": "R7",
    "overpayment": "R8",
    "payment_without_gr": "R9",
}

print("\n--- Recall against injected ground truth ---")
for exc_type, rule_code in rule_for_exception.items():
    gt_subset = gt[gt["exception_type"] == exc_type]
    rule_flags = flags[flags["rule_code"] == rule_code]

    if exc_type == "duplicate_payment" or exc_type == "overpayment":
        found = set(gt_subset["payment_id"].dropna().astype(int)) & set(rule_flags["payment_id"].dropna().astype(int))
        total = len(set(gt_subset["payment_id"].dropna().astype(int)))
    else:
        found = set(gt_subset["invoice_id"].dropna().astype(int)) & set(rule_flags["invoice_id"].dropna().astype(int))
        total = len(set(gt_subset["invoice_id"].dropna().astype(int)))

    recall = len(found) / total * 100 if total else 0
    print(f"{rule_code} {exc_type:28s} injected={total:4d}  caught={len(found):4d}  recall={recall:5.1f}%")
