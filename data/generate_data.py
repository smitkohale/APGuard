"""
APGuard synthetic data generator.

Generates vendors, contracts, purchase_orders, goods_receipts, invoices, payments
with realistic clean data, then injects a known set of exceptions:

  1. duplicate_invoice      - same invoice re-billed (same PO/vendor/amount, new invoice_id)
  2. three_way_mismatch     - invoice quantity/amount doesn't match PO + goods receipt
  3. contract_price_violation - invoice unit_price != contracted price for that SKU/vendor
  4. missing_po              - invoice has no linked PO at all
  5. duplicate_payment       - same invoice paid twice
  6. tax_mismatch            - invoice tax_amount doesn't match contract tax_rate * subtotal
  7. invoice_splitting       - one PO amount split into multiple invoices just under an approval threshold
  8. overpayment             - payment_amount > invoice total_amount

Every injected exception is tagged in `injected_exceptions.csv` with its type and the
relevant ids, so rule recall can be measured against ground truth.
"""
import random
import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

N_VENDORS = 120
N_CONTRACTS = 200
N_POS = 13000
INVOICE_APPROVAL_THRESHOLD = 5000.00  # used for invoice-splitting injection

OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

SKUS = [f"SKU-{i:04d}" for i in range(1, 151)]

ground_truth_rows = []  # (exception_type, invoice_id, po_id, payment_id, vendor_id)


def log_gt(exc_type, invoice_id=None, po_id=None, payment_id=None, vendor_id=None):
    ground_truth_rows.append({
        "exception_type": exc_type,
        "invoice_id": invoice_id,
        "po_id": po_id,
        "payment_id": payment_id,
        "vendor_id": vendor_id,
    })


# ---------------------------------------------------------------------------
# 1. Vendors
# ---------------------------------------------------------------------------
vendors = []
for vid in range(1, N_VENDORS + 1):
    vendors.append({
        "vendor_id": vid,
        "vendor_name": fake.company(),
        "tax_id": fake.bothify(text="??-#######").upper(),
        "country": fake.country(),
        "risk_category": random.choices(["low", "medium", "high"], weights=[0.7, 0.22, 0.08])[0],
    })
vendors_df = pd.DataFrame(vendors)

# ---------------------------------------------------------------------------
# 2. Contracts (one price+tax_rate per vendor/sku pair, for a subset of SKUs)
# ---------------------------------------------------------------------------
contracts = []
cid = 1
contract_lookup = {}  # (vendor_id, sku) -> contract row
for vid in range(1, N_VENDORS + 1):
    vendor_skus = random.sample(SKUS, k=random.randint(3, 8))
    for sku in vendor_skus:
        if cid > N_CONTRACTS:
            break
        price = round(random.uniform(20, 2000), 2)
        tax_rate = random.choice([0.05, 0.07, 0.0825, 0.10, 0.12])
        start = date(2023, 1, 1)
        end = date(2026, 12, 31)
        row = {
            "contract_id": cid,
            "vendor_id": vid,
            "item_sku": sku,
            "contract_price": price,
            "tax_rate": tax_rate,
            "start_date": start,
            "end_date": end,
        }
        contracts.append(row)
        contract_lookup[(vid, sku)] = row
        cid += 1
    if cid > N_CONTRACTS:
        break
contracts_df = pd.DataFrame(contracts)

# ---------------------------------------------------------------------------
# 3. Purchase Orders
# ---------------------------------------------------------------------------
purchase_orders = []
for po_id in range(1, N_POS + 1):
    vid = random.randint(1, N_VENDORS)
    # 75% of POs use a contracted SKU for that vendor if one exists
    vendor_contract_skus = [s for (v, s) in contract_lookup if v == vid]
    if vendor_contract_skus and random.random() < 0.75:
        sku = random.choice(vendor_contract_skus)
        unit_price = contract_lookup[(vid, sku)]["contract_price"]
        contract_id = contract_lookup[(vid, sku)]["contract_id"]
    else:
        sku = random.choice(SKUS)
        unit_price = round(random.uniform(20, 2000), 2)
        contract_id = None

    qty = random.randint(1, 200)
    po_date = fake.date_between(start_date=date(2024, 1, 1), end_date=date(2026, 6, 30))
    purchase_orders.append({
        "po_id": po_id,
        "vendor_id": vid,
        "contract_id": contract_id,
        "item_sku": sku,
        "quantity": qty,
        "unit_price": unit_price,
        "po_date": po_date,
        "po_amount": round(qty * unit_price, 2),
    })
po_df = pd.DataFrame(purchase_orders)
po_by_id = po_df.set_index("po_id").to_dict("index")

# ---------------------------------------------------------------------------
# 4. Goods Receipts (one per PO, usually matching qty; some short/over receipts;
#    a small set of POs deliberately get NO goods receipt at all — held back
#    below so we can inject "paid without goods receipt" exceptions in step 9)
# ---------------------------------------------------------------------------
NO_GR_POOL_SIZE = 210  # candidates for the payment_without_gr injection later
no_gr_po_ids = set(random.sample(list(po_by_id.keys()), k=NO_GR_POOL_SIZE))

goods_receipts = []
gr_id = 1
for po_id, po in po_by_id.items():
    if po_id in no_gr_po_ids:
        continue  # no GR row at all for these — "not yet received" / never confirmed
    received_qty = po["quantity"]
    if random.random() < 0.04:  # small natural variance, not an injected exception
        received_qty = max(1, po["quantity"] + random.randint(-3, 3))
    received_date = po["po_date"] + timedelta(days=random.randint(2, 21))
    goods_receipts.append({
        "gr_id": gr_id,
        "po_id": po_id,
        "received_qty": received_qty,
        "received_date": received_date,
    })
    gr_id += 1
gr_df = pd.DataFrame(goods_receipts)

# ---------------------------------------------------------------------------
# 5. Invoices (clean baseline, one invoice per PO)
# ---------------------------------------------------------------------------
invoices = []
invoice_id = 1
for po_id, po in po_by_id.items():
    inv_date = po["po_date"] + timedelta(days=random.randint(1, 30))
    qty = po["quantity"]
    unit_price = po["unit_price"]
    subtotal = round(qty * unit_price, 2)
    contract_id = po["contract_id"]
    tax_rate = contracts_df.set_index("contract_id").loc[contract_id, "tax_rate"] if pd.notna(contract_id) else random.choice([0.05, 0.07, 0.0825, 0.10])
    tax_amount = round(subtotal * tax_rate, 2)
    invoices.append({
        "invoice_id": invoice_id,
        "vendor_id": po["vendor_id"],
        "po_id": po_id,
        "invoice_number": f"INV-{po['vendor_id']:04d}-{invoice_id:06d}",
        "invoice_date": inv_date,
        "item_sku": po["item_sku"],
        "quantity": qty,
        "unit_price": unit_price,
        "subtotal": subtotal,
        "tax_amount": tax_amount,
        "total_amount": round(subtotal + tax_amount, 2),
    })
    invoice_id += 1

# ---------------------------------------------------------------------------
# 6. Inject exceptions into invoices (mutating a sample of the clean rows,
#    plus appending new duplicate/split invoice rows)
# ---------------------------------------------------------------------------
inv_df = pd.DataFrame(invoices)
next_invoice_id = inv_df["invoice_id"].max() + 1

# no_gr_po_ids are reserved for the payment_without_gr injection (step 3g below) —
# excluded from other pools up front so ground-truth attribution stays clean
# (one injected exception per PO, not stacked issues muddying recall numbers)
used_pos = set(no_gr_po_ids)

# --- 3a. Contract price violations: bump unit_price above the contracted price ---
contract_po_ids = [p for p in po_df[po_df["contract_id"].notna()]["po_id"].tolist() if p not in used_pos]
violation_pos = random.sample(contract_po_ids, k=min(260, len(contract_po_ids)))
used_pos |= set(violation_pos)
for po_id in violation_pos:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    true_price = po_by_id[po_id]["unit_price"]
    inflated = round(true_price * random.uniform(1.05, 1.35), 2)
    inv_df.loc[idx, "unit_price"] = inflated
    new_subtotal = round(inv_df.loc[idx, "quantity"] * inflated, 2)
    inv_df.loc[idx, "subtotal"] = new_subtotal
    inv_df.loc[idx, "total_amount"] = round(new_subtotal + inv_df.loc[idx, "tax_amount"], 2)
    log_gt("contract_price_violation", invoice_id=int(inv_df.loc[idx, "invoice_id"]),
           po_id=po_id, vendor_id=int(inv_df.loc[idx, "vendor_id"]))

# --- 3b. Three-way mismatches: invoice qty differs from PO+GR qty ---
mismatch_pos = random.sample([p for p in po_by_id if p not in used_pos], k=290)
used_pos |= set(mismatch_pos)
for po_id in mismatch_pos:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    true_qty = po_by_id[po_id]["quantity"]
    bad_qty = true_qty + random.choice([-1, 1]) * random.randint(5, 30)
    bad_qty = max(1, bad_qty)
    unit_price = inv_df.loc[idx, "unit_price"]
    new_subtotal = round(bad_qty * unit_price, 2)
    inv_df.loc[idx, "quantity"] = bad_qty
    inv_df.loc[idx, "subtotal"] = new_subtotal
    inv_df.loc[idx, "total_amount"] = round(new_subtotal + inv_df.loc[idx, "tax_amount"], 2)
    log_gt("three_way_mismatch", invoice_id=int(inv_df.loc[idx, "invoice_id"]),
           po_id=po_id, vendor_id=int(inv_df.loc[idx, "vendor_id"]))

# --- 3c. Tax mismatches: tax_amount doesn't reflect the contracted tax_rate ---
tax_candidates = [p for p in contract_po_ids if p not in used_pos]
tax_mismatch_pos = random.sample(tax_candidates, k=min(220, len(tax_candidates)))
for po_id in tax_mismatch_pos:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    subtotal = inv_df.loc[idx, "subtotal"]
    wrong_tax = round(subtotal * random.choice([0.0, 0.03, 0.18, 0.20]), 2)
    inv_df.loc[idx, "tax_amount"] = wrong_tax
    inv_df.loc[idx, "total_amount"] = round(subtotal + wrong_tax, 2)
    log_gt("tax_mismatch", invoice_id=int(inv_df.loc[idx, "invoice_id"]),
           po_id=po_id, vendor_id=int(inv_df.loc[idx, "vendor_id"]))
used_pos |= set(tax_mismatch_pos)

# --- 3d. Missing PO: some invoices have po_id set to None entirely ---
missing_po_candidates = [p for p in po_by_id if p not in used_pos]
missing_po_pos = random.sample(missing_po_candidates, k=200)
for po_id in missing_po_pos:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    inv_df.loc[idx, "po_id"] = None
    log_gt("missing_po", invoice_id=int(inv_df.loc[idx, "invoice_id"]),
           po_id=po_id, vendor_id=int(inv_df.loc[idx, "vendor_id"]))
used_pos |= set(missing_po_pos)

# --- 3e. Duplicate invoices: append a near-identical second invoice for a PO ---
dup_candidates = [p for p in po_by_id if p not in used_pos]
dup_pos = random.sample(dup_candidates, k=230)
dup_rows = []
for po_id in dup_pos:
    orig = inv_df[inv_df["po_id"] == po_id]
    if orig.empty:
        continue
    orig = orig.iloc[0]
    dup = orig.copy()
    dup["invoice_id"] = next_invoice_id
    dup["invoice_number"] = f"INV-{int(orig['vendor_id']):04d}-{next_invoice_id:06d}"
    dup["invoice_date"] = orig["invoice_date"] + timedelta(days=random.randint(3, 15))
    dup_rows.append(dup)
    log_gt("duplicate_invoice", invoice_id=next_invoice_id, po_id=po_id, vendor_id=int(orig["vendor_id"]))
    next_invoice_id += 1
if dup_rows:
    inv_df = pd.concat([inv_df, pd.DataFrame(dup_rows)], ignore_index=True)
used_pos |= set(dup_pos)

# --- 3f. Invoice splitting: split one PO's amount into 2-3 invoices, each just
#          under the approval threshold, where a single invoice would have exceeded it ---
# Bounded above: if the PO amount is too large, even a 3-way split still leaves
# each invoice over the threshold, which isn't actually a "dodge approval" pattern.
split_candidates = [
    p for p in po_by_id
    if p not in used_pos
    and INVOICE_APPROVAL_THRESHOLD * 1.15 < po_by_id[p]["po_amount"] < INVOICE_APPROVAL_THRESHOLD * 2.6
]
split_pos = random.sample(split_candidates, k=min(130, len(split_candidates)))
split_new_rows = []
for po_id in split_pos:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    orig = inv_df.loc[idx]
    n_splits = 3 if po_by_id[po_id]["po_amount"] > INVOICE_APPROVAL_THRESHOLD * 1.9 else 2
    total_qty = orig["quantity"]
    unit_price = orig["unit_price"]
    tax_rate_est = round(float(orig["tax_amount"]) / float(orig["subtotal"]), 4) if orig["subtotal"] else 0
    qtys = [total_qty // n_splits] * n_splits
    qtys[-1] += total_qty - sum(qtys)
    split_invoice_ids = []
    for i, q in enumerate(qtys):
        subtotal = round(q * unit_price, 2)
        tax_amount = round(subtotal * tax_rate_est, 2)
        if i == 0:
            inv_df.loc[idx, "quantity"] = q
            inv_df.loc[idx, "subtotal"] = subtotal
            inv_df.loc[idx, "tax_amount"] = tax_amount
            inv_df.loc[idx, "total_amount"] = round(subtotal + tax_amount, 2)
            split_invoice_ids.append(int(orig["invoice_id"]))
        else:
            new_inv = orig.copy()
            new_inv["invoice_id"] = next_invoice_id
            new_inv["invoice_number"] = f"INV-{int(orig['vendor_id']):04d}-{next_invoice_id:06d}"
            new_inv["invoice_date"] = orig["invoice_date"] + timedelta(days=i)
            new_inv["quantity"] = q
            new_inv["subtotal"] = subtotal
            new_inv["tax_amount"] = tax_amount
            new_inv["total_amount"] = round(subtotal + tax_amount, 2)
            split_new_rows.append(new_inv)
            split_invoice_ids.append(next_invoice_id)
            next_invoice_id += 1
    for iid in split_invoice_ids:
        log_gt("invoice_splitting", invoice_id=iid, po_id=po_id, vendor_id=int(orig["vendor_id"]))
if split_new_rows:
    inv_df = pd.concat([inv_df, pd.DataFrame(split_new_rows)], ignore_index=True)

inv_df = inv_df.sort_values("invoice_id").reset_index(drop=True)

# --- 3g. Payment without goods receipt: the no_gr_po_ids reserved in step 4 —
#          these invoices are otherwise completely normal (correct qty/price/tax),
#          the only issue is that AP is about to pay against a PO nothing was
#          ever received against. Logged here at invoice level; the actual
#          "exception" only becomes real once a payment goes out (step 7 below).
for po_id in no_gr_po_ids:
    mask = inv_df["po_id"] == po_id
    idx = inv_df[mask].index
    if len(idx) == 0:
        continue
    idx = idx[0]
    log_gt("payment_without_gr", invoice_id=int(inv_df.loc[idx, "invoice_id"]),
           po_id=po_id, vendor_id=int(inv_df.loc[idx, "vendor_id"]))

# ---------------------------------------------------------------------------
# 7. Payments (one per invoice normally; inject duplicate payments & overpayments)
# ---------------------------------------------------------------------------
payments = []
payment_id = 1
for _, inv in inv_df.iterrows():
    pay_date = inv["invoice_date"] + timedelta(days=random.randint(5, 45))
    payments.append({
        "payment_id": payment_id,
        "invoice_id": int(inv["invoice_id"]),
        "payment_date": pay_date,
        "payment_amount": float(inv["total_amount"]),
        "payment_ref": fake.bothify(text="PMT-########"),
    })
    payment_id += 1

pay_df = pd.DataFrame(payments)

# --- duplicate payments: same invoice paid a second time ---
dup_pay_invoice_ids = random.sample(inv_df["invoice_id"].tolist(), k=190)
dup_pay_rows = []
for iid in dup_pay_invoice_ids:
    orig = pay_df[pay_df["invoice_id"] == iid].iloc[0]
    new_pay = orig.copy()
    new_pay["payment_id"] = payment_id
    new_pay["payment_date"] = orig["payment_date"] + timedelta(days=random.randint(2, 20))
    new_pay["payment_ref"] = fake.bothify(text="PMT-########")
    dup_pay_rows.append(new_pay)
    log_gt("duplicate_payment", invoice_id=int(iid), payment_id=payment_id,
           vendor_id=int(inv_df.set_index("invoice_id").loc[iid, "vendor_id"]))
    payment_id += 1
if dup_pay_rows:
    pay_df = pd.concat([pay_df, pd.DataFrame(dup_pay_rows)], ignore_index=True)

# --- overpayments: payment_amount inflated above invoice total ---
overpay_candidates = [i for i in inv_df["invoice_id"].tolist() if i not in dup_pay_invoice_ids]
overpay_invoice_ids = random.sample(overpay_candidates, k=160)
for iid in overpay_invoice_ids:
    mask = (pay_df["invoice_id"] == iid)
    idx = pay_df[mask].index[0]
    true_amt = pay_df.loc[idx, "payment_amount"]
    pay_df.loc[idx, "payment_amount"] = round(true_amt * random.uniform(1.05, 1.5), 2)
    log_gt("overpayment", invoice_id=int(iid), payment_id=int(pay_df.loc[idx, "payment_id"]),
           vendor_id=int(inv_df.set_index("invoice_id").loc[iid, "vendor_id"]))

pay_df = pay_df.sort_values("payment_id").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 8. Write everything out
# ---------------------------------------------------------------------------
vendors_df.to_csv(OUT_DIR / "vendors.csv", index=False)
contracts_df.to_csv(OUT_DIR / "contracts.csv", index=False)
po_df.to_csv(OUT_DIR / "purchase_orders.csv", index=False)
gr_df.to_csv(OUT_DIR / "goods_receipts.csv", index=False)
inv_df.to_csv(OUT_DIR / "invoices.csv", index=False)
pay_df.to_csv(OUT_DIR / "payments.csv", index=False)

gt_df = pd.DataFrame(ground_truth_rows)
gt_df.to_csv(OUT_DIR / "injected_exceptions.csv", index=False)

print(f"vendors:          {len(vendors_df):,}")
print(f"contracts:        {len(contracts_df):,}")
print(f"purchase_orders:  {len(po_df):,}")
print(f"goods_receipts:   {len(gr_df):,}")
print(f"invoices:         {len(inv_df):,}")
print(f"payments:         {len(pay_df):,}")
print(f"TOTAL records:    {len(vendors_df)+len(contracts_df)+len(po_df)+len(gr_df)+len(inv_df)+len(pay_df):,}")
print(f"injected exceptions logged: {len(gt_df):,}")
print(gt_df['exception_type'].value_counts())
