"""
APGuard synthetic data generator.

Builds a linked AP dataset (vendors, contracts, purchase orders, goods receipts,
invoices, payments) and then does two things to it:

1. Injects known exceptions, each logged to `injected_exceptions.csv` with its
   type, variant and ids. This is the ground truth used to measure recall AND
   precision.

2. Adds legitimate activity that *looks* suspicious, logged to
   `benign_lookalikes.csv`. A detector that flags these is producing false
   positives. Without this step every naive rule scores 100% and the evaluation
   says nothing.

Realism choices
  * Vendor volume follows a power law: a few vendors carry most of the spend.
  * SKUs have a market price; contracts are negotiated below it, ad-hoc POs pay
    around it. Prices and quantities are log-uniform, so clean invoice amounts
    follow Benford's Law the way real AP data tends to.
  * Contracts are amended mid-term (price increases), partial deliveries are
    billed in instalments, recurring services bill the same amount monthly,
    and tax and payment amounts carry cent-level rounding noise.

Injected exception types (detector in brackets)
  duplicate_invoice [R1]        exact resubmission / reformatted number / amount drift
  three_way_mismatch [R2]       invoiced qty != ordered/received (incl. natural receipt variance)
  contract_price_violation [R3] unit price above the contract in force
  missing_po [R4]               invoice with no PO
  duplicate_payment [R5]        invoice paid twice
  tax_mismatch [R6]             wrong tax amount
  invoice_splitting [R7]        one order billed as several invoices under the approval limit
  overpayment [R8]              payment above invoice total
  payment_without_gr [R9]       paid with no goods receipt
  duplicate_vendor [R10]        cloned vendor record sharing a bank account
  new_vendor_large_payment [R11] shell vendor paid large sums days after creation
  threshold_hugging [A1]        vendor billing amounts clustered under the approval limit
  price_outlier [A2]            inflated unit price on a non-contract item (PO inflated too)
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
fake = Faker()
Faker.seed(SEED)
random.seed(SEED)
rng = np.random.default_rng(SEED)

N_BASE_VENDORS = 120
N_CONTRACTS = 200
N_POS = 12000
APPROVAL_THRESHOLD = 5000.00
PERIOD_START, PERIOD_END = date(2024, 1, 1), date(2026, 6, 30)
TAX_RATES = [0.05, 0.07, 0.0825, 0.10, 0.12]

OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

SKUS = [f"SKU-{i:04d}" for i in range(1, 151)]
MARKET_PRICE = {s: round(math.exp(random.uniform(math.log(20), math.log(2000))), 2) for s in SKUS}

ground_truth: list[dict] = []
lookalikes: list[dict] = []


def log_gt(exc_type, variant="", invoice_id=None, po_id=None, payment_id=None, vendor_id=None, source="injected"):
    ground_truth.append(dict(exception_type=exc_type, variant=variant, invoice_id=invoice_id, po_id=po_id,
                             payment_id=payment_id, vendor_id=vendor_id, source=source))


def log_benign(pattern, invoice_id=None, po_id=None, vendor_id=None, contract_id=None, note=""):
    lookalikes.append(dict(pattern=pattern, invoice_id=invoice_id, po_id=po_id, vendor_id=vendor_id,
                           contract_id=contract_id, note=note))


def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def log_uniform_qty(hi=250) -> int:
    return max(1, int(round(math.exp(random.uniform(0, math.log(hi))))))


# --------------------------------------------------------------------------- #
# 1. Vendors
# --------------------------------------------------------------------------- #
vendors: list[dict] = []


def add_vendor(name, created, risk="low", bank=None, tax_id=None):
    vid = len(vendors) + 1
    vendors.append(dict(
        vendor_id=vid, vendor_name=name,
        tax_id=tax_id or fake.unique.bothify(text="??-#######").upper(),
        bank_account=bank or fake.unique.iban(),
        country=fake.country(), risk_category=risk, created_at=created,
    ))
    return vid


for _ in range(N_BASE_VENDORS):
    add_vendor(fake.unique.company(), rand_date(date(2021, 1, 1), date(2023, 6, 30)),
               random.choices(["low", "medium", "high"], weights=[0.7, 0.22, 0.08])[0])
base_vendor_ids = [v["vendor_id"] for v in vendors]

# power-law volume: weight 1/rank^1.1, ranks shuffled across vendors
ranks = list(range(1, N_BASE_VENDORS + 1))
random.shuffle(ranks)
vendor_weight = {vid: 1 / r ** 1.1 for vid, r in zip(base_vendor_ids, ranks, strict=True)}

# --------------------------------------------------------------------------- #
# 2. Contracts (negotiated below market) with mid-term amendments
# --------------------------------------------------------------------------- #
contracts: list[dict] = []
vendor_contract_skus: dict[int, list[str]] = {}
for vid in base_vendor_ids:
    if len(contracts) >= N_CONTRACTS:
        break
    for sku in random.sample(SKUS, k=random.randint(3, 8)):
        if len(contracts) >= N_CONTRACTS:
            break
        contracts.append(dict(contract_id=len(contracts) + 1, vendor_id=vid, item_sku=sku,
                              contract_price=round(MARKET_PRICE[sku] * random.uniform(0.85, 0.98), 2),
                              tax_rate=random.choice(TAX_RATES),
                              start_date=date(2023, 1, 1), end_date=date(2026, 12, 31)))
        vendor_contract_skus.setdefault(vid, []).append(sku)

# Legitimate price increases: the old contract ends, a new one starts at +3-8%.
for c in random.sample(list(contracts), k=30):
    amend = rand_date(date(2024, 6, 1), date(2025, 12, 31))
    new = dict(c, contract_id=len(contracts) + 1, contract_price=round(c["contract_price"] * random.uniform(1.03, 1.08), 2),
               start_date=amend)
    c["end_date"] = amend - timedelta(days=1)
    contracts.append(new)
    log_benign("contract_amendment", vendor_id=c["vendor_id"], contract_id=new["contract_id"],
               note=f"price raised from {c['contract_price']} to {new['contract_price']}")
contracts_df = pd.DataFrame(contracts)


def contract_in_force(vid, sku, on: date):
    for c in contracts:
        if c["vendor_id"] == vid and c["item_sku"] == sku and c["start_date"] <= on <= c["end_date"]:
            return c
    return None


contract_index: dict[tuple, list[dict]] = {}
for c in contracts:
    contract_index.setdefault((c["vendor_id"], c["item_sku"]), []).append(c)


def fast_contract(vid, sku, on):
    for c in contract_index.get((vid, sku), []):
        if c["start_date"] <= on <= c["end_date"]:
            return c
    return None


# --------------------------------------------------------------------------- #
# 3. Purchase orders
# --------------------------------------------------------------------------- #
purchase_orders: list[dict] = []
po_kind: dict[int, str] = {}  # normal / recurring / partial / hugging / clone / shell / newvendor


def add_po(vid, sku, qty, unit_price, po_date, contract_id=None, kind="normal"):
    pid = len(purchase_orders) + 1
    purchase_orders.append(dict(po_id=pid, vendor_id=vid, contract_id=contract_id, item_sku=sku, quantity=qty,
                                unit_price=unit_price, po_date=po_date, po_amount=round(qty * unit_price, 2)))
    po_kind[pid] = kind
    return pid


def market_po(vid, po_date, kind="normal", qty=None, sku=None):
    """Ad-hoc purchase: SKU not under contract for this vendor, price near market."""
    contracted = set(vendor_contract_skus.get(vid, []))
    sku = sku or random.choice([s for s in SKUS if s not in contracted])
    price = round(MARKET_PRICE[sku] * random.uniform(0.95, 1.15), 2)
    return add_po(vid, sku, qty or log_uniform_qty(), price, po_date, None, kind)


weights = [vendor_weight[v] for v in base_vendor_ids]
for vid in random.choices(base_vendor_ids, weights=weights, k=N_POS):
    po_date = rand_date(PERIOD_START, PERIOD_END)
    skus = vendor_contract_skus.get(vid)
    if skus and random.random() < 0.75:
        sku = random.choice(skus)
        c = fast_contract(vid, sku, po_date)
        add_po(vid, sku, log_uniform_qty(), c["contract_price"], po_date, c["contract_id"])
    else:
        market_po(vid, po_date)

# Recurring services: same SKU, qty and price every month on a fresh PO.
for vid in random.sample(base_vendor_ids, k=25):
    sku = random.choice([s for s in SKUS if s not in vendor_contract_skus.get(vid, [])])
    qty, price = random.randint(1, 5), round(MARKET_PRICE[sku] * random.uniform(0.95, 1.1), 2)
    start = rand_date(date(2024, 1, 1), date(2025, 1, 1))
    for m in range(12):
        pid = add_po(vid, sku, qty, price, start + timedelta(days=30 * m), kind="recurring")
        log_benign("recurring_billing", po_id=pid, vendor_id=vid, note="same amount billed monthly on separate POs")


def hug_po(vid, po_date):
    """PO whose invoice total lands between 4,500 and 4,990: just under the limit."""
    while True:
        sku = random.choice([s for s in SKUS if s not in vendor_contract_skus.get(vid, [])])
        rate = random.choice(TAX_RATES)
        target_sub = random.uniform(4500, 4990) / (1 + rate)
        qty = max(1, round(target_sub / MARKET_PRICE[sku]))
        price = round(target_sub / qty, 2)
        if 0.95 <= price / MARKET_PRICE[sku] <= 1.15 and qty * price * (1 + rate) < 4995:
            return add_po(vid, sku, qty, price, po_date, kind="hugging"), rate


# Threshold-hugging vendors: mid-volume vendors who start billing just under the limit.
mid_volume = sorted(base_vendor_ids, key=lambda v: vendor_weight[v])[40:80]
hug_vendors = random.sample(mid_volume, k=3)
hug_tax: dict[int, float] = {}
for vid in hug_vendors:
    for _ in range(90):
        pid, rate = hug_po(vid, rand_date(date(2025, 1, 1), PERIOD_END))
        hug_tax[pid] = rate
    log_gt("threshold_hugging", vendor_id=vid)

# Cloned vendor records: same bank account as a real vendor, slightly different name.
clone_ids = []
for orig in random.sample(vendors[:N_BASE_VENDORS], k=8):
    variant = random.choice([orig["vendor_name"].replace(",", "").replace(" and ", " & "),
                             orig["vendor_name"].upper(),
                             orig["vendor_name"].split()[0] + " Holdings",
                             orig["vendor_name"] + " Ltd"])
    if variant == orig["vendor_name"]:
        variant += " Ltd"
    vid = add_vendor(variant, rand_date(date(2024, 3, 1), date(2025, 6, 30)), bank=orig["bank_account"])
    clone_ids.append(vid)
    for _ in range(random.randint(15, 30)):
        market_po(vid, rand_date(vendors[vid - 1]["created_at"] + timedelta(days=40), PERIOD_END), kind="clone")
    log_gt("duplicate_vendor", vendor_id=vid)

# Shell vendors: created, then paid large sums within days.
shell_ids = []
for _ in range(12):
    created = rand_date(date(2024, 6, 1), date(2026, 3, 31))
    vid = add_vendor(fake.unique.company(), created, risk="low")
    shell_ids.append(vid)
    for _ in range(random.randint(2, 3)):
        sku = random.choice([s for s in SKUS if MARKET_PRICE[s] > 300])
        qty = max(1, round(random.uniform(12000, 45000) / MARKET_PRICE[sku]))
        market_po(vid, created + timedelta(days=random.randint(2, 8)), kind="shell", qty=qty, sku=sku)

# Legitimately onboarded new vendors: first order comes weeks after onboarding.
for _ in range(10):
    created = rand_date(date(2024, 6, 1), date(2025, 12, 31))
    vid = add_vendor(fake.unique.company(), created)
    for _ in range(random.randint(10, 25)):
        market_po(vid, created + timedelta(days=random.randint(45, 200)), kind="newvendor")
    log_benign("new_vendor_onboarding", vendor_id=vid, note="new vendor, first payment > 30 days after creation")

vendors_df = pd.DataFrame(vendors)
po_df = pd.DataFrame(purchase_orders)
po_by_id = {p["po_id"]: p for p in purchase_orders}

# Partial deliveries: 2-3 receipts and matching invoices on one PO.
partial_pool = [p for p, k in po_kind.items() if k == "normal" and po_by_id[p]["quantity"] >= 4]
partial_pos = set(random.sample(partial_pool, k=400))
for p in partial_pos:
    po_kind[p] = "partial"

# POs with no goods receipt (reserved for payment_without_gr)
no_gr_pos = set(random.sample([p for p, k in po_kind.items() if k == "normal"], k=210))
for p in no_gr_pos:
    po_kind[p] = "no_gr"

# --------------------------------------------------------------------------- #
# 4. Goods receipts
# --------------------------------------------------------------------------- #
goods_receipts: list[dict] = []
partial_plan: dict[int, list[tuple[int, date]]] = {}


def add_gr(po_id, qty, d):
    goods_receipts.append(dict(gr_id=len(goods_receipts) + 1, po_id=po_id, received_qty=qty, received_date=d))


natural_variance: list[int] = []
for pid, po in po_by_id.items():
    kind = po_kind[pid]
    if kind == "no_gr":
        continue
    d = po["po_date"] + timedelta(days=random.randint(1, 3) if kind == "shell" else random.randint(2, 21))
    if kind == "partial":
        n = 3 if po["quantity"] >= 9 and random.random() < 0.3 else 2
        qtys = [po["quantity"] // n] * n
        qtys[-1] += po["quantity"] - sum(qtys)
        plan = []
        for q in qtys:
            add_gr(pid, q, d)
            plan.append((q, d))
            d += timedelta(days=random.randint(4, 20))
        partial_plan[pid] = plan
        log_benign("partial_delivery", po_id=pid, vendor_id=po["vendor_id"], note=f"{n} deliveries, {n} invoices")
        continue
    qty = po["quantity"]
    if kind == "normal" and random.random() < 0.012:
        qty = max(1, qty + random.choice([-1, 1]) * random.randint(1, 3))
        if qty != po["quantity"]:
            natural_variance.append(pid)
    add_gr(pid, qty, d)
gr_df = pd.DataFrame(goods_receipts)
gr_date = gr_df.groupby("po_id")["received_date"].max().to_dict()

# --------------------------------------------------------------------------- #
# 5. Invoices (clean baseline)
# --------------------------------------------------------------------------- #
invoices: list[dict] = []
tax_rate_by_contract = contracts_df.set_index("contract_id")["tax_rate"].to_dict()


def inv_number(vid, iid):
    return f"INV-{vid:04d}-{iid:06d}"


def add_invoice(po, qty, inv_date, rate, unit_price=None):
    iid = len(invoices) + 1
    price = po["unit_price"] if unit_price is None else unit_price
    sub = round(qty * price, 2)
    tax = round(sub * rate, 2)
    invoices.append(dict(invoice_id=iid, vendor_id=po["vendor_id"], po_id=po["po_id"],
                         invoice_number=inv_number(po["vendor_id"], iid), invoice_date=inv_date,
                         item_sku=po["item_sku"], quantity=qty, unit_price=price, subtotal=sub,
                         tax_amount=tax, total_amount=round(sub + tax, 2)))
    return iid


po_tax_rate: dict[int, float] = {}
invoice_ids_by_po: dict[int, list[int]] = {}
for pid, po in po_by_id.items():
    if po["contract_id"] is not None and not pd.isna(po["contract_id"]):
        rate = tax_rate_by_contract[po["contract_id"]]
    else:
        rate = hug_tax.get(pid, random.choice(TAX_RATES[:4]))
    po_tax_rate[pid] = rate
    kind = po_kind[pid]
    if kind == "partial":
        ids = [add_invoice(po, q, d + timedelta(days=random.randint(0, 5)), rate) for q, d in partial_plan[pid]]
    else:
        base = gr_date.get(pid, po["po_date"])
        offset = (1, 4) if kind == "shell" else (1, 30)
        ids = [add_invoice(po, po["quantity"], base + timedelta(days=random.randint(*offset)), rate)]
    invoice_ids_by_po[pid] = ids
for pid in natural_variance:
    log_gt("three_way_mismatch", "natural receipt variance", invoice_id=invoice_ids_by_po[pid][0], po_id=pid,
           vendor_id=po_by_id[pid]["vendor_id"], source="natural")

inv_df = pd.DataFrame(invoices).set_index("invoice_id", drop=False)

# cent-level tax rounding noise on 3% of invoices (benign)
for iid in random.sample(list(inv_df.index), k=int(0.03 * len(inv_df))):
    inv_df.loc[iid, "tax_amount"] = round(inv_df.loc[iid, "tax_amount"] + random.choice([-0.01, 0.01]), 2)
    inv_df.loc[iid, "total_amount"] = round(inv_df.loc[iid, "subtotal"] + inv_df.loc[iid, "tax_amount"], 2)
    log_benign("tax_rounding", invoice_id=int(iid), note="tax off by one cent")

# --------------------------------------------------------------------------- #
# 6. Inject exceptions. Each comes from its own pool of otherwise-clean,
#    single-invoice "normal" POs, so attribution stays one-exception-per-PO.
# --------------------------------------------------------------------------- #
available = {p for p, k in po_kind.items() if k == "normal" and p not in natural_variance}
contracted = {p for p in available if po_by_id[p]["contract_id"] is not None and not pd.isna(po_by_id[p]["contract_id"])}


def take(pool, k):
    chosen = random.sample(sorted(pool), k=min(k, len(pool)))
    available.difference_update(chosen)
    return chosen


def first_inv(pid):
    return invoice_ids_by_po[pid][0]


def reprice(iid, qty=None, price=None, tax=None):
    row = inv_df.loc[iid]
    qty = row["quantity"] if qty is None else qty
    price = row["unit_price"] if price is None else price
    sub = round(qty * price, 2)
    tax = round(sub * po_tax_rate[int(row["po_id"])], 2) if tax is None else tax
    inv_df.loc[iid, ["quantity", "unit_price", "subtotal", "tax_amount", "total_amount"]] = \
        [qty, price, sub, tax, round(sub + tax, 2)]


for pid in take(contracted & available, 260):
    iid = first_inv(pid)
    reprice(iid, price=round(po_by_id[pid]["unit_price"] * random.uniform(1.05, 1.35), 2))
    log_gt("contract_price_violation", invoice_id=iid, po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

for pid in take(available, 290):
    iid, true_qty = first_inv(pid), po_by_id[pid]["quantity"]
    bad = true_qty + random.choice([-1, 1]) * random.randint(5, 30)
    if bad < 1:
        bad = true_qty + random.randint(5, 30)
    reprice(iid, qty=bad)
    log_gt("three_way_mismatch", "over-billed" if bad > true_qty else "under-billed",
           invoice_id=iid, po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

for pid in take(contracted & available, 220):
    iid = first_inv(pid)
    sub = inv_df.loc[iid, "subtotal"]
    true_rate = po_tax_rate[pid]
    wrong = random.choice([r for r in [0.0, 0.03, 0.18, 0.20] if r != true_rate])
    reprice(iid, tax=round(sub * wrong, 2))
    log_gt("tax_mismatch", invoice_id=iid, po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

for pid in take(available, 200):
    iid = first_inv(pid)
    inv_df.loc[iid, "po_id"] = np.nan
    log_gt("missing_po", invoice_id=iid, po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

market_pool = {p for p in available if p not in contracted}
for pid in take(market_pool, 120):
    sku = po_by_id[pid]["item_sku"]
    inflated = round(MARKET_PRICE[sku] * random.uniform(1.5, 2.3), 2)
    po_by_id[pid]["unit_price"] = inflated
    po_by_id[pid]["po_amount"] = round(inflated * po_by_id[pid]["quantity"], 2)
    iid = first_inv(pid)
    reprice(iid, price=inflated)
    log_gt("price_outlier", invoice_id=iid, po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

next_iid = int(inv_df.index.max()) + 1
new_rows = []
for pid in take(available, 230):
    orig = inv_df.loc[first_inv(pid)].copy()
    dup = orig.copy()
    dup["invoice_id"] = next_iid
    dup["invoice_date"] = orig["invoice_date"] + timedelta(days=random.randint(3, 40))
    variant = random.choices(["exact resubmission", "reformatted number", "amount drift", "re-keyed without PO"],
                             weights=[3, 2.5, 2.5, 2])[0]
    if variant == "re-keyed without PO":
        # vendor resends; AP keys it as a non-PO invoice with the vendor's new number
        dup["invoice_number"] = inv_number(int(orig["vendor_id"]), next_iid)
        dup["po_id"] = np.nan
    elif variant == "exact resubmission":
        dup["invoice_number"] = inv_number(int(orig["vendor_id"]), next_iid)
    elif variant == "reformatted number":
        n = orig["invoice_number"]
        dup["invoice_number"] = random.choice([n.replace("-", ""), n.lower(), n.replace("-", "/"), n + " ",
                                               " " + n.replace("-", " ")])
    else:
        dup["invoice_number"] = inv_number(int(orig["vendor_id"]), next_iid)
        drift = round(random.choice([-1, 1]) * random.uniform(0.01, min(0.9, 0.004 * orig["total_amount"]) + 0.01), 2)
        dup["total_amount"] = round(orig["total_amount"] + drift, 2)  # mis-keyed total
    new_rows.append(dup)
    invoice_ids_by_po[pid].append(next_iid)
    log_gt("duplicate_invoice", variant, invoice_id=next_iid, po_id=pid, vendor_id=int(orig["vendor_id"]))
    if variant == "re-keyed without PO":  # it genuinely has no PO as well
        log_gt("missing_po", "re-keyed duplicate", invoice_id=next_iid, po_id=pid, vendor_id=int(orig["vendor_id"]))
    next_iid += 1

# Invoice splitting: choose the number of pieces so that each piece, *including
# tax*, stays under the limit. (The first version split the pre-tax amount, so
# some pieces ended up over the limit after tax and were never really "split".)
split_pool = [p for p in available
              if APPROVAL_THRESHOLD * 1.15 < po_by_id[p]["po_amount"] * (1 + po_tax_rate[p]) < APPROVAL_THRESHOLD * 3.5]
for pid in take(set(split_pool), 130):
    po = po_by_id[pid]
    total = po["po_amount"] * (1 + po_tax_rate[pid])
    n = math.ceil(total / (APPROVAL_THRESHOLD * 0.97))
    if po["quantity"] < n:
        continue
    qtys = [po["quantity"] // n] * n
    qtys[-1] += po["quantity"] - sum(qtys)
    if max(qtys) * po["unit_price"] * (1 + po_tax_rate[pid]) >= APPROVAL_THRESHOLD:
        continue
    ids = []
    first = first_inv(pid)
    reprice(first, qty=qtys[0])
    ids.append(first)
    for i, q in enumerate(qtys[1:], start=1):
        row = inv_df.loc[first].copy()
        row["invoice_id"], row["invoice_number"] = next_iid, inv_number(int(po["vendor_id"]), next_iid)
        row["invoice_date"] = row["invoice_date"] + timedelta(days=random.randint(0, 3) + i - 1)
        sub = round(q * po["unit_price"], 2)
        tax = round(sub * po_tax_rate[pid], 2)
        row["quantity"], row["subtotal"], row["tax_amount"], row["total_amount"] = q, sub, tax, round(sub + tax, 2)
        new_rows.append(row)
        ids.append(next_iid)
        next_iid += 1
    invoice_ids_by_po[pid] = ids
    for iid in ids:
        log_gt("invoice_splitting", f"{n}-way", invoice_id=iid, po_id=pid, vendor_id=po["vendor_id"])

inv_df = pd.concat([inv_df, pd.DataFrame(new_rows)], ignore_index=True).sort_values("invoice_id").reset_index(drop=True)
inv_df["invoice_id"] = inv_df["invoice_id"].astype(int)

for pid in no_gr_pos:
    log_gt("payment_without_gr", invoice_id=first_inv(pid), po_id=pid, vendor_id=po_by_id[pid]["vendor_id"])

po_df = pd.DataFrame(list(po_by_id.values()))

# --------------------------------------------------------------------------- #
# 7. Payments
# --------------------------------------------------------------------------- #
created_at = vendors_df.set_index("vendor_id")["created_at"].to_dict()
shell_set = set(shell_ids)
pay_rows = []
for inv in inv_df.itertuples(index=False):
    lag = random.randint(2, 8) if inv.vendor_id in shell_set else random.randint(5, 45)
    amount = float(inv.total_amount)
    pay_rows.append(dict(payment_id=len(pay_rows) + 1, invoice_id=int(inv.invoice_id),
                         payment_date=inv.invoice_date + timedelta(days=lag),
                         payment_amount=amount, payment_ref=fake.unique.bothify(text="PMT-########")))
pay_df = pd.DataFrame(pay_rows)

# cent-level payment rounding (benign)
for idx in random.sample(list(pay_df.index), k=int(0.02 * len(pay_df))):
    pay_df.loc[idx, "payment_amount"] = round(pay_df.loc[idx, "payment_amount"] + 0.01, 2)
    log_benign("payment_rounding", invoice_id=int(pay_df.loc[idx, "invoice_id"]), note="paid one cent over")

inv_vendor = inv_df.set_index("invoice_id")["vendor_id"].to_dict()
for p in pay_df.itertuples(index=False):
    if inv_vendor[p.invoice_id] in shell_set:
        log_gt("new_vendor_large_payment", invoice_id=p.invoice_id, payment_id=p.payment_id,
               vendor_id=inv_vendor[p.invoice_id])

payable = [i for i in inv_df["invoice_id"] if inv_vendor[i] not in shell_set]
dup_pay = random.sample(payable, k=190)
extra = []
next_pid = int(pay_df["payment_id"].max()) + 1
first_pay = pay_df.drop_duplicates("invoice_id").set_index("invoice_id")
for iid in dup_pay:
    orig = first_pay.loc[iid]
    extra.append(dict(payment_id=next_pid, invoice_id=iid,
                      payment_date=orig["payment_date"] + timedelta(days=random.randint(2, 20)),
                      payment_amount=orig["payment_amount"], payment_ref=fake.unique.bothify(text="PMT-########")))
    log_gt("duplicate_payment", invoice_id=iid, payment_id=next_pid, vendor_id=inv_vendor[iid])
    next_pid += 1
pay_df = pd.concat([pay_df, pd.DataFrame(extra)], ignore_index=True)

dup_set = set(dup_pay)
for iid in random.sample([i for i in payable if i not in dup_set], k=160):
    idx = pay_df.index[pay_df["invoice_id"] == iid][0]
    pay_df.loc[idx, "payment_amount"] = round(pay_df.loc[idx, "payment_amount"] * random.uniform(1.05, 1.5), 2)
    log_gt("overpayment", invoice_id=iid, payment_id=int(pay_df.loc[idx, "payment_id"]), vendor_id=inv_vendor[iid])

pay_df = pay_df.sort_values("payment_id").reset_index(drop=True)

# --------------------------------------------------------------------------- #
# 8. Write
# --------------------------------------------------------------------------- #
po_df["contract_id"] = po_df["contract_id"].astype("Int64")
inv_df["po_id"] = inv_df["po_id"].astype("Int64")
tables = {"vendors": vendors_df, "contracts": contracts_df, "purchase_orders": po_df,
          "goods_receipts": gr_df, "invoices": inv_df, "payments": pay_df}
for name, df in tables.items():
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
gt_df = pd.DataFrame(ground_truth)
for c in ["invoice_id", "po_id", "payment_id", "vendor_id"]:
    gt_df[c] = gt_df[c].astype("Int64")
gt_df.to_csv(OUT_DIR / "injected_exceptions.csv", index=False)
pd.DataFrame(lookalikes).to_csv(OUT_DIR / "benign_lookalikes.csv", index=False)

if __name__ == "__main__":
    for name, df in tables.items():
        print(f"{name:16s} {len(df):>8,}")
    print(f"{'TOTAL':16s} {sum(len(d) for d in tables.values()):>8,}")
    print(f"\ninjected exceptions: {len(gt_df):,}")
    print(gt_df["exception_type"].value_counts().to_string())
    print(f"\nbenign look-alikes: {len(lookalikes):,}")
    print(pd.DataFrame(lookalikes)["pattern"].value_counts().to_string())
