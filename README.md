# APGuard — Accounts Payable Leakage Detection

A rule-based analytics system that identifies payment leakage and contract
compliance issues across Accounts Payable transactions: duplicate payments,
three-way match failures, contract price violations, invoice splitting,
tax mismatches, overpayments, and payments issued without a goods receipt
on file.

## Stack
Python · PostgreSQL · FastAPI · SQLAlchemy · Pandas · Streamlit · Plotly · pytest

## What's actually in this repo

- **Synthetic dataset**: 53,158 records across 6 linked entities (vendors,
  contracts, purchase orders, goods receipts, invoices, payments), generated
  with `data/generate_data.py`. 2,089 realistic exceptions are deliberately
  injected and logged to `data/generated/injected_exceptions.csv` as ground
  truth — duplicate invoices, three-way match failures, contract price
  violations, missing POs, duplicate payments, tax mismatches, invoice
  splitting, overpayments, and payments without a goods receipt.
- **9 business validation rules** (`rules/engine.py`), each a pure function
  that takes DataFrames and returns flags — no DB access inside the rule
  logic, so they're independently testable.
- **26 pytest cases** (`tests/test_rules.py`) against hand-built fixtures,
  not the generated data — covering edge cases like triple payments, the
  threshold boundary for invoice splitting, SKUs with no contract on file,
  and making sure R2 and R9 don't double-flag the same missing-receipt case.
- **PostgreSQL schema + ETL** (`db/`): normalized relational schema with
  indexed foreign keys, loaded via `db/load_data.py`.
- **FastAPI backend** (`api/main.py`): `/flags`, `/flags/summary`,
  `/vendors/risk-ranking`, `/vendors/{id}` drill-down, `/flags/export.csv`.
- **Streamlit dashboard** (`dashboard/app.py`): KPIs, $-at-risk by rule,
  severity breakdown, vendor risk ranking, per-vendor drill-down, and
  filterable/exportable flag table.

## Rule recall against injected ground truth

Run `python3 scripts/run_and_evaluate.py` to reproduce this. Recall is
computed by matching flagged invoice/payment IDs against the IDs the
generator deliberately corrupted.

| Rule | Injected | Caught | Recall |
|---|---|---|---|
| R1 Duplicate Invoice | 230 | 230 | 100.0% |
| R2 Three-Way Match Failure | 290 | 289 | 99.7% |
| R3 Contract Price Violation | 260 | 260 | 100.0% |
| R4 Missing Purchase Order | 200 | 200 | 100.0% |
| R5 Duplicate Payment | 190 | 190 | 100.0% |
| R6 Tax Validation Failure | 220 | 220 | 100.0% |
| R7 Invoice Splitting | 329 | 274 | 83.3% |
| R8 Overpayment | 160 | 160 | 100.0% |
| R9 Payment Without Goods Receipt | 210 | 210 | 100.0% |

**The two non-100% numbers, explained rather than hidden:**

- **R2's one miss**: one injected mismatch happened to land on a PO with
  quantity 1, and the random offset applied clamped back to the same
  original quantity — so there genuinely was no mismatch left to detect.
  Not a rule bug, a generator edge case.
- **R7's ~83%**: a handful of split invoices land just over the $5,000
  approval threshold once tax is added on top of the pre-tax split. The
  rule correctly declines to flag these as "splitting to dodge approval,"
  since one of the resulting invoices still required approval on its own —
  which is the correct, stricter interpretation of what invoice splitting
  actually means.

A real bug was found and fixed during development: `rule_duplicate_payment`
originally assumed payment rows arrived in `payment_id` order and used
`iloc[1:]` per group to skip the "first" (real) payment. Since SQL doesn't
guarantee row order without an explicit `ORDER BY`, this sometimes flagged
the original payment instead of the duplicate. Fixed by sorting explicitly
on `payment_id` before grouping — recall went from 47% to 100%.

**R9 (Payment Without Goods Receipt) was added after the fact**, once R1–R8
were already validated. Adding it surfaced a real design question: `R2`
(three-way match) originally defaulted a missing receipt to `received_qty=0`,
which meant every R9 case would have also been double-counted as an R2
quantity mismatch — two rule codes firing for one root cause. Fixed by
having R2 only compare against a receipt when one actually exists; a PO with
*no* receipt at all is R9's job, not R2's. Confirmed with a dedicated test
(`test_three_way_match_does_not_double_flag_missing_receipt`) and by
checking R2's recall stayed at 99.7% (unchanged) after adding R9.

## Setup

```bash
# Requires PostgreSQL running locally (user 'postgres', password 'apguard', db 'apguard')
bash run.sh
```

Then:
```bash
uvicorn api.main:app --reload --port 8000        # API on :8000
streamlit run dashboard/app.py --server.port 8501 # Dashboard on :8501
```

## What's simplified vs. a production system

Worth knowing if this comes up in an interview:

- The "six data sources" are six related tables from one synthetic
  generator, not six genuinely independent upstream systems — a real AP
  system would integrate a separate ERP, procurement tool, contract
  repository, etc.
- Tax and contract-price rules only check against SKUs that have a contract
  on file; uncontracted spend isn't price-checked (by design — there's no
  authoritative price to check against).
- `TAX_TOLERANCE`, `SPLIT_WINDOW_DAYS`, and `APPROVAL_THRESHOLD` in
  `rules/engine.py` are illustrative constants, not derived from a real
  organization's approval policy.
- No auth on the API or dashboard — fine for a local demo, not for
  deployment as-is.
