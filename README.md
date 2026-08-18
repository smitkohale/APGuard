APGuard — Accounts Payable Leakage Detection

A rule-based analytics system that detects payment leakage and contract compliance issues in Accounts Payable data: duplicate payments, three-way match failures, contract price violations, invoice splitting, tax mismatches, overpayments, and payments issued without a goods receipt on file.

Stack: Python · PostgreSQL · FastAPI · SQLAlchemy · Pandas · Streamlit · Plotly · pytest

What's in this repo
Synthetic dataset — 53,158 records across 6 linked entities (vendors, contracts, purchase orders, goods receipts, invoices, payments), produced by data/generate_data.py. 2,089 exceptions are deliberately injected and logged to data/generated/injected_exceptions.csv as ground truth. Generated CSVs are excluded from Git and can be recreated locally with the generator.
9 business validation rules (rules/engine.py) — each a pure function that takes DataFrames and returns flags, with no DB access inside the rule logic, so each rule is independently testable.
26 pytest cases (tests/test_rules.py) against hand-built fixtures rather than the generated data — covering edge cases such as triple payments, the threshold boundary for invoice splitting, SKUs with no contract on file, and confirming R2 and R9 don't double-flag the same missing-receipt case.
PostgreSQL schema and ETL (db/) — normalized relational schema with indexed foreign keys, loaded via db/load_data.py.
FastAPI backend (api/main.py) — /flags, /flags/summary, /vendors/risk-ranking, /vendors/{id} drill-down, and /flags/export.csv.
Streamlit dashboard (dashboard/app.py) — KPIs, dollars-at-risk by rule, severity breakdown, vendor risk ranking, per-vendor drill-down, and a filterable/exportable flag table.
Rule recall against injected ground truth

Run python scripts/run_and_evaluate.py to reproduce these results. Recall is computed by matching flagged invoice/payment IDs against the IDs the generator deliberately corrupted.

Rule	Injected	Caught	Recall
R1 Duplicate Invoice	230	230	100.0%
R2 Three-Way Match Failure	290	289	99.7%
R3 Contract Price Violation	260	260	100.0%
R4 Missing Purchase Order	200	200	100.0%
R5 Duplicate Payment	190	190	100.0%
R6 Tax Validation Failure	220	220	100.0%
R7 Invoice Splitting	329	274	83.3%
R8 Overpayment	160	160	100.0%
R9 Payment Without Goods Receipt	210	210	100.0%

On the two rules below 100%:

R2's single miss — one injected mismatch landed on a PO with quantity 1, and the random offset applied by the generator clamped back to the original quantity, leaving no actual mismatch to detect. This is a generator edge case, not a rule defect.
R7's 83.3% — a subset of split invoices land just over the $5,000 approval threshold once tax is applied to the pre-tax split amount. The rule correctly declines to flag these as threshold evasion, since one of the resulting invoices still required approval on its own. This is the stricter and more accurate interpretation of what invoice splitting means.

Bug found and fixed during development: rule_duplicate_payment originally assumed payment rows arrived in payment_id order and used iloc[1:] per group to skip the "first" (real) payment. Since SQL does not guarantee row order without an explicit ORDER BY, this occasionally flagged the original payment instead of the duplicate. Fixing it to sort explicitly on payment_id before grouping raised recall from 47% to 100%.

Design issue surfaced by R9: R9 (Payment Without Goods Receipt) was added after R1–R8 were already validated. Adding it exposed a flaw in R2 (three-way match), which had defaulted a missing receipt to received_qty=0 — meaning every R9 case was also being double-counted as an R2 quantity mismatch, with two rule codes firing for one root cause. This was fixed by having R2 compare against a receipt only when one exists; a PO with no receipt at all is R9's responsibility, not R2's. Verified with a dedicated test (test_three_way_match_does_not_double_flag_missing_receipt) and by confirming R2's recall held at 99.7% after R9 was added.

Setup

1. Configure PostgreSQL

Create a PostgreSQL database named apguard, then create a .env file in the project root:

DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@127.0.0.1:5432/apguard

.env is excluded from Git.

2. Install dependencies

bash
pip install -r requirements.txt

3. Generate the synthetic data

bash
python data/generate_data.py

Creates the six related CSV datasets and the injected-exception ground truth.

4. Create the database schema

bash
psql -h 127.0.0.1 -U postgres -d apguard -f db/schema.sql

5. Load the data into PostgreSQL

bash
python db/load_data.py

6. Run the detection engine and evaluation

bash
python scripts/run_and_evaluate.py

7. Start the API

bash
uvicorn api.main:app --reload --port 8000

API available at http://127.0.0.1:8000

8. Start the dashboard

bash
streamlit run dashboard/app.py --server.port 8501

Dashboard available at http://localhost:8501

Scope and limitations

For context if this comes up in review or an interview:

The "six data sources" are six related tables from one synthetic generator, not six independent upstream systems. A production AP environment would integrate a separate ERP, procurement tool, and contract repository.
Tax and contract-price rules only check SKUs that have a contract on file. Uncontracted spend is not price-checked, by design — there is no authoritative price to validate against.
TAX_TOLERANCE, SPLIT_WINDOW_DAYS, and APPROVAL_THRESHOLD in rules/engine.py are illustrative constants, not derived from any real organization's approval policy.
The API and dashboard have no authentication. Sufficient for a local demo, not for deployment as-is.
All data is synthetic and does not represent real company, vendor, invoice, or payment records.
