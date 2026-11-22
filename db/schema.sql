-- APGuard schema (PostgreSQL 14+).
-- Re-running this file drops and recreates everything, including review history.

DROP VIEW IF EXISTS v_rule_r4_missing_po, v_rule_r5_duplicate_payment, v_rule_r8_overpayment,
    v_invoice_exposure, v_active_flags CASCADE;
DROP TABLE IF EXISTS flag_events, flags, rule_runs, payments, invoices, goods_receipts,
    purchase_orders, contracts, vendors CASCADE;

-- --------------------------------------------------------------------------
-- Source data
-- --------------------------------------------------------------------------
CREATE TABLE vendors (
    vendor_id       INTEGER PRIMARY KEY,
    vendor_name     TEXT NOT NULL,
    tax_id          TEXT NOT NULL,
    bank_account    TEXT NOT NULL,
    country         TEXT NOT NULL,
    risk_category   TEXT NOT NULL DEFAULT 'low' CHECK (risk_category IN ('low', 'medium', 'high')),
    created_at      DATE NOT NULL
);

CREATE TABLE contracts (
    contract_id     INTEGER PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id),
    item_sku        TEXT NOT NULL,
    contract_price  NUMERIC(12,2) NOT NULL CHECK (contract_price >= 0),
    tax_rate        NUMERIC(5,4) NOT NULL CHECK (tax_rate >= 0),
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    CHECK (end_date >= start_date)
);

CREATE TABLE purchase_orders (
    po_id           INTEGER PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id),
    contract_id     INTEGER REFERENCES contracts(contract_id),
    item_sku        TEXT NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(12,2) NOT NULL,
    po_date         DATE NOT NULL,
    po_amount       NUMERIC(14,2) NOT NULL
);

CREATE TABLE goods_receipts (
    gr_id           INTEGER PRIMARY KEY,
    po_id           INTEGER NOT NULL REFERENCES purchase_orders(po_id),
    received_qty    INTEGER NOT NULL,
    received_date   DATE NOT NULL
);

CREATE TABLE invoices (
    invoice_id      INTEGER PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id),
    po_id           INTEGER REFERENCES purchase_orders(po_id),
    invoice_number  TEXT NOT NULL,
    invoice_date    DATE NOT NULL,
    item_sku        TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL,
    subtotal        NUMERIC(14,2) NOT NULL,
    tax_amount      NUMERIC(14,2) NOT NULL,
    total_amount    NUMERIC(14,2) NOT NULL
);

CREATE TABLE payments (
    payment_id      INTEGER PRIMARY KEY,
    invoice_id      INTEGER NOT NULL REFERENCES invoices(invoice_id),
    payment_date    DATE NOT NULL,
    payment_amount  NUMERIC(14,2) NOT NULL,
    payment_ref     TEXT NOT NULL
);

-- --------------------------------------------------------------------------
-- Detection runs and flags
-- --------------------------------------------------------------------------
CREATE TABLE rule_runs (
    run_id          SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    engine_version  TEXT NOT NULL,
    config          JSONB NOT NULL,          -- exact thresholds used
    flags_raised    INTEGER,
    flags_new       INTEGER,
    flags_retired   INTEGER,
    runtime_seconds NUMERIC(10,3)
);

CREATE TABLE flags (
    flag_id            SERIAL PRIMARY KEY,
    flag_key           TEXT NOT NULL UNIQUE,  -- rule|vendor|invoice|payment: stable across runs
    rule_code          TEXT NOT NULL,
    rule_name          TEXT NOT NULL,
    category           TEXT NOT NULL CHECK (category IN ('control', 'analytic')),
    vendor_id          INTEGER REFERENCES vendors(vendor_id),
    invoice_id         INTEGER REFERENCES invoices(invoice_id),
    po_id              INTEGER REFERENCES purchase_orders(po_id),
    payment_id         INTEGER REFERENCES payments(payment_id),
    severity           TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    amount_at_risk     NUMERIC(14,2) NOT NULL DEFAULT 0,
    details            TEXT NOT NULL,
    -- review workflow
    status             TEXT NOT NULL DEFAULT 'open'
                       CHECK (status IN ('open', 'in_review', 'confirmed', 'false_positive', 'recovered')),
    assignee           TEXT,
    recovered_amount   NUMERIC(14,2) NOT NULL DEFAULT 0,
    -- lineage
    first_seen_run_id  INTEGER NOT NULL REFERENCES rule_runs(run_id),
    last_seen_run_id   INTEGER NOT NULL REFERENCES rule_runs(run_id),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE flag_events (
    event_id        SERIAL PRIMARY KEY,
    flag_id         INTEGER NOT NULL REFERENCES flags(flag_id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,           -- created | status_change | assigned | note | recovery
    from_status     TEXT,
    to_status       TEXT,
    actor           TEXT NOT NULL,
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_contracts_vendor_sku ON contracts(vendor_id, item_sku);
CREATE INDEX idx_po_vendor ON purchase_orders(vendor_id);
CREATE INDEX idx_po_contract ON purchase_orders(contract_id);
CREATE INDEX idx_gr_po ON goods_receipts(po_id);
CREATE INDEX idx_invoices_vendor ON invoices(vendor_id);
CREATE INDEX idx_invoices_po ON invoices(po_id);
CREATE INDEX idx_payments_invoice ON payments(invoice_id);
CREATE INDEX idx_flags_rule ON flags(rule_code);
CREATE INDEX idx_flags_vendor ON flags(vendor_id);
CREATE INDEX idx_flags_invoice ON flags(invoice_id);
CREATE INDEX idx_flags_status ON flags(status);
CREATE INDEX idx_flags_last_seen ON flags(last_seen_run_id);
CREATE INDEX idx_flag_events_flag ON flag_events(flag_id);

-- --------------------------------------------------------------------------
-- Views
-- --------------------------------------------------------------------------
-- Flags raised by the most recent run. Older flags that a re-run no longer
-- reproduces stay in `flags` (with their review history) but drop out of here.
CREATE VIEW v_active_flags AS
SELECT f.*, v.vendor_name
FROM flags f
LEFT JOIN vendors v ON v.vendor_id = f.vendor_id
WHERE f.last_seen_run_id = (SELECT max(run_id) FROM rule_runs);

-- One invoice can trip several controls (e.g. R1 and R4). Summing every flag
-- double-counts it, so net exposure takes the largest single control flag per invoice.
CREATE VIEW v_invoice_exposure AS
SELECT invoice_id, max(vendor_id) AS vendor_id, max(amount_at_risk) AS exposure,
       count(*) AS flag_count, string_agg(rule_code, ',' ORDER BY rule_code) AS rules
FROM v_active_flags
WHERE category = 'control' AND invoice_id IS NOT NULL
  AND status <> 'false_positive'
GROUP BY invoice_id;

-- SQL implementations of three rules. tests/test_sql_parity.py checks that they
-- return exactly the same entities as the pandas engine.
CREATE VIEW v_rule_r4_missing_po AS
SELECT i.invoice_id, i.vendor_id, i.total_amount AS amount_at_risk
FROM invoices i
WHERE i.po_id IS NULL;

CREATE VIEW v_rule_r5_duplicate_payment AS
SELECT payment_id, invoice_id, payment_amount AS amount_at_risk
FROM (
    SELECT p.*, row_number() OVER (PARTITION BY p.invoice_id
                                   ORDER BY p.payment_date, p.payment_id) AS rn
    FROM payments p
) ranked
WHERE rn > 1;

CREATE VIEW v_rule_r8_overpayment AS
SELECT p.payment_id, p.invoice_id, i.vendor_id,
       p.payment_amount - i.total_amount AS amount_at_risk
FROM payments p
JOIN invoices i ON i.invoice_id = p.invoice_id
WHERE p.payment_amount - i.total_amount > GREATEST(0.05, 0.001 * i.total_amount);
