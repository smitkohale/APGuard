-- APGuard schema
DROP TABLE IF EXISTS flags CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS invoices CASCADE;
DROP TABLE IF EXISTS goods_receipts CASCADE;
DROP TABLE IF EXISTS purchase_orders CASCADE;
DROP TABLE IF EXISTS contracts CASCADE;
DROP TABLE IF EXISTS vendors CASCADE;

CREATE TABLE vendors (
    vendor_id       SERIAL PRIMARY KEY,
    vendor_name     TEXT NOT NULL,
    tax_id          TEXT NOT NULL,
    country         TEXT NOT NULL,
    risk_category   TEXT NOT NULL DEFAULT 'low',
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE contracts (
    contract_id      SERIAL PRIMARY KEY,
    vendor_id        INTEGER NOT NULL REFERENCES vendors(vendor_id),
    item_sku         TEXT NOT NULL,
    contract_price   NUMERIC(12,2) NOT NULL,
    tax_rate         NUMERIC(5,4) NOT NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL
);

CREATE TABLE purchase_orders (
    po_id           SERIAL PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id),
    contract_id     INTEGER REFERENCES contracts(contract_id),
    item_sku        TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL,
    po_date         DATE NOT NULL,
    po_amount       NUMERIC(14,2) NOT NULL
);

CREATE TABLE goods_receipts (
    gr_id           SERIAL PRIMARY KEY,
    po_id           INTEGER NOT NULL REFERENCES purchase_orders(po_id),
    received_qty    INTEGER NOT NULL,
    received_date   DATE NOT NULL
);

CREATE TABLE invoices (
    invoice_id      SERIAL PRIMARY KEY,
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
    payment_id      SERIAL PRIMARY KEY,
    invoice_id      INTEGER NOT NULL REFERENCES invoices(invoice_id),
    payment_date    DATE NOT NULL,
    payment_amount  NUMERIC(14,2) NOT NULL,
    payment_ref     TEXT NOT NULL
);

CREATE TABLE flags (
    flag_id         SERIAL PRIMARY KEY,
    rule_code       TEXT NOT NULL,
    rule_name       TEXT NOT NULL,
    vendor_id       INTEGER REFERENCES vendors(vendor_id),
    invoice_id      INTEGER REFERENCES invoices(invoice_id),
    po_id           INTEGER REFERENCES purchase_orders(po_id),
    payment_id      INTEGER REFERENCES payments(payment_id),
    severity        TEXT NOT NULL,
    amount_at_risk  NUMERIC(14,2) NOT NULL DEFAULT 0,
    details         TEXT NOT NULL,
    is_ground_truth TEXT NOT NULL DEFAULT 'unknown', -- 'injected'/'clean'/'unknown', used only for eval
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_invoices_vendor ON invoices(vendor_id);
CREATE INDEX idx_invoices_po ON invoices(po_id);
CREATE INDEX idx_payments_invoice ON payments(invoice_id);
CREATE INDEX idx_po_vendor ON purchase_orders(vendor_id);
CREATE INDEX idx_po_contract ON purchase_orders(contract_id);
CREATE INDEX idx_gr_po ON goods_receipts(po_id);
CREATE INDEX idx_flags_rule ON flags(rule_code);
CREATE INDEX idx_flags_vendor ON flags(vendor_id);
