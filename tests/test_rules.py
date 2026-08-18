import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from rules.engine import (
    rule_duplicate_invoice, rule_three_way_match, rule_contract_price,
    rule_missing_po, rule_duplicate_payment, rule_tax_validation,
    rule_invoice_splitting, rule_overpayment, rule_payment_without_gr, run_all_rules,
)


def inv_row(**kwargs):
    base = dict(invoice_id=1, vendor_id=1, po_id=1, invoice_number="INV-0001-000001",
                invoice_date="2026-01-01", item_sku="SKU-0001", quantity=10,
                unit_price=100.0, subtotal=1000.0, tax_amount=70.0, total_amount=1070.0)
    base.update(kwargs)
    return base


# --------------------------- R1: duplicate invoice ---------------------------
def test_duplicate_invoice_detected():
    invoices = pd.DataFrame([
        inv_row(invoice_id=1, invoice_number="INV-1"),
        inv_row(invoice_id=2, invoice_number="INV-2"),  # exact duplicate of invoice 1
        inv_row(invoice_id=3, invoice_number="INV-3", po_id=2),  # different PO, not a dupe
    ])
    flags = rule_duplicate_invoice(invoices)
    assert len(flags) == 2
    assert set(flags["invoice_id"]) == {1, 2}
    assert (flags["rule_code"] == "R1").all()


def test_no_duplicate_invoice_when_unique():
    invoices = pd.DataFrame([inv_row(invoice_id=1), inv_row(invoice_id=2, po_id=2)])
    flags = rule_duplicate_invoice(invoices)
    assert flags.empty


def test_duplicate_invoice_ignores_missing_po():
    """Invoices with no PO shouldn't be compared to each other for duplication here (R4's job)."""
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=None), inv_row(invoice_id=2, po_id=None)])
    flags = rule_duplicate_invoice(invoices)
    assert flags.empty


# --------------------------- R2: three-way match ---------------------------
def test_three_way_match_flags_qty_mismatch():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, quantity=15)])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None,
                          "item_sku": "SKU-0001", "quantity": 10, "unit_price": 100.0,
                          "po_date": "2026-01-01", "po_amount": 1000.0}])
    receipts = pd.DataFrame([{"gr_id": 1, "po_id": 10, "received_qty": 10, "received_date": "2026-01-05"}])
    flags = rule_three_way_match(invoices, pos, receipts)
    assert len(flags) == 1
    assert flags.iloc[0]["rule_code"] == "R2"
    assert flags.iloc[0]["amount_at_risk"] == 500.0  # 5 units * 100


def test_three_way_match_clean_when_aligned():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, quantity=10)])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None,
                          "item_sku": "SKU-0001", "quantity": 10, "unit_price": 100.0,
                          "po_date": "2026-01-01", "po_amount": 1000.0}])
    receipts = pd.DataFrame([{"gr_id": 1, "po_id": 10, "received_qty": 10, "received_date": "2026-01-05"}])
    flags = rule_three_way_match(invoices, pos, receipts)
    assert flags.empty


def test_three_way_match_skips_invoice_with_no_po():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=None)])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None,
                          "item_sku": "SKU-0001", "quantity": 10, "unit_price": 100.0,
                          "po_date": "2026-01-01", "po_amount": 1000.0}])
    receipts = pd.DataFrame([{"gr_id": 1, "po_id": 10, "received_qty": 10, "received_date": "2026-01-05"}])
    flags = rule_three_way_match(invoices, pos, receipts)
    assert flags.empty


# --------------------------- R3: contract price ---------------------------
def test_contract_price_violation_detected():
    invoices = pd.DataFrame([inv_row(vendor_id=1, item_sku="SKU-0001", unit_price=150.0, quantity=10)])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    flags = rule_contract_price(invoices, contracts)
    assert len(flags) == 1
    assert flags.iloc[0]["amount_at_risk"] == 500.0  # 50 overcharge * 10 qty


def test_contract_price_ok_when_matching():
    invoices = pd.DataFrame([inv_row(vendor_id=1, item_sku="SKU-0001", unit_price=100.0)])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    flags = rule_contract_price(invoices, contracts)
    assert flags.empty


def test_contract_price_skips_sku_with_no_contract():
    invoices = pd.DataFrame([inv_row(vendor_id=1, item_sku="SKU-9999", unit_price=9999.0)])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    flags = rule_contract_price(invoices, contracts)
    assert flags.empty


# --------------------------- R4: missing PO ---------------------------
def test_missing_po_detected():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=None), inv_row(invoice_id=2, po_id=5)])
    flags = rule_missing_po(invoices)
    assert len(flags) == 1
    assert flags.iloc[0]["invoice_id"] == 1


# --------------------------- R5: duplicate payment ---------------------------
def test_duplicate_payment_flags_extra_payments_only():
    payments = pd.DataFrame([
        {"payment_id": 1, "invoice_id": 100, "payment_date": "2026-01-01", "payment_amount": 500.0, "payment_ref": "A"},
        {"payment_id": 2, "invoice_id": 100, "payment_date": "2026-01-10", "payment_amount": 500.0, "payment_ref": "B"},
        {"payment_id": 3, "invoice_id": 200, "payment_date": "2026-01-01", "payment_amount": 300.0, "payment_ref": "C"},
    ])
    flags = rule_duplicate_payment(payments)
    assert len(flags) == 1
    assert flags.iloc[0]["payment_id"] == 2  # first payment kept, second flagged


def test_duplicate_payment_triple_payment_flags_two():
    payments = pd.DataFrame([
        {"payment_id": 1, "invoice_id": 100, "payment_date": "2026-01-01", "payment_amount": 500.0, "payment_ref": "A"},
        {"payment_id": 2, "invoice_id": 100, "payment_date": "2026-01-10", "payment_amount": 500.0, "payment_ref": "B"},
        {"payment_id": 3, "invoice_id": 100, "payment_date": "2026-01-15", "payment_amount": 500.0, "payment_ref": "C"},
    ])
    flags = rule_duplicate_payment(payments)
    assert len(flags) == 2


# --------------------------- R6: tax validation ---------------------------
def test_tax_mismatch_detected():
    invoices = pd.DataFrame([inv_row(po_id=10, subtotal=1000.0, tax_amount=50.0)])  # should be 70 at 7%
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": 1, "item_sku": "SKU-0001",
                          "quantity": 10, "unit_price": 100.0, "po_date": "2026-01-01", "po_amount": 1000.0}])
    flags = rule_tax_validation(invoices, contracts, pos)
    assert len(flags) == 1
    assert flags.iloc[0]["amount_at_risk"] == 20.0


def test_tax_correct_within_tolerance():
    invoices = pd.DataFrame([inv_row(po_id=10, subtotal=1000.0, tax_amount=70.0)])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": 1, "item_sku": "SKU-0001",
                          "quantity": 10, "unit_price": 100.0, "po_date": "2026-01-01", "po_amount": 1000.0}])
    flags = rule_tax_validation(invoices, contracts, pos)
    assert flags.empty


def test_tax_validation_skips_po_without_contract():
    invoices = pd.DataFrame([inv_row(po_id=10, subtotal=1000.0, tax_amount=0.0)])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None, "item_sku": "SKU-0001",
                          "quantity": 10, "unit_price": 100.0, "po_date": "2026-01-01", "po_amount": 1000.0}])
    flags = rule_tax_validation(invoices, contracts, pos)
    assert flags.empty  # no contract to check against -> not our claim to make


# --------------------------- R7: invoice splitting ---------------------------
def test_invoice_splitting_detected():
    invoices = pd.DataFrame([
        inv_row(invoice_id=1, po_id=10, total_amount=3000.0, invoice_date="2026-01-01"),
        inv_row(invoice_id=2, po_id=10, total_amount=3000.0, invoice_date="2026-01-03"),
    ])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None, "item_sku": "SKU-0001",
                          "quantity": 60, "unit_price": 100.0, "po_date": "2025-12-01", "po_amount": 6000.0}])
    flags = rule_invoice_splitting(invoices, pos)
    assert len(flags) == 2


def test_invoice_splitting_not_flagged_when_outside_window():
    invoices = pd.DataFrame([
        inv_row(invoice_id=1, po_id=10, total_amount=3000.0, invoice_date="2026-01-01"),
        inv_row(invoice_id=2, po_id=10, total_amount=3000.0, invoice_date="2026-03-01"),  # 59 days later
    ])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None, "item_sku": "SKU-0001",
                          "quantity": 60, "unit_price": 100.0, "po_date": "2025-12-01", "po_amount": 6000.0}])
    flags = rule_invoice_splitting(invoices, pos)
    assert flags.empty


def test_invoice_splitting_not_flagged_when_one_exceeds_threshold():
    invoices = pd.DataFrame([
        inv_row(invoice_id=1, po_id=10, total_amount=6000.0, invoice_date="2026-01-01"),  # already over threshold
        inv_row(invoice_id=2, po_id=10, total_amount=3000.0, invoice_date="2026-01-03"),
    ])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None, "item_sku": "SKU-0001",
                          "quantity": 90, "unit_price": 100.0, "po_date": "2025-12-01", "po_amount": 9000.0}])
    flags = rule_invoice_splitting(invoices, pos)
    assert flags.empty  # not "splitting" if one invoice already required approval


# --------------------------- R8: overpayment ---------------------------
def test_overpayment_detected():
    invoices = pd.DataFrame([inv_row(invoice_id=1, total_amount=1000.0)])
    payments = pd.DataFrame([{"payment_id": 1, "invoice_id": 1, "payment_date": "2026-01-05",
                               "payment_amount": 1200.0, "payment_ref": "A"}])
    flags = rule_overpayment(payments, invoices)
    assert len(flags) == 1
    assert flags.iloc[0]["amount_at_risk"] == 200.0


def test_no_overpayment_when_exact():
    invoices = pd.DataFrame([inv_row(invoice_id=1, total_amount=1000.0)])
    payments = pd.DataFrame([{"payment_id": 1, "invoice_id": 1, "payment_date": "2026-01-05",
                               "payment_amount": 1000.0, "payment_ref": "A"}])
    flags = rule_overpayment(payments, invoices)
    assert flags.empty


# --------------------------- R9: payment without goods receipt ---------------------------
def test_payment_without_gr_detected():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, vendor_id=1)])
    payments = pd.DataFrame([{"payment_id": 1, "invoice_id": 1, "payment_date": "2026-01-05",
                               "payment_amount": 1070.0, "payment_ref": "A"}])
    receipts = pd.DataFrame(columns=["gr_id", "po_id", "received_qty", "received_date"])  # no GR at all
    flags = rule_payment_without_gr(invoices, payments, receipts)
    assert len(flags) == 1
    assert flags.iloc[0]["rule_code"] == "R9"
    assert flags.iloc[0]["po_id"] == 10


def test_payment_without_gr_not_flagged_when_receipt_exists():
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, vendor_id=1)])
    payments = pd.DataFrame([{"payment_id": 1, "invoice_id": 1, "payment_date": "2026-01-05",
                               "payment_amount": 1070.0, "payment_ref": "A"}])
    receipts = pd.DataFrame([{"gr_id": 1, "po_id": 10, "received_qty": 10, "received_date": "2026-01-05"}])
    flags = rule_payment_without_gr(invoices, payments, receipts)
    assert flags.empty


def test_payment_without_gr_ignores_unpaid_invoices():
    """An invoice with no GR that hasn't been paid yet isn't leakage — nothing's
    gone out the door. R9 only fires once a payment actually exists."""
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, vendor_id=1)])
    payments = pd.DataFrame(columns=["payment_id", "invoice_id", "payment_date", "payment_amount", "payment_ref"])
    receipts = pd.DataFrame(columns=["gr_id", "po_id", "received_qty", "received_date"])
    flags = rule_payment_without_gr(invoices, payments, receipts)
    assert flags.empty


def test_three_way_match_does_not_double_flag_missing_receipt():
    """R2 should not also fire just because a receipt is absent — that's R9's
    job. R2 only checks quantity divergence against a receipt that exists."""
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, quantity=10)])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": None,
                          "item_sku": "SKU-0001", "quantity": 10, "unit_price": 100.0,
                          "po_date": "2026-01-01", "po_amount": 1000.0}])
    receipts = pd.DataFrame(columns=["gr_id", "po_id", "received_qty", "received_date"])  # no GR
    flags = rule_three_way_match(invoices, pos, receipts)
    assert flags.empty  # invoice qty matches PO qty; no receipt to compare against, so no R2 flag


# --------------------------- integration: run_all_rules ---------------------------
def test_run_all_rules_combines_all_flag_types():
    vendors = pd.DataFrame([{"vendor_id": 1, "vendor_name": "Acme", "tax_id": "X", "country": "US", "risk_category": "low"}])
    contracts = pd.DataFrame([{"contract_id": 1, "vendor_id": 1, "item_sku": "SKU-0001",
                                "contract_price": 100.0, "tax_rate": 0.07,
                                "start_date": "2024-01-01", "end_date": "2026-12-31"}])
    pos = pd.DataFrame([{"po_id": 10, "vendor_id": 1, "contract_id": 1, "item_sku": "SKU-0001",
                          "quantity": 10, "unit_price": 100.0, "po_date": "2026-01-01", "po_amount": 1000.0}])
    receipts = pd.DataFrame([{"gr_id": 1, "po_id": 10, "received_qty": 10, "received_date": "2026-01-05"}])
    invoices = pd.DataFrame([inv_row(invoice_id=1, po_id=10, vendor_id=1, item_sku="SKU-0001",
                                      quantity=10, unit_price=150.0, subtotal=1500.0,
                                      tax_amount=70.0, total_amount=1570.0)])
    payments = pd.DataFrame([{"payment_id": 1, "invoice_id": 1, "payment_date": "2026-01-10",
                               "payment_amount": 1570.0, "payment_ref": "A"}])
    flags = run_all_rules(vendors, contracts, pos, receipts, invoices, payments)
    assert "R3" in set(flags["rule_code"])  # contract price violation should be caught
    assert not flags.empty


def test_run_all_rules_empty_input_returns_empty_frame():
    empty = pd.DataFrame()
    flags = run_all_rules(empty, empty, empty, empty, empty, empty)
    assert flags.empty
