"""Unit tests for every detector, built on small hand-written fixtures."""
import pandas as pd
from conftest import contract, df, gr, inv, invoices, pay, po, vendor

from rules.engine import (
    FLAG_COLUMNS,
    benford_test,
    find_duplicate_invoices,
    normalize_invoice_number,
    rule_benford,
    rule_contract_price,
    rule_duplicate_invoice,
    rule_duplicate_payment,
    rule_duplicate_vendor,
    rule_invoice_splitting,
    rule_missing_po,
    rule_new_vendor_payment,
    rule_overpayment,
    rule_payment_without_gr,
    rule_price_outlier,
    rule_tax_validation,
    rule_three_way_match,
    run_all_rules,
)


# ------------------------------------------------------------------ R1 duplicate invoice
def test_duplicate_invoice_flags_only_the_later_copy():
    i = invoices(inv(invoice_id=1, invoice_number="INV-1"),
                 inv(invoice_id=2, invoice_number="INV-2", invoice_date="2026-01-10"),
                 inv(invoice_id=3, invoice_number="INV-3", po_id=20))
    flags = rule_duplicate_invoice(i, df(po(), po(po_id=20)))
    assert flags["invoice_id"].tolist() == [2]
    assert flags.iloc[0]["rule_code"] == "R1"


def test_duplicate_invoice_catches_reformatted_invoice_number():
    i = invoices(inv(invoice_id=1, invoice_number="INV-0001-000123"),
                 inv(invoice_id=2, invoice_number="inv/0001/000123 ", po_id=None, quantity=3, total_amount=10.0))
    assert rule_duplicate_invoice(i, df(po()))["invoice_id"].tolist() == [2]


def test_duplicate_invoice_tolerates_small_amount_drift():
    i = invoices(inv(invoice_id=1, invoice_number="A"),
                 inv(invoice_id=2, invoice_number="B", total_amount=1070.42, invoice_date="2026-01-12"))
    assert rule_duplicate_invoice(i, df(po()))["invoice_id"].tolist() == [2]


def test_exact_mode_misses_amount_drift():
    i = invoices(inv(invoice_id=1, invoice_number="A"),
                 inv(invoice_id=2, invoice_number="B", total_amount=1070.42, invoice_date="2026-01-12"))
    assert rule_duplicate_invoice(i, df(po()), {"duplicate_invoice": {"fuzzy": False}}).empty


def test_duplicate_invoice_ignores_partial_deliveries():
    """Two equal invoices that together make up the ordered quantity are instalments."""
    i = invoices(inv(invoice_id=1, invoice_number="A", quantity=5, total_amount=535.0),
                 inv(invoice_id=2, invoice_number="B", quantity=5, total_amount=535.0, invoice_date="2026-01-15"))
    assert rule_duplicate_invoice(i, df(po(quantity=10))).empty


def test_duplicate_invoice_rekeyed_without_po_within_window():
    i = invoices(inv(invoice_id=1, invoice_number="A"),
                 inv(invoice_id=2, invoice_number="B", po_id=None, invoice_date="2026-01-20"))
    assert rule_duplicate_invoice(i, df(po()))["invoice_id"].tolist() == [2]


def test_duplicate_invoice_rekeyed_outside_window_is_not_matched():
    i = invoices(inv(invoice_id=1, invoice_number="A"),
                 inv(invoice_id=2, invoice_number="B", po_id=None, invoice_date="2026-04-01"))
    assert rule_duplicate_invoice(i, df(po())).empty


def test_recurring_billing_on_separate_pos_is_not_a_duplicate():
    i = invoices(inv(invoice_id=1, invoice_number="A", po_id=10),
                 inv(invoice_id=2, invoice_number="B", po_id=11, invoice_date="2026-02-01"))
    assert rule_duplicate_invoice(i, df(po(po_id=10), po(po_id=11))).empty


def test_duplicate_invoice_without_po_on_both_sides_is_not_content_matched():
    i = invoices(inv(invoice_id=1, invoice_number="A", po_id=None),
                 inv(invoice_id=2, invoice_number="B", po_id=None, invoice_date="2026-03-15"))
    assert rule_duplicate_invoice(i, df(po())).empty


def test_normalize_invoice_number():
    s = pd.Series(["inv-0001/000123 ", " INV 0001 000123", "INV0001000123"])
    assert normalize_invoice_number(s).nunique() == 1


def test_find_duplicates_reports_original():
    i = invoices(inv(invoice_id=1, invoice_number="X-1"), inv(invoice_id=2, invoice_number="x1", po_id=None))
    d = find_duplicate_invoices(i, df(po()))
    assert d.iloc[0]["original_invoice_id"] == 1


# ------------------------------------------------------------------ R2 three-way match
def test_three_way_match_flags_over_billing_with_exposure():
    flags = rule_three_way_match(invoices(inv(quantity=15)), df(po()), df(gr()))
    assert len(flags) == 1
    assert flags.iloc[0]["amount_at_risk"] == 500.0
    assert flags.iloc[0]["severity"] == "high"


def test_three_way_match_under_billing_is_low_with_no_exposure():
    flags = rule_three_way_match(invoices(inv(quantity=7)), df(po()), df(gr()))
    assert flags.iloc[0]["severity"] == "low"
    assert flags.iloc[0]["amount_at_risk"] == 0.0


def test_three_way_match_short_receipt_is_over_billing():
    flags = rule_three_way_match(invoices(inv(quantity=10)), df(po()), df(gr(received_qty=8)))
    assert flags.iloc[0]["amount_at_risk"] == 200.0


def test_three_way_match_reconciles_partial_deliveries():
    i = invoices(inv(invoice_id=1, invoice_number="A", quantity=4, total_amount=428.0),
                 inv(invoice_id=2, invoice_number="B", quantity=6, total_amount=642.0, invoice_date="2026-01-20"))
    r = df(gr(gr_id=1, received_qty=4), gr(gr_id=2, received_qty=6))
    assert rule_three_way_match(i, df(po()), r).empty


def test_three_way_match_clean_when_aligned():
    assert rule_three_way_match(invoices(inv()), df(po()), df(gr())).empty


def test_three_way_match_skips_invoice_with_no_po():
    assert rule_three_way_match(invoices(inv(po_id=None, quantity=99)), df(po()), df(gr())).empty


def test_three_way_match_does_not_double_flag_missing_receipt():
    """A PO with no receipt is R9's problem, not a quantity mismatch."""
    i = invoices(inv())
    assert rule_three_way_match(i, df(po()), df(gr(po_id=999))).empty
    assert len(rule_payment_without_gr(i, df(pay()), df(gr(po_id=999)))) == 1


def test_three_way_match_ignores_same_po_duplicate():
    """The duplicate is reported once, by R1; R2 should not also call it over-billing."""
    i = invoices(inv(invoice_id=1, invoice_number="A"), inv(invoice_id=2, invoice_number="B", invoice_date="2026-01-09"))
    assert rule_three_way_match(i, df(po()), df(gr())).empty


# ------------------------------------------------------------------ R3 contract price
def test_contract_price_violation_detected():
    flags = rule_contract_price(invoices(inv(po_id=None, unit_price=150.0)), df(contract()))
    assert flags.iloc[0]["amount_at_risk"] == 500.0


def test_contract_price_ok_when_matching():
    assert rule_contract_price(invoices(inv(po_id=None)), df(contract())).empty


def test_contract_price_skips_sku_with_no_contract():
    assert rule_contract_price(invoices(inv(po_id=None, item_sku="SKU-9", unit_price=999.0)), df(contract())).empty


def test_contract_price_respects_amendments():
    """Price rose mid-term: an invoice at the new price after the amendment is fine."""
    c = df(contract(contract_id=1, end_date="2025-06-30"),
           contract(contract_id=2, contract_price=106.0, start_date="2025-07-01"))
    i = invoices(inv(invoice_id=1, po_id=None, unit_price=106.0, invoice_date="2025-09-01"),
                 inv(invoice_id=2, po_id=None, unit_price=106.0, invoice_date="2025-03-01"))
    assert rule_contract_price(i, c)["invoice_id"].tolist() == [2]


def test_contract_price_uses_po_contract_when_linked():
    c = df(contract(contract_id=1, contract_price=90.0), contract(contract_id=2, contract_price=100.0, item_sku="SKU-0001"))
    i = invoices(inv(unit_price=95.0))
    flags = rule_contract_price(i, c, df(po(contract_id=1)))
    assert flags.iloc[0]["amount_at_risk"] == 50.0


# ------------------------------------------------------------------ R4 missing PO
def test_missing_po_detected():
    flags = rule_missing_po(invoices(inv(invoice_id=1, po_id=None), inv(invoice_id=2)))
    assert flags["invoice_id"].tolist() == [1]


# ------------------------------------------------------------------ R5 duplicate payment
def test_duplicate_payment_flags_extra_payments_only():
    p = df(pay(payment_id=1, invoice_id=100), pay(payment_id=2, invoice_id=100, payment_date="2026-02-01"),
           pay(payment_id=3, invoice_id=200))
    assert rule_duplicate_payment(p)["payment_id"].tolist() == [2]


def test_duplicate_payment_ordering_is_by_date_not_arrival():
    p = df(pay(payment_id=2, invoice_id=100, payment_date="2026-02-01"), pay(payment_id=1, invoice_id=100))
    assert rule_duplicate_payment(p)["payment_id"].tolist() == [2]


def test_duplicate_payment_triple_flags_two():
    p = df(pay(payment_id=1), pay(payment_id=2, payment_date="2026-02-01"), pay(payment_id=3, payment_date="2026-03-01"))
    assert len(rule_duplicate_payment(p)) == 2


def test_duplicate_payment_carries_vendor():
    """Payment-level flags must carry vendor_id, or they vanish from vendor risk views."""
    p = df(pay(payment_id=1), pay(payment_id=2, payment_date="2026-02-01"))
    flags = rule_duplicate_payment(p, invoices(inv(vendor_id=42)))
    assert flags.iloc[0]["vendor_id"] == 42


# ------------------------------------------------------------------ R6 tax
def test_tax_mismatch_detected():
    flags = rule_tax_validation(invoices(inv(tax_amount=50.0)), df(contract()), df(po(contract_id=1)))
    assert flags.iloc[0]["amount_at_risk"] == 20.0


def test_tax_rounding_cent_is_tolerated():
    i = invoices(inv(subtotal=20.0, tax_amount=1.41))  # expected 1.40
    assert rule_tax_validation(i, df(contract()), df(po(contract_id=1))).empty


def test_tax_validation_skips_po_without_contract():
    assert rule_tax_validation(invoices(inv(tax_amount=0.0)), df(contract()), df(po())).empty


# ------------------------------------------------------------------ R7 splitting
def split_invoices():
    return invoices(inv(invoice_id=1, invoice_number="A", quantity=50, total_amount=3000.0),
                    inv(invoice_id=2, invoice_number="B", quantity=50, total_amount=3000.0, invoice_date="2026-01-03"))


def test_invoice_splitting_detected():
    flags = rule_invoice_splitting(split_invoices(), df(po(quantity=100)), df(gr(received_qty=100)))
    assert sorted(flags["invoice_id"]) == [1, 2]


def test_invoice_splitting_outside_window():
    i = split_invoices()
    i.loc[1, "invoice_date"] = "2026-02-15"
    assert rule_invoice_splitting(i, df(po(quantity=100)), df(gr(received_qty=100))).empty


def test_invoice_splitting_not_flagged_when_one_exceeds_threshold():
    i = split_invoices()
    i.loc[0, "total_amount"] = 5200.0
    assert rule_invoice_splitting(i, df(po(quantity=100)), df(gr(received_qty=100))).empty


def test_invoice_splitting_ignores_delivery_backed_billing():
    r = df(gr(gr_id=1, received_qty=50), gr(gr_id=2, received_qty=50))
    assert rule_invoice_splitting(split_invoices(), df(po(quantity=100)), r).empty


# ------------------------------------------------------------------ R8 overpayment
def test_overpayment_detected():
    flags = rule_overpayment(df(pay(payment_amount=1200.0)), invoices(inv(vendor_id=7)))
    assert flags.iloc[0]["amount_at_risk"] == 130.0
    assert flags.iloc[0]["vendor_id"] == 7


def test_overpayment_ignores_cent_rounding():
    assert rule_overpayment(df(pay(payment_amount=1070.01)), invoices(inv())).empty


# ------------------------------------------------------------------ R9
def test_payment_without_gr_ignores_unpaid_invoices():
    assert rule_payment_without_gr(invoices(inv()), df(pay(invoice_id=999)), df(gr(po_id=1))).empty


def test_payment_without_gr_not_flagged_when_receipt_exists():
    assert rule_payment_without_gr(invoices(inv()), df(pay()), df(gr())).empty


# ------------------------------------------------------------------ R10 duplicate vendor
def test_duplicate_vendor_flags_later_record_sharing_bank_account():
    v = df(vendor(vendor_id=1, created_at="2022-01-01"),
           vendor(vendor_id=2, vendor_name="ACME SUPPLIES LTD", tax_id="ZZ-9", created_at="2025-03-01"),
           vendor(vendor_id=3, vendor_name="Other", tax_id="CC-3", bank_account="GB00OTHER"))
    i = invoices(inv(invoice_id=5, vendor_id=2))
    flags = rule_duplicate_vendor(v, i, df(pay(invoice_id=5, payment_amount=900.0)))
    assert flags["vendor_id"].tolist() == [2]
    assert flags.iloc[0]["amount_at_risk"] == 900.0
    assert "bank account" in flags.iloc[0]["details"]


def test_duplicate_vendor_none_when_unique():
    v = df(vendor(vendor_id=1), vendor(vendor_id=2, tax_id="X", bank_account="Y"))
    assert rule_duplicate_vendor(v).empty


# ------------------------------------------------------------------ R11 new vendor payment
def test_new_vendor_large_payment():
    v = df(vendor(created_at="2026-01-10"))
    i = invoices(inv())
    flags = rule_new_vendor_payment(v, i, df(pay(payment_amount=25000.0, payment_date="2026-01-20")))
    assert len(flags) == 1


def test_new_vendor_small_or_late_payment_not_flagged():
    v = df(vendor(created_at="2026-01-10"))
    i = invoices(inv())
    p = df(pay(payment_id=1, payment_amount=500.0, payment_date="2026-01-20"),
           pay(payment_id=2, payment_amount=25000.0, payment_date="2026-03-20"))
    assert rule_new_vendor_payment(v, i, p).empty


# ------------------------------------------------------------------ A1 Benford, A2 price outlier
def test_benford_conforming_data_passes():
    import numpy as np
    rng = np.random.default_rng(0)
    amounts = pd.Series(10 ** rng.uniform(1, 5, 3000))
    res = benford_test(amounts)
    assert res["mad"] < 0.006


def test_benford_flags_amounts_clustered_under_limit():
    import numpy as np
    rng = np.random.default_rng(1)
    rows = [inv(invoice_id=k, total_amount=float(rng.uniform(4500, 4990))) for k in range(150)]
    flags = rule_benford(invoices(*rows))
    assert flags["vendor_id"].tolist() == [1]
    assert "digit 4" in flags.iloc[0]["details"]


def test_price_outlier_flags_inflated_unit_price():
    rows = [inv(invoice_id=k, unit_price=100.0 + (k % 5)) for k in range(30)]
    rows.append(inv(invoice_id=99, unit_price=190.0))
    flags = rule_price_outlier(invoices(*rows))
    assert flags["invoice_id"].tolist() == [99]


def test_price_outlier_needs_enough_history():
    rows = [inv(invoice_id=k, unit_price=100.0) for k in range(5)] + [inv(invoice_id=99, unit_price=500.0)]
    assert rule_price_outlier(invoices(*rows)).empty


# ------------------------------------------------------------------ orchestration
def test_run_all_rules_combines_flag_types():
    i = invoices(inv(invoice_id=1, invoice_number="A"),
                 inv(invoice_id=2, invoice_number="B", invoice_date="2026-01-08"),
                 inv(invoice_id=3, invoice_number="C", po_id=None))
    p = df(pay(payment_id=1, invoice_id=1), pay(payment_id=2, invoice_id=1, payment_date="2026-02-01"),
           pay(payment_id=3, invoice_id=3, payment_amount=2000.0))
    flags = run_all_rules(df(vendor()), df(contract()), df(po()), df(gr()), i, p)
    assert {"R1", "R4", "R5", "R8"} <= set(flags["rule_code"])
    assert list(flags.columns) == FLAG_COLUMNS


def test_run_all_rules_subset():
    flags = run_all_rules(df(vendor()), df(contract()), df(po()), df(gr()), invoices(inv(po_id=None)), df(pay()),
                          rules=["R4"])
    assert set(flags["rule_code"]) == {"R4"}


def test_run_all_rules_empty_input_returns_empty_frame():
    e = pd.DataFrame()
    empty_inv = invoices(inv()).iloc[0:0]
    flags = run_all_rules(df(vendor()).iloc[0:0], e, df(po()).iloc[0:0], df(gr()).iloc[0:0], empty_inv,
                          df(pay()).iloc[0:0])
    assert flags.empty
    assert list(flags.columns) == FLAG_COLUMNS
