"""Property-based tests: invariants that must hold for any input, not just fixtures."""
import pandas as pd
from conftest import df, gr, inv, invoices, pay, po
from hypothesis import given, settings
from hypothesis import strategies as st

from rules.engine import (
    normalize_invoice_number,
    rule_duplicate_invoice,
    rule_duplicate_payment,
    rule_overpayment,
    rule_three_way_match,
)

money = st.floats(min_value=1, max_value=1_000_000, allow_nan=False).map(lambda x: round(x, 2))
qty = st.integers(min_value=1, max_value=10_000)


@given(total=money, q=qty)
@settings(max_examples=60, deadline=None)
def test_exact_resubmission_is_always_caught(total, q):
    i = invoices(inv(invoice_id=1, invoice_number="A", quantity=q, total_amount=total),
                 inv(invoice_id=2, invoice_number="B", quantity=q, total_amount=total, invoice_date="2026-01-15"))
    flags = rule_duplicate_invoice(i, df(po(quantity=q)))
    assert flags["invoice_id"].tolist() == [2]


@given(total=money, q=qty)
@settings(max_examples=60, deadline=None)
def test_duplicate_invoice_never_flags_the_original(total, q):
    i = invoices(inv(invoice_id=1, invoice_number="A", quantity=q, total_amount=total),
                 inv(invoice_id=2, invoice_number="A ", quantity=q, total_amount=total, invoice_date="2026-02-01"),
                 inv(invoice_id=3, invoice_number="a", quantity=q, total_amount=total, invoice_date="2026-03-01"))
    flagged = set(rule_duplicate_invoice(i, df(po(quantity=q)))["invoice_id"])
    assert 1 not in flagged and flagged == {2, 3}


@given(n=st.integers(min_value=1, max_value=8), amount=money)
@settings(max_examples=40, deadline=None)
def test_duplicate_payment_flags_exactly_n_minus_one(n, amount):
    p = df(*[pay(payment_id=k, payment_amount=amount, payment_date=f"2026-01-{k + 1:02d}") for k in range(n)])
    assert len(rule_duplicate_payment(p)) == n - 1


@given(total=money, extra=st.floats(min_value=0, max_value=0.04))
@settings(max_examples=60, deadline=None)
def test_overpayment_within_tolerance_never_flags(total, extra):
    p = df(pay(payment_amount=round(total + extra, 2)))
    assert rule_overpayment(p, invoices(inv(total_amount=total))).empty


@given(total=money, factor=st.floats(min_value=1.01, max_value=3))
@settings(max_examples=60, deadline=None)
def test_overpayment_beyond_one_percent_always_flags(total, factor):
    p = df(pay(payment_amount=round(total * factor + 0.06, 2)))
    assert len(rule_overpayment(p, invoices(inv(total_amount=total)))) == 1


@given(parts=st.lists(st.integers(min_value=1, max_value=500), min_size=1, max_size=5))
@settings(max_examples=60, deadline=None)
def test_three_way_match_any_partial_split_reconciles(parts):
    """However an order is split into deliveries and matching invoices, R2 stays quiet."""
    total = sum(parts)
    i = invoices(*[inv(invoice_id=k, invoice_number=f"N{k}", quantity=q, total_amount=107.0 * q + k / 100,
                       invoice_date=f"2026-01-{k + 1:02d}") for k, q in enumerate(parts)])
    r = df(*[gr(gr_id=k, received_qty=q) for k, q in enumerate(parts)])
    assert rule_three_way_match(i, df(po(quantity=total)), r).empty


@given(st.text(alphabet="INVinv-/ 0123456789", min_size=1, max_size=20))
def test_normalize_is_idempotent(s):
    once = normalize_invoice_number(pd.Series([s]))
    assert normalize_invoice_number(once).iloc[0] == once.iloc[0]
