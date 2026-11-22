"""Metadata for every detector: what it checks, which control it tests, and how
its flags are keyed. The engine, API, dashboard and evaluation all read from here,
so a rule is described in exactly one place."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuleSpec:
    code: str
    name: str
    category: str        # "control" = deterministic rule, "analytic" = statistical signal
    level: str           # entity the flag is about: invoice | payment | vendor
    severity: str
    control: str         # the AP control this detector tests
    assertion: str       # financial statement assertion at risk
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


RULES: dict[str, RuleSpec] = {s.code: s for s in [
    RuleSpec("R1", "Duplicate Invoice", "control", "invoice", "high",
             "Duplicate invoice prevention", "Occurrence",
             "Same vendor billing the same goods twice: identical or reformatted invoice "
             "number, or same PO/SKU/quantity with an amount within tolerance, where total "
             "billing exceeds the ordered quantity."),
    RuleSpec("R2", "Three-Way Match Failure", "control", "invoice", "high",
             "Three-way match (PO / goods receipt / invoice)", "Accuracy",
             "Total invoiced quantity on a PO disagrees with the ordered or received quantity. "
             "Over-billing is high severity; under-billing is low with no exposure."),
    RuleSpec("R3", "Contract Price Violation", "control", "invoice", "medium",
             "Contract price compliance", "Valuation",
             "Unit price above the contract price in force on the invoice date "
             "(contract amendments are respected)."),
    RuleSpec("R4", "Missing Purchase Order", "control", "invoice", "high",
             "No PO, no pay", "Authorization",
             "Invoice with no purchase order, so no upstream approval trail."),
    RuleSpec("R5", "Duplicate Payment", "control", "payment", "critical",
             "Duplicate payment prevention", "Occurrence",
             "Second or later payment against an invoice that was already paid."),
    RuleSpec("R6", "Tax Validation Failure", "control", "invoice", "low",
             "Tax / VAT validation", "Accuracy",
             "Tax charged differs from subtotal x contracted tax rate beyond tolerance."),
    RuleSpec("R7", "Invoice Splitting", "control", "invoice", "medium",
             "Approval limits and segregation of duties", "Authorization",
             "Several invoices on one PO, each under the approval limit, together over it, "
             "filed within a short window and not backed by separate deliveries."),
    RuleSpec("R8", "Overpayment", "control", "payment", "high",
             "Payment accuracy", "Accuracy",
             "Payment amount exceeds the invoice total beyond tolerance."),
    RuleSpec("R9", "Payment Without Goods Receipt", "control", "payment", "high",
             "Receipt before payment", "Existence",
             "Payment made against a PO with no goods receipt on file."),
    RuleSpec("R10", "Duplicate Vendor Record", "control", "vendor", "critical",
             "Vendor master data integrity", "Occurrence",
             "Vendor shares a bank account or tax ID with an earlier vendor record, a common "
             "route for diverting payments."),
    RuleSpec("R11", "New Vendor Large Payment", "control", "payment", "high",
             "New vendor onboarding and due diligence", "Existence",
             "Large payment to a vendor created only days earlier."),
    RuleSpec("A1", "Benford Nonconformity", "analytic", "vendor", "medium",
             "Forensic analytics (Benford's Law)", "Occurrence",
             "Vendor's invoice first-digit distribution departs significantly from Benford's "
             "Law (chi-square and Nigrini MAD), e.g. amounts clustered under the approval limit."),
    RuleSpec("A2", "Unit Price Outlier", "analytic", "invoice", "medium",
             "Forensic analytics (price benchmarking)", "Valuation",
             "Unit price far above what the organisation normally pays for the SKU "
             "(robust z-score). Catches inflated prices on items with no contract."),
]}

RULE_CODES = list(RULES)
CONTROL_RULES = [c for c, s in RULES.items() if s.category == "control"]
ANALYTIC_RULES = [c for c, s in RULES.items() if s.category == "analytic"]
SEVERITIES = ["critical", "high", "medium", "low"]
