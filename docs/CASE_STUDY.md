# Case study: building and evaluating an AP leakage detector

## The problem

Accounts payable leakage is money that leaves a business when it shouldn't: an invoice paid twice, a price above the contract, goods paid for but never received, a payment to a vendor record that someone quietly cloned. The individual amounts are small relative to total spend, but they recur, and most are only found later by an audit or a recovery firm.

The controls that prevent leakage are well known: three-way matching, "no PO, no pay", approval limits, vendor master reviews. The hard part is not writing a rule for each one. It is writing rules precise enough that an AP team will actually work the queue. A detector that flags every partial delivery as a quantity mismatch is ignored within a week.

So the question this project tries to answer is not "can I write nine SQL checks?" It is **"how good are these checks, and how would I know?"**

## Version 1, and why it wasn't good enough

The first version had nine rules, a synthetic dataset with planted exceptions, a dashboard, and recall figures: seven of nine rules at 100%. It looked finished.

Two problems were hiding in it:

1. **It only measured recall.** 3,599 flags were raised for 2,089 planted exceptions, and nothing explained the other ~1,500. Some came from the duplicate rule flagging *both* the original and the copy. Others came from rules colliding: a duplicate invoice also tripped the splitting rule.
2. **The data was too clean to fail on.** Every purchase order had exactly one receipt and one invoice. Contracts never changed. Amounts never had rounding noise. Any rule that fired on an anomaly scored perfectly, including rules that would drown a real team in false positives.

There was also a correctness bug in the main view: duplicate payments and overpayments carried no vendor, so the two most expensive kinds of leakage were missing from the vendor risk ranking.

## Making the evaluation mean something

The generator was rebuilt around one idea: **for every kind of exception, add legitimate activity that looks like it.**

| Exception | Legitimate look-alike |
|---|---|
| Duplicate invoice | Partial deliveries billed in equal instalments; monthly recurring billing |
| Three-way mismatch | Partial deliveries with several receipts |
| Contract price violation | Mid-term contract amendments with a price increase |
| Tax error / overpayment | Cent-level rounding |
| New-vendor payment | New vendors onboarded weeks before their first order |

It also added cases that are genuinely hard rather than just noisy: duplicates re-keyed without their PO reference (catching them needs a looser match that collides with recurring billing), duplicates with a mis-keyed total, and natural receipt variance.

The rules then went up against a **naive baseline**, the version 1 logic, on the same data. That turned "I improved the duplicate rule" into "precision went from 13% to 96% and recall from 57% to 100%". It also showed where nothing improved: the tax and overpayment rules were already fine, and the report says so.

## What changed in the rules

- **Duplicates** flag only the later copy. Invoice numbers are normalised (`inv/0001/000123` matches `INV-0001-000123`), amounts match within a tolerance, and re-keyed duplicates without a PO are caught within a 45-day window. A pair whose combined quantity fits within the order is treated as instalments, not duplicates.
- **Three-way match** reconciles total invoiced quantity per PO instead of each invoice against the PO, which is what makes partial deliveries clean. It separates over-billing (exposure) from under-billing (no exposure).
- **Contract price** uses the contract in force on the invoice date.
- **Invoice splitting** ignores duplicates and billing that follows separate deliveries.
- **New detectors**: duplicate vendor records sharing a bank account or tax ID, large payments to vendors created days earlier, a Benford's Law first-digit test per vendor, and a robust price-outlier score per SKU.

## Rules vs machine learning

An Isolation Forest was trained on generic invoice features and given the same alert budget as the rules. It reached 63% precision against 99.8% for the rules, and it missed most contract and quantity problems because they don't look unusual in isolation: they are only wrong *relative to another document*.

But it caught 94% of the non-contract price inflation, which no rule was written for. That is a useful result in both directions: structured controls should stay rules, and the gap the model found became a targeted analytic (A2) rather than an opaque score.

## From detector to workflow

A flag on its own isn't worth much. Each flag now has:

- a stable identity across runs, so re-running detection preserves review decisions;
- an evidence view showing the contract, PO, receipts, every invoice on the PO and every payment;
- a status workflow with required notes for dismissals and amounts for recoveries;
- an append-only audit trail.

Reviewer decisions produce an observed precision per detector. On real data, that is the metric that would replace the synthetic one.

## What I'd do with real data

1. Run the rules unchanged and look at observed precision per detector in the first two weeks of review.
2. Expect the look-alikes I didn't think of: credit notes, multi-line invoices, currency conversions, vendor mergers.
3. Use dismissals to tune thresholds per vendor segment, and keep the sensitivity sweeps as the audit trail for why each threshold is where it is.

## Limitations

The generator and the rules were written by the same person, so high recall shows the rules do what they claim, not how they'd perform in the wild. Exceptions are injected far more densely than real leakage rates so each detector has enough cases to measure. The demo's review decisions are simulated. These limits are stated in the README and the dashboard rather than left for a reader to discover.
