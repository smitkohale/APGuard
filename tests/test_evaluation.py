"""Scoring logic, plus regression guards on the generated dataset.

The dataset tests are skipped when data/generated is absent; CI generates it first.
They pin the evaluation so a change that quietly degrades precision or recall fails.
"""
from pathlib import Path

import pandas as pd
import pytest

from apguard.evaluation import load_dataset, score_all, score_rule
from rules.baseline import BASELINE_RULES, run_baseline
from rules.engine import run_all_rules

DATA = Path(__file__).resolve().parent.parent / "data" / "generated"


def test_score_rule_counts_entities_once():
    flags = pd.DataFrame({"rule_code": ["R4", "R4", "R4"], "invoice_id": [1, 1, 2]})
    truth = pd.DataFrame({"exception_type": ["missing_po", "missing_po"], "invoice_id": [1, 3],
                          "payment_id": [None, None], "vendor_id": [1, 1]})
    s = score_rule("R4", flags, truth)
    assert (s["flagged"], s["true_positives"], s["false_positives"], s["missed"]) == (2, 1, 1, 1)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


@pytest.fixture(scope="module")
def dataset():
    if not (DATA / "invoices.csv").exists():
        pytest.skip("generated data not present: run python data/generate_data.py")
    return load_dataset(DATA)


@pytest.fixture(scope="module")
def scores(dataset):
    return score_all(run_all_rules(*dataset.tables), dataset.truth).set_index("rule")


@pytest.mark.parametrize("rule", ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "A1"])
def test_control_recall_floor(scores, rule):
    assert scores.loc[rule, "recall"] >= 0.98


@pytest.mark.parametrize("rule", ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"])
def test_control_precision_floor(scores, rule):
    assert scores.loc[rule, "precision"] >= 0.95


def test_analytic_precision_floor(scores):
    assert scores.loc["A2", "precision"] >= 0.75


def test_apguard_beats_naive_baseline(dataset, scores):
    base = score_all(run_baseline(*dataset.tables), dataset.truth, BASELINE_RULES).set_index("rule")
    for rule in BASELINE_RULES:
        assert scores.loc[rule, "f1"] >= base.loc[rule, "f1"]
    assert scores.loc["R1", "precision"] - base.loc["R1", "precision"] > 0.5
