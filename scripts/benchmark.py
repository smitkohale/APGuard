"""Benchmark the engine at increasing scale by tiling the generated dataset.

    python scripts/benchmark.py [--scales 1 5 20] [--legacy]

--legacy also times the original row-by-row engine, loaded straight from the
first commit in git history, on the same data.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apguard.evaluation import load_dataset  # noqa: E402
from rules.engine import run_all_rules  # noqa: E402

ID_COLS = ["vendor_id", "contract_id", "po_id", "gr_id", "invoice_id", "payment_id"]
LEGACY_COMMIT = "9e62eab"


def tile(tables: tuple, k: int) -> tuple:
    out = []
    for t in tables:
        parts = []
        for i in range(k):
            c = t.copy()
            for col in ID_COLS:
                if col in c.columns:
                    c[col] = c[col] + i * 10_000_000
            for col in ("bank_account", "tax_id", "invoice_number", "payment_ref"):
                if col in c.columns:
                    c[col] = c[col].astype(str) + f"-{i}"
            parts.append(c)
        out.append(pd.concat(parts, ignore_index=True))
    return tuple(out)


def load_legacy():
    try:
        src = subprocess.run(["git", "show", f"{LEGACY_COMMIT}:rules/engine.py"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    path = Path(tempfile.mkdtemp()) / "legacy_engine.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("legacy_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed(fn, *args, repeat=1) -> float:
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--legacy", action="store_true")
    args = ap.parse_args()
    ds = load_dataset(ROOT / "data" / "generated")
    legacy = load_legacy() if args.legacy else None
    warnings.filterwarnings("ignore")  # the legacy engine emits pandas deprecation warnings

    print(f"{'scale':>5} {'invoices':>10} {'payments':>10} {'engine (s)':>11} {'legacy (s)':>11}")
    for k in args.scales:
        t = tile(ds.tables, k)
        cur = timed(run_all_rules, *t, repeat=2 if k <= 5 else 1)
        leg = ""
        if legacy is not None and k <= 5:
            leg = f"{timed(legacy.run_all_rules, *t):>11.2f}"
        print(f"{k:>5} {len(t[4]):>10,} {len(t[5]):>10,} {cur:>11.2f} {leg:>11}")


if __name__ == "__main__":
    main()
