"""One command to (re)build a database: generate data, apply the schema, load,
run detection and simulate a round of reviews.

    python scripts/setup_database.py --yes

DESTRUCTIVE: drops and recreates every APGuard table in DATABASE_URL, including
review history. Used by the "Set up database" GitHub Actions workflow, so a hosted
database can be refreshed without a local copy of the project.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "db"))

from apguard.db import get_engine  # noqa: E402


def step(n: int, total: int, text: str) -> None:
    print(f"\n== {n}/{total} {text}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="confirm that every APGuard table may be dropped")
    ap.add_argument("--no-reviews", action="store_true", help="skip the simulated review round")
    args = ap.parse_args()
    if not args.yes:
        sys.exit("Refusing to run without --yes: this drops and recreates every APGuard table.")

    total = 4 if args.no_reviews else 5
    engine = get_engine()
    host = engine.url.host or "local socket"
    print(f"Target database: {engine.url.database} on {host}")

    step(1, total, "Generating synthetic data")
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_data.py")], check=True)

    step(2, total, "Applying schema")
    with engine.begin() as conn:
        conn.exec_driver_sql((ROOT / "db" / "schema.sql").read_text())
    print("schema applied")

    step(3, total, "Loading data")
    import load_data
    sys.argv = ["load_data.py"]
    load_data.main()

    step(4, total, "Running detection")
    from apguard.pipeline import run_detection
    r = run_detection(engine)
    print(f"run {r['run_id']}: {r['flags_raised']:,} flags in {r['runtime_seconds']}s")

    if not args.no_reviews:
        step(5, total, "Simulating a review round (demo only)")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "simulate_reviews.py")], check=True)

    print("\nDatabase ready.")


if __name__ == "__main__":
    main()
