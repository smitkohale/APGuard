"""Run every detector against the database and upsert the flags.

Safe to re-run: reviewer decisions on existing flags are preserved.

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apguard.pipeline import run_detection  # noqa: E402


def main() -> None:
    r = run_detection()
    print(f"run {r['run_id']}: {r['flags_raised']:,} flags ({r['flags_new']:,} new, "
          f"{r['flags_retired']:,} no longer raised) in {r['runtime_seconds']}s")
    for code, n in r["by_rule"].items():
        print(f"  {code:4s} {n:>6,}")


if __name__ == "__main__":
    main()
