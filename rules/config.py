"""Loads detection thresholds from config/rules.yaml, falling back to defaults."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.yaml"

DEFAULTS: dict[str, Any] = {
    "approval_threshold": 5000.00,
    "duplicate_invoice": {"fuzzy": True, "amount_tolerance_abs": 1.00,
                          "amount_tolerance_pct": 0.005, "window_days": 90,
                          "cross_po_window_days": 45, "cross_po_scope": "missing_po_only"},
    "three_way_match": {"qty_tolerance": 0},
    "contract_price": {"price_tolerance_pct": 0.001},
    "tax_validation": {"tolerance_pct": 0.01, "tolerance_abs": 0.02},
    "invoice_splitting": {"window_days": 10},
    "overpayment": {"tolerance_pct": 0.001, "tolerance_abs": 0.05},
    "new_vendor_payment": {"window_days": 30, "min_payment": 10000.00},
    "benford": {"min_invoices": 100, "mad_threshold": 0.015, "p_value": 0.001},
    "price_outlier": {"min_observations": 15, "robust_z_threshold": 3.5, "min_log_mad": 0.05},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=8)
def _read_yaml(path: str, mtime: float) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Return the effective config: defaults <- yaml file <- explicit overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        cfg = _merge(cfg, _read_yaml(str(p), p.stat().st_mtime))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
