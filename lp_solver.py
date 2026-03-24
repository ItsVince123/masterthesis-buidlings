"""LP solver — shared greedy optimisation and day simulation.

This module contains:

* :func:`greedy_lp` — the core scheduling algorithm (provably optimal
  for linear cost + box/sum constraints).
* :func:`simulate_day` — asset-driven historical day simulation that
  compares baseline vs LP-optimised cost.
* CSV loading helpers for historical building data and solar archives.

Both the live dashboard (via ``smpc_calculator``) and the historical
analysis dialog import from here, eliminating code duplication.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from energy_assets import (
    EnergyAsset, EXTRA_COST_EUR_MWH, GENERATOR, SHIFTABLE_LOAD,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Core LP algorithm
# ===================================================================

def greedy_lp(
    n_steps: int,
    total_price: np.ndarray,
    total_energy: float,
    max_per_step: float,
) -> np.ndarray:
    """Schedule *total_energy* kWh across the cheapest time-steps.

    Fills intervals from cheapest to most expensive, capping each at
    *max_per_step*.  Provably optimal for a linear objective with
    box + sum constraints (no inter-temporal coupling beyond the total).
    """
    schedule = np.zeros(n_steps)
    remaining = total_energy
    for t in np.argsort(total_price):
        if remaining <= 0:
            break
        amount = min(max_per_step, remaining)
        schedule[t] = amount
        remaining -= amount
    return schedule


# ===================================================================
# CSV loaders
# ===================================================================

def parse_eu_float(text: str) -> float:
    """Parse a European-format number (comma = decimal separator)."""
    text = text.strip()
    if not text:
        return 0.0
    return float(text.replace(",", "."))


def load_historical_csv(path: Path) -> dict[str, list[dict]]:
    """Load the building CSV, grouped by day string → list of row dicts.

    Expected columns (semicolon-separated, European decimals)::

        From Timestamp;TotalUsage;NetUsage;ProductionWKK;
        TotalChiller;RemainingUsage;PricesElec;ExtraCost;TotalPrices

    Returns ``{day_str: [row_dict, …]}`` keyed by ``"D/MM/YYYY"``.
    """
    days: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for raw in reader:
            row = {
                "timestamp":       raw["From Timestamp"].strip(),
                "total_usage":     parse_eu_float(raw["TotalUsage"]),
                "net_usage":       parse_eu_float(raw["NetUsage"]),
                "production_wkk":  parse_eu_float(raw["ProductionWKK"]),
                "total_chiller":   parse_eu_float(raw["TotalChiller"]),
                "remaining_usage": parse_eu_float(raw["RemainingUsage"]),
                "prices_elec":     parse_eu_float(raw["PricesElec"]),
                "extra_cost":      parse_eu_float(raw["ExtraCost"]),
            }
            day_key = row["timestamp"].split(" ")[0]
            days.setdefault(day_key, []).append(row)
    return days


def load_solar_csv(path: Path) -> dict[str, list[float]]:
    """Load ``solar_2022.csv`` into ``{day_key: [kwh_h0, …]}``.

    Day key format matches DATA.csv: ``"D/MM/YYYY"``.
    """
    days: dict[str, list[tuple[int, float]]] = {}
    if not path.exists():
        logger.warning("Solar CSV not found: %s — solar will be 0", path)
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f, delimiter=";"):
            ts = raw["Timestamp"].strip()
            day_key = ts.split(" ")[0]
            time_part = ts.split(" ")[1] if " " in ts else "0:00"
            hour = int(time_part.split(":")[0])
            val = float(raw["Solar_kWh"].strip().replace(",", "."))
            days.setdefault(day_key, []).append((hour, val))

    result: dict[str, list[float]] = {}
    for k, pairs in days.items():
        pairs.sort(key=lambda x: x[0])
        result[k] = [v for _, v in pairs]
    return result


# ===================================================================
# Day key helpers
# ===================================================================

def day_key_candidates(d) -> list[str]:
    """Return possible day-key strings for a ``datetime.date``."""
    return [
        f"{d.day}/{d.month:02d}/{d.year}",
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day}/{d.month}/{d.year}",
    ]


def find_day_rows(
    csv_data: dict[str, list[dict]], selected,
) -> list[dict]:
    """Look up CSV rows for the given date, trying multiple key formats."""
    for key in day_key_candidates(selected):
        if key in csv_data:
            return csv_data[key]
    return []


# ===================================================================
# Historical day simulation (asset-driven)
# ===================================================================

# Column name → parsed row-dict key mapping
_COL_MAP = {
    "TotalChiller":  "total_chiller",
    "ProductionWKK": "production_wkk",
}


def simulate_day(
    day_rows: list[dict],
    assets: list[EnergyAsset],
    solar_hours: dict[str, list[float]],
    day_keys: list[str],
) -> dict:
    """Compare baseline vs LP-optimised cost for one historical day.

    Parameters
    ----------
    day_rows : list[dict]
        Parsed rows from :func:`load_historical_csv` for a single day.
    assets : list[EnergyAsset]
        Enabled energy assets.
    solar_hours : dict
        ``{day_key: [kwh_h0, …]}`` solar production lookup (from CSV).
    day_keys : list[str]
        Candidate day-key strings for solar lookups.

    Returns
    -------
    dict with keys:
        baseline, optimised          — per-hour cost arrays (EUR)
        baseline_load_kwh, optimised_load_kwh — per-hour load arrays
        baseline_grid_kwh, optimised_grid_kwh — per-hour grid draw arrays
        prices_elec                  — per-hour spot price array
        load_shifted                 — total shiftable kWh
        total_generation             — effective generation kWh
        n_slots                      — number of hourly slots
    """
    n = len(day_rows)
    remaining   = np.array([r["remaining_usage"] for r in day_rows])
    net_usage   = np.array([r["net_usage"]       for r in day_rows])
    prices_elec = np.array([r["prices_elec"]     for r in day_rows])

    total_price = prices_elec + EXTRA_COST_EUR_MWH

    # ── Shiftable loads ─────────────────────────────────────────────
    baseline_shiftable = np.zeros(n)
    lp_shiftable = np.zeros(n)
    total_daily_charged = 0.0

    for asset in assets:
        if asset.asset_type != SHIFTABLE_LOAD or not asset.csv_column:
            continue
        col = _COL_MAP.get(asset.csv_column, asset.csv_column)
        actual = np.array([r.get(col, 0.0) for r in day_rows])
        daily_sum = float(np.sum(actual))
        total_daily_charged += daily_sum
        hourly_max = max(asset.hourly_max_kwh, float(np.max(actual)))

        baseline_shiftable += actual
        lp_shiftable += greedy_lp(n, total_price, daily_sum, hourly_max)

    baseline_load = remaining + baseline_shiftable
    baseline_grid = net_usage
    baseline_cost = baseline_grid * total_price / 1000.0

    # ── Generators ──────────────────────────────────────────────────
    total_gen_lp_effective = np.zeros(n)

    for asset in assets:
        if asset.asset_type != GENERATOR:
            continue

        gen_kwh = np.zeros(n)

        # Solar CSV
        if asset.solar_csv:
            for k in day_keys:
                if k in solar_hours:
                    arr = np.array(solar_hours[k][:n])
                    if len(arr) < n:
                        arr = np.pad(arr, (0, n - len(arr)))
                    gen_kwh += arr
                    break

        # CSV column (e.g. WKK)
        if asset.csv_gen_column:
            col = _COL_MAP.get(asset.csv_gen_column, asset.csv_gen_column)
            gen_kwh += np.array([r.get(col, 0.0) for r in day_rows])

        # Decoupling
        if asset.decouple_below_eur_mwh is not None:
            gen_kwh = np.where(
                total_price >= asset.decouple_below_eur_mwh,
                gen_kwh, 0.0,
            )
        total_gen_lp_effective += gen_kwh

    # ── LP result ───────────────────────────────────────────────────
    lp_load = remaining + lp_shiftable
    lp_grid = lp_load - total_gen_lp_effective
    lp_cost = lp_grid * total_price / 1000.0

    return {
        "baseline":           baseline_cost,
        "optimised":          lp_cost,
        "baseline_load_kwh":  baseline_load,
        "optimised_load_kwh": lp_load,
        "baseline_grid_kwh":  baseline_grid,
        "optimised_grid_kwh": lp_grid,
        "prices_elec":        prices_elec,
        "load_shifted":       total_daily_charged,
        "total_generation":   float(np.sum(total_gen_lp_effective)),
        "n_slots":            n,
    }
