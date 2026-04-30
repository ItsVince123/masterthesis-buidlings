"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  THIS IS A CORE THESIS ALGORITHM                                 ║
║  Deterministic greedy LP scheduler used for historical           ║
║  analysis and as the baseline reference in the SMPC module.      ║
╚══════════════════════════════════════════════════════════════════╝

LP solver — shared greedy optimisation and day simulation.

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
    GAS_HEATER, HEAT_PUMP,
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

    This is the core LP scheduling algorithm.

    MATHEMATICAL BASIS
    ------------------
    We want to minimise:
        sum_t ( price[t] * x[t] )
    subject to:
        sum_t x[t] = total_energy        (must consume exactly the daily target)
        0 <= x[t] <= max_per_step        (box constraints per interval)

    For a LINEAR objective with only box + sum constraints (no inter-temporal
    coupling like storage dynamics), the greedy solution IS the global optimum:
    sort intervals by price and fill from cheapest to most expensive.

    PROOF SKETCH: Suppose an optimal solution x* doesn't fill the cheapest
    intervals first.  Then there exist intervals i, j where price[i] < price[j]
    but x*[i] < x*[j].  Shifting ε units from j to i (while keeping the sum
    fixed) strictly lowers cost — contradiction.

    Parameters
    ----------
    n_steps      : number of scheduling intervals (e.g. 96 for 24 h at 15 min)
    total_price  : array of length n_steps — total cost per kWh at each step
                   (should include spot price + distribution fee)
    total_energy : total kWh to schedule over the horizon
    max_per_step : maximum kWh allowed in any single interval (power limit)

    Returns
    -------
    schedule : np.ndarray of length n_steps with the optimal kWh per interval
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


def greedy_lp_ramped(
    n_steps: int,
    total_price: np.ndarray,
    total_energy: float,
    max_per_step: float,
    ramp_up_pct: float = 100.0,
    ramp_down_pct: float = 100.0,
) -> np.ndarray:
    """Schedule with ramp-rate constraints (post-processing smoothing).

    First runs the unconstrained greedy LP, then iteratively smooths
    the schedule so that consecutive steps respect ramp limits.

    Parameters
    ----------
    ramp_up_pct   : max increase between steps, as % of max_per_step
    ramp_down_pct : max decrease between steps, as % of max_per_step
    """
    if ramp_up_pct >= 100.0 and ramp_down_pct >= 100.0:
        return greedy_lp(n_steps, total_price, total_energy, max_per_step)

    schedule = greedy_lp(n_steps, total_price, total_energy, max_per_step)

    ramp_up = max_per_step * ramp_up_pct / 100.0
    ramp_down = max_per_step * ramp_down_pct / 100.0

    # Forward pass: enforce ramp-up limit
    for t in range(1, n_steps):
        if schedule[t] - schedule[t - 1] > ramp_up:
            schedule[t] = schedule[t - 1] + ramp_up

    # Backward pass: enforce ramp-down limit
    for t in range(1, n_steps):
        if schedule[t - 1] - schedule[t] > ramp_down:
            schedule[t] = schedule[t - 1] - ramp_down

    # Clip to box constraints
    np.clip(schedule, 0, max_per_step, out=schedule)

    # Re-scale to preserve total energy (adjust proportionally)
    current_total = schedule.sum()
    if current_total > 0:
        scale = total_energy / current_total
        schedule *= scale
        np.clip(schedule, 0, max_per_step, out=schedule)

    return schedule


# ===================================================================
# Building thermal simulation
# ===================================================================

def simulate_building_thermal(
    n_steps: int,
    dt_hours: float,
    initial_temp_c: float,
    setpoint_c: float,
    deadband_c: float,
    cooldown_rate_c_per_hour: float,
    thermal_mass_kwh_per_c: float,
    assets: list[EnergyAsset],
    prices: np.ndarray | None = None,
    beo_initial_temp_c: float = 12.0,
    chp_heat_kw: np.ndarray | None = None,
) -> dict:
    """Simulate building temperature over a horizon with heating assets.

    Simple rule-based thermal model:
    - Each step the building loses heat at ``cooldown_rate_c_per_hour * dt``
    - CHP exhaust heat (if provided) is applied first as free heating
    - When temperature falls below ``setpoint_c - deadband_c``, heating
      is activated using available gas heaters and heat pumps
    - Heat pumps with ``is_ground_source=True`` extract from BEO-veld,
      which tracks its own temperature

    Parameters
    ----------
    n_steps : number of time steps
    dt_hours : duration of each step in hours (1.0 for hourly, 0.25 for 15-min)
    initial_temp_c : starting building temperature
    setpoint_c : target temperature
    deadband_c : allowed deviation below setpoint before heating activates
    cooldown_rate_c_per_hour : natural cooling rate [°C/h]
    thermal_mass_kwh_per_c : thermal inertia — kWh needed to raise building 1°C
    assets : list of EnergyAsset (only GAS_HEATER and HEAT_PUMP are used)
    prices : optional price array for cost tracking
    beo_initial_temp_c : starting BEO-veld ground temperature
    chp_heat_kw : optional array of CHP exhaust thermal power per step [kW]

    Returns
    -------
    dict with:
        temp_profile   — array of building temperature at each step [°C]
        heating_kw     — array of total heating power at each step [kW]
        beo_temp       — array of BEO-veld temperature at each step [°C]
        heating_cost   — array of heating energy cost at each step [EUR]
        gas_used_m3    — total gas consumed [m³]
        elec_used_kwh  — total electricity consumed by heat pumps [kWh]
    """
    temp = np.zeros(n_steps)
    heating_kw = np.zeros(n_steps)
    beo_temp = np.zeros(n_steps)
    heating_cost = np.zeros(n_steps)
    gas_total = 0.0
    elec_total = 0.0

    current_temp = initial_temp_c
    current_beo = beo_initial_temp_c

    # Gather heating assets
    gas_heaters = [a for a in assets if a.asset_type == GAS_HEATER and a.enabled]
    heat_pumps = [a for a in assets if a.asset_type == HEAT_PUMP and a.enabled]

    # BEO-veld capacity from the first ground-source HP (simplified)
    beo_cap = 50000.0
    for hp in heat_pumps:
        if hp.is_ground_source and hp.beo_capacity_kwh > 0:
            beo_cap = hp.beo_capacity_kwh
            break

    for t in range(n_steps):
        # Natural cooling
        current_temp -= cooldown_rate_c_per_hour * dt_hours

        # CHP exhaust heat (free heating from cogeneration)
        chp_kwh = 0.0
        if chp_heat_kw is not None and t < len(chp_heat_kw):
            chp_kwh = float(chp_heat_kw[t]) * dt_hours
            if thermal_mass_kwh_per_c > 0:
                current_temp += chp_kwh / thermal_mass_kwh_per_c

        # Check if heating is needed
        heat_needed_c = 0.0
        if current_temp < setpoint_c - deadband_c:
            heat_needed_c = setpoint_c - current_temp
        elif current_temp < setpoint_c:
            # Gentle heating to reach setpoint
            heat_needed_c = (setpoint_c - current_temp) * 0.5

        heat_delivered_kwh = 0.0
        step_heating_kw = 0.0
        step_gas_m3 = 0.0
        step_elec_kwh = 0.0

        if heat_needed_c > 0:
            heat_needed_kwh = heat_needed_c * thermal_mass_kwh_per_c

            # 1. Gas heaters
            for gh in gas_heaters:
                if heat_delivered_kwh >= heat_needed_kwh:
                    break
                avail = gh.thermal_output_kw * dt_hours  # kWh this step
                deliver = min(avail, heat_needed_kwh - heat_delivered_kwh)
                heat_delivered_kwh += deliver
                step_heating_kw += deliver / dt_hours
                gas_m3 = (deliver / gh.gas_efficiency) / 9.8 if gh.gas_efficiency > 0 else 0
                step_gas_m3 += gas_m3

            # 2. Heat pumps
            for hp in heat_pumps:
                if heat_delivered_kwh >= heat_needed_kwh:
                    break
                avail = hp.heating_capacity_kw * dt_hours  # kWh this step
                deliver = min(avail, heat_needed_kwh - heat_delivered_kwh)
                heat_delivered_kwh += deliver
                step_heating_kw += deliver / dt_hours
                elec_kwh = deliver / hp.cop if hp.cop > 0 else deliver
                step_elec_kwh += elec_kwh

                # BEO-veld interaction for ground-source
                if hp.is_ground_source:
                    extracted = deliver - elec_kwh  # heat from ground
                    if beo_cap > 0:
                        current_beo -= extracted / (beo_cap / 50.0)  # simplified

        # Apply heating to building temperature
        if thermal_mass_kwh_per_c > 0:
            current_temp += heat_delivered_kwh / thermal_mass_kwh_per_c

        temp[t] = current_temp
        heating_kw[t] = step_heating_kw + (chp_kwh / dt_hours if dt_hours > 0 else 0.0)
        beo_temp[t] = current_beo
        gas_total += step_gas_m3
        elec_total += step_elec_kwh

        # Cost: gas + electricity
        gas_cost = step_gas_m3 * 0.35  # default gas price
        elec_price = prices[t] / 1000.0 if prices is not None and t < len(prices) else 0.05
        elec_cost = step_elec_kwh * elec_price
        heating_cost[t] = gas_cost + elec_cost

    return {
        "temp_profile": temp,
        "heating_kw": heating_kw,
        "beo_temp": beo_temp,
        "heating_cost": heating_cost,
        "gas_used_m3": gas_total,
        "elec_used_kwh": elec_total,
    }


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
    # Extract hourly arrays from the historical CSV rows
    remaining   = np.array([r["remaining_usage"] for r in day_rows])  # non-shiftable base load [kWh]
    net_usage   = np.array([r["net_usage"]       for r in day_rows])  # actual grid draw (baseline reference)
    prices_elec = np.array([r["prices_elec"]     for r in day_rows])  # ENTSO-E spot price [EUR/MWh]

    # Total cost per kWh = spot price + fixed grid/distribution fee
    # EXTRA_COST_EUR_MWH = 50 EUR/MWh covering network charges, taxes, etc.
    total_price = prices_elec + EXTRA_COST_EUR_MWH

    # ── Shiftable loads ─────────────────────────────────────────────
    # For each shiftable load (e.g. ice bank chiller):
    #   - baseline: keep the actual historical schedule
    #   - optimised: run greedy_lp to reschedule to cheapest hours
    baseline_shiftable = np.zeros(n)
    lp_shiftable = np.zeros(n)
    total_daily_charged = 0.0

    for asset in assets:
        if asset.asset_type != SHIFTABLE_LOAD or not asset.csv_column:
            continue
        col = _COL_MAP.get(asset.csv_column, asset.csv_column)
        actual = np.array([r.get(col, 0.0) for r in day_rows])  # historical kWh per hour
        daily_sum = float(np.sum(actual))                        # total kWh consumed that day
        total_daily_charged += daily_sum
        # Allow the LP to charge up to the observed peak (or the configured limit)
        hourly_max = max(asset.hourly_max_kwh, float(np.max(actual)))

        baseline_shiftable += actual
        # The LP keeps the same TOTAL daily energy but redistributes it
        # to hours with the lowest total_price → cost saving via time-shifting
        lp_shiftable += greedy_lp_ramped(
            n, total_price, daily_sum, hourly_max,
            ramp_up_pct=asset.ramp_up_pct_per_hour,
            ramp_down_pct=asset.ramp_down_pct_per_hour,
        )

    # Baseline grid draw = non-shiftable base + actual shiftable schedule
    baseline_load = remaining + baseline_shiftable
    baseline_grid = net_usage                                    # use recorded net usage as-is
    baseline_cost = baseline_grid * total_price / 1000.0        # EUR (price is per MWh → /1000)

    # ── Generators ──────────────────────────────────────────────────
    # Aggregate on-site generation for the LP scenario (solar + CHP/WKK).
    # Generator output reduces the net grid draw, lowering cost.
    total_gen_lp_effective = np.zeros(n)

    for asset in assets:
        if asset.asset_type != GENERATOR:
            continue

        gen_kwh = np.zeros(n)

        # Solar: look up from the solar archive CSV (pre-computed)
        if asset.solar_csv:
            for k in day_keys:
                if k in solar_hours:
                    arr = np.array(solar_hours[k][:n])
                    if len(arr) < n:
                        arr = np.pad(arr, (0, n - len(arr)))
                    gen_kwh += arr
                    break

        # CHP / WKK: read historical production from building CSV
        if asset.csv_gen_column:
            col = _COL_MAP.get(asset.csv_gen_column, asset.csv_gen_column)
            gen_kwh += np.array([r.get(col, 0.0) for r in day_rows])

        # Price-based decoupling: some generators (e.g. solar feed-in) can be
        # disabled when the spot price drops below a configured threshold.
        # This models arbitrage decisions (e.g. don't export to grid at negative prices).
        if asset.decouple_below_eur_mwh is not None:
            gen_kwh = np.where(
                total_price >= asset.decouple_below_eur_mwh,
                gen_kwh, 0.0,
            )
        total_gen_lp_effective += gen_kwh

    # ── Startup costs for generators ────────────────────────────────
    # Count on/off transitions and multiply by configured startup cost.
    total_startup_cost = 0.0
    for asset in assets:
        if asset.asset_type != GENERATOR or asset.startup_cost_eur <= 0:
            continue
        gen_kwh = np.zeros(n)
        if asset.csv_gen_column:
            col = _COL_MAP.get(asset.csv_gen_column, asset.csv_gen_column)
            gen_kwh = np.array([r.get(col, 0.0) for r in day_rows])
        on = gen_kwh > 0
        starts = int(np.sum(on[1:] & ~on[:-1]))
        if on[0]:
            starts += 1
        total_startup_cost += starts * asset.startup_cost_eur

    # ── CHP exhaust heat for building thermal model ───────────────
    # CHP heat output = gas_burned [m³] × calorific_value [kWh/m³] × heat_eff
    # For historical sim, read CHP production from CSV and estimate gas burned.
    chp_heat_kw = np.zeros(n)
    for asset in assets:
        if asset.asset_type != GENERATOR or not asset.csv_gen_column:
            continue
        col = _COL_MAP.get(asset.csv_gen_column, asset.csv_gen_column)
        chp_elec = np.array([r.get(col, 0.0) for r in day_rows])  # kWh elec per hour
        # Estimate heat from CHP: heat_eff / elec_eff × elec output
        # Default: 0.45 / 0.40 = 1.125 kWh heat per kWh electricity
        chp_heat_kw += chp_elec * (0.45 / 0.40)  # kW (hourly steps → kWh=kW)

    # ── Building thermal simulation ─────────────────────────────────
    thermal_result = simulate_building_thermal(
        n_steps=n,
        dt_hours=1.0,  # hourly steps for historical
        initial_temp_c=21.0,  # default; overridden by config in dialog
        setpoint_c=21.0,
        deadband_c=1.0,
        cooldown_rate_c_per_hour=0.5,
        thermal_mass_kwh_per_c=500.0,
        assets=assets,
        prices=total_price,
        chp_heat_kw=chp_heat_kw,
    )

    # ── LP result ───────────────────────────────────────────────────
    # Net grid draw under the LP schedule = base load + shifted demand - generation
    lp_load = remaining + lp_shiftable
    lp_grid = lp_load - total_gen_lp_effective
    lp_cost = lp_grid * total_price / 1000.0   # EUR
    # Add startup costs to optimised total
    if n > 0 and total_startup_cost > 0:
        lp_cost[0] += total_startup_cost
    # Add heating costs
    lp_cost += thermal_result["heating_cost"]

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
        "startup_cost":       total_startup_cost,
        "temp_profile":       thermal_result["temp_profile"],
        "heating_kw":         thermal_result["heating_kw"],
        "beo_temp":           thermal_result["beo_temp"],
        "n_slots":            n,
    }
