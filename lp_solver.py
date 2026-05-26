"""
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
from dashboard_config import load_dashboard_config as _load_cfg

logger = logging.getLogger(__name__)


# ===================================================================
# Thermal config cache — read from dashboard_config.json once,
# always reflects the latest GUI-saved settings.
# ===================================================================

def _read_thermal_cfg() -> dict:
    """Return the smpc.building.thermal block from dashboard_config.json."""
    try:
        return _load_cfg().get("smpc", {}).get("building", {}).get("thermal", {})
    except Exception:
        return {}

_THERMAL_CFG: dict = _read_thermal_cfg()


def reload_thermal_config() -> None:
    """Re-read thermal settings from disk (call after GUI saves)."""
    global _THERMAL_CFG
    _THERMAL_CFG = _read_thermal_cfg()


def _read_hw_cfg() -> dict:
    """Return hot water tank settings — from a HOT_WATER_HEATER asset if one
    exists, otherwise fall back to smpc.building.hot_water_tank in config."""
    try:
        from energy_assets import load_assets, HOT_WATER_HEATER
        for a in load_assets():
            if a.asset_type == HOT_WATER_HEATER and a.enabled:
                return {
                    "enabled":          True,
                    "tank_volume_l":    a.hw_tank_volume_l,
                    "min_temp_c":       a.hw_min_temp_c,
                    "max_temp_c":       a.hw_max_temp_c,
                    "initial_temp_c":   a.hw_initial_temp_c,
                    "heat_loss_w":      a.hw_heat_loss_w,
                    "heater_power_kw":  a.hw_heater_power_kw,
                }
    except Exception:
        pass
    try:
        return _load_cfg().get("smpc", {}).get("building", {}).get("hot_water_tank", {})
    except Exception:
        return {}


_HW_CFG: dict = _read_hw_cfg()


def reload_hw_config() -> None:
    """Re-read hot water tank settings from disk (call after GUI saves)."""
    global _HW_CFG
    _HW_CFG = _read_hw_cfg()


# ===================================================================
# Utility
# ===================================================================

def _in_window(h: int, start_h: int, end_h: int) -> bool:
    """Return True if hour *h* is inside [start_h, end_h) with midnight wrap.

    Handles overnight windows (start_h > end_h), e.g. 21–07:
      _in_window(23, 21, 7) → True   _in_window(3, 21, 7) → True
      _in_window(10, 21, 7) → False
    A degenerate window (start_h == end_h) is treated as empty → False.
    """
    if start_h == end_h:
        return False
    if start_h < end_h:
        return start_h <= h < end_h   # normal window
    # Overnight: e.g. 21 → 7  means h >= 21 OR h < 7
    return h >= start_h or h < end_h


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
        if not np.isfinite(total_price[t]):
            break  # remaining slots all have inf price (outside flex window)
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


def _effective_prices(
    total_price: np.ndarray,
    gen_surplus: np.ndarray,
    max_per_step: float,
) -> np.ndarray:
    """Return prices reduced where free solar/CHP generation creates a surplus.

    For each slot, the fraction ``min(surplus, max_per_step) / max_per_step``
    of any scheduled load is covered by zero-marginal-cost generation.
    The effective marginal grid cost is scaled down accordingly, making
    surplus slots appear cheaper to the greedy LP scheduler.

    Parameters
    ----------
    total_price  : grid price array [any consistent units, e.g. EUR/MWh]
    gen_surplus  : free generation above fixed base load per slot [kWh]
    max_per_step : maximum shiftable load per slot for this asset [kWh]
    """
    if max_per_step <= 0:
        return total_price
    frac_free = np.minimum(gen_surplus, max_per_step) / max_per_step
    return total_price * np.maximum(0.0, 1.0 - frac_free)


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
    gas_price_eur_m3: float = 0.35,
    price_aware: bool = False,
    outside_temp_c: np.ndarray | None = None,
    ua_kwh_per_c_per_h: float = 2.5,
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
    price_aware : if True, use price-aware deadband pre-heating (heat when cheap,
        defer when expensive within the deadband flex zone).  If False (default),
        always heat whenever temperature falls below setpoint (baseline rule).

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
    hp_elec_kwh_per_step = np.zeros(n_steps)
    gas_heater_cost_per_step = np.zeros(n_steps)
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

    # ── Physical parameters for price-aware look-ahead ──────────────
    # drift_per_step is the fixed-rate fallback used to pre-compute
    # steps_to_recover (a conservative worst-case estimate).
    # Per-step drift is computed dynamically in the loop using Newton's law
    # when outside_temp_c is provided.
    drift_per_step = cooldown_rate_c_per_hour * dt_hours  # °C lost per step (no heating)

    # Total installed heating capacity across all enabled assets [kW]
    total_heating_cap_kw = (
        sum(a.heating_capacity_kw for a in heat_pumps)
        + sum(a.thermal_output_kw for a in gas_heaters)
    )
    # Temperature rise per step at full heating capacity [°C/step]
    heat_rate_per_step = (
        total_heating_cap_kw * dt_hours / thermal_mass_kwh_per_c
        if thermal_mass_kwh_per_c > 0 and total_heating_cap_kw > 0
        else float("inf")
    )
    # Net rise per step = heating rate minus concurrent drift
    net_heat_rate = max(heat_rate_per_step - drift_per_step, 1e-6)
    # Steps needed to recover from lower_limit back to setpoint.
    # Subtract 1 because heating fires within the triggered step itself,
    # so a full-step head-start is not required — this ensures at least 1
    # slot of defer flexibility even with high-capacity / coarse time resolution.
    steps_to_recover = max(0, int(np.ceil(deadband_c / net_heat_rate)) - 1)

    for t in range(n_steps):
        # Natural cooling — Newton's law when outside temp is available,
        # else fall back to fixed rate.
        if (outside_temp_c is not None and t < len(outside_temp_c)
                and thermal_mass_kwh_per_c > 0):
            _delta_out = current_temp - float(outside_temp_c[t])
            step_drift = ua_kwh_per_c_per_h * _delta_out * dt_hours / thermal_mass_kwh_per_c
        else:
            step_drift = cooldown_rate_c_per_hour * dt_hours
        current_temp -= step_drift

        # CHP exhaust heat (free heating from cogeneration)
        chp_kwh = 0.0
        if chp_heat_kw is not None and t < len(chp_heat_kw):
            chp_kwh = float(chp_heat_kw[t]) * dt_hours
            if thermal_mass_kwh_per_c > 0:
                current_temp += chp_kwh / thermal_mass_kwh_per_c

        # Determine how much heating is needed this step
        heat_needed_c = 0.0
        if price_aware:
            # Dynamic look-ahead using physical thermal parameters:
            #
            #  lower_limit = setpoint - deadband  (comfort floor, e.g. 20 °C)
            #  steps_until_forced = how long temp can drift before hitting floor
            #  steps_to_recover   = steps needed to heat from floor → setpoint
            #                       at full capacity (accounts for thermal mass)
            #  defer_window       = steps_until_forced - steps_to_recover
            #                       → the slots we can still choose to defer into
            #
            # Heat NOW only if current slot is the cheapest in defer_window.
            # If defer_window ≤ 0 the recovery deadline has arrived — heat now.
            lower_limit = setpoint_c - deadband_c
            if current_temp < lower_limit:
                # Past comfort floor — forced heat regardless of price
                heat_needed_c = setpoint_c - current_temp
            elif current_temp < setpoint_c and prices is not None:
                slack = current_temp - lower_limit          # °C above floor
                steps_until_forced = (
                    int(slack / step_drift) if step_drift > 0 else n_steps
                )
                defer_window = max(0, steps_until_forced - steps_to_recover)
                if defer_window == 0:
                    # Must start heating now to make it back to setpoint in time
                    heat_needed_c = setpoint_c - current_temp
                else:
                    # Look ahead over the deferrable window and heat only if
                    # the current price is the cheapest available slot
                    lookahead_end = min(t + defer_window + 1, n_steps)
                    if float(prices[t]) <= float(np.min(prices[t:lookahead_end])):
                        heat_needed_c = setpoint_c - current_temp
                    # else: cheaper slot is coming — defer
        else:
            # Baseline (constant thermostat): always heat to setpoint
            if current_temp < setpoint_c:
                heat_needed_c = setpoint_c - current_temp

        heat_delivered_kwh = 0.0
        step_heating_kw = 0.0
        step_gas_m3 = 0.0
        step_elec_kwh = 0.0

        if heat_needed_c > 0:
            heat_needed_kwh = heat_needed_c * thermal_mass_kwh_per_c

            # Cost-aware dispatch: sort all heating sources cheapest-first
            # per unit of thermal energy delivered this step.
            elec_price_kwh = (
                prices[t] / 1000.0
                if prices is not None and t < len(prices)
                else 0.15
            )

            candidates: list[tuple[float, str, object]] = []
            for gh in gas_heaters:
                gp  = getattr(gh, "gas_price_eur_m3",  0.0) or gas_price_eur_m3
                hv  = getattr(gh, "gas_energy_kwh_m3", 9.8) or 9.8
                eff = gh.gas_efficiency if gh.gas_efficiency > 0 else 1.0
                candidates.append((gp / (hv * eff), "gas", gh))
            for hp in heat_pumps:
                cop = hp.cop if hp.cop > 0 else 1.0
                candidates.append((elec_price_kwh / cop, "hp", hp))
            candidates.sort(key=lambda x: x[0])  # cheapest first

            for _cost, kind, a in candidates:
                if heat_delivered_kwh >= heat_needed_kwh:
                    break
                if kind == "gas":
                    avail = a.thermal_output_kw * dt_hours
                    deliver = min(avail, heat_needed_kwh - heat_delivered_kwh)
                    heat_delivered_kwh += deliver
                    step_heating_kw += deliver / dt_hours
                    hv = getattr(a, "gas_energy_kwh_m3", 9.8) or 9.8
                    gas_m3 = (deliver / a.gas_efficiency) / hv if a.gas_efficiency > 0 else 0.0
                    step_gas_m3 += gas_m3
                else:  # heat pump
                    avail = a.heating_capacity_kw * dt_hours
                    deliver = min(avail, heat_needed_kwh - heat_delivered_kwh)
                    heat_delivered_kwh += deliver
                    step_heating_kw += deliver / dt_hours
                    elec_kwh = deliver / a.cop if a.cop > 0 else deliver
                    step_elec_kwh += elec_kwh
                    if a.is_ground_source:
                        extracted = deliver - elec_kwh
                        if beo_cap > 0:
                            current_beo -= extracted / (beo_cap / 50.0)

        # Apply heating to building temperature
        if thermal_mass_kwh_per_c > 0:
            current_temp += heat_delivered_kwh / thermal_mass_kwh_per_c

        temp[t] = current_temp
        heating_kw[t] = step_heating_kw + (chp_kwh / dt_hours if dt_hours > 0 else 0.0)
        beo_temp[t] = current_beo
        gas_total += step_gas_m3
        elec_total += step_elec_kwh

        # Cost: gas + electricity
        gas_cost = step_gas_m3 * gas_price_eur_m3
        elec_price = prices[t] / 1000.0 if prices is not None and t < len(prices) else 0.05
        elec_cost = step_elec_kwh * elec_price
        heating_cost[t] = gas_cost + elec_cost
        gas_heater_cost_per_step[t] = gas_cost
        hp_elec_kwh_per_step[t] = step_elec_kwh

    return {
        "temp_profile": temp,
        "heating_kw": heating_kw,
        "beo_temp": beo_temp,
        "heating_cost": heating_cost,
        "gas_heater_cost": gas_heater_cost_per_step,
        "hp_elec_kwh_per_step": hp_elec_kwh_per_step,
        "gas_used_m3": gas_total,
        "elec_used_kwh": elec_total,
    }


# ===================================================================
# Hot water tank simulation (electric, COP = 1)
# ===================================================================

def simulate_hot_water_tank(
    n_steps: int,
    dt_hours: float,
    initial_temp_c: float,
    min_temp_c: float,
    max_temp_c: float,
    heat_loss_w: float,
    heater_power_kw: float,
    tank_volume_l: float,
    prices: np.ndarray | None = None,
    price_aware: bool = False,
) -> dict:
    """Simulate a domestic hot water tank with price-aware electric heating.

    The heating element has COP = 1 (pure electric resistance).

    Physics
    -------
    - Thermal mass: C_water ≈ 1.163 Wh/(L·°C)  →  mass = volume × C_water [kWh/°C]
    - Heat loss per step: heat_loss_w / 1000 × dt_hours [kWh]
    - Max heat per step: heater_power_kw × dt_hours [kWh]
    - Temperature boundary: [min_temp_c, max_temp_c]

    Price-aware logic mirrors the building thermal model:
    - Baseline: heat whenever temperature is below max_temp_c (naive thermostat)
    - LP: pre-heat during cheapest slots; defer when a cheaper slot is
      available within the remaining thermal slack window.

    Parameters
    ----------
    n_steps         : number of time steps
    dt_hours        : step duration [h]
    initial_temp_c  : starting water temperature [°C]
    min_temp_c      : minimum allowed temperature — safety / comfort floor [°C]
    max_temp_c      : target / maximum temperature [°C]
    heat_loss_w     : constant standby heat loss [W]
    heater_power_kw : rated heater power [kW]  (COP = 1 → elec = thermal)
    tank_volume_l   : tank size [L]
    prices          : total electricity price per slot [EUR/MWh]
    price_aware     : if True use LP look-ahead; else naive thermostat

    Returns
    -------
    dict with:
        temp_profile    — water temperature at each step [°C]
        elec_kwh        — electricity consumed per step [kWh]
        heating_cost    — electricity cost per step [EUR]
        total_elec_kwh  — total electricity consumed over horizon [kWh]
    """
    C_WATER_KWH_PER_L_PER_C = 1.163e-3           # kWh / (L · °C)
    thermal_mass = max(tank_volume_l * C_WATER_KWH_PER_L_PER_C, 1e-9)  # kWh/°C

    heat_loss_kwh_per_step = heat_loss_w / 1000.0 * dt_hours   # kWh lost per step
    heat_loss_c_per_step   = heat_loss_kwh_per_step / thermal_mass  # °C drop per step

    max_heat_kwh = heater_power_kw * dt_hours           # max thermal kWh per step
    max_heat_c   = max_heat_kwh / thermal_mass          # max °C rise per step

    deadband_c    = max_temp_c - min_temp_c
    net_heat_rate = max(max_heat_c - heat_loss_c_per_step, 1e-6)  # °C net rise/step at full power
    steps_to_recover = max(0, int(np.ceil(deadband_c / net_heat_rate)) - 1)

    temp         = np.zeros(n_steps)
    elec_kwh     = np.zeros(n_steps)
    heating_cost = np.zeros(n_steps)

    current_temp = float(initial_temp_c)

    for t in range(n_steps):
        # Standby heat loss
        current_temp -= heat_loss_c_per_step

        # Determine heating need
        heat_needed_c = 0.0
        if price_aware and prices is not None:
            if current_temp < min_temp_c:
                # Past safety floor — force heat regardless of price
                heat_needed_c = max_temp_c - current_temp
            elif current_temp < max_temp_c:
                slack = current_temp - min_temp_c
                steps_until_forced = (
                    int(slack / heat_loss_c_per_step)
                    if heat_loss_c_per_step > 0 else n_steps
                )
                defer_window = max(0, steps_until_forced - steps_to_recover)
                if defer_window == 0:
                    heat_needed_c = max_temp_c - current_temp
                else:
                    lookahead_end = min(t + defer_window + 1, n_steps)
                    if float(prices[t]) <= float(np.min(prices[t:lookahead_end])):
                        heat_needed_c = max_temp_c - current_temp
                    # else: cheaper slot ahead — defer
        else:
            # Naive thermostat: always heat to max when below it
            if current_temp < max_temp_c:
                heat_needed_c = max_temp_c - current_temp

        # Apply heating  (COP = 1 → electrical kWh = thermal kWh delivered)
        if heat_needed_c > 0:
            heat_needed_kwh  = heat_needed_c * thermal_mass
            heat_deliver_kwh = min(heat_needed_kwh, max_heat_kwh)
            current_temp    += heat_deliver_kwh / thermal_mass
            elec_kwh[t]      = heat_deliver_kwh
            elec_price        = prices[t] / 1000.0 if prices is not None else 0.15
            heating_cost[t]  = heat_deliver_kwh * elec_price

        # Clamp to physical range (sanity guard against floating-point drift)
        current_temp = min(max(current_temp, min_temp_c - 20.0), max_temp_c + 5.0)
        temp[t] = current_temp

    return {
        "temp_profile":   temp,
        "elec_kwh":       elec_kwh,
        "heating_cost":   heating_cost,
        "total_elec_kwh": float(elec_kwh.sum()),
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
        TotalChiller;RemainingUsage;PricesElec;ExtraCost;TotalPrices;ChillerBanks

    Returns ``{day_str: [row_dict, …]}`` keyed by ``"D/MM/YYYY"``.
    """
    days: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for raw in reader:
            row = {
                "timestamp":       raw["From Timestamp"].strip(),
                "total_usage":     parse_eu_float(raw.get("TotalUsage", "0")),
                "net_usage":       parse_eu_float(raw.get("NetUsage", "0")),
                "production_wkk":  parse_eu_float(raw.get("ProductionWKK", "0")),
                "total_chiller":   parse_eu_float(raw.get("TotalChiller", "0")),
                "chiller_banks":   parse_eu_float(raw.get("ChillerBanks", "0")),
                "remaining_usage": parse_eu_float(raw.get("RemainingUsage", "0")),
                "prices_elec":     parse_eu_float(raw.get("PricesElec", "0")),
                "extra_cost":      parse_eu_float(raw.get("ExtraCost", "0")),
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
# CHP scheduling helper
# ===================================================================

def _chp_optimal_schedule(
    n: int,
    total_price: np.ndarray,
    asset: EnergyAsset,
    dt_hours: float = 1.0,
) -> np.ndarray:
    """Compute the LP-optimal CHP firing schedule based on spark spread.

    The CHP fires only in hours where the market electricity price exceeds
    the variable gas cost of generating that electricity, minus any startup
    cost amortised over each consecutive run block.

    Parameters
    ----------
    n           : number of time slots
    total_price : total electricity price per slot [EUR/MWh] (spot + fees)
    asset       : EnergyAsset with ``chp_elec_efficiency > 0``
    dt_hours    : slot duration in hours

    Returns
    -------
    gen_kwh : np.ndarray — electrical output per slot [kWh]
    """
    if asset.chp_elec_efficiency <= 0 or asset.gas_energy_kwh_m3 <= 0:
        return np.zeros(n)

    gp = asset.gas_price_eur_m3 if asset.gas_price_eur_m3 > 0 else 0.35

    # Variable gas cost to generate 1 kWh electrical [EUR/kWh → EUR/MWh]
    var_gas_cost_mwh = gp / (asset.gas_energy_kwh_m3 * asset.chp_elec_efficiency) * 1000.0

    cap_kwh = asset.capacity_kwp * dt_hours  # max kWh output per slot

    # Step 1: fire in every profitable slot
    schedule = np.where(total_price > var_gas_cost_mwh, cap_kwh, 0.0)

    # Step 2: prune run-blocks whose gross profit does not cover startup cost
    if asset.startup_cost_eur > 0 and cap_kwh > 0:
        on = schedule > 0
        i = 0
        while i < n:
            if on[i]:
                j = i + 1
                while j < n and on[j]:
                    j += 1
                # Gross profit of this block (EUR)
                block_profit = float(
                    np.sum((total_price[i:j] - var_gas_cost_mwh) * cap_kwh / 1000.0)
                )
                if block_profit < asset.startup_cost_eur:
                    schedule[i:j] = 0.0
                i = j
            else:
                i += 1

    return schedule


# ===================================================================
# Unified LP simulation core
# ===================================================================

def simulate_slots(
    n: int,
    prices_elec: np.ndarray,
    base_load: np.ndarray,
    assets: list[EnergyAsset],
    *,
    solar_hours: dict | None = None,
    day_keys: list[str] | None = None,
    start_hour: float = 0.0,
    dt_hours: float = 1.0,
    baseline_gen_override: dict | None = None,
    baseline_load_override: dict | None = None,
    thermal_params: dict | None = None,
    outside_temp_c: np.ndarray | None = None,
) -> dict:
    """Asset-driven LP simulation over arbitrary time slots.

    This is the **unified simulation core** used by both the historical
    analysis dialog (via :func:`simulate_day`) and the future simulation
    dialog (via direct call with a synthetic load profile).

    The two use cases differ only in how the input arrays are built:

    * **Historical** — ``base_load`` and ``prices_elec`` come from the
      building CSV.  ``baseline_gen_override`` / ``baseline_load_override``
      contain measured production / load data for the baseline comparison.
    * **Future** — ``base_load`` is a synthetic sinusoidal profile;
      ``prices_elec`` comes from the user-supplied price CSV.  No
      override dicts are passed, so baselines are built from the
      configured ``baseline_start_hour`` / ``baseline_end_hour`` windows.

    Parameters
    ----------
    n : int
        Number of time slots.
    prices_elec : np.ndarray
        Spot electricity price per slot [EUR/MWh].
    base_load : np.ndarray
        Fixed (non-shiftable) building load per slot [kWh].
    assets : list[EnergyAsset]
        Enabled energy assets from the configuration.
    solar_hours : dict, optional
        ``{day_key: [kwh_h0, …]}`` from a solar archive CSV.
    day_keys : list[str], optional
        Candidate day-key strings for solar archive lookups.
    start_hour : float
        Hour-of-day index of slot 0 (0.0 = midnight).
    dt_hours : float
        Slot duration in hours (1.0 for hourly data).
    baseline_gen_override : dict, optional
        ``{asset.name: np.ndarray}`` — measured generation used as the
        *baseline* generator schedule (historical mode).
    baseline_load_override : dict, optional
        ``{asset.name: np.ndarray}`` — measured shiftable load used as
        the *baseline* schedule (historical mode).
    thermal_params : dict, optional
        Building thermal model overrides.  Supported keys:
        ``initial_temp_c``, ``setpoint_c``, ``deadband_c``,
        ``cooldown_rate_c_per_hour``, ``thermal_mass_kwh_per_c``,
        ``beo_initial_temp_c``, ``ua_kwh_per_c_per_h``.
    outside_temp_c : np.ndarray, optional
        Outside air temperature per slot [°C].  When provided the thermal
        model uses Newton’s law of cooling instead of the fixed rate.

    Returns
    -------
    dict
        Same structure as :func:`simulate_day`.
    """
    total_price = prices_elec + EXTRA_COST_EUR_MWH

    # Infer gas price from first asset that has it configured
    gas_price_eur_m3 = 0.35
    for a in assets:
        if a.enabled and getattr(a, "gas_price_eur_m3", 0.0) > 0:
            gas_price_eur_m3 = a.gas_price_eur_m3
            break

    bgo = baseline_gen_override  or {}
    blo = baseline_load_override or {}

    # ── Pre-compute LP generation for surplus-aware pricing ──────────
    # Solar and CHP produce at zero/low marginal cost.  Slots where their
    # output already exceeds the fixed base load create a surplus that can
    # cover shiftable demand for free — those slots should appear cheaper.
    _pre_gen_lp = np.zeros(n)
    for _a in assets:
        if _a.asset_type != GENERATOR or not _a.enabled:
            continue
        if _a.solar_csv:
            if solar_hours and day_keys:
                for _k in day_keys:
                    if _k in solar_hours:
                        _arr = np.array(solar_hours[_k][:n])
                        if len(_arr) < n:
                            _arr = np.pad(_arr, (0, n - len(_arr)))
                        _pre_gen_lp += _arr
                        break
            else:
                _cap = _a.capacity_kwp * dt_hours
                for _t in range(n):
                    _h = (start_hour + _t * dt_hours) % 24
                    if 6 <= _h <= 18:
                        _pre_gen_lp[_t] += max(0.0, np.sin(np.pi * (_h - 6) / 12)) * _cap
            if _a.decouple_below_eur_mwh is not None:
                _pre_gen_lp = np.where(total_price >= _a.decouple_below_eur_mwh, _pre_gen_lp, 0.0)
        elif _a.chp_elec_efficiency > 0:
            _pre_gen_lp += _chp_optimal_schedule(n, total_price, _a, dt_hours)
    _gen_surplus = np.maximum(0.0, _pre_gen_lp - base_load)

    # ── Shiftable loads ─────────────────────────────────────────────
    baseline_shiftable = np.zeros(n)
    lp_shiftable       = np.zeros(n)
    asset_loads: dict[str, dict] = {}

    for asset in assets:
        if asset.asset_type != SHIFTABLE_LOAD or not asset.enabled:
            continue

        # Baseline: historical CSV data if available, else synthetic block
        if asset.name in blo:
            actual = blo[asset.name].copy()
        else:
            actual = np.zeros(n)
            if asset.daily_energy_kwh > 0:
                start_h  = int(asset.baseline_start_hour)
                end_h    = int(asset.baseline_end_hour)
                if start_h < end_h:
                    window_h = end_h - start_h
                elif start_h > end_h:
                    window_h = (24 - start_h) + end_h
                else:
                    window_h = 0
                window_h = max(1, window_h)
                load_per_h = min(
                    asset.hourly_max_kwh,
                    asset.daily_energy_kwh / window_h,
                )
                for t in range(n):
                    h = int((start_hour + t * dt_hours)) % 24
                    if _in_window(h, start_h, end_h):
                        actual[t] = load_per_h

        daily_sum = float(np.sum(actual))
        if daily_sum <= 0:
            continue

        hourly_max = max(asset.hourly_max_kwh, float(np.max(actual)))
        baseline_shiftable += actual

        # Optional flex window: restrict LP to certain hours only
        fs = int(getattr(asset, "flex_start_hour", 0))
        fe = int(getattr(asset, "flex_end_hour", 24))
        eff_price = _effective_prices(total_price, _gen_surplus, hourly_max)
        if fs == 0 and fe == 24:
            sched_prices = eff_price
        else:
            sched_prices = eff_price.copy()
            for t in range(n):
                h = int((start_hour + t * dt_hours)) % 24
                if not _in_window(h, fs, fe):
                    sched_prices[t] = np.inf
            # Cap daily_sum to what physically fits in the restricted flex window.
            # Without this, greedy_lp overflows energy into inf-priced slots
            # (outside the window) when the window is too small.
            n_window_slots = sum(
                1 for _t in range(n)
                if _in_window(int((start_hour + _t * dt_hours)) % 24, fs, fe)
            )
            daily_sum = min(daily_sum, float(n_window_slots) * hourly_max)

        lp_sched = greedy_lp_ramped(
            n, sched_prices, daily_sum, hourly_max,
            ramp_up_pct=asset.ramp_up_pct_per_hour,
            ramp_down_pct=asset.ramp_down_pct_per_hour,
        )
        lp_shiftable += lp_sched

        asset_loads[asset.name] = {
            "baseline":        actual.copy(),
            "optimised":       lp_sched.copy(),
            "daily_kwh":       daily_sum,
            "cost_saving_eur": float(np.sum((actual - lp_sched) * total_price / 1000.0)),
        }

    # ── Generators ──────────────────────────────────────────────────
    total_gen_baseline    = np.zeros(n)
    total_gen_lp_effective = np.zeros(n)
    asset_generators: dict[str, dict] = {}
    chp_heat_kw_lp = np.zeros(n)  # LP CHP exhaust heat → fed to thermal model
    chp_gas_cost_baseline = np.zeros(n)  # Fuel cost of baseline CHP operation [EUR/slot]
    chp_gas_cost_lp       = np.zeros(n)  # Fuel cost of LP CHP operation [EUR/slot]

    for asset in assets:
        if asset.asset_type != GENERATOR or not asset.enabled:
            continue

        gen_full = np.zeros(n)  # baseline generation

        # ─ Solar ──────────────────────────────────────────────────
        if asset.solar_csv:
            if solar_hours and day_keys:
                for k in day_keys:
                    if k in solar_hours:
                        arr = np.array(solar_hours[k][:n])
                        if len(arr) < n:
                            arr = np.pad(arr, (0, n - len(arr)))
                        gen_full += arr
                        break
            else:
                # Synthetic sinusoidal solar model (future mode)
                cap = asset.capacity_kwp * dt_hours
                for t in range(n):
                    h = (start_hour + t * dt_hours) % 24
                    if 6 <= h <= 18:
                        gen_full[t] = max(0.0, np.sin(np.pi * (h - 6) / 12)) * cap

            gen_lp = gen_full.copy()
            if asset.decouple_below_eur_mwh is not None:
                gen_lp = np.where(total_price >= asset.decouple_below_eur_mwh, gen_lp, 0.0)

            total_gen_baseline    += gen_full
            total_gen_lp_effective += gen_lp
            asset_generators[asset.name] = {
                "gen_kwh":          gen_lp.copy(),
                "gen_kwh_baseline": gen_full.copy(),
                "cost_saving_eur":  float(np.sum((gen_lp - gen_full) * total_price / 1000.0)),
            }
            continue

        # ─ CHP (spark-spread scheduled) ───────────────────────────
        if asset.chp_elec_efficiency > 0:
            gen_lp = _chp_optimal_schedule(n, total_price, asset, dt_hours)

            # Baseline: historical CSV production OR synthetic window schedule
            if asset.name in bgo:
                gen_full = bgo[asset.name].copy()
            else:
                start_h = int(asset.baseline_start_hour)
                end_h   = int(asset.baseline_end_hour)
                cap_kwh = asset.capacity_kwp * dt_hours
                for t in range(n):
                    h = int((start_hour + t * dt_hours)) % 24
                    if _in_window(h, start_h, end_h):
                        gen_full[t] = cap_kwh

            total_gen_baseline    += gen_full
            total_gen_lp_effective += gen_lp

            # Exhaust heat from LP CHP schedule
            if asset.chp_heat_efficiency > 0:
                heat_ratio = asset.chp_heat_efficiency / asset.chp_elec_efficiency
                chp_heat_kw_lp += gen_lp * heat_ratio / dt_hours  # kW

            # Gas fuel cost — subtract from savings so only the spark spread is shown
            gp = asset.gas_price_eur_m3 if asset.gas_price_eur_m3 > 0 else 0.35
            hv = asset.gas_energy_kwh_m3 if asset.gas_energy_kwh_m3 > 0 else 9.8
            gas_cost_per_kwh_elec = gp / (hv * asset.chp_elec_efficiency)  # EUR/kWh
            chp_gas_cost_baseline += gen_full * gas_cost_per_kwh_elec
            chp_gas_cost_lp       += gen_lp   * gas_cost_per_kwh_elec

            asset_generators[asset.name] = {
                "gen_kwh":          gen_lp.copy(),
                "gen_kwh_baseline": gen_full.copy(),
                # Net saving = electricity value of extra generation minus extra gas cost
                "cost_saving_eur":  float(np.sum(
                    (gen_lp - gen_full) * (total_price / 1000.0 - gas_cost_per_kwh_elec)
                )),
            }
            continue

        # ─ Other generators (fixed CSV or capacity-based) ─────────
        if asset.name in bgo:
            gen_full = bgo[asset.name].copy()
        elif asset.capacity_kwp > 0:
            # Fixed-output generator — run during baseline window
            start_h = int(asset.baseline_start_hour)
            end_h   = int(asset.baseline_end_hour)
            cap_kwh = asset.capacity_kwp * dt_hours
            for t in range(n):
                h = int((start_hour + t * dt_hours)) % 24
                if _in_window(h, start_h, end_h):
                    gen_full[t] = cap_kwh

        gen_lp = gen_full.copy()
        if asset.decouple_below_eur_mwh is not None:
            gen_lp = np.where(total_price >= asset.decouple_below_eur_mwh, gen_lp, 0.0)

        total_gen_baseline    += gen_full
        total_gen_lp_effective += gen_lp
        asset_generators[asset.name] = {
            "gen_kwh":          gen_lp.copy(),
            "gen_kwh_baseline": gen_full.copy(),
            "cost_saving_eur":  float(np.sum((gen_lp - gen_full) * total_price / 1000.0)),
        }

    # ── Startup costs ────────────────────────────────────────────────
    total_startup_cost = 0.0
    for asset in assets:
        if asset.asset_type != GENERATOR or asset.startup_cost_eur <= 0 or not asset.enabled:
            continue
        gen_kwh = asset_generators.get(asset.name, {}).get("gen_kwh", np.zeros(n))
        on = gen_kwh > 0
        starts = int(np.sum(on[1:] & ~on[:-1]))
        if len(on) > 0 and on[0]:
            starts += 1
        total_startup_cost += starts * asset.startup_cost_eur

    # ── Building thermal simulation ──────────────────────────────────
    # Priority: explicit thermal_params arg → GUI-saved config (disk) → hardcoded
    tp = thermal_params or {}
    def _tp(key: str, fallback: float) -> float:
        if key in tp:
            return tp[key]
        if key in _THERMAL_CFG:
            return _THERMAL_CFG[key]
        return fallback

    thermal_common = dict(
        n_steps=n,
        dt_hours=dt_hours,
        initial_temp_c=_tp("initial_temp_c", 21.0),
        setpoint_c=_tp("setpoint_c", 21.0),
        deadband_c=_tp("deadband_c", 1.0),
        cooldown_rate_c_per_hour=_tp("cooldown_rate_c_per_hour", 0.5),
        thermal_mass_kwh_per_c=_tp("thermal_mass_kwh_per_c", 50.0),
        assets=assets,
        prices=total_price,
        beo_initial_temp_c=_tp("beo_initial_temp_c", 12.0),
        chp_heat_kw=chp_heat_kw_lp,
        gas_price_eur_m3=gas_price_eur_m3,
        outside_temp_c=outside_temp_c,
        ua_kwh_per_c_per_h=_tp("ua_kwh_per_c_per_h", 2.5),
    )
    # Baseline thermal: constant rule — always maintain setpoint
    baseline_thermal = simulate_building_thermal(**thermal_common, price_aware=False)
    # LP thermal: price-aware — pre-heat during cheap slots
    thermal_result = simulate_building_thermal(**thermal_common, price_aware=True)

    # ── Hot water tank simulation ─────────────────────────────────────
    # Config is read from _HW_CFG (refreshed by reload_hw_config after GUI save).
    _hw_enabled = bool(_HW_CFG.get("enabled", False))
    if _hw_enabled:
        _hw_common = dict(
            n_steps=n,
            dt_hours=dt_hours,
            initial_temp_c=float(_HW_CFG.get("initial_temp_c", 55.0)),
            min_temp_c=float(_HW_CFG.get("min_temp_c", 45.0)),
            max_temp_c=float(_HW_CFG.get("max_temp_c", 60.0)),
            heat_loss_w=float(_HW_CFG.get("heat_loss_w", 50.0)),
            heater_power_kw=float(_HW_CFG.get("heater_power_kw", 3.0)),
            tank_volume_l=float(_HW_CFG.get("tank_volume_l", 200.0)),
            prices=total_price,
        )
        baseline_hw = simulate_hot_water_tank(**_hw_common, price_aware=False)
        lp_hw       = simulate_hot_water_tank(**_hw_common, price_aware=True)
    else:
        _zero_hw: dict = {
            "heating_cost": np.zeros(n),
            "elec_kwh": np.zeros(n),
            "temp_profile": np.zeros(n),
            "total_elec_kwh": 0.0,
        }
        baseline_hw = _zero_hw
        lp_hw       = _zero_hw

    # ── Costs ────────────────────────────────────────────────────────
    baseline_load = base_load + baseline_shiftable
    baseline_grid = np.maximum(baseline_load - total_gen_baseline, 0.0)
    baseline_cost = baseline_grid * total_price / 1000.0 + chp_gas_cost_baseline
    baseline_cost += baseline_thermal["heating_cost"]
    baseline_cost += baseline_hw["heating_cost"]

    lp_load = base_load + lp_shiftable
    lp_grid = np.maximum(lp_load - total_gen_lp_effective, 0.0)
    lp_cost = lp_grid * total_price / 1000.0 + chp_gas_cost_lp
    if n > 0 and total_startup_cost > 0:
        lp_cost[0] += total_startup_cost
    lp_cost += thermal_result["heating_cost"]
    lp_cost += lp_hw["heating_cost"]

    heating_saving_eur = float(baseline_thermal["heating_cost"].sum()) - float(thermal_result["heating_cost"].sum())
    heating_baseline_cost_eur = float(baseline_thermal["heating_cost"].sum())
    heating_saving_pct = (
        (heating_saving_eur / heating_baseline_cost_eur * 100)
        if heating_baseline_cost_eur > 0 else 0.0
    )

    hw_saving_eur = float(baseline_hw["heating_cost"].sum()) - float(lp_hw["heating_cost"].sum())
    hw_baseline_cost_eur = float(baseline_hw["heating_cost"].sum())
    hw_saving_pct = (
        (hw_saving_eur / hw_baseline_cost_eur * 100)
        if hw_baseline_cost_eur > 0 else 0.0
    )

    total_load_shifted = sum(float(np.sum(ad["baseline"])) for ad in asset_loads.values())

    return {
        "baseline":           baseline_cost,
        "optimised":          lp_cost,
        "baseline_load_kwh":  baseline_load,
        "optimised_load_kwh": lp_load,
        "baseline_grid_kwh":  baseline_grid,
        "optimised_grid_kwh": lp_grid,
        "prices_elec":        prices_elec,
        "load_shifted":       total_load_shifted,
        "total_generation":   float(np.sum(total_gen_lp_effective)),
        "startup_cost":       total_startup_cost,
        "temp_profile":       thermal_result["temp_profile"],
        "heating_kw":         thermal_result["heating_kw"],
        "beo_temp":           thermal_result["beo_temp"],
        "heating_saving_eur": heating_saving_eur,
        "heating_saving_pct": heating_saving_pct,
        "heating_baseline_cost_eur": heating_baseline_cost_eur,
        "hw_temp_profile":    lp_hw["temp_profile"],
        "hw_elec_kwh":        lp_hw["elec_kwh"],
        "hw_saving_eur":      hw_saving_eur,
        "hw_saving_pct":      hw_saving_pct,
        "hw_baseline_cost_eur": hw_baseline_cost_eur,
        "n_slots":            n,
        "asset_loads":        asset_loads,
        "asset_generators":   asset_generators,
    }


# ===================================================================
# Historical day simulation (asset-driven)
# ===================================================================

# Column name → parsed row-dict key mapping
_COL_MAP = {
    "TotalChiller":  "total_chiller",
    "ChillerBanks":  "chiller_banks",
    "ProductionWKK": "production_wkk",
}


def simulate_day(
    day_rows: list[dict],
    assets: list[EnergyAsset],
    solar_hours: dict[str, list[float]],
    day_keys: list[str],
    thermal_params: dict | None = None,
) -> dict:
    """Compare baseline vs LP-optimised cost for one historical day.

    Thin wrapper around :func:`simulate_slots` that extracts input
    arrays from the historical building CSV rows and builds baseline
    override dicts from measured production and load data.
    """
    n = len(day_rows)
    remaining   = np.array([r["remaining_usage"] for r in day_rows])
    prices_elec = np.array([r["prices_elec"]     for r in day_rows])

    # Build baseline override dicts from measured CSV data
    bgo: dict[str, np.ndarray] = {}  # generator baselines
    blo: dict[str, np.ndarray] = {}  # shiftable-load baselines

    for asset in assets:
        if not asset.enabled:
            continue
        if asset.asset_type == GENERATOR and asset.csv_gen_column:
            col = _COL_MAP.get(asset.csv_gen_column, asset.csv_gen_column)
            bgo[asset.name] = np.array([r.get(col, 0.0) for r in day_rows])
        if asset.asset_type == SHIFTABLE_LOAD and asset.csv_column:
            col = _COL_MAP.get(asset.csv_column,
                               asset.csv_column.lower().replace(" ", "_"))
            arr = np.array([r.get(col, 0.0) for r in day_rows])
            # Only use CSV if it has meaningful data
            if float(np.sum(arr)) >= asset.hourly_max_kwh * 0.5:
                blo[asset.name] = arr

    return simulate_slots(
        n=n,
        prices_elec=prices_elec,
        base_load=remaining,
        assets=assets,
        solar_hours=solar_hours,
        day_keys=day_keys,
        dt_hours=1.0,
        baseline_gen_override=bgo,
        baseline_load_override=blo,
        thermal_params=thermal_params,
    )


# ────────────────────────────────────────────────────────────────────
# Parallel full-year worker (used by historical_dialog._do_year_analysis)
# ────────────────────────────────────────────────────────────────────
# Module-level state set by the pool initializer so we don't re-pickle
# the (possibly large) asset list + solar_hours dict for every day.
_YEAR_WORKER_ASSETS: list | None = None
_YEAR_WORKER_SOLAR: dict | None = None
_YEAR_WORKER_THERMAL: dict | None = None


def _year_worker_init(assets, solar_hours, thermal_params=None):
    """ProcessPoolExecutor initializer — store shared inputs as globals."""
    global _YEAR_WORKER_ASSETS, _YEAR_WORKER_SOLAR, _YEAR_WORKER_THERMAL
    _YEAR_WORKER_ASSETS = assets
    _YEAR_WORKER_SOLAR = solar_hours
    _YEAR_WORKER_THERMAL = thermal_params
    # Avoid HIGHS / BLAS oversubscription when many workers run in parallel.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _year_worker_task(day_iso: str, day_rows: list[dict], day_keys: list[str]) -> tuple[str, dict]:
    """Solve one historical day in a worker process; return (day_iso, results)."""
    res = simulate_day(
        day_rows,
        _YEAR_WORKER_ASSETS or [],
        _YEAR_WORKER_SOLAR or {},
        day_keys,
        thermal_params=_YEAR_WORKER_THERMAL,
    )
    return day_iso, res

