"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
╚══════════════════════════════════════════════════════════════════╝

Shared energy-asset definitions used by the dashboard UI and LP solver.

Asset categories
----------------
**Shiftable load** (``"shiftable_load"``):
    A load that must consume a fixed total daily energy but whose hourly
    schedule can be shifted to the cheapest hours.  Example: ice-bank chillers.

**Generator** (``"generator"``):
    On-site generation that reduces grid draw.  Can optionally be *decoupled*
    when the electricity price drops below a configurable threshold.
    Examples: solar panels, CHP / WKK.

**Fixed load** (``"fixed_load"``):
    Non-controllable background consumption.  Always present (RemainingUsage
    from the CSV).  Not user-configurable — it comes straight from the data.

Persistence
-----------
Assets are stored under ``dashboard_config.json`` → ``"energy_assets"`` and
loaded/saved via :func:`load_assets` / :func:`save_assets`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from settings import CONFIG_JSON

# ── Asset type constants ────────────────────────────────────────────
SHIFTABLE_LOAD = "shiftable_load"
GENERATOR = "generator"
FIXED_LOAD = "fixed_load"
STORAGE = "storage"
GAS_HEATER = "gas_heater"
HEAT_PUMP = "heat_pump"

ASSET_TYPES = {
    SHIFTABLE_LOAD: "Shiftable Load",
    GENERATOR: "Generator",
    FIXED_LOAD: "Fixed Load",
    STORAGE: "Storage",
    GAS_HEATER: "Gas Heater",
    HEAT_PUMP: "Heat Pump",
}

# Extra cost added on top of spot price (distribution fees, taxes, etc.)
EXTRA_COST_EUR_MWH = 50.0


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class EnergyAsset:
    """One user-defined energy asset."""

    uid: str = ""                      # unique id (auto-generated)
    name: str = ""                     # display name
    asset_type: str = SHIFTABLE_LOAD   # "shiftable_load" or "generator"
    category: str = "input"            # "input" or "output"
    icon: str = "plug"                 # icon key for the dashboard column
    enabled: bool = True               # toggle without deleting

    # ── Shiftable-load properties ───────────────────────────────────
    csv_column: str = ""               # CSV column (optional – blank = no link)
    hourly_max_kwh: float = 720.0      # max kWh in any single hour
    daily_energy_kwh: float = 0.0      # fixed daily energy demand (0 = from CSV)
    ramp_up_pct_per_hour: float = 100.0    # max ramp-up rate [% of hourly_max / hour]
    ramp_down_pct_per_hour: float = 100.0  # max ramp-down rate [% of hourly_max / hour]
    # Fixed-block baseline schedule (used when csv_column is empty)
    baseline_start_hour: int = 9           # hour index when fixed baseline starts (inclusive)
    baseline_end_hour: int = 19            # hour index when fixed baseline ends (exclusive)
    # Flexible hours window (optional LP constraint — 0/24 = unrestricted)
    flex_start_hour: int = 0               # LP may only schedule from this hour (inclusive)
    flex_end_hour: int = 24                # LP may only schedule until this hour (exclusive)

    # ── Generator properties ────────────────────────────────────────
    capacity_kwp: float = 0.0          # installed capacity (for solar)
    solar_csv: str = ""                # file with hourly solar production (optional)
    csv_gen_column: str = ""           # CSV column for non-solar gen (optional)
    decouple_below_eur_mwh: float | None = None  # disconnect when price < this
    startup_cost_eur: float = 0.0      # one-off cost each time unit starts up

    # ── CHP / cogeneration properties (GENERATOR type) ─────────────
    chp_elec_efficiency: float = 0.0   # electrical efficiency; >0 flags this as a CHP
    chp_heat_efficiency: float = 0.0   # recoverable exhaust-heat fraction of fuel input
    gas_price_eur_m3: float = 0.0      # gas price [EUR/m³] (0 → use SMPC config default)
    gas_energy_kwh_m3: float = 9.8     # calorific value of gas [kWh/m³]

    # ── Storage properties ──────────────────────────────────────────
    storage_capacity_kwh: float = 0.0  # total storage capacity
    charge_rate_kw: float = 0.0        # max charge rate
    discharge_rate_kw: float = 0.0     # max discharge rate
    efficiency: float = 0.90           # round-trip efficiency

    # ── Gas heater properties ───────────────────────────────────────
    thermal_output_kw: float = 0.0     # rated thermal output [kW]
    gas_efficiency: float = 0.92       # thermal efficiency (HHV)
    gas_consumption_m3_per_hour: float = 0.0   # max gas consumption [m³/h]

    # ── Heat pump properties ────────────────────────────────────────
    cop: float = 3.5                   # coefficient of performance
    electrical_input_kw: float = 0.0   # rated electrical input [kW]
    heating_capacity_kw: float = 0.0   # rated heating capacity [kW]
    cooling_capacity_kw: float = 0.0   # rated cooling capacity [kW] (0 = heating-only)

    # ── Heat pump source (air vs ground / BEO-veld) ─────────────────
    is_ground_source: bool = False     # True = ground-source (BEO), False = air-source
    beo_capacity_kwh: float = 50000.0  # BEO-veld thermal storage capacity [kWh]
    beo_extraction_rate_kw: float = 0.0  # max heat extraction from ground [kW]
    beo_injection_rate_kw: float = 0.0   # max heat injection into ground [kW]
    beo_initial_temp_c: float = 12.0   # initial ground temperature [°C]

    def __post_init__(self):
        if not self.uid:
            self.uid = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnergyAsset":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Persistence helpers ─────────────────────────────────────────────

def _read_config() -> dict:
    if not CONFIG_JSON.exists():
        return {}
    with CONFIG_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(data: dict) -> None:
    with CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_assets() -> list[EnergyAsset]:
    """Load energy assets from ``dashboard_config.json``."""
    raw = _read_config()
    return [EnergyAsset.from_dict(d) for d in raw.get("energy_assets", [])]


def save_assets(assets: list[EnergyAsset]) -> None:
    """Save energy assets to ``dashboard_config.json``."""
    data = _read_config()
    data["energy_assets"] = [a.to_dict() for a in assets]
    _write_config(data)


# ── Default assets (created on first run) ───────────────────────────

def get_default_assets() -> list[EnergyAsset]:
    """Return the default asset set matching the current DATA.csv structure."""
    return [
        EnergyAsset(
            uid="ice_banks",
            name="Ice Banks",
            asset_type=SHIFTABLE_LOAD,
            category="output",
            icon="snowflake",
            csv_column="TotalChiller",
            hourly_max_kwh=720.0,
        ),
        EnergyAsset(
            uid="solar",
            name="Solar Panels",
            asset_type=GENERATOR,
            category="input",
            icon="sun",
            capacity_kwp=1.525,
            solar_csv="solar_2022.csv",
            decouple_below_eur_mwh=0.0,
        ),
        EnergyAsset(
            uid="wkk",
            name="CHP / WKK",
            asset_type=GENERATOR,
            category="input",
            icon="fire",
            csv_gen_column="ProductionWKK",
        ),
        EnergyAsset(
            uid="gas_heater_1",
            name="Gas Boiler",
            asset_type=GAS_HEATER,
            category="input",
            icon="fire",
            thermal_output_kw=500.0,
            gas_efficiency=0.92,
            gas_consumption_m3_per_hour=55.0,
        ),
        EnergyAsset(
            uid="heat_pump_1",
            name="Ground-Source Heat Pump",
            asset_type=HEAT_PUMP,
            category="input",
            icon="heat",
            cop=4.2,
            electrical_input_kw=120.0,
            heating_capacity_kw=500.0,
            cooling_capacity_kw=400.0,
            is_ground_source=True,
            beo_capacity_kwh=50000.0,
            beo_extraction_rate_kw=350.0,
            beo_injection_rate_kw=300.0,
            beo_initial_temp_c=12.0,
        ),
    ]


# ── Site-level helpers ──────────────────────────────────────────────

def active_assets() -> list[EnergyAsset]:
    """Return only enabled assets."""
    return [a for a in load_assets() if a.enabled]


def ensure_defaults() -> list[EnergyAsset]:
    """Load assets; if none exist yet, create the defaults and save them."""
    assets = load_assets()
    if not assets:
        assets = get_default_assets()
        save_assets(assets)
    return assets
