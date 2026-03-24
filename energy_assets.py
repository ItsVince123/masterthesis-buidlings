"""Shared energy-asset definitions used by the dashboard UI and LP solver.

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

ASSET_TYPES = {
    SHIFTABLE_LOAD: "Shiftable Load",
    GENERATOR: "Generator",
    FIXED_LOAD: "Fixed Load",
    STORAGE: "Storage",
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

    # ── Generator properties ────────────────────────────────────────
    capacity_kwp: float = 0.0          # installed capacity (for solar)
    solar_csv: str = ""                # file with hourly solar production (optional)
    csv_gen_column: str = ""           # CSV column for non-solar gen (optional)
    decouple_below_eur_mwh: float | None = None  # disconnect when price < this

    # ── Storage properties ──────────────────────────────────────────
    storage_capacity_kwh: float = 0.0  # total storage capacity
    charge_rate_kw: float = 0.0        # max charge rate
    discharge_rate_kw: float = 0.0     # max discharge rate
    efficiency: float = 0.90           # round-trip efficiency

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
