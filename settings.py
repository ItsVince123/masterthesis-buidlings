"""
Central configuration constants for the Dashboard application.

All hardcoded values (paths, coordinates, API config) live here so they
can be changed in one place.  Every other module imports the constants it
needs rather than duplicating magic values.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
WEATHER_CSV = DASHBOARD_DIR / "weather.csv"
PREDICT_CSV = DASHBOARD_DIR / "predict.csv"
PRICES_CSV = DASHBOARD_DIR / "prices.csv"
CONFIG_JSON = DASHBOARD_DIR / "dashboard_config.json"

# Historical building data CSV (hourly data with chiller, CHP, prices).
# Used by HistoricalAnalysisDialog.
HISTORICAL_CSV = DASHBOARD_DIR / "DATA.csv"

# ---------------------------------------------------------------------------
# Timezone & default location (Brussels)
# ---------------------------------------------------------------------------
LOCAL_TZ = ZoneInfo("Europe/Brussels")
DEFAULT_LATITUDE = 50.85045
DEFAULT_LONGITUDE = 4.34878

# ---------------------------------------------------------------------------
# ENTSO-E transparency platform
# ---------------------------------------------------------------------------
def get_entsoe_api_key() -> str:
    """Return the ENTSO-E API key.

    Priority: dashboard_config.json > ENTSOE_API_KEY env var > built-in fallback.
    """
    import json
    try:
        with open(CONFIG_JSON, "r", encoding="utf-8") as _f:
            _key = json.load(_f).get("api_keys", {}).get("entsoe_api_key", "").strip()
        if _key:
            return _key
    except Exception:
        pass
    return os.environ.get("ENTSOE_API_KEY", "a13c900f-96f6-4fdf-ba15-4ce38bdd651b")


# Set at import time so modules that do `from settings import ENTSOE_API_KEY` work.
ENTSOE_API_KEY = get_entsoe_api_key()
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_DOMAIN = "10YBE----------2"  # Belgium bidding zone

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
# Grid/distribution/tax charges that apply on top of the ENTSO-E spot price.
# Typical Belgian industrial value is ~50 EUR/MWh  (= 0.05 EUR/kWh).
# This fee is added in BOTH the LP solver and the SMPC calculator so the
# optimiser includes the real total cost, not just the volatile spot component.
GRID_FEE_EUR_MWH = 50.0       # EUR/MWh = EXTRA_COST_EUR_MWH in energy_assets.py

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
DAILY_FETCH_HOUR = 14           # Local hour at which daily data refresh runs
                                # (14:00 ensures tomorrow's DAM prices are available)
SOLAR_CAPACITY_KWP = 1.525      # Installed PV capacity (kWp) — fallback only
INTERVAL_MINUTES = 15           # Time resolution used throughout the system
                                # (must match ENTSO-E data resolution after Oct 2025)
