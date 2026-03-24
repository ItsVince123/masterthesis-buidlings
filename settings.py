"""Central configuration constants for the Dashboard application.

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
# Prefer setting the ENTSOE_API_KEY environment variable.
# If unset, the fallback value below is used (rotate if compromised).
ENTSOE_API_KEY = os.environ.get(
    "ENTSOE_API_KEY",
    "a13c900f-96f6-4fdf-ba15-4ce38bdd651b",
)
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_DOMAIN = "10YBE----------2"  # Belgium bidding zone

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
# Grid/distribution/tax charges that apply on top of the ENTSO-E spot price.
# Typical Belgian industrial value is ~50 EUR/MWh  (= 0.05 EUR/kWh).
GRID_FEE_EUR_MWH = 50.0

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
DAILY_FETCH_HOUR = 14           # Local hour at which daily data refresh runs
SOLAR_CAPACITY_KWP = 1.525     # Installed PV capacity (kWp)
INTERVAL_MINUTES = 15           # Time resolution used throughout the system
