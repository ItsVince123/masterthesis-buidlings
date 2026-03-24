"""Fetch 2022 historical weather from Open-Meteo and generate hourly solar predictions.

Uses Open-Meteo's free Historical Weather API (archive-api) which covers data
back to 1940.  Uses shortwave radiation (W/m²) for accurate PV estimation
with temperature derating.

Output: solar_2022.csv  (semicolon-delimited, hourly)
Columns: Timestamp;Solar_kWh

The Timestamp format matches DATA.csv: ``D/MM/YYYY H:MM`` (e.g. ``1/01/2022 0:00``).

Usage::

    python generate_solar_2022.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from settings import DASHBOARD_DIR, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
SOLAR_CAPACITY_KWP = 1.525
OUTPUT_PATH = DASHBOARD_DIR / "solar_2022.csv"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Standard Test Conditions irradiance (W/m²)
STC_IRRADIANCE = 1000.0


def predict_solar_kwh(
    shortwave_radiation_wm2: float,
    temperature_c: float,
    capacity_kwp: float,
) -> float:
    """Estimate hourly solar yield (kWh) from irradiance and temperature.

    Uses shortwave radiation (W/m²) normalised by STC (1000 W/m²)
    with -0.4%/°C temperature derating above 25 °C (clamped at 75%).
    """
    irradiance_factor = max(0.0, shortwave_radiation_wm2 / STC_IRRADIANCE)
    temp_derate = 1.0 - max(0.0, temperature_c - 25.0) * 0.004
    temp_derate = max(0.75, temp_derate)
    power_kw = max(0.0, capacity_kwp * irradiance_factor * temp_derate)
    return power_kw  # 1 hour interval -> kWh = kW × 1 h


def _format_timestamp(ts: pd.Timestamp) -> str:
    """Format timestamp to match DATA.csv: ``D/MM/YYYY H:MM`` (no leading zeros on day/hour)."""
    return f"{ts.day}/{ts.month:02d}/{ts.year} {ts.hour}:{ts.minute:02d}"


def main():
    cache = requests_cache.CachedSession(".cache_historical", expire_after=-1)
    retry_sess = retry(cache, retries=5, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_sess)

    all_rows: list[tuple[str, float]] = []

    # Fetch in monthly chunks to stay within API limits
    for month in range(1, 13):
        start = f"2022-{month:02d}-01"
        if month == 12:
            end = "2022-12-31"
        else:
            end = (pd.Timestamp(f"2022-{month + 1:02d}-01") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info("Fetching %s to %s ...", start, end)

        params = {
            "latitude": DEFAULT_LATITUDE,
            "longitude": DEFAULT_LONGITUDE,
            "hourly": ["temperature_2m", "shortwave_radiation"],
            "timezone": "Europe/Brussels",
            "start_date": start,
            "end_date": end,
        }

        response = client.weather_api(ARCHIVE_URL, params=params)[0]
        hourly = response.Hourly()
        temperature = hourly.Variables(0).ValuesAsNumpy()
        radiation = hourly.Variables(1).ValuesAsNumpy()

        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ).tz_convert("Europe/Brussels")

        n = min(len(timestamps), len(temperature), len(radiation))
        for i in range(n):
            ts_str = _format_timestamp(timestamps[i])
            rad = float(radiation[i]) if not np.isnan(radiation[i]) else 0.0
            temp = float(temperature[i]) if not np.isnan(temperature[i]) else 15.0
            solar = predict_solar_kwh(rad, temp, SOLAR_CAPACITY_KWP)
            all_rows.append((ts_str, solar))

        logger.info("  -> %d hours fetched", n)

    # Export CSV (semicolon-delimited, European decimals)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Timestamp", "Solar_kWh"])
        for ts, solar in all_rows:
            writer.writerow([ts, f"{solar:.4f}".replace(".", ",")])

    logger.info("Saved %d rows to %s", len(all_rows), OUTPUT_PATH)
    print(f"\nDone! {len(all_rows)} rows saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
