"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  Weather data fetcher (Open-Meteo API).                          ║
║  Retrieves 48h of temperature, UV index, and wind speed at       ║
║  15-minute resolution and caches to weather.csv.                 ║
╚══════════════════════════════════════════════════════════════════╝

Fetch 48 h weather data (temperature, UV index, wind speed) from Open-Meteo.

Exports a semicolon-delimited CSV with 15-minute resolution.

Usage::

    python getWeather.py --lat 50.85045 --lon 4.34878 --output weather.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from settings import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, WEATHER_CSV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def build_client():
    """Build an Open-Meteo client with local caching and retry logic."""
    cache = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_sess = retry(cache, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_sess)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_weather(
    client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    """Fetch weather data from Open-Meteo for a date range.

    Returns a dict with keys ``timestamps``, ``temperature``, ``uv_index``,
    ``wind_speed`` — all aligned at 15-minute resolution.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": ["temperature_2m", "wind_speed_10m"],
        "hourly": "uv_index",
        "timezone": "Europe/Brussels",
        "start_date": start_date,
        "end_date": end_date,
    }

    logger.info(
        "Fetching weather for (%.5f, %.5f) | %s \u2192 %s",
        latitude, longitude, start_date, end_date,
    )
    response = client.weather_api(OPEN_METEO_URL, params=params)[0]

    # --- 15-minute data: temperature & wind speed --------------------------
    minutely = response.Minutely15()
    temperature = minutely.Variables(0).ValuesAsNumpy()
    wind_speed = minutely.Variables(1).ValuesAsNumpy()

    ts_15m = pd.date_range(
        start=pd.to_datetime(minutely.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=minutely.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Brussels")

    # --- Hourly UV index \u2192 interpolate to 15-min -------------------------
    hourly = response.Hourly()
    uv_hourly = hourly.Variables(0).ValuesAsNumpy()

    ts_hourly = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Brussels")

    uv_series = pd.Series(uv_hourly, index=ts_hourly)
    uv_reindexed = uv_series.reindex(uv_series.index.union(ts_15m))
    uv_interp = (
        uv_reindexed
        .interpolate(method="time")
        .reindex(ts_15m)
        .fillna(0.0)
    )

    # Align all arrays to the shortest common length
    n = min(len(ts_15m), len(temperature), len(wind_speed), len(uv_interp))
    logger.info("Fetched %d data points (15-min resolution)", n)

    return {
        "timestamps": ts_15m[:n],
        "temperature": temperature[:n].tolist(),
        "uv_index": uv_interp.values[:n].tolist(),
        "wind_speed": wind_speed[:n].tolist(),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(data: dict, output_path: Path) -> None:
    """Write weather data to a semicolon-delimited CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Timestamp", "Temperature (\u00b0C)", "UV Index",
            "Wind Speed (km/h)",
        ])
        for i, ts in enumerate(data["timestamps"]):
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                f"{data['temperature'][i]:.2f}".replace(".", ","),
                f"{data['uv_index'][i]:.2f}".replace(".", ","),
                f"{data['wind_speed'][i]:.2f}".replace(".", ","),
            ])

    logger.info("Exported %d rows to %s", len(data["timestamps"]), output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch 48 h weather data from Open-Meteo.",
    )
    parser.add_argument(
        "--lat", type=float, default=DEFAULT_LATITUDE,
        help="Latitude (default: Brussels)",
    )
    parser.add_argument(
        "--lon", type=float, default=DEFAULT_LONGITUDE,
        help="Longitude (default: Brussels)",
    )
    parser.add_argument(
        "--output", default=str(WEATHER_CSV),
        help="Output CSV file path",
    )
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    client = build_client()
    data = fetch_weather(client, args.lat, args.lon, today, tomorrow)
    export_csv(data, Path(args.output))

    n = len(data["timestamps"])
    print(f"\nDone! {n} rows saved to: {args.output}")
    print(f"  Temperature : {min(data['temperature']):.1f}"
          f" \u2013 {max(data['temperature']):.1f} \u00b0C")
    print(f"  UV Index    : {min(data['uv_index']):.1f}"
          f" \u2013 {max(data['uv_index']):.1f}")
    print(f"  Wind Speed  : {min(data['wind_speed']):.1f}"
          f" \u2013 {max(data['wind_speed']):.1f} km/h")


if __name__ == "__main__":
    main()
