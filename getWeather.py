"""
Fetch 48h weather data (temperature, UV index, wind speed) from Open-Meteo.
Exports a CSV with 15-minute resolution timestamps.

Usage: python fetch_weather.py --lat 50.85045 --lon 4.34878 --output weather.csv
"""

import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"


def build_client():
    """Build Open-Meteo client with caching and retry logic."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_weather(client, latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """
    Fetch weather data from Open-Meteo for a date range.

    Returns a dict with lists: timestamps, temperatures, uv_index, wind_speed
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

    logger.info(f"Fetching weather for ({latitude}, {longitude}) | {start_date} → {end_date}")
    responses = client.weather_api(BASE_URL, params=params)
    response = responses[0]

    # --- 15-minute data: temperature & wind ---
    minutely = response.Minutely15()
    temperature = minutely.Variables(0).ValuesAsNumpy()
    wind_speed  = minutely.Variables(1).ValuesAsNumpy()

    timestamps_15min = pd.date_range(
        start=pd.to_datetime(minutely.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=minutely.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Brussels")

    # --- Hourly data: UV index → interpolate to 15-min ---
    hourly = response.Hourly()
    uv_hourly = hourly.Variables(0).ValuesAsNumpy()

    timestamps_hourly = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert("Europe/Brussels")

    # Interpolate UV to 15-min resolution
    uv_series = pd.Series(uv_hourly, index=timestamps_hourly)
    uv_reindexed = uv_series.reindex(uv_series.index.union(timestamps_15min))
    uv_interpolated = uv_reindexed.interpolate(method="time").reindex(timestamps_15min).fillna(0.0)

    # Align all to the same length
    n = min(len(timestamps_15min), len(temperature), len(wind_speed), len(uv_interpolated))
    logger.info(f"Fetched {n} data points (15-min resolution)")

    return {
        "timestamps": timestamps_15min[:n],
        "temperature": temperature[:n].tolist(),
        "uv_index": uv_interpolated.values[:n].tolist(),
        "wind_speed": wind_speed[:n].tolist(),
    }


def export_csv(data: dict, output_path: Path) -> None:
    """Write weather data to a semicolon-delimited CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Timestamp", "Temperature (°C)", "UV Index", "Wind Speed (km/h)"])
        for i, ts in enumerate(data["timestamps"]):
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                f"{data['temperature'][i]:.2f}".replace(".", ","),
                f"{data['uv_index'][i]:.2f}".replace(".", ","),
                f"{data['wind_speed'][i]:.2f}".replace(".", ","),
            ])

    logger.info(f"Exported {len(data['timestamps'])} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch 48h weather data from Open-Meteo.")
    parser.add_argument("--lat",    type=float, default=50.85045, help="Latitude  (default: Brussels)")
    parser.add_argument("--lon",    type=float, default=4.34878,  help="Longitude (default: Brussels)")
    parser.add_argument("--output", default=r"C:\Users\32488\Documents\4de jaar\Masterproef\Dashboard\weather.csv", help="Output CSV file")
    args = parser.parse_args()

    today    = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    client = build_client()
    data   = fetch_weather(client, args.lat, args.lon, today, tomorrow)

    export_csv(data, Path(args.output))

    # Quick summary
    print(f"\nDone! {len(data['timestamps'])} rows saved to: {args.output}")
    print(f"  Temperature : {min(data['temperature']):.1f} – {max(data['temperature']):.1f} °C")
    print(f"  UV Index    : {min(data['uv_index']):.1f} – {max(data['uv_index']):.1f}")
    print(f"  Wind Speed  : {min(data['wind_speed']):.1f} – {max(data['wind_speed']):.1f} km/h")


if __name__ == "__main__":
    main()