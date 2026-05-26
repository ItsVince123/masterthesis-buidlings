"""
Predict solar yield from weather CSV and export to predict.csv.

The model uses UV index as a proxy for irradiance and applies a temperature
derating factor for panel efficiency.

Usage::

    python predict.py --weather weather.csv --output predict.csv --capacity-kwp 100
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from settings import (
    DASHBOARD_DIR, INTERVAL_MINUTES, LOCAL_TZ, SOLAR_CAPACITY_KWP,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WeatherRow:
    """One 15-minute weather observation."""
    timestamp: str
    temperature_c: float
    uv_index: float
    wind_kmh: float


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_float_local(value: str) -> float:
    """Parse a number that may use comma as decimal separator."""
    return float(value.strip().replace(",", "."))


def normalize_local_timestamp(timestamp: str) -> str:
    """Normalise a timestamp into Brussels local format: ``YYYY-mm-dd HH:MM:SS``."""
    ts = timestamp.strip()
    if not ts:
        return ts
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts[:-1] + "+00:00").astimezone(LOCAL_TZ)
        elif "T" in ts and ("+" in ts[10:] or "-" in ts[10:]):
            dt = datetime.fromisoformat(ts).astimezone(LOCAL_TZ)
        else:
            # Naive timestamps are assumed to already be Brussels local.
            dt = datetime.strptime(ts.replace("T", " "), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_weather_csv(path: Path) -> list[WeatherRow]:
    """Read a semicolon-delimited weather CSV into a list of ``WeatherRow``."""
    rows: list[WeatherRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for item in csv.DictReader(f, delimiter=";"):
            rows.append(WeatherRow(
                timestamp=normalize_local_timestamp(item["Timestamp"]),
                temperature_c=parse_float_local(item["Temperature (\u00b0C)"]),
                uv_index=parse_float_local(item["UV Index"]),
                wind_kmh=parse_float_local(item["Wind Speed (km/h)"]),
            ))
    return rows


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------

def predict_power_kw(row: WeatherRow, capacity_kwp: float) -> float:
    """Estimate instantaneous PV power (kW) from weather conditions.

    SOLAR MODEL
    -----------
    PV power is estimated using two factors:

    1. IRRADIANCE PROXY:
       UV index is used as a proxy for solar irradiance (GHI).
       Mapping: UV 8 ≈ 1000 W/m² (full sun) → factor = 1.0
                UV 0 → factor = 0.0 (no production)
       Formula: irradiance_factor = clip(UV / 8.0, 0, 1)

    2. TEMPERATURE DERATING:
       PV panels lose efficiency when hot.  Standard rating is at 25°C.
       Above 25°C: -0.4%/°C (typical silicon panel temperature coefficient)
       The derate is clamped at 75% (worst case at ~87.5°C, unrealistic).
       Formula: temp_derate = max(0.75, 1 - max(0, T - 25) × 0.004)

    Combined:
       power_kw = capacity_kwp × irradiance_factor × temp_derate

    LIMITATIONS: This is a simplified proxy model.  A production-grade
    system would use measured GHI/DNI data and a PVWatts or PVLIB model.
    """
    irradiance_factor = max(0.0, min(row.uv_index / 8.0, 1.0))
    temp_derate = 1.0 - max(0.0, row.temperature_c - 25.0) * 0.004
    temp_derate = max(0.75, temp_derate)
    return max(0.0, capacity_kwp * irradiance_factor * temp_derate)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_predictions(
    weather_rows: list[WeatherRow],
    output_path: Path,
    capacity_kwp: float,
    interval_minutes: int = INTERVAL_MINUTES,
) -> None:
    """Write solar-yield predictions to a semicolon-delimited CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Timestamp", "Predicted Power (kW)", "Predicted Yield (kWh)",
        ])
        for row in weather_rows:
            power = predict_power_kw(row, capacity_kwp)
            yld = power * (interval_minutes / 60.0)
            writer.writerow([
                row.timestamp,
                f"{power:.2f}".replace(".", ","),
                f"{yld:.3f}".replace(".", ","),
            ])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict solar yield from weather.csv",
    )
    parser.add_argument(
        "--weather", default=str(DASHBOARD_DIR / "weather.csv"),
        help="Input weather CSV path",
    )
    parser.add_argument(
        "--output", default=str(DASHBOARD_DIR / "predict.csv"),
        help="Output prediction CSV path",
    )
    parser.add_argument(
        "--capacity-kwp", type=float, default=SOLAR_CAPACITY_KWP,
        help="Installed solar capacity (kWp)",
    )
    args = parser.parse_args()

    weather_rows = read_weather_csv(Path(args.weather))
    if not weather_rows:
        raise ValueError("No weather rows found in input CSV.")

    export_predictions(weather_rows, Path(args.output), args.capacity_kwp)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
