"""Predict solar yield from weather CSV and export to predict.csv.

The model uses UV index as a proxy for irradiance and applies a small temperature
derating factor for panel efficiency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Brussels")


@dataclass
class WeatherRow:
	timestamp: str
	temperature_c: float
	uv_index: float
	wind_kmh: float


def parse_float_local(value: str) -> float:
	"""Parse numbers that may use comma as decimal separator."""
	return float(value.strip().replace(",", "."))


def normalize_local_timestamp(timestamp: str) -> str:
	"""Normalize timestamp into Brussels local format: YYYY-mm-dd HH:MM:SS."""
	ts = timestamp.strip()
	if not ts:
		return ts

	try:
		# If timezone is provided, convert to local timezone.
		if ts.endswith("Z"):
			dt = datetime.fromisoformat(ts[:-1] + "+00:00").astimezone(LOCAL_TZ)
		elif "T" in ts and ("+" in ts[10:] or "-" in ts[10:]):
			dt = datetime.fromisoformat(ts).astimezone(LOCAL_TZ)
		else:
			# Naive timestamps are assumed to already be Brussels local.
			dt = datetime.strptime(ts.replace("T", " "), "%Y-%m-%d %H:%M:%S")
		return dt.strftime("%Y-%m-%d %H:%M:%S")
	except ValueError:
		# Leave as-is if format is unexpected; downstream matching stays deterministic.
		return ts


def read_weather_csv(path: Path) -> list[WeatherRow]:
	rows: list[WeatherRow] = []
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter=";")
		for item in reader:
			rows.append(
				WeatherRow(
					timestamp=normalize_local_timestamp(item["Timestamp"]),
					temperature_c=parse_float_local(item["Temperature (°C)"]),
					uv_index=parse_float_local(item["UV Index"]),
					wind_kmh=parse_float_local(item["Wind Speed (km/h)"]),
				)
			)
	return rows


def predict_power_kw(row: WeatherRow, capacity_kwp: float) -> float:
	"""Estimate instantaneous PV power (kW) from weather inputs.

	Assumptions:
	- UV index mapped to irradiance proxy via normalized factor uv/8.
	- Temperature derating of -0.4% / degC above 25C.
	"""
	irradiance_factor = max(0.0, min(row.uv_index / 8.0, 1.0))
	temp_derate = 1.0 - max(0.0, row.temperature_c - 25.0) * 0.004
	temp_derate = max(0.75, temp_derate)

	power_kw = capacity_kwp * irradiance_factor * temp_derate
	return max(0.0, power_kw)


def export_predictions(
	weather_rows: list[WeatherRow],
	output_path: Path,
	capacity_kwp: float,
	interval_minutes: int = 15,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter=";")
		writer.writerow(["Timestamp", "Predicted Power (kW)", "Predicted Yield (kWh)"])

		for row in weather_rows:
			power_kw = predict_power_kw(row, capacity_kwp)
			yield_kwh = power_kw * (interval_minutes / 60.0)
			writer.writerow(
				[
					row.timestamp,
					f"{power_kw:.2f}".replace(".", ","),
					f"{yield_kwh:.3f}".replace(".", ","),
				]
			)


def main() -> None:
	dashboard_dir = Path(r"C:/Users/32488/Documents/4de jaar/Masterproef/Dashboard")

	parser = argparse.ArgumentParser(description="Predict solar yield from weather.csv")
	parser.add_argument("--weather", default=str(dashboard_dir / "weather.csv"), help="Input weather CSV path")
	parser.add_argument("--output", default=str(dashboard_dir / "predict.csv"), help="Output prediction CSV path")
	parser.add_argument("--capacity-kwp", type=float, default=100.0, help="Installed solar capacity (kWp)")
	args = parser.parse_args()

	weather_path = Path(args.weather)
	output_path = Path(args.output)

	weather_rows = read_weather_csv(weather_path)
	if not weather_rows:
		raise ValueError("No weather rows found in input CSV.")

	export_predictions(weather_rows, output_path, args.capacity_kwp)
	print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
	main()
