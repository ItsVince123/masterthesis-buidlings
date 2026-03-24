"""Data layer — fetching, caching, and scheduled refresh of live data.

All external I/O (weather, price, prediction, UV) is handled here so that
the dashboard UI never touches network calls or CSV parsing directly.

Usage::

    dm = DataManager()          # one instance, owned by the dashboard
    dm.refresh_all(force=True)  # first load
    dm.tick()                   # call every second; refreshes when due

    price = dm.current_price(slot)
    power, yld = dm.solar_prediction(slot)
    uv = dm.uv_index(slot)
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import getPrice
import getWeather
import predict
from settings import (
    DASHBOARD_DIR, DAILY_FETCH_HOUR, DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
    INTERVAL_MINUTES, LOCAL_TZ, PREDICT_CSV, SOLAR_CAPACITY_KWP,
    WEATHER_CSV,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Time helpers
# ===================================================================

def current_slot() -> datetime:
    """Return the current 15-min slot as a tz-aware datetime."""
    now = datetime.now(LOCAL_TZ)
    return now.replace(
        minute=(now.minute // INTERVAL_MINUTES) * INTERVAL_MINUTES,
        second=0, microsecond=0,
    )


def next_quarter(now: datetime) -> datetime:
    """Return the start of the next 15-minute interval."""
    aligned = now.replace(
        minute=(now.minute // INTERVAL_MINUTES) * INTERVAL_MINUTES,
        second=0, microsecond=0,
    )
    return aligned + timedelta(minutes=INTERVAL_MINUTES)


# ===================================================================
# DataManager
# ===================================================================

class DataManager:
    """Fetches, caches, and schedules all external live data."""

    def __init__(self):
        # Price data
        self.price_rows: list = []            # [(tz-aware ts, EUR/MWh, flag)]
        self.avg_48h: float | None = None

        # Solar predictions  {timestamp_str: (power_kw, yield_kwh)}
        self.predictions: dict[str, tuple[float, float]] = {}

        # UV index  {timestamp_str: uv_float}
        self.uv_data: dict[str, float] = {}

        # Derived current-slot values (updated via update_slot_values)
        self.current_price: float | None = None     # EUR/MWh
        self.current_power_kw: float | None = None
        self.current_yield_kwh: float | None = None
        self.current_uv: float | None = None

        # Scheduling (next allowed refresh time)
        self._price_next: datetime | None = None
        self._predict_next: datetime | None = None
        self._weather_next: datetime | None = None
        self._pipeline_next: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_all(self, *, force: bool = False) -> None:
        """Run the full data pipeline + all refreshes."""
        self._run_pipeline(force=force)
        self.refresh_prices(force=force)
        self.refresh_predictions(force=force)
        self.refresh_weather(force=force)
        self.update_slot_values()

    def tick(self) -> None:
        """Called every second by the dashboard timer.

        Checks schedules and refreshes data sources when due.
        """
        self._maybe_run_pipeline()
        self.refresh_prices()
        self.refresh_predictions()
        self.refresh_weather()
        self.update_slot_values()

    def update_slot_values(self) -> None:
        """Derive current-slot scalar values from cached data."""
        slot = current_slot()
        slot_key = slot.strftime("%Y-%m-%d %H:%M:%S")

        # Price
        self.current_price = None
        for ts, p, _ in self.price_rows:
            if ts.astimezone(LOCAL_TZ) == slot:
                self.current_price = p
                break

        # Solar prediction
        pred = self.predictions.get(slot_key)
        if pred:
            self.current_power_kw, self.current_yield_kwh = pred
        else:
            self.current_power_kw = None
            self.current_yield_kwh = None

        # UV
        self.current_uv = self.uv_data.get(slot_key)

    # ------------------------------------------------------------------
    # Forecast builders (used by the LP solver)
    # ------------------------------------------------------------------

    def build_price_forecast(self, slot: datetime, horizon: int) -> np.ndarray:
        """Build an array of EUR/kWh prices for *horizon* 15-min steps."""
        if not self.price_rows:
            return np.full(horizon, 0.10)
        timeline = sorted(
            [
                (ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0),
                 price / 1000.0)
                for ts, price, _ in self.price_rows
            ],
            key=lambda r: r[0],
        )
        prices = np.zeros(horizon)
        for t in range(horizon):
            target = slot + timedelta(minutes=INTERVAL_MINUTES * t)
            best = timeline[0][1]
            for ts, p in timeline:
                if ts <= target:
                    best = p
                else:
                    break
            prices[t] = best
        return prices

    def build_load_and_solar(
        self, slot: datetime, horizon: int,
        base_kw: float, peak_kw: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (base_load_kwh, solar_kwh) arrays for *horizon* steps.

        base_load_kwh is the gross non-shiftable building consumption
        (sinusoidal day profile).  solar_kwh is predicted PV output.
        """
        start_h = slot.hour + slot.minute / 60.0
        base_load = np.zeros(horizon)
        solar = np.zeros(horizon)
        for t in range(horizon):
            h = (start_h + t * 0.25) % 24
            if 6 <= h <= 18:
                kw = base_kw + (peak_kw - base_kw) * max(
                    0.0, np.sin(np.pi * (h - 6) / 12),
                )
            else:
                kw = base_kw
            base_load[t] = kw * 0.25
            key = (slot + timedelta(minutes=INTERVAL_MINUTES * t)).strftime(
                "%Y-%m-%d %H:%M:%S",
            )
            pred = self.predictions.get(key)
            if pred:
                solar[t] = max(0.0, pred[1])
        return base_load, solar

    # ------------------------------------------------------------------
    # Individual data refreshes
    # ------------------------------------------------------------------

    def refresh_prices(self, *, force: bool = False) -> None:
        now = datetime.now(LOCAL_TZ)
        if not force and self._price_next and now < self._price_next:
            return
        try:
            rows, avg = getPrice.get_flagged_next_day_prices()
            self.price_rows = rows
            self.avg_48h = avg
            self._price_next = next_quarter(now)
        except Exception as exc:
            logger.warning("Price refresh failed: %s", exc)

    def refresh_predictions(self, *, force: bool = False) -> None:
        now = datetime.now(LOCAL_TZ)
        if not force and self._predict_next and now < self._predict_next:
            return
        if not PREDICT_CSV.exists():
            return
        try:
            loaded: dict = {}
            with PREDICT_CSV.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    ts = row.get("Timestamp", "").strip()
                    if not ts:
                        continue
                    pw = float(row["Predicted Power (kW)"].replace(",", "."))
                    yl = float(row["Predicted Yield (kWh)"].replace(",", "."))
                    loaded[ts] = (pw, yl)
            self.predictions = loaded
            self._predict_next = next_quarter(now)
        except Exception as exc:
            logger.warning("Predict refresh failed: %s", exc)

    def refresh_weather(self, *, force: bool = False) -> None:
        now = datetime.now(LOCAL_TZ)
        if not force and self._weather_next and now < self._weather_next:
            return
        if not WEATHER_CSV.exists():
            return
        try:
            loaded: dict = {}
            with WEATHER_CSV.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    ts = row.get("Timestamp", "").strip()
                    if ts:
                        loaded[ts] = float(
                            row.get("UV Index", "0").replace(",", "."),
                        )
            self.uv_data = loaded
            self._weather_next = next_quarter(now)
        except Exception as exc:
            logger.warning("Weather refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Daily data pipeline
    # ------------------------------------------------------------------

    def _next_pipeline_time(self, now: datetime | None = None) -> datetime:
        if now is None:
            now = datetime.now(LOCAL_TZ)
        target = now.replace(
            hour=DAILY_FETCH_HOUR, minute=0, second=0, microsecond=0,
        )
        return target if now < target else target + timedelta(days=1)

    def _run_pipeline(self, *, force: bool = False) -> None:
        now = datetime.now(LOCAL_TZ)
        if not force and self._pipeline_next and now < self._pipeline_next:
            return

        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            client = getWeather.build_client()
            data = getWeather.fetch_weather(
                client, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, today, tomorrow,
            )
            getWeather.export_csv(data, WEATHER_CSV)
        except Exception as exc:
            logger.warning("Weather pipeline: %s", exc)

        try:
            rows = predict.read_weather_csv(WEATHER_CSV)
            if rows:
                predict.export_predictions(rows, PREDICT_CSV, SOLAR_CAPACITY_KWP)
        except Exception as exc:
            logger.warning("Predict pipeline: %s", exc)

        try:
            getPrice.fetch_and_save_prices(
                output_filename="prices.csv",
                output_dir=DASHBOARD_DIR,
                reference_time=now,
            )
        except Exception as exc:
            logger.warning("Price pipeline: %s", exc)

        self._pipeline_next = self._next_pipeline_time(now)

    def _maybe_run_pipeline(self) -> None:
        now = datetime.now(LOCAL_TZ)
        if self._pipeline_next is None:
            self._pipeline_next = self._next_pipeline_time(now)
        if now >= self._pipeline_next:
            self._run_pipeline(force=True)
            self.refresh_predictions(force=True)
            self.refresh_weather(force=True)
            self.refresh_prices(force=True)
