"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  Data layer — fetches, caches, and schedules all external data.  ║
║  The UI (dashboard.py) only reads from DataManager; it never     ║
║  touches the network or CSV files directly.                      ║
╚══════════════════════════════════════════════════════════════════╝

Data layer — fetching, caching, and scheduled refresh of live data.

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
    """Return the start of the current 15-minute interval (tz-aware Brussels).

    Example: called at 14:23:47 → returns 14:15:00.
    The dashboard and LP solver use this as the 'now' anchor for all
    forecasts and graph indices.
    """
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
    """Fetches, caches, and schedules all external live data.

    DESIGN: The DataManager is the single source of truth for all live data.
    The dashboard UI and LP solver read from it; they never call external
    APIs or parse CSVs themselves.

    CACHING STRATEGY
    ----------------
    All external data is cached in memory after the first load.
    Refreshes are rate-limited using ``_*_next`` timestamps so we don't
    hammer the APIs on every tick.

    DATA SOURCES
    ------------
    prices.csv   ← fetched from ENTSO-E Transparency Platform (getPrice.py)
    predict.csv  ← derived from weather.csv via the solar prediction model (predict.py)
    weather.csv  ← fetched from Open-Meteo API (getWeather.py)

    DAILY PIPELINE
    --------------
    At DAILY_FETCH_HOUR (default 14:00), the pipeline runs:
      1. Fetch fresh weather data for today & tomorrow
      2. Re-run solar yield prediction
      3. Fetch tomorrow's day-ahead electricity prices
    This ensures the LP always has a 24–48 h price forecast.
    """

    def __init__(self):
        # ── Price data (from prices.csv) ────────────────────────────
        # Each row: (timezone-aware datetime, EUR/MWh, is_below_48h_avg bool)
        self.price_rows: list = []
        self.avg_48h: float | None = None    # rolling 48 h average for colour coding

        # ── Solar predictions (from predict.csv) ────────────────────
        # Key: "YYYY-MM-DD HH:MM:SS" string → (power_kW, yield_kWh)
        self.predictions: dict[str, tuple[float, float]] = {}

        # ── UV index (from weather.csv) ─────────────────────────────
        # Key: "YYYY-MM-DD HH:MM:SS" string → UV float (0–11 scale)
        self.uv_data: dict[str, float] = {}
        # ── Outside temperature (from weather.csv) ─────────────────────
        # Key: "YYYY-MM-DD HH:MM:SS" string → temperature in °C
        self.temp_data: dict[str, float] = {}
        # ── Current-slot scalar values ──────────────────────────────
        # These are re-derived each tick by update_slot_values()
        self.current_price: float | None = None      # EUR/MWh at current 15-min slot
        self.current_power_kw: float | None = None   # predicted PV output [kW]
        self.current_yield_kwh: float | None = None  # predicted PV yield this interval [kWh]
        self.current_uv: float | None = None         # UV index at current slot

        # ── Rate-limiting timestamps ────────────────────────────────
        # Each source refreshes at most once per 15-min interval.
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
        """Build an array of EUR/kWh prices for the next *horizon* 15-min steps.

        The LP solver needs prices in EUR/kWh (not EUR/MWh) aligned to
        15-minute intervals starting from *slot*.

        If no price data is loaded yet, returns a flat 0.10 EUR/kWh fallback.
        Uses forward-fill: each interval gets the most recent known price.
        """
        if not self.price_rows:
            return np.full(horizon, 0.10)          # safe fallback
        # Convert from EUR/MWh → EUR/kWh, normalise timestamps to minute precision
        timeline = sorted(
            [
                (ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0),
                 price / 1000.0)                   # EUR/MWh ÷ 1000 = EUR/kWh
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
        dt_hours: float = INTERVAL_MINUTES / 60.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (base_load_kwh, solar_kwh) arrays for the next *horizon* MPC steps.

        BASE LOAD MODEL
        ---------------
        Building consumption is modelled as a flat constant at *base_kw* during
        night hours (outside 06:00–18:00) and a sinusoidal ramp up to *peak_kw*
        during daytime.  Both values are configured in the Building tab of the
        MPC Settings dialog (smpc.building.base_load_kw / peak_load_kw).

        Set base_load_kw = peak_load_kw to use a flat constant profile.
        Set both to 0 if you have no load to model (optimiser only covers HVAC
        and hot-water heating driven by the thermal sub-model).

        SOLAR
        -----
        Solar yield is read from predict.csv (output of predict.py), which stores
        kWh per INTERVAL_MINUTES-minute slot.  When dt_hours differs from
        INTERVAL_MINUTES/60 (e.g. 1-hour MPC steps), sub-slots within each MPC
        step are summed so that the returned array is in kWh per MPC step.
        """
        dt_min = dt_hours * 60.0                          # MPC step in minutes
        n_sub  = max(1, round(dt_min / INTERVAL_MINUTES)) # 15-min sub-slots per MPC step
        start_h = slot.hour + slot.minute / 60.0

        base_load = np.zeros(horizon)
        solar = np.zeros(horizon)

        for t in range(horizon):
            h = (start_h + t * dt_hours) % 24            # fractional hour at step start
            if 6.0 <= h <= 18.0:
                kw = base_kw + (peak_kw - base_kw) * max(
                    0.0, np.sin(np.pi * (h - 6.0) / 12.0),
                )
            else:
                kw = base_kw
            base_load[t] = kw * dt_hours                 # kWh per MPC step

            # Aggregate sub-slot solar kWh into one MPC-step total
            kwh_sum = 0.0
            for s in range(n_sub):
                offset_min = int(t * dt_min + s * INTERVAL_MINUTES)
                key = (slot + timedelta(minutes=offset_min)).strftime(
                    "%Y-%m-%d %H:%M:%S",
                )
                pred = self.predictions.get(key)
                if pred:
                    kwh_sum += max(0.0, pred[1])          # kWh per 15-min sub-slot
            solar[t] = kwh_sum                            # kWh per MPC step

        return base_load, solar

    def build_outside_temp(
        self, slot: datetime, horizon: int,
        dt_hours: float = INTERVAL_MINUTES / 60.0,
    ) -> np.ndarray | None:
        """Return outside temperature [°C] for the next *horizon* MPC steps.

        Reads from the cached weather.csv temperature data.  Returns ``None``
        if no weather data is loaded so the thermal model can fall back to
        the fixed cooling rate.

        When dt_hours matches INTERVAL_MINUTES/60 (the default), each MPC
        step maps directly to one weather record.  For longer steps (e.g.
        1-hour MPC with 15-min weather data), the temperature is taken from
        the first sub-slot in each MPC step (nearest-sample, not averaged).
        """
        if not self.temp_data:
            return None
        dt_min = dt_hours * 60.0
        arr = np.zeros(horizon)
        found_any = False
        for t in range(horizon):
            offset_min = int(round(t * dt_min))
            key = (slot + timedelta(minutes=offset_min)).strftime(
                "%Y-%m-%d %H:%M:%S",
            )
            val = self.temp_data.get(key)
            if val is not None:
                arr[t] = val
                found_any = True
            elif found_any and t > 0:
                arr[t] = arr[t - 1]   # forward-fill
            else:
                arr[t] = 10.0         # safe fallback if nothing loaded yet
        return arr if found_any else None

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
            loaded_uv: dict = {}
            loaded_temp: dict = {}
            with WEATHER_CSV.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    ts = row.get("Timestamp", "").strip()
                    if ts:
                        loaded_uv[ts] = float(
                            row.get("UV Index", "0").replace(",", "."),
                        )
                        temp_str = row.get("Temperature (°C)", "").strip()
                        if temp_str:
                            loaded_temp[ts] = float(temp_str.replace(",", "."))
            self.uv_data = loaded_uv
            self.temp_data = loaded_temp
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
