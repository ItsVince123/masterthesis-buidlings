"""
generate_future_prices.py
─────────────────────────
Generates a synthetic one-year CSV of 15-minute day-ahead electricity prices
that reflects a grid with ~5 years more renewable penetration than today.

Key modelling choices
─────────────────────
* Lower average base price  — more zero-marginal-cost generation suppresses
  the market clearing price (merit-order effect).
* Stronger duck curve       — deep midday solar valley, sharper evening ramp.
* More negative prices      — ~10 % of spring/autumn midday steps go negative
  (wind + solar surplus during low-demand periods).
* Higher day-to-day volatility — wind forecast errors drive larger swings.
* Seasonal patterns         — summer: solar-dominated (midday cheap, evening peak);
                               winter: wind-dominated (less predictable, higher avg).
* Weekend demand reduction  — ~15 % lower base load → lower prices.
* Autocorrelation           — prices within a day are correlated (same weather).

Output format matches prices.csv:
    Timestamp (Brussels);Price (EUR/MWh);Below 48h Avg
    2031-01-01 00:00:00;87,42;false
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_FILE   = Path(__file__).parent / "prices_future.csv"
YEAR          = 2025          # Synthetic year
STEPS_PER_DAY = 96            # 15-min intervals
DT_MINUTES    = 15
SEED          = 42

# Base price statistics (EUR/MWh)
BASE_MEAN_EUR_MWH  =  65.0   # Lower than today (~100-120) due to more renewables
BASE_STD_DAY       =  28.0   # Day-to-day volatility (weather-driven)

# Seasonal adjustment (additive, EUR/MWh): positive = more expensive that season
SEASONAL_AMPLITUDE =  18.0   # Winter peak vs summer trough
# Winter peaks due to heating demand + less solar; summer dip from solar surplus
# Phase: peak in January, trough in July

# Duck-curve intraday profile (additive, EUR/MWh)
# Represents the typical shape relative to the daily mean
DUCK_MORNING_RAMP  =  30.0   # Morning peak (07-09h) above daily mean
DUCK_MIDDAY_DIP    = -35.0   # Solar valley (11-15h) below daily mean
DUCK_EVENING_RAMP  =  45.0   # Evening peak (18-21h) above daily mean
DUCK_NIGHT_BASE    = -15.0   # Night-time below daily mean

# Negative price probability during solar peak hours in spring/autumn
NEG_PRICE_PROB_SPRING_AUTUMN = 0.12   # 12 % of midday steps in Mar-May, Sep-Oct
NEG_PRICE_PROB_SUMMER        = 0.06   # 6 % in June-August (more curtailment)
NEG_PRICE_FLOOR_EUR_MWH      = -55.0  # Deepest plausible negative price

# Price spike: low-wind / high-demand events (winter evening peaks)
SPIKE_PROB_WINTER_EVENING     = 0.04   # 4 % of winter evening steps
SPIKE_MAGNITUDE_EUR_MWH       = 180.0  # Additional premium during a spike

# Weekend demand discount
WEEKEND_DISCOUNT_EUR_MWH = 12.0

# ── Helpers ──────────────────────────────────────────────────────────────────

def _day_of_year(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5   # Saturday=5, Sunday=6


def _seasonal_offset(doy: int) -> float:
    """
    Cosine seasonal adjustment.
    Peak (positive, expensive) around day 15 (mid-Jan),
    trough (negative, cheap) around day 196 (mid-Jul).
    """
    return SEASONAL_AMPLITUDE * math.cos(2 * math.pi * (doy - 15) / 365)


def _duck_curve(hour: float) -> float:
    """
    Piecewise duck-curve intraday shape as a function of the hour (0-24).
    Returns additive offset in EUR/MWh relative to the day's mean price.
    """
    if 0.0 <= hour < 6.0:
        # Night: flat low
        return DUCK_NIGHT_BASE
    elif 6.0 <= hour < 9.0:
        # Morning ramp up
        t = (hour - 6.0) / 3.0
        return DUCK_NIGHT_BASE + (DUCK_MORNING_RAMP - DUCK_NIGHT_BASE) * t
    elif 9.0 <= hour < 11.0:
        # Morning peak coasting → start of solar dip
        t = (hour - 9.0) / 2.0
        return DUCK_MORNING_RAMP + (DUCK_MIDDAY_DIP - DUCK_MORNING_RAMP) * t
    elif 11.0 <= hour < 15.0:
        # Solar valley
        return DUCK_MIDDAY_DIP
    elif 15.0 <= hour < 18.0:
        # Afternoon ramp up (solar fading)
        t = (hour - 15.0) / 3.0
        return DUCK_MIDDAY_DIP + (DUCK_EVENING_RAMP - DUCK_MIDDAY_DIP) * t
    elif 18.0 <= hour < 21.0:
        # Evening peak
        return DUCK_EVENING_RAMP
    elif 21.0 <= hour < 24.0:
        # Evening descent back to night
        t = (hour - 21.0) / 3.0
        return DUCK_EVENING_RAMP + (DUCK_NIGHT_BASE - DUCK_EVENING_RAMP) * t
    return DUCK_NIGHT_BASE


def _solar_strength(doy: int) -> float:
    """0–1 index of how strong the solar effect is that day (1 = midsummer)."""
    return max(0.0, math.sin(math.pi * (doy - 80) / 185))   # Apr–Oct


def _is_spring_autumn(month: int) -> bool:
    return month in (3, 4, 5, 9, 10)


def _is_summer(month: int) -> bool:
    return month in (6, 7, 8)


def _is_winter(month: int) -> bool:
    return month in (11, 12, 1, 2)


# ── Main generation ──────────────────────────────────────────────────────────

def generate_prices(year: int = YEAR) -> list[tuple[datetime, float]]:
    rng = random.Random(SEED)

    start = datetime(year, 1, 1, 0, 0)
    end   = datetime(year + 1, 1, 1, 0, 0)

    records: list[tuple[datetime, float]] = []
    current = start

    # Day-level state: AR(1) wind noise that persists across the day
    wind_noise = 0.0          # persistent day-to-day component
    AR_COEFF   = 0.65         # autocorrelation between consecutive days

    while current < end:
        # ── Day-level parameters ──────────────────────────────────────────
        doy   = _day_of_year(current)
        month = current.month

        # Update wind noise (AR-1 day-to-day)
        wind_noise = AR_COEFF * wind_noise + rng.gauss(0, BASE_STD_DAY * math.sqrt(1 - AR_COEFF**2))

        day_mean = BASE_MEAN_EUR_MWH + _seasonal_offset(doy) + wind_noise
        if _is_weekend(current):
            day_mean -= WEEKEND_DISCOUNT_EUR_MWH

        solar = _solar_strength(doy)

        # ── Step-level prices ─────────────────────────────────────────────
        for step in range(STEPS_PER_DAY):
            hour = step * DT_MINUTES / 60.0   # 0.0 – 23.75
            ts   = current + timedelta(minutes=step * DT_MINUTES)

            # Intraday shape
            duck = _duck_curve(hour) * (0.7 + 0.6 * solar)   # stronger in summer

            # Step-level noise (within-day autocorrelation via smoothing)
            step_noise = rng.gauss(0, 8.0)

            price = day_mean + duck + step_noise

            # Negative price event during midday solar surplus
            midday = 10.5 <= hour <= 15.5
            if midday and solar > 0.2:
                if _is_spring_autumn(month) and rng.random() < NEG_PRICE_PROB_SPRING_AUTUMN:
                    price = rng.uniform(NEG_PRICE_FLOOR_EUR_MWH, -5.0)
                elif _is_summer(month) and rng.random() < NEG_PRICE_PROB_SUMMER:
                    price = rng.uniform(NEG_PRICE_FLOOR_EUR_MWH, 0.0)

            # Price spike: cold dark winter evening
            if _is_winter(month) and 17.5 <= hour <= 21.0:
                if rng.random() < SPIKE_PROB_WINTER_EVENING:
                    price += rng.uniform(SPIKE_MAGNITUDE_EUR_MWH * 0.5,
                                         SPIKE_MAGNITUDE_EUR_MWH)

            records.append((ts, price))

        current += timedelta(days=1)

    return records


def _compute_rolling_48h_mean(records: list[tuple[datetime, float]]) -> list[float]:
    """Rolling mean over the preceding 48 h (192 steps)."""
    prices = [p for _, p in records]
    window = 192
    means  = []
    for i in range(len(prices)):
        start_i = max(0, i - window + 1)
        means.append(sum(prices[start_i:i + 1]) / (i - start_i + 1))
    return means


def write_csv(records: list[tuple[datetime, float]], output: Path) -> None:
    rolling = _compute_rolling_48h_mean(records)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Timestamp (Brussels)", "Price (EUR/MWh)", "Below 48h Avg"])
        for (ts, price), mean_48h in zip(records, rolling):
            ts_str    = ts.strftime("%Y-%m-%d %H:%M:%S")
            price_str = f"{price:.2f}".replace(".", ",")
            below     = "true" if price < mean_48h else "false"
            writer.writerow([ts_str, price_str, below])
    print(f"Written {len(records):,} rows → {output}")


def print_stats(records: list[tuple[datetime, float]]) -> None:
    prices = [p for _, p in records]
    neg    = sum(1 for p in prices if p < 0)
    print(f"  Total steps  : {len(prices):,}")
    print(f"  Mean price   : {sum(prices)/len(prices):.1f} EUR/MWh")
    print(f"  Min price    : {min(prices):.1f} EUR/MWh")
    print(f"  Max price    : {max(prices):.1f} EUR/MWh")
    print(f"  Negative     : {neg:,} steps  ({100*neg/len(prices):.1f} %)")


if __name__ == "__main__":
    print(f"Generating synthetic {YEAR} electricity prices (renewable-heavy grid)…")
    records = generate_prices(YEAR)
    print_stats(records)
    write_csv(records, OUTPUT_FILE)
