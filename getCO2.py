"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  Grid CO2 intensity calculator.                                  ║
║  Fetches actual generation mix per fuel type from ENTSO-E and   ║
║  converts to CO2 intensity using IPCC lifecycle emission factors.║
╚══════════════════════════════════════════════════════════════════╝

Fetch grid CO2 intensity (gCO2eq/kWh) for the Belgian zone.

Data source: ENTSO-E Transparency Platform — Actual Generation per Type
(document type ``A75``).  The generation mix is converted to CO2 intensity
using standard lifecycle emission factors per fuel.

Usage::

    # In code
    from getCO2 import get_hourly_co2, load_co2_csv

    # CLI — fetch and cache a full year
    python getCO2.py --year 2022
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import xmltodict

from settings import (
    DASHBOARD_DIR,
    ENTSOE_API_KEY,
    ENTSOE_BASE_URL,
    ENTSOE_DOMAIN,
    LOCAL_TZ,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifecycle CO2 emission factors (gCO2eq per kWh_elec)
# Source: IPCC 2014 median values & EU reference documents
# ---------------------------------------------------------------------------
_CO2_FACTORS: dict[str, float] = {
    # ENTSO-E PsrType code → gCO2eq/kWh
    "B01": 820,    # Biomass (conservative — can be 0 if sustainable)
    "B02": 230,    # Fossil Brown coal / Lignite
    "B03": 0,      # Fossil Coal-derived gas  (rarely used, mapped to coal)
    "B04": 490,    # Fossil Gas
    "B05": 820,    # Fossil Hard coal
    "B06": 780,    # Fossil Oil
    "B07": 380,    # Fossil Oil shale
    "B08": 490,    # Fossil Peat
    "B09": 24,     # Geothermal
    "B10": 4,      # Hydro Pumped Storage (lifecycle emissions only)
    "B11": 4,      # Hydro Run-of-river
    "B12": 4,      # Hydro Water Reservoir
    "B13": 0,      # Marine
    "B14": 12,     # Nuclear
    "B15": 0,      # Other renewable
    "B16": 45,     # Solar
    "B17": 0,      # Waste
    "B18": 11,     # Wind Offshore
    "B19": 11,     # Wind Onshore
    "B20": 0,      # Other
}

# Fallback when no API data is available (Belgian grid average ~150-180)
FALLBACK_CO2_GRAMS_PER_KWH = 170.0

# Direct combustion CO2 factor for natural gas (HHV basis, combustion only)
# Source: IPCC / IEA — approx 202 gCO2/kWh_gas (56.1 kgCO2/GJ × 3.6 MJ/kWh)
GAS_CO2_G_PER_KWH_GAS = 202.0

CO2_CSV = DASHBOARD_DIR / "co2_intensity.csv"


# ---------------------------------------------------------------------------
# ENTSO-E generation mix fetch
# ---------------------------------------------------------------------------

def _fetch_generation_xml(
    domain: str,
    period_start: str,
    period_end: str,
) -> str:
    """Fetch Actual Generation Per Type (A75) from ENTSO-E."""
    params = {
        "securityToken": ENTSOE_API_KEY,
        "documentType": "A75",
        "processType": "A16",  # Realised
        "in_Domain": domain,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    resp = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def _parse_generation(xml_text: str) -> dict[datetime, dict[str, float]]:
    """Parse generation XML into ``{timestamp: {psr_type: mw}}``.

    Returns a dict mapping each timestamp to a fuel-type → MW dict.
    """
    data = xmltodict.parse(xml_text)
    doc = (
        data.get("GL_MarketDocument")
        or data.get("Publication_MarketDocument")
        or next(iter(data.values()), {})
    )

    timeseries = doc.get("TimeSeries", [])
    if isinstance(timeseries, dict):
        timeseries = [timeseries]

    result: dict[datetime, dict[str, float]] = {}

    for ts in timeseries:
        psr_type = (
            ts.get("MktPSRType", {}).get("psrType", "B20")
        )
        period = ts.get("Period", {})
        periods = period if isinstance(period, list) else [period]

        for p in periods:
            start_str = p.get("timeInterval", {}).get("start", "")
            if start_str.endswith("Z"):
                start_str = start_str[:-1] + "+00:00"
            start_dt = datetime.fromisoformat(start_str)

            res_str = p.get("resolution", "PT60M")
            if "15M" in res_str:
                minutes = 15
            else:
                minutes = 60

            points = p.get("Point", [])
            if isinstance(points, dict):
                points = [points]

            for point in points:
                pos = int(point.get("position", 1))
                mw = float(point.get("quantity", 0))
                ts_out = start_dt + timedelta(minutes=minutes * (pos - 1))
                result.setdefault(ts_out, {})[psr_type] = mw

    return result


def _generation_to_co2(gen_mix: dict[str, float]) -> float:
    """Convert a fuel mix dict ``{psr_type: MW}`` to gCO2eq/kWh."""
    total_mw = sum(gen_mix.values())
    if total_mw <= 0:
        return FALLBACK_CO2_GRAMS_PER_KWH
    weighted = sum(
        _CO2_FACTORS.get(psr, 0) * mw for psr, mw in gen_mix.items()
    )
    return weighted / total_mw


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def get_hourly_co2(
    date: datetime,
    domain: str = ENTSOE_DOMAIN,
) -> list[tuple[datetime, float]]:
    """Fetch hourly CO2 intensity (gCO2eq/kWh) for a single day.

    Returns sorted ``[(timestamp, gCO2/kWh), …]``.
    """
    utc = ZoneInfo("UTC")
    day_start = datetime(
        date.year, date.month, date.day, tzinfo=LOCAL_TZ,
    ).astimezone(utc)
    day_end = day_start + timedelta(days=1)

    start_s = day_start.strftime("%Y%m%d%H%M")
    end_s = day_end.strftime("%Y%m%d%H%M")

    xml = _fetch_generation_xml(domain, start_s, end_s)
    gen_data = _parse_generation(xml)

    if not gen_data:
        logger.warning("No generation data for %s — using fallback", date)
        return [
            (day_start + timedelta(hours=h), FALLBACK_CO2_GRAMS_PER_KWH)
            for h in range(24)
        ]

    results = [
        (ts.astimezone(LOCAL_TZ), _generation_to_co2(mix))
        for ts, mix in sorted(gen_data.items())
    ]
    return results


# ---------------------------------------------------------------------------
# CSV cache — fetch once, reuse for historical analysis
# ---------------------------------------------------------------------------

def fetch_and_save_year(
    year: int,
    domain: str = ENTSOE_DOMAIN,
    output: Path = CO2_CSV,
) -> Path:
    """Fetch CO2 intensity for every day of *year* and save to CSV.

    The CSV has columns ``Timestamp;CO2_grams_per_kWh`` (semicolon-
    delimited, Brussels local time) matching the project convention.
    """
    logger.info("Fetching CO2 intensity for %d — this may take a while…", year)
    all_rows: list[tuple[str, float]] = []

    day = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    while day < end:
        try:
            hourly = get_hourly_co2(day, domain)
            for ts, co2 in hourly:
                ts_local = ts.astimezone(LOCAL_TZ)
                all_rows.append((
                    ts_local.strftime("%Y-%m-%d %H:%M"),
                    round(co2, 1),
                ))
        except Exception as exc:
            logger.warning("CO2 fetch failed for %s: %s — filling fallback", day, exc)
            for h in range(24):
                ts_local = datetime(day.year, day.month, day.day, h, tzinfo=LOCAL_TZ)
                all_rows.append((
                    ts_local.strftime("%Y-%m-%d %H:%M"),
                    FALLBACK_CO2_GRAMS_PER_KWH,
                ))
        day += timedelta(days=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Timestamp", "CO2_grams_per_kWh"])
        for row in all_rows:
            w.writerow(row)

    logger.info("Saved %d rows to %s", len(all_rows), output)
    return output


def load_co2_csv(
    path: Path = CO2_CSV,
) -> dict[str, list[float]]:
    """Load cached CO2 CSV into ``{day_key: [co2_h0, co2_h1, …]}``.

    Day key format: ``D/MM/YYYY`` to match DATA.csv conventions.
    Falls back to an empty dict if the file doesn't exist.
    """
    if not path.exists():
        return {}

    days: dict[str, list[tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f, delimiter=";"):
            ts = raw["Timestamp"].strip()
            # Parse "2022-01-01 00:00" format
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            # Build day key matching DATA.csv: D/MM/YYYY
            day_key = f"{dt.day}/{dt.month:02d}/{dt.year}"
            hour = dt.hour
            co2 = float(raw["CO2_grams_per_kWh"])
            days.setdefault(day_key, []).append((hour, co2))

    result: dict[str, list[float]] = {}
    for k, pairs in days.items():
        pairs.sort(key=lambda x: x[0])
        result[k] = [v for _, v in pairs]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Fetch hourly grid CO2 intensity from ENTSO-E generation data.",
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Year to fetch (default: 2022)",
    )
    parser.add_argument(
        "--domain", default=ENTSOE_DOMAIN,
        help="ENTSO-E bidding zone (default: Belgium)",
    )
    parser.add_argument(
        "--output", default=str(CO2_CSV),
        help="Output CSV path",
    )
    args = parser.parse_args()

    saved = fetch_and_save_year(
        args.year, args.domain, Path(args.output),
    )
    print(f"\nDone! Saved to: {saved}")


if __name__ == "__main__":
    main()
