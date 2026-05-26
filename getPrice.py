"""
Fetch day-ahead electricity prices from ENTSO-E and export to CSV.

Usage::

    python getPrice.py --domain 10YBE----------2 --output prices.csv

The API key is read from the ``ENTSOE_API_KEY`` environment variable (with a
fallback in ``settings.py``).
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import xmltodict

from settings import (
    DASHBOARD_DIR, ENTSOE_BASE_URL, ENTSOE_DOMAIN, LOCAL_TZ, get_entsoe_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ENTSO-E switched to 15-minute resolution on this date.
_RESOLUTION_CHANGE = datetime(2025, 10, 1)


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def fetch_prices(domain: str, period_start: str, period_end: str) -> str:
    """Fetch day-ahead prices XML from the ENTSO-E Transparency Platform."""
    params = {
        "securityToken": get_entsoe_api_key(),
        "documentType": "A44",
        "out_Domain": domain,
        "in_Domain": domain,
        "periodStart": period_start,
        "periodEnd": period_end,
        "contract_MarketAgreement.type": "A01",
    }
    resp = requests.get(ENTSOE_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_prices(xml_text: str) -> list[tuple[datetime, float]]:
    """Parse the ENTSO-E XML response into ``(timestamp, price)`` tuples."""
    data = xmltodict.parse(xml_text)
    doc = (
        data.get("Publication_MarketDocument")
        or data.get("GL_MarketDocument")
        or data.get("Acknowledgement_MarketDocument")
        or next(iter(data.values()), {})
    )

    timeseries = doc.get("TimeSeries", [])
    if isinstance(timeseries, dict):
        timeseries = [timeseries]

    results: list[tuple[datetime, float]] = []
    for ts in timeseries:
        period = ts.get("Period", {})
        periods = period if isinstance(period, list) else [period]

        for p in periods:
            start_str = p.get("timeInterval", {}).get("start", "")
            if start_str.endswith("Z"):
                start_str = start_str[:-1] + "+00:00"
            start_dt = datetime.fromisoformat(start_str)

            # Determine interval length based on the resolution-change date.
            naive = start_dt.replace(tzinfo=None)
            minutes = 15 if naive >= _RESOLUTION_CHANGE else 60

            points = p.get("Point", [])
            if isinstance(points, dict):
                points = [points]

            for point in points:
                pos = int(point.get("position", 1))
                price = float(point.get("price.amount", 0))
                ts_out = start_dt + timedelta(minutes=minutes * (pos - 1))
                results.append((ts_out, price))

    return sorted(results, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv_with_flag(
    data: list[tuple[datetime, float, bool]],
    output_path: Path,
) -> Path:
    """Write price data to a semicolon-delimited CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "Timestamp (Brussels)", "Price (EUR/MWh)", "Below 48h Avg",
            ])
            for ts, price, is_below in data:
                ts_local = (
                    ts.astimezone(LOCAL_TZ) if ts.tzinfo
                    else ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(LOCAL_TZ)
                )
                writer.writerow([
                    ts_local.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{price:.2f}".replace(".", ","),
                    "true" if is_below else "false",
                ])
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite '{output_path}'. Close the file in other "
            "programs and try again."
        ) from exc

    logger.info("Exported %d rows to %s", len(data), output_path)
    return output_path


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def get_flagged_next_day_prices(
    domain: str = ENTSOE_DOMAIN,
    reference_time: datetime | None = None,
) -> tuple[list[tuple[datetime, float, bool]], float]:
    """Return 48 h prices with a *below-average* flag.

    Fetches today + tomorrow (Brussels local) and flags entries whose price
    falls below the combined 48 h average.

    WHY 48 H?
    The dashboard shows today's data + tomorrow's day-ahead auction results.
    By computing the average over both days the colour-coding (green = cheap
    relative to the full two-day window, red = expensive) gives a useful
    visual indication of whether NOW is a good time to consume.

    BELOW-AVERAGE FLAG
    Each price entry is flagged True when price < 48h_average.
    The dashboard colours these green (cheap hours to shift load into).
    """
    now = (
        reference_time.astimezone(LOCAL_TZ) if reference_time
        else datetime.now(LOCAL_TZ)
    )

    today = datetime(now.year, now.month, now.day, tzinfo=LOCAL_TZ)
    tomorrow = today + timedelta(days=1)
    day_after = tomorrow + timedelta(days=1)

    logger.info(
        "Fetching prices: %s and %s | Domain: %s",
        today.date(), tomorrow.date(), domain,
    )

    utc = ZoneInfo("UTC")
    prev_s = today.astimezone(utc).strftime("%Y%m%d%H%M")
    prev_e = tomorrow.astimezone(utc).strftime("%Y%m%d%H%M")
    next_s = prev_e
    next_e = day_after.astimezone(utc).strftime("%Y%m%d%H%M")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(fetch_prices, domain, prev_s, prev_e)
        f2 = pool.submit(fetch_prices, domain, next_s, next_e)
        try:
            prev_data = parse_prices(f1.result())
        except Exception as exc:
            logger.warning("Could not fetch today's prices: %s", exc)
            prev_data = []
        try:
            next_data = parse_prices(f2.result())
        except Exception as exc:
            logger.warning("Could not fetch tomorrow's prices: %s", exc)
            next_data = []

    if not prev_data and not next_data:
        raise ValueError("No price data available for today or tomorrow.")

    combined = sorted(prev_data + next_data, key=lambda x: x[0])
    avg = sum(p for _, p in combined) / len(combined)
    flagged = [(ts, price, price < avg) for ts, price in combined]

    return flagged, avg


def fetch_and_save_prices(
    domain: str = ENTSOE_DOMAIN,
    output_filename: str = "prices.csv",
    output_dir: Path | None = None,
    reference_time: datetime | None = None,
) -> tuple[Path, float]:
    """Convenience wrapper: fetch, flag, and save day-ahead prices to CSV."""
    data, avg = get_flagged_next_day_prices(
        domain=domain, reference_time=reference_time,
    )
    out_dir = output_dir if output_dir is not None else DASHBOARD_DIR
    path = out_dir / Path(output_filename).name
    saved = export_csv_with_flag(data, path)
    return saved, avg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch next-day electricity prices from ENTSO-E.",
    )
    parser.add_argument(
        "--domain", default=ENTSOE_DOMAIN,
        help="EIC bidding zone code (default: Belgium)",
    )
    parser.add_argument(
        "--output", default="prices.csv",
        help="Output CSV file name",
    )
    args = parser.parse_args()

    try:
        saved, avg = fetch_and_save_prices(
            domain=args.domain, output_filename=args.output,
        )
    except (ValueError, PermissionError) as exc:
        logger.error(str(exc))
        return

    logger.info("48 h average price: %.2f EUR/MWh", avg)
    print(f"\nDone! Saved to: {saved}")


if __name__ == "__main__":
    main()
