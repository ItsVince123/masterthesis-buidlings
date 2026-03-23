"""
Fetch next 24h day-ahead electricity prices from ENTSO-E and export to CSV.
Usage: python fetch_prices.py --domain 10YBE----------2 --output prices.csv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import xmltodict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = "a13c900f-96f6-4fdf-ba15-4ce38bdd651b"
BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Resolution changed to 15-min intervals on 2025-10-01
RESOLUTION_CHANGE_DATE = datetime(2025, 10, 1)


def fetch_prices(domain: str, period_start: str, period_end: str) -> str:
    """Fetch day-ahead prices XML from ENTSO-E API."""
    params = {
        "securityToken": API_KEY,
        "documentType": "A44",
        "out_Domain": domain,
        "in_Domain": domain,
        "periodStart": period_start,
        "periodEnd": period_end,
        "contract_MarketAgreement.type": "A01",  # Day-ahead
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def parse_prices(xml_text: str) -> list[tuple[datetime, float]]:
    """Parse XML response into a list of (timestamp, price) tuples."""
    data = xmltodict.parse(xml_text)

    # ENTSO-E can return different root tags depending on endpoint/version.
    # Prefer known market document roots, then fall back to the first key.
    doc = (
        data.get("Publication_MarketDocument")
        or data.get("GL_MarketDocument")
        or data.get("Acknowledgement_MarketDocument")
        or next(iter(data.values()), {})
    )

    # Navigate to TimeSeries (handle both single and multiple)
    timeseries = doc.get("TimeSeries", [])
    if isinstance(timeseries, dict):
        timeseries = [timeseries]

    results = []

    for ts in timeseries:
        period = ts.get("Period", {})
        if isinstance(period, list):
            periods = period
        else:
            periods = [period]

        for p in periods:
            # Parse start time
            start_str = p.get("timeInterval", {}).get("start", "")
            if start_str.endswith("Z"):
                start_str = start_str[:-1] + "+00:00"
            start_dt = datetime.fromisoformat(start_str)

            # Determine interval length based on date
            naive_start = start_dt.replace(tzinfo=None)
            minutes = 15 if naive_start >= RESOLUTION_CHANGE_DATE else 60

            # Parse price points
            points = p.get("Point", [])
            if isinstance(points, dict):
                points = [points]

            for point in points:
                position = int(point.get("position", 1))
                price = float(point.get("price.amount", 0))
                timestamp = start_dt + timedelta(minutes=minutes * (position - 1))
                results.append((timestamp, price))

    return sorted(results, key=lambda x: x[0])


def export_csv_with_flag(data: list[tuple[datetime, float, bool]], output_path: Path) -> Path:
    """Write price data to CSV, overwriting the target file if it exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_csv(target_path: Path) -> None:
        with target_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Timestamp (Brussels)", "Price (EUR/MWh)", "Below 48h Avg"])
            brussels = ZoneInfo("Europe/Brussels")
            for ts, price, is_below in data:
                ts_local = ts.astimezone(brussels) if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(brussels)
                writer.writerow([
                    ts_local.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{price:.2f}".replace(".", ","),
                    "true" if is_below else "false"
                ])

    try:
        _write_csv(output_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite '{output_path}'. The file is likely open in another program (e.g. Excel). "
            "Close it and run again."
        ) from exc

    logger.info(f"Exported {len(data)} rows to {output_path}")
    return output_path


def get_flagged_next_day_prices(
    domain: str = "10YBE----------2",
    reference_time: datetime | None = None,
) -> tuple[list[tuple[datetime, float, bool]], float]:
    """Return previous+next day 15min prices with a below-48h-average flag.

    Uses Brussels local days:
    - previous day (today local)
    - next day (tomorrow local)
    """
    brussels = ZoneInfo("Europe/Brussels")
    now = reference_time.astimezone(brussels) if reference_time else datetime.now(brussels)

    today_local = datetime(now.year, now.month, now.day, tzinfo=brussels)
    next_start = today_local + timedelta(days=1)
    next_end = next_start + timedelta(days=1)
    prev_start = today_local
    prev_end = next_start

    logger.info(f"Fetching prices for previous day: {prev_start.date()} | Domain: {domain}")
    logger.info(f"Fetching prices for next day: {next_start.date()} | Domain: {domain}")

    prev_start_utc = prev_start.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")
    prev_end_utc = prev_end.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")
    next_start_utc = next_start.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")
    next_end_utc = next_end.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")

    with ThreadPoolExecutor(max_workers=2) as executor:
        prev_future = executor.submit(fetch_prices, domain, prev_start_utc, prev_end_utc)
        next_future = executor.submit(fetch_prices, domain, next_start_utc, next_end_utc)
        prev_xml = prev_future.result()
        next_xml = next_future.result()

    prev_data = parse_prices(prev_xml)
    next_data = parse_prices(next_xml)

    if not prev_data or not next_data:
        raise ValueError("No complete 48h price data found in the response.")

    all_data = sorted(prev_data + next_data, key=lambda x: x[0])
    all_prices = [price for _, price in all_data]
    avg_48h = sum(all_prices) / len(all_prices)

    flagged_48h = []
    for ts, price in all_data:
        flagged_48h.append((ts, price, price < avg_48h))

    return flagged_48h, avg_48h


def fetch_and_save_prices(
    domain: str = "10YBE----------2",
    output_filename: str = "prices.csv",
    output_dir: Path | None = None,
    reference_time: datetime | None = None,
) -> tuple[Path, float]:
    """Convenience function for other modules: fetch, flag, and save CSV."""
    data, avg_48h = get_flagged_next_day_prices(domain=domain, reference_time=reference_time)

    if output_dir is None:
        output_dir = Path(r"C:/Users/32488/Documents/4de jaar/Masterproef/Dashboard")

    output_path = output_dir / Path(output_filename).name
    saved_path = export_csv_with_flag(data, output_path)
    return saved_path, avg_48h


def main():
    parser = argparse.ArgumentParser(description="Fetch next 24h electricity prices from ENTSO-E.")
    parser.add_argument("--domain", default="10YBE----------2", help="EIC bidding zone code (default: Belgium)")
    parser.add_argument("--output", default="prices.csv", help="Output CSV file name (default: prices.csv)")
    args = parser.parse_args()

    try:
        saved_path, avg_48h = fetch_and_save_prices(domain=args.domain, output_filename=args.output)
    except ValueError as exc:
        logger.error(str(exc))
        return
    except PermissionError as exc:
        logger.error(str(exc))
        return

    logger.info(f"48h average price: {avg_48h:.2f} EUR/MWh")
    print(f"\nDone! Saved to: {saved_path}")


if __name__ == "__main__":
    main()