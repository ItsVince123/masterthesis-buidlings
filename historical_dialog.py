"""Historical SMPC cost-analysis dialog.

Two modes:
1. **Live API** — Fetches real ENTSO-E prices and weather data for a past date,
   runs the SMPC optimiser for each 15-minute slot, and compares vs baseline.
2. **Historical CSV** — Loads pre-computed hourly building data (consumption,
   chiller, CHP, prices) from ``adjusted_DATA_edit.csv``, runs an hourly SMPC
   simulation, and shows a 3-way comparison: baseline vs simple optimisation
   (from the CSV) vs SMPC.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDateEdit, QDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)

import getPrice
import getWeather
import predict
from graph_renderer import draw_comparison_graph, draw_three_way_graph
from settings import (
    DEFAULT_LATITUDE, DEFAULT_LONGITUDE, ENTSOE_DOMAIN,
    HISTORICAL_CSV, LOCAL_TZ, SOLAR_CAPACITY_KWP,
)
from smpc_calculator import SMPCInputs, _get_season
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

# KPI rows displayed in the results panel
_KPI_ROWS = [
    ("baseline",        "Baseline cost (no optimisation)"),
    ("simple_opt",      "Simple optimisation (chiller shift)"),
    ("smpc",            "SMPC optimised cost"),
    ("saving_simple",   "Savings — simple vs baseline"),
    ("saving_smpc",     "Savings — SMPC vs baseline"),
    ("pct_simple",      "Savings % — simple"),
    ("pct_smpc",        "Savings % — SMPC"),
    ("slots",           "Time slots simulated"),
    ("season",          "Detected season"),
    ("ice_charged",     "Total ice bank charged"),
    ("ice_discharged",  "Total ice bank discharged"),
    ("wkk_gas",         "Total WKK gas used"),
]


# ===================================================================
# CSV data loader
# ===================================================================

def _parse_eu_float(text: str) -> float:
    """Parse a European-format number (comma = decimal separator)."""
    return float(text.strip().replace(",", "."))


def load_historical_csv(path: Path) -> dict[str, list[dict]]:
    """Load the historical building CSV, grouped by day string.

    Returns ``{day_str: [row_dict, …]}`` where *day_str* is ``"DD/MM/YYYY"``
    and each row dict contains floats for the building fields.
    """
    days: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for raw in reader:
            row = {
                "date": raw["Date"].strip(),
                "total_usage": _parse_eu_float(raw["TotalUsage"]),
                "net_usage": _parse_eu_float(raw["NetUsage"]),
                "production_wkk": _parse_eu_float(raw["ProductionWKK"]),
                "total_chiller": _parse_eu_float(raw["TotalChiller"]),
                "remaining_usage": _parse_eu_float(raw["RemainingUsage"]),
                "electricity_price": _parse_eu_float(raw["ElectricityPrice"]),
                "extra_cost": _parse_eu_float(raw["ExtraCost"]),
                "adjusted_chiller": _parse_eu_float(raw["AdjustedChiller"]),
                "adjusted_net_usage": _parse_eu_float(raw["AdjustedNetUsage"]),
                "original_cost": _parse_eu_float(raw["OriginalCost"]),
                "adjusted_cost": _parse_eu_float(raw["AdjustedCost"]),
                "savings": _parse_eu_float(raw["Savings"]),
            }
            # Day key from the "Day" column or parsed from date
            day_key = raw.get("Day", "").strip()
            if not day_key:
                day_key = row["date"].split(" ")[0]
            days.setdefault(day_key, []).append(row)
    return days


class HistoricalAnalysisDialog(QDialog):
    """Dialog that simulates a past day to compare baseline vs SMPC costs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Historical SMPC Analysis")
        self.setMinimumSize(760, 740)
        self.setStyleSheet(HISTORICAL_DIALOG_STYLE)

        self._smpc_cfg = getattr(parent, "smpc_cfg", None)
        self._smpc_calc = getattr(parent, "smpc_calculator", None)
        self._building = {
            "base_load_kw":        getattr(parent, "smpc_base_load_kw", 80.0),
            "peak_load_kw":        getattr(parent, "smpc_peak_load_kw", 200.0),
            "wkk_max_gas_m3":      getattr(parent, "smpc_wkk_max_gas_m3", 9.0),
            "heat_demand_base_kwh": getattr(parent, "smpc_heat_demand_base_kwh", 20.0),
        }

        # Historical CSV data (loaded lazily)
        self._csv_data: dict[str, list[dict]] | None = None
        self._csv_available = HISTORICAL_CSV.exists()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("Historical SMPC cost analysis")
        title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #0b3a6e;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet("font-weight: 600;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Live API (15-min slots)")
        if self._csv_available:
            self.mode_combo.addItem("Historical CSV (hourly)")
        else:
            self.mode_combo.addItem("Historical CSV (not found)")
            # Disable the CSV option if file doesn't exist
            model = self.mode_combo.model()
            if model is not None:
                item = model.item(1)
                if item is not None:
                    item.setEnabled(False)
        self.mode_combo.setMinimumWidth(200)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.mode_combo, stretch=1)
        layout.addLayout(mode_row)

        # Date selector row
        sel = QHBoxLayout()
        sel.setSpacing(10)
        lbl = QLabel("Date:")
        lbl.setStyleSheet("font-weight: 600;")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate().addDays(-1))
        self.date_edit.setMaximumDate(QDate.currentDate().addDays(-1))
        self.date_edit.setMinimumWidth(180)

        run_btn = QPushButton("Run Analysis")
        run_btn.setMinimumHeight(36)
        run_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        run_btn.clicked.connect(self._run_analysis)
        sel.addWidget(lbl)
        sel.addWidget(self.date_edit, stretch=1)
        sel.addWidget(run_btn)
        layout.addLayout(sel)

        # Status / progress
        self.status_label = QLabel("Pick a date and click 'Run Analysis'")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; font-style: italic;")
        layout.addWidget(self.status_label)

        # Graph area
        self.graph_label = QLabel("")
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setMinimumHeight(260)
        self.graph_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px;"
        )
        layout.addWidget(self.graph_label)

        # KPI card
        self._build_kpi_card(layout)

    def _build_kpi_card(self, parent_layout):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #d6dfeb;"
            " border-radius: 10px; }"
        )
        kpi_layout = QVBoxLayout(frame)
        kpi_layout.setContentsMargins(16, 12, 16, 12)
        kpi_layout.setSpacing(6)

        self.kpi_title = QLabel("Results")
        self.kpi_title.setStyleSheet(
            "font-size: 11pt; font-weight: 700; color: #0b3a6e; border: none;"
        )
        kpi_layout.addWidget(self.kpi_title)

        self.kpi_labels: dict[str, QLabel] = {}
        for key, text in _KPI_ROWS:
            row = QHBoxLayout()
            name = QLabel(text)
            name.setStyleSheet("font-weight: 600; border: none;")
            val = QLabel("--")
            val.setStyleSheet(
                "font-family: 'Consolas'; font-weight: 700;"
                " color: #0f766e; border: none;"
            )
            row.addWidget(name)
            row.addStretch()
            row.addWidget(val)
            kpi_layout.addLayout(row)
            self.kpi_labels[key] = val

        parent_layout.addWidget(frame)

    # ------------------------------------------------------------------
    # Analysis logic
    # ------------------------------------------------------------------

    def _run_analysis(self):
        """Dispatch to API or CSV mode based on combo selection."""
        mode = self.mode_combo.currentIndex()
        if mode == 0:
            self._run_api_analysis()
        elif mode == 1 and self._csv_available:
            self._run_csv_analysis()
        else:
            self.status_label.setText("Historical CSV file not found.")

    # ------------------------------------------------------------------
    # MODE 1: Live API analysis (15-min resolution)
    # ------------------------------------------------------------------

    def _run_api_analysis(self):
        """Fetch data from APIs → simulate 15-min slots → display results."""
        selected = self.date_edit.date().toPyDate()
        day_str = selected.strftime("%Y-%m-%d")

        if self._smpc_cfg is None or self._smpc_calc is None:
            self.status_label.setText("SMPC not initialised — cannot run analysis.")
            return

        # 1. Prices
        self.status_label.setText(f"Fetching prices for {day_str}\u2026")
        QApplication.processEvents()
        raw_prices = self._fetch_day_prices(selected)
        if raw_prices is None:
            return

        # 2. Weather → solar
        self.status_label.setText(f"Fetching weather for {day_str}\u2026")
        QApplication.processEvents()
        solar = self._fetch_day_solar(day_str)
        if solar is None:
            return

        # 3. Build price timeline
        timeline = self._price_timeline(raw_prices)

        # 4. Generate 96 slots
        day_start = datetime(
            selected.year, selected.month, selected.day, tzinfo=LOCAL_TZ,
        )
        slots = [day_start + timedelta(minutes=15 * s) for s in range(96)]
        labels = [s.strftime("%H:%M") for s in slots]

        # 5. Simulate
        self.status_label.setText(f"Running SMPC simulation for {day_str}\u2026")
        QApplication.processEvents()
        results = self._simulate_day_api(
            slots, timeline, solar, selected.month,
        )

        # 6. Render (2-way: baseline vs SMPC)
        self.graph_label.setPixmap(
            draw_comparison_graph(
                results["baseline"], results["smpc"], labels,
            )
        )
        self._display_kpis_api(day_str, results, labels)
        self.status_label.setText(f"Simulation complete for {day_str}.")

    # ------------------------------------------------------------------
    # MODE 2: Historical CSV analysis (hourly resolution)
    # ------------------------------------------------------------------

    def _run_csv_analysis(self):
        """Load CSV data for the selected date and run SMPC comparison."""
        selected = self.date_edit.date().toPyDate()
        day_str = selected.strftime("%Y-%m-%d")

        if self._smpc_cfg is None or self._smpc_calc is None:
            self.status_label.setText("SMPC not initialised — cannot run analysis.")
            return

        # 1. Load CSV (once)
        self.status_label.setText("Loading historical data\u2026")
        QApplication.processEvents()
        if self._csv_data is None:
            try:
                self._csv_data = load_historical_csv(HISTORICAL_CSV)
            except Exception as exc:
                self.status_label.setText(f"CSV load failed: {exc}")
                return

        # 2. Find the day's data.  CSV uses "D/MM/YYYY" format in the Day column.
        day_rows = self._find_day_rows(selected)
        if not day_rows:
            self.status_label.setText(
                f"No data for {day_str} in the historical CSV."
            )
            return

        # 3. Run SMPC over the hourly data
        self.status_label.setText(
            f"Running SMPC simulation ({len(day_rows)} hourly slots)\u2026"
        )
        QApplication.processEvents()
        results = self._simulate_day_csv(day_rows, selected.month)

        # 4. Render 3-way graph
        labels = [r["date"].split(" ")[-1].rsplit(":", 1)[0] for r in day_rows]
        self.graph_label.setPixmap(
            draw_three_way_graph(
                results["baseline"],
                results["simple_opt"],
                results["smpc"],
                labels,
            )
        )
        self._display_kpis_csv(day_str, results)
        self.status_label.setText(f"Analysis complete for {day_str}.")

    def _find_day_rows(self, selected) -> list[dict]:
        """Look up CSV rows for the given date, trying multiple key formats."""
        if self._csv_data is None:
            return []
        # The CSV "Day" column may use D/MM/YYYY or DD/MM/YYYY format.
        # Try several variants to be robust.
        candidates = [
            f"{selected.day}/{selected.month:02d}/{selected.year}",      # 1/01/2022
            f"{selected.day:02d}/{selected.month:02d}/{selected.year}",  # 01/01/2022
            f"{selected.day}/{selected.month}/{selected.year}",          # 1/1/2022
        ]
        for key in candidates:
            if key in self._csv_data:
                return self._csv_data[key]
        return []

    # ------------------------------------------------------------------
    # Simulation — CSV hourly data
    # ------------------------------------------------------------------

    def _simulate_day_csv(self, day_rows: list[dict], month: int) -> dict:
        """Run SMPC over each hour using the real building data from the CSV.

        For each hour:
        * baseline = OriginalCost from CSV (real operation)
        * simple_opt = AdjustedCost from CSV (greedy chiller shift)
        * smpc = SMPC-optimised cost using real consumption & prices

        The SMPC uses a 24-step horizon (one step = 1 hour) for these hourly
        simulations, which is different from the live dashboard's 96 × 15-min.
        """
        cfg = self._smpc_cfg
        n = len(day_rows)

        baseline = np.array([r["original_cost"] for r in day_rows])
        simple_opt = np.array([r["adjusted_cost"] for r in day_rows])
        smpc = np.zeros(n)

        # Build full-day price and consumption arrays for the forecast horizon
        prices_eur_kwh = np.array(
            [r["electricity_price"] / 1000.0 for r in day_rows]
        )
        # Extra cost (grid fee) per kWh — from the CSV
        extra_eur_kwh = np.array(
            [r["extra_cost"] / 1000.0 for r in day_rows]
        )
        total_price_eur_kwh = prices_eur_kwh + extra_eur_kwh

        # Consumption that needs to be supplied from grid (MWh → kWh per hour)
        # RemainingUsage = non-chiller load  (MWh/h)
        remaining_kwh = np.array(
            [r["remaining_usage"] * 1000.0 for r in day_rows]
        )
        chiller_kwh = np.array(
            [r["total_chiller"] * 1000.0 for r in day_rows]
        )
        wkk_kwh = np.array(
            [r["production_wkk"] * 1000.0 for r in day_rows]
        )

        # Ice bank state tracking
        ice_bank = cfg.ice_bank_initial_kwh
        heat_buffer = cfg.heat_buffer_initial_kwh
        ice_charged = 0.0
        ice_discharged = 0.0
        wkk_gas_total = 0.0

        for idx in range(n):
            # For each hour, build a price forecast for the remaining hours
            horizon = min(n - idx, 24)
            price_fc = prices_eur_kwh[idx:idx + horizon]
            # The "consumption" from the grid's perspective = remaining + chiller - WKK
            # But SMPC controls the *chiller* part via ice bank, so the
            # controllable load is the chiller, and remaining is fixed.
            consumption_fc = remaining_kwh[idx:idx + horizon]

            # Heat demand (simplified, from config)
            if month in cfg.winter_months:
                heat_fc = np.full(horizon, self._building["heat_demand_base_kwh"])
            elif month in (4, 10):
                heat_fc = np.full(
                    horizon, self._building["heat_demand_base_kwh"] * 0.5,
                )
            else:
                heat_fc = np.zeros(horizon)

            # Build inputs — note we use the ACTUAL grid fee from CSV data
            cur_price = float(prices_eur_kwh[idx])
            cur_remaining = float(remaining_kwh[idx])

            inputs = SMPCInputs(
                electricity_price_eur_kwh=cur_price,
                price_forecast_eur_kwh=price_fc,
                consumption_kwh=cur_remaining,
                consumption_forecast_kwh=consumption_fc,
                ice_bank_kwh=ice_bank,
                heat_buffer_kwh=heat_buffer,
                heat_demand_forecast_kwh=heat_fc,
                wkk_max_gas_m3=self._building["wkk_max_gas_m3"],
                month=month,
            )

            try:
                out = self._smpc_calc.solve(inputs)
                # Net power from grid = remaining + charge - discharge - wkk_elec
                net_power = out.net_power_kwh
                # Cost = net_power × total price (spot + grid fee)
                grid_fee = float(extra_eur_kwh[idx])
                smpc[idx] = net_power * (cur_price + grid_fee) + (
                    out.wkk_gas_setpoint_m3 * cfg.gas_price_eur_m3
                )
                ice_bank = out.ice_bank_next_kwh
                heat_buffer = out.heat_buffer_next_kwh
                ice_charged += out.ice_bank_charge_kwh
                ice_discharged += out.ice_bank_discharge_kwh
                wkk_gas_total += out.wkk_gas_setpoint_m3
            except Exception as exc:
                logger.warning("SMPC solve error at slot %d: %s", idx, exc)
                smpc[idx] = baseline[idx]

        # Determine season from first slot
        season = _get_season(month, cfg)

        return {
            "baseline": baseline,
            "simple_opt": simple_opt,
            "smpc": smpc,
            "ice_charged": ice_charged,
            "ice_discharged": ice_discharged,
            "wkk_gas": wkk_gas_total,
            "season": season,
            "n_slots": n,
        }

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------

    def _fetch_day_prices(self, selected) -> list | None:
        day_start = datetime(
            selected.year, selected.month, selected.day, tzinfo=LOCAL_TZ,
        )
        day_end = day_start + timedelta(days=1)
        utc = ZoneInfo("UTC")
        start_utc = day_start.astimezone(utc).strftime("%Y%m%d%H%M")
        end_utc = day_end.astimezone(utc).strftime("%Y%m%d%H%M")
        try:
            xml = getPrice.fetch_prices(ENTSOE_DOMAIN, start_utc, end_utc)
            prices = getPrice.parse_prices(xml)
            if not prices:
                self.status_label.setText(
                    f"No ENTSO-E prices available for {selected}."
                )
                return None
            return prices
        except Exception as exc:
            self.status_label.setText(f"Price fetch failed: {exc}")
            return None

    def _fetch_day_solar(self, day_str: str) -> dict | None:
        try:
            client = getWeather.build_client()
            weather = getWeather.fetch_weather(
                client, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, day_str, day_str,
            )
        except Exception as exc:
            self.status_label.setText(f"Weather fetch failed: {exc}")
            return None

        solar: dict[str, float] = {}
        for i, ts in enumerate(weather["timestamps"]):
            ts_key = (
                ts.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(ts, "strftime") else str(ts)
            )
            row = predict.WeatherRow(
                timestamp=ts_key,
                temperature_c=weather["temperature"][i],
                uv_index=weather["uv_index"][i],
                wind_kmh=weather["wind_speed"][i],
            )
            power = predict.predict_power_kw(row, SOLAR_CAPACITY_KWP)
            solar[ts_key] = power * 0.25  # kWh per 15-min interval
        return solar

    @staticmethod
    def _price_timeline(raw_prices):
        """Convert raw (UTC datetime, EUR/MWh) → sorted (local datetime, EUR/kWh)."""
        out = []
        for ts, price_mwh in sorted(raw_prices, key=lambda r: r[0]):
            ts_local = ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0)
            out.append((ts_local, price_mwh / 1000.0))
        return out

    # ------------------------------------------------------------------
    # Simulation — API 15-min slots (original approach)
    # ------------------------------------------------------------------

    def _simulate_day_api(self, slots, price_timeline, solar_by_slot, month):
        cfg = self._smpc_cfg
        calc = self._smpc_calc
        bp = self._building
        H = cfg.horizon_steps

        ice_bank = cfg.ice_bank_initial_kwh
        heat_buffer = cfg.heat_buffer_initial_kwh

        baseline = np.zeros(96)
        smpc = np.zeros(96)
        ice_charged = 0.0
        ice_discharged = 0.0
        wkk_gas = 0.0
        season = "unknown"

        def _price_at(target):
            best = price_timeline[0][1] if price_timeline else 0.10
            for ts, p in price_timeline:
                if ts <= target:
                    best = p
                else:
                    break
            return best

        for idx, slot in enumerate(slots):
            slot_h = slot.hour + slot.minute / 60.0

            # Price forecast
            price_fc = np.array([
                _price_at(slot + timedelta(minutes=15 * t)) for t in range(H)
            ])

            # Consumption = load − solar
            load_kwh = np.zeros(H)
            solar_kwh = np.zeros(H)
            for t in range(H):
                h = (slot_h + t * 0.25) % 24
                if 6 <= h <= 18:
                    kw = bp["base_load_kw"] + (
                        bp["peak_load_kw"] - bp["base_load_kw"]
                    ) * max(0.0, np.sin(np.pi * (h - 6) / 12))
                else:
                    kw = bp["base_load_kw"]
                load_kwh[t] = kw * 0.25
                key = (slot + timedelta(minutes=15 * t)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                solar_kwh[t] = max(0.0, solar_by_slot.get(key, 0.0))

            consumption_fc = np.maximum(load_kwh - solar_kwh, 0.0)

            # Heat demand
            if month in cfg.winter_months:
                heat = np.full(H, bp["heat_demand_base_kwh"])
            elif month in (4, 10):
                heat = np.full(H, bp["heat_demand_base_kwh"] * 0.5)
            else:
                heat = np.zeros(H)

            cur_price = float(price_fc[0])
            cur_cons = float(consumption_fc[0])
            # Baseline includes grid fee (spot + distribution charges)
            baseline[idx] = cur_cons * (cur_price + cfg.grid_fee_eur_kwh)

            inputs = SMPCInputs(
                electricity_price_eur_kwh=cur_price,
                price_forecast_eur_kwh=price_fc,
                consumption_kwh=cur_cons,
                consumption_forecast_kwh=consumption_fc,
                ice_bank_kwh=ice_bank,
                heat_buffer_kwh=heat_buffer,
                heat_demand_forecast_kwh=heat,
                wkk_max_gas_m3=bp["wkk_max_gas_m3"],
                month=month,
            )

            try:
                out = calc.solve(inputs)
                smpc[idx] = out.smpc_cost_eur
                ice_bank = out.ice_bank_next_kwh
                heat_buffer = out.heat_buffer_next_kwh
                ice_charged += out.ice_bank_charge_kwh
                ice_discharged += out.ice_bank_discharge_kwh
                wkk_gas += out.wkk_gas_setpoint_m3
                if idx == 0:
                    season = out.season
            except Exception:
                smpc[idx] = baseline[idx]

        return {
            "baseline": baseline,
            "smpc": smpc,
            "ice_charged": ice_charged,
            "ice_discharged": ice_discharged,
            "wkk_gas": wkk_gas,
            "season": season,
        }

    # ------------------------------------------------------------------
    # Results display — API mode (2-way)
    # ------------------------------------------------------------------

    def _display_kpis_api(self, day_str, results, x_labels):
        bl = results["baseline"]
        sm = results["smpc"]
        total_bl = bl.sum()
        total_sm = sm.sum()
        saving_smpc = total_bl - total_sm
        pct_smpc = (saving_smpc / total_bl * 100) if total_bl != 0 else 0.0

        colour = "#16a34a" if saving_smpc >= 0 else "#dc2626"
        coloured = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )

        self.kpi_title.setText(f"Results for {day_str} (Live API)")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_bl:.4f}")
        self.kpi_labels["simple_opt"].setText("N/A (API mode)")
        self.kpi_labels["smpc"].setText(f"\u20ac{total_sm:.4f}")
        self.kpi_labels["saving_simple"].setText("N/A")
        self.kpi_labels["saving_smpc"].setText(f"\u20ac{saving_smpc:.4f}")
        self.kpi_labels["saving_smpc"].setStyleSheet(coloured)
        self.kpi_labels["pct_simple"].setText("N/A")
        self.kpi_labels["pct_smpc"].setText(f"{pct_smpc:.2f}%")
        self.kpi_labels["pct_smpc"].setStyleSheet(coloured)
        self.kpi_labels["slots"].setText("96 (15-min)")
        self.kpi_labels["season"].setText(results["season"].capitalize())
        self.kpi_labels["ice_charged"].setText(
            f"{results['ice_charged']:.2f} kWh"
        )
        self.kpi_labels["ice_discharged"].setText(
            f"{results['ice_discharged']:.2f} kWh"
        )
        self.kpi_labels["wkk_gas"].setText(f"{results['wkk_gas']:.2f} m\u00b3")

    # ------------------------------------------------------------------
    # Results display — CSV mode (3-way)
    # ------------------------------------------------------------------

    def _display_kpis_csv(self, day_str, results):
        bl = results["baseline"]
        so = results["simple_opt"]
        sm = results["smpc"]
        total_bl = bl.sum()
        total_so = so.sum()
        total_sm = sm.sum()

        saving_simple = total_bl - total_so
        saving_smpc = total_bl - total_sm
        pct_simple = (saving_simple / total_bl * 100) if total_bl != 0 else 0.0
        pct_smpc = (saving_smpc / total_bl * 100) if total_bl != 0 else 0.0

        def _colour(val):
            return "#16a34a" if val >= 0 else "#dc2626"

        def _styled(val):
            c = _colour(val)
            return (
                f"font-family: 'Consolas'; font-weight: 700;"
                f" color: {c}; border: none;"
            )

        self.kpi_title.setText(f"Results for {day_str} (Historical CSV)")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_bl:.2f}")
        self.kpi_labels["simple_opt"].setText(f"\u20ac{total_so:.2f}")
        self.kpi_labels["smpc"].setText(f"\u20ac{total_sm:.2f}")
        self.kpi_labels["saving_simple"].setText(f"\u20ac{saving_simple:.2f}")
        self.kpi_labels["saving_simple"].setStyleSheet(_styled(saving_simple))
        self.kpi_labels["saving_smpc"].setText(f"\u20ac{saving_smpc:.2f}")
        self.kpi_labels["saving_smpc"].setStyleSheet(_styled(saving_smpc))
        self.kpi_labels["pct_simple"].setText(f"{pct_simple:.2f}%")
        self.kpi_labels["pct_simple"].setStyleSheet(_styled(saving_simple))
        self.kpi_labels["pct_smpc"].setText(f"{pct_smpc:.2f}%")
        self.kpi_labels["pct_smpc"].setStyleSheet(_styled(saving_smpc))
        self.kpi_labels["slots"].setText(f"{results['n_slots']} (hourly)")
        self.kpi_labels["season"].setText(results["season"].capitalize())
        self.kpi_labels["ice_charged"].setText(
            f"{results['ice_charged']:.2f} kWh"
        )
        self.kpi_labels["ice_discharged"].setText(
            f"{results['ice_discharged']:.2f} kWh"
        )
        self.kpi_labels["wkk_gas"].setText(f"{results['wkk_gas']:.2f} m\u00b3")
