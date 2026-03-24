"""Historical LP cost-analysis dialog.

Loads hourly building data from ``DATA.csv`` and runs a deterministic LP
for ice bank charge scheduling, comparing baseline vs LP-optimised cost.

Asset configuration (shiftable loads, generators) is read from
:mod:`energy_assets`, which persists to ``dashboard_config.json``.

Architecture
------------
* **UI only** — layout, KPI display, graph rendering.
* **Simulation** — delegated to :mod:`lp_solver` (shared with the live LP).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtWidgets import (
    QApplication, QDateEdit, QDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from energy_assets import EnergyAsset, ensure_defaults, load_assets
from getCO2 import load_co2_csv, FALLBACK_CO2_GRAMS_PER_KWH
from graph_renderer import draw_comparison_graph, draw_power_comparison_graph
from lp_solver import (
    day_key_candidates, find_day_rows, load_historical_csv,
    load_solar_csv, simulate_day,
)
from settings import DASHBOARD_DIR, HISTORICAL_CSV
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

# KPI rows displayed in the results panel
_KPI_ROWS = [
    ("baseline",        "Baseline cost"),
    ("optimised",       "LP optimised cost"),
    ("saving",          "Savings"),
    ("saving_pct",      "Savings %"),
    ("co2_saved",       "CO\u2082 saved"),
    ("co2_saved_pct",   "CO\u2082 saved %"),
    ("slots",           "Time slots"),
    ("load_shifted",    "Total load shifted"),
    ("total_generation","Total on-site generation"),
]


class HistoricalAnalysisDialog(QDialog):
    """Dialog that simulates a past day to compare baseline vs LP costs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Historical LP Analysis")
        self.setMinimumSize(820, 800)
        self.resize(860, 900)
        self.setStyleSheet(HISTORICAL_DIALOG_STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Historical CSV data (loaded lazily)
        self._csv_data: dict[str, list[dict]] | None = None
        self._csv_available = HISTORICAL_CSV.exists()

        # Per-hour solar data keyed by CSV path (loaded lazily)
        self._solar_cache: dict[str, dict[str, list[float]]] = {}

        # CO2 intensity data (loaded lazily)
        self._co2_data: dict[str, list[float]] | None = None

        # Energy asset config (loaded fresh each time dialog opens)
        self._assets: list[EnergyAsset] = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Top bar: title + date selector (stays outside scroll area)
        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(20, 16, 20, 4)
        top_lay.setSpacing(10)

        title = QLabel("Historical LP Cost Analysis")
        title.setStyleSheet("font-size: 13pt; font-weight: 700; color: #0b3a6e;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_lay.addWidget(title)

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
        self._run_btn = run_btn

        year_btn = QPushButton("Full Year")
        year_btn.setMinimumHeight(36)
        year_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        year_btn.clicked.connect(self._run_year_analysis)
        self._year_btn = year_btn

        sel.addWidget(lbl)
        sel.addWidget(self.date_edit, stretch=1)
        sel.addWidget(run_btn)
        sel.addWidget(year_btn)
        top_lay.addLayout(sel)

        self.status_label = QLabel("Pick a date and click 'Run Analysis'")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; font-style: italic;")
        top_lay.addWidget(self.status_label)

        outer.addWidget(top)

        # Scrollable content area (graphs + KPIs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 8, 20, 16)
        layout.setSpacing(12)

        # Cost graph
        self.graph_label = QLabel("")
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setMinimumHeight(270)
        self.graph_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px;"
        )
        self.graph_label.setScaledContents(False)
        layout.addWidget(self.graph_label)

        # Power / load graph
        self.power_graph_label = QLabel("")
        self.power_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_graph_label.setMinimumHeight(270)
        self.power_graph_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px;"
        )
        self.power_graph_label.setScaledContents(False)
        layout.addWidget(self.power_graph_label)

        # KPI card
        self._build_kpi_card(layout)

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

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
        """Run CSV-based historical analysis."""
        if not self._csv_available:
            self.status_label.setText("Historical CSV file not found.")
            return
        self._run_btn.setEnabled(False)
        try:
            self._run_csv_analysis()
        except Exception as exc:
            logger.exception("Historical analysis failed")
            self.status_label.setText(f"Analysis error: {exc}")
        finally:
            self._run_btn.setEnabled(True)

    def _run_csv_analysis(self):
        """Load CSV data for the selected date and run LP optimisation."""
        selected = self.date_edit.date().toPyDate()
        day_str = selected.strftime("%Y-%m-%d")

        # Reset visual state so stale results never linger
        self._clear_results()

        # Reload assets fresh (may have changed since dialog opened)
        self._assets = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        # 1. Load CSV (once)
        self.status_label.setText("Loading historical data\u2026")
        QApplication.processEvents()
        if self._csv_data is None:
            try:
                self._csv_data = load_historical_csv(HISTORICAL_CSV)
            except Exception as exc:
                self.status_label.setText(f"CSV load failed: {exc}")
                return

        # 2. Find the day's data.
        day_keys = day_key_candidates(selected)
        day_rows = find_day_rows(self._csv_data, selected)
        if not day_rows:
            self.status_label.setText(
                f"No data for {day_str} in the historical CSV."
            )
            return

        # 3. Run LP optimisation
        self.status_label.setText(
            f"Running LP optimisation ({len(day_rows)} hourly slots)\u2026"
        )
        QApplication.processEvents()

        solar_hours = self._get_solar_hours_all(day_keys)
        results = simulate_day(
            day_rows, [a for a in self._assets if a.enabled],
            solar_hours, day_keys,
        )

        # 3b. Compute CO2 impact
        co2_hourly = self._get_co2_hours(day_keys, len(day_rows))
        co2_arr = np.array(co2_hourly[:len(day_rows)])
        baseline_co2_g = float(np.sum(results["baseline_grid_kwh"] * co2_arr))
        optimised_co2_g = float(np.sum(results["optimised_grid_kwh"] * co2_arr))
        results["co2_saved_kg"] = (baseline_co2_g - optimised_co2_g) / 1000.0
        results["co2_baseline_g"] = baseline_co2_g
        results["co2_optimised_g"] = optimised_co2_g

        # 4. Render graphs
        gw = max(780, self.width() - 60)
        labels = [r["timestamp"].split(" ")[-1] for r in day_rows]
        self.graph_label.setPixmap(
            draw_comparison_graph(
                results["baseline"], results["optimised"], labels,
                width=gw, height=270,
            )
        )
        self.power_graph_label.setPixmap(
            draw_power_comparison_graph(
                results["baseline_load_kwh"], results["optimised_load_kwh"],
                labels, results["prices_elec"],
                width=gw, height=270,
            )
        )
        self._display_kpis_csv(day_str, results)
        self.status_label.setText(f"Analysis complete for {day_str}.")

    def _clear_results(self):
        """Reset all result widgets to their initial empty state."""
        self.graph_label.clear()
        self.power_graph_label.clear()
        self.kpi_title.setText("Results")
        for lbl in self.kpi_labels.values():
            lbl.setText("--")
            lbl.setStyleSheet(
                "font-family: 'Consolas'; font-weight: 700;"
                " color: #0f766e; border: none;"
            )

    # ------------------------------------------------------------------
    # Solar CSV cache helper
    # ------------------------------------------------------------------

    def _get_solar_hours_all(
        self, keys: list[str],
    ) -> dict[str, list[float]]:
        """Return the full solar dict, loading each CSV lazily."""
        result: dict[str, list[float]] = {}
        for asset in self._assets:
            if not asset.solar_csv:
                continue
            csv_path = str(asset.solar_csv)
            if csv_path not in self._solar_cache:
                path = DASHBOARD_DIR / csv_path
                self._solar_cache[csv_path] = load_solar_csv(path)
            result.update(self._solar_cache[csv_path])
        return result

    # ------------------------------------------------------------------
    # CO2 intensity cache helper
    # ------------------------------------------------------------------

    def _get_co2_hours(
        self, day_key_candidates: list[str], n_hours: int,
    ) -> list[float]:
        """Return per-hour CO2 intensity (gCO2/kWh) for a day.

        Loads ``co2_intensity.csv`` lazily.  If the file doesn't exist or
        the day is missing, returns the Belgian grid fallback value.
        """
        if self._co2_data is None:
            self._co2_data = load_co2_csv()
        for k in day_key_candidates:
            if k in self._co2_data:
                hours = self._co2_data[k]
                if len(hours) >= n_hours:
                    return hours[:n_hours]
                return hours + [FALLBACK_CO2_GRAMS_PER_KWH] * (n_hours - len(hours))
        return [FALLBACK_CO2_GRAMS_PER_KWH] * n_hours

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _display_kpis_csv(self, day_str, results):
        bl = results["baseline"]
        opt = results["optimised"]
        total_bl = bl.sum()
        total_opt = opt.sum()

        saving = total_bl - total_opt
        pct = (saving / total_bl * 100) if total_bl != 0 else 0.0

        colour = "#16a34a" if saving >= 0 else "#dc2626"
        coloured = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )

        self.kpi_title.setText(f"Results for {day_str}")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_bl:.2f}")
        self.kpi_labels["optimised"].setText(f"\u20ac{total_opt:.2f}")
        self.kpi_labels["saving"].setText(f"\u20ac{saving:.2f}")
        self.kpi_labels["saving"].setStyleSheet(coloured)
        self.kpi_labels["saving_pct"].setText(f"{pct:.2f}%")
        self.kpi_labels["saving_pct"].setStyleSheet(coloured)

        co2_kg = results.get("co2_saved_kg", 0.0)
        co2_baseline_g = results.get("co2_baseline_g", 0.0)
        co2_pct = (
            (co2_kg * 1000 / co2_baseline_g * 100) if co2_baseline_g > 0 else 0.0
        )
        co2_colour = "#16a34a" if co2_kg >= 0 else "#dc2626"
        co2_styled = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {co2_colour}; border: none;"
        )
        self.kpi_labels["co2_saved"].setText(f"{co2_kg:.2f} kg CO\u2082")
        self.kpi_labels["co2_saved"].setStyleSheet(co2_styled)
        self.kpi_labels["co2_saved_pct"].setText(f"{co2_pct:.2f}%")
        self.kpi_labels["co2_saved_pct"].setStyleSheet(co2_styled)

        self.kpi_labels["slots"].setText(f"{results['n_slots']} hourly")
        self.kpi_labels["load_shifted"].setText(
            f"{results['load_shifted']:.1f} kWh"
        )
        self.kpi_labels["total_generation"].setText(
            f"{results['total_generation']:.1f} kWh"
        )

    # ------------------------------------------------------------------
    # Full-year analysis
    # ------------------------------------------------------------------

    def _run_year_analysis(self):
        """Run LP simulation for every day in the selected year."""
        if not self._csv_available:
            self.status_label.setText("Historical CSV file not found.")
            return

        self._run_btn.setEnabled(False)
        self._year_btn.setEnabled(False)
        try:
            self._do_year_analysis()
        except Exception as exc:
            logger.exception("Year analysis failed")
            self.status_label.setText(f"Year analysis error: {exc}")
        finally:
            self._run_btn.setEnabled(True)
            self._year_btn.setEnabled(True)

    def _do_year_analysis(self):
        """Aggregate daily LP results across all available days in the year."""
        year = self.date_edit.date().year()
        self._clear_results()

        # Reload assets
        self._assets = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        # Load CSV data
        self.status_label.setText("Loading historical data\u2026")
        QApplication.processEvents()
        if self._csv_data is None:
            try:
                self._csv_data = load_historical_csv(HISTORICAL_CSV)
            except Exception as exc:
                self.status_label.setText(f"CSV load failed: {exc}")
                return

        # Pre-load solar data
        solar_hours = self._get_solar_hours_all([])
        enabled_assets = [a for a in self._assets if a.enabled]

        # Accumulators
        total_baseline_cost = 0.0
        total_optimised_cost = 0.0
        total_co2_baseline_g = 0.0
        total_co2_optimised_g = 0.0
        total_load_shifted = 0.0
        total_generation = 0.0
        total_slots = 0
        days_processed = 0

        # Iterate every day in the year
        day = date(year, 1, 1)
        end = date(year + 1, 1, 1)
        while day < end:
            if days_processed % 30 == 0:
                self.status_label.setText(
                    f"Processing {year}\u2026 {day.strftime('%b %d')} "
                    f"({days_processed} days done)"
                )
                QApplication.processEvents()

            day_rows = find_day_rows(self._csv_data, day)
            if not day_rows:
                day += timedelta(days=1)
                continue

            keys = day_key_candidates(day)
            results = simulate_day(day_rows, enabled_assets, solar_hours, keys)

            # Cost accumulators
            total_baseline_cost += float(results["baseline"].sum())
            total_optimised_cost += float(results["optimised"].sum())
            total_load_shifted += results["load_shifted"]
            total_generation += results["total_generation"]
            total_slots += results["n_slots"]

            # CO2 accumulators
            n = len(day_rows)
            co2_hourly = self._get_co2_hours(keys, n)
            co2_arr = np.array(co2_hourly[:n])
            total_co2_baseline_g += float(
                np.sum(results["baseline_grid_kwh"] * co2_arr)
            )
            total_co2_optimised_g += float(
                np.sum(results["optimised_grid_kwh"] * co2_arr)
            )

            days_processed += 1
            day += timedelta(days=1)

        if days_processed == 0:
            self.status_label.setText(f"No data found for {year}.")
            return

        # Build aggregated results dict
        saving = total_baseline_cost - total_optimised_cost
        saving_pct = (
            (saving / total_baseline_cost * 100)
            if total_baseline_cost != 0 else 0.0
        )
        co2_saved_kg = (total_co2_baseline_g - total_co2_optimised_g) / 1000.0
        co2_pct = (
            ((total_co2_baseline_g - total_co2_optimised_g)
             / total_co2_baseline_g * 100)
            if total_co2_baseline_g > 0 else 0.0
        )

        colour = "#16a34a" if saving >= 0 else "#dc2626"
        coloured = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )
        co2_colour = "#16a34a" if co2_saved_kg >= 0 else "#dc2626"
        co2_styled = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {co2_colour}; border: none;"
        )

        self.kpi_title.setText(f"Full Year Results for {year} ({days_processed} days)")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_baseline_cost:.2f}")
        self.kpi_labels["optimised"].setText(f"\u20ac{total_optimised_cost:.2f}")
        self.kpi_labels["saving"].setText(f"\u20ac{saving:.2f}")
        self.kpi_labels["saving"].setStyleSheet(coloured)
        self.kpi_labels["saving_pct"].setText(f"{saving_pct:.2f}%")
        self.kpi_labels["saving_pct"].setStyleSheet(coloured)
        self.kpi_labels["co2_saved"].setText(f"{co2_saved_kg:.2f} kg CO\u2082")
        self.kpi_labels["co2_saved"].setStyleSheet(co2_styled)
        self.kpi_labels["co2_saved_pct"].setText(f"{co2_pct:.2f}%")
        self.kpi_labels["co2_saved_pct"].setStyleSheet(co2_styled)
        self.kpi_labels["slots"].setText(f"{total_slots} hourly")
        self.kpi_labels["load_shifted"].setText(
            f"{total_load_shifted:.1f} kWh"
        )
        self.kpi_labels["total_generation"].setText(
            f"{total_generation:.1f} kWh"
        )

        self.graph_label.setText(
            "Full-year mode — per-day graphs not shown"
        )
        self.power_graph_label.setText(
            "Full-year mode — per-day graphs not shown"
        )
        self.status_label.setText(
            f"Year analysis complete — {days_processed} days processed."
        )
