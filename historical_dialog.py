"""
╔══════════════════════════════════════════════════════════════════╗
║  FRONTEND FILE — student is NOT responsible for this module      ║
║                                                                  ║
║  Historical analysis dialog.  UI glue only: date picker,        ║
║  graph display, KPI table.  All simulation logic is in           ║
║  lp_solver.simulate_day() (backend).                             ║
╚══════════════════════════════════════════════════════════════════╝

Historical LP cost-analysis dialog.

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
    QApplication, QDateEdit, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from energy_assets import EnergyAsset, ensure_defaults, load_assets
from getCO2 import (
    load_co2_csv, get_hourly_co2,
    FALLBACK_CO2_GRAMS_PER_KWH, GAS_CO2_G_PER_KWH_GAS,
)
from graph_renderer import draw_comparison_graph, draw_power_comparison_graph
from lp_solver import (
    day_key_candidates, find_day_rows, load_historical_csv,
    load_solar_csv, simulate_day,
)
from settings import DASHBOARD_DIR, HISTORICAL_CSV
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

# Top-level KPI rows (always shown)
_KPI_ROWS_TOP = [
    ("baseline",             "Baseline cost"),
    ("optimised",            "LP optimised cost"),
    ("saving",               "Total savings"),
    ("saving_pct",           "Savings %"),
    ("co2_saved",            "CO\u2082 saved"),
    ("co2_saved_pct",        "CO\u2082 saved %"),
    ("heating_saving",       "Heating savings"),
    ("heating_saving_pct",   "Heating savings %"),
    ("hw_saving",            "Hot water savings"),
    ("hw_saving_pct",        "Hot water savings %"),
    ("slots",                "Time slots"),
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

        # Last simulation results (for CSV export)
        self._last_results: dict | None = None
        self._last_day_str: str = ""
        self._last_day_rows: list[dict] = []

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

        export_btn = QPushButton("Export CSV")
        export_btn.setMinimumHeight(36)
        export_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        export_btn.clicked.connect(self._export_csv)
        export_btn.setEnabled(False)
        self._export_btn = export_btn

        sel.addWidget(lbl)
        sel.addWidget(self.date_edit, stretch=1)
        sel.addWidget(run_btn)
        sel.addWidget(year_btn)
        sel.addWidget(export_btn)
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
        for key, text in _KPI_ROWS_TOP:
            row = QHBoxLayout()
            name = QLabel(text)
            name.setStyleSheet("font-weight: 600; border: none;")
            val = QLabel("--")
            val.setStyleSheet(
                "font-family: 'Calibri'; font-weight: 700;"
                " color: #0f766e; border: none;"
            )
            row.addWidget(name)
            row.addStretch()
            row.addWidget(val)
            kpi_layout.addLayout(row)
            self.kpi_labels[key] = val

        # Dynamic per-asset section (populated after a run)
        self._kpi_frame = frame
        self._kpi_frame_layout = kpi_layout
        # Track widgets added dynamically so we can remove them without
        # touching the static rows above.
        self._asset_kpi_widgets: list = []

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
        # ── BACKEND CALL: lp_solver.simulate_day() ──────────────────
        # This is where the actual optimisation happens.
        # simulate_day() runs greedy_lp() for each shiftable load and
        # compares the rescheduled cost to the historical (baseline) cost.
        # The dialog only displays the results — it contains no calculations.
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

        # Per-generator CO2 impact — CHP is shown separately, NOT in total CO2 saved
        asset_lookup = {a.name: a for a in self._assets}
        for name, gd in results.get("asset_generators", {}).items():
            grid_co2_displaced_g = float(np.sum(gd["gen_kwh"] * co2_arr))
            chp_asset = asset_lookup.get(name)
            if chp_asset and getattr(chp_asset, "chp_elec_efficiency", 0.0) > 0:
                gas_co2_per_kwh_elec = GAS_CO2_G_PER_KWH_GAS / chp_asset.chp_elec_efficiency
                gas_co2_emitted_g = float(np.sum(gd["gen_kwh"])) * gas_co2_per_kwh_elec
                # Store raw values for the CHP CO2 breakdown panel
                gd["chp_gas_co2_emitted_kg"] = gas_co2_emitted_g / 1000.0
                gd["chp_grid_co2_equiv_kg"]  = grid_co2_displaced_g / 1000.0
                gd["co2_saving_kg"] = (grid_co2_displaced_g - gas_co2_emitted_g) / 1000.0
            else:
                gd["co2_saving_kg"] = grid_co2_displaced_g / 1000.0

        # Overall CO2 saved = grid reduction from LP load-shifting only (CHP excluded)
        results["co2_saved_kg"]  = (baseline_co2_g - optimised_co2_g) / 1000.0
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

        # Store for CSV export
        self._last_results = results
        self._last_day_str = day_str
        self._last_day_rows = day_rows
        self._export_btn.setEnabled(True)

    def _clear_results(self):
        """Reset all result widgets to their initial empty state."""
        self.graph_label.clear()
        self.power_graph_label.clear()
        self.kpi_title.setText("Results")
        for lbl in self.kpi_labels.values():
            lbl.setText("--")
            lbl.setStyleSheet(
                "font-family: 'Calibri'; font-weight: 700;"
                " color: #0f766e; border: none;"
            )
        self._remove_asset_kpi_widgets()

    def _remove_asset_kpi_widgets(self):
        """Remove all dynamically added per-asset widgets from the KPI card."""
        for w in self._asset_kpi_widgets:
            w.setParent(None)  # detach from layout
            w.deleteLater()
        self._asset_kpi_widgets.clear()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Export last simulation results to a CSV file with per-asset breakdown."""
        if self._last_results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", f"historical_{self._last_day_str}.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        r = self._last_results
        n = r["n_slots"]
        timestamps = [row["timestamp"] for row in self._last_day_rows[:n]]

        asset_loads: dict = r.get("asset_loads", {})       # {name: {baseline, optimised}}
        asset_gens: dict  = r.get("asset_generators", {})  # {name: {gen_kwh}}

        load_names = list(asset_loads.keys())
        gen_names  = list(asset_gens.keys())

        lines: list[str] = []

        # ── Section 1: per-hour detail ──────────────────────────────
        lines.append("# === HOURLY DETAIL ===")
        # Build dynamic header
        col_headers = [
            "Timestamp", "Price_EUR_MWh",
            "Baseline_Grid_kWh", "Optimised_Grid_kWh",
            "Baseline_Cost_EUR", "Optimised_Cost_EUR",
        ]
        for name in load_names:
            safe = name.replace(";", "_")
            col_headers += [f"{safe}_Baseline_kWh", f"{safe}_Optimised_kWh"]
        for name in gen_names:
            safe = name.replace(";", "_")
            col_headers.append(f"{safe}_Gen_kWh")
        col_headers.append("Decision")
        lines.append(";".join(col_headers))

        for i in range(n):
            bl_grid  = r["baseline_grid_kwh"][i]
            opt_grid = r["optimised_grid_kwh"][i]
            bl_cost  = r["baseline"][i]
            opt_cost = r["optimised"][i]
            price    = r["prices_elec"][i]

            # Decision based on total shiftable load delta
            opt_load = r["optimised_load_kwh"][i]
            bl_load  = r["baseline_load_kwh"][i]
            delta    = opt_load - bl_load
            if abs(delta) < 0.01:
                decision = "No change"
            elif delta > 0:
                decision = f"Shifted +{delta:.1f} kWh here (cheap hour)"
            else:
                decision = f"Shifted {delta:.1f} kWh away (expensive hour)"

            row_vals = [
                timestamps[i],
                f"{price:.2f}",
                f"{bl_grid:.3f}", f"{opt_grid:.3f}",
                f"{bl_cost:.4f}", f"{opt_cost:.4f}",
            ]
            for name in load_names:
                ad = asset_loads[name]
                row_vals += [f"{ad['baseline'][i]:.3f}", f"{ad['optimised'][i]:.3f}"]
            for name in gen_names:
                row_vals.append(f"{asset_gens[name]['gen_kwh'][i]:.3f}")
            row_vals.append(decision)
            lines.append(";".join(row_vals))

        # ── Section 2: per-asset summary ───────────────────────────
        lines.append("")
        lines.append("# === PER-ASSET SUMMARY ===")
        lines.append("Asset;Type;Baseline_Total_kWh;Optimised_Total_kWh;Saving_kWh;Daily_kWh")
        for name, ad in asset_loads.items():
            bl_t  = float(np.sum(ad["baseline"]))
            opt_t = float(np.sum(ad["optimised"]))
            lines.append(
                f"{name};Shiftable Load;{bl_t:.2f};{opt_t:.2f};"
                f"{bl_t - opt_t:.2f};{ad.get('daily_kwh', bl_t):.2f}"
            )
        for name, gd in asset_gens.items():
            gen_t = float(np.sum(gd["gen_kwh"]))
            lines.append(f"{name};Generator;----;{gen_t:.2f};----;----")

        # ── Section 3: overall summary ─────────────────────────────
        total_bl  = float(r["baseline"].sum())
        total_opt = float(r["optimised"].sum())
        saving    = total_bl - total_opt
        lines.append("")
        lines.append("# === OVERALL SUMMARY ===")
        lines.append(f"# Date;{self._last_day_str}")
        lines.append(f"# Baseline total cost;EUR {total_bl:.2f}")
        lines.append(f"# Optimised total cost;EUR {total_opt:.2f}")
        lines.append(f"# Saving;EUR {saving:.2f}")
        lines.append(f"# Load shifted;{r['load_shifted']:.1f} kWh")
        lines.append(f"# Total generation;{r['total_generation']:.1f} kWh")
        startup = r.get("startup_cost", 0.0)
        if startup > 0:
            lines.append(f"# Startup costs;EUR {startup:.2f}")
        co2_kg = r.get("co2_saved_kg")
        if co2_kg is not None:
            lines.append(f"# CO2 saved;{co2_kg:.2f} kg")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.status_label.setText(f"Exported to {path}")
        except OSError as exc:
            self.status_label.setText(f"Export failed: {exc}")

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

        1. Try the cached ``co2_intensity.csv`` first (fast, offline).
        2. If the day is missing, call ``get_hourly_co2()`` live from ENTSO-E
           and store the result back into the in-memory cache.
        3. If the live fetch also fails, fall back to the Belgian grid average.
        """
        if self._co2_data is None:
            self._co2_data = load_co2_csv()

        # 1. CSV cache hit
        for k in day_key_candidates:
            if k in self._co2_data:
                hours = self._co2_data[k]
                if len(hours) >= n_hours:
                    return hours[:n_hours]
                return hours + [FALLBACK_CO2_GRAMS_PER_KWH] * (n_hours - len(hours))

        # 2. Live ENTSO-E fetch — parse key like "D/MM/YYYY"
        for k in day_key_candidates:
            try:
                parts = k.split("/")
                if len(parts) == 3:
                    from datetime import datetime as _dt
                    day_dt = _dt(int(parts[2]), int(parts[1]), int(parts[0]))
                    hourly = get_hourly_co2(day_dt)
                    if hourly:
                        values = [co2 for _, co2 in hourly]
                        self._co2_data[k] = values  # persist in memory cache
                        if len(values) >= n_hours:
                            return values[:n_hours]
                        return values + [FALLBACK_CO2_GRAMS_PER_KWH] * (n_hours - len(values))
            except Exception:
                logger.debug("Live CO2 fetch failed for key %s", k)
                continue

        # 3. Static fallback
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
            f"font-family: 'Calibri'; font-weight: 700;"
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
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {co2_colour}; border: none;"
        )
        self.kpi_labels["co2_saved"].setText(f"{co2_kg:.2f} kg CO\u2082")
        self.kpi_labels["co2_saved"].setStyleSheet(co2_styled)
        self.kpi_labels["co2_saved_pct"].setText(f"{co2_pct:.2f}%")
        self.kpi_labels["co2_saved_pct"].setStyleSheet(co2_styled)

        heat_save = results.get("heating_saving_eur", 0.0)
        heat_pct  = results.get("heating_saving_pct", 0.0)
        heat_colour = "#16a34a" if heat_save >= 0 else "#dc2626"
        heat_styled = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {heat_colour}; border: none;"
        )
        self.kpi_labels["heating_saving"].setText(f"\u20ac{heat_save:.2f}")
        self.kpi_labels["heating_saving"].setStyleSheet(heat_styled)
        self.kpi_labels["heating_saving_pct"].setText(f"{heat_pct:.2f}%")
        self.kpi_labels["heating_saving_pct"].setStyleSheet(heat_styled)

        hw_save = results.get("hw_saving_eur", 0.0)
        hw_pct  = results.get("hw_saving_pct",  0.0)
        hw_colour = "#16a34a" if hw_save >= 0 else "#dc2626"
        hw_styled = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {hw_colour}; border: none;"
        )
        self.kpi_labels["hw_saving"].setText(f"\u20ac{hw_save:.2f}")
        self.kpi_labels["hw_saving"].setStyleSheet(hw_styled)
        self.kpi_labels["hw_saving_pct"].setText(f"{hw_pct:.2f}%")
        self.kpi_labels["hw_saving_pct"].setStyleSheet(hw_styled)

        self.kpi_labels["slots"].setText(f"{results['n_slots']} hourly")

        # ── Per-asset breakdown ─────────────────────────────────────
        self._rebuild_asset_kpis(results)

    def _rebuild_asset_kpis(self, results):
        """Dynamically add per-asset saving rows to the KPI card."""
        self._remove_asset_kpi_widgets()

        def _add_section(title):
            sep = QLabel(title)
            sep.setStyleSheet(
                "font-weight: 700; color: #0b3a6e; border: none;"
                " padding-top: 6px;"
            )
            self._kpi_frame_layout.addWidget(sep)
            self._asset_kpi_widgets.append(sep)

        def _add_row(label_text, value_text, colour="#0f766e"):
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            name_lbl = QLabel(label_text)
            name_lbl.setStyleSheet("font-weight: 600; border: none;")
            val_lbl = QLabel(value_text)
            val_lbl.setStyleSheet(
                f"font-family: 'Calibri'; font-weight: 700;"
                f" color: {colour}; border: none;"
            )
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            self._kpi_frame_layout.addWidget(container)
            self._asset_kpi_widgets.append(container)

        # ── Outputs: generators ─────────────────────────────────────
        asset_gens = results.get("asset_generators", {})
        if asset_gens:
            _add_section("\u26a1 Output savings (generators)")
            for name, gd in asset_gens.items():
                gen_total = float(np.sum(gd["gen_kwh"]))
                cost_save = gd.get("cost_saving_eur", 0.0)
                _add_row(f"  {name} \u2014 generation",
                         f"{gen_total:.1f} kWh", "#0e7490")
                _add_row(f"  {name} \u2014 cost saving",
                         f"\u20ac{cost_save:.2f}",
                         "#16a34a" if cost_save >= 0 else "#dc2626")

        # ── CHP CO2 breakdown (separate from total CO2 saved) ───────
        chp_co2_rows = [
            (name, gd) for name, gd in asset_gens.items()
            if "chp_gas_co2_emitted_kg" in gd
        ]
        if chp_co2_rows:
            _add_section("\U0001f4a8 CHP CO\u2082 vs Grid CO\u2082 (informational)")
            for name, gd in chp_co2_rows:
                emitted = gd["chp_gas_co2_emitted_kg"]
                equiv   = gd["chp_grid_co2_equiv_kg"]
                net     = equiv - emitted  # positive = CHP is cleaner than grid
                _add_row(f"  {name} \u2014 gas CO\u2082 emitted",
                         f"{emitted:.2f} kg", "#dc2626")
                _add_row(f"  {name} \u2014 grid CO\u2082 equiv.",
                         f"{equiv:.2f} kg", "#64748b")
                _add_row(f"  {name} \u2014 net vs grid",
                         f"{net:+.2f} kg",
                         "#16a34a" if net >= 0 else "#dc2626")

        # ── Inputs: shiftable loads ─────────────────────────────────
        asset_loads = results.get("asset_loads", {})
        if asset_loads:
            _add_section("\U0001f504 Input savings (load shifting)")
            for name, ad in asset_loads.items():
                bl_kwh  = float(np.sum(ad["baseline"]))
                opt_kwh = float(np.sum(ad["optimised"]))
                cost_save = ad.get("cost_saving_eur", 0.0)
                _add_row(f"  {name} \u2014 baseline schedule",
                         f"{bl_kwh:.1f} kWh", "#64748b")
                _add_row(f"  {name} \u2014 LP schedule",
                         f"{opt_kwh:.1f} kWh", "#0e7490")
                _add_row(f"  {name} \u2014 cost saving",
                         f"\u20ac{cost_save:.2f}",
                         "#16a34a" if cost_save >= 0 else "#dc2626")

        # ── Thermal savings ─────────────────────────────────────────
        heat_save = results.get("heating_saving_eur")
        if heat_save is not None:
            heat_bl   = results.get("heating_baseline_cost_eur", 0.0)
            heat_pct  = results.get("heating_saving_pct", 0.0)
            _add_section("\U0001f321 Thermal savings (smart pre-heating)")
            _add_row("  Baseline heating cost",
                     f"\u20ac{heat_bl:.2f}", "#64748b")
            _add_row("  Smart heating cost",
                     f"\u20ac{heat_bl - heat_save:.2f}", "#0e7490")
            _add_row("  Heating saving",
                     f"\u20ac{heat_save:.2f}",
                     "#16a34a" if heat_save >= 0 else "#dc2626")
            _add_row("  Heating saving %",
                     f"{heat_pct:.1f}%",
                     "#16a34a" if heat_pct >= 0 else "#dc2626")

        # ── Hot water tank savings ───────────────────────────────────
        hw_save = results.get("hw_saving_eur")
        if hw_save is not None:
            hw_bl  = results.get("hw_baseline_cost_eur", 0.0)
            hw_pct = results.get("hw_saving_pct", 0.0)
            _add_section("\U0001f6bf Hot water tank (smart scheduling, COP\u202f=\u202f1)")
            _add_row("  Baseline DHW cost",
                     f"\u20ac{hw_bl:.2f}", "#64748b")
            _add_row("  Smart DHW cost",
                     f"\u20ac{hw_bl - hw_save:.2f}", "#0e7490")
            _add_row("  DHW saving",
                     f"\u20ac{hw_save:.2f}",
                     "#16a34a" if hw_save >= 0 else "#dc2626")
            _add_row("  DHW saving %",
                     f"{hw_pct:.1f}%",
                     "#16a34a" if hw_pct >= 0 else "#dc2626")

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
        asset_lookup = {a.name: a for a in self._assets}

        # Accumulators
        total_baseline_cost = 0.0
        total_optimised_cost = 0.0
        total_co2_baseline_g = 0.0
        total_co2_optimised_g = 0.0
        total_heating_saving = 0.0
        total_heating_baseline = 0.0
        total_hw_saving = 0.0
        total_hw_baseline = 0.0
        total_slots = 0
        days_processed = 0
        # Per-asset accumulators (keyed by asset name)
        year_asset_generators: dict[str, dict] = {}  # name → {gen_kwh, cost_saving_eur, co2_saving_kg}
        year_asset_loads: dict[str, dict] = {}        # name → {baseline_total, optimised_total, cost_saving_eur}

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

            # CO2 per day
            n = len(day_rows)
            co2_hourly = self._get_co2_hours(keys, n)
            co2_arr = np.array(co2_hourly[:n])

            # Attribute per-generator CO2 saving for this day
            for name, gd in results.get("asset_generators", {}).items():
                grid_co2_displaced_g = float(np.sum(gd["gen_kwh"] * co2_arr))
                chp_a = asset_lookup.get(name)
                if chp_a and getattr(chp_a, "chp_elec_efficiency", 0.0) > 0:
                    gas_co2_per_kwh_elec = GAS_CO2_G_PER_KWH_GAS / chp_a.chp_elec_efficiency
                    gas_co2_emitted_g = float(np.sum(gd["gen_kwh"])) * gas_co2_per_kwh_elec
                    gd["chp_gas_co2_emitted_kg"] = gas_co2_emitted_g / 1000.0
                    gd["chp_grid_co2_equiv_kg"]  = grid_co2_displaced_g / 1000.0
                    gd["co2_saving_kg"] = (grid_co2_displaced_g - gas_co2_emitted_g) / 1000.0
                else:
                    gd["co2_saving_kg"] = grid_co2_displaced_g / 1000.0
                acc = year_asset_generators.setdefault(name, {
                    "gen_kwh": np.zeros(1), "cost_saving_eur": 0.0, "co2_saving_kg": 0.0,
                    "chp_gas_co2_emitted_kg": 0.0, "chp_grid_co2_equiv_kg": 0.0,
                })
                acc["gen_kwh"] = np.array([acc.get("_gen_total", 0.0) + float(np.sum(gd["gen_kwh"]))])
                acc["_gen_total"] = float(acc["gen_kwh"][0])
                acc["cost_saving_eur"] += gd.get("cost_saving_eur", 0.0)
                acc["co2_saving_kg"]   += gd.get("co2_saving_kg", 0.0)
                acc["chp_gas_co2_emitted_kg"] += gd.get("chp_gas_co2_emitted_kg", 0.0)
                acc["chp_grid_co2_equiv_kg"]  += gd.get("chp_grid_co2_equiv_kg", 0.0)

            # Accumulate per-load cost savings
            for name, ad in results.get("asset_loads", {}).items():
                acc = year_asset_loads.setdefault(name, {
                    "baseline": np.zeros(1), "optimised": np.zeros(1),
                    "_bl_total": 0.0, "_opt_total": 0.0,
                    "cost_saving_eur": 0.0,
                })
                acc["_bl_total"]  += float(np.sum(ad["baseline"]))
                acc["_opt_total"] += float(np.sum(ad["optimised"]))
                acc["baseline"]    = np.array([acc["_bl_total"]])
                acc["optimised"]   = np.array([acc["_opt_total"]])
                acc["cost_saving_eur"] += ad.get("cost_saving_eur", 0.0)

            # Cost accumulators
            total_baseline_cost  += float(results["baseline"].sum())
            total_optimised_cost += float(results["optimised"].sum())
            total_slots          += results["n_slots"]
            total_co2_baseline_g += float(np.sum(results["baseline_grid_kwh"] * co2_arr))
            total_co2_optimised_g += float(np.sum(results["optimised_grid_kwh"] * co2_arr))
            total_heating_saving  += results.get("heating_saving_eur", 0.0)
            total_heating_baseline += results.get("heating_baseline_cost_eur", 0.0)
            total_hw_saving   += results.get("hw_saving_eur", 0.0)
            total_hw_baseline += results.get("hw_baseline_cost_eur", 0.0)

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
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )
        co2_colour = "#16a34a" if co2_saved_kg >= 0 else "#dc2626"
        co2_styled = (
            f"font-family: 'Calibri'; font-weight: 700;"
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

        heat_colour = "#16a34a" if total_heating_saving >= 0 else "#dc2626"
        heat_styled = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {heat_colour}; border: none;"
        )
        heat_pct = (
            (total_heating_saving / total_heating_baseline * 100)
            if total_heating_baseline > 0 else 0.0
        )
        self.kpi_labels["heating_saving"].setText(f"\u20ac{total_heating_saving:.2f}")
        self.kpi_labels["heating_saving"].setStyleSheet(heat_styled)
        self.kpi_labels["heating_saving_pct"].setText(f"{heat_pct:.2f}%")
        self.kpi_labels["heating_saving_pct"].setStyleSheet(heat_styled)

        hw_colour = "#16a34a" if total_hw_saving >= 0 else "#dc2626"
        hw_styled = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {hw_colour}; border: none;"
        )
        hw_pct = (
            (total_hw_saving / total_hw_baseline * 100)
            if total_hw_baseline > 0 else 0.0
        )
        self.kpi_labels["hw_saving"].setText(f"\u20ac{total_hw_saving:.2f}")
        self.kpi_labels["hw_saving"].setStyleSheet(hw_styled)
        self.kpi_labels["hw_saving_pct"].setText(f"{hw_pct:.2f}%")
        self.kpi_labels["hw_saving_pct"].setStyleSheet(hw_styled)

        self.kpi_labels["slots"].setText(f"{total_slots} hourly")

        self.graph_label.setText(
            "Full-year mode — per-day graphs not shown"
        )
        self.power_graph_label.setText(
            "Full-year mode — per-day graphs not shown"
        )

        # Show year-level per-asset totals (accumulated)
        self._rebuild_asset_kpis({
            "asset_generators": year_asset_generators,
            "asset_loads":      year_asset_loads,
            "heating_saving_eur":          total_heating_saving,
            "heating_saving_pct":          heat_pct,
            "heating_baseline_cost_eur":   total_heating_baseline,
            "hw_saving_eur":               total_hw_saving,
            "hw_saving_pct":               hw_pct,
            "hw_baseline_cost_eur":        total_hw_baseline,
        })

        self.status_label.setText(
            f"Year analysis complete — {days_processed} days processed."
        )
