"""
Future LP simulation dialog.

Lets the user load a CSV with custom price data (timestamp + price)
and runs the LP to compare baseline vs LP-optimised cost using the
same building model as the live dashboard (sinusoidal load profile
from ``dashboard_config.json`` + solar predictions).

The user CSV must have columns:
    Timestamp;PricesElec

where Timestamp is ``D/MM/YYYY H:MM`` and PricesElec is the spot price
in EUR/MWh (European decimals accepted).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from dashboard_config import load_dashboard_config
from energy_assets import (
    EnergyAsset, EXTRA_COST_EUR_MWH, GENERATOR, SHIFTABLE_LOAD,
    ensure_defaults, load_assets,
)
from graph_renderer import draw_comparison_graph, draw_power_comparison_graph
from lp_solver import greedy_lp_ramped, parse_eu_float
from settings import LOCAL_TZ
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

_KPI_ROWS = [
    ("baseline",        "Baseline cost"),
    ("optimised",       "LP optimised cost"),
    ("saving",          "Savings"),
    ("saving_pct",      "Savings %"),
    ("slots",           "Time slots"),
    ("load_shifted",    "Total load shifted"),
    ("total_generation","Total on-site generation"),
]


class FutureSimulationDialog(QDialog):
    """Dialog that simulates future days using user-supplied price CSV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Future LP Simulation")
        self.setMinimumSize(820, 800)
        self.resize(860, 900)
        self.setStyleSheet(HISTORICAL_DIALOG_STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._assets: list[EnergyAsset] = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        # Building parameters (same source as the live dashboard)
        raw = load_dashboard_config()
        building = raw.get("smpc", {}).get("building", {})
        self._base_load_kw: float = building.get("base_load_kw", 80)
        self._peak_load_kw: float = building.get("peak_load_kw", 200)

        # User-supplied price CSV data: list of (timestamp_str, price_eur_mwh)
        self._user_price_rows: list[tuple[str, float]] | None = None
        self._user_csv_path: str = ""

        # Last simulation results (for CSV export)
        self._last_results: dict | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(20, 16, 20, 4)
        top_lay.setSpacing(10)

        title = QLabel("Future LP Price Simulation")
        title.setStyleSheet("font-size: 13pt; font-weight: 700; color: #0b3a6e;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_lay.addWidget(title)

        desc = QLabel(
            "Load a CSV with columns <b>Timestamp;PricesElec</b> "
            "(timestamp format: <code>D/MM/YYYY H:MM</code>, "
            "price in EUR/MWh).<br>"
            "The building load profile from the live simulation model "
            "will be used and the LP will reschedule shiftable loads "
            "to your prices."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #475569; font-size: 9pt;")
        top_lay.addWidget(desc)

        sel = QHBoxLayout()
        sel.setSpacing(10)

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(
            "font-weight: 600; color: #64748b; font-style: italic;"
        )

        browse_btn = QPushButton("Browse CSV\u2026")
        browse_btn.setMinimumHeight(36)
        browse_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        browse_btn.clicked.connect(self._browse_csv)

        run_btn = QPushButton("Run Simulation")
        run_btn.setMinimumHeight(36)
        run_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        run_btn.clicked.connect(self._run_simulation)
        self._run_btn = run_btn

        export_btn = QPushButton("Export CSV")
        export_btn.setMinimumHeight(36)
        export_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        export_btn.clicked.connect(self._export_csv)
        export_btn.setEnabled(False)
        self._export_btn = export_btn

        sel.addWidget(self.file_label, stretch=1)
        sel.addWidget(browse_btn)
        sel.addWidget(run_btn)
        sel.addWidget(export_btn)
        top_lay.addLayout(sel)

        self.status_label = QLabel("Load a price CSV and click 'Run Simulation'")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; font-style: italic;")
        top_lay.addWidget(self.status_label)

        outer.addWidget(top)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 8, 20, 16)
        layout.setSpacing(12)

        self.graph_label = QLabel("")
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setMinimumHeight(270)
        self.graph_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px;"
        )
        self.graph_label.setScaledContents(False)
        layout.addWidget(self.graph_label)

        self.power_graph_label = QLabel("")
        self.power_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_graph_label.setMinimumHeight(270)
        self.power_graph_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px;"
        )
        self.power_graph_label.setScaledContents(False)
        layout.addWidget(self.power_graph_label)

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
    # CSV browsing
    # ------------------------------------------------------------------

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Price CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            self._user_price_rows = self._load_price_csv(Path(path))
            self._user_csv_path = path
            n_rows = len(self._user_price_rows)
            self.file_label.setText(
                f"{Path(path).name}  ({n_rows} rows)"
            )
            self.file_label.setStyleSheet("font-weight: 600; color: #0b3a6e;")
            self.status_label.setText("CSV loaded \u2014 click 'Run Simulation'")
        except Exception as exc:
            logger.exception("Failed to load price CSV")
            self.file_label.setText("Load failed")
            self.file_label.setStyleSheet("font-weight: 600; color: #dc2626;")
            self.status_label.setText(f"CSV error: {exc}")
            self._user_price_rows = None

    @staticmethod
    def _load_price_csv(path: Path) -> list[tuple[str, float]]:
        """Parse a user-supplied price CSV.

        Expected format (semicolon-separated)::

            Timestamp;PricesElec
            1/01/2025 0:00;82,02
            1/01/2025 1:00;67,07

        Returns list of (timestamp_str, price_eur_mwh).
        """
        rows: list[tuple[str, float]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            fields = reader.fieldnames or []
            if "Timestamp" not in fields or "PricesElec" not in fields:
                raise ValueError(
                    "CSV must have 'Timestamp' and 'PricesElec' columns "
                    f"(found: {fields})"
                )
            for raw in reader:
                ts = raw["Timestamp"].strip()
                price = parse_eu_float(raw["PricesElec"])
                rows.append((ts, price))
        if not rows:
            raise ValueError("CSV contains no data rows")
        return rows

    # ------------------------------------------------------------------
    # Building load model (same as live dashboard)
    # ------------------------------------------------------------------

    def _build_load_profile(self, n: int, start_hour: float) -> np.ndarray:
        """Generate a sinusoidal building load profile (kWh per hour).

        Uses the same day/night model as ``DataManager.build_load_and_solar``:
        - Night (outside 06:00-18:00): constant at base_load_kw
        - Day (06:00-18:00): base + peak * sin(pi*(h-6)/12)

        Returns kWh per 1-hour slot (since the CSV is hourly).
        """
        load = np.zeros(n)
        for t in range(n):
            h = (start_hour + t) % 24
            if 6 <= h <= 18:
                kw = self._base_load_kw + (
                    self._peak_load_kw - self._base_load_kw
                ) * max(0.0, np.sin(np.pi * (h - 6) / 12))
            else:
                kw = self._base_load_kw
            load[t] = kw * 1.0  # 1 hour per slot → kWh = kW * 1h
        return load

    # ------------------------------------------------------------------
    # Simulation logic
    # ------------------------------------------------------------------

    def _run_simulation(self):
        if self._user_price_rows is None:
            self.status_label.setText("Please load a price CSV first.")
            return
        self._run_btn.setEnabled(False)
        try:
            self._do_simulation()
        except Exception as exc:
            logger.exception("Future simulation failed")
            self.status_label.setText(f"Simulation error: {exc}")
        finally:
            self._run_btn.setEnabled(True)

    def _do_simulation(self):
        self._clear_results()

        self._assets = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        n = len(self._user_price_rows)
        self.status_label.setText(f"Running LP on {n} time slots\u2026")
        QApplication.processEvents()

        # Extract price array (EUR/MWh)
        prices_eur_mwh = np.array([p for _, p in self._user_price_rows])
        timestamps = [ts for ts, _ in self._user_price_rows]

        # Total cost per kWh = spot + distribution fee
        total_price = prices_eur_mwh + EXTRA_COST_EUR_MWH

        # Determine starting hour from the first timestamp
        first_ts = timestamps[0]
        try:
            time_part = first_ts.split(" ")[1] if " " in first_ts else "0:00"
            start_hour = float(time_part.split(":")[0])
        except (IndexError, ValueError):
            start_hour = 0.0

        # Build the building load profile (sinusoidal model)
        base_load = self._build_load_profile(n, start_hour)

        # ── Shiftable loads ─────────────────────────────────────────
        baseline_shiftable = np.zeros(n)
        lp_shiftable = np.zeros(n)
        total_daily_shifted = 0.0

        for asset in self._assets:
            if asset.asset_type != SHIFTABLE_LOAD or not asset.enabled:
                continue
            daily_kwh = asset.daily_energy_kwh
            if daily_kwh <= 0:
                continue
            hourly_max = asset.hourly_max_kwh

            # Scale daily energy to the number of slots
            # (if CSV covers multiple days, scale proportionally)
            n_days = max(1, n / 24)
            total_kwh = daily_kwh * n_days

            baseline_shiftable += total_kwh / n  # uniform baseline
            lp_shiftable += greedy_lp_ramped(
                n, total_price, total_kwh, hourly_max,
                ramp_up_pct=asset.ramp_up_pct_per_hour,
                ramp_down_pct=asset.ramp_down_pct_per_hour,
            )
            total_daily_shifted += total_kwh

        # ── Generators ──────────────────────────────────────────────
        total_gen = np.zeros(n)

        for asset in self._assets:
            if asset.asset_type != GENERATOR or not asset.enabled:
                continue
            gen = np.zeros(n)

            # Simple solar model: assume generation follows a sinusoidal
            # pattern during daylight hours, similar to load profile
            if asset.solar_csv:
                for t in range(n):
                    h = (start_hour + t) % 24
                    if 6 <= h <= 18:
                        gen[t] = max(0.0, np.sin(np.pi * (h - 6) / 12)) * 5.0
                    else:
                        gen[t] = 0.0

            if asset.decouple_below_eur_mwh is not None:
                gen = np.where(
                    total_price >= asset.decouple_below_eur_mwh, gen, 0.0,
                )
            total_gen += gen

        # ── Cost comparison ─────────────────────────────────────────
        baseline_load = base_load + baseline_shiftable
        lp_load = base_load + lp_shiftable

        baseline_grid = baseline_load - total_gen
        lp_grid = lp_load - total_gen

        baseline_cost = baseline_grid * total_price / 1000.0  # EUR
        lp_cost = lp_grid * total_price / 1000.0

        # ── Display ─────────────────────────────────────────────────
        gw = max(780, self.width() - 60)
        labels = []
        for ts in timestamps:
            parts = ts.split(" ")
            labels.append(parts[-1] if len(parts) > 1 else ts)

        self.graph_label.setPixmap(
            draw_comparison_graph(
                baseline_cost, lp_cost, labels,
                width=gw, height=270,
            )
        )
        self.power_graph_label.setPixmap(
            draw_power_comparison_graph(
                baseline_load, lp_load, labels, prices_eur_mwh,
                width=gw, height=270,
            )
        )

        self._display_kpis(
            baseline_cost, lp_cost, total_daily_shifted,
            float(np.sum(total_gen)), n,
        )
        self.status_label.setText(
            f"Simulation complete \u2014 {n} time slots processed."
        )

        # Store for CSV export
        self._last_results = {
            "baseline": baseline_cost,
            "optimised": lp_cost,
            "baseline_load_kwh": baseline_load,
            "optimised_load_kwh": lp_load,
            "baseline_grid_kwh": baseline_grid,
            "optimised_grid_kwh": lp_grid,
            "prices_elec": prices_eur_mwh,
            "load_shifted": total_daily_shifted,
            "total_generation": float(np.sum(total_gen)),
            "n_slots": n,
            "timestamps": timestamps,
        }
        self._export_btn.setEnabled(True)

    def _clear_results(self):
        self.graph_label.clear()
        self.power_graph_label.clear()
        self.kpi_title.setText("Results")
        for lbl in self.kpi_labels.values():
            lbl.setText("--")
            lbl.setStyleSheet(
                "font-family: 'Consolas'; font-weight: 700;"
                " color: #0f766e; border: none;"
            )

    def _display_kpis(self, baseline_cost, lp_cost, shifted, generation, n):
        total_bl = float(baseline_cost.sum())
        total_opt = float(lp_cost.sum())
        saving = total_bl - total_opt
        pct = (saving / total_bl * 100) if total_bl != 0 else 0.0

        colour = "#16a34a" if saving >= 0 else "#dc2626"
        coloured = (
            f"font-family: 'Consolas'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )

        self.kpi_title.setText("Simulation Results")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_bl:.2f}")
        self.kpi_labels["optimised"].setText(f"\u20ac{total_opt:.2f}")
        self.kpi_labels["saving"].setText(f"\u20ac{saving:.2f}")
        self.kpi_labels["saving"].setStyleSheet(coloured)
        self.kpi_labels["saving_pct"].setText(f"{pct:.2f}%")
        self.kpi_labels["saving_pct"].setStyleSheet(coloured)
        self.kpi_labels["slots"].setText(f"{n} hourly")
        self.kpi_labels["load_shifted"].setText(f"{shifted:.1f} kWh")
        self.kpi_labels["total_generation"].setText(f"{generation:.1f} kWh")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Export last simulation results to a CSV file with decision log."""
        if self._last_results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "future_simulation.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        r = self._last_results
        n = r["n_slots"]
        timestamps = r["timestamps"]

        lines: list[str] = []
        lines.append(
            "Timestamp;Baseline_Cost_EUR;Optimised_Cost_EUR;"
            "Baseline_Load_kWh;Optimised_Load_kWh;"
            "Baseline_Grid_kWh;Optimised_Grid_kWh;"
            "Price_EUR_MWh;Decision"
        )
        for i in range(n):
            bl_cost = r["baseline"][i]
            opt_cost = r["optimised"][i]
            bl_load = r["baseline_load_kwh"][i]
            opt_load = r["optimised_load_kwh"][i]
            bl_grid = r["baseline_grid_kwh"][i]
            opt_grid = r["optimised_grid_kwh"][i]
            price = r["prices_elec"][i]

            delta_load = opt_load - bl_load
            if abs(delta_load) < 0.01:
                decision = "No change"
            elif delta_load > 0:
                decision = f"Shifted +{delta_load:.1f} kWh here (cheap hour)"
            else:
                decision = f"Shifted {delta_load:.1f} kWh away (expensive hour)"

            lines.append(
                f"{timestamps[i]};{bl_cost:.4f};{opt_cost:.4f};"
                f"{bl_load:.2f};{opt_load:.2f};"
                f"{bl_grid:.2f};{opt_grid:.2f};"
                f"{price:.2f};{decision}"
            )

        total_bl = float(r["baseline"].sum())
        total_opt = float(r["optimised"].sum())
        saving = total_bl - total_opt
        lines.append("")
        lines.append("# Summary")
        lines.append(f"# Source CSV;{self._user_csv_path}")
        lines.append(f"# Baseline total;EUR {total_bl:.2f}")
        lines.append(f"# Optimised total;EUR {total_opt:.2f}")
        lines.append(f"# Saving;EUR {saving:.2f}")
        lines.append(f"# Load shifted;{r['load_shifted']:.1f} kWh")
        lines.append(f"# Generation;{r['total_generation']:.1f} kWh")
        lines.append("")
        lines.append("# Decision logic: The LP schedules shiftable loads")
        lines.append("# to the cheapest hours using the user-supplied prices,")
        lines.append("# respecting daily energy and ramp-rate constraints.")
        lines.append("# Building load follows a sinusoidal day/night model.")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.status_label.setText(f"Exported to {path}")
        except OSError as exc:
            self.status_label.setText(f"Export failed: {exc}")
