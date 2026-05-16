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
    EnergyAsset, ensure_defaults, load_assets,
)
from graph_renderer import draw_comparison_graph, draw_power_comparison_graph
from lp_solver import simulate_slots, load_solar_csv, parse_eu_float
from settings import DASHBOARD_DIR, LOCAL_TZ
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

_KPI_ROWS = [
    ("baseline",           "Baseline cost"),
    ("optimised",          "LP optimised cost"),
    ("saving",             "Savings"),
    ("saving_pct",         "Savings %"),
    ("heating_saving",     "Heating savings"),
    ("heating_saving_pct", "Heating savings %"),
    ("hw_saving",          "Hot water savings"),
    ("hw_saving_pct",      "Hot water savings %"),
    ("slots",              "Time slots"),
    ("load_shifted",       "Total load shifted"),
    ("total_generation",   "Total on-site generation"),
    ("chp_firing",         "CHP firing hours"),
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
        self._asset_kpi_widgets: list = []

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

        prices_eur_mwh = np.array([p for _, p in self._user_price_rows])
        timestamps = [ts for ts, _ in self._user_price_rows]

        # Determine starting hour from the first timestamp
        first_ts = timestamps[0]
        try:
            time_part = first_ts.split(" ")[1] if " " in first_ts else "0:00"
            start_hour = float(time_part.split(":")[0])
        except (IndexError, ValueError):
            start_hour = 0.0

        # Build the building load profile (sinusoidal model)
        base_load = self._build_load_profile(n, start_hour)

        # Collect unique day-keys and load solar archive if any asset has one
        day_keys = list({ts.split(" ")[0] for ts in timestamps})
        solar_hours: dict = {}
        for asset in self._assets:
            if asset.enabled and asset.solar_csv:
                p = DASHBOARD_DIR / asset.solar_csv
                if p.exists():
                    solar_hours.update(load_solar_csv(p))

        # Run the unified LP simulation (same core as historical dialog)
        results = simulate_slots(
            n=n,
            prices_elec=prices_eur_mwh,
            base_load=base_load,
            assets=[a for a in self._assets if a.enabled],
            solar_hours=solar_hours or None,
            day_keys=day_keys,
            start_hour=start_hour,
            dt_hours=1.0,
        )

        # ── Display ─────────────────────────────────────────────────
        gw = max(780, self.width() - 60)
        labels = []
        for ts in timestamps:
            parts = ts.split(" ")
            labels.append(parts[-1] if len(parts) > 1 else ts)

        self.graph_label.setPixmap(
            draw_comparison_graph(
                results["baseline"], results["optimised"], labels,
                width=gw, height=270,
            )
        )
        self.power_graph_label.setPixmap(
            draw_power_comparison_graph(
                results["baseline_load_kwh"], results["optimised_load_kwh"],
                labels, prices_eur_mwh,
                width=gw, height=270,
            )
        )

        self._display_kpis(results, n)

        self.status_label.setText(
            f"Simulation complete \u2014 {n} time slots processed."
        )

        # Store for CSV export
        results["timestamps"] = timestamps
        self._last_results = results
        self._export_btn.setEnabled(True)

    def _clear_results(self):
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
        for w in self._asset_kpi_widgets:
            w.setParent(None)
            w.deleteLater()
        self._asset_kpi_widgets.clear()

    def _rebuild_asset_kpis(self, results):
        """Dynamically add per-asset saving rows to the KPI card."""
        self._remove_asset_kpi_widgets()

        def _add_section(title):
            sep = QLabel(title)
            sep.setStyleSheet(
                "font-weight: 700; color: #0b3a6e; border: none; padding-top: 6px;"
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

        asset_gens = results.get("asset_generators", {})
        if asset_gens:
            _add_section("\u26a1 Output savings (generators)")
            for name, gd in asset_gens.items():
                gen_total = float(np.sum(gd["gen_kwh"]))
                gen_bl    = float(np.sum(gd.get("gen_kwh_baseline", gd["gen_kwh"])))
                cost_save = gd.get("cost_saving_eur", 0.0)
                fire_h    = int(np.sum(gd["gen_kwh"] > 0))
                _add_row(f"  {name} \u2014 generation",
                         f"{gen_total:.1f} kWh", "#0e7490")
                _add_row(f"  {name} \u2014 baseline generation",
                         f"{gen_bl:.1f} kWh", "#64748b")
                _add_row(f"  {name} \u2014 firing hours",
                         f"{fire_h} h", "#64748b")
                _add_row(f"  {name} \u2014 spark saving",
                         f"\u20ac{cost_save:.2f}",
                         "#16a34a" if cost_save >= 0 else "#dc2626")

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
            heat_bl  = results.get("heating_baseline_cost_eur", 0.0)
            heat_pct = results.get("heating_saving_pct", 0.0)
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

    def _display_kpis(self, results: dict, n: int):
        baseline_cost = results["baseline"]
        lp_cost       = results["optimised"]
        total_bl  = float(baseline_cost.sum())
        total_opt = float(lp_cost.sum())
        saving = total_bl - total_opt
        pct = (saving / total_bl * 100) if total_bl != 0 else 0.0

        colour = "#16a34a" if saving >= 0 else "#dc2626"
        coloured = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {colour}; border: none;"
        )

        self.kpi_title.setText("Simulation Results")
        self.kpi_labels["baseline"].setText(f"\u20ac{total_bl:.2f}")
        self.kpi_labels["optimised"].setText(f"\u20ac{total_opt:.2f}")
        self.kpi_labels["saving"].setText(f"\u20ac{saving:.2f}")
        self.kpi_labels["saving"].setStyleSheet(coloured)
        self.kpi_labels["saving_pct"].setText(f"{pct:.2f}%")
        self.kpi_labels["saving_pct"].setStyleSheet(coloured)

        heat_save = results.get("heating_saving_eur", 0.0)
        heat_pct  = results.get("heating_saving_pct", 0.0)
        heat_colour = "#16a34a" if heat_save >= 0 else "#dc2626"
        heat_coloured = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {heat_colour}; border: none;"
        )
        self.kpi_labels["heating_saving"].setText(f"\u20ac{heat_save:.2f}")
        self.kpi_labels["heating_saving"].setStyleSheet(heat_coloured)
        self.kpi_labels["heating_saving_pct"].setText(f"{heat_pct:.2f}%")
        self.kpi_labels["heating_saving_pct"].setStyleSheet(heat_coloured)

        hw_save = results.get("hw_saving_eur", 0.0)
        hw_pct  = results.get("hw_saving_pct",  0.0)
        hw_colour = "#16a34a" if hw_save >= 0 else "#dc2626"
        hw_coloured = (
            f"font-family: 'Calibri'; font-weight: 700;"
            f" color: {hw_colour}; border: none;"
        )
        self.kpi_labels["hw_saving"].setText(f"\u20ac{hw_save:.2f}")
        self.kpi_labels["hw_saving"].setStyleSheet(hw_coloured)
        self.kpi_labels["hw_saving_pct"].setText(f"{hw_pct:.2f}%")
        self.kpi_labels["hw_saving_pct"].setStyleSheet(hw_coloured)

        self.kpi_labels["slots"].setText(f"{n} hourly")
        self.kpi_labels["load_shifted"].setText(
            f"{results.get('load_shifted', 0.0):.1f} kWh"
        )
        self.kpi_labels["total_generation"].setText(
            f"{results.get('total_generation', 0.0):.1f} kWh"
        )

        # CHP firing hours across all CHP generators
        chp_fire_h = 0
        for gd in results.get("asset_generators", {}).values():
            chp_fire_h += int(np.sum(gd["gen_kwh"] > 0))
        self.kpi_labels["chp_firing"].setText(f"{chp_fire_h} h")

        self._rebuild_asset_kpis(results)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Export last simulation results to a CSV file with per-asset breakdown."""
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

        asset_loads: dict = r.get("asset_loads", {})
        asset_gens: dict  = r.get("asset_generators", {})
        load_names = list(asset_loads.keys())
        gen_names  = list(asset_gens.keys())

        lines: list[str] = []

        # ── Section 1: per-hour detail ──────────────────────────────
        lines.append("# === HOURLY DETAIL ===")
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

            delta = r["optimised_load_kwh"][i] - r["baseline_load_kwh"][i]
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
        lines.append("Asset;Type;Baseline_Total_kWh;Optimised_Total_kWh;Saving_kWh;Configured_Daily_kWh")
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
        lines.append(f"# Source CSV;{self._user_csv_path}")
        lines.append(f"# Baseline total cost;EUR {total_bl:.2f}")
        lines.append(f"# Optimised total cost;EUR {total_opt:.2f}")
        lines.append(f"# Saving;EUR {saving:.2f}")
        lines.append(f"# Load shifted;{r['load_shifted']:.1f} kWh")
        lines.append(f"# Total generation;{r['total_generation']:.1f} kWh")
        lines.append("# Note: Building load uses sinusoidal day/night model.")
        lines.append("# Shiftable loads are scheduled to cheapest hours via LP.")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.status_label.setText(f"Exported to {path}")
        except OSError as exc:
            self.status_label.setText(f"Export failed: {exc}")
