"""
╔══════════════════════════════════════════════════════════════════╗
║  FRONTEND FILE — student is NOT responsible for this module      ║
║                                                                  ║
║  Main PyQt6 window.  Contains only UI layout, widget wiring,    ║
║  and display logic.  All calculations happen in the backend.     ║
╚══════════════════════════════════════════════════════════════════╝

Main Dashboard window — Building Management System.

Run with::

    python dashboard.py

The window has three columns:

* **Left**   — system inputs  (price, solar, CHP, …)
* **Centre** — graphs, KPIs, historical-analysis button
* **Right**  — system outputs (CHP state, ice banks, …)

Architecture
------------
* **UI only** — all layout, widgets, and rendering live here.
* **Data**    — fetched / cached by :class:`data_manager.DataManager`.
* **LP**      — solved by :class:`smpc_calculator.SMPCCalculator`.
"""

import logging
import multiprocessing as _mp
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Must be set before any cvxpy/HIGHS import to prevent OpenMP ↔ Qt thread conflict
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HIGHS_NUM_THREADS", "1")

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from dashboard_config import load_dashboard_config
from data_manager import DataManager, current_slot
from graph_renderer import draw_price_graph, draw_solar_graph, draw_temperature_graph, draw_thermal_graph
from future_dialog import FutureSimulationDialog
from historical_dialog import HistoricalAnalysisDialog
from settings import INTERVAL_MINUTES, LOCAL_TZ
from smpc_calculator import SMPCCalculator
from styles import (
    COLUMN_BUTTON_STYLE, FUTURE_BUTTON_STYLE, HISTORICAL_BUTTON_STYLE,
    MAIN_WINDOW_STYLE, SLOT_SLIDER_STYLE, value_css,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Icon lookup tables
# ---------------------------------------------------------------------------
_ICON_KEY = {
    "plug": "\U0001f50c", "sun": "\u2600\ufe0f", "fire": "\U0001f525",
    "heat": "\u2668\ufe0f", "target": "\U0001f3af", "snowflake": "\U0001f9ca",
    "car": "\U0001f697", "globe": "\U0001f30d",
    "battery": "\U0001f50b", "lightning": "\u26a1", "factory": "\U0001f3ed",
    "house": "\U0001f3e0", "wind": "\U0001f32c\ufe0f", "leaf": "\U0001f33f",
    "gear": "\u2699\ufe0f", "thermo": "\U0001f321\ufe0f",
}
_ICON_LEGACY = {
    "GRID": "\U0001f50c", "Solar Panels": "\u2600\ufe0f",
    "CHP": "\U0001f525", "Heat Pump": "\u2668\ufe0f",
    "Setpoint": "\U0001f3af", "Ice Banks": "\U0001f9ca",
    "EVs": "\U0001f697", "BEO": "\U0001f30d",
}

SIDE_COL_MIN_W = 260


# ===================================================================
# Async subprocess helper for LP solve
# (HIGHS DLL conflicts with Qt on Windows when both are in the same process.
#  multiprocessing.Pool(spawn) re-imports dashboard.py → Qt loads in worker
#  → same conflict.  Solution: use a standalone _lp_worker.py script via
#  subprocess.Popen so Qt is never imported in the solver process.)
# ===================================================================

_LP_WORKER = Path(__file__).parent / "_lp_worker.py"


class _AsyncSubprocessResult:
    """Run _lp_worker.py in a subprocess; expose .ready() / .get() interface."""

    def __init__(self, *args):
        import pickle
        import subprocess
        import threading

        self._result = None
        self._error: Exception | None = None
        self._done = threading.Event()

        payload = pickle.dumps(args)

        def _run():
            try:
                proc = subprocess.run(
                    [sys.executable, str(_LP_WORKER)],
                    input=payload,
                    capture_output=True,
                    timeout=120,
                )
                if proc.returncode == 0:
                    self._result = pickle.loads(proc.stdout)
                else:
                    self._error = RuntimeError(
                        f"LP worker exited {proc.returncode}: "
                        f"{proc.stderr.decode(errors='replace')[:500]}"
                    )
            except Exception as exc:
                self._error = exc
            finally:
                self._done.set()

        threading.Thread(target=_run, daemon=True).start()

    def ready(self) -> bool:
        return self._done.is_set()

    def get(self):
        if self._error:
            raise self._error
        return self._result


# ===================================================================
# Main window
# ===================================================================

class ScadaWindow(QMainWindow):
    """Three-column SCADA-style dashboard."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thesis \u2014 Building Management System")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        self.resize(1400, 900)

        self._init_state()
        self._init_data()
        self._init_lp()
        self._build_ui()
        self._start_timer()

        # First data load
        self.data.refresh_all(force=True)
        self._update_all_widgets()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Declare widget references and tag bookkeeping."""
        self.value_labels: dict = {}
        self.tag_definitions: dict = {"input": {}, "output": {}}

        # Slot preview slider (0 = Now, negative = past, positive = future 15-min steps)
        self._selected_step: int = 0
        self._lp_now_idx:    int = 0   # steps from midnight to current slot in the LP plan

        # Widget references (populated during UI build)
        self.clock_label = None
        self.price_current_label = None
        self.price_avg_label = None
        self.predicted_solar_label = None
        self.center_price_graph_label = None
        self.center_solar_graph_label = None
        self.center_temp_graph_label = None
        self.center_uv_value_label = None
        self.center_solar_value_label = None
        self.slot_slider = None
        self.slot_label = None

        # ── Next-day LP solve state ──────────────────────────────────
        self.last_lp_next_day_outputs = None     # SMPCOutputs for tomorrow
        self._nd_lp_async_result = None          # pending subprocess result
        self._nd_lp_submit_time  = None          # monotonic time of submit
        self._nd_lp_last_date: str | None = None # date string "YYYY-MM-DD" already solved

    def _init_data(self):
        """Create the DataManager (handles all fetching and caching).

        BACKEND CALL: DataManager lives in data_manager.py (backend).
        The dashboard never fetches data directly — it only reads from this object.
        """
        self.data = DataManager()

    def _init_lp(self):
        """Create the LP calculator and load building parameters.

        BACKEND CALL: SMPCCalculator lives in smpc_calculator.py (backend).
        base_load_kw / peak_load_kw stay in 'smpc.building' for DataManager
        compatibility.  Building thermal + HW state come from the 'mpc' block.
        """
        raw = load_dashboard_config()
        building_smpc = raw.get("smpc", {}).get("building", {})
        self.base_load_kw = building_smpc.get("base_load_kw", 80)
        self.peak_load_kw = building_smpc.get("peak_load_kw", 200)

        mpc_raw = raw.get("mpc", {})
        self._building_temp_c      = float(mpc_raw.get("building", {}).get("T_init_c", 20.0))
        self._building_setpoint_c  = float(mpc_raw.get("building", {}).get("Tset_c",   21.0))
        self._hw_temp_c            = float(mpc_raw.get("hot_water_tank", {}).get("T_init_c", 55.0))

        _cfg_abs = str(Path(__file__).parent / "dashboard_config.json")
        self.lp = SMPCCalculator(_cfg_abs)
        self.lp_last_slot = None
        self.last_lp_outputs = None
        self.last_lp_next_day_outputs = None
        self._lp_async_result = None   # pending AsyncResult from the pool
        self._lp_submit_time  = None   # monotonic time of last submit (for timeout warning)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the three-column layout and populate from energy assets."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        cols = QHBoxLayout()
        cols.setSpacing(14)
        root.addLayout(cols)

        # Left — inputs
        self.input_col, self.input_content = self._make_column(
            "System inputs", "#004d40",
        )
        self.input_col.setMinimumWidth(SIDE_COL_MIN_W)
        self.input_col.setMaximumWidth(360)
        cols.addWidget(self.input_col, stretch=0)

        # Centre — graphs / KPIs
        cols.addWidget(self._build_center_panel(), stretch=1)

        # Right — outputs
        self.output_col, self.output_content = self._make_column(
            "System outputs", "#4d1a1a",
        )
        self.output_col.setMinimumWidth(SIDE_COL_MIN_W)
        self.output_col.setMaximumWidth(360)
        cols.addWidget(self.output_col, stretch=0)

        self._populate_columns()

        # "⊕ Manage Assets" buttons (one per column, both open same dialog)
        for container in (self.input_col, self.output_col):
            btn = QPushButton("⊕  Manage Assets")
            btn.setMinimumHeight(52)
            btn.setMinimumWidth(50)
            btn.setStyleSheet(COLUMN_BUTTON_STYLE)
            btn.clicked.connect(self._open_mpc_asset_manager)
            container.layout().addWidget(btn)



    def _populate_columns(self):
        """Build input/output tag rows from asset_instances config.

        Layout
        ------
        INPUTS:   Time & Price | Generation (PV, CHP) | Heating (COP, Bld-temp, Tank)
        OUTPUTS:  Heating (HP kW, Boiler kW) | Hot Water | Flex | Storage | Grid
        """
        from dashboard_config import load_dashboard_config as _ldc
        raw      = _ldc()
        instances = raw.get("mpc", {}).get("asset_instances", [])

        # Fallback: if no instances defined yet, derive from mpc_cfg flags
        if not instances:
            cfg = self.lp.mpc_cfg
            if cfg.pv_enabled:
                instances.append({"id":"pv_1","type":"pv","name":"Solar Panels","enabled":True,"baseline_mode":"always_off","baseline_power_kw":0.0})
            if cfg.hp_enabled:
                instances.append({"id":"hp_1","type":"heat_pump","name":"Heat Pump","enabled":True,"baseline_mode":"constant","baseline_power_kw":30.0})
            if cfg.boiler_enabled:
                instances.append({"id":"boiler_1","type":"gas_boiler","name":"Gas Boiler","enabled":True,"baseline_mode":"constant","baseline_power_kw":50.0})
            if cfg.chp_enabled:
                instances.append({"id":"chp_1","type":"chp","name":"CHP","enabled":True,"baseline_mode":"always_off","baseline_power_kw":0.0})
            if cfg.bat_enabled:
                instances.append({"id":"bat_1","type":"battery","name":"Battery","enabled":True,"baseline_mode":"always_off","baseline_power_kw":0.0})
            if cfg.flex_enabled:
                instances.append({"id":"flex_1","type":"flex","name":"Flex Load","enabled":True,"baseline_mode":"always_off","baseline_power_kw":0.0})
            if cfg.hw_enabled:
                instances.append({"id":"hw_1","type":"hot_water","name":"Hot Water Tank","enabled":True,"baseline_mode":"constant","baseline_power_kw":2.0})

        enabled = [i for i in instances if i.get("enabled", True)]

        input_tags:  list[dict] = []
        output_tags: list[dict] = []

        # ── INPUT: GENERATION ─────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            iname = inst.get("name", iid)
            if itype == "pv":
                input_tags.append({
                    "id": f"{iid}_solar", "name": "Solar Predicted", "icon": "sun",
                    "section": "GENERATION",
                    "simulation": {"mode": "predicted_solar",
                                   "unit": "kW", "decimals": 1, "color": "#f59e0b"},
                })
            elif itype == "chp":
                input_tags.append({
                    "id": f"{iid}_elec_kw", "name": f"{iname} Elec", "icon": "lightning",
                    "section": "GENERATION",
                    "simulation": {"mode": "mpc_scalar", "field": "chp_elec_kw",
                                   "plan_field": "plan_chp_elec_kw",
                                   "unit": "kW", "decimals": 1, "color": "#0e7490"},
                })
                input_tags.append({
                    "id": f"{iid}_heat_kw", "name": f"{iname} Heat", "icon": "fire",
                    "section": "GENERATION",
                    "simulation": {"mode": "mpc_scalar", "field": "chp_heat_kw",
                                   "plan_field": "plan_chp_heat_kw",
                                   "unit": "kW", "decimals": 1, "color": "#f97316"},
                })

        # ── INPUT: HEATING ─────────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            if itype == "heat_pump":
                input_tags.append({
                    "id": f"{iid}_cop", "name": "Heat Pump COP", "icon": "gear",
                    "section": "HEATING",
                    "simulation": {"mode": "mpc_scalar", "field": "cop_now",
                                   "unit": "", "decimals": 2, "color": "#16a34a",
                                   "plan_field": "plan_COP"},
                })

        # Building temperature always shown in HEATING
        input_tags.append({
            "id": "bld_temp", "name": "Building Temp", "icon": "thermo",
            "section": "HEATING",
            "simulation": {"mode": "mpc_building_temp",
                           "plan_field": "plan_building_temp_c"},
        })

        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            if itype == "hot_water":
                input_tags.append({
                    "id": f"{iid}_temp", "name": "Water Tank", "icon": "thermo",
                    "section": "HEATING",
                    "simulation": {"mode": "mpc_scalar", "field": "hw_temp_c",
                                   "unit": "\u00b0C", "decimals": 1, "color": "#0891b2",
                                   "plan_field": "plan_hw_temp_c"},
                })

        # ── OUTPUT: HEATING ────────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            if itype == "heat_pump":
                output_tags.append({
                    "id": f"{iid}_kw", "name": "Heat Pump", "icon": "heat",
                    "section": "HEATING",
                    "simulation": {"mode": "asset_power", "uid": "heat_pump",
                                   "unit": "kW", "decimals": 1, "color": "#f59e0b"},
                })
            elif itype == "gas_boiler":
                output_tags.append({
                    "id": f"{iid}_kw", "name": "Gas Boiler", "icon": "fire",
                    "section": "HEATING",
                    "simulation": {"mode": "asset_power", "uid": "gas_boiler",
                                   "unit": "kW", "decimals": 1, "color": "#ef4444"},
                })

        # ── OUTPUT: HOT WATER ──────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            if itype == "hot_water":
                output_tags.append({
                    "id": f"{iid}_heater_kw", "name": "HW Heater", "icon": "lightning",
                    "section": "HOT WATER",
                    "simulation": {"mode": "asset_power", "uid": "hot_water_tank",
                                   "unit": "kW", "decimals": 1, "color": "#dc2626"},
                })

        # ── OUTPUT: FLEX ───────────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            iname = inst.get("name", iid)
            if itype == "flex":
                output_tags.append({
                    "id": f"{iid}_kw", "name": iname, "icon": "lightning",
                    "section": "FLEX",
                    "simulation": {"mode": "asset_power", "uid": "flexible_load",
                                   "unit": "kW", "decimals": 1, "color": "#0891b2"},
                })

        # ── OUTPUT: STORAGE ────────────────────────────────────────────────
        for inst in enabled:
            iid   = inst["id"]
            itype = inst.get("type", "")
            iname = inst.get("name", iid)
            if itype == "battery":
                output_tags.append({
                    "id": f"{iid}_status", "name": iname, "icon": "battery",
                    "section": "STORAGE",
                    "simulation": {"mode": "mpc_battery_status"},
                })

        # ── OUTPUT: GRID ───────────────────────────────────────────────────
        output_tags += [
            {
                "id": "grid_import", "name": "Grid Import", "icon": "plug",
                "section": "GRID",
                "simulation": {"mode": "mpc_scalar", "field": "pgrid_kw",
                               "unit": "kW", "decimals": 1, "color": "#dc2626",
                               "plan_field": "plan_pgrid_kw"},
            },
            {
                "id": "cost_saving", "name": "Saving Today", "icon": "leaf",
                "section": "GRID",
                "simulation": {"mode": "mpc_scalar", "field": "cost_saving_eur",
                               "unit": "\u20ac", "decimals": 3, "color": "#16a34a"},
            },
        ]

        norm_in  = self._normalise_tags(input_tags)
        norm_out = self._normalise_tags(output_tags)
        self._register_definitions("input",  norm_in)
        self._register_definitions("output", norm_out)

        self._add_price_widgets(self.input_content)
        self._add_grouped_tags(
            self.input_content,
            self._group_by_section(norm_in, "INPUTS"),
            "input",
        )
        self._add_grouped_tags(
            self.output_content,
            self._group_by_section(norm_out, "OUTPUTS"),
            "output",
        )

    def _make_column(self, title: str, colour: str):
        """Create a scrollable side column with a coloured header."""
        frame = QFrame()
        frame.setObjectName("ColumnContainer")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel(title)
        header.setObjectName("ColumnHeader")
        header.setStyleSheet(f"background-color: {colour};")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(12)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        return frame, content

    def _build_center_panel(self) -> QFrame:
        """Construct the centre panel: graphs, metrics, analysis button."""
        panel = QFrame()
        panel.setObjectName("ColumnContainer")
        outer_lay = QVBoxLayout(panel)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        title = QLabel(
            "SYSTEM OVERVIEW / SCHEMATIC",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        title.setObjectName("CenterTitle")
        lay.addWidget(title)

        # Price graph card
        self.center_price_graph_label = self._graph_card(
            lay, "Price Graph (48h)", "Loading price graph\u2026",
        )
        # Solar graph card
        self.center_solar_graph_label = self._graph_card(
            lay, "Predicted Solar Graph", "Loading solar graph\u2026",
        )
        # Temperature / thermal graph card (taller to fit heating bars)
        self.center_temp_graph_label = self._graph_card(
            lay, "Outside Temperature (48h)", "Loading temperature graph\u2026",
            height=280,
        )

        # Metrics card
        metrics = QFrame()
        metrics.setObjectName("TagRow")
        ml = QVBoxLayout(metrics)
        ml.setContentsMargins(12, 10, 12, 10)
        ml.setSpacing(8)
        self.center_uv_value_label = self._metric_row(ml, "UV Index", "--")
        self.center_solar_value_label = self._metric_row(
            ml, "Estimated Solar", "-- kW",
        )
        lay.addWidget(metrics)

        # ── Time-slot preview slider ──────────────────────────────
        slider_card = QFrame()
        slider_card.setObjectName("TagRow")
        sl = QVBoxLayout(slider_card)
        sl.setContentsMargins(12, 10, 12, 10)
        sl.setSpacing(6)

        slider_header = QHBoxLayout()
        slider_title = QLabel("\U0001f55b  Preview time slot")
        slider_title.setObjectName("TagName")
        self.slot_label = QLabel("Now")
        self.slot_label.setObjectName("TagValue")
        slider_header.addWidget(slider_title)
        slider_header.addStretch()
        slider_header.addWidget(self.slot_label)
        sl.addLayout(slider_header)

        self.slot_slider = QSlider(Qt.Orientation.Horizontal)
        self.slot_slider.setMinimum(0)
        _horizon = getattr(getattr(self, "lp", None), "mpc_cfg", None)
        _max = _horizon.horizon_steps - 1 if _horizon else 23
        self.slot_slider.setMaximum(_max)
        self.slot_slider.setValue(0)
        self.slot_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slot_slider.setTickInterval(4)  # tick every hour
        self.slot_slider.setStyleSheet(SLOT_SLIDER_STYLE)
        self.slot_slider.valueChanged.connect(self._on_slot_slider_changed)
        sl.addWidget(self.slot_slider)

        reset_btn = QPushButton("Reset to Now")
        reset_btn.setMinimumHeight(28)
        reset_btn.setStyleSheet(
            "QPushButton { background: #475569; color: white; border: none; "
            "border-radius: 4px; padding: 4px 12px; font-size: 9pt; } "
            "QPushButton:hover { background: #334155; }"
        )
        reset_btn.clicked.connect(lambda: self.slot_slider.setValue(0))
        sl.addWidget(reset_btn)

        lay.addWidget(slider_card)

        # Historical analysis button
        hist_btn = QPushButton("Historical LP Analysis")
        hist_btn.setMinimumHeight(44)
        hist_btn.setStyleSheet(HISTORICAL_BUTTON_STYLE)
        hist_btn.clicked.connect(self._open_historical)
        lay.addWidget(hist_btn)

        # Future simulation button
        future_btn = QPushButton("Future LP Simulation")
        future_btn.setMinimumHeight(44)
        future_btn.setStyleSheet(FUTURE_BUTTON_STYLE)
        future_btn.clicked.connect(self._open_future)
        lay.addWidget(future_btn)

        lay.addStretch()

        scroll.setWidget(content)
        outer_lay.addWidget(scroll)
        return panel

    def _on_slot_slider_changed(self, value: int):
        """Handle slider position change — updates all three panels."""
        self._selected_step = value
        if value == 0:
            self.slot_label.setText("Now")
            self.slot_label.setStyleSheet(value_css("#0e7490"))
        elif value > 0:
            target = current_slot() + timedelta(minutes=value * INTERVAL_MINUTES)
            self.slot_label.setText(f"+{value}  ({target.strftime('%H:%M')})")
            self.slot_label.setStyleSheet(value_css("#7c3aed"))
        else:
            target = current_slot() + timedelta(minutes=value * INTERVAL_MINUTES)
            self.slot_label.setText(f"{value}  ({target.strftime('%H:%M')})")
            self.slot_label.setStyleSheet(value_css("#94a3b8"))
        # Update ALL panels (inputs, outputs, centre)
        self._update_tag_labels()
        self._update_price_for_slot()
        self._update_center()

    # ------------------------------------------------------------------
    # Reusable widget builders
    # ------------------------------------------------------------------

    @staticmethod
    def _graph_card(parent_layout, title_text: str, placeholder: str, height: int = 200) -> QLabel:
        """Add a titled graph card to *parent_layout*; return the graph label."""
        card = QFrame()
        card.setObjectName("TagRow")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(8)
        t = QLabel(title_text)
        t.setObjectName("TagName")
        cl.addWidget(t)
        lbl = QLabel(placeholder)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFixedHeight(height)
        lbl.setObjectName("TagValue")
        cl.addWidget(lbl)
        parent_layout.addWidget(card)
        return lbl

    @staticmethod
    def _metric_row(parent_layout, name: str, default: str) -> QLabel:
        """Add an inline name/value row; return the value label."""
        row = QHBoxLayout()
        n = QLabel(name)
        n.setObjectName("TagName")
        v = QLabel(default)
        v.setObjectName("TagValue")
        row.addWidget(n)
        row.addStretch()
        row.addWidget(v)
        parent_layout.addLayout(row)
        return v

    @staticmethod
    def _info_row(container, name: str, default: str) -> QLabel:
        """Add a single info row (icon + label + value); return value label."""
        row = QFrame()
        row.setObjectName("TagRow")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(12, 8, 12, 8)
        n = QLabel(name)
        n.setObjectName("TagName")
        n.setWordWrap(True)
        n.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        v = QLabel(default)
        v.setObjectName("TagValue")
        rl.addWidget(n)
        rl.addStretch()
        rl.addWidget(v)
        container.layout().addWidget(row)
        return v

    def _add_price_widgets(self, container):
        """Add clock, current-price and 48 h average rows."""
        self.clock_label = self._info_row(
            container, "\U0001f552  Local time", "--",
        )
        self.price_current_label = self._info_row(
            container, "\u26a1  Current 15m price", "-- EUR/MWh",
        )
        self.price_avg_label = self._info_row(
            container, "\U0001f4c8  48h average", "-- EUR/MWh",
        )
        self._update_clock()

    # ------------------------------------------------------------------
    # Tag handling
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_name(tag) -> str:
        if isinstance(tag, dict):
            return str(tag.get("name", tag.get("id", "Unnamed")))
        return str(tag)

    @staticmethod
    def _tag_id(tag) -> str:
        if isinstance(tag, dict):
            return str(tag.get("id", tag.get("name", "Unnamed")))
        return str(tag)

    @staticmethod
    def _normalise_tags(items: list) -> list[dict]:
        """De-duplicate and normalise raw config tag entries."""
        result, seen = [], set()
        for item in items:
            if isinstance(item, str):
                item = {"id": item, "name": item}
            elif isinstance(item, dict):
                item = dict(item)
                item.setdefault("name", item.get("id", "Unnamed"))
                item.setdefault("id", item["name"])
            else:
                continue
            key = ScadaWindow._tag_id(item).strip().lower()
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result

    def _register_definitions(self, tag_type: str, items: list[dict]):
        for item in items:
            self.tag_definitions[tag_type][self._tag_id(item)] = item

    @staticmethod
    def _group_by_section(items, default):
        groups: dict[str, list] = {}
        for item in items:
            sec = str(item.get("section", default)).strip() or default
            groups.setdefault(sec, []).append(item)
        return list(groups.items())

    def _resolve_icon(self, tag, tag_type: str) -> str:
        key = ""
        if isinstance(tag, dict):
            key = str(tag.get("icon", "")).strip().lower()
        if key in _ICON_KEY:
            return _ICON_KEY[key]
        name = self._tag_name(tag)
        return _ICON_LEGACY.get(
            name, "\U0001f7e2" if tag_type == "input" else "\u2699\ufe0f",
        )

    def _add_tags_to_column(self, container, tags, tag_type):
        for tag in tags:
            name = self._tag_name(tag)
            tid = self._tag_id(tag)
            icon = self._resolve_icon(tag, tag_type)

            row = QFrame()
            row.setMinimumHeight(42)
            row.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
            )
            row.setObjectName("TagRow")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 8, 12, 8)

            nlbl = QLabel(f"{icon}  {name}")
            nlbl.setObjectName("TagName")
            nlbl.setWordWrap(True)
            nlbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
            )
            vlbl = QLabel("0.0")
            vlbl.setObjectName("TagValue")

            rl.addWidget(nlbl)
            rl.addStretch()
            rl.addWidget(vlbl)

            props = tag.get("properties", {}) if isinstance(tag, dict) else {}
            if props:
                row.setToolTip(str(props))

            container.layout().addWidget(row)
            self.value_labels[(tag_type, tid)] = vlbl

    def _add_grouped_tags(self, container, grouped, tag_type):
        first = True
        for section, tags in grouped:
            if not tags:
                continue
            if not first:
                div = QFrame()
                div.setObjectName("SectionDivider")
                container.layout().addWidget(div)
            hdr = QLabel(section)
            hdr.setObjectName("SectionHeader")
            container.layout().addWidget(hdr)
            self._add_tags_to_column(container, tags, tag_type)
            first = False

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_timer(self):
        """Start the 1-second tick timer that drives all live updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(1000)

    # ------------------------------------------------------------------
    # Main tick (called every second by QTimer)
    # ------------------------------------------------------------------

    def _on_tick(self):
        """Central update loop — runs every second.

        ORDER OF OPERATIONS (important):
        1. Update clock display (cheap, always runs)
        2. DataManager.tick() — check if data refresh is due (backend)
        3. _collect_lp_result() — pick up finished async LP result (non-blocking)
        4. _run_lp_if_needed() — submit new LP solve to subprocess (non-blocking)
        5. _update_all_widgets() — propagate results to the UI labels (frontend)
        """
        self._update_clock()
        self.data.tick()               # may refresh prices/predictions/weather
        self._collect_lp_result()      # grab result if the subprocess finished
        self._collect_next_day_lp_result()   # grab next-day result if ready
        self._run_lp_if_needed()       # submit async LP once per 15-min slot
        self._run_next_day_lp_if_needed()    # submit next-day LP when prices available
        self._update_all_widgets()

    # ------------------------------------------------------------------
    # Widget updates (pure UI — reads from DataManager)
    # ------------------------------------------------------------------

    def _update_all_widgets(self):
        self._update_price_for_slot()
        self._update_predict_labels()
        self._update_center()
        self._update_tag_labels()

    def _update_price_labels(self):
        dm = self.data
        if dm.current_price is None or dm.avg_48h is None:
            return
        if self.price_current_label:
            self.price_current_label.setText(f"{dm.current_price:.2f} EUR/MWh")
            colour = "#16a34a" if dm.current_price < dm.avg_48h else "#dc2626"
            self.price_current_label.setStyleSheet(value_css(colour))
        if self.price_avg_label:
            self.price_avg_label.setText(f"{dm.avg_48h:.2f} EUR/MWh")
            self.price_avg_label.setStyleSheet(value_css("#0f766e"))

    def _update_price_for_slot(self):
        """Update the current price label for the selected future slot."""
        if not self.price_current_label:
            return
        dm = self.data
        if self._selected_step == 0:
            self._update_price_labels()
            return
        from datetime import timedelta
        target = current_slot() + timedelta(minutes=self._selected_step * INTERVAL_MINUTES)
        price = None
        for ts, p, _ in dm.price_rows:
            if ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0) == target:
                price = p
                break
        if price is not None and dm.avg_48h is not None:
            self.price_current_label.setText(f"{price:.2f} EUR/MWh")
            colour = "#16a34a" if price < dm.avg_48h else "#dc2626"
            self.price_current_label.setStyleSheet(value_css(colour))
        else:
            self.price_current_label.setText("-- EUR/MWh")
            self.price_current_label.setStyleSheet(value_css("#64748b"))

    def _update_predict_labels(self):
        dm = self.data
        if self.predicted_solar_label:
            if dm.current_power_kw is None:
                self.predicted_solar_label.setText("-- kW")
                self.predicted_solar_label.setStyleSheet(value_css("#64748b"))
            else:
                self.predicted_solar_label.setText(f"{dm.current_power_kw:.1f} kW")
                colour = "#f59e0b" if dm.current_power_kw > 0 else "#64748b"
                self.predicted_solar_label.setStyleSheet(value_css(colour))

    # ------------------------------------------------------------------
    # Centre panel updates
    # ------------------------------------------------------------------

    def _update_center(self):
        slot = current_slot()
        dm = self.data

        # ── Shared 48-h timeline: today 00:00 … tomorrow 23:45 in 15-min steps ──
        now_local = datetime.now(LOCAL_TZ)
        today_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        n_steps_48h = 2 * 24 * (60 // INTERVAL_MINUTES)   # 192 for 15-min slots
        shared_times = [
            today_midnight + timedelta(minutes=i * INTERVAL_MINUTES)
            for i in range(n_steps_48h)
        ]
        shared_keys = [t.strftime("%Y-%m-%d %H:%M:%S") for t in shared_times]
        start_label = shared_times[0].strftime("%d/%m %H:%M")
        end_label   = shared_times[-1].strftime("%d/%m %H:%M")

        # Current-slot index and selected-slot index in the shared timeline
        now_idx = next(
            (i for i, t in enumerate(shared_times) if t == slot), None
        )
        if now_idx is None:
            # Clamp to closest past entry
            for i, t in reversed(list(enumerate(shared_times))):
                if t <= slot:
                    now_idx = i
                    break

        target_ts = slot + timedelta(minutes=self._selected_step * INTERVAL_MINUTES)
        sel_idx = next(
            (i for i, t in enumerate(shared_times) if t == target_ts), None
        )
        if sel_idx is None:
            # Clamp to closest past entry for the selected slot
            for i, t in reversed(list(enumerate(shared_times))):
                if t <= target_ts:
                    sel_idx = i
                    break

        # Price graph — map price_rows onto the shared timeline
        if self.center_price_graph_label is not None:
            price_map: dict = {}
            for ts, p, _ in dm.price_rows:
                ts_loc = ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0)
                price_map[ts_loc.strftime("%Y-%m-%d %H:%M:%S")] = p
            # Forward-fill: each slot gets the most recent known price
            last_p = None
            prices = []
            for key in shared_keys:
                if key in price_map:
                    last_p = price_map[key]
                prices.append(last_p if last_p is not None else float("nan"))
            gw = max(self.center_price_graph_label.width(), 300)
            gh = self.center_price_graph_label.height() or 200
            self.center_price_graph_label.setPixmap(
                draw_price_graph(prices, start_label, end_label,
                                 now_idx, sel_idx, gw, gh),
            )
            # Slider range: from midnight (negative) to end of tomorrow (positive)
            if now_idx is not None:
                max_future = n_steps_48h - 1 - now_idx
                min_past   = -now_idx
                if self.slot_slider.maximum() != max_future:
                    self.slot_slider.setMaximum(max(max_future, 1))
                if self.slot_slider.minimum() != min_past:
                    self.slot_slider.setMinimum(min_past)

        # Solar graph — map predictions onto the shared timeline
        if self.center_solar_graph_label is not None:
            solar_values = [
                dm.predictions[key][0] if key in dm.predictions else 0.0
                for key in shared_keys
            ]
            gw = max(self.center_solar_graph_label.width(), 300)
            gh = self.center_solar_graph_label.height() or 200
            self.center_solar_graph_label.setPixmap(
                draw_solar_graph(solar_values, start_label, end_label,
                                 now_idx, sel_idx, gw, gh),
            )

        # Temperature / heating graph — overlay LP building temp + heating sources
        if self.center_temp_graph_label is not None:
            # Outside temperature (forward-filled from weather.csv)
            temp_values = []
            last_t = None
            for key in shared_keys:
                v = dm.temp_data.get(key)
                if v is not None:
                    last_t = v
                temp_values.append(last_t if last_t is not None else float("nan"))

            # Helper: build a 192-step list from today's + tomorrow's LP plan arrays
            H    = self.lp.mpc_cfg.horizon_steps  # typically 96
            dt_h = self.lp.mpc_cfg.dt_hours        # step duration [h]
            # Each LP step spans n_sub fifteen-minute slots on the shared timeline.
            # e.g. dt=0.25h → n_sub=1;  dt=1.0h → n_sub=4
            _n_sub = max(1, round(dt_h * 60 / INTERVAL_MINUTES))

            def _lp_plan_48h(attr: str, scale: float = 1.0) -> list[float]:
                """Concatenate today's and tomorrow's plan array into 192 fifteen-min slots.

                Each LP step of dt_hours is expanded to n_sub consecutive 15-min slots
                so the graph always covers the correct wall-clock duration.
                """
                result = [float("nan")] * n_steps_48h
                for offs, out in (
                    (0,  self.last_lp_outputs),
                    (96, self.last_lp_next_day_outputs),
                ):
                    if out is None:
                        continue
                    arr = getattr(out, attr, None)
                    if arr is None or len(arr) == 0:
                        continue
                    for j, v in enumerate(arr):
                        for s in range(_n_sub):
                            idx = offs + j * _n_sub + s
                            if idx < n_steps_48h:
                                result[idx] = float(v) * scale
                return result

            bld_temps   = _lp_plan_48h("plan_building_temp_c")
            heat_chp_kw = _lp_plan_48h("plan_chp_heat_kw")

            # asset_schedules values are in kWh/step; convert to kW: ÷ dt_hours
            _kw_factor = 1.0 / max(dt_h, 1e-9)
            # asset_schedules is a dict, not a plain array — handle separately
            heat_hp_kw     = [float("nan")] * n_steps_48h
            heat_boiler_kw = [float("nan")] * n_steps_48h
            for offs, out in (
                (0,  self.last_lp_outputs),
                (96, self.last_lp_next_day_outputs),
            ):
                if out is None:
                    continue
                for result_list, sched_key in (
                    (heat_hp_kw,     "heat_pump"),
                    (heat_boiler_kw, "gas_boiler"),
                ):
                    sched = out.asset_schedules.get(sched_key)
                    if sched is None:
                        continue
                    for j, v in enumerate(sched):   # use full array length
                        for s in range(_n_sub):
                            idx = offs + j * _n_sub + s
                            if idx < n_steps_48h:
                                result_list[idx] = float(v) * _kw_factor

            setpoint = (
                self.last_lp_outputs.building_setpoint_c
                if self.last_lp_outputs is not None else 21.0
            )
            gw = max(self.center_temp_graph_label.width(), 300)
            gh = self.center_temp_graph_label.height() or 280
            self.center_temp_graph_label.setPixmap(
                draw_thermal_graph(
                    temp_values, bld_temps, setpoint,
                    heat_hp_kw, heat_boiler_kw, heat_chp_kw,
                    start_label, end_label, now_idx, sel_idx, gw, gh,
                ),
            )

        # Scalar metrics
        if self.center_uv_value_label:
            self.center_uv_value_label.setText(
                "--" if dm.current_uv is None else f"{dm.current_uv:.2f}",
            )
        if self.center_solar_value_label:
            if dm.current_power_kw is None:
                self.center_solar_value_label.setText("-- kW")
            else:
                self.center_solar_value_label.setText(
                    f"{dm.current_power_kw:.1f} kW",
                )

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def _update_clock(self):
        if self.clock_label is None:
            return
        self.clock_label.setText(
            datetime.now(LOCAL_TZ).strftime("%d/%m/%Y %H:%M:%S"),
        )
        self.clock_label.setStyleSheet(value_css("#0e7490"))

    def _update_tag_labels(self):
        for (ttype, tid), label in self.value_labels.items():
            defn = self.tag_definitions.get(ttype, {}).get(tid, {})
            text, colour = self._sim_value(defn)
            label.setStyleSheet(value_css(colour))
            label.setText(text)

    # ------------------------------------------------------------------
    # LP integration (called once per 15-min slot)
    # ------------------------------------------------------------------

    def _collect_lp_result(self):
        """Pick up a finished async LP result without blocking."""
        if self._lp_async_result is None:
            return
        # Warn if subprocess hasn't returned after 30 seconds
        submit_time = getattr(self, "_lp_submit_time", None)
        if submit_time is not None:
            elapsed = __import__("time").monotonic() - submit_time
            if elapsed > 30 and not self._lp_async_result.ready():
                logger.warning("LP subprocess still running after %.0f s — possible hang", elapsed)
                self._lp_submit_time = None   # warn once
        if not self._lp_async_result.ready():
            return                     # subprocess still solving — check next tick
        try:
            result = self._lp_async_result.get()
            self.last_lp_outputs = result
            # Reset next-day date guard so it re-solves with the updated
            # end-of-day temperature as the correct initial condition.
            self._nd_lp_last_date = None
            logger.info(
                "LP result collected: status=%s solver=%s",
                getattr(result, "solver_status", "?"),
                getattr(result, "solver_used", "?"),
            )
        except Exception as exc:
            logger.warning("LP result error: %s", exc)
        finally:
            self._lp_async_result = None

    def _run_lp_if_needed(self):
        """Run the LP solver at the start of each new 15-minute interval.

        BACKEND CALL: SMPCCalculator.solve_lp() in smpc_calculator.py.

        This implements the RECEDING HORIZON principle of MPC:
          - At every 15-min slot boundary, re-solve the full 24 h problem.
          - Execute only the FIRST-step command (self.last_lp_outputs.* at step 0).
          - Repeat at the next slot with fresh measurements and forecasts.

        Input data flow:
          DataManager.build_price_forecast() → EUR/kWh array [96 steps]
          DataManager.build_load_and_solar() → (base_load_kwh, solar_kwh) arrays
          SMPCCalculator.solve_lp()          → SMPCOutputs (schedules + KPIs)
        """
        slot = current_slot()
        if self.lp_last_slot == slot:
            return                     # already solved for this interval
        self.lp_last_slot = slot

        H    = self.lp.mpc_cfg.horizon_steps   # MPC horizon steps
        dt_h = self.lp.mpc_cfg.dt_hours          # step duration [h]
        month = datetime.now(LOCAL_TZ).month

        # Always plan from today midnight so the full day is visible
        today_midnight = slot.replace(hour=0, minute=0, second=0, microsecond=0)
        lp_slot = today_midnight

        # Steps from midnight to current slot (used by display methods)
        self._lp_now_idx = int(
            (slot - today_midnight).total_seconds() / (INTERVAL_MINUTES * 60)
        )

        # Horizon: from midnight to end of known price data
        if self.data.price_rows:
            last_ts = (self.data.price_rows[-1][0]
                       .astimezone(LOCAL_TZ).replace(second=0, microsecond=0))
            known_steps = max(1, int(
                (last_ts - today_midnight).total_seconds() / (INTERVAL_MINUTES * 60)
            ) + 1)
            H = min(H, known_steps)

        # Build forecasts from cached data (backend call)
        price_fc = self.data.build_price_forecast(lp_slot, H)
        base_load, solar_kwh = self.data.build_load_and_solar(
            lp_slot, H, self.base_load_kw, self.peak_load_kw, dt_hours=dt_h,
        )
        outside_temp = self.data.build_outside_temp(lp_slot, H, dt_hours=dt_h)

        # ── Submit async LP solve to subprocess (non-blocking) ──────────
        # _lp_worker.py is a standalone script that only imports smpc_calculator
        # (no Qt), so HIGHS DLLs never clash with Qt DLLs.
        try:
            _cfg = self.lp.mpc_cfg
            logger.info("LP submit: slot=%s H=%d lp_now_idx=%d", slot, H, self._lp_now_idx)
            self._lp_async_result = _AsyncSubprocessResult(
                self.lp.config_path,
                price_fc,
                base_load,
                solar_kwh,
                month,
                _cfg.T_init_c,
                outside_temp,
                _cfg.hw_T_init_c,
            )
            self._lp_submit_time = __import__("time").monotonic()
            logger.info("LP worker subprocess started")
        except Exception as exc:
            logger.warning("LP submit failed: %s", exc)

    # ------------------------------------------------------------------
    # Next-day LP (runs once when tomorrow's prices become available)
    # ------------------------------------------------------------------

    def _collect_next_day_lp_result(self):
        """Pick up the finished next-day async LP result without blocking."""
        if self._nd_lp_async_result is None:
            return
        submit_time = self._nd_lp_submit_time
        if submit_time is not None:
            elapsed = __import__("time").monotonic() - submit_time
            if elapsed > 30 and not self._nd_lp_async_result.ready():
                logger.warning(
                    "Next-day LP subprocess still running after %.0f s", elapsed
                )
                self._nd_lp_submit_time = None
        if not self._nd_lp_async_result.ready():
            return
        try:
            result = self._nd_lp_async_result.get()
            self.last_lp_next_day_outputs = result
            logger.info(
                "Next-day LP result collected: status=%s solver=%s",
                getattr(result, "solver_status", "?"),
                getattr(result, "solver_used", "?"),
            )
        except Exception as exc:
            logger.warning("Next-day LP result error: %s", exc)
        finally:
            self._nd_lp_async_result = None

    def _run_next_day_lp_if_needed(self):
        """Solve the LP for all of tomorrow when next-day prices are available.

        ENTSO-E publishes tomorrow's day-ahead prices around 13:00 CET.
        Once prices.csv contains data for tomorrow midnight onwards, this
        method submits a 96-step LP covering 00:00–23:45 tomorrow.

        The solve runs once per calendar day (tracked by self._nd_lp_last_date).
        A pending async result is never replaced — we wait for it first.
        """
        if self._nd_lp_async_result is not None:
            return                      # a solve is already in flight

        now = datetime.now(LOCAL_TZ)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        tomorrow_key = tomorrow.strftime("%Y-%m-%d")

        if self._nd_lp_last_date == tomorrow_key:
            return                      # already solved for tomorrow today

        # Check whether prices exist for tomorrow midnight
        has_tomorrow = any(
            ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0) >= tomorrow
            for ts, _, _ in self.data.price_rows
        )
        if not has_tomorrow:
            return                      # prices not yet published — check next tick

        H    = self.lp.mpc_cfg.horizon_steps
        dt_h = self.lp.mpc_cfg.dt_hours
        month = tomorrow.month

        price_fc    = self.data.build_price_forecast(tomorrow, H)
        base_load, solar_kwh = self.data.build_load_and_solar(
            tomorrow, H, self.base_load_kw, self.peak_load_kw, dt_hours=dt_h,
        )
        outside_temp = self.data.build_outside_temp(tomorrow, H, dt_hours=dt_h)

        try:
            _cfg = self.lp.mpc_cfg
            # Use the predicted end-of-today temps as initial conditions for tomorrow,
            # so the building/HW temperature lines connect smoothly at midnight.
            today_out = self.last_lp_outputs
            if (today_out is not None
                    and len(getattr(today_out, "plan_building_temp_c", [])) > 0):
                t_init = float(today_out.plan_building_temp_c[-1])
            else:
                t_init = _cfg.T_init_c
            if (today_out is not None
                    and len(getattr(today_out, "plan_hw_temp_c", [])) > 0):
                hw_t_init = float(today_out.plan_hw_temp_c[-1])
            else:
                hw_t_init = _cfg.hw_T_init_c
            logger.info("Next-day LP submit: date=%s H=%d T_init=%.1f hw_T_init=%.1f",
                        tomorrow_key, H, t_init, hw_t_init)
            self._nd_lp_async_result = _AsyncSubprocessResult(
                self.lp.config_path,
                price_fc,
                base_load,
                solar_kwh,
                month,
                t_init,
                outside_temp,
                hw_t_init,
            )
            self._nd_lp_submit_time = __import__("time").monotonic()
            self._nd_lp_last_date   = tomorrow_key
            logger.info("Next-day LP worker subprocess started")
        except Exception as exc:
            logger.warning("Next-day LP submit failed: %s", exc)

    # ------------------------------------------------------------------
    # Tag value simulation
    # ------------------------------------------------------------------

    def _lp_for_abs_step(self, abs_step: int):
        """Return (SMPCOutputs, plan_step) for an absolute step from today midnight.

        Today's LP covers plan indices 0 … H-1.
        Next-day LP covers plan indices 0 … H-1 of tomorrow, mapped to
        abs_step H … 2H-1 in the shared 48-h timeline.
        Returns (None, 0) if no result is available for that range.
        """
        H = self.lp.mpc_cfg.horizon_steps   # typically 96
        if abs_step < H:
            return self.last_lp_outputs, abs_step
        nd = self.last_lp_next_day_outputs
        if nd is not None:
            return nd, abs_step - H
        # next-day LP not solved yet — clamp to last step of today
        if self.last_lp_outputs is not None:
            return self.last_lp_outputs, min(abs_step, H - 1)
        return None, 0

    def _sim_value(self, defn):
        """Determine the display text and colour for a single tag.

        Reads from self.last_lp_outputs (set by _run_lp_if_needed) and from
        self.data (DataManager).  Each tag has a 'mode' that selects
        which backend field to read:

        Modes:
          predicted_solar  — reads PV power from DataManager.predictions
          smpc             — reads a numeric KPI from SMPCOutputs
          smpc_state       — reads a state string (ON/OFF/CHARGE/etc.)
          smpc_ice_status  — reads ice bank charge/discharge state
          smpc_setpoint    — reads a setpoint value
          asset_power      — reads per-asset kW from asset_schedules

        IMPORTANT: this method only READS the outputs; all calculations
        are done in the backend (smpc_calculator.py / data_manager.py).
        """
        sim = defn.get("simulation", {}) if isinstance(defn, dict) else {}
        mode = str(sim.get("mode", "")).strip().lower()

        if mode == "predicted_solar":
            return self._sim_predicted_solar(sim)
        if mode == "smpc":
            return self._sim_smpc(sim)
        if mode in ("smpc_state", "mpc_state"):
            return self._sim_smpc_state(sim)
        if mode == "smpc_ice_status":
            return self._sim_ice_status(sim)
        if mode == "smpc_setpoint":
            return self._sim_setpoint(sim)
        if mode == "asset_power":
            return self._sim_asset_power(sim)
        # ── MPC modes ────────────────────────────────────────────────
        if mode == "mpc_scalar":
            return self._sim_mpc_scalar(sim)
        if mode == "mpc_battery_status":
            return self._sim_mpc_battery_status(sim)
        if mode == "mpc_building_temp":
            return self._sim_mpc_building_temp(sim)
        if mode == "mpc_text":
            return self._sim_mpc_text(sim)
        return "--", "#64748b"

    def _sim_predicted_solar(self, sim):
        unit = sim.get("unit", "kW")
        dec = int(sim.get("decimals", 1))
        col = sim.get("color", "#0e7490")

        step = self._selected_step
        if step > 0:
            # Look up predicted solar at the future slot
            from datetime import timedelta
            future = current_slot() + timedelta(minutes=step * INTERVAL_MINUTES)
            key = future.strftime("%Y-%m-%d %H:%M:%S")
            pred = self.data.predictions.get(key)
            if pred is None:
                return f"-- {unit}", col
            return f"{pred[0]:.{dec}f} {unit}", col

        if self.data.current_power_kw is None:
            return f"-- {unit}", col
        return f"{self.data.current_power_kw:.{dec}f} {unit}", col

    def _sim_smpc(self, sim):
        field = sim.get("field", "")
        unit = sim.get("unit", "")
        dec = int(sim.get("decimals", 1))
        mult = float(sim.get("multiplier", 1))
        col = sim.get("color", "#0e7490")
        if self.last_lp_outputs is None:
            return f"-- {unit}".strip(), col
        kpis = SMPCCalculator.outputs_to_dashboard_dict(
            self.last_lp_outputs, step=self._selected_step,
        )
        val = kpis.get(field)
        if val is None:
            return f"-- {unit}".strip(), col
        txt = f"{val * mult:.{dec}f}"
        if unit:
            txt += f" {unit}"
        return txt, col

    def _sim_smpc_state(self, sim):
        field = sim.get("field", "")
        thresh = float(sim.get("threshold", 0.1))
        above = sim.get("above", "ON")
        below = sim.get("below", "OFF")
        cols = sim.get("colors", {})
        if self.last_lp_outputs is None:
            return "--", "#64748b"
        kpis = SMPCCalculator.outputs_to_dashboard_dict(
            self.last_lp_outputs, step=self._selected_step,
        )
        state = above if kpis.get(field, 0) > thresh else below
        return state, cols.get(state, "#0e7490")

    def _sim_ice_status(self, sim):
        cols = sim.get("colors", {
            "CHARGE": "#0e7490", "DISCHARGE": "#f59e0b", "IDLE": "#16a34a",
        })
        show_power = sim.get("show_power", False)
        if self.last_lp_outputs is None:
            return "--", "#64748b"
        out = self.last_lp_outputs
        step = self._selected_step
        if step > 0 and len(out.plan_ice_charge_kwh) > step:
            charge = float(out.plan_ice_charge_kwh[step])
            discharge = float(out.plan_ice_discharge_kwh[step]) if len(out.plan_ice_discharge_kwh) > step else 0.0
        else:
            charge = out.ice_bank_charge_kwh
            discharge = out.ice_bank_discharge_kwh
        if charge > 0.1:
            state = "CHARGE"
            power_kw = charge * 4.0
        elif discharge > 0.1:
            state = "DISCHARGE"
            power_kw = discharge * 4.0
        else:
            state = "IDLE"
            power_kw = 0.0
        txt = state
        if show_power and power_kw > 0.1:
            txt = f"{state} {power_kw:.1f} kW"
        return txt, cols.get(state, "#0e7490")

    def _sim_asset_power(self, sim):
        """Show per-asset scheduled power from the LP solution."""
        uid  = sim.get("uid", "")
        unit = sim.get("unit", "kW")
        dec  = int(sim.get("decimals", 1))
        col  = sim.get("color", "#0e7490")
        abs_step = max(0, self._lp_now_idx + self._selected_step)
        out, plan_step = self._lp_for_abs_step(abs_step)
        if out is None:
            return f"-- {unit}", "#64748b"
        sched = out.asset_schedules.get(uid)
        if sched is not None and len(sched) > plan_step:
            kw = float(sched[plan_step]) * 4.0   # kWh/15-min step → kW
        else:
            kw = out.asset_power_kw.get(uid, 0.0)
        col_out = col if kw >= 0.1 else "#64748b"
        return f"{kw:.{dec}f} {unit}", col_out

    def _sim_setpoint(self, sim):
        cols = sim.get("colors", {
            "Lowered": "#16a34a", "Normal": "#0e7490", "Higher": "#dc2626",
        })
        if self.last_lp_outputs is None:
            return "--", "#64748b"
        saving = self.last_lp_outputs.cost_saving_eur
        if saving > 0.001:
            state = "Lowered"
        elif saving < -0.001:
            state = "Higher"
        else:
            state = "Normal"
        return state, cols.get(state, "#0e7490")

    # ── New MPC simulation modes ─────────────────────────────────────

    def _sim_mpc_scalar(self, sim: dict):
        """Read a named SMPCOutputs field from the plan array at the selected slot."""
        field      = sim.get("field", "")
        plan_field = sim.get("plan_field", "")
        unit       = sim.get("unit", "")
        dec        = int(sim.get("decimals", 1))
        mult       = float(sim.get("multiplier", 1.0))
        col        = sim.get("color", "#0e7490")

        placeholder = (f"-- {unit}".strip()) or "--"

        abs_step = max(0, self._lp_now_idx + self._selected_step)
        out, plan_step = self._lp_for_abs_step(abs_step)
        if out is None:
            return placeholder, col

        # Always prefer plan array (covers past, now, and future)
        if plan_field:
            arr = getattr(out, plan_field, None)
            if arr is not None and len(arr) > plan_step:
                val = float(arr[plan_step]) * mult
                txt = f"{val:.{dec}f}"
                if unit:
                    txt += f" {unit}"
                return txt, col

        # Fallback: scalar attribute on SMPCOutputs
        val = getattr(out, field, None)
        if val is None:
            kpis = SMPCCalculator.outputs_to_dashboard_dict(out, step=0)
            val  = kpis.get(field)
        if val is None:
            return placeholder, col

        val = float(val) * mult
        txt = f"{val:.{dec}f}"
        if unit:
            txt += f" {unit}"
        return txt, col

    def _sim_mpc_battery_status(self, sim: dict):
        """Show CHARGING / DISCHARGING / IDLE with power and SOC."""
        abs_step = max(0, self._lp_now_idx + self._selected_step)
        out, plan_step = self._lp_for_abs_step(abs_step)
        if out is None:
            return "--", "#64748b"
        sched_ch  = out.asset_schedules.get("battery_charge")
        sched_dis = out.asset_schedules.get("battery_discharge")
        charge    = float(sched_ch[plan_step])  * 4.0 if (sched_ch  is not None and len(sched_ch)  > plan_step) else 0.0
        discharge = float(sched_dis[plan_step]) * 4.0 if (sched_dis is not None and len(sched_dis) > plan_step) else 0.0
        soc_arr   = out.plan_SOC
        soc = float(soc_arr[abs_step]) if (soc_arr is not None and len(soc_arr) > abs_step) else out.battery_soc_kwh
        if charge > 0.1:
            return f"CHARGING  {charge:.1f} kW", "#0891b2"
        if discharge > 0.1:
            return f"DISCHARGING  {discharge:.1f} kW", "#f59e0b"
        return f"IDLE  ({soc:.0f} kWh)", "#16a34a"

    def _sim_mpc_building_temp(self, sim: dict):
        """Building temperature coloured by distance from comfort setpoint."""
        if self.last_lp_outputs is None:
            return "-- \u00b0C", "#64748b"
        abs_step   = max(0, self._lp_now_idx + self._selected_step)
        out, plan_step = self._lp_for_abs_step(abs_step)
        if out is None:
            return "-- \u00b0C", "#64748b"
        plan_field = sim.get("plan_field", "plan_building_temp_c")
        arr = getattr(out, plan_field, None)
        if arr is not None and len(arr) > plan_step:
            temp = float(arr[plan_step])
        else:
            temp = float(out.building_temp_c)
        sp   = float(out.building_setpoint_c)
        diff = abs(temp - sp)
        col  = "#16a34a" if diff < 1.0 else "#f59e0b" if diff < 2.0 else "#dc2626"
        return f"{temp:.1f} \u00b0C", col

    def _sim_mpc_text(self, sim: dict):
        """Display a plain-text string field from SMPCOutputs."""
        field = sim.get("field", "")
        if self.last_lp_outputs is None:
            return "--", "#64748b"
        val = getattr(self.last_lp_outputs, field, None)
        return (str(val) if val is not None else "--"), "#0f766e"

    # ------------------------------------------------------------------
    # Dialogs / popups
    # ------------------------------------------------------------------

    def _open_historical(self):
        HistoricalAnalysisDialog(self).exec()

    def _open_future(self):
        FutureSimulationDialog(self).exec()

    def _open_building_settings(self):
        """Open MPC Settings dialog for general building & solver configuration."""
        from mpc_config_dialog import MpcConfigDialog
        dlg = MpcConfigDialog(self)
        dlg.exec()
        # Reload all building state from updated config
        cfg = self.lp.mpc_cfg
        self._building_setpoint_c = cfg.Tset_c
        self._building_temp_c     = cfg.T_init_c
        self._hw_temp_c           = cfg.hw_T_init_c
        # Reload electrical load profile (lives in smpc.building in JSON)
        raw = load_dashboard_config()
        bld_smpc = raw.get("smpc", {}).get("building", {})
        self.base_load_kw = float(bld_smpc.get("base_load_kw", self.base_load_kw))
        self.peak_load_kw = float(bld_smpc.get("peak_load_kw", self.peak_load_kw))
        # Resize slider to new horizon
        if self.slot_slider:
            self.slot_slider.setMaximum(cfg.horizon_steps - 1)
            self.slot_slider.setValue(0)
        # Force LP re-solve with new settings at the next tick
        self.lp_last_slot = None

    def _open_mpc_asset_manager(self):
        """Open the asset selector dialog; rebuild columns if user saved."""
        from mpc_config_dialog import MPCAssetSelectorDialog
        dlg = MPCAssetSelectorDialog(self)
        if dlg.exec():
            self._rebuild_columns()

    def _rebuild_columns(self):
        """Clear and re-populate both side columns after asset changes."""
        for content in (self.input_content, self.output_content):
            layout = content.layout()
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

        self.value_labels.clear()
        self.tag_definitions = {"input": {}, "output": {}}

        self._populate_columns()


# ===================================================================
# Entry point
# ===================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s \u2014 %(message)s",
    )
    app = QApplication(sys.argv)
    # Force Fusion light style so OS dark mode cannot bleed through the
    # hardcoded light-colour stylesheet.
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor("#eef3f9"))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor("#1f2937"))
    pal.setColor(QPalette.ColorRole.Base,            QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#f8fbff"))
    pal.setColor(QPalette.ColorRole.Text,            QColor("#1f2937"))
    pal.setColor(QPalette.ColorRole.Button,          QColor("#d6dfeb"))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor("#1f2937"))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor("#2563eb"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor("#1f2937"))
    app.setPalette(pal)
    window = ScadaWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _mp.freeze_support()   # required for Windows frozen executables
    main()
