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
import sys
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from asset_dialogs import AssetManagerDialog
from dashboard_config import load_dashboard_config
from data_manager import DataManager, current_slot
from graph_renderer import draw_price_graph, draw_solar_graph
from future_dialog import FutureSimulationDialog
from historical_dialog import HistoricalAnalysisDialog
from thermal_dialog import BuildingThermalDialog
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

        # Slot preview slider (0 = Now, 1..95 = future 15-min steps)
        self._selected_step: int = 0

        # Widget references (populated during UI build)
        self.clock_label = None
        self.price_current_label = None
        self.price_avg_label = None
        self.actual_yield_label = None
        self.center_price_graph_label = None
        self.center_solar_graph_label = None
        self.center_uv_value_label = None
        self.center_solar_value_label = None
        self.slot_slider = None
        self.slot_label = None

    def _init_data(self):
        """Create the DataManager (handles all fetching and caching).

        BACKEND CALL: DataManager lives in data_manager.py (backend).
        The dashboard never fetches data directly — it only reads from this object.
        """
        self.data = DataManager()

    def _init_lp(self):
        """Create the LP calculator and load building parameters.

        BACKEND CALL: SMPCCalculator lives in smpc_calculator.py (backend).
        Building parameters (base_load_kw, peak_load_kw) come from
        dashboard_config.json → 'smpc' → 'building'.
        """
        raw = load_dashboard_config()
        building = raw.get("smpc", {}).get("building", {})

        self.lp = SMPCCalculator()
        self.lp_last_slot = None
        self.last_lp_outputs = None

        self.base_load_kw = building.get("base_load_kw", 80)
        self.peak_load_kw = building.get("peak_load_kw", 200)

        # Building thermal state
        thermal = building.get("thermal", {})
        self._building_temp_c = thermal.get("initial_temp_c", 21.0)
        self._building_setpoint_c = thermal.get("setpoint_c", 21.0)

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

        # "Add" buttons
        for container, label, key in [
            (self.input_col, "+ Add input", "input"),
            (self.output_col, "+ Add output", "output"),
        ]:
            btn = QPushButton(label)
            btn.setMinimumHeight(52)
            btn.setMinimumWidth(50)
            btn.setStyleSheet(COLUMN_BUTTON_STYLE)
            btn.clicked.connect(lambda _=False, k=key: self._open_popup(k))
            container.layout().addWidget(btn)

    @staticmethod
    def _simulation_for_asset(asset, side):
        """Return simulation config dict for a known asset.

        *side* is ``"input"`` (power / value) or ``"output"`` (status).
        Generators get power on input, ON/OFF on output.
        Shiftable loads only appear on output with power + status.
        Heating assets (HEAT_PUMP, GAS_HEATER) show total heating kW.
        """
        from energy_assets import (
            SHIFTABLE_LOAD as _SL, GENERATOR as _GEN,
            HEAT_PUMP as _HP, GAS_HEATER as _GH,
        )

        uid = asset.uid

        # ── Generators ──────────────────────────────────────────
        if asset.asset_type == _GEN:
            if uid == "solar":
                if side == "input":
                    return {"mode": "predicted_solar", "unit": "kW", "decimals": 1, "color": "#f59e0b"}
                return {
                    "mode": "smpc_state",
                    "field": "net_power_kwh",
                    "threshold": 0.0,
                    "above": "PRODUCING",
                    "below": "IDLE",
                    "colors": {"PRODUCING": "#16a34a", "IDLE": "#64748b"},
                }
            # Generic generator (CHP / WKK / any new one)
            if side == "input":
                return {
                    "mode": "smpc",
                    "field": "wkk_elec_kwh",
                    "unit": "kW",
                    "decimals": 1,
                    "multiplier": 4,
                    "color": "#0e7490",
                }
            return {
                "mode": "smpc_state",
                "field": "wkk_elec_kwh",
                "threshold": 0.1,
                "above": "ON",
                "below": "OFF",
                "colors": {"ON": "#16a34a", "OFF": "#dc2626"},
            }

        # ── Shiftable loads (output side only) ──────────────────
        if asset.asset_type == _SL:
            return {
                "mode": "asset_power",
                "uid": uid,
                "unit": "kW",
                "decimals": 1,
            }

        # ── Heat pumps ───────────────────────────────────────────
        if asset.asset_type == _HP:
            if side == "input":
                return {
                    "mode": "smpc",
                    "field": "heating_power_kw",
                    "unit": "kW",
                    "decimals": 1,
                    "multiplier": 1,
                    "color": "#f59e0b",
                }
            return {
                "mode": "smpc_state",
                "field": "heating_power_kw",
                "threshold": 0.1,
                "above": "HEATING",
                "below": "IDLE",
                "colors": {"HEATING": "#dc2626", "IDLE": "#16a34a"},
            }

        # ── Gas heaters ──────────────────────────────────────────
        if asset.asset_type == _GH:
            if side == "input":
                return {
                    "mode": "smpc",
                    "field": "heating_power_kw",
                    "unit": "kW",
                    "decimals": 1,
                    "multiplier": 1,
                    "color": "#ef4444",
                }
            return {
                "mode": "smpc_state",
                "field": "heating_power_kw",
                "threshold": 0.1,
                "above": "HEATING",
                "below": "IDLE",
                "colors": {"HEATING": "#dc2626", "IDLE": "#16a34a"},
            }

        return {}

    def _populate_columns(self):
        """Build input/output tag lists from energy assets and populate."""
        from energy_assets import (
            load_assets as _load_ea,
            SHIFTABLE_LOAD as _SL, HEAT_PUMP as _HP, GAS_HEATER as _GH,
        )
        input_tags: list[dict] = []
        output_tags: list[dict] = []
        for asset in _load_ea():
            base = {
                "id": asset.uid,
                "name": asset.name,
                "icon": asset.icon,
                "section": (
                    "CONTROLLABLE" if asset.asset_type == _SL
                    else "HEATING" if asset.asset_type in (_HP, _GH)
                    else "GENERATION"
                ),
            }
            if asset.asset_type == _SL:
                # Shiftable loads → output column only
                out_tag = dict(base, simulation=self._simulation_for_asset(asset, "output"))
                output_tags.append(out_tag)
            else:
                # Generators + heating assets → both columns (power on input, status on output)
                in_tag = dict(base, simulation=self._simulation_for_asset(asset, "input"))
                out_tag = dict(base, simulation=self._simulation_for_asset(asset, "output"))
                input_tags.append(in_tag)
                output_tags.append(out_tag)

        norm_in = self._normalise_tags(input_tags)
        norm_out = self._normalise_tags(output_tags)
        self._register_definitions("input", norm_in)
        self._register_definitions("output", norm_out)

        self._add_price_widgets(self.input_content)
        self._add_thermal_widgets(self.input_content)
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
        self.slot_slider.setMinimum(-96)
        self.slot_slider.setMaximum(191)
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

        # Building thermal settings button
        thermal_btn = QPushButton("\U0001f321\ufe0f  Building Thermal Settings")
        thermal_btn.setMinimumHeight(44)
        thermal_btn.setStyleSheet(
            "QPushButton { background: #7c3aed; color: white; border: none; "
            "border-radius: 8px; font-weight: 700; font-size: 10pt; } "
            "QPushButton:hover { background: #6d28d9; }"
        )
        thermal_btn.clicked.connect(self._open_thermal_settings)
        lay.addWidget(thermal_btn)

        lay.addStretch()

        scroll.setWidget(content)
        outer_lay.addWidget(scroll)
        return panel

    def _on_slot_slider_changed(self, value: int):
        """Handle slider position change."""
        self._selected_step = value
        if value == 0:
            self.slot_label.setText("Now")
            self.slot_label.setStyleSheet(value_css("#0e7490"))
        else:
            from datetime import timedelta
            target = current_slot() + timedelta(minutes=value * INTERVAL_MINUTES)
            self.slot_label.setText(target.strftime("%H:%M"))
            if value < 0:
                self.slot_label.setStyleSheet(value_css("#94a3b8"))
            else:
                self.slot_label.setStyleSheet(value_css("#7c3aed"))
        self._update_tag_labels()
        self._update_price_for_slot()
        self._update_center()
        self._update_thermal_labels()

    def wheelEvent(self, event):
        """Scroll wheel adjusts the time-slot slider."""
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        step = 1 if delta > 0 else -1
        self.slot_slider.setValue(self.slot_slider.value() + step)
        event.accept()

    # ------------------------------------------------------------------
    # Reusable widget builders
    # ------------------------------------------------------------------

    @staticmethod
    def _graph_card(parent_layout, title_text: str, placeholder: str) -> QLabel:
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
        lbl.setFixedHeight(200)
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
        """Add clock, current-price, 48 h average, and yield rows."""
        self.clock_label = self._info_row(
            container, "\U0001f552  Local time", "--",
        )
        self.price_current_label = self._info_row(
            container, "\u26a1  Current 15m price", "-- EUR/MWh",
        )
        self.price_avg_label = self._info_row(
            container, "\U0001f4c8  48h average", "-- EUR/MWh",
        )
        self.actual_yield_label = self._info_row(
            container, "\U0001f506  Actual Yield", "-- kWh",
        )
        self._update_clock()

    def _add_thermal_widgets(self, container):
        """Add building temperature, setpoint, and BEO-veld info rows."""
        self.building_temp_label = self._info_row(
            container, "\U0001f321\ufe0f  Building temp", "-- °C",
        )
        self.building_setpoint_label = self._info_row(
            container, "\U0001f3af  Setpoint", "-- °C",
        )
        self.heating_power_label = self._info_row(
            container, "\U0001f525  Heating", "-- kW",
        )
        self.beo_temp_label = self._info_row(
            container, "\U0001f30d  BEO-veld", "-- °C",
        )

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
        3. _run_lp_if_needed() — run backend LP solver if we're in a new slot
        4. _update_all_widgets() — propagate results to the UI labels (frontend)
        """
        self._update_clock()
        self.data.tick()               # may refresh prices/predictions/weather
        self._run_lp_if_needed()       # runs once per 15-min slot
        self._update_all_widgets()

    # ------------------------------------------------------------------
    # Widget updates (pure UI — reads from DataManager)
    # ------------------------------------------------------------------

    def _update_all_widgets(self):
        self._update_price_for_slot()
        self._update_predict_labels()
        self._update_center()
        self._update_tag_labels()
        self._update_thermal_labels()

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
        if self.actual_yield_label:
            self.actual_yield_label.setText("-- kWh")
            self.actual_yield_label.setStyleSheet(value_css("#64748b"))

    def _update_thermal_labels(self):
        """Update building temperature display from LP outputs."""
        out = self.last_lp_outputs
        if out is None:
            return

        step = max(0, self._selected_step)

        def _plan(arr, fallback):
            """Return plan value at `step`, falling back to first-step scalar."""
            if arr is not None and len(arr) > step:
                return float(arr[step])
            return fallback

        temp = _plan(out.plan_building_temp_c, out.building_temp_c)
        sp   = out.building_setpoint_c
        heat = _plan(out.plan_heating_kw, out.heating_power_kw)
        beo  = _plan(out.plan_beo_temp_c, out.beo_temp_c)

        # Temperature colour: green=OK, orange=near limit, red=out of band
        diff = abs(temp - sp)
        if diff < 1.0:
            colour = "#16a34a"
        elif diff < 2.0:
            colour = "#f59e0b"
        else:
            colour = "#dc2626"

        self.building_temp_label.setText(f"{temp:.1f} °C")
        self.building_temp_label.setStyleSheet(value_css(colour))

        self.building_setpoint_label.setText(f"{sp:.1f} °C")
        self.building_setpoint_label.setStyleSheet(value_css("#0f766e"))

        heat_colour = "#dc2626" if heat > 0 else "#64748b"
        self.heating_power_label.setText(
            f"{heat:.1f} kW" if heat > 0 else "Off"
        )
        self.heating_power_label.setStyleSheet(value_css(heat_colour))

        self.beo_temp_label.setText(f"{beo:.1f} °C")
        self.beo_temp_label.setStyleSheet(value_css("#0f766e"))

    # ------------------------------------------------------------------
    # Centre panel updates
    # ------------------------------------------------------------------

    def _update_center(self):
        slot = current_slot()
        slot_key = slot.strftime("%Y-%m-%d %H:%M:%S")
        dm = self.data

        # Price graph
        if self.center_price_graph_label is not None:
            prices = [p for _, p, _ in dm.price_rows]
            idx = None
            if dm.price_rows:
                start = dm.price_rows[0][0].astimezone(LOCAL_TZ).strftime("%d/%m %H:%M")
                end = dm.price_rows[-1][0].astimezone(LOCAL_TZ).strftime("%d/%m %H:%M")
                for i, (ts, _, _) in enumerate(dm.price_rows):
                    if ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0) == slot:
                        idx = i
                        break
            else:
                start, end = "", ""
            if idx is not None and self._selected_step != 0:
                _si = idx + self._selected_step
                sel_idx = _si if 0 <= _si < len(prices) else None
            else:
                sel_idx = None
            gw = max(self.center_price_graph_label.width(), 300)
            gh = self.center_price_graph_label.height() or 200
            self.center_price_graph_label.setPixmap(
                draw_price_graph(prices, start, end, idx, sel_idx, gw, gh),
            )
            # Clamp slider to remaining future data
            if idx is not None:
                max_future = max(len(prices) - 1 - idx, 0)
                if self.slot_slider.maximum() != max_future:
                    self.slot_slider.setMaximum(max(max_future, 1))

        # Solar graph
        if self.center_solar_graph_label is not None:
            items = sorted(dm.predictions.items())
            values = [pw for _, (pw, _) in items]
            idx = None
            if items:
                start = datetime.strptime(
                    items[0][0], "%Y-%m-%d %H:%M:%S",
                ).strftime("%d/%m %H:%M")
                end = datetime.strptime(
                    items[-1][0], "%Y-%m-%d %H:%M:%S",
                ).strftime("%d/%m %H:%M")
                for i, (k, _) in enumerate(items):
                    if k == slot_key:
                        idx = i
                        break
            else:
                start, end = "", ""
            if idx is not None and self._selected_step != 0:
                _si = idx + self._selected_step
                sel_idx = _si if 0 <= _si < len(values) else None
            else:
                sel_idx = None
            gw = max(self.center_solar_graph_label.width(), 300)
            gh = self.center_solar_graph_label.height() or 200
            self.center_solar_graph_label.setPixmap(
                draw_solar_graph(values, start, end, idx, sel_idx, gw, gh),
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

        H = self.lp.cfg.horizon_steps          # 96 steps = 24 hours
        month = datetime.now(LOCAL_TZ).month

        # Build forecasts from cached data (backend call)
        price_fc = self.data.build_price_forecast(slot, H)
        base_load, solar_kwh = self.data.build_load_and_solar(
            slot, H, self.base_load_kw, self.peak_load_kw,
        )

        try:
            # ── BACKEND CALL: run the greedy LP optimiser ──────────
            self.last_lp_outputs = self.lp.solve_lp(
                price_forecast_eur_kwh=price_fc,
                base_load_kwh=base_load,
                solar_pred_kwh=solar_kwh,
                month=month,
                building_temp_c=self._building_temp_c,
            )
            # Carry building temperature forward for the next LP call.
            # building_temp_c = temp at END of step 0 = start of next slot.
            if self.last_lp_outputs is not None:
                self._building_temp_c = self.last_lp_outputs.building_temp_c
        except Exception as exc:
            logger.warning("LP solve: %s", exc)

    # ------------------------------------------------------------------
    # Tag value simulation
    # ------------------------------------------------------------------

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
        if mode == "smpc_state":
            return self._sim_smpc_state(sim)
        if mode == "smpc_ice_status":
            return self._sim_ice_status(sim)
        if mode == "smpc_setpoint":
            return self._sim_setpoint(sim)
        if mode == "asset_power":
            return self._sim_asset_power(sim)
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
        uid = sim.get("uid", "")
        unit = sim.get("unit", "kW")
        dec = int(sim.get("decimals", 1))
        if self.last_lp_outputs is None:
            return f"-- {unit}", "#64748b"
        out = self.last_lp_outputs
        step = self._selected_step
        # Use per-asset full schedule if looking at a future step
        if step > 0:
            sched = out.asset_schedules.get(uid)
            if sched is not None and len(sched) > step:
                kw = float(sched[step]) * 4.0  # kWh/15min → kW
            else:
                kw = 0.0
        else:
            kw = out.asset_power_kw.get(uid, 0.0)
        if kw < 0.1:
            return "IDLE", "#16a34a"
        return f"{kw:.{dec}f} {unit}", "#0e7490"

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

    # ------------------------------------------------------------------
    # Dialogs / popups
    # ------------------------------------------------------------------

    def _open_historical(self):
        HistoricalAnalysisDialog(self).exec()

    def _open_future(self):
        FutureSimulationDialog(self).exec()

    def _open_thermal_settings(self):
        dlg = BuildingThermalDialog(self)
        if dlg.exec():
            # Reload thermal config into running state
            cfg = load_dashboard_config()
            th = cfg.get("smpc", {}).get("building", {}).get("thermal", {})
            self._building_setpoint_c = th.get("setpoint_c", 21.0)
            self._building_temp_c = th.get("initial_temp_c", 21.0)

    def _open_popup(self, key):
        dlg = AssetManagerDialog(category=key, parent=self)
        dlg.exec()
        if dlg.changed:
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
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    app = QApplication(sys.argv)
    window = ScadaWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
