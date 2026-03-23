"""Main Dashboard window — Building Management System.

Run with::

    python dashboard.py

The window has three columns:
  * **Left** — system inputs (price, solar, CHP, …)
  * **Centre** — graphs, KPIs, historical-analysis button
  * **Right** — system outputs (CHP state, heat-pump, setpoint, ice banks, …)

Values update every second via SMPC and live data refreshes.
"""

import csv
import logging
import sys
from datetime import datetime, timedelta

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

import getPrice
import getWeather
import predict
from dashboard_config import load_dashboard_config
from graph_renderer import draw_price_graph, draw_solar_graph
from historical_dialog import HistoricalAnalysisDialog
from settings import (
    DASHBOARD_DIR, DAILY_FETCH_HOUR, DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
    INTERVAL_MINUTES, LOCAL_TZ, PREDICT_CSV, SOLAR_CAPACITY_KWP,
    WEATHER_CSV,
)
from smpc_calculator import SMPCCalculator, SMPCInputs
from styles import (
    COLUMN_BUTTON_STYLE, HISTORICAL_BUTTON_STYLE, MAIN_WINDOW_STYLE,
    value_css,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Icon lookup tables
# ---------------------------------------------------------------------------
_ICON_KEY = {
    "plug": "\U0001f50c", "sun": "\u2600\ufe0f", "fire": "\U0001f525",
    "heat": "\u2668\ufe0f", "target": "\U0001f3af", "snowflake": "\U0001f9ca",
    "car": "\U0001f697", "globe": "\U0001f30d",
}
_ICON_LEGACY = {
    "GRID": "\U0001f50c", "Solar Panels": "\u2600\ufe0f",
    "CHP": "\U0001f525", "Heat Pump": "\u2668\ufe0f",
    "Setpoint": "\U0001f3af", "Ice Banks": "\U0001f9ca",
    "EVs": "\U0001f697", "BEO": "\U0001f30d",
}

SIDE_COL_MIN_W = 340


# ===================================================================
# Main window
# ===================================================================

class ScadaWindow(QMainWindow):
    """Three-column SCADA-style dashboard."""

    def __init__(self, input_tags: list, output_tags: list):
        super().__init__()
        self.setWindowTitle("Thesis \u2014 Building Management System")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        self._init_state()
        self._init_smpc()
        self._build_ui(input_tags, output_tags)
        self._start_timer()
        self._load_initial_data()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Declare every mutable attribute with its default value."""
        self.small_windows: dict = {}
        self.value_labels: dict = {}
        self.tag_definitions: dict = {"input": {}, "output": {}}

        # Widget references (populated during UI build)
        self.clock_label = None
        self.price_current_label = None
        self.price_avg_label = None
        self.actual_yield_label = None
        self.center_price_graph_label = None
        self.center_solar_graph_label = None
        self.center_uv_value_label = None
        self.center_solar_value_label = None

        # Data caches
        self.cached_price_rows: list = []
        self.cached_avg_48h: float | None = None
        self.predicted_by_timestamp: dict = {}
        self.uv_by_timestamp: dict = {}

        # Current-slot derived values
        self.current_predicted_power_kw: float | None = None
        self.current_predicted_yield_kwh: float | None = None
        self.current_uv_index: float | None = None

        # Refresh scheduling
        self.price_next_refresh = None
        self.predict_next_refresh = None
        self.weather_next_refresh = None
        self.next_data_pipeline_run = None

    def _init_smpc(self):
        """Load SMPC configuration and create the calculator."""
        raw = load_dashboard_config()
        building = raw.get("smpc", {}).get("building", {})

        self.smpc_calculator = SMPCCalculator()
        self.smpc_cfg = self.smpc_calculator.cfg

        self.ice_bank_kwh = self.smpc_cfg.ice_bank_initial_kwh
        self.heat_buffer_kwh = self.smpc_cfg.heat_buffer_initial_kwh
        self.last_smpc_outputs = None
        self.smpc_last_slot = None

        self.smpc_base_load_kw = building.get("base_load_kw", 80)
        self.smpc_peak_load_kw = building.get("peak_load_kw", 200)
        self.smpc_wkk_max_gas_m3 = building.get("wkk_max_gas_m3", 9.0)
        self.smpc_heat_demand_base_kwh = building.get("heat_demand_base_kwh", 20)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, input_tags, output_tags):
        """Build the three-column layout and populate from config tags."""
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
        cols.addWidget(self.input_col, stretch=1)

        # Centre — graphs / KPIs
        cols.addWidget(self._build_center_panel(), stretch=1)

        # Right — outputs
        self.output_col, self.output_content = self._make_column(
            "System outputs", "#4d1a1a",
        )
        self.output_col.setMinimumWidth(SIDE_COL_MIN_W)
        cols.addWidget(self.output_col, stretch=1)

        # Populate from config
        norm_in = self._normalise_tags(input_tags)
        norm_out = self._normalise_tags(output_tags)
        self._register_definitions("input", norm_in)
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
        lay = QVBoxLayout(panel)
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

        # Historical analysis button
        hist_btn = QPushButton("Historical SMPC Analysis")
        hist_btn.setMinimumHeight(44)
        hist_btn.setStyleSheet(HISTORICAL_BUTTON_STYLE)
        hist_btn.clicked.connect(self._open_historical)
        lay.addWidget(hist_btn)

        lay.addStretch()
        return panel

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
        lbl.setMinimumHeight(170)
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
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(1000)

    def _load_initial_data(self):
        self._run_data_pipeline(force=True)
        self._refresh_prices(force=True)
        self._refresh_predictions(force=True)
        self._refresh_weather(force=True)
        self._update_center()

    # ------------------------------------------------------------------
    # Data refresh — prices
    # ------------------------------------------------------------------

    def _refresh_prices(self, force=False):
        if self.price_current_label is None:
            return
        now = datetime.now(LOCAL_TZ)
        if not force and self.price_next_refresh and now < self.price_next_refresh:
            return
        try:
            rows, avg = getPrice.get_flagged_next_day_prices()
            self.cached_price_rows = rows
            self.cached_avg_48h = avg
            self.price_next_refresh = self._next_quarter(now)
        except Exception as exc:
            if force:
                self.price_current_label.setText("API error")
                self.price_avg_label.setText("API error")
            logger.warning("Price refresh: %s", exc)
            return
        self._update_price_labels()

    def _update_price_labels(self):
        slot = self._current_slot()
        if not self.cached_price_rows or self.cached_avg_48h is None:
            return
        price = None
        for ts, p, _ in self.cached_price_rows:
            if ts.astimezone(LOCAL_TZ) == slot:
                price = p
                break
        if price is None:
            return

        self.price_current_label.setText(f"{price:.2f} EUR/MWh")
        self.price_avg_label.setText(f"{self.cached_avg_48h:.2f} EUR/MWh")
        colour = "#16a34a" if price < self.cached_avg_48h else "#dc2626"
        self.price_current_label.setStyleSheet(value_css(colour))
        self.price_avg_label.setStyleSheet(value_css("#0f766e"))

    # ------------------------------------------------------------------
    # Data refresh — predictions
    # ------------------------------------------------------------------

    def _refresh_predictions(self, force=False):
        now = datetime.now(LOCAL_TZ)
        if not force and self.predict_next_refresh and now < self.predict_next_refresh:
            return
        if not PREDICT_CSV.exists():
            return
        try:
            loaded: dict = {}
            with PREDICT_CSV.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    ts = row.get("Timestamp", "").strip()
                    if not ts:
                        continue
                    pw = float(row["Predicted Power (kW)"].replace(",", "."))
                    yl = float(row["Predicted Yield (kWh)"].replace(",", "."))
                    loaded[ts] = (pw, yl)
            self.predicted_by_timestamp = loaded
        except Exception as exc:
            logger.warning("Predict refresh: %s", exc)
            return
        self.predict_next_refresh = self._next_quarter(now)
        self._update_predict_labels()

    def _update_predict_labels(self):
        key = self._current_slot().strftime("%Y-%m-%d %H:%M:%S")
        cur = self.predicted_by_timestamp.get(key)
        if cur is None:
            self.current_predicted_power_kw = None
            self.current_predicted_yield_kwh = None
            if self.actual_yield_label:
                self.actual_yield_label.setText("-- kWh")
            return
        self.current_predicted_power_kw, self.current_predicted_yield_kwh = cur
        if self.actual_yield_label:
            self.actual_yield_label.setText("-- kWh")
            self.actual_yield_label.setStyleSheet(value_css("#64748b"))

    # ------------------------------------------------------------------
    # Data refresh — weather
    # ------------------------------------------------------------------

    def _refresh_weather(self, force=False):
        now = datetime.now(LOCAL_TZ)
        if not force and self.weather_next_refresh and now < self.weather_next_refresh:
            return
        if not WEATHER_CSV.exists():
            return
        try:
            loaded: dict = {}
            with WEATHER_CSV.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    ts = row.get("Timestamp", "").strip()
                    if ts:
                        loaded[ts] = float(
                            row.get("UV Index", "0").replace(",", ".")
                        )
            self.uv_by_timestamp = loaded
        except Exception as exc:
            logger.warning("Weather refresh: %s", exc)
            return
        self.weather_next_refresh = self._next_quarter(now)

    # ------------------------------------------------------------------
    # Centre panel updates
    # ------------------------------------------------------------------

    def _update_center(self):
        slot = self._current_slot()
        slot_key = slot.strftime("%Y-%m-%d %H:%M:%S")

        # Price graph
        if self.center_price_graph_label is not None:
            prices = [p for _, p, _ in self.cached_price_rows]
            idx = None
            if self.cached_price_rows:
                start = self.cached_price_rows[0][0].astimezone(LOCAL_TZ).strftime("%d/%m %H:%M")
                end = self.cached_price_rows[-1][0].astimezone(LOCAL_TZ).strftime("%d/%m %H:%M")
                for i, (ts, _, _) in enumerate(self.cached_price_rows):
                    if ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0) == slot:
                        idx = i
                        break
            else:
                start, end = "", ""
            self.center_price_graph_label.setPixmap(
                draw_price_graph(prices, start, end, idx),
            )

        # Solar graph
        if self.center_solar_graph_label is not None:
            items = sorted(self.predicted_by_timestamp.items())
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
            self.center_solar_graph_label.setPixmap(
                draw_solar_graph(values, start, end, idx),
            )

        # UV & solar value labels
        self.current_uv_index = self.uv_by_timestamp.get(slot_key)
        if self.center_uv_value_label:
            self.center_uv_value_label.setText(
                "--" if self.current_uv_index is None
                else f"{self.current_uv_index:.2f}"
            )
        if self.center_solar_value_label:
            if self.current_predicted_power_kw is None:
                self.center_solar_value_label.setText("-- kW")
            else:
                self.center_solar_value_label.setText(
                    f"{self.current_predicted_power_kw:.1f} kW"
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

    # ------------------------------------------------------------------
    # Data pipeline (daily scheduled refresh)
    # ------------------------------------------------------------------

    def _next_pipeline_time(self, now=None):
        if now is None:
            now = datetime.now(LOCAL_TZ)
        target = now.replace(
            hour=DAILY_FETCH_HOUR, minute=0, second=0, microsecond=0,
        )
        return target if now < target else target + timedelta(days=1)

    def _run_data_pipeline(self, force=False):
        now = datetime.now(LOCAL_TZ)
        if not force and self.next_data_pipeline_run and now < self.next_data_pipeline_run:
            return

        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            client = getWeather.build_client()
            data = getWeather.fetch_weather(
                client, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, today, tomorrow,
            )
            getWeather.export_csv(data, WEATHER_CSV)
        except Exception as exc:
            logger.warning("Weather pipeline: %s", exc)

        try:
            rows = predict.read_weather_csv(WEATHER_CSV)
            if rows:
                predict.export_predictions(
                    rows, PREDICT_CSV, SOLAR_CAPACITY_KWP,
                )
        except Exception as exc:
            logger.warning("Predict pipeline: %s", exc)

        try:
            getPrice.fetch_and_save_prices(
                output_filename="prices.csv",
                output_dir=DASHBOARD_DIR,
                reference_time=now,
            )
        except Exception as exc:
            logger.warning("Price pipeline: %s", exc)

        self.next_data_pipeline_run = self._next_pipeline_time(now)

    def _maybe_run_pipeline(self):
        now = datetime.now(LOCAL_TZ)
        if self.next_data_pipeline_run is None:
            self.next_data_pipeline_run = self._next_pipeline_time(now)
        if now >= self.next_data_pipeline_run:
            self._run_data_pipeline(force=True)
            self._refresh_predictions(force=True)
            self._refresh_weather(force=True)
            self._refresh_prices(force=True)
            self._update_center()

    # ------------------------------------------------------------------
    # SMPC integration
    # ------------------------------------------------------------------

    def _build_price_forecast(self, slot):
        H = self.smpc_cfg.horizon_steps
        if not self.cached_price_rows:
            return np.full(H, 0.10)
        timeline = sorted(
            [
                (ts.astimezone(LOCAL_TZ).replace(second=0, microsecond=0),
                 price / 1000.0)
                for ts, price, _ in self.cached_price_rows
            ],
            key=lambda r: r[0],
        )
        prices = np.zeros(H)
        for t in range(H):
            target = slot + timedelta(minutes=INTERVAL_MINUTES * t)
            best = timeline[0][1]
            for ts, p in timeline:
                if ts <= target:
                    best = p
                else:
                    break
            prices[t] = best
        return prices

    def _build_consumption_profile(self, slot):
        H = self.smpc_cfg.horizon_steps
        start_h = slot.hour + slot.minute / 60.0
        load = np.zeros(H)
        solar = np.zeros(H)
        for t in range(H):
            h = (start_h + t * 0.25) % 24
            if 6 <= h <= 18:
                kw = self.smpc_base_load_kw + (
                    self.smpc_peak_load_kw - self.smpc_base_load_kw
                ) * max(0.0, np.sin(np.pi * (h - 6) / 12))
            else:
                kw = self.smpc_base_load_kw
            load[t] = kw * 0.25
            key = (slot + timedelta(minutes=INTERVAL_MINUTES * t)).strftime(
                "%Y-%m-%d %H:%M:%S",
            )
            pred = self.predicted_by_timestamp.get(key)
            if pred:
                _, yld = pred
                solar[t] = max(0.0, yld)
        return np.maximum(load - solar, 0.0)

    def _run_smpc_if_needed(self):
        slot = self._current_slot()
        if self.smpc_last_slot == slot:
            return
        self.smpc_last_slot = slot

        cfg = self.smpc_cfg
        month = datetime.now(LOCAL_TZ).month
        price_fc = self._build_price_forecast(slot)
        cons_fc = self._build_consumption_profile(slot)

        if month in cfg.winter_months:
            heat = np.full(cfg.horizon_steps, self.smpc_heat_demand_base_kwh)
        elif month in (4, 10):
            heat = np.full(
                cfg.horizon_steps, self.smpc_heat_demand_base_kwh * 0.5,
            )
        else:
            heat = np.zeros(cfg.horizon_steps)

        inputs = SMPCInputs(
            electricity_price_eur_kwh=float(price_fc[0]),
            price_forecast_eur_kwh=price_fc,
            consumption_kwh=float(cons_fc[0]),
            consumption_forecast_kwh=cons_fc,
            ice_bank_kwh=self.ice_bank_kwh,
            heat_buffer_kwh=self.heat_buffer_kwh,
            heat_demand_forecast_kwh=heat,
            wkk_max_gas_m3=self.smpc_wkk_max_gas_m3,
            month=month,
        )
        try:
            self.last_smpc_outputs = self.smpc_calculator.solve(inputs)
            self.ice_bank_kwh = self.last_smpc_outputs.ice_bank_next_kwh
            self.heat_buffer_kwh = self.last_smpc_outputs.heat_buffer_next_kwh
        except Exception as exc:
            logger.warning("SMPC solve: %s", exc)

    # ------------------------------------------------------------------
    # Main tick (called every second)
    # ------------------------------------------------------------------

    def _on_tick(self):
        self._update_clock()
        self._maybe_run_pipeline()
        self._refresh_prices()
        self._update_price_labels()
        self._refresh_predictions()
        self._update_predict_labels()
        self._refresh_weather()
        self._update_center()
        self._run_smpc_if_needed()

        for (ttype, tid), label in self.value_labels.items():
            defn = self.tag_definitions.get(ttype, {}).get(tid, {})
            text, colour = self._sim_value(ttype, defn)
            label.setStyleSheet(value_css(colour))
            label.setText(text)

    # ------------------------------------------------------------------
    # Tag value simulation
    # ------------------------------------------------------------------

    def _sim_value(self, tag_type, defn):
        """Determine the display text and colour for a single tag."""
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
        return "--", "#64748b"

    def _sim_predicted_solar(self, sim):
        unit = sim.get("unit", "kW")
        dec = int(sim.get("decimals", 1))
        col = sim.get("color", "#0e7490")
        if self.current_predicted_power_kw is None:
            return f"-- {unit}", col
        return f"{self.current_predicted_power_kw:.{dec}f} {unit}", col

    def _sim_smpc(self, sim):
        field = sim.get("field", "")
        unit = sim.get("unit", "")
        dec = int(sim.get("decimals", 1))
        mult = float(sim.get("multiplier", 1))
        col = sim.get("color", "#0e7490")
        if self.last_smpc_outputs is None:
            return f"-- {unit}".strip(), col
        kpis = SMPCCalculator.outputs_to_dashboard_dict(self.last_smpc_outputs)
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
        if self.last_smpc_outputs is None:
            return "--", "#64748b"
        kpis = SMPCCalculator.outputs_to_dashboard_dict(self.last_smpc_outputs)
        state = above if kpis.get(field, 0) > thresh else below
        return state, cols.get(state, "#0e7490")

    def _sim_ice_status(self, sim):
        cols = sim.get("colors", {
            "CHARGE": "#0e7490", "DISCHARGE": "#f59e0b", "IDLE": "#16a34a",
        })
        if self.last_smpc_outputs is None:
            return "--", "#64748b"
        out = self.last_smpc_outputs
        if out.ice_bank_charge_kwh > 0.1:
            state = "CHARGE"
        elif out.ice_bank_discharge_kwh > 0.1:
            state = "DISCHARGE"
        else:
            state = "IDLE"
        return state, cols.get(state, "#0e7490")

    def _sim_setpoint(self, sim):
        cols = sim.get("colors", {
            "Lowered": "#16a34a", "Normal": "#0e7490", "Higher": "#dc2626",
        })
        if self.last_smpc_outputs is None:
            return "--", "#64748b"
        saving = self.last_smpc_outputs.cost_saving_eur
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

    def _open_popup(self, key):
        w = max(280, self.width() // 4)
        h = max(180, self.height() // 2)
        if key not in self.small_windows:
            dlg = QDialog(self)
            dlg.setWindowTitle(f"{key.capitalize()} Small Window")
            lay = QVBoxLayout(dlg)
            lay.addWidget(QLabel(
                "Placeholder window. Add your custom UI here.",
                alignment=Qt.AlignmentFlag.AlignCenter,
            ))
            self.small_windows[key] = dlg
        win = self.small_windows[key]
        win.resize(w, h)
        win.show()
        win.raise_()
        win.activateWindow()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _current_slot():
        """Return the current 15-minute slot as a tz-aware datetime."""
        now = datetime.now(LOCAL_TZ)
        return now.replace(
            minute=(now.minute // INTERVAL_MINUTES) * INTERVAL_MINUTES,
            second=0, microsecond=0,
        )

    @staticmethod
    def _next_quarter(now):
        """Return the start of the next 15-minute interval."""
        aligned = now.replace(
            minute=(now.minute // INTERVAL_MINUTES) * INTERVAL_MINUTES,
            second=0, microsecond=0,
        )
        return aligned + timedelta(minutes=INTERVAL_MINUTES)


# ===================================================================
# Entry point
# ===================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    config = load_dashboard_config()
    app = QApplication(sys.argv)
    window = ScadaWindow(
        config.get("inputs", []),
        config.get("outputs", []),
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
