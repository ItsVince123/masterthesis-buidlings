import sys
import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QScrollArea,
                             QPushButton, QDialog, QSizePolicy, QDateEdit)
from PyQt6.QtCore import Qt, QTimer, QPoint, QDate
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QBrush, QPolygon

import getPrice
import getWeather
import predict
import numpy as np
from dashboard_config import load_dashboard_config
from smpc_calculator import SMPCCalculator, SMPCInputs

def launch_scada_hmi(input_tags, output_tags):
    """
    Launches a corrected PyQt6 SCADA-style HMI.
    """
    
    class ScadaWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Thesis - Building Management System")
            self.setMinimumSize(1000, 700)
            self.small_windows = {}
            
            # Light dashboard theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #eef3f9;
                }

                QWidget {
                    color: #1f2937;
                    font-family: 'Segoe UI';
                    font-size: 11pt;
                }

                QFrame#ColumnContainer {
                    background-color: #ffffff;
                    border: 1px solid #d6dfeb;
                    border-radius: 14px;
                }

                QLabel#ColumnHeader {
                    color: #ffffff;
                    font-size: 10pt;
                    font-weight: 700;
                    letter-spacing: 1px;
                    padding: 12px;
                    border-top-left-radius: 14px;
                    border-top-right-radius: 14px;
                }

                QLabel#CenterTitle {
                    color: #364152;
                    font-size: 13pt;
                    font-weight: 600;
                    padding-top: 8px;
                }

                QFrame#EnergyRow {
                    background-color: #f5f9ff;
                    border: 1px solid #d8e3f3;
                    border-radius: 12px;
                }

                QLabel#EnergyLabel {
                    color: #334155;
                    font-size: 10pt;
                    font-weight: 600;
                }

                QLabel#EnergyValue {
                    color: #0b7a75;
                    font-family: 'Consolas';
                    font-size: 16pt;
                    font-weight: 700;
                }

                QFrame#TagRow {
                    background-color: #f8fbff;
                    border: 1px solid #dce5f2;
                    border-radius: 10px;
                }

                QLabel#TagName {
                    color: #27364f;
                    font-size: 10pt;
                    font-weight: 600;
                }

                QLabel#TagValue {
                    color: #0f766e;
                    font-family: 'Consolas';
                    font-size: 11pt;
                    font-weight: 700;
                }

                QLabel#SectionHeader {
                    color: #0b3a6e;
                    font-size: 9pt;
                    font-weight: 800;
                    letter-spacing: 1px;
                    padding: 6px 2px 2px 2px;
                }

                QFrame#SectionDivider {
                    background-color: #d4e2f4;
                    min-height: 1px;
                    max-height: 1px;
                    border: none;
                }
            """)

            # Main Layout setup
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            root_layout = QVBoxLayout(central_widget)
            root_layout.setContentsMargins(16, 16, 16, 16)
            root_layout.setSpacing(12)
            self.main_layout = QHBoxLayout()
            self.main_layout.setSpacing(14)
            root_layout.addLayout(self.main_layout)

            # Dictionary to track labels for updates
            self.value_labels = {}
            self.tag_definitions = {"input": {}, "output": {}}
            self.energy_labels = {}
            self.clock_label = None
            self.price_current_label = None
            self.price_avg_label = None
            self.price_next_refresh = None
            self.cached_price_rows = []
            self.cached_avg_48h = None
            self.actual_yield_label = None
            self.predict_next_refresh = None
            self.predicted_by_timestamp = {}
            self.current_predicted_power_kw = None
            self.current_predicted_yield_kwh = None
            self.weather_next_refresh = None
            self.uv_by_timestamp = {}
            self.current_uv_index = None
            self.center_price_graph_label = None
            self.center_solar_graph_label = None
            self.center_uv_value_label = None
            self.center_solar_value_label = None
            self.local_tz = ZoneInfo("Europe/Brussels")
            self.daily_fetch_hour = 14
            self.next_data_pipeline_run = None

            # --- SMPC integration ---
            raw_config = load_dashboard_config()
            smpc_building = raw_config.get("smpc", {}).get("building", {})
            self.smpc_calculator = SMPCCalculator()
            self.smpc_cfg = self.smpc_calculator.cfg
            self.ice_bank_kwh = self.smpc_cfg.ice_bank_initial_kwh
            self.heat_buffer_kwh = self.smpc_cfg.heat_buffer_initial_kwh
            self.last_smpc_outputs = None
            self.smpc_last_slot = None
            self.smpc_base_load_kw = smpc_building.get("base_load_kw", 80)
            self.smpc_peak_load_kw = smpc_building.get("peak_load_kw", 200)
            self.smpc_wkk_max_gas_m3 = smpc_building.get("wkk_max_gas_m3", 9.0)
            self.smpc_heat_demand_base_kwh = smpc_building.get("heat_demand_base_kwh", 20)

            side_column_min_width = 340

            # --- 1. LEFT COLUMN: INPUTS ---
            self.input_column, self.input_content = self.create_scada_column("System inputs", "#004d40")
            self.input_column.setMinimumWidth(side_column_min_width)
            self.main_layout.addWidget(self.input_column, stretch=1)

            # --- 2. CENTER PANEL ---
            self.center_panel = QFrame()
            self.center_panel.setObjectName("ColumnContainer")
            center_layout = QVBoxLayout(self.center_panel)
            center_layout.setContentsMargins(20, 18, 20, 18)
            center_layout.setSpacing(12)
            center_title = QLabel("SYSTEM OVERVIEW / SCHEMATIC", alignment=Qt.AlignmentFlag.AlignCenter)
            center_title.setObjectName("CenterTitle")
            center_layout.addWidget(center_title)
            self.add_center_widgets(center_layout)
            center_layout.addStretch()
            self.main_layout.addWidget(self.center_panel, stretch=1)

            # --- 3. RIGHT COLUMN: OUTPUTS ---
            self.output_column, self.output_content = self.create_scada_column("System outputs", "#4d1a1a")
            self.output_column.setMinimumWidth(side_column_min_width)
            self.main_layout.addWidget(self.output_column, stretch=1)

            normalized_inputs = self.normalize_tag_items(input_tags)
            normalized_outputs = self.normalize_tag_items(output_tags)
            self.register_tag_definitions("input", normalized_inputs)
            self.register_tag_definitions("output", normalized_outputs)

            # Populate side columns as flat row lists (no Energy In/Out sections).
            self.add_price_widgets(self.input_content)
            self.add_grouped_tags_to_column(self.input_content, self.build_groups_from_items(normalized_inputs, "INPUTS"), "input")
            self.add_grouped_tags_to_column(self.output_content, self.build_groups_from_items(normalized_outputs, "OUTPUTS"), "output")

            # Add one button below inputs and one below outputs.
            input_btn = self.create_column_button("+ Add input")
            input_btn.clicked.connect(lambda: self.open_small_window("input"))
            self.input_column.layout().addWidget(input_btn)

            output_btn = self.create_column_button("+ Add output")
            output_btn.clicked.connect(lambda: self.open_small_window("output"))
            self.output_column.layout().addWidget(output_btn)

            # Simulation Timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_simulation)
            self.timer.start(1000)

            # Run all data-get pipelines at launch, then schedule daily refresh at 14:00.
            self.run_data_pipeline(force=True)
            self.refresh_energy_prices(force=True)
            self.refresh_predict_data(force=True)
            self.refresh_weather_data(force=True)
            self.update_center_panel()

        def add_center_widgets(self, center_layout):
            """Add middle-column graph and KPI rows."""
            graph_card = QFrame()
            graph_card.setObjectName("TagRow")
            graph_layout = QVBoxLayout(graph_card)
            graph_layout.setContentsMargins(12, 10, 12, 10)
            graph_layout.setSpacing(8)

            graph_title = QLabel("Price Graph (48h)")
            graph_title.setObjectName("TagName")
            graph_layout.addWidget(graph_title)

            self.center_price_graph_label = QLabel("Loading price graph...")
            self.center_price_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.center_price_graph_label.setMinimumHeight(170)
            self.center_price_graph_label.setObjectName("TagValue")
            graph_layout.addWidget(self.center_price_graph_label)

            solar_graph_card = QFrame()
            solar_graph_card.setObjectName("TagRow")
            solar_graph_layout = QVBoxLayout(solar_graph_card)
            solar_graph_layout.setContentsMargins(12, 10, 12, 10)
            solar_graph_layout.setSpacing(8)

            solar_graph_title = QLabel("Predicted Solar Graph")
            solar_graph_title.setObjectName("TagName")
            solar_graph_layout.addWidget(solar_graph_title)

            self.center_solar_graph_label = QLabel("Loading solar graph...")
            self.center_solar_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.center_solar_graph_label.setMinimumHeight(170)
            self.center_solar_graph_label.setObjectName("TagValue")
            solar_graph_layout.addWidget(self.center_solar_graph_label)

            metrics_card = QFrame()
            metrics_card.setObjectName("TagRow")
            metrics_layout = QVBoxLayout(metrics_card)
            metrics_layout.setContentsMargins(12, 10, 12, 10)
            metrics_layout.setSpacing(8)

            uv_row = QHBoxLayout()
            uv_name = QLabel("UV Index")
            uv_name.setObjectName("TagName")
            self.center_uv_value_label = QLabel("--")
            self.center_uv_value_label.setObjectName("TagValue")
            uv_row.addWidget(uv_name)
            uv_row.addStretch()
            uv_row.addWidget(self.center_uv_value_label)

            solar_row = QHBoxLayout()
            solar_name = QLabel("Estimated Solar")
            solar_name.setObjectName("TagName")
            self.center_solar_value_label = QLabel("-- kW")
            self.center_solar_value_label.setObjectName("TagValue")
            solar_row.addWidget(solar_name)
            solar_row.addStretch()
            solar_row.addWidget(self.center_solar_value_label)

            metrics_layout.addLayout(uv_row)
            metrics_layout.addLayout(solar_row)

            center_layout.addWidget(graph_card)
            center_layout.addWidget(solar_graph_card)
            center_layout.addWidget(metrics_card)

            # Historical analysis button
            hist_btn = QPushButton("Historical SMPC Analysis")
            hist_btn.setMinimumHeight(44)
            hist_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0b3a6e;
                    color: white;
                    border: 1px solid #082f59;
                    border-radius: 8px;
                    padding: 10px 16px;
                    font-size: 10pt;
                    font-weight: 700;
                }
                QPushButton:hover { background-color: #0e4d8f; }
                QPushButton:pressed { background-color: #072a50; }
            """)
            hist_btn.clicked.connect(self.open_historical_analysis)
            center_layout.addWidget(hist_btn)

        def draw_series_graph(self, values, unit_label, start_label, end_label, line_color, current_index=None):
            """Render a line graph pixmap with timestamp and unit annotations."""
            width, height = 420, 170
            pixmap = QPixmap(width, height)
            pixmap.fill(QColor("#f8fbff"))

            if not values:
                painter = QPainter(pixmap)
                painter.setPen(QColor("#64748b"))
                painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, "No price data")
                painter.end()
                return pixmap

            pad_left = 48
            pad_right = 12
            pad_top = 18
            pad_bottom = 28

            min_val = min(values)
            max_val = max(values)
            spread = max(max_val - min_val, 1e-6)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Grid lines
            grid_pen = QPen(QColor("#dbe7f7"))
            grid_pen.setWidth(1)
            painter.setPen(grid_pen)
            for i in range(5):
                y = pad_top + i * (height - pad_top - pad_bottom) / 4
                painter.drawLine(pad_left, int(y), width - pad_right, int(y))

            # Data line
            line_pen = QPen(QColor(line_color))
            line_pen.setWidth(2)
            painter.setPen(line_pen)

            n = len(values)
            for i in range(n - 1):
                x1 = pad_left + i * (width - pad_left - pad_right) / max(n - 1, 1)
                x2 = pad_left + (i + 1) * (width - pad_left - pad_right) / max(n - 1, 1)
                y1 = height - pad_bottom - ((values[i] - min_val) / spread) * (height - pad_top - pad_bottom)
                y2 = height - pad_bottom - ((values[i + 1] - min_val) / spread) * (height - pad_top - pad_bottom)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # Current time marker.
            if current_index is not None and 0 <= current_index < n:
                x_now = pad_left + current_index * (width - pad_left - pad_right) / max(n - 1, 1)
                now_pen = QPen(QColor("#dc2626"))
                now_pen.setWidth(2)
                painter.setPen(now_pen)
                painter.drawLine(int(x_now), pad_top, int(x_now), height - pad_bottom)
                painter.drawText(int(x_now) + 4, pad_top + 12, "Now")

            text_pen = QPen(QColor("#334155"))
            painter.setPen(text_pen)
            painter.drawText(4, 16, f"{max_val:.1f} {unit_label}")
            painter.drawText(4, height - pad_bottom + 4, f"{min_val:.1f} {unit_label}")
            painter.drawText(pad_left, height - 6, start_label)
            painter.drawText(width - 120, height - 6, end_label)

            painter.end()
            return pixmap

        def draw_price_graph(self, prices, start_label, end_label, current_index=None):
            return self.draw_series_graph(prices, "EUR/MWh", start_label, end_label, "#1d4ed8", current_index)

        def draw_solar_graph(self, powers, start_label, end_label, current_index=None):
            return self.draw_series_graph(powers, "kW", start_label, end_label, "#f59e0b", current_index)

        def refresh_weather_data(self, force=False):
            """Load weather.csv periodically and cache UV index by timestamp."""
            now_local = datetime.now(self.local_tz)
            if not force and self.weather_next_refresh and now_local < self.weather_next_refresh:
                return

            weather_path = Path(r"C:/Users/32488/Documents/4de jaar/Masterproef/Dashboard/weather.csv")
            if not weather_path.exists():
                return

            try:
                loaded = {}
                with weather_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    for row in reader:
                        ts = row.get("Timestamp", "").strip()
                        if ts:
                            loaded[ts] = float(row.get("UV Index", "0").replace(",", "."))
                self.uv_by_timestamp = loaded
            except Exception as exc:
                print(f"Weather refresh warning: {exc}")
                return

            self.weather_next_refresh = now_local.replace(second=0, microsecond=0)
            self.weather_next_refresh = self.weather_next_refresh.replace(
                minute=(self.weather_next_refresh.minute // 15) * 15
            )
            self.weather_next_refresh += timedelta(minutes=15)

        def update_center_panel(self):
            """Refresh center graph and KPI labels."""
            now_local = datetime.now(self.local_tz)
            slot = now_local.replace(minute=(now_local.minute // 15) * 15, second=0, microsecond=0)
            slot_key = slot.strftime("%Y-%m-%d %H:%M:%S")

            if self.center_price_graph_label is not None:
                # Use full cached range so the current-slot marker is always in view.
                price_rows = self.cached_price_rows
                prices = [price for _, price, _ in price_rows]
                current_index = None
                if price_rows:
                    start_label = price_rows[0][0].astimezone(self.local_tz).strftime("%d/%m %H:%M")
                    end_label = price_rows[-1][0].astimezone(self.local_tz).strftime("%d/%m %H:%M")
                    for idx, (ts, _, _) in enumerate(price_rows):
                        if ts.astimezone(self.local_tz).replace(second=0, microsecond=0) == slot:
                            current_index = idx
                            break
                else:
                    start_label, end_label = "", ""
                self.center_price_graph_label.setPixmap(
                    self.draw_price_graph(prices, start_label, end_label, current_index)
                )

            if self.center_solar_graph_label is not None:
                # Use full cached range so the current-slot marker is always in view.
                solar_items = sorted(self.predicted_by_timestamp.items())
                solar_values = [power for _, (power, _) in solar_items]
                current_index = None
                if solar_items:
                    start_label = datetime.strptime(solar_items[0][0], "%Y-%m-%d %H:%M:%S").strftime("%d/%m %H:%M")
                    end_label = datetime.strptime(solar_items[-1][0], "%Y-%m-%d %H:%M:%S").strftime("%d/%m %H:%M")
                    for idx, (ts_key, _) in enumerate(solar_items):
                        if ts_key == slot_key:
                            current_index = idx
                            break
                else:
                    start_label, end_label = "", ""
                self.center_solar_graph_label.setPixmap(
                    self.draw_solar_graph(solar_values, start_label, end_label, current_index)
                )

            self.current_uv_index = self.uv_by_timestamp.get(slot_key)
            if self.center_uv_value_label is not None:
                self.center_uv_value_label.setText("--" if self.current_uv_index is None else f"{self.current_uv_index:.2f}")

            if self.center_solar_value_label is not None:
                if self.current_predicted_power_kw is None:
                    self.center_solar_value_label.setText("-- kW")
                else:
                    self.center_solar_value_label.setText(f"{self.current_predicted_power_kw:.1f} kW")

        def create_scada_column(self, title, color):
            """Creates a column with a header and no visible scrollbars."""
            outer_frame = QFrame()
            outer_frame.setObjectName("ColumnContainer")
            outer_layout = QVBoxLayout(outer_frame)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(0)

            header = QLabel(title)
            header.setObjectName("ColumnHeader")
            header.setStyleSheet(f"background-color: {color};")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            outer_layout.addWidget(header)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            # This widget holds the actual rows
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(10, 10, 10, 10)
            content_layout.setSpacing(12)
            content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(content_widget)
            
            outer_layout.addWidget(scroll)
            # Return both: the mounted frame and inner content host for tag rows.
            return outer_frame, content_widget

        def get_tag_name(self, tag_item):
            if isinstance(tag_item, dict):
                return str(tag_item.get("name", tag_item.get("id", "Unnamed")))
            return str(tag_item)

        def get_tag_id(self, tag_item):
            if isinstance(tag_item, dict):
                return str(tag_item.get("id", self.get_tag_name(tag_item)))
            return str(tag_item)

        def normalize_tag_items(self, tag_items):
            """Preserve order and remove duplicates by tag id."""
            normalized = []
            seen_ids = set()

            for item in tag_items:
                if isinstance(item, str):
                    item = {"id": item, "name": item}
                elif isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("name", item.get("id", "Unnamed"))
                    item.setdefault("id", item["name"])
                else:
                    continue

                tag_id = self.get_tag_id(item).strip().lower()
                if tag_id in seen_ids:
                    continue

                normalized.append(item)
                seen_ids.add(tag_id)

            return normalized

        def register_tag_definitions(self, tag_type, tag_items):
            for item in tag_items:
                self.tag_definitions[tag_type][self.get_tag_id(item)] = item

        def build_groups_from_items(self, tag_items, default_section):
            """Group tags based on configurable section names."""
            grouped = {}
            for item in tag_items:
                section = str(item.get("section", default_section)).strip() or default_section
                grouped.setdefault(section, []).append(item)
            return list(grouped.items())

        def resolve_icon(self, tag_item, tag_type):
            icon_key = ""
            if isinstance(tag_item, dict):
                icon_key = str(tag_item.get("icon", "")).strip().lower()

            icon_key_to_glyph = {
                "plug": "🔌",
                "sun": "☀️",
                "fire": "🔥",
                "heat": "♨️",
                "target": "🎯",
                "snowflake": "🧊",
                "car": "🚗",
                "globe": "🌍",
            }
            if icon_key in icon_key_to_glyph:
                return icon_key_to_glyph[icon_key]

            name = self.get_tag_name(tag_item)
            legacy_icon_map = {
                "GRID": "🔌",
                "Solar Panels": "☀️",
                "CHP": "🔥",
                "Heat Pump": "♨️",
                "Setpoint": "🎯",
                "Ice Banks": "🧊",
                "EVs": "🚗",
                "BEO": "🌍",
            }
            return legacy_icon_map.get(name, "🟢" if tag_type == "input" else "⚙️")

        def add_tags_to_column(self, container, tags, tag_type, row_height=42):
            for tag_item in tags:
                tag_name = self.get_tag_name(tag_item)
                tag_id = self.get_tag_id(tag_item)
                row = QFrame()
                row.setMinimumHeight(row_height)
                row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                row.setObjectName("TagRow")
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(12, 8, 12, 8)

                icon = self.resolve_icon(tag_item, tag_type)
                name_lbl = QLabel(f"{icon}  {tag_name}")
                name_lbl.setObjectName("TagName")
                name_lbl.setWordWrap(True)
                name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                
                val_lbl = QLabel("0.0")
                val_lbl.setObjectName("TagValue")
                
                row_layout.addWidget(name_lbl)
                row_layout.addStretch()
                row_layout.addWidget(val_lbl)

                properties = tag_item.get("properties", {}) if isinstance(tag_item, dict) else {}
                if properties:
                    row.setToolTip(str(properties))
                
                container.layout().addWidget(row)
                self.value_labels[(tag_type, tag_id)] = val_lbl

        def add_grouped_tags_to_column(self, container, grouped_tags, tag_type):
            """Render grouped tags with section headers and dividers."""
            first_group = True
            for section_title, tags in grouped_tags:
                if not tags:
                    continue

                if not first_group:
                    divider = QFrame()
                    divider.setObjectName("SectionDivider")
                    container.layout().addWidget(divider)

                header = QLabel(section_title)
                header.setObjectName("SectionHeader")
                container.layout().addWidget(header)

                self.add_tags_to_column(container, tags, tag_type)
                first_group = False

        def create_column_button(self, text):
            button = QPushButton(text)
            button.setMinimumHeight(52)
            button.setMinimumWidth(50)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2563eb;
                    color: white;
                    border: 1px solid #1d4ed8;
                    border-radius: 8px;
                    padding: 10px 16px;
                    font-size: 10pt;
                    font-weight: 700;
                }
                QPushButton:hover { background-color: #1d4ed8; }
                QPushButton:pressed { background-color: #1e40af; }
            """)
            return button

        def add_energy_block(self, container, column_key, tags, tag_type):
            split_index = (len(tags) + 1) // 2
            energy_in_tags = tags[:split_index]
            energy_out_tags = tags[split_index:]

            # Ensure required source tags are always shown in System Inputs / Energy In.
            if column_key == "System Inputs":
                required_energy_in = ["GRID", "Solar Panels", "CHP"]

                # Canonical names for case-insensitive cleanup (e.g. Grid/GRID, chp/CHP).
                canonical = {
                    "grid": "GRID",
                    "solar panels": "Solar Panels",
                    "chp": "CHP",
                }

                # Remove any variant of required tags from both sections.
                cleaned_in = []
                seen_in = set()
                for tag in energy_in_tags:
                    key = tag.strip().lower()
                    normalized = canonical.get(key, tag)
                    nkey = normalized.lower()
                    if nkey in canonical and normalized not in required_energy_in:
                        normalized = canonical[nkey]
                    if nkey not in seen_in and nkey not in {"grid", "solar panels", "chp"}:
                        cleaned_in.append(normalized)
                        seen_in.add(nkey)

                cleaned_out = []
                seen_out = set()
                for tag in energy_out_tags:
                    key = tag.strip().lower()
                    normalized = canonical.get(key, tag)
                    nkey = normalized.lower()
                    if nkey not in seen_out and nkey not in {"grid", "solar panels", "chp"}:
                        cleaned_out.append(normalized)
                        seen_out.add(nkey)

                energy_in_tags = required_energy_in + cleaned_in
                energy_out_tags = cleaned_out

            energy_in_row, energy_in_value, energy_in_tag_container = self.create_energy_row("Energy In", "-- kW")
            energy_out_row, energy_out_value, energy_out_tag_container = self.create_energy_row("Energy Out", "-- kW")

            if column_key == "System Inputs":
                self.add_price_widgets(energy_in_tag_container)

            self.add_tags_to_column(energy_in_tag_container, energy_in_tags, tag_type)
            self.add_tags_to_column(energy_out_tag_container, energy_out_tags, tag_type)

            container.layout().addWidget(energy_in_row, 1)
            container.layout().addWidget(energy_out_row, 1)

            self.energy_labels[(column_key, "Energy In")] = energy_in_value
            self.energy_labels[(column_key, "Energy Out")] = energy_out_value

        def add_price_widgets(self, tag_container):
            """Add clock and price/yield rows to the inputs column."""
            clock_row = QFrame()
            clock_row.setObjectName("TagRow")
            clock_layout = QHBoxLayout(clock_row)
            clock_layout.setContentsMargins(12, 8, 12, 8)
            clock_name = QLabel("🕒  Local time")
            clock_name.setObjectName("TagName")
            clock_name.setWordWrap(True)
            clock_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.clock_label = QLabel("--")
            self.clock_label.setObjectName("TagValue")
            clock_layout.addWidget(clock_name)
            clock_layout.addStretch()
            clock_layout.addWidget(self.clock_label)

            current_row = QFrame()
            current_row.setObjectName("TagRow")
            current_layout = QHBoxLayout(current_row)
            current_layout.setContentsMargins(12, 8, 12, 8)
            current_name = QLabel("⚡  Current 15m price")
            current_name.setObjectName("TagName")
            current_name.setWordWrap(True)
            current_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.price_current_label = QLabel("-- EUR/MWh")
            self.price_current_label.setObjectName("TagValue")
            current_layout.addWidget(current_name)
            current_layout.addStretch()
            current_layout.addWidget(self.price_current_label)

            avg_row = QFrame()
            avg_row.setObjectName("TagRow")
            avg_layout = QHBoxLayout(avg_row)
            avg_layout.setContentsMargins(12, 8, 12, 8)
            avg_name = QLabel("📈  48h average")
            avg_name.setObjectName("TagName")
            avg_name.setWordWrap(True)
            avg_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.price_avg_label = QLabel("-- EUR/MWh")
            self.price_avg_label.setObjectName("TagValue")
            avg_layout.addWidget(avg_name)
            avg_layout.addStretch()
            avg_layout.addWidget(self.price_avg_label)

            actual_row = QFrame()
            actual_row.setObjectName("TagRow")
            actual_layout = QHBoxLayout(actual_row)
            actual_layout.setContentsMargins(12, 8, 12, 8)
            actual_name = QLabel("🔆  Actual Yield")
            actual_name.setObjectName("TagName")
            actual_name.setWordWrap(True)
            actual_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.actual_yield_label = QLabel("-- kWh")
            self.actual_yield_label.setObjectName("TagValue")
            actual_layout.addWidget(actual_name)
            actual_layout.addStretch()
            actual_layout.addWidget(self.actual_yield_label)

            tag_container.layout().addWidget(clock_row)
            tag_container.layout().addWidget(current_row)
            tag_container.layout().addWidget(avg_row)
            tag_container.layout().addWidget(actual_row)

            self.update_clock_label()

        def update_clock_label(self):
            if self.clock_label is None:
                return
            now_local = datetime.now(self.local_tz)
            self.clock_label.setText(now_local.strftime("%d/%m/%Y %H:%M:%S"))
            self.clock_label.setStyleSheet("color: #0e7490; font-family: 'Consolas'; font-weight: 700;")

        def refresh_predict_data(self, force=False):
            """Load predict.csv periodically and cache values by timestamp."""
            now_local = datetime.now(self.local_tz)
            if not force and self.predict_next_refresh and now_local < self.predict_next_refresh:
                return

            predict_path = Path(r"C:/Users/32488/Documents/4de jaar/Masterproef/Dashboard/predict.csv")
            if not predict_path.exists():
                return

            try:
                loaded = {}
                with predict_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    for row in reader:
                        ts = row.get("Timestamp", "").strip()
                        if not ts:
                            continue
                        power = float(row.get("Predicted Power (kW)", "0").replace(",", "."))
                        yld = float(row.get("Predicted Yield (kWh)", "0").replace(",", "."))
                        loaded[ts] = (power, yld)

                self.predicted_by_timestamp = loaded
            except Exception as exc:
                print(f"Predict refresh warning: {exc}")
                return

            self.predict_next_refresh = now_local.replace(second=0, microsecond=0)
            self.predict_next_refresh = self.predict_next_refresh.replace(
                minute=(self.predict_next_refresh.minute // 15) * 15
            )
            self.predict_next_refresh += timedelta(minutes=15)

            self.update_predict_labels()

        def update_predict_labels(self):
            """Update current predicted solar power and actual yield label."""
            now_local = datetime.now(self.local_tz)
            slot = now_local.replace(minute=(now_local.minute // 15) * 15, second=0, microsecond=0)
            slot_key = slot.strftime("%Y-%m-%d %H:%M:%S")

            current = self.predicted_by_timestamp.get(slot_key)
            if current is None:
                self.current_predicted_power_kw = None
                self.current_predicted_yield_kwh = None
                if self.actual_yield_label is not None:
                    self.actual_yield_label.setText("-- kWh")
                return

            self.current_predicted_power_kw, self.current_predicted_yield_kwh = current

            if self.actual_yield_label is not None:
                self.actual_yield_label.setText("-- kWh")
                self.actual_yield_label.setStyleSheet("color: #64748b; font-family: 'Consolas'; font-weight: 700;")

        def refresh_energy_prices(self, force=False):
            """Refresh ENTSO-E prices periodically and color current value vs 48h average."""
            if self.price_current_label is None or self.price_avg_label is None:
                return

            now_local = datetime.now(self.local_tz)
            if not force and self.price_next_refresh and now_local < self.price_next_refresh:
                return

            try:
                rows, avg_48h = getPrice.get_flagged_next_day_prices()
                self.cached_price_rows = rows
                self.cached_avg_48h = avg_48h
                self.price_next_refresh = now_local.replace(second=0, microsecond=0)
                self.price_next_refresh = self.price_next_refresh.replace(
                    minute=(self.price_next_refresh.minute // 15) * 15
                )
                self.price_next_refresh += timedelta(minutes=15)
            except (ValueError, ConnectionError, TimeoutError) as exc:
                if force:
                    self.price_current_label.setText("API error")
                    self.price_avg_label.setText("API error")
                print(f"Price refresh warning: {exc}")
                return
            except Exception as exc:
                if force:
                    self.price_current_label.setText("API error")
                    self.price_avg_label.setText("API error")
                print(f"Unexpected price refresh warning: {exc}")
                return

            self.update_price_labels()

        def update_price_labels(self):
            """Update price labels based on the current Brussels 15-minute slot."""
            now_local = datetime.now(self.local_tz)
            slot = now_local.replace(minute=(now_local.minute // 15) * 15, second=0, microsecond=0)

            if not self.cached_price_rows or self.cached_avg_48h is None:
                return

            current_item = None
            for ts, price, _ in self.cached_price_rows:
                ts_local = ts.astimezone(self.local_tz)
                if ts_local == slot:
                    current_item = (ts, price)
                    break

            if current_item is None:
                return

            _, current_price = current_item
            self.price_current_label.setText(f"{current_price:.2f} EUR/MWh")

            self.price_avg_label.setText(f"{self.cached_avg_48h:.2f} EUR/MWh")

            # Green if current price is lower than average, red if higher/equal.
            if current_price < self.cached_avg_48h:
                self.price_current_label.setStyleSheet("color: #16a34a; font-family: 'Consolas'; font-weight: 700;")
            else:
                self.price_current_label.setStyleSheet("color: #dc2626; font-family: 'Consolas'; font-weight: 700;")

            self.price_avg_label.setStyleSheet("color: #0f766e; font-family: 'Consolas'; font-weight: 700;")

        def create_energy_row(self, label_text, initial_value):
            row = QFrame()
            row.setObjectName("EnergyRow")
            row.setMinimumHeight(130)

            layout = QVBoxLayout(row)
            layout.setContentsMargins(14, 10, 14, 10)
            layout.setSpacing(8)

            header_layout = QHBoxLayout()

            title = QLabel(label_text)
            title.setObjectName("EnergyLabel")

            value = QLabel(initial_value)
            value.setObjectName("EnergyValue")

            header_layout.addWidget(title)
            header_layout.addStretch()
            header_layout.addWidget(value)

            tags_widget = QWidget()
            tags_layout = QVBoxLayout(tags_widget)
            tags_layout.setContentsMargins(0, 0, 0, 0)
            tags_layout.setSpacing(6)
            tags_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            layout.addLayout(header_layout)
            layout.addWidget(tags_widget)

            return row, value, tags_widget

        def compute_next_pipeline_run(self, now_local=None):
            """Return the next local datetime when data pipeline must run (daily 14:00)."""
            if now_local is None:
                now_local = datetime.now(self.local_tz)

            target_today = now_local.replace(
                hour=self.daily_fetch_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
            if now_local < target_today:
                return target_today
            return target_today + timedelta(days=1)

        def run_data_pipeline(self, force=False):
            """Run weather/predict/price data getters now, then plan next daily 14:00 run."""
            now_local = datetime.now(self.local_tz)
            if not force and self.next_data_pipeline_run and now_local < self.next_data_pipeline_run:
                return

            dashboard_dir = Path(__file__).resolve().parent

            # Weather fetch/export.
            try:
                client = getWeather.build_client()
                today = now_local.strftime("%Y-%m-%d")
                tomorrow = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
                weather_data = getWeather.fetch_weather(client, 50.85045, 4.34878, today, tomorrow)
                getWeather.export_csv(weather_data, dashboard_dir / "weather.csv")
            except Exception as exc:
                print(f"Weather pipeline warning: {exc}")

            # Prediction export from weather CSV.
            try:
                weather_rows = predict.read_weather_csv(dashboard_dir / "weather.csv")
                if weather_rows:
                    predict.export_predictions(weather_rows, dashboard_dir / "predict.csv", capacity_kwp=100.0)
            except Exception as exc:
                print(f"Predict pipeline warning: {exc}")

            # Price fetch/export.
            try:
                getPrice.fetch_and_save_prices(output_filename="prices.csv", output_dir=dashboard_dir, reference_time=now_local)
            except Exception as exc:
                print(f"Price pipeline warning: {exc}")

            self.next_data_pipeline_run = self.compute_next_pipeline_run(now_local)

        def maybe_run_scheduled_pipeline(self):
            """Check whether it's time for the daily pipeline and run it if due."""
            now_local = datetime.now(self.local_tz)
            if self.next_data_pipeline_run is None:
                self.next_data_pipeline_run = self.compute_next_pipeline_run(now_local)

            if now_local >= self.next_data_pipeline_run:
                self.run_data_pipeline(force=True)
                # Ensure view cache refresh after new files/data are generated.
                self.refresh_predict_data(force=True)
                self.refresh_weather_data(force=True)
                self.refresh_energy_prices(force=True)
                self.update_center_panel()

        # --- SMPC helper methods ---

        def _build_price_forecast(self, slot):
            """Build price forecast array (EUR/kWh) for the SMPC horizon."""
            H = self.smpc_cfg.horizon_steps
            if not self.cached_price_rows:
                return np.full(H, 0.10)

            sorted_prices = sorted(self.cached_price_rows, key=lambda r: r[0])
            price_timeline = []
            for ts, price_mwh, _ in sorted_prices:
                ts_local = ts.astimezone(self.local_tz).replace(second=0, microsecond=0)
                price_timeline.append((ts_local, price_mwh / 1000.0))

            prices = np.zeros(H)
            for t in range(H):
                target = slot + timedelta(minutes=15 * t)
                best_price = price_timeline[0][1]
                for ts, p in price_timeline:
                    if ts <= target:
                        best_price = p
                    else:
                        break
                prices[t] = best_price
            return prices

        def _build_consumption_profile(self, slot):
            """Build synthetic net-consumption forecast (kWh per 15-min step)."""
            H = self.smpc_cfg.horizon_steps
            start_hour = slot.hour + slot.minute / 60.0

            load_kwh = np.zeros(H)
            solar_kwh = np.zeros(H)

            for t in range(H):
                h = (start_hour + t * 0.25) % 24
                if 6 <= h <= 18:
                    load_kw = self.smpc_base_load_kw + (
                        self.smpc_peak_load_kw - self.smpc_base_load_kw
                    ) * max(0.0, np.sin(np.pi * (h - 6) / 12))
                else:
                    load_kw = self.smpc_base_load_kw
                load_kwh[t] = load_kw * 0.25

                ts = slot + timedelta(minutes=15 * t)
                ts_key = ts.strftime("%Y-%m-%d %H:%M:%S")
                pred = self.predicted_by_timestamp.get(ts_key)
                if pred:
                    _, yld = pred
                    solar_kwh[t] = max(0.0, yld)

            return np.maximum(load_kwh - solar_kwh, 0.0)

        def run_smpc_if_needed(self):
            """Run SMPC optimisation when the 15-min slot changes."""
            now_local = datetime.now(self.local_tz)
            slot = now_local.replace(
                minute=(now_local.minute // 15) * 15, second=0, microsecond=0
            )
            if self.smpc_last_slot == slot:
                return
            self.smpc_last_slot = slot

            cfg = self.smpc_cfg
            month = now_local.month

            price_fc = self._build_price_forecast(slot)
            consumption_fc = self._build_consumption_profile(slot)

            if month in cfg.winter_months:
                heat_demand = np.full(cfg.horizon_steps, self.smpc_heat_demand_base_kwh)
            elif month in (4, 10):
                heat_demand = np.full(cfg.horizon_steps, self.smpc_heat_demand_base_kwh * 0.5)
            else:
                heat_demand = np.zeros(cfg.horizon_steps)

            inputs = SMPCInputs(
                electricity_price_eur_kwh=float(price_fc[0]),
                price_forecast_eur_kwh=price_fc,
                consumption_kwh=float(consumption_fc[0]),
                consumption_forecast_kwh=consumption_fc,
                ice_bank_kwh=self.ice_bank_kwh,
                heat_buffer_kwh=self.heat_buffer_kwh,
                heat_demand_forecast_kwh=heat_demand,
                wkk_max_gas_m3=self.smpc_wkk_max_gas_m3,
                month=month,
            )

            try:
                self.last_smpc_outputs = self.smpc_calculator.solve(inputs)
                self.ice_bank_kwh = self.last_smpc_outputs.ice_bank_next_kwh
                self.heat_buffer_kwh = self.last_smpc_outputs.heat_buffer_next_kwh
            except Exception as exc:
                print(f"SMPC solve warning: {exc}")

        def update_simulation(self):
            self.update_clock_label()
            self.maybe_run_scheduled_pipeline()
            self.refresh_energy_prices()
            self.update_price_labels()
            self.refresh_predict_data()
            self.update_predict_labels()
            self.refresh_weather_data()
            self.update_center_panel()
            self.run_smpc_if_needed()

            for (tag_type, tag_id), label in self.value_labels.items():
                tag_definition = self.tag_definitions.get(tag_type, {}).get(tag_id, {})
                val, color = self.simulate_tag_value(tag_type, tag_definition)
                label.setStyleSheet(f"color: {color}; font-family: 'Consolas'; font-weight: 700;")
                label.setText(val)

        def simulate_tag_value(self, tag_type, tag_definition):
            simulation = tag_definition.get("simulation", {}) if isinstance(tag_definition, dict) else {}
            mode = str(simulation.get("mode", "legacy")).strip().lower()

            if mode == "predicted_solar":
                unit = simulation.get("unit", "kW")
                decimals = int(simulation.get("decimals", 1))
                if self.current_predicted_power_kw is None:
                    return f"-- {unit}", simulation.get("color", "#0e7490")
                return f"{self.current_predicted_power_kw:.{decimals}f} {unit}", simulation.get("color", "#0e7490")

            if mode == "smpc":
                field = simulation.get("field", "")
                unit = simulation.get("unit", "")
                decimals = int(simulation.get("decimals", 1))
                multiplier = float(simulation.get("multiplier", 1))
                color = simulation.get("color", "#0e7490")
                if self.last_smpc_outputs is None:
                    return f"-- {unit}".strip(), color
                kpis = SMPCCalculator.outputs_to_dashboard_dict(self.last_smpc_outputs)
                value = kpis.get(field)
                if value is None:
                    return f"-- {unit}".strip(), color
                value = value * multiplier
                rendered = f"{value:.{decimals}f}"
                if unit:
                    rendered = f"{rendered} {unit}"
                return rendered, color

            if mode == "smpc_state":
                field = simulation.get("field", "")
                threshold = float(simulation.get("threshold", 0.1))
                above_label = simulation.get("above", "ON")
                below_label = simulation.get("below", "OFF")
                color_map = simulation.get("colors", {})
                if self.last_smpc_outputs is None:
                    return "--", "#64748b"
                kpis = SMPCCalculator.outputs_to_dashboard_dict(self.last_smpc_outputs)
                value = kpis.get(field, 0)
                state = above_label if value > threshold else below_label
                return state, color_map.get(state, "#0e7490")

            if mode == "smpc_ice_status":
                color_map = simulation.get("colors", {
                    "CHARGE": "#0e7490",
                    "DISCHARGE": "#f59e0b",
                    "IDLE": "#16a34a"
                })
                if self.last_smpc_outputs is None:
                    return "--", "#64748b"
                charge = self.last_smpc_outputs.ice_bank_charge_kwh
                discharge = self.last_smpc_outputs.ice_bank_discharge_kwh
                if charge > 0.1:
                    state = "CHARGE"
                elif discharge > 0.1:
                    state = "DISCHARGE"
                else:
                    state = "IDLE"
                return state, color_map.get(state, "#0e7490")

            if mode == "smpc_setpoint":
                color_map = simulation.get("colors", {
                    "Lowered": "#16a34a",
                    "Normal": "#0e7490",
                    "Higher": "#dc2626"
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
                return state, color_map.get(state, "#0e7490")

            if mode == "random_range":
                unit = simulation.get("unit", "")
                rendered = f"-- {unit}".strip()
                return rendered, "#64748b"

            if mode == "state_options":
                return "--", "#64748b"

            return "--", "#64748b"

        def open_historical_analysis(self):
            """Open the historical SMPC cost analysis dialog."""
            dialog = HistoricalAnalysisDialog(self)
            dialog.exec()

        def open_small_window(self, window_key):
            # Approximate 1/8 area of the main window.
            popup_width = max(280, self.width() // 4)
            popup_height = max(180, self.height() // 2)

            if window_key not in self.small_windows:
                dialog = QDialog(self)
                dialog.setWindowTitle(f"{window_key.capitalize()} Small Window")
                popup_layout = QVBoxLayout(dialog)
                popup_layout.addWidget(
                    QLabel("Placeholder window. Add your custom UI here.", alignment=Qt.AlignmentFlag.AlignCenter)
                )
                self.small_windows[window_key] = dialog

            window = self.small_windows[window_key]
            window.resize(popup_width, popup_height)
            window.show()
            window.raise_()
            window.activateWindow()

    class HistoricalAnalysisDialog(QDialog):
        """Dialog that simulates a past day using SMPC and shows baseline vs optimised costs."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Historical SMPC Analysis")
            self.setMinimumSize(740, 680)
            self.setStyleSheet("""
                QDialog { background-color: #eef3f9; }
                QLabel { color: #1f2937; font-family: 'Segoe UI'; font-size: 10pt; }
                QDateEdit {
                    background-color: #ffffff;
                    border: 1px solid #d6dfeb;
                    border-radius: 6px;
                    padding: 6px 10px;
                    font-size: 10pt;
                    min-height: 28px;
                }
            """)

            # Grab SMPC config from parent ScadaWindow
            self._smpc_cfg = parent.smpc_cfg if parent else None
            self._smpc_calculator = parent.smpc_calculator if parent else None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 16, 20, 16)
            layout.setSpacing(12)

            title = QLabel("Select a date to simulate SMPC optimisation")
            title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #0b3a6e;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title)

            # Date selector row
            selector_row = QHBoxLayout()
            selector_row.setSpacing(10)
            selector_label = QLabel("Date:")
            selector_label.setStyleSheet("font-weight: 600;")
            self.date_edit = QDateEdit()
            self.date_edit.setCalendarPopup(True)
            self.date_edit.setDisplayFormat("yyyy-MM-dd")
            self.date_edit.setDate(QDate.currentDate().addDays(-1))
            self.date_edit.setMaximumDate(QDate.currentDate().addDays(-1))
            self.date_edit.setMinimumWidth(180)

            run_btn = QPushButton("Run Analysis")
            run_btn.setMinimumHeight(36)
            run_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2563eb; color: white;
                    border: 1px solid #1d4ed8; border-radius: 6px;
                    padding: 6px 18px; font-size: 10pt; font-weight: 700;
                }
                QPushButton:hover { background-color: #1d4ed8; }
            """)
            run_btn.clicked.connect(self._run_analysis)
            selector_row.addWidget(selector_label)
            selector_row.addWidget(self.date_edit, stretch=1)
            selector_row.addWidget(run_btn)
            layout.addLayout(selector_row)

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

            # KPI card area
            self.kpi_frame = QFrame()
            self.kpi_frame.setStyleSheet(
                "QFrame { background-color: #ffffff; border: 1px solid #d6dfeb; border-radius: 10px; }"
            )
            kpi_layout = QVBoxLayout(self.kpi_frame)
            kpi_layout.setContentsMargins(16, 12, 16, 12)
            kpi_layout.setSpacing(6)
            self.kpi_title = QLabel("Results")
            self.kpi_title.setStyleSheet("font-size: 11pt; font-weight: 700; color: #0b3a6e; border: none;")
            kpi_layout.addWidget(self.kpi_title)

            self.kpi_labels = {}
            for key, label_text in [
                ("baseline", "Baseline cost (no optimisation)"),
                ("smpc", "SMPC optimised cost"),
                ("saving", "Total savings"),
                ("pct", "Savings %"),
                ("slots", "15-min slots simulated"),
                ("season", "Detected season"),
                ("ice_charged", "Total ice bank charged"),
                ("ice_discharged", "Total ice bank discharged"),
                ("wkk_gas", "Total WKK gas used"),
                ("peak_baseline", "Peak cost slot (baseline)"),
                ("peak_smpc", "Peak cost slot (SMPC)"),
            ]:
                row = QHBoxLayout()
                name = QLabel(label_text)
                name.setStyleSheet("font-weight: 600; border: none;")
                val = QLabel("--")
                val.setStyleSheet("font-family: 'Consolas'; font-weight: 700; color: #0f766e; border: none;")
                row.addWidget(name)
                row.addStretch()
                row.addWidget(val)
                kpi_layout.addLayout(row)
                self.kpi_labels[key] = val

            layout.addWidget(self.kpi_frame)

        def _run_analysis(self):
            """Fetch real data for selected day, run SMPC for each slot, show results."""
            selected = self.date_edit.date().toPyDate()
            day_str = selected.strftime("%Y-%m-%d")
            next_day_str = (selected + timedelta(days=1)).strftime("%Y-%m-%d")

            if self._smpc_cfg is None or self._smpc_calculator is None:
                self.status_label.setText("SMPC not initialised — cannot run analysis.")
                return

            self.status_label.setText(f"Fetching prices for {day_str}...")
            QApplication.processEvents()

            brussels = ZoneInfo("Europe/Brussels")
            cfg = self._smpc_cfg
            calc = self._smpc_calculator

            # --- 1. Fetch ENTSO-E prices for the selected day ---
            try:
                day_start = datetime(selected.year, selected.month, selected.day, tzinfo=brussels)
                day_end = day_start + timedelta(days=1)
                start_utc = day_start.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")
                end_utc = day_end.astimezone(ZoneInfo("UTC")).strftime("%Y%m%d%H%M")

                xml_text = getPrice.fetch_prices("10YBE----------2", start_utc, end_utc)
                raw_prices = getPrice.parse_prices(xml_text)
                if not raw_prices:
                    self.status_label.setText(f"No ENTSO-E prices available for {day_str}.")
                    return
            except Exception as exc:
                self.status_label.setText(f"Price fetch failed: {exc}")
                return

            self.status_label.setText(f"Fetching weather for {day_str}...")
            QApplication.processEvents()

            # --- 2. Fetch weather for solar prediction ---
            try:
                client = getWeather.build_client()
                weather_data = getWeather.fetch_weather(
                    client, 50.85045, 4.34878, day_str, day_str
                )
            except Exception as exc:
                self.status_label.setText(f"Weather fetch failed: {exc}")
                return

            # Build solar predictions from weather
            solar_by_slot = {}
            for i, ts in enumerate(weather_data["timestamps"]):
                ts_key = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, 'strftime') else str(ts)
                row = predict.WeatherRow(
                    timestamp=ts_key,
                    temperature_c=weather_data["temperature"][i],
                    uv_index=weather_data["uv_index"][i],
                    wind_kmh=weather_data["wind_speed"][i],
                )
                power_kw = predict.predict_power_kw(row, capacity_kwp=100.0)
                yield_kwh = power_kw * 0.25  # 15-min interval
                solar_by_slot[ts_key] = yield_kwh

            # --- 3. Build 15-min price timeline for the day ---
            sorted_prices = sorted(raw_prices, key=lambda r: r[0])
            price_timeline = []
            for ts, price_mwh in sorted_prices:
                ts_local = ts.astimezone(brussels).replace(second=0, microsecond=0)
                price_timeline.append((ts_local, price_mwh / 1000.0))  # EUR/kWh

            # Generate 96 slots (24h × 4 per hour)
            slots = []
            for s in range(96):
                slots.append(day_start + timedelta(minutes=15 * s))

            def price_at(target):
                best = price_timeline[0][1] if price_timeline else 0.10
                for ts, p in price_timeline:
                    if ts <= target:
                        best = p
                    else:
                        break
                return best

            self.status_label.setText(f"Running SMPC simulation for {day_str}...")
            QApplication.processEvents()

            # --- 4. Simulate each 15-min slot with SMPC ---
            month = selected.month
            H = cfg.horizon_steps
            ice_bank = cfg.ice_bank_initial_kwh
            heat_buffer = cfg.heat_buffer_initial_kwh

            # Building load parameters (same as dashboard)
            base_load_kw = 80.0
            peak_load_kw = 200.0
            wkk_max_gas = 9.0
            heat_demand_base = 20.0

            try:
                parent = self.parent()
                if parent and hasattr(parent, 'smpc_base_load_kw'):
                    base_load_kw = parent.smpc_base_load_kw
                    peak_load_kw = parent.smpc_peak_load_kw
                    wkk_max_gas = parent.smpc_wkk_max_gas_m3
                    heat_demand_base = parent.smpc_heat_demand_base_kwh
            except Exception:
                pass

            baseline_costs = np.zeros(96)
            smpc_costs = np.zeros(96)
            total_ice_charged = 0.0
            total_ice_discharged = 0.0
            total_wkk_gas = 0.0
            season_detected = "unknown"

            for s_idx, slot in enumerate(slots):
                slot_hour = slot.hour + slot.minute / 60.0

                # Price forecast from this slot onward
                price_fc = np.zeros(H)
                for t in range(H):
                    target = slot + timedelta(minutes=15 * t)
                    price_fc[t] = price_at(target)

                # Consumption profile (synthetic load - solar)
                load_kwh = np.zeros(H)
                solar_kwh = np.zeros(H)
                for t in range(H):
                    h = (slot_hour + t * 0.25) % 24
                    if 6 <= h <= 18:
                        load_kw = base_load_kw + (peak_load_kw - base_load_kw) * max(0.0, np.sin(np.pi * (h - 6) / 12))
                    else:
                        load_kw = base_load_kw
                    load_kwh[t] = load_kw * 0.25

                    ts_target = slot + timedelta(minutes=15 * t)
                    ts_key = ts_target.strftime("%Y-%m-%d %H:%M:%S")
                    solar_kwh[t] = max(0.0, solar_by_slot.get(ts_key, 0.0))

                consumption_fc = np.maximum(load_kwh - solar_kwh, 0.0)

                # Heat demand
                if month in cfg.winter_months:
                    heat_demand = np.full(H, heat_demand_base)
                elif month in (4, 10):
                    heat_demand = np.full(H, heat_demand_base * 0.5)
                else:
                    heat_demand = np.zeros(H)

                current_price = float(price_fc[0])
                current_consumption = float(consumption_fc[0])

                # Baseline: no optimisation, just grid draw × price
                baseline_costs[s_idx] = current_consumption * current_price

                inputs = SMPCInputs(
                    electricity_price_eur_kwh=current_price,
                    price_forecast_eur_kwh=price_fc,
                    consumption_kwh=current_consumption,
                    consumption_forecast_kwh=consumption_fc,
                    ice_bank_kwh=ice_bank,
                    heat_buffer_kwh=heat_buffer,
                    heat_demand_forecast_kwh=heat_demand,
                    wkk_max_gas_m3=wkk_max_gas,
                    month=month,
                )

                try:
                    outputs = calc.solve(inputs)
                    smpc_costs[s_idx] = outputs.smpc_cost_eur
                    ice_bank = outputs.ice_bank_next_kwh
                    heat_buffer = outputs.heat_buffer_next_kwh
                    total_ice_charged += outputs.ice_bank_charge_kwh
                    total_ice_discharged += outputs.ice_bank_discharge_kwh
                    total_wkk_gas += outputs.wkk_gas_setpoint_m3
                    if s_idx == 0:
                        season_detected = outputs.season
                except Exception:
                    smpc_costs[s_idx] = baseline_costs[s_idx]

            # --- 5. Draw results ---
            x_labels = [sl.strftime("%H:%M") for sl in slots]
            pixmap = self._draw_comparison_graph(baseline_costs, smpc_costs, x_labels)
            self.graph_label.setPixmap(pixmap)

            # --- 6. Update KPIs ---
            total_baseline = baseline_costs.sum()
            total_smpc = smpc_costs.sum()
            saving = total_baseline - total_smpc
            pct = (saving / total_baseline * 100) if total_baseline != 0 else 0.0

            peak_base_idx = int(np.argmax(baseline_costs))
            peak_smpc_idx = int(np.argmax(smpc_costs))

            self.kpi_title.setText(f"Results for {day_str}")
            self.kpi_labels["baseline"].setText(f"\u20ac{total_baseline:.4f}")
            self.kpi_labels["smpc"].setText(f"\u20ac{total_smpc:.4f}")

            saving_color = "#16a34a" if saving >= 0 else "#dc2626"
            self.kpi_labels["saving"].setText(f"\u20ac{saving:.4f}")
            self.kpi_labels["saving"].setStyleSheet(
                f"font-family: 'Consolas'; font-weight: 700; color: {saving_color}; border: none;"
            )
            self.kpi_labels["pct"].setText(f"{pct:.2f}%")
            self.kpi_labels["pct"].setStyleSheet(
                f"font-family: 'Consolas'; font-weight: 700; color: {saving_color}; border: none;"
            )
            self.kpi_labels["slots"].setText("96")
            self.kpi_labels["season"].setText(season_detected.capitalize())
            self.kpi_labels["ice_charged"].setText(f"{total_ice_charged:.2f} kWh")
            self.kpi_labels["ice_discharged"].setText(f"{total_ice_discharged:.2f} kWh")
            self.kpi_labels["wkk_gas"].setText(f"{total_wkk_gas:.2f} m\u00b3")
            self.kpi_labels["peak_baseline"].setText(
                f"{x_labels[peak_base_idx]} (\u20ac{baseline_costs[peak_base_idx]:.4f})"
            )
            self.kpi_labels["peak_smpc"].setText(
                f"{x_labels[peak_smpc_idx]} (\u20ac{smpc_costs[peak_smpc_idx]:.4f})"
            )

            self.status_label.setText(f"Simulation complete for {day_str}.")

        def _draw_comparison_graph(self, series_a, series_b, x_labels):
            """Draw two overlaid line series: baseline (red) vs SMPC (green)."""
            width, height = 700, 260
            pixmap = QPixmap(width, height)
            pixmap.fill(QColor("#f8fbff"))

            pad_left, pad_right, pad_top, pad_bottom = 60, 20, 24, 40
            draw_w = width - pad_left - pad_right
            draw_h = height - pad_top - pad_bottom

            all_vals = np.concatenate([series_a, series_b])
            min_val = float(np.min(all_vals))
            max_val = float(np.max(all_vals))
            spread = max(max_val - min_val, 1e-6)

            n = len(series_a)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Grid lines
            grid_pen = QPen(QColor("#dbe7f7"))
            grid_pen.setWidth(1)
            painter.setPen(grid_pen)
            for i in range(5):
                y = pad_top + i * draw_h / 4
                painter.drawLine(pad_left, int(y), width - pad_right, int(y))

            def x_pos(i):
                return pad_left + i * draw_w / max(n - 1, 1)

            def y_pos(v):
                return height - pad_bottom - ((v - min_val) / spread) * draw_h

            # Fill savings area (where baseline > smpc)
            for i in range(n - 1):
                x1 = int(x_pos(i))
                x2 = int(x_pos(i + 1))
                ya1, ya2 = int(y_pos(series_a[i])), int(y_pos(series_a[i + 1]))
                yb1, yb2 = int(y_pos(series_b[i])), int(y_pos(series_b[i + 1]))
                if series_a[i] > series_b[i] or series_a[i + 1] > series_b[i + 1]:
                    fill_pen = QPen(Qt.PenStyle.NoPen)
                    painter.setPen(fill_pen)
                    painter.setBrush(QBrush(QColor(22, 163, 74, 40)))
                    poly = QPolygon([
                        QPoint(x1, ya1), QPoint(x2, ya2),
                        QPoint(x2, yb2), QPoint(x1, yb1),
                    ])
                    painter.drawPolygon(poly)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

            # Baseline line (red)
            pen_a = QPen(QColor("#dc2626"))
            pen_a.setWidth(2)
            painter.setPen(pen_a)
            for i in range(n - 1):
                painter.drawLine(
                    int(x_pos(i)), int(y_pos(series_a[i])),
                    int(x_pos(i + 1)), int(y_pos(series_a[i + 1])),
                )

            # SMPC optimised line (green)
            pen_b = QPen(QColor("#16a34a"))
            pen_b.setWidth(2)
            painter.setPen(pen_b)
            for i in range(n - 1):
                painter.drawLine(
                    int(x_pos(i)), int(y_pos(series_b[i])),
                    int(x_pos(i + 1)), int(y_pos(series_b[i + 1])),
                )

            # Axis labels
            text_pen = QPen(QColor("#334155"))
            painter.setPen(text_pen)
            painter.drawText(4, pad_top + 4, f"{max_val:.4f} \u20ac")
            painter.drawText(4, height - pad_bottom + 4, f"{min_val:.4f} \u20ac")

            # X-axis time labels (every ~4 hours for 96 slots)
            step = max(1, n // 6)
            for i in range(0, n, step):
                x = int(x_pos(i))
                painter.drawText(x - 14, height - 8, x_labels[i])

            # Legend
            legend_y = pad_top - 6
            painter.setPen(pen_a)
            painter.drawLine(pad_left + 10, legend_y, pad_left + 30, legend_y)
            painter.setPen(text_pen)
            painter.drawText(pad_left + 34, legend_y + 4, "Baseline")
            painter.setPen(pen_b)
            painter.drawLine(pad_left + 120, legend_y, pad_left + 140, legend_y)
            painter.setPen(text_pen)
            painter.drawText(pad_left + 144, legend_y + 4, "SMPC Optimised")

            painter.end()
            return pixmap

    app = QApplication(sys.argv)
    window = ScadaWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    dashboard_config = load_dashboard_config()
    launch_scada_hmi(
        dashboard_config.get("inputs", []),
        dashboard_config.get("outputs", []),
    )