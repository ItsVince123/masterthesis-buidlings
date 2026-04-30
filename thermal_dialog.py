"""
Building Thermal Settings dialog.

Allows the user to edit building thermal model parameters
(setpoint, deadband, cooldown rate, thermal mass) and persist
them to dashboard_config.json.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)

from dashboard_config import load_dashboard_config, save_dashboard_config

logger = logging.getLogger(__name__)

_DIALOG_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Segoe UI'; font-size: 10pt; }
QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #d6dfeb;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    min-height: 28px;
}
"""

_SAVE_BUTTON_STYLE = """
QPushButton {
    background-color: #7c3aed;
    color: white;
    border: 1px solid #6d28d9;
    border-radius: 6px;
    padding: 8px 24px;
    font-size: 10pt;
    font-weight: 700;
}
QPushButton:hover { background-color: #6d28d9; }
"""


class BuildingThermalDialog(QDialog):
    """Dialog for editing building thermal model parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building Thermal Settings")
        self.setMinimumWidth(400)
        self.setStyleSheet(_DIALOG_STYLE)

        self._load_current()
        self._build_ui()

    # ------------------------------------------------------------------
    def _load_current(self):
        cfg = load_dashboard_config()
        thermal = cfg.get("smpc", {}).get("building", {}).get("thermal", {})
        self._setpoint = thermal.get("setpoint_c", 21.0)
        self._deadband = thermal.get("deadband_c", 1.0)
        self._initial_temp = thermal.get("initial_temp_c", 21.0)
        self._cooldown_rate = thermal.get("cooldown_rate_c_per_hour", 0.5)
        self._thermal_mass = thermal.get("thermal_mass_kwh_per_c", 500.0)

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("\U0001f321\ufe0f  Building Thermal Model")
        title.setStyleSheet("font-size: 14pt; font-weight: 700; color: #7c3aed;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(8)

        self._sp_setpoint = self._spin(10.0, 35.0, self._setpoint, " °C", 1)
        form.addRow("Setpoint temperature:", self._sp_setpoint)

        self._sp_deadband = self._spin(0.1, 5.0, self._deadband, " °C", 1)
        form.addRow("Deadband:", self._sp_deadband)

        self._sp_initial = self._spin(5.0, 35.0, self._initial_temp, " °C", 1)
        form.addRow("Initial temperature:", self._sp_initial)

        self._sp_cooldown = self._spin(0.01, 5.0, self._cooldown_rate, " °C/h", 2)
        form.addRow("Cooldown rate:", self._sp_cooldown)

        self._sp_thermal_mass = self._spin(10.0, 50000.0, self._thermal_mass, " kWh/°C", 0)
        form.addRow("Thermal mass:", self._sp_thermal_mass)

        layout.addLayout(form)

        # --- hint ---
        hint = QLabel(
            "Cooldown rate = how fast the building loses temperature "
            "per hour without heating.\n"
            "Thermal mass = energy needed to change building temp by 1 °C."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6b7280; font-size: 9pt;")
        layout.addWidget(hint)

        # --- buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(_SAVE_BUTTON_STYLE)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { padding: 8px 24px; font-size: 10pt; border-radius: 6px; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _spin(self, lo, hi, val, suffix, decimals):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setSuffix(suffix)
        sb.setDecimals(decimals)
        return sb

    # ------------------------------------------------------------------
    def _on_save(self):
        cfg = load_dashboard_config()
        smpc = cfg.setdefault("smpc", {})
        building = smpc.setdefault("building", {})
        thermal = building.setdefault("thermal", {})

        thermal["setpoint_c"] = self._sp_setpoint.value()
        thermal["deadband_c"] = self._sp_deadband.value()
        thermal["initial_temp_c"] = self._sp_initial.value()
        thermal["cooldown_rate_c_per_hour"] = self._sp_cooldown.value()
        thermal["thermal_mass_kwh_per_c"] = self._sp_thermal_mass.value()

        save_dashboard_config(cfg)
        logger.info("Saved building thermal config: %s", thermal)
        self.accept()
