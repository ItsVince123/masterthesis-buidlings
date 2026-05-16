"""
MPC Settings Dialog.

Allows editing all MPCConfig parameters (efficiencies, capacities,
COP model, weights) and persists them to the ``'mpc'`` block in
dashboard_config.json.

Tabs
----
* Grid & Horizon  — Pgrid_max, distribution fee, horizon steps / dt
* Heat Pump       — Php_max, COP0, T0, cop_alpha, COP_min
* Gas Boiler      — Pgas_max, eta_boiler, gas price, HV
* CHP             — enable, Fchp_max, eta_elec/heat, startup cost, gas price
* Battery         — enable, SOC capacity/limits, charge/discharge rates, efficiencies
* Hot Water Tank  — enable, volume, temperatures, heater power, heat loss
* Building        — Tset, Tmin/Tmax, T_init, Cth, UA
* Flexible Load   — enable, Pflex_max, daily energy
* Weights         — w_peak, w_comfort, w_tank
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QTime
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QTabWidget, QTimeEdit, QVBoxLayout, QWidget,
)

from dashboard_config import load_dashboard_config, save_dashboard_config
from mpc_lp import MPCConfig

logger = logging.getLogger(__name__)

# ── Styles ─────────────────────────────────────────────────────────────────

_DIALOG_STYLE = """
QDialog { background-color: #eef3f9; }
QTabWidget::pane { border: 1px solid #d6dfeb; border-radius: 6px; }
QTabBar::tab {
    background: #d6dfeb; color: #1f2937; padding: 6px 14px;
    border-radius: 4px 4px 0 0; font-size: 9pt;
}
QTabBar::tab:selected { background: #7c3aed; color: white; font-weight: 700; }
QLabel { color: #1f2937; font-family: 'Calibri'; font-size: 10pt; }
QDoubleSpinBox, QSpinBox {
    background-color: #ffffff; border: 1px solid #d6dfeb;
    border-radius: 6px; padding: 6px 10px; font-size: 10pt; min-height: 28px;
}
QCheckBox { font-size: 10pt; color: #1f2937; }
"""

_SAVE_STYLE = """
QPushButton {
    background-color: #7c3aed; color: white; border: 1px solid #6d28d9;
    border-radius: 6px; padding: 8px 24px; font-size: 10pt; font-weight: 700;
}
QPushButton:hover { background-color: #6d28d9; }
"""

_RESET_STYLE = """
QPushButton {
    background-color: #64748b; color: white; border: none;
    border-radius: 6px; padding: 8px 18px; font-size: 10pt;
}
QPushButton:hover { background-color: #475569; }
"""


# ── Helper widget builders ───────────────────────────────────────────────────

def _dspin(lo: float, hi: float, val: float, suffix: str = "",
           decimals: int = 3, step: float = 0.01) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(val)
    if suffix:
        sb.setSuffix(f" {suffix}")
    return sb


def _ispin(lo: int, hi: int, val: int, suffix: str = "") -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    if suffix:
        sb.setSuffix(f" {suffix}")
    return sb


def _section(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size: 11pt; font-weight: 700; color: #7c3aed; margin-top: 8px;"
    )
    return lbl


def _tab_scroll(widget: QWidget) -> QWidget:
    """Wrap a form widget in a scroll area for tall tabs."""
    from PyQt6.QtWidgets import QScrollArea
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    outer = QWidget()
    lay = QVBoxLayout(outer)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(scroll)
    return outer


# ===========================================================================
# MPC Settings dialog
# ===========================================================================

class MpcConfigDialog(QDialog):
    """Tabbed dialog for editing all MPC LP parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙  MPC Settings")
        self.setMinimumSize(520, 600)
        self.setStyleSheet(_DIALOG_STYLE)

        self._cfg = self._load_cfg()
        self._build_ui()

    # ── Load / save ─────────────────────────────────────────────────────────

    def _load_cfg(self) -> MPCConfig:
        try:
            raw = load_dashboard_config()
            return MPCConfig.from_dict(raw.get("mpc", {}))
        except Exception:
            return MPCConfig()

    def _save(self):
        """Update only the fields owned by this dialog; leave asset params untouched."""
        try:
            raw = load_dashboard_config()
            mpc = raw.get("mpc", {})
            mpc.setdefault("horizon", {}).update({
                "steps":    self._steps.value(),
                "dt_hours": self._dt.value(),
            })
            mpc.setdefault("grid", {}).update({
                "Pgrid_max_kw":       self._Pgrid_max.value(),
                "fee_eur_kwh":        self._fee.value(),
                "cap_tariff_Plim_kw":   self._cap_Plim.value(),
                "cap_tariff_epsilon_l": self._cap_eps.value(),
            })
            mpc.setdefault("building", {}).update({
                "Tset_c":        self._Tset.value(),
                "Tmin_c":        self._Tmin_b.value(),
                "Tmax_c":        self._Tmax_b.value(),
                "T_init_c":      self._Tinit_b.value(),
                "Cth_kwh_per_c": self._Cth.value(),
                "UA_kw_per_c":   self._UA.value(),
                "use_night_setback": self._night_chk.isChecked(),
                "T_set_night_c":     self._T_set_night.value(),
                "T_cool_night_c":    self._T_cool_night.value(),
                "night_start_h":     self._night_start.value(),
                "night_end_h":       self._night_end.value(),
            })
            # base/peak load lives in smpc.building, not mpc.building
            raw.setdefault("smpc", {}).setdefault("building", {}).update({
                "base_load_kw": self._base_load.value(),
                "peak_load_kw": self._peak_load.value(),
            })
            raw["mpc"] = mpc
            save_dashboard_config(raw)
            try:
                from smpc_calculator import SMPCCalculator
                SMPCCalculator._reload_mpc_config()
            except Exception:
                pass
            self.accept()
        except Exception as exc:
            logger.error("Could not save MPC config: %s", exc)

    def _reset(self):
        """Restore the visible fields to MPCConfig defaults."""
        d = MPCConfig()
        self._steps.setValue(d.horizon_steps)
        self._dt.setValue(d.dt_hours)
        self._Pgrid_max.setValue(d.Pgrid_max_kw)
        self._fee.setValue(d.fee_eur_kwh)
        self._cap_Plim.setValue(d.cap_tariff_Plim_kw)
        self._cap_eps.setValue(d.cap_tariff_epsilon_l)
        # reset base/peak load to smpc defaults
        self._base_load.setValue(80.0)
        self._peak_load.setValue(200.0)
        self._Tset.setValue(d.Tset_c)
        self._Tmin_b.setValue(d.Tmin_c)
        self._Tmax_b.setValue(d.Tmax_c)
        self._Tinit_b.setValue(d.T_init_c)
        self._Cth.setValue(d.Cth_kwh_per_c)
        self._UA.setValue(d.UA_kw_per_c)
        self._night_chk.setChecked(d.use_night_setback)
        self._T_set_night.setValue(d.T_set_night_c)
        self._T_cool_night.setValue(d.T_cool_night_c)
        self._night_start.setValue(d.night_start_h)
        self._night_end.setValue(d.night_end_h)

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        title = QLabel("\u2699\ufe0f  Building & Solver Settings")
        title.setStyleSheet("font-size: 14pt; font-weight: 700; color: #7c3aed;")
        root.addWidget(title)

        sub = QLabel(
            "Asset-specific parameters (heat pump COP, battery capacity, "
            "flex schedule \u2026) are configured per-asset in the Asset Manager."
        )
        sub.setStyleSheet("color: #64748b; font-size: 9pt;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        self._build_tab_horizon()
        self._build_tab_building()

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet(_SAVE_STYLE)
        save_btn.clicked.connect(self._save)
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.setStyleSheet(_RESET_STYLE)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

    # ── Individual tabs ──────────────────────────────────────────────────────

    def _build_tab_horizon(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)
        form.addRow(_section("Optimisation Horizon"))
        self._steps  = _ispin(4, 96, self._cfg.horizon_steps, "steps")
        self._dt     = _dspin(0.25, 4.0, self._cfg.dt_hours, "h", 2, 0.25)
        form.addRow("Horizon steps:", self._steps)
        form.addRow("Step duration dt:", self._dt)

        form.addRow(_section("Grid Connection"))
        self._Pgrid_max  = _dspin(10.0, 50000.0, self._cfg.Pgrid_max_kw,  "kW",    1, 10.0)
        self._fee        = _dspin(0.0,  2.0,     self._cfg.fee_eur_kwh,   "€/kWh", 4, 0.005)
        form.addRow("Grid peak limit (hard):",  self._Pgrid_max)
        form.addRow("Fixed electricity cost:",  self._fee)

        form.addRow(_section("Capacity Tariff"))
        self._cap_Plim = _dspin(0.0, 50000.0, self._cfg.cap_tariff_Plim_kw,   "kW",    1, 10.0)
        self._cap_eps  = _dspin(0.0, 1e6,     self._cfg.cap_tariff_epsilon_l, "€/kW²",  4, 0.001)
        _cap_note = QLabel(
            "Quadratic peak penalty: ε_L \u00b7 Σ(max(0, P_grid,k \u2212 P_lim))². "
            "Discourages grid import above P_lim without a hard cut-off. "
            "Set P_lim to 0 to disable."
        )
        _cap_note.setStyleSheet("color: #64748b; font-size: 9pt;")
        _cap_note.setWordWrap(True)
        form.addRow("Power limit P_lim:",    self._cap_Plim)
        form.addRow("Penalty ε_L:",         self._cap_eps)
        form.addRow("",                       _cap_note)
        self._tabs.addTab(w, "Grid / Horizon")

    def _build_tab_building(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(10)

        form.addRow(_section("Electrical Load Profile"))
        raw = load_dashboard_config()
        _smpc_bld = raw.get("smpc", {}).get("building", {})
        self._base_load = _dspin(0.0, 100000.0, _smpc_bld.get("base_load_kw", 80.0), "kW", 1, 10.0)
        self._peak_load = _dspin(0.0, 100000.0, _smpc_bld.get("peak_load_kw", 200.0), "kW", 1, 10.0)
        form.addRow("Night load (22:00–07:00):", self._base_load)
        form.addRow("Day load   (07:00–22:00):", self._peak_load)

        form.addRow(_section("Building Thermal Model"))
        self._Tset    = _dspin(10.0, 35.0,  self._cfg.Tset_c,        "°C",        1, 0.5)
        self._Tmin_b  = _dspin(5.0,  30.0,  self._cfg.Tmin_c,        "°C",        1, 0.5)
        self._Tmax_b  = _dspin(15.0, 40.0,  self._cfg.Tmax_c,        "°C",        1, 0.5)
        self._Tinit_b = _dspin(5.0,  35.0,  self._cfg.T_init_c,      "°C",        1, 0.5)
        self._Cth     = _dspin(1.0,  1e6,   self._cfg.Cth_kwh_per_c, "kWh/°C",    1, 100.0)
        self._UA      = _dspin(0.01, 500.0, self._cfg.UA_kw_per_c,   "kW/°C",     3, 0.1)
        form.addRow("Setpoint temperature:", self._Tset)
        form.addRow("Min comfort temperature:", self._Tmin_b)
        form.addRow("Max comfort temperature:", self._Tmax_b)
        form.addRow("Initial temperature:", self._Tinit_b)
        form.addRow("Thermal mass Cth:", self._Cth)
        form.addRow("Heat transfer coefficient UA:", self._UA)

        form.addRow(_section("Night Setback"))
        self._night_chk = QCheckBox("Enable night setback schedule")
        self._night_chk.setChecked(self._cfg.use_night_setback)
        form.addRow("", self._night_chk)
        self._T_set_night  = _dspin(5.0,  30.0, self._cfg.T_set_night_c,  "\u00b0C", 1, 0.5)
        self._T_cool_night = _dspin(15.0, 40.0, self._cfg.T_cool_night_c, "\u00b0C", 1, 0.5)
        self._night_start  = _dspin(0.0,  23.0, self._cfg.night_start_h,  "h",       1, 0.5)
        self._night_end    = _dspin(0.0,  23.0, self._cfg.night_end_h,    "h",       1, 0.5)
        form.addRow("Night heating floor:",   self._T_set_night)
        form.addRow("Night cooling ceiling:", self._T_cool_night)
        form.addRow("Night starts at hour:",  self._night_start)
        form.addRow("Night ends at hour:",    self._night_end)

        self._tabs.addTab(_tab_scroll(w), "Building")


# ===========================================================================
# Asset Instance Dialog  (per-instance settings incl. type-specific params)
# ===========================================================================

_TYPE_ICONS = {
    "pv":         "\u2600\ufe0f  PV / Solar",
    "heat_pump":  "\U0001f321\ufe0f  Heat Pump",
    "gas_boiler": "\U0001f525  Gas Boiler",
    "chp":        "\u26a1  CHP / Cogen",
    "battery":    "\U0001f50b  Battery",
    "flex":       "\u21c6  Flexible Load",
    "hot_water":  "\U0001f6bf  Hot Water Tank",
}

_INSTANCE_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Calibri'; font-size: 10pt; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTimeEdit {
    background: white; border: 1px solid #cbd5e1; border-radius: 5px;
    padding: 4px 8px; font-size: 10pt; color: #1f2937;
}
QCheckBox { font-size: 10pt; color: #1f2937; spacing: 8px; padding: 4px; }
QCheckBox::indicator {
    width: 18px; height: 18px; border-radius: 4px;
    border: 2px solid #94a3b8; background: white;
}
QCheckBox::indicator:checked { background: #1e40af; border-color: #1e40af; }
QPushButton {
    background: #1e40af; color: white; border: none;
    border-radius: 6px; padding: 8px 20px; font-size: 10pt; font-weight: 700;
}
QPushButton:hover { background: #1d4ed8; }
QPushButton#cancel { background: #64748b; }
QPushButton#cancel:hover { background: #475569; }
"""

# Per-type parameter specs: (json_key, label, lo, hi, default, suffix, decimals, step)
_TYPE_PARAM_SPECS: dict[str, list] = {
    "pv": [
        ("capacity_kwp", "Installed capacity", 0.0, 9_999_999.0, 100.0, "kWp", 1, 10.0),
    ],
    "heat_pump": [
        ("Php_max_kw",      "Max thermal output",        1.0, 99_999.0,  50.0, "kW",     1,  5.0),
        ("COP0",            "Nominal COP (heating)",      1.0,     10.0,   4.0, "",       2,  0.1),
        ("T0_c",            "Reference temperature T0", -20.0,    30.0,   7.0, "\u00b0C", 1,  1.0),
        ("cop_alpha",       "COP degradation \u03b1",    0.0,      0.1,  0.02, "/\u00b0C", 4, 0.001),
        ("COP_min",         "Minimum COP",                0.5,      5.0,   1.0, "",       2,  0.1),
        ("ramp_pct",        "Ramp up/down limit",         1.0,    100.0, 100.0, "%/step", 1,  5.0),
        ("Php_cool_max_kw", "Max cooling output",         0.0, 99_999.0,  50.0, "kW",     1,  5.0),
        ("COP_cool",        "Cooling COP",                0.5,     10.0,   3.0, "",       2,  0.1),
    ],
    "gas_boiler": [
        ("Pgas_max_kw",      "Max thermal output",  1.0, 9_999_999.0, 100.0, "kW",      1, 5.0),
        ("eta_boiler",       "Thermal efficiency",  0.5,     1.0,  0.92, "",         3, 0.01),
        ("gas_price_eur_kwh", "Gas price",           0.0,     1.0,  0.035, "\u20ac/kWh", 4, 0.001),
        ("ramp_pct",         "Ramp up/down limit",  1.0,   100.0, 100.0, "%/step",   1, 5.0),
    ],
    "chp": [
        ("Pchp_max_kw",      "Max elec. output",          1.0, 9_999_999.0, 39.2, "kW",           1, 5.0),
        ("eta_elec",         "Electrical efficiency",      0.1,     0.9,  0.40, "",             3, 0.01),
        ("eta_heat",         "Thermal efficiency",         0.1,     0.9,  0.45, "",             3, 0.01),
        ("startup_cost_eur", "Startup cost per event",     0.0, 1_000.0,   5.0, "\u20ac",       2, 0.5),
        ("gas_price_eur_m3", "Gas price",                  0.0,     5.0,  0.35, "\u20ac/m\u00b3", 4, 0.01),
        ("ramp_pct",         "Ramp up/down limit",         1.0,   100.0, 100.0, "%/step",       1, 5.0),
    ],
    "battery": [
        ("SOC_cap_kwh",  "Capacity",               1.0, 100_000.0, 100.0, "kWh", 1, 10.0),
        ("SOC_min_kwh",  "Minimum SOC (reserve)",  0.0,  10_000.0,  10.0, "kWh", 1,  5.0),
        ("SOC_init_kwh", "Initial SOC",            0.0, 100_000.0,  50.0, "kWh", 1, 10.0),
        ("SOC_end_kwh",  "End-of-day SOC target",  0.0, 100_000.0,  50.0, "kWh", 1, 10.0),
        ("Pch_max_kw",   "Max charge power",       0.1,  10_000.0,  25.0, "kW",  1,  5.0),
        ("Pdis_max_kw",  "Max discharge power",    0.1,  10_000.0,  25.0, "kW",  1,  5.0),
        ("eta_ch",       "Charge efficiency",      0.5,       1.0,  0.95, "",    3, 0.01),
        ("eta_dis",      "Discharge efficiency",   0.5,       1.0,  0.95, "",    3, 0.01),
        ("ramp_pct",     "Ramp up/down limit",     1.0,     100.0, 100.0, "%/step", 1, 5.0),
    ],
    "hot_water": [
        ("Ptank_max_kw", "Heater power",         0.1, 9_999_999.0,   3.0, "kW",  2, 0.5),
        ("volume_l",     "Tank volume",          10.0, 9_999_999.0, 200.0, "L",   1, 10.0),
        ("T_min_c",      "Min temperature",      10.0,    80.0,  45.0, "\u00b0C", 1, 1.0),
        ("T_max_c",      "Max temperature",      10.0,    95.0,  60.0, "\u00b0C", 1, 1.0),
        ("T_init_c",     "Initial temperature",  10.0,    95.0,  55.0, "\u00b0C", 1, 1.0),
        ("heat_loss_w",  "Standby heat loss",     0.0,   500.0,  50.0, "W",   1, 5.0),
        ("ramp_pct",     "Ramp up/down limit",    1.0,   100.0, 100.0, "%/step", 1, 5.0),
    ],
}

# Per-type baseline mode options shown in the combo box.
# Keys match the "type" field stored in asset instances.
_BASELINE_MODES_BY_TYPE: dict[str, list[str]] = {
    "pv":         ["always_off"],
    "heat_pump":  ["on_off", "constant", "always_off"],
    "gas_boiler": ["on_off", "constant", "always_off"],
    "battery":    ["always_off"],
    "chp":        ["always_off", "heat_demand", "constant"],
    "hot_water":  ["on_off", "constant", "always_off"],
    "flex":       ["fixed_window", "always_off"],
}

# Human-readable labels for raw mode keys (used in the combo box).
_BL_MODE_DISPLAY: dict[str, str] = {
    "on_off":       "On/off (bang-bang thermostat)",
    "constant":     "Constant power",
    "always_off":   "Always off (disabled)",
    "heat_demand":  "Heat-demand led",
    "fixed_window": "Fixed time window",
}

# Asset types whose baseline behaviour is fixed — no user configuration needed.
_BL_FIXED_ASSET_NOTES: dict[str, str] = {
    "pv":      "PV always uses all available generation in the baseline.",
    "battery": "Battery is always idle in the baseline (no price arbitrage).",
}


class AssetInstanceDialog(QDialog):
    """Edit a single asset instance — common settings + type-specific parameters."""

    def __init__(self, instance: dict, parent=None):
        super().__init__(parent)
        self._inst = dict(instance)
        self.setWindowTitle("Edit Asset")
        self.setMinimumWidth(420)
        self.setStyleSheet(_INSTANCE_STYLE)
        self._param_widgets: dict[str, QDoubleSpinBox | QTimeEdit] = {}
        self._bool_widgets:  dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setSpacing(10)
        form.setContentsMargins(20, 20, 20, 10)

        itype = self._inst.get("type", "")
        type_label = _TYPE_ICONS.get(itype, itype)
        form.addRow(_section(type_label))

        # ── Common fields ────────────────────────────────────────────────────
        self._name_edit = QLineEdit(self._inst.get("name", ""))
        form.addRow("Name:", self._name_edit)

        self._enabled_chk = QCheckBox("Active in optimisation")
        self._enabled_chk.setChecked(self._inst.get("enabled", True))
        form.addRow("", self._enabled_chk)

        # ── Baseline comparison ──────────────────────────────────────────────
        div1 = QFrame(); div1.setFrameShape(QFrame.Shape.HLine)
        form.addRow(div1)
        form.addRow(_section("Baseline comparison"))

        if itype in _BL_FIXED_ASSET_NOTES:
            # PV / battery: baseline behaviour is fixed, nothing to configure.
            _note = QLabel(_BL_FIXED_ASSET_NOTES[itype])
            _note.setStyleSheet("color: #64748b; font-size: 9pt;")
            _note.setWordWrap(True)
            form.addRow(_note)
            self._baseline_combo = None
            self._baseline_power = None
            self._bl_t_on  = None
            self._bl_t_off = None
        else:
            hint = QLabel(
                "The baseline models what the system would consume WITHOUT optimisation.\n"
                "It is used to calculate the reported energy savings."
            )
            hint.setStyleSheet("color: #64748b; font-size: 9pt;")
            hint.setWordWrap(True)
            form.addRow(hint)

            # Mode combo — items store raw key as userData, show friendly display label.
            self._baseline_combo = QComboBox()
            _bl_modes = _BASELINE_MODES_BY_TYPE.get(itype, [])
            for m in _bl_modes:
                self._baseline_combo.addItem(_BL_MODE_DISPLAY.get(m, m), userData=m)
            cur = self._inst.get("baseline_mode", _bl_modes[0] if _bl_modes else "")
            idx = next(
                (i for i in range(self._baseline_combo.count())
                 if self._baseline_combo.itemData(i) == cur),
                0,
            )
            self._baseline_combo.setCurrentIndex(idx)
            form.addRow("Baseline mode:", self._baseline_combo)

            # Power spinbox — only meaningful for assets with a configurable run power.
            # CHP always fires at its rated gas flow; no separate power setting needed.
            if itype in {"heat_pump", "gas_boiler", "hot_water", "flex"}:
                self._baseline_power = QDoubleSpinBox()
                self._baseline_power.setRange(0.0, 10_000.0)
                self._baseline_power.setSuffix(" kW")
                self._baseline_power.setDecimals(1)
                self._baseline_power.setValue(float(self._inst.get("baseline_power_kw", 0.0)))
                form.addRow("Baseline power (0 = use max):", self._baseline_power)
            else:
                self._baseline_power = None
                if itype == "chp":
                    _chp_note = QLabel("When active, CHP always runs at its rated output.")
                    _chp_note.setStyleSheet("color: #64748b; font-size: 9pt;")
                    form.addRow(_chp_note)

            # Flex-specific baseline time window ─────────────────────────────
            if itype == "flex":
                hint_bl = QLabel(
                    "Baseline window: the hours during which the load runs without"
                    " optimisation (independent from the MPC active window above)."
                )
                hint_bl.setStyleSheet("color: #64748b; font-size: 9pt;")
                hint_bl.setWordWrap(True)
                form.addRow(hint_bl)

                def _te_bl(s: str) -> QTimeEdit:
                    t = QTimeEdit()
                    t.setDisplayFormat("HH:mm")
                    try:
                        h2, m2 = map(int, s.split(":"))
                        t.setTime(QTime(h2, m2))
                    except Exception:
                        t.setTime(QTime(0, 0))
                    return t

                self._bl_t_on  = _te_bl(self._inst.get("baseline_time_start", "00:00"))
                self._bl_t_off = _te_bl(self._inst.get("baseline_time_end",   "23:59"))
                form.addRow("Baseline window start:", self._bl_t_on)
                form.addRow("Baseline window end:",   self._bl_t_off)
            else:
                self._bl_t_on  = None
                self._bl_t_off = None

        # ── Type-specific parameters ─────────────────────────────────────────
        if itype in _TYPE_PARAM_SPECS or itype == "flex":
            div2 = QFrame(); div2.setFrameShape(QFrame.Shape.HLine)
            form.addRow(div2)
            form.addRow(_section("Asset parameters"))

        if itype in _TYPE_PARAM_SPECS:
            for key, label, lo, hi, default, suffix, decimals, step in _TYPE_PARAM_SPECS[itype]:
                val = float(self._inst.get(key, default))
                w = _dspin(lo, hi, val, suffix, decimals, step)
                form.addRow(f"{label}:", w)
                self._param_widgets[key] = w

            if itype == "heat_pump":
                div_cool = QFrame(); div_cool.setFrameShape(QFrame.Shape.HLine)
                form.addRow(div_cool)
                form.addRow(_section("Cooling"))
                cooling_chk = QCheckBox("Enable HP cooling (reverse-cycle)")
                cooling_chk.setChecked(bool(self._inst.get("cooling_enabled", False)))
                form.addRow("", cooling_chk)
                self._bool_widgets["cooling_enabled"] = cooling_chk


        elif itype == "flex":
            # Flex power and energy budget
            val_max = float(self._inst.get("Pflex_max_kw", 50.0))
            self._flex_max = _dspin(0.1, 5_000.0, val_max, "kW", 1, 5.0)
            form.addRow("Max power:", self._flex_max)

            val_kwh = float(self._inst.get("daily_energy_kwh", 400.0))
            self._flex_kwh = _dspin(0.0, 1e6, val_kwh, "kWh", 1, 50.0)
            form.addRow("Daily energy target:", self._flex_kwh)

            # Time window
            form.addRow(_section("Active time window"))
            hint2 = QLabel(
                "The flex load may only run between these hours.\n"
                "Set 00:00 \u2013 23:59 to allow any time."
            )
            hint2.setStyleSheet("color: #64748b; font-size: 9pt;")
            hint2.setWordWrap(True)
            form.addRow(hint2)

            def _te(s: str) -> QTimeEdit:
                t = QTimeEdit()
                t.setDisplayFormat("HH:mm")
                try:
                    h, m = map(int, s.split(":"))
                    t.setTime(QTime(h, m))
                except Exception:
                    t.setTime(QTime(0, 0))
                return t

            self._flex_t_on  = _te(self._inst.get("time_start", "00:00"))
            self._flex_t_off = _te(self._inst.get("time_end",   "23:59"))
            form.addRow("Window start:", self._flex_t_on)
            form.addRow("Window end:",   self._flex_t_off)

            # Ramp rates
            form.addRow(_section("Ramp-rate limits"))
            hint3 = QLabel(
                "Maximum power change between consecutive MPC steps.\n"
                "Leave at 9999 to disable (unconstrained)."
            )
            hint3.setStyleSheet("color: #64748b; font-size: 9pt;")
            hint3.setWordWrap(True)
            form.addRow(hint3)

            val_up   = float(self._inst.get("ramp_up_kw",   9999.0))
            val_down = float(self._inst.get("ramp_down_kw", 9999.0))
            self._ramp_up   = _dspin(0.1, 9999.0, val_up,   "kW/step", 1, 5.0)
            self._ramp_down = _dspin(0.1, 9999.0, val_down, "kW/step", 1, 5.0)
            form.addRow("Max ramp up:",   self._ramp_up)
            form.addRow("Max ramp down:", self._ramp_down)

        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(20, 8, 20, 16)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("\u2713  Save")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        outer.addLayout(btn_row)

    def _save(self):
        self._inst["name"]    = self._name_edit.text().strip() or self._inst.get("name", "Asset")
        self._inst["enabled"] = self._enabled_chk.isChecked()
        if self._baseline_combo is not None:
            self._inst["baseline_mode"] = self._baseline_combo.currentData()
        if self._baseline_power is not None:
            self._inst["baseline_power_kw"] = self._baseline_power.value()
        # Generic spinbox params
        for key, widget in self._param_widgets.items():
            self._inst[key] = widget.value()
        # Boolean params (checkboxes)
        for key, widget in self._bool_widgets.items():
            self._inst[key] = widget.isChecked()
        # Flex-specific
        itype = self._inst.get("type", "")
        if itype == "flex":
            self._inst["Pflex_max_kw"]          = self._flex_max.value()
            self._inst["daily_energy_kwh"]       = self._flex_kwh.value()
            self._inst["time_start"]             = self._flex_t_on.time().toString("HH:mm")
            self._inst["time_end"]               = self._flex_t_off.time().toString("HH:mm")
            self._inst["ramp_up_kw"]             = self._ramp_up.value()
            self._inst["ramp_down_kw"]           = self._ramp_down.value()
            if self._bl_t_on is not None:
                self._inst["baseline_time_start"] = self._bl_t_on.time().toString("HH:mm")
            if self._bl_t_off is not None:
                self._inst["baseline_time_end"]   = self._bl_t_off.time().toString("HH:mm")
        self.accept()

    def result_instance(self) -> dict:
        return self._inst


# ===========================================================================
# Asset Selector Dialog  (multi-instance list manager)
# ===========================================================================

_SELECTOR_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Calibri'; font-size: 10pt; }
QScrollArea { background: transparent; border: none; }
QFrame#InstanceRow {
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    margin: 2px 0;
}
"""

_ASSET_BTN_STYLE = """
QPushButton {
    background-color: #1e40af; color: white; border: none;
    border-radius: 6px; padding: 8px 20px; font-size: 10pt; font-weight: 700;
}
QPushButton:hover { background-color: #1d4ed8; }
"""

_CONFIG_BTN_STYLE = """
QPushButton {
    background-color: #64748b; color: white; border: none;
    border-radius: 6px; padding: 8px 16px; font-size: 10pt;
}
QPushButton:hover { background-color: #475569; }
"""

_SMALL_BTN = (
    "QPushButton { background: %s; color: white; border: none; "
    "border-radius: 4px; padding: 3px 10px; font-size: 9pt; } "
    "QPushButton:hover { background: %s; }"
)
_EDIT_BTN_STYLE = _SMALL_BTN % ("#2563eb", "#1d4ed8")
_DEL_BTN_STYLE  = _SMALL_BTN % ("#dc2626", "#b91c1c")
_ADD_BTN_STYLE  = _SMALL_BTN % ("#059669", "#047857")


# Ordered list of types for the Add menu
_ADD_TYPES = [
    ("pv",         "\u2600\ufe0f  PV / Solar panels"),
    ("heat_pump",  "\U0001f321\ufe0f  Heat Pump"),
    ("gas_boiler", "\U0001f525  Gas Boiler"),
    ("chp",        "\u26a1  CHP / Cogeneration"),
    ("battery",    "\U0001f50b  Battery Storage"),
    ("flex",       "\u21c6  Flexible Load"),
    ("hot_water",  "\U0001f6bf  Hot Water Tank"),
]

# Map asset type to the mpc_cfg enabled flag so the solver stays in sync
_TYPE_TO_CFG_FLAG = {
    "pv":         "pv",
    "heat_pump":  "heat_pump",
    "gas_boiler": "gas_boiler",
    "chp":        "chp",
    "battery":    "battery",
    "flex":       "flexible_load",
    "hot_water":  "hot_water_tank",
}


class MPCAssetSelectorDialog(QDialog):
    """
    Multi-instance asset manager.

    Each row represents one asset instance that will appear as its own
    row in the input / output panels.  Multiple instances of the same
    type are allowed (they share solver data but show separately).
    Each instance also stores baseline comparison settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\u2295  Manage Assets")
        self.setMinimumWidth(520)
        self.setMinimumHeight(480)
        self.setStyleSheet(_SELECTOR_STYLE)
        self._instances: list[dict] = self._load_instances()
        self._build_ui()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _load_instances(self) -> list[dict]:
        try:
            raw = load_dashboard_config()
            return list(raw.get("mpc", {}).get("asset_instances", []))
        except Exception:
            return []

    def _unique_id(self, type_key: str) -> str:
        existing = {inst["id"] for inst in self._instances}
        n = 1
        while f"{type_key}_{n}" in existing:
            n += 1
        return f"{type_key}_{n}"

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(20, 20, 20, 20)

        title = QLabel("\u2295  Asset Instances")
        title.setStyleSheet("font-size: 13pt; font-weight: 700; color: #1e40af;")
        outer.addWidget(title)

        sub = QLabel(
            "Each instance appears as its own row in the panels. "
            "Multiple instances of the same type are allowed.\n"
            "Use 'Edit' to change name, enable/disable, and set baseline rules."
        )
        sub.setStyleSheet("color: #64748b; font-size: 9pt;")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color: #d6dfeb;")
        outer.addWidget(div)

        # Scrollable list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_lay = QVBoxLayout(self._list_widget)
        self._list_lay.setSpacing(6)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.addStretch()
        self._scroll.setWidget(self._list_widget)
        outer.addWidget(self._scroll, stretch=1)

        # Add / bottom buttons
        add_row = QHBoxLayout()
        add_btn = QPushButton("\u2795  Add Asset")
        add_btn.setStyleSheet(_ADD_BTN_STYLE)
        add_btn.clicked.connect(self._add_asset)
        add_row.addWidget(add_btn)
        add_row.addStretch()
        outer.addLayout(add_row)

        outer.addWidget(div)  # second divider

        btn_row = QHBoxLayout()
        cfg_btn = QPushButton("\u2699  Solver Config")
        cfg_btn.setStyleSheet(_CONFIG_BTN_STYLE)
        cfg_btn.clicked.connect(self._open_full_settings)
        btn_row.addWidget(cfg_btn)
        btn_row.addStretch()
        save_btn = QPushButton("\u2713  Save")
        save_btn.setStyleSheet(_ASSET_BTN_STYLE)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        outer.addLayout(btn_row)

        self._refresh_list()

    def _refresh_list(self):
        """Rebuild the scrollable list from self._instances."""
        # Remove all existing rows (keep the trailing stretch)
        while self._list_lay.count() > 1:
            item = self._list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, inst in enumerate(self._instances):
            row = QFrame()
            row.setObjectName("InstanceRow")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(10, 8, 10, 8)
            rl.setSpacing(8)

            # Enabled checkbox
            chk = QCheckBox()
            chk.setChecked(inst.get("enabled", True))
            chk.stateChanged.connect(lambda state, idx=i: self._toggle_enabled(idx, state))
            rl.addWidget(chk)

            # Icon + type badge
            type_icon = _TYPE_ICONS.get(inst.get("type", ""), inst.get("type", ""))
            icon_lbl = QLabel(type_icon.split("  ")[0])
            icon_lbl.setStyleSheet("font-size: 16pt;")
            rl.addWidget(icon_lbl)

            # Name + baseline info
            info_col = QVBoxLayout()
            info_col.setSpacing(2)
            name_lbl = QLabel(f"<b>{inst.get('name', inst.get('id', '?'))}</b>")
            name_lbl.setStyleSheet("color: #1f2937; font-size: 10pt;")
            bmode = inst.get("baseline_mode", "always_off")
            bkw   = inst.get("baseline_power_kw", 0.0)
            baseline_lbl = QLabel(f"Baseline: {bmode}  ·  {bkw:.1f} kW")
            baseline_lbl.setStyleSheet("color: #64748b; font-size: 8pt;")
            info_col.addWidget(name_lbl)
            info_col.addWidget(baseline_lbl)
            rl.addLayout(info_col, stretch=1)

            # Edit / Delete buttons
            edit_btn = QPushButton("Edit")
            edit_btn.setStyleSheet(_EDIT_BTN_STYLE)
            edit_btn.clicked.connect(lambda _, idx=i: self._edit_instance(idx))
            del_btn = QPushButton("Delete")
            del_btn.setStyleSheet(_DEL_BTN_STYLE)
            del_btn.clicked.connect(lambda _, idx=i: self._delete_instance(idx))
            rl.addWidget(edit_btn)
            rl.addWidget(del_btn)

            self._list_lay.insertWidget(self._list_lay.count() - 1, row)

    # ── Actions ─────────────────────────────────────────────────────────────

    def _toggle_enabled(self, idx: int, state: int):
        self._instances[idx]["enabled"] = bool(state)

    def _edit_instance(self, idx: int):
        dlg = AssetInstanceDialog(self._instances[idx], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._instances[idx] = dlg.result_instance()
            self._refresh_list()

    def _delete_instance(self, idx: int):
        self._instances.pop(idx)
        self._refresh_list()

    def _add_asset(self):
        menu = QMenu(self)
        for type_key, label in _ADD_TYPES:
            act = menu.addAction(label)
            act.setData(type_key)
        chosen = menu.exec(self.cursor().pos())
        if chosen is None:
            return
        type_key = chosen.data()
        new_inst = {
            "id":                self._unique_id(type_key),
            "type":              type_key,
            "name":              _TYPE_ICONS.get(type_key, type_key).split("  ")[-1].strip(),
            "enabled":           True,
            "baseline_mode":     _BASELINE_MODES_BY_TYPE.get(type_key, ["always_off"])[0],
            "baseline_power_kw": 0.0,
        }
        if type_key == "flex":
            new_inst["baseline_time_start"] = "00:00"
            new_inst["baseline_time_end"]   = "23:59"
        self._instances.append(new_inst)
        self._refresh_list()

    def _save(self):
        raw = load_dashboard_config()
        mpc = raw.get("mpc", {})
        mpc["asset_instances"] = self._instances

        # Keys that belong to the common instance header (not solver params)
        _skip_keys = {"id", "type", "name", "enabled"}

        # For each asset type: sync enabled flag + propagate params from first
        # enabled instance of that type into the corresponding solver JSON section
        for type_key, json_section in _TYPE_TO_CFG_FLAG.items():
            any_enabled = any(
                inst.get("type") == type_key and inst.get("enabled", True)
                for inst in self._instances
            )
            mpc.setdefault(json_section, {})["enabled"] = any_enabled
            for inst in self._instances:
                if inst.get("type") == type_key and inst.get("enabled", True):
                    params = {k: v for k, v in inst.items() if k not in _skip_keys}
                    # Remap flex baseline time keys: instance uses baseline_time_*,
                    # section dict (and from_dict) expects bl_time_*
                    if type_key == "flex":
                        if "baseline_time_start" in params:
                            params["bl_time_start"] = params.pop("baseline_time_start")
                        if "baseline_time_end" in params:
                            params["bl_time_end"] = params.pop("baseline_time_end")
                    mpc[json_section].update(params)
                    break

        raw["mpc"] = mpc
        save_dashboard_config(raw)
        self.accept()

    def _open_full_settings(self):
        dlg = MpcConfigDialog(self)
        dlg.exec()


