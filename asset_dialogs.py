"""Dialogs for adding, editing, and removing energy assets.

Provides :class:`AssetManagerDialog` (list of all assets with add/edit/remove)
and :class:`AssetEditorDialog` (form for a single asset's properties).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from energy_assets import (
    ASSET_TYPES, FIXED_LOAD, GENERATOR, SHIFTABLE_LOAD, STORAGE,
    EnergyAsset, load_assets, save_assets, ensure_defaults,
)

# Icon choices offered in the editor
_ICON_OPTIONS = {
    "plug":      "\U0001f50c  Plug",
    "sun":       "\u2600\ufe0f  Sun",
    "fire":      "\U0001f525  Fire",
    "heat":      "\u2668\ufe0f  Heat",
    "snowflake": "\U0001f9ca  Snowflake",
    "car":       "\U0001f697  Car",
    "globe":     "\U0001f30d  Globe",
    "target":    "\U0001f3af  Target",
    "battery":   "\U0001f50b  Battery",
    "lightning": "\u26a1  Lightning",
    "factory":   "\U0001f3ed  Factory",
    "house":     "\U0001f3e0  House",
    "wind":      "\U0001f32c\ufe0f  Wind",
    "leaf":      "\U0001f33f  Leaf",
    "gear":      "\u2699\ufe0f  Gear",
    "thermo":    "\U0001f321\ufe0f  Thermometer",
}

_DIALOG_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Segoe UI'; font-size: 10pt; }
QLineEdit, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #d6dfeb;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    min-height: 24px;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #d6dfeb;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    min-height: 24px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #d6dfeb;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #475569;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d6dfeb;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}
"""

_BTN_PRIMARY = """
QPushButton {
    background-color: #2563eb; color: white; border: 1px solid #1d4ed8;
    border-radius: 6px; padding: 6px 18px; font-size: 10pt; font-weight: 700;
}
QPushButton:hover { background-color: #1d4ed8; }
"""

_BTN_DANGER = """
QPushButton {
    background-color: #dc2626; color: white; border: 1px solid #b91c1c;
    border-radius: 6px; padding: 6px 18px; font-size: 10pt; font-weight: 700;
}
QPushButton:hover { background-color: #b91c1c; }
"""

_BTN_SECONDARY = """
QPushButton {
    background-color: #64748b; color: white; border: 1px solid #475569;
    border-radius: 6px; padding: 6px 14px; font-size: 10pt; font-weight: 600;
}
QPushButton:hover { background-color: #475569; }
"""


# ═══════════════════════════════════════════════════════════════════
# Single-asset editor dialog
# ═══════════════════════════════════════════════════════════════════

class AssetEditorDialog(QDialog):
    """Form dialog for creating or editing one :class:`EnergyAsset`."""

    def __init__(self, asset: EnergyAsset | None = None, category: str = "input",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._is_new = asset is None
        self._asset = asset if asset else EnergyAsset(category=category)
        self.setWindowTitle("New Asset" if self._is_new else f"Edit — {self._asset.name}")
        self.setMinimumWidth(420)
        self.setStyleSheet(_DIALOG_STYLE)
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)

        # Name
        self._name = QLineEdit(self._asset.name)
        self._name.setPlaceholderText("e.g. Ice Banks, Solar Panels")
        form.addRow("Name:", self._name)

        # Asset type
        self._type = QComboBox()
        for key, label in ASSET_TYPES.items():
            self._type.addItem(label, key)
        idx = self._type.findData(self._asset.asset_type)
        if idx >= 0:
            self._type.setCurrentIndex(idx)
        self._type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Type:", self._type)

        # Icon
        self._icon = QComboBox()
        for key, label in _ICON_OPTIONS.items():
            self._icon.addItem(label, key)
        idx = self._icon.findData(self._asset.icon)
        if idx >= 0:
            self._icon.setCurrentIndex(idx)
        form.addRow("Icon:", self._icon)

        # Enabled
        self._enabled = QCheckBox("Active")
        self._enabled.setChecked(self._asset.enabled)
        self._enabled.setToolTip(
            "When unchecked the asset stays in the list\n"
            "but is excluded from simulations."
        )
        form.addRow("Enabled:", self._enabled)

        layout.addLayout(form)

        # ── Shiftable-load properties ───────────────────────────────
        self._shift_frame = QFrame()
        self._shift_frame.setObjectName("PropertyGroup")
        sf = QFormLayout(self._shift_frame)
        sf.setContentsMargins(0, 8, 0, 0)
        sf.setSpacing(8)

        hdr = QLabel("Shiftable Load Properties")
        hdr.setStyleSheet("font-weight: 700; color: #0b3a6e;")
        sf.addRow(hdr)

        self._hourly_max = QDoubleSpinBox()
        self._hourly_max.setRange(0, 100_000)
        self._hourly_max.setDecimals(1)
        self._hourly_max.setSuffix(" kWh")
        self._hourly_max.setValue(self._asset.hourly_max_kwh)
        self._hourly_max.setToolTip("Maximum energy this load can draw in one hour.")
        sf.addRow("Max per hour:", self._hourly_max)

        self._shift_daily = QDoubleSpinBox()
        self._shift_daily.setRange(0, 1_000_000)
        self._shift_daily.setDecimals(1)
        self._shift_daily.setSuffix(" kWh")
        self._shift_daily.setValue(self._asset.daily_energy_kwh)
        self._shift_daily.setToolTip(
            "Fixed total daily energy demand in kWh.\n"
            "Set to 0 to derive from CSV data."
        )
        sf.addRow("Daily total:", self._shift_daily)

        csv_hdr = QLabel("Historical CSV Linking (optional)")
        csv_hdr.setStyleSheet("font-weight: 600; color: #475569; font-size: 9pt;")
        sf.addRow(csv_hdr)

        self._csv_col = QLineEdit(self._asset.csv_column)
        self._csv_col.setPlaceholderText("e.g. TotalChiller — leave blank if none")
        self._csv_col.setToolTip(
            "Column name in the historical CSV that holds\n"
            "this load's hourly consumption. Leave blank\n"
            "if this asset has no historical data."
        )
        sf.addRow("CSV column:", self._csv_col)

        layout.addWidget(self._shift_frame)

        # ── Generator properties ────────────────────────────────────
        self._gen_frame = QFrame()
        self._gen_frame.setObjectName("PropertyGroup")
        gf = QFormLayout(self._gen_frame)
        gf.setContentsMargins(0, 8, 0, 0)
        gf.setSpacing(8)

        hdr2 = QLabel("Generator Properties")
        hdr2.setStyleSheet("font-weight: 700; color: #0b3a6e;")
        gf.addRow(hdr2)

        self._capacity = QDoubleSpinBox()
        self._capacity.setRange(0, 100_000)
        self._capacity.setDecimals(3)
        self._capacity.setSuffix(" kWp")
        self._capacity.setValue(self._asset.capacity_kwp)
        self._capacity.setToolTip("Installed peak capacity (relevant for solar).")
        gf.addRow("Capacity:", self._capacity)

        self._decouple = QDoubleSpinBox()
        self._decouple.setRange(-10_000, 10_000)
        self._decouple.setDecimals(1)
        self._decouple.setSuffix(" €/MWh")
        self._decouple.setSpecialValueText("Always connected")
        self._decouple.setMinimum(-10_000)
        if self._asset.decouple_below_eur_mwh is not None:
            self._decouple.setValue(self._asset.decouple_below_eur_mwh)
        else:
            self._decouple.setValue(-10_000)
        self._decouple.setToolTip(
            "Disconnect this generator when the electricity\n"
            "price drops below this value (avoids paying to\n"
            "inject). Set to 'Always connected' to disable."
        )
        gf.addRow("Decouple below:", self._decouple)

        csv_hdr2 = QLabel("Historical CSV Linking (optional)")
        csv_hdr2.setStyleSheet("font-weight: 600; color: #475569; font-size: 9pt;")
        gf.addRow(csv_hdr2)

        self._solar_csv = QLineEdit(self._asset.solar_csv)
        self._solar_csv.setPlaceholderText("e.g. solar_2022.csv — leave blank if none")
        self._solar_csv.setToolTip(
            "Path to a CSV file with hourly solar generation.\n"
            "Only needed for historical simulation."
        )
        gf.addRow("Solar CSV file:", self._solar_csv)

        self._csv_gen_col = QLineEdit(self._asset.csv_gen_column)
        self._csv_gen_col.setPlaceholderText("e.g. ProductionWKK — leave blank if none")
        self._csv_gen_col.setToolTip(
            "Column name in the historical CSV for this\n"
            "generator's hourly output. Leave blank if not\n"
            "available (e.g. pure solar with separate CSV)."
        )
        gf.addRow("CSV column:", self._csv_gen_col)

        layout.addWidget(self._gen_frame)

        # ── Fixed-load properties ───────────────────────────────────
        self._fixed_frame = QFrame()
        self._fixed_frame.setObjectName("PropertyGroup")
        ff = QFormLayout(self._fixed_frame)
        ff.setContentsMargins(0, 8, 0, 0)
        ff.setSpacing(8)

        hdr3 = QLabel("Fixed Load Properties")
        hdr3.setStyleSheet("font-weight: 700; color: #0b3a6e;")
        ff.addRow(hdr3)

        self._fixed_csv_col = QLineEdit(self._asset.csv_column)
        self._fixed_csv_col.setPlaceholderText("e.g. RemainingUsage — leave blank if none")
        self._fixed_csv_col.setToolTip(
            "Column name in the historical CSV for this\n"
            "fixed load's hourly consumption."
        )
        ff.addRow("CSV column:", self._fixed_csv_col)

        self._daily_energy = QDoubleSpinBox()
        self._daily_energy.setRange(0, 1_000_000)
        self._daily_energy.setDecimals(1)
        self._daily_energy.setSuffix(" kWh")
        self._daily_energy.setValue(self._asset.daily_energy_kwh)
        self._daily_energy.setToolTip(
            "Fixed daily energy demand in kWh.\n"
            "Set to 0 to derive from CSV data."
        )
        ff.addRow("Daily energy:", self._daily_energy)

        layout.addWidget(self._fixed_frame)

        # ── Storage properties ──────────────────────────────────────
        self._stor_frame = QFrame()
        self._stor_frame.setObjectName("PropertyGroup")
        stf = QFormLayout(self._stor_frame)
        stf.setContentsMargins(0, 8, 0, 0)
        stf.setSpacing(8)

        hdr4 = QLabel("Storage Properties")
        hdr4.setStyleSheet("font-weight: 700; color: #0b3a6e;")
        stf.addRow(hdr4)

        self._stor_capacity = QDoubleSpinBox()
        self._stor_capacity.setRange(0, 1_000_000)
        self._stor_capacity.setDecimals(1)
        self._stor_capacity.setSuffix(" kWh")
        self._stor_capacity.setValue(self._asset.storage_capacity_kwh)
        self._stor_capacity.setToolTip("Total usable storage capacity.")
        stf.addRow("Capacity:", self._stor_capacity)

        self._charge_rate = QDoubleSpinBox()
        self._charge_rate.setRange(0, 100_000)
        self._charge_rate.setDecimals(1)
        self._charge_rate.setSuffix(" kW")
        self._charge_rate.setValue(self._asset.charge_rate_kw)
        self._charge_rate.setToolTip("Maximum charging power.")
        stf.addRow("Charge rate:", self._charge_rate)

        self._discharge_rate = QDoubleSpinBox()
        self._discharge_rate.setRange(0, 100_000)
        self._discharge_rate.setDecimals(1)
        self._discharge_rate.setSuffix(" kW")
        self._discharge_rate.setValue(self._asset.discharge_rate_kw)
        self._discharge_rate.setToolTip("Maximum discharging power.")
        stf.addRow("Discharge rate:", self._discharge_rate)

        self._efficiency = QDoubleSpinBox()
        self._efficiency.setRange(0, 100)
        self._efficiency.setDecimals(1)
        self._efficiency.setSuffix(" %")
        self._efficiency.setValue(self._asset.efficiency * 100)
        self._efficiency.setToolTip("Round-trip efficiency in percent.")
        stf.addRow("Efficiency:", self._efficiency)

        layout.addWidget(self._stor_frame)

        # ── Buttons ─────────────────────────────────────────────────
        btns = QHBoxLayout()
        btns.setSpacing(10)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(_BTN_PRIMARY)
        save_btn.clicked.connect(self._on_save)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_BTN_SECONDARY)
        cancel_btn.clicked.connect(self.reject)

        btns.addStretch()
        btns.addWidget(cancel_btn)
        btns.addWidget(save_btn)
        layout.addLayout(btns)

        self._on_type_changed()

    def _on_type_changed(self):
        current = self._type.currentData()
        self._shift_frame.setVisible(current == SHIFTABLE_LOAD)
        self._gen_frame.setVisible(current == GENERATOR)
        self._fixed_frame.setVisible(current == FIXED_LOAD)
        self._stor_frame.setVisible(current == STORAGE)

    def _on_save(self):
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Name is required.")
            return
        self._asset.name = name
        self._asset.asset_type = self._type.currentData()
        self._asset.icon = self._icon.currentData()
        self._asset.enabled = self._enabled.isChecked()

        atype = self._asset.asset_type
        if atype == SHIFTABLE_LOAD:
            self._asset.csv_column = self._csv_col.text().strip()
            self._asset.hourly_max_kwh = self._hourly_max.value()
            self._asset.daily_energy_kwh = self._shift_daily.value()
        elif atype == GENERATOR:
            self._asset.capacity_kwp = self._capacity.value()
            self._asset.solar_csv = self._solar_csv.text().strip()
            self._asset.csv_gen_column = self._csv_gen_col.text().strip()
            decouple_val = self._decouple.value()
            self._asset.decouple_below_eur_mwh = (
                None if decouple_val <= -10_000 else decouple_val
            )
        elif atype == FIXED_LOAD:
            self._asset.csv_column = self._fixed_csv_col.text().strip()
            self._asset.daily_energy_kwh = self._daily_energy.value()
        elif atype == STORAGE:
            self._asset.storage_capacity_kwh = self._stor_capacity.value()
            self._asset.charge_rate_kw = self._charge_rate.value()
            self._asset.discharge_rate_kw = self._discharge_rate.value()
            self._asset.efficiency = self._efficiency.value() / 100.0
        self.accept()

    def get_asset(self) -> EnergyAsset:
        return self._asset


# ═══════════════════════════════════════════════════════════════════
# Asset manager dialog  (list + add / edit / remove)
# ═══════════════════════════════════════════════════════════════════

class AssetManagerDialog(QDialog):
    """Lists all energy assets with Add / Edit / Remove controls.

    Opens from the "+ Add input" / "+ Add output" buttons.  The
    *category* argument (``"input"`` or ``"output"``) controls which
    column new assets default to.
    """

    def __init__(self, category: str = "input", parent: QWidget | None = None):
        super().__init__(parent)
        self._category = category
        self.setWindowTitle(f"Manage Energy Assets — {category.capitalize()}s")
        self.setMinimumSize(520, 400)
        self.setStyleSheet(_DIALOG_STYLE)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._assets: list[EnergyAsset] = []
        self._changed = False
        self._build_ui()
        self._refresh_list()

    @property
    def changed(self) -> bool:
        return self._changed

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel(f"Energy Assets — {self._category.capitalize()}s")
        title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #0b3a6e;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Scrollable list area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll, stretch=1)

        # Bottom buttons
        btns = QHBoxLayout()
        btns.setSpacing(10)

        add_btn = QPushButton("+ Add Asset")
        add_btn.setStyleSheet(_BTN_PRIMARY)
        add_btn.clicked.connect(self._on_add)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_BTN_SECONDARY)
        close_btn.clicked.connect(self.accept)

        btns.addWidget(add_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    # ── List management ─────────────────────────────────────────────

    def _refresh_list(self):
        self._assets = load_assets()
        if not self._assets:
            self._assets = ensure_defaults()

        # Clear old rows
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Filter to current category
        shown = [a for a in self._assets if a.category == self._category]
        if not shown:
            lbl = QLabel("No assets defined yet. Click '+ Add Asset' to begin.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #64748b; font-style: italic;")
            self._list_layout.addWidget(lbl)
            return

        for asset in shown:
            self._list_layout.addWidget(self._make_row(asset))

    def _make_row(self, asset: EnergyAsset) -> QFrame:
        row = QFrame()
        row.setStyleSheet(
            "QFrame { background: #fff; border: 1px solid #d6dfeb;"
            " border-radius: 8px; }"
        )
        rl = QHBoxLayout(row)
        rl.setContentsMargins(14, 10, 14, 10)
        rl.setSpacing(10)

        type_label = ASSET_TYPES.get(asset.asset_type, asset.asset_type)
        status = "" if asset.enabled else "  (disabled)"
        info = QLabel(f"<b>{asset.name}</b>  —  {type_label}{status}")
        info.setStyleSheet("border: none;")
        if not asset.enabled:
            info.setStyleSheet("border: none; color: #94a3b8;")
        rl.addWidget(info, stretch=1)

        edit_btn = QPushButton("Edit")
        edit_btn.setStyleSheet(_BTN_SECONDARY)
        edit_btn.setFixedWidth(70)
        edit_btn.clicked.connect(lambda _, a=asset: self._on_edit(a))
        rl.addWidget(edit_btn)

        del_btn = QPushButton("Remove")
        del_btn.setStyleSheet(_BTN_DANGER)
        del_btn.setFixedWidth(80)
        del_btn.clicked.connect(lambda _, a=asset: self._on_remove(a))
        rl.addWidget(del_btn)

        return row

    # ── Actions ─────────────────────────────────────────────────────

    def _on_add(self):
        dlg = AssetEditorDialog(category=self._category, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            asset = dlg.get_asset()
            self._assets.append(asset)
            save_assets(self._assets)
            self._changed = True
            self._refresh_list()

    def _on_edit(self, asset: EnergyAsset):
        dlg = AssetEditorDialog(asset=asset, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            save_assets(self._assets)
            self._changed = True
            self._refresh_list()

    def _on_remove(self, asset: EnergyAsset):
        reply = QMessageBox.question(
            self, "Remove Asset",
            f"Remove '{asset.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._assets = [a for a in self._assets if a.uid != asset.uid]
            save_assets(self._assets)
            self._changed = True
            self._refresh_list()
