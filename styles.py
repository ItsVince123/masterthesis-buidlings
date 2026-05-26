"""
QSS stylesheet definitions for the Dashboard UI.

Keeping all styling in one place makes it easy to adjust colours, fonts, and
spacing without digging through widget code.
"""

# ---------------------------------------------------------------------------
# Main window & widget base styles
# ---------------------------------------------------------------------------
MAIN_WINDOW_STYLE = """
QMainWindow, QWidget {
    background-color: #eef3f9;
    color: #1f2937;
    font-family: 'Calibri';
    font-size: 11pt;
}

QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #eef3f9;
    color: #1f2937;
}

QScrollBar:vertical {
    background: #e2e8f0; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #94a3b8; border-radius: 4px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QScrollBar:horizontal {
    background: #e2e8f0; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #94a3b8; border-radius: 4px; min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

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
    font-family: 'Calibri';
    font-size: 16pt;
    font-weight: 700;
}

QFrame#TagRow {
    background-color: #f8fbff;
    border: 1px solid #dce5f2;
    border-radius: 10px;
}

QLabel#TagName {
    background-color: transparent;
    color: #27364f;
    font-size: 10pt;
    font-weight: 600;
}

QLabel#TagValue {
    background-color: transparent;
    color: #0f766e;
    font-family: 'Calibri';
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

QLabel#BaselineLabel {
    color: #64748b;
    font-size: 9pt;
    font-style: italic;
    padding: 1px 2px;
}

QFrame#SectionDivider {
    background-color: #d4e2f4;
    min-height: 1px;
    max-height: 1px;
    border: none;
}
"""

# ---------------------------------------------------------------------------
# Button styles
# ---------------------------------------------------------------------------

SLOT_SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: 1px solid #cbd5e1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2563eb;
    border: 2px solid #1d4ed8;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #1d4ed8;
}
QSlider::sub-page:horizontal {
    background: #93c5fd;
    border-radius: 3px;
}
"""
COLUMN_BUTTON_STYLE = """
QPushButton {
    background-color: #2563eb;
    color: white;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 10pt;
    font-weight: 700;
}
QPushButton:hover  { background-color: #1d4ed8; }
QPushButton:pressed { background-color: #1e40af; }
"""

HISTORICAL_BUTTON_STYLE = """
QPushButton {
    background-color: #0b3a6e;
    color: white;
    border: 1px solid #082f59;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 10pt;
    font-weight: 700;
}
QPushButton:hover  { background-color: #0e4d8f; }
QPushButton:pressed { background-color: #072a50; }
"""

FUTURE_BUTTON_STYLE = """
QPushButton {
    background-color: #166534;
    color: white;
    border: 1px solid #14532d;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 10pt;
    font-weight: 700;
}
QPushButton:hover  { background-color: #15803d; }
QPushButton:pressed { background-color: #14532d; }
"""

# ---------------------------------------------------------------------------
# Historical analysis dialog
# ---------------------------------------------------------------------------
HISTORICAL_DIALOG_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Calibri'; font-size: 10pt; }
QDateEdit {
    background-color: #ffffff;
    border: 1px solid #d6dfeb;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    min-height: 28px;
}
"""

RUN_ANALYSIS_BUTTON_STYLE = """
QPushButton {
    background-color: #2563eb;
    color: white;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 10pt;
    font-weight: 700;
}
QPushButton:hover { background-color: #1d4ed8; }
"""

# ---------------------------------------------------------------------------
# Inline value colours (used by dashboard widget setStyleSheet calls)
# ---------------------------------------------------------------------------
VALUE_STYLE = "font-family: 'Calibri'; font-weight: 700;"

def value_css(colour: str) -> str:
    """Return an inline style string for a tag-value label."""
    return f"color: {colour}; background-color: transparent; {VALUE_STYLE}"
