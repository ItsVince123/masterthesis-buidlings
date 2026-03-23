"""QSS stylesheet definitions for the Dashboard UI.

Keeping all styling in one place makes it easy to adjust colours, fonts, and
spacing without digging through widget code.
"""

# ---------------------------------------------------------------------------
# Main window & widget base styles
# ---------------------------------------------------------------------------
MAIN_WINDOW_STYLE = """
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
"""

# ---------------------------------------------------------------------------
# Button styles
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Historical analysis dialog
# ---------------------------------------------------------------------------
HISTORICAL_DIALOG_STYLE = """
QDialog { background-color: #eef3f9; }
QLabel  { color: #1f2937; font-family: 'Segoe UI'; font-size: 10pt; }
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
VALUE_STYLE = "font-family: 'Consolas'; font-weight: 700;"

def value_css(colour: str) -> str:
    """Return an inline style string for a tag-value label."""
    return f"color: {colour}; {VALUE_STYLE}"
