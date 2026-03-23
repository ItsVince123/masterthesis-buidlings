"""Graph rendering functions using QPainter on QPixmap.

All functions are stateless: they accept data arrays and return a QPixmap
ready to be set on a QLabel via ``label.setPixmap(pixmap)``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QBrush, QPolygon


# ---------------------------------------------------------------------------
# Generic line-graph renderer
# ---------------------------------------------------------------------------

def draw_series_graph(
    values: list[float],
    unit_label: str,
    start_label: str,
    end_label: str,
    line_color: str,
    current_index: int | None = None,
    width: int = 420,
    height: int = 170,
) -> QPixmap:
    """Render a line graph with unit and timestamp annotations.

    Parameters
    ----------
    values        – data points to plot.
    unit_label    – unit shown on the Y axis (e.g. ``"EUR/MWh"``).
    start_label   – text drawn below the first data point.
    end_label     – text drawn below the last data point.
    line_color    – hex colour for the data line.
    current_index – if set, draws a vertical red "Now" marker at that index.
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fbff"))

    if not values:
        painter = QPainter(pixmap)
        painter.setPen(QColor("#64748b"))
        painter.drawText(
            0, 0, width, height, Qt.AlignmentFlag.AlignCenter, "No data",
        )
        painter.end()
        return pixmap

    pad_l, pad_r, pad_t, pad_b = 48, 12, 18, 28
    draw_w = width - pad_l - pad_r
    draw_h = height - pad_t - pad_b

    min_val = min(values)
    max_val = max(values)
    spread = max(max_val - min_val, 1e-6)
    n = len(values)

    def x_at(i: int) -> float:
        return pad_l + i * draw_w / max(n - 1, 1)

    def y_at(v: float) -> float:
        return height - pad_b - ((v - min_val) / spread) * draw_h

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Horizontal grid lines
    grid_pen = QPen(QColor("#dbe7f7"))
    grid_pen.setWidth(1)
    painter.setPen(grid_pen)
    for i in range(5):
        y = pad_t + i * draw_h / 4
        painter.drawLine(pad_l, int(y), width - pad_r, int(y))

    # Data line
    line_pen = QPen(QColor(line_color))
    line_pen.setWidth(2)
    painter.setPen(line_pen)
    for i in range(n - 1):
        painter.drawLine(
            int(x_at(i)), int(y_at(values[i])),
            int(x_at(i + 1)), int(y_at(values[i + 1])),
        )

    # "Now" marker
    if current_index is not None and 0 <= current_index < n:
        now_pen = QPen(QColor("#dc2626"))
        now_pen.setWidth(2)
        painter.setPen(now_pen)
        x_now = int(x_at(current_index))
        painter.drawLine(x_now, pad_t, x_now, height - pad_b)
        painter.drawText(x_now + 4, pad_t + 12, "Now")

    # Axis labels
    text_pen = QPen(QColor("#334155"))
    painter.setPen(text_pen)
    painter.drawText(4, 16, f"{max_val:.1f} {unit_label}")
    painter.drawText(4, height - pad_b + 4, f"{min_val:.1f} {unit_label}")
    painter.drawText(pad_l, height - 6, start_label)
    painter.drawText(width - 120, height - 6, end_label)

    painter.end()
    return pixmap


# ---------------------------------------------------------------------------
# Convenience wrappers with pre-set colours
# ---------------------------------------------------------------------------

def draw_price_graph(
    prices: list[float],
    start_label: str,
    end_label: str,
    current_index: int | None = None,
) -> QPixmap:
    """Line graph styled for electricity prices (blue)."""
    return draw_series_graph(
        prices, "EUR/MWh", start_label, end_label, "#1d4ed8", current_index,
    )


def draw_solar_graph(
    powers: list[float],
    start_label: str,
    end_label: str,
    current_index: int | None = None,
) -> QPixmap:
    """Line graph styled for solar power output (amber)."""
    return draw_series_graph(
        powers, "kW", start_label, end_label, "#f59e0b", current_index,
    )


# ---------------------------------------------------------------------------
# Dual-series comparison graph (used by historical analysis)
# ---------------------------------------------------------------------------

def draw_comparison_graph(
    baseline: np.ndarray,
    optimised: np.ndarray,
    x_labels: list[str],
    width: int = 700,
    height: int = 260,
) -> QPixmap:
    """Draw baseline (red) vs optimised (green) cost series.

    The area where baseline > optimised is shaded green to visualise savings.
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fbff"))

    pad_l, pad_r, pad_t, pad_b = 60, 20, 24, 40
    draw_w = width - pad_l - pad_r
    draw_h = height - pad_t - pad_b

    all_vals = np.concatenate([baseline, optimised])
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    spread = max(max_val - min_val, 1e-6)
    n = len(baseline)

    def x_pos(i: int) -> float:
        return pad_l + i * draw_w / max(n - 1, 1)

    def y_pos(v: float) -> float:
        return height - pad_b - ((v - min_val) / spread) * draw_h

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Grid lines
    grid_pen = QPen(QColor("#dbe7f7"))
    grid_pen.setWidth(1)
    painter.setPen(grid_pen)
    for i in range(5):
        y = pad_t + i * draw_h / 4
        painter.drawLine(pad_l, int(y), width - pad_r, int(y))

    # Savings shading (where baseline > optimised)
    for i in range(n - 1):
        if baseline[i] > optimised[i] or baseline[i + 1] > optimised[i + 1]:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(22, 163, 74, 40)))
            poly = QPolygon([
                QPoint(int(x_pos(i)),     int(y_pos(baseline[i]))),
                QPoint(int(x_pos(i + 1)), int(y_pos(baseline[i + 1]))),
                QPoint(int(x_pos(i + 1)), int(y_pos(optimised[i + 1]))),
                QPoint(int(x_pos(i)),     int(y_pos(optimised[i]))),
            ])
            painter.drawPolygon(poly)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    # Baseline line (red)
    pen_base = QPen(QColor("#dc2626"))
    pen_base.setWidth(2)
    painter.setPen(pen_base)
    for i in range(n - 1):
        painter.drawLine(
            int(x_pos(i)), int(y_pos(baseline[i])),
            int(x_pos(i + 1)), int(y_pos(baseline[i + 1])),
        )

    # Optimised line (green)
    pen_opt = QPen(QColor("#16a34a"))
    pen_opt.setWidth(2)
    painter.setPen(pen_opt)
    for i in range(n - 1):
        painter.drawLine(
            int(x_pos(i)), int(y_pos(optimised[i])),
            int(x_pos(i + 1)), int(y_pos(optimised[i + 1])),
        )

    # Axis labels
    text_pen = QPen(QColor("#334155"))
    painter.setPen(text_pen)
    painter.drawText(4, pad_t + 4, f"{max_val:.4f} \u20ac")
    painter.drawText(4, height - pad_b + 4, f"{min_val:.4f} \u20ac")

    # X-axis time labels (roughly every 4 hours for 96 slots)
    step = max(1, n // 6)
    for i in range(0, n, step):
        painter.drawText(int(x_pos(i)) - 14, height - 8, x_labels[i])

    # Legend
    legend_y = pad_t - 6
    painter.setPen(pen_base)
    painter.drawLine(pad_l + 10, legend_y, pad_l + 30, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 34, legend_y + 4, "Baseline")
    painter.setPen(pen_opt)
    painter.drawLine(pad_l + 120, legend_y, pad_l + 140, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 144, legend_y + 4, "SMPC Optimised")

    painter.end()
    return pixmap


# ---------------------------------------------------------------------------
# Three-series comparison graph (baseline vs simple-opt vs SMPC)
# ---------------------------------------------------------------------------

def draw_three_way_graph(
    baseline: np.ndarray,
    simple_opt: np.ndarray,
    smpc: np.ndarray,
    x_labels: list[str],
    width: int = 700,
    height: int = 300,
) -> QPixmap:
    """Draw baseline (red) vs simple optimisation (orange) vs SMPC (green).

    Used by the historical analysis dialog when comparing against real CSV data.
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fbff"))

    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 40
    draw_w = width - pad_l - pad_r
    draw_h = height - pad_t - pad_b

    all_vals = np.concatenate([baseline, simple_opt, smpc])
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    spread = max(max_val - min_val, 1e-6)
    n = len(baseline)

    def x_pos(i: int) -> float:
        return pad_l + i * draw_w / max(n - 1, 1)

    def y_pos(v: float) -> float:
        return height - pad_b - ((v - min_val) / spread) * draw_h

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Grid lines
    grid_pen = QPen(QColor("#dbe7f7"))
    grid_pen.setWidth(1)
    painter.setPen(grid_pen)
    for i in range(5):
        y = pad_t + i * draw_h / 4
        painter.drawLine(pad_l, int(y), width - pad_r, int(y))

    # Baseline line (red)
    pen_base = QPen(QColor("#dc2626"))
    pen_base.setWidth(2)
    painter.setPen(pen_base)
    for i in range(n - 1):
        painter.drawLine(
            int(x_pos(i)), int(y_pos(baseline[i])),
            int(x_pos(i + 1)), int(y_pos(baseline[i + 1])),
        )

    # Simple optimisation line (orange)
    pen_simple = QPen(QColor("#ea580c"))
    pen_simple.setWidth(2)
    painter.setPen(pen_simple)
    for i in range(n - 1):
        painter.drawLine(
            int(x_pos(i)), int(y_pos(simple_opt[i])),
            int(x_pos(i + 1)), int(y_pos(simple_opt[i + 1])),
        )

    # SMPC line (green)
    pen_smpc = QPen(QColor("#16a34a"))
    pen_smpc.setWidth(2)
    painter.setPen(pen_smpc)
    for i in range(n - 1):
        painter.drawLine(
            int(x_pos(i)), int(y_pos(smpc[i])),
            int(x_pos(i + 1)), int(y_pos(smpc[i + 1])),
        )

    # Axis labels
    text_pen = QPen(QColor("#334155"))
    painter.setPen(text_pen)
    painter.drawText(4, pad_t + 4, f"{max_val:.2f} \u20ac")
    painter.drawText(4, height - pad_b + 4, f"{min_val:.2f} \u20ac")

    # X-axis time labels
    step = max(1, n // 6)
    for i in range(0, n, step):
        painter.drawText(int(x_pos(i)) - 14, height - 8, x_labels[i])

    # Legend (3 series)
    legend_y = pad_t - 10
    painter.setPen(pen_base)
    painter.drawLine(pad_l + 10, legend_y, pad_l + 30, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 34, legend_y + 4, "Baseline")

    painter.setPen(pen_simple)
    painter.drawLine(pad_l + 120, legend_y, pad_l + 140, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 144, legend_y + 4, "Simple Opt")

    painter.setPen(pen_smpc)
    painter.drawLine(pad_l + 240, legend_y, pad_l + 260, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 264, legend_y + 4, "SMPC")

    painter.end()
    return pixmap
