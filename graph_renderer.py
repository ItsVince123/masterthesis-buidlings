"""
╔══════════════════════════════════════════════════════════════════╗
║  FRONTEND FILE — student is NOT responsible for this module      ║
║                                                                  ║
║  All graph rendering via QPainter on QPixmap.                    ║
║  Pure display logic — no calculations here.                      ║
╚══════════════════════════════════════════════════════════════════╝

Graph rendering functions using QPainter on QPixmap.

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
    selected_index: int | None = None,
    width: int = 420,
    height: int = 200,
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

    # Data step blocks (bar / column style)
    slot_w = draw_w / max(n, 1)
    bar_pen = QPen(QColor(line_color))
    bar_pen.setWidth(2)
    painter.setPen(bar_pen)
    for i in range(n):
        x0 = int(pad_l + i * slot_w)
        x1 = int(pad_l + (i + 1) * slot_w)
        y = int(y_at(values[i]))
        painter.drawLine(x0, y, x1, y)
        if i > 0:
            y_prev = int(y_at(values[i - 1]))
            painter.drawLine(x0, y_prev, x0, y)

    # "Now" marker
    if current_index is not None and 0 <= current_index < n:
        now_pen = QPen(QColor("#dc2626"))
        now_pen.setWidth(2)
        painter.setPen(now_pen)
        x_now = int(pad_l + current_index * slot_w)
        painter.drawLine(x_now, pad_t, x_now, height - pad_b)
        painter.drawText(x_now + 4, pad_t + 12, "Now")

    # Selected slot marker (purple dashed line)
    if selected_index is not None and 0 <= selected_index < n:
        sel_pen = QPen(QColor("#7c3aed"))
        sel_pen.setWidth(2)
        sel_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(sel_pen)
        x_sel = int(pad_l + selected_index * slot_w)
        painter.drawLine(x_sel, pad_t, x_sel, height - pad_b)
        painter.setPen(QPen(QColor("#7c3aed")))
        fm = painter.fontMetrics()
        arrow_w = fm.horizontalAdvance("\u25bc")
        painter.drawText(x_sel - arrow_w // 2, pad_t + 12, "\u25bc")

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
    selected_index: int | None = None,
    width: int = 420,
    height: int = 200,
) -> QPixmap:
    """Line graph styled for electricity prices (blue)."""
    return draw_series_graph(
        prices, "EUR/MWh", start_label, end_label, "#1d4ed8", current_index,
        selected_index, width, height,
    )


def draw_solar_graph(
    powers: list[float],
    start_label: str,
    end_label: str,
    current_index: int | None = None,
    selected_index: int | None = None,
    width: int = 420,
    height: int = 200,
) -> QPixmap:
    """Line graph styled for solar power output (amber)."""
    return draw_series_graph(
        powers, "kW", start_label, end_label, "#f59e0b", current_index,
        selected_index, width, height,
    )


def draw_temperature_graph(
    temps: list[float],
    start_label: str,
    end_label: str,
    current_index: int | None = None,
    selected_index: int | None = None,
    width: int = 420,
    height: int = 200,
) -> QPixmap:
    """Line graph styled for outside temperature (teal/green). Legacy wrapper."""
    return draw_series_graph(
        temps, "°C", start_label, end_label, "#0d9488", current_index,
        selected_index, width, height,
    )


def draw_thermal_graph(
    outdoor_temps: list[float],
    building_temps: list[float],
    setpoint_c: float,
    heat_hp_kw: list[float],
    heat_boiler_kw: list[float],
    heat_chp_kw: list[float],
    start_label: str,
    end_label: str,
    current_index: int | None = None,
    selected_index: int | None = None,
    width: int = 420,
    height: int = 280,
) -> QPixmap:
    """Dual-axis step-line graph: temperatures (left °C) + heating sources (right kW).

    Left Y-axis  (°C):  outdoor temp (teal), building temp (orange), setpoint (dashed gray)
    Right Y-axis (kW):  CHP heat (amber), boiler (red), heat pump (green)
    """
    import math

    def _clean(lst: list[float]) -> list[float | None]:
        """Replace NaN/None with None for safe rendering."""
        out = []
        for v in lst:
            out.append(v if (v is not None and not math.isnan(v)) else None)
        return out

    n = max(len(outdoor_temps), len(building_temps),
            len(heat_hp_kw), len(heat_boiler_kw), len(heat_chp_kw), 1)

    out_t  = _clean(outdoor_temps)
    bld_t  = _clean(building_temps)
    hp_kw  = _clean(heat_hp_kw)
    bl_kw  = _clean(heat_boiler_kw)
    chp_kw = _clean(heat_chp_kw)

    def _get(lst, i):
        return lst[i] if i < len(lst) else None

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fbff"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    pad_l, pad_r, pad_t, pad_b = 46, 46, 14, 36
    dw = width  - pad_l - pad_r
    dh = height - pad_t - pad_b
    slot_w = dw / max(n, 1)

    # ── Left Y scale: °C ──────────────────────────────────────────────
    all_t = [v for v in out_t + bld_t if v is not None] + [setpoint_c]
    t_min = (min(all_t) - 2.0) if all_t else -5.0
    t_max = (max(all_t) + 2.0) if all_t else 30.0
    t_span = max(t_max - t_min, 1e-6)

    def ly_t(v: float) -> int:
        return int(pad_t + dh - (v - t_min) / t_span * dh)

    # ── Right Y scale: kW ─────────────────────────────────────────────
    all_kw = [v for v in hp_kw + bl_kw + chp_kw if v is not None]
    kw_max = max(max(all_kw) * 1.10, 1.0) if all_kw else 1.0
    kw_min = 0.0

    def ry_kw(v: float) -> int:
        return int(pad_t + dh - (v - kw_min) / kw_max * dh)

    # ── Grid lines ────────────────────────────────────────────────────
    grid_pen = QPen(QColor("#e2ecf7"))
    grid_pen.setWidth(1)
    painter.setPen(grid_pen)
    for i in range(5):
        y = int(pad_t + i * dh / 4)
        painter.drawLine(pad_l, y, width - pad_r, y)

    # ── Helper: draw one step-line series, skipping None gaps ─────────
    def _draw_step_line(series: list, y_fn, pen: QPen):
        painter.setPen(pen)
        prev = None
        for i in range(n):
            v = _get(series, i)
            x0 = int(pad_l + i * slot_w)
            x1 = int(pad_l + (i + 1) * slot_w)
            if v is not None:
                y = y_fn(v)
                painter.drawLine(x0, y, x1, y)          # horizontal segment
                if prev is not None:
                    painter.drawLine(x0, y_fn(prev), x0, y)  # vertical step
            prev = v if v is not None else prev          # forward-fill across gaps

    # ── Setpoint dashed line ──────────────────────────────────────────
    sp_pen = QPen(QColor("#94a3b8"))
    sp_pen.setWidth(1)
    sp_pen.setStyle(Qt.PenStyle.DashLine)
    painter.setPen(sp_pen)
    y_sp = ly_t(setpoint_c)
    painter.drawLine(pad_l, y_sp, width - pad_r, y_sp)

    # ── Heating lines (right axis) ────────────────────────────────────
    _draw_step_line(chp_kw, ry_kw, QPen(QColor("#f59e0b"), 2))
    _draw_step_line(bl_kw,  ry_kw, QPen(QColor("#ef4444"), 2))
    _draw_step_line(hp_kw,  ry_kw, QPen(QColor("#16a34a"), 2))

    # ── Temperature lines (left axis) — drawn on top ──────────────────
    _draw_step_line(out_t, ly_t, QPen(QColor("#0d9488"), 2))
    _draw_step_line(bld_t, ly_t, QPen(QColor("#f97316"), 2))

    # ── Now / selected markers ────────────────────────────────────────
    if current_index is not None and 0 <= current_index < n:
        now_pen = QPen(QColor("#dc2626"), 2)
        painter.setPen(now_pen)
        x_now = int(pad_l + current_index * slot_w)
        painter.drawLine(x_now, pad_t, x_now, height - pad_b)
        painter.setPen(QPen(QColor("#dc2626")))
        painter.drawText(x_now + 3, pad_t + 11, "Now")

    if selected_index is not None and 0 <= selected_index < n:
        sel_pen = QPen(QColor("#7c3aed"), 2)
        sel_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(sel_pen)
        x_sel = int(pad_l + selected_index * slot_w)
        painter.drawLine(x_sel, pad_t, x_sel, height - pad_b)
        painter.setPen(QPen(QColor("#7c3aed")))
        fm = painter.fontMetrics()
        painter.drawText(x_sel - fm.horizontalAdvance("▼") // 2, pad_t + 11, "▼")

    # ── Axis tick labels ──────────────────────────────────────────────
    text_pen = QPen(QColor("#475569"))
    painter.setPen(text_pen)
    # Left: °C
    for frac, val in ((0.0, t_max), (0.5, (t_max + t_min) / 2), (1.0, t_min)):
        y = int(pad_t + frac * dh)
        painter.drawText(2, y + 4, f"{val:.0f}°")
    # Right: kW
    for frac, val in ((0.0, kw_max), (0.5, kw_max / 2), (1.0, 0.0)):
        y = int(pad_t + frac * dh)
        lbl = f"{val:.0f}kW"
        painter.drawText(width - pad_r + 3, y + 4, lbl)

    # ── Border lines (left + bottom) ─────────────────────────────────
    border_pen = QPen(QColor("#cbd5e1"), 1)
    painter.setPen(border_pen)
    painter.drawLine(pad_l, pad_t, pad_l, height - pad_b)
    painter.drawLine(pad_l, height - pad_b, width - pad_r, height - pad_b)
    painter.drawLine(width - pad_r, pad_t, width - pad_r, height - pad_b)

    # ── Legend ────────────────────────────────────────────────────────
    series_legend = [
        ("#0d9488", "Out °C"),
        ("#f97316", "Bld °C"),
        ("#94a3b8", "Setpoint"),
        ("#16a34a", "HP"),
        ("#ef4444", "Boiler"),
        ("#f59e0b", "CHP"),
    ]
    painter.setPen(text_pen)
    fm = painter.fontMetrics()
    lx = pad_l
    ly = height - 6
    for col, label in series_legend:
        painter.fillRect(lx, ly - 8, 12, 4, QColor(col))
        painter.drawText(lx + 15, ly, label)
        lx += 15 + fm.horizontalAdvance(label) + 8
        if lx + 50 > width - pad_r:
            break  # don't overflow

    # ── Time labels ───────────────────────────────────────────────────
    painter.setPen(QPen(QColor("#94a3b8")))
    painter.drawText(pad_l, height - pad_b + 12, start_label)
    mid_lbl = f"{start_label.split(' ')[0]}  →  {end_label.split(' ')[0]}"
    painter.drawText(width - pad_r - fm.horizontalAdvance(end_label), height - pad_b + 12, end_label)

    painter.end()
    return pixmap


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
    """Draw baseline (red) vs optimised (green) cost series as step blocks."""
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
    slot_w = draw_w / max(n, 1)

    def x_pos(i: int) -> float:
        return pad_l + i * slot_w

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

    y_zero = int(y_pos(0)) if min_val < 0 else height - pad_b

    # Savings shading (where baseline > optimised)
    for i in range(n):
        if baseline[i] > optimised[i]:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(22, 163, 74, 40)))
            x0 = int(x_pos(i))
            x1 = int(x_pos(i + 1))
            y_bl = int(y_pos(baseline[i]))
            y_op = int(y_pos(optimised[i]))
            painter.drawRect(x0, min(y_bl, y_op), x1 - x0, abs(y_bl - y_op))
            painter.setBrush(Qt.BrushStyle.NoBrush)

    # Baseline step blocks (red)
    pen_base = QPen(QColor("#dc2626"))
    pen_base.setWidth(2)
    painter.setPen(pen_base)
    for i in range(n):
        x0 = int(x_pos(i))
        x1 = int(x_pos(i + 1))
        y = int(y_pos(baseline[i]))
        painter.drawLine(x0, y, x1, y)
        if i > 0:
            y_prev = int(y_pos(baseline[i - 1]))
            painter.drawLine(x0, y_prev, x0, y)

    # Optimised step blocks (green)
    pen_opt = QPen(QColor("#16a34a"))
    pen_opt.setWidth(2)
    painter.setPen(pen_opt)
    for i in range(n):
        x0 = int(x_pos(i))
        x1 = int(x_pos(i + 1))
        y = int(y_pos(optimised[i]))
        painter.drawLine(x0, y, x1, y)
        if i > 0:
            y_prev = int(y_pos(optimised[i - 1]))
            painter.drawLine(x0, y_prev, x0, y)

    # Axis labels
    text_pen = QPen(QColor("#334155"))
    painter.setPen(text_pen)
    painter.drawText(4, pad_t + 4, f"{max_val:.4f} \u20ac")
    painter.drawText(4, height - pad_b + 4, f"{min_val:.4f} \u20ac")

    # X-axis time labels
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
    painter.drawText(pad_l + 144, legend_y + 4, "LP Optimised")

    painter.end()
    return pixmap


# ---------------------------------------------------------------------------
# Dual-series power/load comparison graph
# ---------------------------------------------------------------------------

def draw_power_comparison_graph(
    baseline_kwh: np.ndarray,
    smpc_kwh: np.ndarray,
    x_labels: list[str],
    prices_eur_mwh: np.ndarray | None = None,
    width: int = 700,
    height: int = 260,
) -> QPixmap:
    """Draw baseline load (red) vs SMPC load (green) as step blocks.

    If *prices_eur_mwh* is provided, an orange price curve is overlaid
    on a secondary Y axis (right side).
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f8fbff"))

    pad_l = 60
    pad_r = 60 if prices_eur_mwh is not None else 20
    pad_t, pad_b = 24, 40
    draw_w = width - pad_l - pad_r
    draw_h = height - pad_t - pad_b

    all_vals = np.concatenate([baseline_kwh, smpc_kwh])
    min_val = float(np.min(all_vals))
    max_val = float(np.max(all_vals))
    spread = max(max_val - min_val, 1e-6)
    n = len(baseline_kwh)
    slot_w = draw_w / max(n, 1)

    def x_pos(i: int) -> float:
        return pad_l + i * slot_w

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

    # Price overlay (orange step blocks, secondary Y axis)
    if prices_eur_mwh is not None:
        p_min = float(np.min(prices_eur_mwh))
        p_max = float(np.max(prices_eur_mwh))
        p_spread = max(p_max - p_min, 1e-6)

        def y_price(v: float) -> float:
            return height - pad_b - ((v - p_min) / p_spread) * draw_h

        # Semi-transparent orange fill
        for i in range(n):
            x0 = int(x_pos(i))
            x1 = int(x_pos(i + 1))
            yp = int(y_price(prices_eur_mwh[i]))
            y_bottom = height - pad_b
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(245, 158, 11, 35)))
            painter.drawRect(x0, yp, x1 - x0, y_bottom - yp)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        # Orange step outline
        pen_price = QPen(QColor("#f59e0b"))
        pen_price.setWidth(1)
        painter.setPen(pen_price)
        for i in range(n):
            x0 = int(x_pos(i))
            x1 = int(x_pos(i + 1))
            yp = int(y_price(prices_eur_mwh[i]))
            painter.drawLine(x0, yp, x1, yp)
            if i > 0:
                yp_prev = int(y_price(prices_eur_mwh[i - 1]))
                painter.drawLine(x0, yp_prev, x0, yp)

        # Right-side price axis labels
        text_pen_price = QPen(QColor("#b45309"))
        painter.setPen(text_pen_price)
        painter.drawText(width - pad_r + 4, pad_t + 4, f"{p_max:.0f}")
        painter.drawText(width - pad_r + 4, height - pad_b + 4, f"{p_min:.0f}")
        painter.drawText(width - pad_r + 4, pad_t + 16, "\u20ac/MWh")

    # Baseline load step blocks (red)
    pen_base = QPen(QColor("#dc2626"))
    pen_base.setWidth(2)
    painter.setPen(pen_base)
    for i in range(n):
        x0 = int(x_pos(i))
        x1 = int(x_pos(i + 1))
        y = int(y_pos(baseline_kwh[i]))
        painter.drawLine(x0, y, x1, y)
        if i > 0:
            y_prev = int(y_pos(baseline_kwh[i - 1]))
            painter.drawLine(x0, y_prev, x0, y)

    # SMPC load step blocks (green)
    pen_smpc = QPen(QColor("#16a34a"))
    pen_smpc.setWidth(2)
    painter.setPen(pen_smpc)
    for i in range(n):
        x0 = int(x_pos(i))
        x1 = int(x_pos(i + 1))
        y = int(y_pos(smpc_kwh[i]))
        painter.drawLine(x0, y, x1, y)
        if i > 0:
            y_prev = int(y_pos(smpc_kwh[i - 1]))
            painter.drawLine(x0, y_prev, x0, y)

    # Left axis labels (kWh)
    text_pen = QPen(QColor("#334155"))
    painter.setPen(text_pen)
    painter.drawText(4, pad_t + 4, f"{max_val:.0f} kWh")
    painter.drawText(4, height - pad_b + 4, f"{min_val:.0f} kWh")

    # X-axis time labels
    step = max(1, n // 6)
    for i in range(0, n, step):
        painter.drawText(int(x_pos(i)) - 14, height - 8, x_labels[i])

    # Legend
    legend_y = pad_t - 6
    painter.setPen(pen_base)
    painter.drawLine(pad_l + 10, legend_y, pad_l + 30, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 34, legend_y + 4, "Baseline load")
    painter.setPen(pen_smpc)
    painter.drawLine(pad_l + 150, legend_y, pad_l + 170, legend_y)
    painter.setPen(text_pen)
    painter.drawText(pad_l + 174, legend_y + 4, "LP load")
    if prices_eur_mwh is not None:
        pen_price_leg = QPen(QColor("#f59e0b"))
        pen_price_leg.setWidth(2)
        painter.setPen(pen_price_leg)
        painter.drawLine(pad_l + 270, legend_y, pad_l + 290, legend_y)
        painter.setPen(text_pen)
        painter.drawText(pad_l + 294, legend_y + 4, "Price")

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
