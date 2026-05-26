"""
Three analysis modes
--------------------
1. Date Range  — pick start/end dates; weather auto-fetched from Open-Meteo;
                 prices loaded from a user-supplied CSV.
2. Full Year   — run every day that has price data for a chosen year;
                 progress bar updates as each day is solved.
3. Custom CSV  — upload a price CSV; date range is auto-detected from the
                 file; weather auto-fetched for that range.

Price CSV formats supported
---------------------------
* Tab-separated with header  ``Timestamp\\tPricesElec``
  - Timestamp: ``D/MM/YYYY H:MM``  (e.g. ``1/01/2022 0:00``)
  - Price unit: EUR/kWh  (e.g. 0.1)
* Semicolon-separated with header  ``Timestamp (Brussels);Price (EUR/MWh);…``
  - Price unit: EUR/MWh  → auto-divided by 1000 when max value > 10

Pre-2025 hourly prices are automatically up-sampled to 15-min by repeating
each value 4 times.
"""

from __future__ import annotations

import csv
import datetime
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QTabWidget, QWidget, QLabel, QPushButton, QDateEdit, QSpinBox,
    QProgressBar, QFileDialog, QScrollArea, QFrame, QDoubleSpinBox,
    QSizePolicy, QMessageBox, QTextEdit, QCheckBox,
)

from dashboard_config import load_dashboard_config, save_dashboard_config
from mpc_lp import (
    MPCConfig, MPCInputs, MPCOutputs,
    load_mpc_config, solve_mpc,
    compute_baseline_arrays, compute_asset_savings, compute_cop,
)
from graph_renderer import draw_price_graph, draw_solar_graph, draw_thermal_graph
from settings import DEFAULT_LATITUDE, DEFAULT_LONGITUDE
from styles import HISTORICAL_DIALOG_STYLE, RUN_ANALYSIS_BUTTON_STYLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weather fetching (archive + forecast APIs)
# ---------------------------------------------------------------------------

def _fetch_weather_historical(
    lat: float,
    lon: float,
    start_str: str,
    end_str: str,
) -> dict:
    """
    Fetch historical weather data from Open-Meteo.

    Uses the archive API (``archive-api.open-meteo.com``) for dates older
    than 5 days, and the standard forecast API otherwise.  Results are
    upsampled to 15-minute resolution.

    Parameters
    ----------
    lat, lon    : coordinates
    start_str   : ``YYYY-MM-DD`` string (first day, inclusive)
    end_str     : ``YYYY-MM-DD`` string (last day, inclusive)

    Returns
    -------
    dict with keys ``timestamps``, ``temperature``, ``uv_index``, ``wind_speed``
    (all lists of equal length, 15-min resolution).
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    today      = datetime.date.today()
    start_date = datetime.date.fromisoformat(start_str)
    use_archive = start_date < today - datetime.timedelta(days=5)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        if use_archive
        else "https://api.open-meteo.com/v1/forecast"
    )

    # Use a persistent cache so full-year runs don't refetch the same data
    expire = -1 if use_archive else 3600
    cache_session  = requests_cache.CachedSession(".cache_analysis", expire_after=expire)
    retry_session  = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params: dict = {
        "latitude":   lat,
        "longitude":  lon,
        "timezone":   "Europe/Brussels",
        "start_date": start_str,
        "end_date":   end_str,
        "hourly":     ["temperature_2m", "uv_index", "shortwave_radiation"],
    }

    logger.info("Fetching weather %s → %s (archive=%s)", start_str, end_str, use_archive)
    response = client.weather_api(url, params=params)[0]

    hourly    = response.Hourly()
    temps_h   = hourly.Variables(0).ValuesAsNumpy()   # temperature_2m
    uv_h      = hourly.Variables(1).ValuesAsNumpy()   # uv_index
    sw_h      = hourly.Variables(2).ValuesAsNumpy()   # shortwave_radiation [W/m²]

    n_h = min(len(temps_h), len(uv_h), len(sw_h))
    temps_h = np.where(np.isnan(temps_h[:n_h]), 10.0, temps_h[:n_h])
    uv_raw  = uv_h[:n_h]
    sw_raw  = np.where(np.isnan(sw_h[:n_h]), 0.0, sw_h[:n_h])
    # uv_index ≈ shortwave_radiation [W/m²] / 125  (UV 8 ≈ 1000 W/m²)
    # Use UV index where valid (> 0); fall back to shortwave-derived value otherwise.
    uv_h = np.where(
        np.isnan(uv_raw) | (uv_raw <= 0.0),
        sw_raw / 125.0,
        uv_raw,
    )
    uv_h = np.maximum(0.0, uv_h)

    # Upsample hourly → 15-min  (4 slots per hour)
    temps_15 = np.repeat(temps_h, 4)
    uv_15    = np.maximum(0.0, np.repeat(uv_h, 4))

    # Generate timestamps starting at midnight of start_str
    start_dt = datetime.datetime.fromisoformat(start_str + "T00:00:00")
    n = len(temps_15)
    timestamps = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(n)]

    return {
        "timestamps":  timestamps,
        "temperature": temps_15.tolist(),
        "uv_index":    uv_15.tolist(),
        "wind_speed":  [0.0] * n,
    }


# ---------------------------------------------------------------------------
# Price CSV parsing
# ---------------------------------------------------------------------------

def _parse_price_csv(filepath: str) -> "pd.DataFrame":
    """
    Parse a price CSV into a pandas DataFrame with DatetimeIndex and
    a ``price_eur_kwh`` column.

    Handles:
    * Tab and semicolon separators
    * ``D/MM/YYYY H:MM`` and ISO ``YYYY-MM-DD HH:MM`` timestamp formats
    * EUR/kWh and EUR/MWh (auto-detected by magnitude)
    * Comma-as-decimal-separator in price values
    * Hourly and 15-min resolutions
    """
    if not _PANDAS_OK:
        raise ImportError("pandas is required for price CSV parsing")

    raw = Path(filepath).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")

    # Auto-detect separator
    sep = "\t"
    for candidate in ["\t", ";", ","]:
        first_line = text.split("\n")[0]
        if candidate in first_line:
            sep = candidate
            break

    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Find timestamp and price columns
    ts_col = next(
        (c for c in df.columns if "timestamp" in c.lower() or "time" in c.lower()),
        df.columns[0],
    )
    price_col = next(
        (c for c in df.columns
         if any(k in c.lower() for k in ["price", "elec", "prijs", "eur"])),
        df.columns[1] if len(df.columns) > 1 else df.columns[0],
    )

    # Parse prices (handle comma decimal separator)
    prices_raw = (
        df[price_col]
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    prices_num = pd.to_numeric(prices_raw, errors="coerce").fillna(0.0)

    # Auto-detect EUR/MWh vs EUR/kWh
    max_val = prices_num.max()
    if max_val > 10.0:
        prices_num = prices_num / 1000.0   # EUR/MWh → EUR/kWh

    # Parse timestamps
    ts_series = df[ts_col].str.strip()
    # Try multiple formats
    parsed_ts = None
    for fmt in [
        "%Y-%m-%d %H:%M:%S",   # ISO with seconds (e.g. prices_future.csv)
        "%d/%m/%Y %H:%M",
        "%-d/%-m/%Y %-H:%M",   # may not work on Windows
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%d-%m-%Y %H:%M",
    ]:
        try:
            parsed_ts = pd.to_datetime(ts_series, format=fmt)
            break
        except (ValueError, TypeError):
            continue

    if parsed_ts is None:
        # Fallback: let pandas infer
        parsed_ts = pd.to_datetime(ts_series, dayfirst=True, errors="coerce")

    result_df = pd.DataFrame(
        {"price_eur_kwh": prices_num.values},
        index=parsed_ts,
    )
    result_df = result_df[~result_df.index.isna()]
    result_df = result_df.sort_index()
    return result_df


def _get_day_prices(price_df: "pd.DataFrame", date: datetime.date) -> Optional[np.ndarray]:
    """
    Extract 96 × 15-min price slots for *date* from *price_df*.

    Handles hourly (24 rows → repeat ×4) and 15-min (96 rows) resolutions.
    Returns ``None`` when no data is available for that date.
    """
    day_str = date.isoformat()
    # Slice the day (timezone-naive or tz-aware)
    try:
        mask = price_df.index.date == date
    except AttributeError:
        mask = price_df.index.normalize().date == date

    day_df = price_df[mask]
    if len(day_df) == 0:
        return None

    vals = day_df["price_eur_kwh"].values

    if len(vals) <= 24:
        # Hourly → upsample to 15-min
        arr = np.repeat(vals, 4)
    elif len(vals) < 96:
        arr = np.repeat(vals, max(1, 96 // len(vals)))
    else:
        arr = vals

    # Clip / pad to exactly 96
    if len(arr) < 96:
        arr = np.pad(arr, (0, 96 - len(arr)), "edge")
    return np.asarray(arr[:96], dtype=float)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _AnalysisWorker(QThread):
    """Solves MPC + baseline for a list of days in a background thread."""

    progress_signal = pyqtSignal(int, int)   # (current_day_index+1, total)
    day_done_signal = pyqtSignal(dict)        # per-day result dict
    error_signal    = pyqtSignal(str)         # non-fatal warning message
    finished_signal = pyqtSignal(list)        # list of error strings (may be empty)

    def __init__(
        self,
        days: list[dict],
        cfg: MPCConfig,
    ):
        """
        Parameters
        ----------
        days     : list of dicts, each with:
                     'date'        str (YYYY-MM-DD)
                     'price'       np.ndarray (96,) EUR/kWh
                     'temperature' np.ndarray (96,) °C
                     'uv_index'    np.ndarray (96,)
        cfg      : MPCConfig (day/night base load is read from cfg)
        """
        super().__init__()
        self.days     = days
        self.cfg      = cfg
        self._abort   = False

    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D102
        # ── Step 1: compute PV forecasts in QThread (pure Python, no DLL) ─
        from predict import predict_power_kw, WeatherRow

        total  = len(self.days)
        errors: list[str] = []
        H      = self.cfg.horizon_steps
        dt     = self.cfg.dt_hours
        days_with_ppv: list[dict] = []

        for day in self.days:
            temps = np.asarray(day["temperature"], dtype=float)
            uv    = np.asarray(day["uv_index"],    dtype=float)
            price = np.asarray(day["price"],        dtype=float)

            if len(price) < H:
                price = np.pad(price, (0, H - len(price)), "edge")
            if len(temps) < H:
                temps = np.pad(temps, (0, H - len(temps)), "edge")
            if len(uv) < H:
                uv = np.pad(uv, (0, H - len(uv)), "edge")

            ppv = []
            for k in range(H):
                row = WeatherRow(
                    timestamp=day["date"],
                    temperature_c=float(temps[k]),
                    uv_index=float(max(0.0, uv[k])),
                    wind_kmh=0.0,
                )
                ppv.append(predict_power_kw(row, self.cfg.pv_capacity_kwp))

            days_with_ppv.append({
                "date":        day["date"],
                "price":       price[:H].tolist(),
                "ppv":         ppv,
                "temperature": temps[:H].tolist(),
            })

        if self._abort:
            self.finished_signal.emit(errors)
            return

        # ── Step 2: solve all days in a subprocess ─────────────────────────
        # HIGHS DLL conflicts with Qt DLLs when loaded in the same process.
        # Running the solver in a child process (no Qt imported there) avoids
        # the silent native crash.  Same pattern as _lp_worker.py.
        import pickle
        import subprocess
        import sys as _sys
        from pathlib import Path as _Path

        _worker = _Path(__file__).parent / "_analysis_day_worker.py"
        payload = pickle.dumps({"days": days_with_ppv, "cfg": self.cfg})

        try:
            timeout_s = max(180, total * 60)   # at least 3 min; 1 min per day
            proc = subprocess.run(
                [_sys.executable, str(_worker)],
                input=payload,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"Analysis timed out (> {timeout_s} s)")
            self.finished_signal.emit(errors)
            return
        except Exception as exc:
            errors.append(f"Subprocess launch failed: {exc}")
            self.finished_signal.emit(errors)
            return

        if proc.returncode != 0:
            stderr_txt = proc.stderr.decode(errors="replace")[:1000]
            errors.append(f"Solver subprocess exited {proc.returncode}:\n{stderr_txt}")
            self.finished_signal.emit(errors)
            return

        try:
            sub_results: list[dict] = pickle.loads(proc.stdout)
        except Exception as exc:
            errors.append(f"Could not parse subprocess output: {exc}")
            self.finished_signal.emit(errors)
            return

        # ── Step 3: emit results ───────────────────────────────────────────
        for i, res in enumerate(sub_results):
            if self._abort:
                break
            if res.get("ok"):
                self.day_done_signal.emit(res)
            else:
                msg = f"Day {res.get('date', '?')}: {res.get('error', 'unknown')}"
                logger.warning(msg)
                self.error_signal.emit(msg)
                errors.append(msg)
            self.progress_signal.emit(i + 1, total)

        self.finished_signal.emit(errors)

    def abort(self) -> None:
        """Request early termination."""
        self._abort = True

    # ------------------------------------------------------------------

    def _run_day(self, day: dict) -> dict:
        from predict import predict_power_kw, WeatherRow

        H  = self.cfg.horizon_steps
        dt = self.cfg.dt_hours

        price = np.asarray(day["price"],       dtype=float)
        temps = np.asarray(day["temperature"], dtype=float)
        uv    = np.asarray(day["uv_index"],    dtype=float)

        # Pad to H slots
        for arr_ref in ["price", "temperature", "uv_index"]:
            a = np.asarray(day[arr_ref], dtype=float)
            if len(a) < H:
                a = np.pad(a, (0, H - len(a)), "edge")
            if arr_ref == "price":
                price = a[:H]
            elif arr_ref == "temperature":
                temps = a[:H]
            else:
                uv = a[:H]

        # Solar forecast from UV + temperature (predict_power_kw returns kW directly)
        ppv = []
        for k in range(H):
            row = WeatherRow(
                timestamp=day["date"],
                temperature_c=float(temps[k]),
                uv_index=float(max(0.0, uv[k])),
                wind_kmh=0.0,
            )
            ppv.append(predict_power_kw(row, self.cfg.pv_capacity_kwp))
        Ppv_forecast = np.array(ppv)

        from mpc_lp import build_pload_vector
        Pload = build_pload_vector(self.cfg, 0.0, H)

        inputs = MPCInputs(
            price_eur_kwh     = price,
            Pload_kw          = Pload,
            Ppv_forecast_kw   = Ppv_forecast,
            Tamb_c            = temps,
            SOC_init_kwh      = self.cfg.SOC_init_kwh,
            T_building_init_c = self.cfg.T_init_c,
            T_tank_init_c     = self.cfg.hw_T_init_c,
        )

        # Derive weekday from the simulated date so flex day-of-week gating
        # uses the historical day, not "today" (Mon=0 … Sun=6).
        _start_wd = None
        try:
            from datetime import datetime as _dtcls
            _start_wd = _dtcls.strptime(str(day["date"]), "%Y-%m-%d").weekday()
        except Exception:
            _start_wd = None

        mpc_out = solve_mpc(inputs, self.cfg, start_weekday=_start_wd)
        bl      = compute_baseline_arrays(inputs, self.cfg, start_weekday=_start_wd)
        savings = compute_asset_savings(mpc_out, bl, inputs, self.cfg)
        dt      = self.cfg.dt_hours

        return {
            "date":               day["date"],
            "baseline_cost_eur":  bl["total_cost"],
            "mpc_cost_eur":       mpc_out.mpc_cost_eur,
            "total_saving_eur":   bl["total_cost"] - mpc_out.mpc_cost_eur,
            # Gross baseline cost (imports + gas, no export credit). Useful as
            # a denominator for "% saving of the energy bill" because PV
            # export revenue otherwise shrinks the denominator and inflates %.
            "baseline_gross_eur": bl.get("gross_cost", bl["total_cost"]),
            "baseline_export_revenue_eur": bl.get("export_revenue", 0.0),
            "savings":            savings,
            "dt":                 dt,
            # MPC dispatch arrays (kW, as plain lists for Qt signal safety)
            "plan_Pgrid":  mpc_out.plan_Pgrid.tolist(),
            "plan_Php":    mpc_out.plan_Php.tolist(),
            "plan_Php_cool": mpc_out.plan_Php_cool.tolist() if len(mpc_out.plan_Php_cool) > 0 else [],
            "plan_Ppv":    mpc_out.plan_Ppv.tolist(),
            "plan_Pflex":  mpc_out.plan_Pflex.tolist(),
            "plan_Pch":    mpc_out.plan_Pch.tolist(),
            "plan_Pdis":   mpc_out.plan_Pdis.tolist(),
            "plan_Pgas":   mpc_out.plan_Pgas.tolist(),
            "plan_Ptank":  mpc_out.plan_Ptank.tolist(),
            "plan_Pchp":   mpc_out.plan_Pchp.tolist() if len(mpc_out.plan_Pchp) > 0 else [],
            "plan_Qchp":   mpc_out.plan_Qchp.tolist() if len(mpc_out.plan_Qchp) > 0 else [],
            # Baseline dispatch arrays
            "bl_Pgrid":    bl["Pgrid"].tolist(),
            "bl_Php":      bl["Php"].tolist(),
            "bl_Ppv":      bl["Ppv"].tolist(),
            "bl_Pflex":    bl["Pflex"].tolist(),
            "bl_Ptank":    bl["Ptank"].tolist(),
            "prices":      price.tolist(),
        }


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

_ANALYSIS_STYLE = HISTORICAL_DIALOG_STYLE + """
QGroupBox {
    font-family: 'Calibri';
    font-weight: 700;
    font-size: 10pt;
    color: #0b3a6e;
    border: 1px solid #c7d7ed;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QTabWidget::pane { border: 1px solid #c7d7ed; border-radius: 6px; }
QTabBar::tab {
    font-family: 'Calibri'; font-size: 9pt;
    padding: 5px 16px;
    background: #dce9f5;
    border: 1px solid #c7d7ed;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
}
QTabBar::tab:selected { background: #eef3f9; font-weight: 700; }
QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #ffffff;
    border: 1px solid #d6dfeb;
    border-radius: 6px;
    padding: 4px 8px;
    font-family: 'Calibri'; font-size: 10pt;
    min-height: 26px;
}
QProgressBar {
    border: 1px solid #c7d7ed; border-radius: 6px;
    background: #eef3f9; text-align: center;
    font-family: 'Calibri'; font-size: 9pt;
    min-height: 20px;
}
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
"""

_EXPORT_BTN_STYLE = """
QPushButton {
    background-color: #0f766e;
    color: white;
    border: 1px solid #0d665e;
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 10pt;
    font-weight: 700;
    font-family: 'Calibri';
}
QPushButton:hover   { background-color: #0d665e; }
QPushButton:disabled { background-color: #94a3b8; }
"""

_ABORT_BTN_STYLE = """
QPushButton {
    background-color: #dc2626;
    color: white;
    border: 1px solid #b91c1c;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 10pt;
    font-weight: 700;
    font-family: 'Calibri';
}
QPushButton:hover { background-color: #b91c1c; }
"""


def _kpi_label(text: str, bold: bool = False, color: str = "#1f2937") -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Calibri", 10, QFont.Weight.Bold if bold else QFont.Weight.Normal))
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def _val_label(text: str, color: str = "#0f766e") -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Calibri", 10, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


class AnalysisDialog(QDialog):
    """Unified MPC Analysis & Simulation dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MPC Analysis & Simulation")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(1080, avail.width() - 80) if avail else 1000
        h = min(820, avail.height() - 80) if avail else 760
        self.resize(w, h)
        self.setMinimumWidth(780)
        self.setStyleSheet(_ANALYSIS_STYLE)

        # State
        self._cfg: Optional[MPCConfig] = None
        self._price_df = None            # parsed price DataFrame (tabs 1 & 2)
        self._price_path = ""
        self._custom_price_df = None     # price DataFrame for tab 3
        self._custom_price_path = ""
        self._worker: Optional[_AnalysisWorker] = None
        self._results: list[dict] = []

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title
        title_lbl = QLabel("MPC Analysis & Simulation")
        title_lbl.setFont(QFont("Calibri", 14, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #0b3a6e;")
        root.addWidget(title_lbl)

        # ── Setup row ──────────────────────────────────────────────────────
        setup_row = QHBoxLayout()
        setup_row.setSpacing(12)

        # Mode tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_date_range_tab(), "Date Range")
        self._tabs.addTab(self._build_full_year_tab(),  "Full Year")
        self._tabs.addTab(self._build_custom_csv_tab(), "Custom CSV")
        setup_row.addWidget(self._tabs, 3)

        # Settings + run button panel
        setup_row.addWidget(self._build_settings_panel(), 2)
        root.addLayout(setup_row)

        # Restore last-used price CSV path silently
        self._restore_price_csv()

        # ── Progress bar ───────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m days")
        self._progress.hide()
        prog_row.addWidget(self._progress)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet(_ABORT_BTN_STYLE)
        self._abort_btn.setFixedWidth(80)
        self._abort_btn.hide()
        self._abort_btn.clicked.connect(self._abort_run)
        prog_row.addWidget(self._abort_btn)
        root.addLayout(prog_row)

        # ── Status label ──────────────────────────────────────────────────
        self._status_lbl = QLabel("Load a price CSV and choose a mode, then click Run.")
        self._status_lbl.setStyleSheet("color: #64748b; font-family: 'Calibri';")
        root.addWidget(self._status_lbl)

        # ── Results scroll area ────────────────────────────────────────────
        self._results_outer = QScrollArea()
        self._results_outer.setWidgetResizable(True)
        self._results_outer.setStyleSheet(
            "QScrollArea { border: 1px solid #c7d7ed; border-radius: 6px; }"
        )
        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.setSpacing(10)
        self._results_layout.setContentsMargins(10, 10, 10, 10)
        # Placeholder
        self._placeholder_lbl = QLabel("Results will appear here after analysis.")
        self._placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_lbl.setStyleSheet("color: #94a3b8; font-family: 'Calibri'; font-size: 11pt;")
        self._results_layout.addWidget(self._placeholder_lbl)
        self._results_layout.addStretch()
        self._results_outer.setWidget(self._results_container)
        self._results_outer.setMinimumHeight(280)
        root.addWidget(self._results_outer, 1)

        # ── Export buttons ────────────────────────────────────────────────
        export_row = QHBoxLayout()
        export_row.addStretch()
        self._export_summary_by_year_btn = QPushButton("Export Summary by Year as CSV")
        self._export_summary_by_year_btn.setStyleSheet(_EXPORT_BTN_STYLE)
        self._export_summary_by_year_btn.setMinimumHeight(36)
        self._export_summary_by_year_btn.setEnabled(False)
        self._export_summary_by_year_btn.setVisible(False)
        self._export_summary_by_year_btn.clicked.connect(self._export_summary_by_year_csv)
        export_row.addWidget(self._export_summary_by_year_btn)
        self._export_summary_btn = QPushButton("Export Summary as CSV")
        self._export_summary_btn.setStyleSheet(_EXPORT_BTN_STYLE)
        self._export_summary_btn.setMinimumHeight(36)
        self._export_summary_btn.setEnabled(False)
        self._export_summary_btn.clicked.connect(self._export_summary_csv)
        export_row.addWidget(self._export_summary_btn)
        self._export_btn = QPushButton("Export Results as CSV")
        self._export_btn.setStyleSheet(_EXPORT_BTN_STYLE)
        self._export_btn.setMinimumHeight(36)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        export_row.addWidget(self._export_btn)
        root.addLayout(export_row)

    # ======================================================================
    # Tab builders
    # ======================================================================

    def _build_date_range_tab(self) -> QWidget:
        tab = QWidget()
        lay = QGridLayout(tab)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)

        lay.addWidget(_kpi_label("From:"), 0, 0)
        self._from_date = QDateEdit()
        self._from_date.setCalendarPopup(True)
        self._from_date.setDate(QDate.currentDate().addDays(-1))
        self._from_date.setDisplayFormat("dd/MM/yyyy")
        lay.addWidget(self._from_date, 0, 1)

        lay.addWidget(_kpi_label("To:"), 1, 0)
        self._to_date = QDateEdit()
        self._to_date.setCalendarPopup(True)
        self._to_date.setDate(QDate.currentDate().addDays(-1))
        self._to_date.setDisplayFormat("dd/MM/yyyy")
        lay.addWidget(self._to_date, 1, 1)

        hint = _kpi_label("Weather is fetched automatically\nfrom Open-Meteo.", color="#64748b")
        hint.setWordWrap(True)
        lay.addWidget(hint, 2, 0, 1, 2)
        lay.setRowStretch(3, 1)
        return tab

    def _build_full_year_tab(self) -> QWidget:
        tab = QWidget()
        lay = QGridLayout(tab)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)

        lay.addWidget(_kpi_label("Year:"), 0, 0)
        self._year_spin = QSpinBox()
        self._year_spin.setRange(2020, datetime.date.today().year)
        self._year_spin.setValue(datetime.date.today().year - 1)
        lay.addWidget(self._year_spin, 0, 1)

        hint = _kpi_label(
            "Runs every day that has price data\nin the loaded CSV.\n"
            "This may take several minutes.",
            color="#64748b",
        )
        hint.setWordWrap(True)
        lay.addWidget(hint, 1, 0, 1, 2)
        lay.setRowStretch(2, 1)
        return tab

    def _build_custom_csv_tab(self) -> QWidget:
        """Tab 3: upload a CSV — its date range defines what to analyse."""
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 10, 10, 10)

        row = QHBoxLayout()
        self._custom_csv_btn = QPushButton("Browse price CSV…")
        self._custom_csv_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        self._custom_csv_btn.clicked.connect(self._browse_custom_csv)
        row.addWidget(self._custom_csv_btn)
        self._custom_csv_lbl = QLabel("No file selected")
        self._custom_csv_lbl.setStyleSheet("color: #64748b;")
        row.addWidget(self._custom_csv_lbl, 1)
        lay.addLayout(row)

        self._custom_range_lbl = QLabel("")
        self._custom_range_lbl.setStyleSheet("color: #0f766e; font-weight: 700;")
        lay.addWidget(self._custom_range_lbl)

        hint = _kpi_label(
            "CSV must have a Timestamp column and a price column.\n"
            "Supported formats: tab- or semicolon-separated,\n"
            "EUR/kWh or EUR/MWh (auto-detected).",
            color="#64748b",
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch()
        return tab

    def _build_settings_panel(self) -> QGroupBox:
        grp = QGroupBox("Settings & Run")
        lay = QGridLayout(grp)
        lay.setSpacing(8)
        lay.setContentsMargins(10, 18, 10, 10)

        # Price CSV (shared by tabs 1 and 2)
        lay.addWidget(_kpi_label("Price source:"), 0, 0, 1, 2)

        lay.addWidget(_kpi_label("Price CSV (tabs 1 & 2):"), 1, 0, 1, 2)
        self._price_btn = QPushButton("Browse…")
        self._price_btn.clicked.connect(self._browse_price_csv)
        lay.addWidget(self._price_btn, 2, 0)
        self._price_lbl = QLabel("No file")
        self._price_lbl.setStyleSheet("color: #64748b; font-size: 9pt;")
        self._price_lbl.setWordWrap(True)
        lay.addWidget(self._price_lbl, 2, 1)

        # Note: base electrical load is read from Solver Config
        # (Manage Assets → Night load / Day load).  No manual input here.
        _note = QLabel(
            "Base electrical load is taken from Solver Config\n"
            "(Manage Assets → Night load / Day load)."
        )
        _note.setStyleSheet("color: #64748b; font-size: 9pt;")
        _note.setWordWrap(True)
        lay.addWidget(_note, 3, 0, 1, 3)

        lay.addWidget(QLabel(""), 4, 0)   # spacer row

        # Run button
        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setStyleSheet(RUN_ANALYSIS_BUTTON_STYLE)
        self._run_btn.setMinimumHeight(40)
        self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn, 5, 0, 1, 2)

        lay.setRowStretch(6, 1)
        return grp

    # ======================================================================
    # Slots — file browsing
    # ======================================================================

    def _restore_price_csv(self) -> None:
        """Re-load the last-used price CSV from saved config (silent on failure)."""
        try:
            saved = load_dashboard_config().get("analysis_price_csv", "")
            if saved and Path(saved).exists():
                self._price_df   = _parse_price_csv(saved)
                self._price_path = saved
                fname  = Path(saved).name
                n_days = len(set(self._price_df.index.date))
                self._price_lbl.setText(f"{fname}\n({n_days} days)")
        except Exception:
            pass

    def _browse_price_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Price CSV", "",
            "CSV files (*.csv *.txt);;All files (*)",
        )
        if not path:
            return
        try:
            self._price_df   = _parse_price_csv(path)
            self._price_path = path
            fname = Path(path).name
            n_days = len(set(self._price_df.index.date))
            self._price_lbl.setText(f"{fname}\n({n_days} days)")
            self._status_lbl.setText(
                f"Loaded {fname}: {n_days} days, "
                f"{self._price_df.index.min().date()} – "
                f"{self._price_df.index.max().date()}"
            )
            # Persist path so the next dialog open reloads it automatically
            try:
                _cfg = load_dashboard_config()
                _cfg["analysis_price_csv"] = path
                save_dashboard_config(_cfg)
            except Exception:
                pass
        except Exception as exc:
            QMessageBox.critical(self, "Parse error", f"Could not parse price CSV:\n{exc}")

    def _browse_custom_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Price CSV", "",
            "CSV files (*.csv *.txt);;All files (*)",
        )
        if not path:
            return
        try:
            self._custom_price_df   = _parse_price_csv(path)
            self._custom_price_path = path
            fname   = Path(path).name
            n_days  = len(set(self._custom_price_df.index.date))
            d_min   = self._custom_price_df.index.min().date()
            d_max   = self._custom_price_df.index.max().date()
            self._custom_csv_lbl.setText(fname)
            self._custom_range_lbl.setText(
                f"Detected {n_days} days:  {d_min} → {d_max}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Parse error", f"Could not parse price CSV:\n{exc}")

    # ======================================================================
    # Main run logic
    # ======================================================================

    def _run(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        # Load config
        try:
            self._cfg = load_mpc_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config error", f"Cannot load MPC config:\n{exc}")
            return

        tab = self._tabs.currentIndex()

        # Build list of days to simulate
        try:
            days_data = self._build_days_data(tab)
        except Exception as exc:
            QMessageBox.critical(self, "Input error", str(exc))
            return

        if not days_data:
            QMessageBox.warning(self, "No data", "No days with price data found for the selected period.")
            return

        # Clear previous results
        self._results = []
        self._clear_results_area()
        self._placeholder_lbl.hide()

        # Kick off worker
        self._worker = _AnalysisWorker(days_data, self._cfg)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.day_done_signal.connect(self._on_day_done)
        self._worker.error_signal.connect(lambda msg: self._status_lbl.setText(f"⚠ {msg}"))
        self._worker.finished_signal.connect(self._on_finished)
        n = len(days_data)
        self._progress.setRange(0, n)
        self._progress.setValue(0)
        self._progress.setFormat(f"%v / {n} day(s)")
        self._progress.show()
        self._abort_btn.show()
        self._run_btn.setEnabled(False)
        self._status_lbl.setText(f"Running analysis for {n} day(s)…")

        self._worker.start()

    def _build_days_data(self, tab_index: int) -> list[dict]:
        """Collect all (date, price, temperature, uv_index) records to simulate."""
        if tab_index == 0:       # Date Range
            qfrom = self._from_date.date().toPyDate()
            qto   = self._to_date.date().toPyDate()
            if qfrom > qto:
                raise ValueError("'From' date must be ≤ 'To' date.")
            dates = [qfrom + datetime.timedelta(days=i) for i in range((qto - qfrom).days + 1)]
            price_df = self._require_price_df()

        elif tab_index == 1:     # Full Year
            yr = self._year_spin.value()
            dates = [
                datetime.date(yr, 1, 1) + datetime.timedelta(days=i)
                for i in range(366)
                if (datetime.date(yr, 1, 1) + datetime.timedelta(days=i)).year == yr
            ]
            price_df = self._require_price_df()

        else:                    # Custom CSV
            if self._custom_price_df is None:
                raise ValueError("Please load a price CSV in the 'Custom CSV' tab first.")
            price_df = self._custom_price_df
            dates = sorted(set(price_df.index.date))

        # Filter to dates with price data
        valid_dates = [d for d in dates if _get_day_prices(price_df, d) is not None]
        if not valid_dates:
            return []

        # Fetch weather for the full range in one API call
        start_str = valid_dates[0].isoformat()
        end_str   = valid_dates[-1].isoformat()
        self._status_lbl.setText(
            f"Fetching weather {start_str} → {end_str}…"
        )
        # Force UI to update before blocking network call
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        weather = _fetch_weather_historical(
            DEFAULT_LATITUDE, DEFAULT_LONGITUDE,
            start_str, end_str,
        )

        # Build a dict keyed by date string for quick lookup
        weather_by_date: dict[str, dict] = {}
        if _PANDAS_OK:
            import pandas as pd
            ts_list  = weather["timestamps"]
            temp_arr = np.array(weather["temperature"])
            uv_arr   = np.array(weather["uv_index"])
            for d in valid_dates:
                d_str = d.isoformat()
                # slots for this day
                slots = [
                    i for i, t in enumerate(ts_list)
                    if isinstance(t, datetime.datetime) and t.date() == d
                ]
                if not slots:
                    continue
                s, e = slots[0], slots[-1] + 1
                t96 = temp_arr[s:e]
                u96 = uv_arr[s:e]
                if len(t96) < 96:
                    t96 = np.pad(t96, (0, 96 - len(t96)), "edge")
                    u96 = np.pad(u96, (0, 96 - len(u96)), "edge")
                weather_by_date[d_str] = {
                    "temperature": t96[:96],
                    "uv_index":    u96[:96],
                }
        else:
            # Simple fallback: repeat the full array sequentially
            temp_arr = np.array(weather["temperature"])
            uv_arr   = np.array(weather["uv_index"])
            for idx, d in enumerate(valid_dates):
                s = idx * 96
                e = s + 96
                weather_by_date[d.isoformat()] = {
                    "temperature": temp_arr[s:min(e, len(temp_arr))],
                    "uv_index":    uv_arr[s:min(e, len(uv_arr))],
                }

        # Assemble final list
        days_data = []
        for d in valid_dates:
            d_str = d.isoformat()
            price = _get_day_prices(price_df, d)
            if price is None:
                continue
            wx = weather_by_date.get(d_str, {})
            t96 = np.asarray(wx.get("temperature", np.full(96, 10.0)), dtype=float)
            u96 = np.asarray(wx.get("uv_index",    np.zeros(96)),     dtype=float)
            if len(t96) < 96:
                t96 = np.pad(t96, (0, 96 - len(t96)), "edge")
            if len(u96) < 96:
                u96 = np.pad(u96, (0, 96 - len(u96)), "edge")
            days_data.append({
                "date":        d_str,
                "price":       price,
                "temperature": t96[:96],
                "uv_index":    u96[:96],
            })

        return days_data

    def _require_price_df(self):
        if self._price_df is None:
            raise ValueError(
                "No price CSV loaded. Click 'Browse…' in the Settings panel to load one."
            )
        return self._price_df

    # ======================================================================
    # Worker signal handlers
    # ======================================================================

    def _on_progress(self, current: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(current)

    def _on_day_done(self, result: dict) -> None:
        self._results.append(result)

    def _on_finished(self, errors: list) -> None:
        self._progress.hide()
        self._abort_btn.hide()
        self._run_btn.setEnabled(True)

        if not self._results:
            if errors:
                # Show the first error prominently so the user knows what failed
                first = errors[0]
                self._status_lbl.setText(f"No results — {first}")
                QMessageBox.critical(
                    self, "Analysis failed",
                    f"Every day failed with an error.\nFirst error:\n\n{first}\n\n"
                    f"({len(errors)} total error(s))",
                )
            else:
                self._status_lbl.setText("No results produced.")
            return

        n = len(self._results)
        total_bl  = sum(r["baseline_cost_eur"] for r in self._results)
        total_mpc = sum(r["mpc_cost_eur"]      for r in self._results)
        total_sav = total_bl - total_mpc
        pct       = 100.0 * total_sav / max(total_bl, 1e-9)

        self._status_lbl.setText(
            f"Done — {n} day(s): baseline {total_bl:.2f} €, "
            f"MPC {total_mpc:.2f} €, saving {total_sav:.2f} € ({pct:.1f}%)"
        )

        self._render_results()
        self._export_btn.setEnabled(True)
        self._export_summary_btn.setEnabled(True)

        # Show "Export Summary by Year" button only when results span multiple years
        years = {r["date"][:4] for r in self._results}
        multi_year = len(years) > 1
        self._export_summary_by_year_btn.setVisible(multi_year)
        self._export_summary_by_year_btn.setEnabled(multi_year)

    # ======================================================================
    # Results rendering
    # ======================================================================

    def _clear_results_area(self) -> None:
        """Remove all widgets from the results layout (except placeholder)."""
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            if item.widget() and item.widget() is not self._placeholder_lbl:
                item.widget().deleteLater()
            elif item.layout():
                # nested layout — clear it
                pass

    def _render_results(self) -> None:
        self._clear_results_area()

        results = self._results
        n = len(results)
        total_bl  = sum(r["baseline_cost_eur"] for r in results)
        total_mpc = sum(r["mpc_cost_eur"]      for r in results)
        total_sav = total_bl - total_mpc
        pct_sav   = 100.0 * total_sav / max(total_bl, 1e-9)
        # Gross baseline (imports + gas, no PV export credit) gives a more
        # conservative "% of energy bill" metric. Without this, PV export
        # revenue shrinks the denominator and inflates the saving %.
        total_bl_gross = sum(r.get("baseline_gross_eur", r["baseline_cost_eur"])
                             for r in results)
        total_bl_export_rev = sum(r.get("baseline_export_revenue_eur", 0.0)
                                  for r in results)
        pct_sav_gross = 100.0 * total_sav / max(total_bl_gross, 1e-9)

        # --- Summary KPI block -------------------------------------------
        kpi_grp = QGroupBox("Summary")
        kg = QGridLayout(kpi_grp)
        kg.setSpacing(6)

        def _add_kpi_row(row: int, lbl: str, val: str, val_color: str = "#0f766e"):
            kg.addWidget(_kpi_label(lbl), row, 0)
            kg.addWidget(_val_label(val, val_color), row, 1)

        _add_kpi_row(0, "Days analysed",        str(n))
        _add_kpi_row(1, "Baseline cost",         f"{total_bl:.2f} €")
        _add_kpi_row(2, "MPC cost",              f"{total_mpc:.2f} €")
        _add_kpi_row(3, "Total saving",          f"{total_sav:.2f} €", "#0b3a6e")
        _add_kpi_row(4, "Saving %",              f"{pct_sav:.1f} %",   "#0b3a6e")
        self._results_layout.addWidget(kpi_grp)

        # --- Per-asset savings breakdown ----------------------------------
        asset_keys = [
            ("hp_eur",      "Heat Pump"),
            ("cooling_eur", "HP Cooling"),
            ("boiler_eur",  "Gas Boiler"),
            ("flex_eur",    "Flex Load"),
            ("battery_eur", "Battery"),
            ("chp_eur",     "CHP"),
            ("hw_eur",      "Hot Water"),
            ("pv_eur",      "PV"),
        ]
        asset_grp = QGroupBox("Savings by Asset")
        ag = QGridLayout(asset_grp)
        ag.setSpacing(6)
        ag.setColumnStretch(1, 1)
        ag.setColumnMinimumWidth(1, 200)

        for row_i, (key, label) in enumerate(asset_keys):
            val = sum(r["savings"].get(key, 0.0) for r in results)
            color = "#0f766e" if val >= 0 else "#dc2626"
            pct = 100.0 * val / max(total_bl, 1e-9)
            ag.addWidget(_kpi_label(label), row_i, 0)
            ag.addWidget(_val_label(f"{val:+.2f} € ({pct:+.1f}%)", color), row_i, 1)
        self._results_layout.addWidget(asset_grp)

        # --- Per-rule savings breakdown -----------------------------------
        # Tuple: (dict_key, display_label, sub_row)
        # sub_row=True → indented ↳ line under "Thermal building"
        rule_keys = [
            ("thermal_building_eur",   "Thermal building (HP + boiler)",      False),
            ("fuel_switching_eur",     "\u21b3 Fuel switching (HP vs boiler)", True),
            ("thermal_storage_eur",    "\u21b3 Thermal storage (pre-heat / coast)", True),
            ("hw_thermal_eur",         "Hot water pre-heating",                False),
            ("flex_shifting_eur",      "Flex load shifting",                   False),
            ("battery_arbitrage_eur",  "Battery arbitrage",                    False),
            ("chp_spark_eur",          "CHP spark spread",                     False),
            ("pv_selfconsumption_eur", "Solar self-consumption",               False),
            ("peak_shaving_eur",       "Capacity tariff / peak shaving",       False),
        ]
        rule_grp = QGroupBox("Savings by Rule")
        rg = QGridLayout(rule_grp)
        rg.setSpacing(6)
        rg.setColumnStretch(1, 1)
        rg.setColumnMinimumWidth(1, 200)

        for row_i, (key, label, sub_row) in enumerate(rule_keys):
            val = sum(r["savings"].get(key, 0.0) for r in results)
            color = "#0f766e" if val >= 0 else "#dc2626"
            pct = 100.0 * val / max(total_bl, 1e-9)
            lbl = _kpi_label(label, color="#64748b" if sub_row else "#1f2937")
            if sub_row:
                lbl.setContentsMargins(16, 0, 0, 0)
            rg.addWidget(lbl, row_i, 0)
            rg.addWidget(_val_label(f"{val:+.2f} € ({pct:+.1f}%)", color), row_i, 1)
        self._results_layout.addWidget(rule_grp)

        # --- Daily savings table (if multiple days) -----------------------
        if n > 1:
            daily_grp = QGroupBox("Daily Breakdown")
            dg_lay = QVBoxLayout(daily_grp)
            dg_lay.setSpacing(2)

            # Header
            hdr = QGridLayout()
            for ci, txt in enumerate(["Date", "Baseline €", "MPC €", "Saving €", "Saving %"]):
                lbl = _kpi_label(txt, bold=True)
                lbl.setAlignment(Qt.AlignmentFlag.AlignRight if ci > 0 else Qt.AlignmentFlag.AlignLeft)
                hdr.addWidget(lbl, 0, ci)
            dg_lay.addLayout(hdr)

            # Divider
            div = QFrame()
            div.setFrameShape(QFrame.Shape.HLine)
            div.setStyleSheet("color: #c7d7ed;")
            dg_lay.addWidget(div)

            for r in sorted(results, key=lambda x: x["date"]):
                bl_r   = r["baseline_cost_eur"]
                mpc_r  = r["mpc_cost_eur"]
                sav_r  = r["total_saving_eur"]
                pct_r  = 100.0 * sav_r / max(bl_r, 1e-9)
                row_lay = QGridLayout()
                color_r = "#0f766e" if sav_r >= 0 else "#dc2626"
                row_lay.addWidget(_kpi_label(r["date"]),               0, 0)
                row_lay.addWidget(_val_label(f"{bl_r:.2f}",  "#1f2937"), 0, 1)
                row_lay.addWidget(_val_label(f"{mpc_r:.2f}", "#1f2937"), 0, 2)
                row_lay.addWidget(_val_label(f"{sav_r:+.2f}", color_r), 0, 3)
                row_lay.addWidget(_val_label(f"{pct_r:.1f}%", color_r), 0, 4)
                dg_lay.addLayout(row_lay)

            self._results_layout.addWidget(daily_grp)

        # --- Charts -------------------------------------------------------
        if n == 1:
            self._render_day_chart(results[0])
            self._render_comparison_graphs(results[0])
            self._render_rt_graphs(results[0])
        elif n > 1:
            self._render_multi_day_chart(results)
            # Asset breakdown aggregated across ALL days in the range, so the
            # bar values match the "Savings by Asset" totals shown above.
            agg_savings: dict[str, float] = {}
            for r in results:
                for k, v in (r.get("savings") or {}).items():
                    try:
                        agg_savings[k] = agg_savings.get(k, 0.0) + float(v)
                    except (TypeError, ValueError):
                        continue
            agg_result = {
                "date": f"{results[0]['date']} \u2192 {results[-1]['date']}",
                "baseline_cost_eur": total_bl,
                "mpc_cost_eur":      total_mpc,
                "savings":           agg_savings,
            }
            self._render_comparison_graphs(
                agg_result,
                title_suffix=f" — {n} days (total)",
            )

        self._results_layout.addStretch()

    # ======================================================================
    # Charts (matplotlib, with QPainter fallback)
    # ======================================================================

    def _render_day_chart(self, result: dict) -> None:
        """Draw MPC vs baseline grid power + price overlay for a single day."""
        try:
            self._render_day_chart_mpl(result)
        except Exception:
            pass   # chart is optional; KPIs are already shown

    def _render_multi_day_chart(self, results: list[dict]) -> None:
        """Draw daily savings bar chart for multi-day runs."""
        try:
            self._render_multi_day_chart_mpl(results)
        except Exception:
            pass

    def _render_day_chart_mpl(self, result: dict) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        H  = len(result.get("plan_Pgrid", []))
        if H == 0:
            return
        dt = float(result.get("dt", 0.25))
        x  = np.arange(H)

        # X-axis: time labels every 4 h (16 × 15-min slots)
        xticks  = list(range(0, H, 16))
        xlabels = [f"{(k * int(dt * 60)) // 60:02d}:00" for k in xticks]

        def _a(key):
            return np.asarray(result.get(key, np.zeros(H)), dtype=float)[:H]

        pgrid_mpc = _a("plan_Pgrid")
        pgrid_bl  = _a("bl_Pgrid")
        prices    = _a("prices")
        ppv_mpc   = _a("plan_Ppv")
        pdis_mpc  = _a("plan_Pdis")
        php_mpc   = _a("plan_Php")
        pgas_mpc  = _a("plan_Pgas")
        pflex_mpc = _a("plan_Pflex")
        ptank_mpc = _a("plan_Ptank")
        pch_mpc   = _a("plan_Pch")
        pcool_mpc = _a("plan_Php_cool")

        # ── Shared style constants ────────────────────────────────────────
        BG_FIG  = "#eef3f9"
        BG_AX   = "#f8fbff"
        C_GRID  = "#dbe7f7"
        C_TEXT  = "#334155"
        C_SPINE = "#cbd5e1"

        def _style(ax, show_grid=True):
            ax.set_facecolor(BG_AX)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(C_SPINE)
            ax.spines["bottom"].set_color(C_SPINE)
            ax.tick_params(labelsize=8, colors=C_TEXT)
            ax.yaxis.label.set_color(C_TEXT)
            if show_grid:
                ax.grid(True, color=C_GRID, linewidth=0.7, linestyle="-", zorder=0)

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 9.5), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 2.0, 1.8]},
        )
        fig.patch.set_facecolor(BG_FIG)
        for ax in axes:
            _style(ax)

        # ── Panel 1: Grid power (MPC vs Baseline) + Price (right axis) ───
        ax1 = axes[0]
        ax1.fill_between(
            x, pgrid_bl, pgrid_mpc,
            where=pgrid_mpc < pgrid_bl,
            alpha=0.18, color="#16a34a", step="post",
        )
        ax1.step(x, pgrid_bl,  where="post", color="#94a3b8", linewidth=1.4,
                 linestyle="--", label="Baseline", alpha=0.9, zorder=3)
        ax1.step(x, pgrid_mpc, where="post", color="#2563eb", linewidth=2.0,
                 label="MPC optimised", zorder=4)
        ax1.set_ylabel("Grid draw [kW]", fontsize=8.5)
        ax1.set_title(
            f"MPC dispatch  —  {result['date']}",
            fontsize=10, fontweight="bold", color="#0b3a6e", pad=10,
        )
        # Price on twin right axis
        ax1b = ax1.twinx()
        ax1b.step(x, prices * 100, where="post", color="#f59e0b",
                  linewidth=1.1, linestyle=":", alpha=0.95, label="Price")
        ax1b.fill_between(x, 0, prices * 100, alpha=0.07, color="#f59e0b", step="post")
        ax1b.set_ylabel("Price [ct/kWh]", fontsize=8, color="#b45309")
        ax1b.tick_params(labelsize=7.5, colors="#b45309")
        ax1b.spines["top"].set_visible(False)
        ax1b.spines["right"].set_color(C_SPINE)
        # Combined legend with savings patch
        h1, l1   = ax1.get_legend_handles_labels()
        h1b, l1b = ax1b.get_legend_handles_labels()
        sav_patch = mpatches.Patch(color="#16a34a", alpha=0.4, label="Grid saving")
        ax1.legend(h1 + h1b + [sav_patch], l1 + l1b + ["Grid saving"],
                   fontsize=8, loc="upper right", framealpha=0.85)

        # ── Panel 2: Stacked asset dispatch + ambient temperature (right) ─
        ax2 = axes[1]
        asset_palette = [
            (ppv_mpc,   "#f59e0b", "PV output"),
            (pdis_mpc,  "#0ea5e9", "Battery discharge"),
            (php_mpc,   "#f97316", "Heat pump (heating)"),
            (pcool_mpc, "#38bdf8", "Heat pump (cooling)"),
            (pgas_mpc,  "#a78bfa", "Boiler (thermal)"),
            (pflex_mpc, "#22c55e", "Flex load"),
            (ptank_mpc, "#06b6d4", "Hot water heater"),
            (pch_mpc,   "#64748b", "Battery charge"),
        ]
        active = [(v, c, l) for v, c, l in asset_palette if v.sum() > 0.5]
        if active:
            ax2.stackplot(
                x,
                *[v for v, c, l in active],
                colors=[c for v, c, l in active],
                labels=[l for v, c, l in active],
                alpha=0.80, zorder=2,
            )
        ax2.set_ylabel("Asset power [kW]", fontsize=8.5)
        ax2.set_title(
            "Controllable asset dispatch (MPC decision)",
            fontsize=9, color=C_TEXT, style="italic",
        )
        # Ambient temperature on twin right axis
        temps = _a("temperature")
        if temps.sum() != 0 or temps.min() < -0.5:
            ax2b = ax2.twinx()
            ax2b.plot(x, temps, color="#64748b", linewidth=1.2, linestyle="--",
                      alpha=0.75, label="Ambient temp", zorder=5)
            ax2b.set_ylabel("Amb. temp [°C]", fontsize=8, color="#64748b")
            ax2b.tick_params(labelsize=7.5, colors="#64748b")
            ax2b.spines["top"].set_visible(False)
            ax2b.spines["right"].set_color(C_SPINE)
            h2, l2   = ax2.get_legend_handles_labels()
            h2b, l2b = ax2b.get_legend_handles_labels()
            ax2.legend(h2 + h2b, l2 + l2b,
                       fontsize=7.5, loc="upper right", ncol=2, framealpha=0.85)
        else:
            ax2.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.85)

        # ── Panel 3: Cumulative electricity cost over the day ─────────────
        ax3 = axes[2]
        cum_bl  = np.cumsum(pgrid_bl  * prices * dt)
        cum_mpc = np.cumsum(pgrid_mpc * prices * dt)
        ax3.fill_between(
            x, cum_mpc, cum_bl,
            where=cum_bl >= cum_mpc, alpha=0.20, color="#16a34a", step="post",
        )
        ax3.step(x, cum_bl,  where="post", color="#dc2626", linewidth=1.5,
                 linestyle="--", label="Baseline cost", alpha=0.9)
        ax3.step(x, cum_mpc, where="post", color="#2563eb", linewidth=2.0,
                 label="MPC cost")
        # Annotate final saving
        final_sav = float(cum_bl[-1] - cum_mpc[-1]) if len(cum_bl) else 0.0
        if abs(final_sav) > 0.001:
            mid_y = (float(cum_bl[-1]) + float(cum_mpc[-1])) / 2
            ax3.annotate(
                f"  Saving: {final_sav:+.2f} €",
                xy=(H - 1, float(cum_bl[-1])),
                xytext=(int(H * 0.55), mid_y),
                fontsize=8.5, color="#16a34a", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.1),
            )
        ax3.set_ylabel("Cumulative cost [€]", fontsize=8.5)
        ax3.set_xlabel("Time of day", fontsize=8.5, color=C_TEXT)
        ax3.legend(fontsize=8, loc="upper left", framealpha=0.85)
        ax3.set_title(
            "Cumulative electricity cost — savings build up over the day",
            fontsize=9, color=C_TEXT, style="italic",
        )

        # ── X-axis ticks ──────────────────────────────────────────────────
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels(xlabels, fontsize=8.5)

        fig.tight_layout(pad=1.2, h_pad=1.0)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        plt.close(fig)
        buf.seek(0)

        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())

        chart_lbl = QLabel()
        chart_lbl.setPixmap(pixmap)
        chart_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        chart_grp = QGroupBox(f"Dispatch Overview — {result['date']}")
        ch_lay = QVBoxLayout(chart_grp)
        ch_lay.addWidget(chart_lbl)
        self._results_layout.addWidget(chart_grp)

    def _render_comparison_graphs(self, result: dict, title_suffix: str = "") -> None:
        """Horizontal bar chart of per-asset savings breakdown."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        try:
            savings  = result.get("savings", {})
            total_bl = max(float(result.get("baseline_cost_eur", 1.0)), 1e-6)

            # (saving_key, bl_cost_key, label, color)
            asset_items = [
                ("pv_eur",      "bl_pv_saving",    "PV self-consumption (load shifted to solar peak)", "#f59e0b"),
                ("battery_eur", "bl_battery_cost", "Battery (buy low/sell high)",  "#0ea5e9"),
                ("hp_eur",      "bl_hp_cost",      "Heat pump (run at right time)", "#f97316"),
                ("cooling_eur", "bl_cooling_cost", "HP cooling (shift to off-peak)", "#38bdf8"),
                ("boiler_eur",  "bl_boiler_cost",  "Boiler (gas vs electricity)",   "#a78bfa"),
                ("flex_eur",    "bl_flex_cost",    "Flex load (time shift)",        "#22c55e"),
                ("hw_eur",      "bl_hw_cost",      "Hot water (pre-heat off-peak)", "#06b6d4"),
                ("chp_eur",     "bl_chp_income",   "CHP (generate at peak price)",  "#8b5cf6"),
            ]
            labels = [l for _, _, l, _ in asset_items]
            values = [savings.get(sk, 0.0) for sk, _, _, _ in asset_items]

            if not any(abs(v) > 0.001 for v in values):
                return

            max_abs    = max(abs(v) for v in values) or 1.0
            bar_colors = [
                "#16a34a" if v > 0.001 else "#dc2626" if v < -0.001 else "#94a3b8"
                for v in values
            ]

            BG_FIG  = "#eef3f9"
            BG_AX   = "#f8fbff"
            C_SPINE = "#cbd5e1"
            C_TEXT  = "#334155"

            fig, ax = plt.subplots(figsize=(9, 3.6))
            fig.patch.set_facecolor(BG_FIG)
            ax.set_facecolor(BG_AX)

            bars = ax.barh(labels, values, color=bar_colors, alpha=0.82, height=0.55)
            ax.axvline(0, color="#64748b", linewidth=0.9)

            for bar, (sk, bk, _, _), val in zip(bars, asset_items, values):
                if abs(val) > 0.001:
                    # % relative to total baseline cost — gives a consistent
                    # cross-asset comparison ("CHP earns X% of total bill", etc.)
                    pct = 100.0 * val / total_bl
                    pct_label = f"{pct:+.1f}% of bl."
                    ha  = "left"  if val >= 0 else "right"
                    off = 0.02 * max_abs
                    ax.text(
                        val + (off if val >= 0 else -off),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.2f} €  ({pct_label})",
                        va="center", ha=ha, fontsize=8,
                        color="#0b3a6e" if val >= 0 else "#dc2626",
                        fontweight="bold",
                    )

            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(C_SPINE)
            ax.spines["left"].set_color(C_SPINE)
            ax.grid(axis="x", color="#dbe7f7", linewidth=0.7)
            ax.tick_params(labelsize=8.5, colors=C_TEXT)
            ax.set_xlabel("Saving [€]", fontsize=8.5, color=C_TEXT)
            ax.set_title(
                f"Savings breakdown by asset{title_suffix}",
                fontsize=10, fontweight="bold", color="#0b3a6e",
            )

            fig.tight_layout(pad=0.9)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)

            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())

            sav_lbl = QLabel()
            sav_lbl.setPixmap(pixmap)
            sav_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            sav_grp = QGroupBox(f"Asset Savings Breakdown{title_suffix}")
            s_lay = QVBoxLayout(sav_grp)
            s_lay.addWidget(sav_lbl)
            self._results_layout.addWidget(sav_grp)

        except Exception as exc:
            logger.debug("_render_comparison_graphs failed: %s", exc)

    def _render_multi_day_chart_mpl(self, results: list[dict]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        sorted_r = sorted(results, key=lambda x: x["date"])
        dates    = [r["date"] for r in sorted_r]
        n        = len(dates)
        xs       = np.arange(n)
        savings  = np.array([r["total_saving_eur"] for r in sorted_r])
        cum      = np.cumsum(savings)

        asset_palette = [
            ("pv_eur",      "PV",        "#f5c400"),   # golden yellow
            ("battery_eur", "Battery",   "#0ea5e9"),   # sky blue
            ("hp_eur",      "Heat pump", "#e63946"),   # red
            ("cooling_eur", "HP cooling","#38bdf8"),   # light cyan
            ("boiler_eur",  "Boiler",    "#ff7f0e"),   # vivid orange
            ("flex_eur",    "Flex load", "#2ca02c"),   # forest green
            ("hw_eur",      "HW tank",   "#17becf"),   # teal/cyan
            ("chp_eur",     "CHP",       "#9467bd"),   # purple
        ]

        BG_FIG  = "#eef3f9"
        BG_AX   = "#f8fbff"
        C_GRID  = "#dbe7f7"
        C_TEXT  = "#334155"
        C_SPINE = "#cbd5e1"

        def _style(ax):
            ax.set_facecolor(BG_AX)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.spines["left"].set_color(C_SPINE)
            ax.spines["bottom"].set_color(C_SPINE)
            ax.tick_params(labelsize=8, colors=C_TEXT)
            ax.yaxis.label.set_color(C_TEXT)

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 10),
            gridspec_kw={"height_ratios": [1.6, 2.2, 1.6]},
        )
        fig.patch.set_facecolor(BG_FIG)
        for ax in axes:
            _style(ax)

        step  = max(1, n // 12)
        bar_w = max(0.3, min(0.8, 15.0 / max(n, 1)))

        # ── Panel 1: Daily savings bars ──────────────────────────────────
        ax1 = axes[0]
        bar_colors = ["#16a34a" if s >= 0 else "#dc2626" for s in savings]
        ax1.bar(xs, savings, color=bar_colors, alpha=0.82, width=bar_w, zorder=2)
        ax1.axhline(0, color="#64748b", linewidth=0.8)
        ax1.set_ylabel("Daily saving [€]", fontsize=8.5)
        ax1.set_title(
            f"MPC vs baseline  —  {n} days  |  Total: {savings.sum():+.2f} €",
            fontsize=10, fontweight="bold", color="#0b3a6e", pad=8,
        )
        ax1.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
        ax1.set_xticks(xs[::step])
        ax1.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=7.5)

        # ── Panel 2: Stacked asset savings per day ───────────────────────
        ax2 = axes[1]
        bottom_pos = np.zeros(n)
        bottom_neg = np.zeros(n)
        for key, label, color in asset_palette:
            vals = np.array([r.get("savings", {}).get(key, 0.0) for r in sorted_r])
            pos  = np.maximum(vals, 0)
            neg  = np.minimum(vals, 0)
            lbl_used = False
            if pos.sum() > 0.01:
                ax2.bar(xs, pos, bottom=bottom_pos, color=color, alpha=0.78,
                        width=bar_w, label=label, zorder=2)
                bottom_pos += pos
                lbl_used = True
            if neg.sum() < -0.01:
                ax2.bar(xs, neg, bottom=bottom_neg, color=color, alpha=0.78,
                        width=bar_w, label=(None if lbl_used else label), zorder=2)
                bottom_neg += neg
        ax2.axhline(0, color="#64748b", linewidth=0.8)
        ax2.set_ylabel("Asset saving [€/day]", fontsize=8.5)
        ax2.set_title(
            "Which assets save on each day  (stacked contribution)",
            fontsize=9, color=C_TEXT, style="italic",
        )
        ax2.legend(fontsize=7.5, loc="upper right", ncol=3, framealpha=0.85)
        ax2.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
        ax2.set_xticks(xs[::step])
        ax2.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=7.5)

        # ── Panel 3: Cumulative savings ──────────────────────────────────
        ax3 = axes[2]
        ax3.fill_between(xs, 0, cum, where=cum >= 0, alpha=0.18,
                         color="#16a34a", step="post")
        ax3.fill_between(xs, 0, cum, where=cum < 0,  alpha=0.18,
                         color="#dc2626", step="post")
        ax3.step(xs, cum, where="post", color="#0f766e", linewidth=2.0)
        ax3.axhline(0, color="#64748b", linewidth=0.8)
        if len(cum):
            ax3.annotate(
                f"Total: {float(cum[-1]):+.2f} €",
                xy=(n - 1, float(cum[-1])),
                xytext=(max(0, int(n * 0.55)),
                        float(cum[-1]) * 0.55 if abs(float(cum[-1])) > 0.5 else 0.5),
                fontsize=8.5, color="#0f766e", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.1),
            )
        ax3.set_ylabel("Cumulative saving [€]", fontsize=8.5)
        ax3.set_xlabel("Date", fontsize=8.5, color=C_TEXT)
        ax3.set_title(
            "Cumulative savings over the period",
            fontsize=9, color=C_TEXT, style="italic",
        )
        ax3.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
        ax3.set_xticks(xs[::step])
        ax3.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=7.5)

        fig.tight_layout(pad=1.2, h_pad=1.2)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        plt.close(fig)
        buf.seek(0)

        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())

        chart_lbl = QLabel()
        chart_lbl.setPixmap(pixmap)
        chart_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        chart_grp = QGroupBox(f"Multi-Day Analysis — {n} days")
        ch_lay = QVBoxLayout(chart_grp)
        ch_lay.addWidget(chart_lbl)
        self._results_layout.addWidget(chart_grp)

    def _render_rt_graphs(self, result: dict) -> None:
        """Render price / solar / thermal graphs identical to the real-time dashboard."""
        from PyQt6.QtGui import QPixmap

        H = len(result.get("plan_Pgrid", []))
        if H == 0:
            return

        dt = float(result.get("dt", 0.25))

        # Time labels
        end_h  = H * dt
        end_hh = int(end_h) % 24
        end_mm = int((end_h % 1) * 60)
        start_label = "00:00"
        end_label   = f"{end_hh:02d}:{end_mm:02d}"

        def _a(key, default=0.0):
            arr = result.get(key) or []
            out = list(arr)[:H]
            if len(out) < H:
                out += [default] * (H - len(out))
            return out

        prices_mwh  = [p * 1000.0 for p in _a("prices")]   # EUR/kWh → EUR/MWh
        pgrid_mpc   = _a("plan_Pgrid")
        ppv_mpc     = _a("plan_Ppv")
        pchp_mpc    = _a("plan_Pchp")
        qchp_mpc    = _a("plan_Qchp")
        temps       = _a("temperature", 10.0)
        bld_temps   = _a("plan_Tbuilding", float("nan"))
        php_mpc     = _a("plan_Php")
        pgas_mpc    = _a("plan_Pgas")
        pcool_mpc   = _a("plan_Php_cool")

        # Comfort-band schedules from MPC config
        cfg = self._cfg
        tmin_sched: list[float] = []
        tmax_sched: list[float] = []
        for k in range(H):
            _hour = (k * dt) % 24.0
            if cfg and cfg.use_night_setback and (
                _hour >= cfg.night_start_h or _hour < cfg.night_end_h
            ):
                tmin_sched.append(cfg.T_set_night_c)
                tmax_sched.append(cfg.T_cool_night_c)
            else:
                tmin_sched.append(cfg.Tmin_c if cfg else 20.0)
                tmax_sched.append(cfg.Tmax_c if cfg else 23.0)

        setpoint = cfg.Tmin_c if cfg else 21.0
        W = 1020

        grp = QGroupBox(f"Real-time graphs — {result['date']}")
        grp_lay = QVBoxLayout(grp)

        # Price + grid overlay (+ CHP electrical output)
        price_lbl = QLabel()
        price_overlays = [{"array": pgrid_mpc, "color": "#f97316", "label": "MPC Grid [kW]"}]
        if any(v > 0.1 for v in pchp_mpc):
            price_overlays.append({"array": pchp_mpc, "color": "#8b5cf6", "label": "CHP Elec [kW]"})
        price_lbl.setPixmap(draw_price_graph(
            prices_mwh, start_label, end_label, None, None, W, 240,
            overlays=price_overlays,
        ))
        price_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grp_lay.addWidget(price_lbl)

        # Solar / PV
        solar_lbl = QLabel()
        solar_lbl.setPixmap(draw_solar_graph(ppv_mpc, start_label, end_label, None, None, W, 200))
        solar_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grp_lay.addWidget(solar_lbl)

        # Thermal: outdoor + building temp + comfort band + HP/boiler/CHP heat
        thermal_lbl = QLabel()
        thermal_lbl.setPixmap(draw_thermal_graph(
            temps, bld_temps, setpoint,
            php_mpc, pgas_mpc, qchp_mpc,
            start_label, end_label, None, None, W, 280,
            tmin_schedule=tmin_sched,
            tmax_schedule=tmax_sched,
            cool_hp_kw=pcool_mpc if any(v > 1e-6 for v in pcool_mpc) else None,
        ))
        thermal_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grp_lay.addWidget(thermal_lbl)

        self._results_layout.addWidget(grp)

    # ======================================================================
    # Export
    # ======================================================================

    def _export_csv(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "mpc_analysis_results.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        def _fmt(v: float) -> str:
            """Format a float with comma decimal separator for Excel (BE/NL locale)."""
            return f"{v:.4f}".replace(".", ",")

        def _fmt2(v: float) -> str:
            return f"{v:.2f}".replace(".", ",")

        try:
            # Use semicolon delimiter + comma decimal so Excel (BE/NL locale)
            # opens the file directly without an import wizard.
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                # Header
                writer.writerow([
                    "Date",
                    "Baseline cost (€)", "MPC cost (€)", "Total saving (€)", "Saving (%)",
                    "HP saving (€)", "HP cooling saving (€)", "Boiler saving (€)", "Flex saving (€)",
                    "Battery saving (€)", "CHP saving (€)", "HW saving (€)", "PV saving (€)",
                    "Thermal building (€)", "Fuel switching (€)", "Thermal storage (€)",
                    "HW thermal (€)", "Flex shifting (€)",
                    "Battery arbitrage (€)", "CHP spark (€)",
                    "PV self-consumption (€)", "Peak shaving (€)",
                ])
                for r in sorted(self._results, key=lambda x: x["date"]):
                    bl_r  = r["baseline_cost_eur"]
                    mpc_r = r["mpc_cost_eur"]
                    sav_r = r["total_saving_eur"]
                    pct_r = 100.0 * sav_r / max(bl_r, 1e-9)
                    s     = r.get("savings", {})
                    writer.writerow([
                        r["date"],
                        _fmt(bl_r),  _fmt(mpc_r),
                        _fmt(sav_r), _fmt2(pct_r),
                        _fmt(s.get("hp_eur",      0.0)),
                        _fmt(s.get("cooling_eur", 0.0)),
                        _fmt(s.get("boiler_eur",  0.0)),
                        _fmt(s.get("flex_eur",    0.0)),
                        _fmt(s.get("battery_eur", 0.0)),
                        _fmt(s.get("chp_eur",     0.0)),
                        _fmt(s.get("hw_eur",      0.0)),
                        _fmt(s.get("pv_eur",      0.0)),
                        _fmt(s.get("thermal_building_eur",   0.0)),
                        _fmt(s.get("fuel_switching_eur",     0.0)),
                        _fmt(s.get("thermal_storage_eur",    0.0)),
                        _fmt(s.get("hw_thermal_eur",         0.0)),
                        _fmt(s.get("flex_shifting_eur",      0.0)),
                        _fmt(s.get("battery_arbitrage_eur",  0.0)),
                        _fmt(s.get("chp_spark_eur",          0.0)),
                        _fmt(s.get("pv_selfconsumption_eur", 0.0)),
                        _fmt(s.get("peak_shaving_eur",       0.0)),
                    ])
            self._status_lbl.setText(f"Exported to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Could not write CSV:\n{exc}")

    def _export_summary_csv(self) -> None:
        """Export aggregated summary, savings by asset, and savings by rule.

        Mirrors the totals shown in the on-screen results panel — a compact
        view suitable for pasting into reports / the thesis.
        """
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary CSV", "mpc_analysis_summary.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        def _fmt(v: float) -> str:
            return f"{v:.4f}".replace(".", ",")

        def _fmt2(v: float) -> str:
            return f"{v:.2f}".replace(".", ",")

        results = self._results
        n = len(results)
        total_bl  = sum(r["baseline_cost_eur"] for r in results)
        total_mpc = sum(r["mpc_cost_eur"]      for r in results)
        total_sav = total_bl - total_mpc
        pct_sav   = 100.0 * total_sav / max(total_bl, 1e-9)

        asset_keys = [
            ("hp_eur",      "Heat Pump"),
            ("cooling_eur", "HP Cooling"),
            ("boiler_eur",  "Gas Boiler"),
            ("flex_eur",    "Flex Load"),
            ("battery_eur", "Battery"),
            ("chp_eur",     "CHP"),
            ("hw_eur",      "Hot Water"),
            ("pv_eur",      "PV"),
        ]
        rule_keys = [
            ("thermal_building_eur",   "Thermal building (HP + boiler)"),
            ("fuel_switching_eur",     "  Fuel switching (HP vs boiler)"),
            ("thermal_storage_eur",    "  Thermal storage (pre-heat / coast)"),
            ("hw_thermal_eur",         "Hot water pre-heating"),
            ("flex_shifting_eur",      "Flex load shifting"),
            ("battery_arbitrage_eur",  "Battery arbitrage"),
            ("chp_spark_eur",          "CHP spark spread"),
            ("pv_selfconsumption_eur", "Solar self-consumption"),
            ("peak_shaving_eur",       "Capacity tariff / peak shaving"),
        ]

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")

                # ── Summary ─────────────────────────────────────────────
                writer.writerow(["Summary"])
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Days analysed", n])
                writer.writerow(["Baseline cost (€)", _fmt2(total_bl)])
                writer.writerow(["MPC cost (€)",      _fmt2(total_mpc)])
                writer.writerow(["Total saving (€)",  _fmt2(total_sav)])
                writer.writerow(["Saving (%)",        _fmt2(pct_sav)])
                writer.writerow([])

                # ── Savings by Asset ────────────────────────────────────
                writer.writerow(["Savings by Asset"])
                writer.writerow(["Asset", "Saving (€)", "Saving (% of baseline)"])
                for key, label in asset_keys:
                    val = sum(r["savings"].get(key, 0.0) for r in results)
                    pct = 100.0 * val / max(total_bl, 1e-9)
                    writer.writerow([label, _fmt(val), _fmt2(pct)])
                writer.writerow([])

                # ── Savings by Rule ─────────────────────────────────────
                writer.writerow(["Savings by Rule"])
                writer.writerow(["Rule", "Saving (€)", "Saving (% of baseline)"])
                for key, label in rule_keys:
                    val = sum(r["savings"].get(key, 0.0) for r in results)
                    pct = 100.0 * val / max(total_bl, 1e-9)
                    writer.writerow([label, _fmt(val), _fmt2(pct)])

            self._status_lbl.setText(f"Exported summary to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Could not write CSV:\n{exc}")

    # ======================================================================
    # Abort
    # ======================================================================

    def _export_summary_by_year_csv(self) -> None:
        """Export a side-by-side summary grouped by year.

        Produces the same structure as the full-year summary but with one
        column pair (Saving €, Saving %) per year, so years can be compared
        directly in Excel / Calc.
        """
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary by Year CSV", "mpc_summary_by_year.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        def _fmt(v: float) -> str:
            return f"{v:.4f}".replace(".", ",")

        def _fmt2(v: float) -> str:
            return f"{v:.2f}".replace(".", ",")

        # Group results by year (string key like "2020")
        from collections import defaultdict
        by_year: dict[str, list] = defaultdict(list)
        for r in self._results:
            by_year[r["date"][:4]].append(r)
        years = sorted(by_year.keys())

        asset_keys = [
            ("hp_eur",      "Heat Pump"),
            ("cooling_eur", "HP Cooling"),
            ("boiler_eur",  "Gas Boiler"),
            ("flex_eur",    "Flex Load"),
            ("battery_eur", "Battery"),
            ("chp_eur",     "CHP"),
            ("hw_eur",      "Hot Water"),
            ("pv_eur",      "PV"),
        ]
        rule_keys = [
            ("thermal_building_eur",   "Thermal building (HP + boiler)"),
            ("fuel_switching_eur",     "  Fuel switching (HP vs boiler)"),
            ("thermal_storage_eur",    "  Thermal storage (pre-heat / coast)"),
            ("hw_thermal_eur",         "Hot water pre-heating"),
            ("flex_shifting_eur",      "Flex load shifting"),
            ("battery_arbitrage_eur",  "Battery arbitrage"),
            ("chp_spark_eur",          "CHP spark spread"),
            ("pv_selfconsumption_eur", "Solar self-consumption"),
            ("peak_shaving_eur",       "Capacity tariff / peak shaving"),
        ]

        # Pre-compute per-year totals
        yr_n:    dict[str, int]   = {}
        yr_bl:   dict[str, float] = {}
        yr_mpc:  dict[str, float] = {}
        yr_sav:  dict[str, float] = {}
        yr_pct:  dict[str, float] = {}
        for yr in years:
            rr = by_year[yr]
            bl  = sum(r["baseline_cost_eur"] for r in rr)
            mpc = sum(r["mpc_cost_eur"]      for r in rr)
            sav = bl - mpc
            yr_n[yr]   = len(rr)
            yr_bl[yr]  = bl
            yr_mpc[yr] = mpc
            yr_sav[yr] = sav
            yr_pct[yr] = 100.0 * sav / max(bl, 1e-9)

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")

                # ── Header row: year names (2 columns each) ─────────────
                header = [""]
                for yr in years:
                    header += [yr, ""]
                writer.writerow(header)

                # ── Summary block ────────────────────────────────────────
                def _summary_row(label: str, values: list[str]) -> list[str]:
                    row = [label]
                    for v in values:
                        row += [v, ""]
                    return row

                writer.writerow(_summary_row("Days analysed",    [str(yr_n[yr])          for yr in years]))
                writer.writerow(_summary_row("Baseline cost (€)", [_fmt2(yr_bl[yr])       for yr in years]))
                writer.writerow(_summary_row("MPC cost (€)",      [_fmt2(yr_mpc[yr])      for yr in years]))
                writer.writerow(_summary_row("Total saving (€)",  [_fmt2(yr_sav[yr])      for yr in years]))
                writer.writerow(_summary_row("Saving (%)",        [_fmt2(yr_pct[yr])      for yr in years]))
                writer.writerow([])

                # ── Savings by Asset ─────────────────────────────────────
                # Sub-header: "Saving (€)" / "Saving (%)" per year
                sub_header = [""]
                for yr in years:
                    sub_header += ["Saving (€)", "Saving (%)"]
                writer.writerow(sub_header)

                for key, label in asset_keys:
                    row = [label]
                    for yr in years:
                        val = sum(r["savings"].get(key, 0.0) for r in by_year[yr])
                        pct = 100.0 * val / max(yr_bl[yr], 1e-9)
                        row += [_fmt(val), _fmt2(pct)]
                    writer.writerow(row)
                writer.writerow([])

                # ── Savings by Rule ──────────────────────────────────────
                for key, label in rule_keys:
                    row = [label]
                    for yr in years:
                        val = sum(r["savings"].get(key, 0.0) for r in by_year[yr])
                        pct = 100.0 * val / max(yr_bl[yr], 1e-9)
                        row += [_fmt(val), _fmt2(pct)]
                    writer.writerow(row)

            self._status_lbl.setText(f"Exported summary by year to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Could not write CSV:\n{exc}")

    # ======================================================================
    # Abort
    # ======================================================================

    def _abort_run(self) -> None:
        if self._worker:
            self._worker.abort()
        self._status_lbl.setText("Aborted by user.")
        self._progress.hide()
        self._abort_btn.hide()
        self._run_btn.setEnabled(True)
