"""
mpc_lp.py — Model Predictive Control LP/MILP Solver
====================================================

Solves the building energy management problem over a receding horizon H.

Decision variables per time step k = 0 … H-1
---------------------------------------------
  Pgrid[k]      grid import power            [kW]
  Php[k]        heat pump thermal output     [kW]
  Pgas[k]       gas boiler thermal output    [kW]
  Pch[k]        battery charging power       [kW]
  Pdis[k]       battery discharging power    [kW]
  SOC[k]        battery state of charge      [kWh]  (H+1 values)
  Pflex[k]      flexible / shiftable load    [kW]
  Ppv[k]        PV power actually used       [kW]
  Ptank[k]      hot water heater power       [kW]
  Ttank[k]      hot water tank temperature   [°C]   (H+1 values)
  Fchp[k]       CHP gas flow rate            [m³/h]
  Pchp[k]       CHP electrical output        [kW]   (derived from Fchp)
  Qchp[k]       CHP thermal output           [kW]   (derived from Fchp)
  zchp[k]       CHP on/off binary            {0,1}
  ychp[k]       CHP startup binary           {0,1}
  Tbuilding[k]  building temperature         [°C]   (H+1 values)

Precomputed parameter per step
-------------------------------
  COP[k] = COP0 × max(COP_min, 1 − cop_alpha × (Tamb[k] − T0))

Electrical power balance (kW)
------------------------------
  Pgrid[k] + Ppv[k] + Pchp[k] + Pdis[k]
      = Pload[k] + Php[k]/COP[k] + Ptank[k] + Pflex[k] + Pch[k]

Building thermal dynamics (dt in hours, Cth in kWh/°C, UA in kW/°C)
---------------------------------------------------------------------
  Tbuilding[k+1] = (1 − UA·dt/Cth)·Tbuilding[k]
                 + (dt/Cth)·(Php[k] + Pgas[k] + Qchp[k])
                 + (UA·dt/Cth)·Tamb[k]

Battery dynamics
----------------
  SOC[k+1] = SOC[k] + (η_ch·Pch[k] − Pdis[k]/η_dis)·dt

Hot water tank dynamics (Ctank = volume·1.163e-3 kWh/°C)
---------------------------------------------------------
  Ttank[k+1] = Ttank[k] + (dt/Ctank)·(Ptank[k] − heat_loss_kw)

CHP (bilinear → MILP via big-M)
---------------------------------
  Pchp[k] = Fchp[k] · HV · η_elec
  Qchp[k] = Fchp[k] · HV · η_heat
  0 ≤ Fchp[k] ≤ Fchp_max · zchp[k]
  ychp[k] ≥ zchp[k] − zchp[k−1]    (startup detection)

Objective
---------
  min Σ_k dt·[(λ_e[k]+fee)·Pgrid[k]
              + λ_gas_chp·Fchp[k]
              + (λ_gas_boiler/η_boiler)·Pgas[k]]
      + startup_cost·Σ_k ychp[k]
      + w_peak·Σ_k slack_peak[k]²
      + w_comfort·Σ_k slack_Tmin[k]²
      + w_tank·Σ_k slack_tank[k]²
      + w_bat_end·(SOC[H]−SOC_end)²          [if battery enabled]
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn(
        "cvxpy not found — MPC solver will use rule-based fallback.\n"
        "Install with: pip install cvxpy",
        stacklevel=2,
    )

logger = logging.getLogger(__name__)


def _compute_flex_active_mask(
    t_start_str: str,
    t_end_str: str,
    H: int,
    dt: float,
    start_hour: float | None = None,
    start_weekday: int | None = None,
    active_days: list | None = None,
) -> np.ndarray:
    """Return a per-step boolean mask of when a flex load may consume power.

    Step ``k`` corresponds to wall-clock ``start_hour + k*dt`` (hours past
    midnight, wrapped mod 24).  When ``start_hour`` is None the current
    wall-clock time is used instead (legacy behaviour).  Supports overnight
    windows (start > end).  Invalid time strings → all-True (no restriction).

    Day-of-week filter: when ``active_days`` is a non-empty list of ints in
    {0..6} (Mon=0 … Sun=6) AND ``start_weekday`` is known, steps whose
    wall-clock weekday is not in the set are masked out.  When either is
    missing the day filter is skipped (all days allowed).
    """
    import datetime as _dt
    mask = np.ones(H, dtype=bool)
    try:
        t_on  = _dt.time.fromisoformat(t_start_str)
        t_off = _dt.time.fromisoformat(t_end_str)
        on_min  = t_on.hour  * 60 + t_on.minute
        off_min = t_off.hour * 60 + t_off.minute
        if start_hour is None:
            _now = _dt.datetime.now()
            base_min = _now.hour * 60 + _now.minute
            if start_weekday is None:
                start_weekday = _now.weekday()
        else:
            base_min = int(round(float(start_hour) * 60.0)) % (24 * 60)
        step_min = dt * 60.0
        day_min  = 24 * 60
        # Normalise active_days; None / empty list → no day restriction.
        if active_days:
            _active_set = {int(d) % 7 for d in active_days}
        else:
            _active_set = None
        for k in range(H):
            total_min = base_min + k * step_min
            m = int(total_min) % day_min
            if on_min <= off_min:
                ok = on_min <= m <= off_min
            else:                     # overnight (e.g. 22:00–06:00)
                ok = m >= on_min or m <= off_min
            if ok and _active_set is not None and start_weekday is not None:
                wd = (int(start_weekday) + int(total_min // day_min)) % 7
                if wd not in _active_set:
                    ok = False
            mask[k] = ok
    except ValueError:
        pass
    return mask


# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

@dataclass
class MPCConfig:
    """
    All physical and cost parameters for the MPC LP.

    Construct via MPCConfig.from_dict(raw["mpc"]) to load from
    dashboard_config.json, or use defaults directly.

    All efficiencies and capacity limits are configurable through the
    MPC Settings dialog (mpc_config_dialog.py).
    """

    # ── Optimisation horizon ────────────────────────────────────────────────
    horizon_steps: int   = 96      # Number of steps (e.g. 96 × 15 min = 24 h)
    dt_hours:      float = 0.25    # Step duration [h]

    # ── Grid ────────────────────────────────────────────────────────────────
    Pgrid_max_kw:       float = 500.0   # Grid import power limit [kW]
    fee_eur_kwh:        float = 0.05    # Fixed electricity cost (distribution, taxes, levies) [€/kWh]
    cap_tariff_Plim_kw:   float = 0.0   # Capacity tariff: power limit P_lim [kW] (0 = disabled)
    cap_tariff_epsilon_l: float = 0.0   # Capacity tariff: quadratic penalty ε_L [€/kW²]

    # ── Battery ─────────────────────────────────────────────────────────────
    bat_enabled:   bool  = False
    SOC_cap_kwh:   float = 100.0   # Usable capacity [kWh]
    SOC_min_kwh:   float =  10.0   # Safety reserve [kWh]
    SOC_init_kwh:  float =  50.0   # Initial SOC [kWh]
    SOC_end_kwh:   float =  50.0   # Soft end-of-horizon SOC target [kWh]
    Pch_max_kw:    float =  25.0   # Max charging power [kW]
    Pdis_max_kw:   float =  25.0   # Max discharging power [kW]
    eta_ch:        float =  0.95   # Charging efficiency
    eta_dis:       float =  0.95   # Discharging efficiency
    bat_ramp_pct:  float = 100.0   # Max charge/discharge ramp [% of max per step]

    # ── Heat pump ───────────────────────────────────────────────────────────
    hp_enabled:    bool  = True
    Php_max_kw:    float = 50.0    # Max thermal output [kW]
    COP0:          float =  4.0    # Nominal COP at reference temperature T0
    T0_c:          float =  7.0    # Reference ambient temperature [°C]
    cop_alpha:     float =  0.02   # COP degradation per °C above T0
    COP_min:       float =  1.0    # Minimum allowable COP (safety clamp)
    hp_ramp_pct:   float = 100.0   # Max heat pump ramp [% of Php_max_kw per step]
    # Cooling (reverse-cycle of the heat pump)
    cooling_enabled:   bool  = False
    Php_cool_max_kw:   float = 50.0    # Max cooling thermal output [kW]
    COP_cool:          float =  3.0    # Cooling COP (constant approximation)    # Night setback schedule
    use_night_setback: bool  = False
    T_set_night_c:     float = 18.0    # Heating comfort floor at night [°C]
    T_cool_night_c:    float = 28.0    # Cooling comfort ceiling at night [°C]
    night_start_h:     float = 22.0    # Hour (0–24) when night period begins
    night_end_h:       float =  7.0    # Hour (0–24) when day resumes
    # Optional: list of weekdays (Mon=0..Sun=6) treated as fully night.
    # Useful e.g. for offices closed on weekends — the whole day uses the
    # T_set_night_c / T_cool_night_c bounds (in both MPC and baseline).
    night_setback_days: list = field(default_factory=list)

    # ── Gas boiler ──────────────────────────────────────────────────────────
    boiler_enabled: bool  = True
    Pgas_max_kw:    float = 100.0  # Max thermal output [kW]
    eta_boiler:     float =  0.92  # Thermal efficiency (fraction of gas HHV)
    gas_price_boiler_eur_kwh: float = 0.035  # Gas price [€/kWh]
    gas_HV_kwh_m3:  float =  9.8   # Gas calorific value [kWh/m³] (kept for backward compat)
    boiler_ramp_pct: float = 100.0  # Max boiler ramp [% of Pgas_max_kw per step]

    # ── CHP / cogeneration ──────────────────────────────────────────────────
    chp_enabled:         bool  = False
    Pchp_max_kw:         float = 39.2   # Max CHP electrical output [kW]
    Fchp_max_m3_h:       float = 10.0   # Derived: Pchp_max_kw / (HV × η_elec) [m³/h]
    chp_eta_elec:        float =  0.40  # Electrical efficiency
    chp_eta_heat:        float =  0.45  # Thermal efficiency
    chp_startup_cost_eur: float =  5.0  # One-off startup cost [€]
    gas_price_chp_eur_m3: float =  0.35 # Gas price for CHP [€/m³]
    chp_gas_HV_kwh_m3:   float =  9.8   # Gas calorific value [kWh/m³]
    chp_use_milp:        bool  = True   # True = MILP (binary z,y); False = LP relaxation
    chp_ramp_pct:        float = 100.0  # Max CHP ramp [% of Pchp_max_kw per step]
    chp_heat_dump_enabled: bool = True   # External heat-dump radiator. When False,
                                         # all CHP waste heat must be absorbed by
                                         # the building (CHP will throttle if it
                                         # would breach the cooling ceiling).

    # ── PV ──────────────────────────────────────────────────────────────────
    pv_enabled:    bool  = True
    pv_capacity_kwp: float = 100.0  # Installed PV capacity [kWp]
    # Sell excess PV back to grid at spot price (no grid fee on the revenue).
    # When False, excess PV is curtailed (legacy behaviour, no income).
    pv_export_enabled: bool = True

    # ── Flexible / shiftable load ────────────────────────────────────────────
    # Legacy single-flex-load fields (used when ``flex_loads`` is empty so that
    # older configs keep working).  Real installations should populate
    # ``flex_loads`` so each shiftable load gets its own time window, energy
    # budget and ramp constraints — without that, multiple flex assets in the
    # GUI all collapse into one solver variable and the solution is wrong.
    flex_enabled:      bool  = False
    Pflex_max_kw:      float =  50.0   # Max instantaneous power [kW]
    flex_daily_kwh:    float = 400.0   # Daily energy requirement [kWh]
    flex_time_start:   str   = "00:00" # Active window start [HH:MM]
    flex_time_end:     str   = "23:59" # Active window end [HH:MM]
    flex_ramp_up_kw:   float = 9999.0  # Max power increase per step [kW]
    flex_ramp_down_kw: float = 9999.0  # Max power decrease per step [kW]

    # Per-instance flex loads — one dict per shiftable asset, each with its own
    # window/limits.  Populated from ``mpc.asset_instances`` (type=="flex").
    # When present, the MPC creates N independent Pflex variables and applies
    # the constraints separately for each.  Keys per entry:
    #   id, name, Pmax_kw, daily_kwh, time_start, time_end,
    #   ramp_up_kw, ramp_down_kw,
    #   bl_mode, bl_time_start, bl_time_end, bl_power_kw
    flex_loads:        list  = field(default_factory=list)

    # ── Hot water tank ───────────────────────────────────────────────────────
    hw_enabled:      bool  = True
    Ptank_max_kw:    float =  3.0    # Heater rated power [kW]  (COP = 1)
    hw_volume_l:     float = 200.0   # Tank volume [L]
    hw_T_min_c:      float =  45.0   # Minimum temperature (comfort/safety) [°C]
    hw_T_max_c:      float =  60.0   # Maximum temperature [°C]
    hw_T_init_c:     float =  55.0   # Initial temperature [°C]
    hw_heat_loss_w:  float =  50.0   # Standby heat loss [W] at nominal temperature
    hw_draw_kw:      float =   0.5   # Constant hot-water draw [kW]
    hw_ramp_pct:     float = 100.0   # Max hot-water heater ramp [% of Ptank_max_kw per step]
    hw_indoor_amb_c: float =  18.0   # Tank's surrounding plant-room temp [°C]
                                     # — heat loss is to INDOOR air, not outdoor

    # ── Building ─────────────────────────────────────────────────────────────
    Tset_c:          float =  21.0    # Setpoint temperature [°C]
    Tmin_c:          float =  19.0    # Minimum comfort temperature [°C]
    Tmax_c:          float =  23.0    # Maximum comfort temperature [°C]
    T_init_c:        float =  20.0    # Initial building temperature [°C]
    Cth_kwh_per_c:   float = 2000.0   # Thermal mass [kWh/°C]
    UA_kw_per_c:     float =  100.0   # Envelope heat transfer coefficient [kW/°C]

    # Base (non-controllable) electrical load — day/night profile
    # Day window is [night_end_h, night_start_h); the rest is night.
    base_load_day_kw:   float = 200.0   # Daytime base load [kW]
    base_load_night_kw: float =  80.0   # Nighttime base load [kW]

    # ── Baseline simulation modes ─────────────────────────────────────────────
    # These define the rule-based 'no-optimisation' reference scenario used to
    # calculate energy cost savings attributable to the MPC.
    #
    # heat_pump / gas_boiler:
    #   "on_off"     — bang-bang controller around the active heating setpoint
    #                  (day/night): heat below Tset_active-0.5 and stop
    #                  above Tset_active+0.5.
    #                  Both share one heating_on state flag (same thermostat).
    #   "constant"   — fixed power (hp_bl_power_kw / boiler_bl_power_kw) every step.
    #   "always_off" — asset not used in the baseline.
    # battery:
    #   "always_off" — idle (no arbitrage without optimisation).
    # chp:
    #   "always_off"   — not used in baseline.
    #   "heat_demand"  — heat-led dispatch with hysteresis: switch ON when building
    #                    T drops below (Tset_c - 0.5°C), switch OFF when T exceeds
    #                    (Tset_c + 0.5°C). Mirrors a real heat-led CHP controller.
    #                    Electricity is a byproduct.
    #   "ambient_temp" — outdoor-temperature led: run at max output whenever the
    #                    outdoor temperature is below ``chp_bl_Tamb_threshold_c``
    #                    (heating season). This is the most common real-world
    #                    CHP baseline ("run when it's cold outside").
    #   "constant"     — always on at max output.
    # hot_water_tank:
    #   "on_off"     — bang-bang: heat at baseline_power_kw when T < hw_T_min_c,
    #                  stop when T >= hw_T_max_c. Uses hw_T_min_c / hw_T_max_c as
    #                  the deadband (no price shifting).
    #   "constant"   — fixed power (hw_bl_power_kw) every step.
    #   "always_off" — not used in baseline.
    # flexible_load:
    #   "fixed_window" — run at baseline_power_kw between flex_bl_time_start and
    #                    flex_bl_time_end every day (separate from the MPC window),
    #                    regardless of price.
    #   "always_off"   — not used in baseline.
    #
    # For on_off and fixed_window: baseline_power_kw = 0 means "use the asset max".
    hp_bl_mode:         str   = "on_off"
    hp_bl_power_kw:     float = 0.0
    boiler_bl_mode:     str   = "on_off"
    boiler_bl_power_kw: float = 0.0
    bat_bl_mode:        str   = "always_off"
    chp_bl_mode:        str   = "always_off"
    chp_bl_Tamb_threshold_c: float = 15.0   # CHP runs when Tamb < this [°C]
    hw_bl_mode:         str   = "on_off"
    hw_bl_power_kw:     float = 0.0
    flex_bl_mode:       str   = "fixed_window"
    flex_bl_time_start: str   = "00:00"   # Baseline run window start [HH:MM]
    flex_bl_time_end:   str   = "23:59"   # Baseline run window end   [HH:MM]
    flex_bl_power_kw:   float = 0.0       # Baseline fixed power (0 → use Pflex_max_kw)

    # ── Config I/O ───────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "MPCConfig":
        """Build MPCConfig from the ``'mpc'`` block in dashboard_config.json."""
        def _g(section: str, key: str, default):
            return d.get(section, {}).get(key, default)

        c = cls()

        c.horizon_steps = int(_g("horizon", "steps",    c.horizon_steps))
        c.dt_hours      = float(_g("horizon", "dt_hours", c.dt_hours))

        c.Pgrid_max_kw       = float(_g("grid", "Pgrid_max_kw",       c.Pgrid_max_kw))
        c.fee_eur_kwh        = float(_g("grid", "fee_eur_kwh",        c.fee_eur_kwh))
        c.cap_tariff_Plim_kw   = float(_g("grid", "cap_tariff_Plim_kw",   c.cap_tariff_Plim_kw))
        c.cap_tariff_epsilon_l = float(_g("grid", "cap_tariff_epsilon_l", c.cap_tariff_epsilon_l))

        bat = d.get("battery", {})
        c.bat_enabled  = bool (bat.get("enabled",     c.bat_enabled))
        c.SOC_cap_kwh  = float(bat.get("SOC_cap_kwh", c.SOC_cap_kwh))
        c.SOC_min_kwh  = float(bat.get("SOC_min_kwh", c.SOC_min_kwh))
        c.SOC_init_kwh = float(bat.get("SOC_init_kwh",c.SOC_init_kwh))
        c.SOC_end_kwh  = float(bat.get("SOC_end_kwh", c.SOC_end_kwh))
        c.Pch_max_kw   = float(bat.get("Pch_max_kw",  c.Pch_max_kw))
        c.Pdis_max_kw  = float(bat.get("Pdis_max_kw", c.Pdis_max_kw))
        c.eta_ch       = float(bat.get("eta_ch",       c.eta_ch))
        c.eta_dis      = float(bat.get("eta_dis",      c.eta_dis))
        c.bat_ramp_pct = float(bat.get("ramp_pct",     c.bat_ramp_pct))
        hp = d.get("heat_pump", {})
        c.hp_enabled       = bool (hp.get("enabled",          c.hp_enabled))
        c.Php_max_kw       = float(hp.get("Php_max_kw",       c.Php_max_kw))
        c.COP0             = float(hp.get("COP0",              c.COP0))
        c.T0_c             = float(hp.get("T0_c",              c.T0_c))
        c.cop_alpha        = float(hp.get("cop_alpha",         c.cop_alpha))
        c.COP_min          = float(hp.get("COP_min",           c.COP_min))
        c.hp_ramp_pct      = float(hp.get("ramp_pct",          c.hp_ramp_pct))
        c.hp_bl_mode       = str  (hp.get("baseline_mode",     c.hp_bl_mode))
        c.hp_bl_power_kw   = float(hp.get("baseline_power_kw", c.hp_bl_power_kw))
        c.cooling_enabled   = bool (hp.get("cooling_enabled",   c.cooling_enabled))
        c.Php_cool_max_kw   = float(hp.get("Php_cool_max_kw",   c.Php_cool_max_kw))
        c.COP_cool          = float(hp.get("COP_cool",           c.COP_cool))

        bl = d.get("gas_boiler", {})
        c.boiler_enabled          = bool (bl.get("enabled",          c.boiler_enabled))
        c.Pgas_max_kw             = float(bl.get("Pgas_max_kw",      c.Pgas_max_kw))
        c.eta_boiler              = float(bl.get("eta_boiler",        c.eta_boiler))
        c.gas_HV_kwh_m3           = float(bl.get("gas_HV_kwh_m3",    c.gas_HV_kwh_m3))
        if "gas_price_eur_kwh" in bl:
            c.gas_price_boiler_eur_kwh = float(bl["gas_price_eur_kwh"])
        elif "gas_price_eur_m3" in bl:
            c.gas_price_boiler_eur_kwh = float(bl["gas_price_eur_m3"]) / max(c.gas_HV_kwh_m3, 1e-9)
        c.boiler_ramp_pct         = float(bl.get("ramp_pct",         c.boiler_ramp_pct))
        c.boiler_bl_mode          = str  (bl.get("baseline_mode",     c.boiler_bl_mode))
        c.boiler_bl_power_kw      = float(bl.get("baseline_power_kw", c.boiler_bl_power_kw))

        chp = d.get("chp", {})
        c.chp_enabled          = bool (chp.get("enabled",          c.chp_enabled))
        c.chp_eta_elec         = float(chp.get("eta_elec",         c.chp_eta_elec))
        c.chp_eta_heat         = float(chp.get("eta_heat",         c.chp_eta_heat))
        c.chp_startup_cost_eur = float(chp.get("startup_cost_eur", c.chp_startup_cost_eur))
        # CHP gas price: prefer EUR/kWh (same unit as boiler) to avoid the
        # easy unit mix-up.  Legacy ``gas_price_eur_m3`` is still honoured.
        c.chp_gas_HV_kwh_m3    = float(chp.get("gas_HV_kwh_m3",   c.chp_gas_HV_kwh_m3))
        if "gas_price_eur_kwh" in chp:
            c.gas_price_chp_eur_m3 = float(chp["gas_price_eur_kwh"]) * c.chp_gas_HV_kwh_m3
        elif "gas_price_eur_m3" in chp:
            c.gas_price_chp_eur_m3 = float(chp["gas_price_eur_m3"])
        c.chp_use_milp         = bool (chp.get("use_milp",        c.chp_use_milp))
        c.chp_ramp_pct         = float(chp.get("ramp_pct",         c.chp_ramp_pct))
        c.chp_heat_dump_enabled = bool(chp.get("heat_dump_enabled", c.chp_heat_dump_enabled))
        c.chp_bl_mode          = str  (chp.get("baseline_mode",    c.chp_bl_mode))
        c.chp_bl_Tamb_threshold_c = float(chp.get("bl_Tamb_threshold_c", c.chp_bl_Tamb_threshold_c))
        # CHP capacity reconciliation.
        # ``Pchp_max_kw`` is the primary, user-facing nameplate (kW electric)
        # and is what the dashboard GUI edits.  ``Fchp_max_m3_h`` (gas flow cap)
        # is always derived from it via:  Fchp = Pchp / (HV * η_elec).
        # If only ``Fchp_max_m3_h`` is given (legacy configs), Pchp is derived
        # the other way round.  We no longer take ``min(...)`` of the two caps,
        # which used to silently throttle the CHP when the JSON values were
        # inconsistent.
        _hv_e = c.chp_gas_HV_kwh_m3 * c.chp_eta_elec
        if "Pchp_max_kw" in chp:
            c.Pchp_max_kw   = float(chp["Pchp_max_kw"])
            c.Fchp_max_m3_h = c.Pchp_max_kw / max(_hv_e, 1e-9)
        elif "Fchp_max_m3_h" in chp:
            c.Fchp_max_m3_h = float(chp["Fchp_max_m3_h"])
            c.Pchp_max_kw   = c.Fchp_max_m3_h * _hv_e

        pv = d.get("pv", {})
        c.pv_enabled       = bool (pv.get("enabled",        c.pv_enabled))
        c.pv_capacity_kwp  = float(pv.get("capacity_kwp",   c.pv_capacity_kwp))
        c.pv_export_enabled = bool(pv.get("export_enabled", c.pv_export_enabled))

        fx = d.get("flexible_load", {})
        c.flex_enabled      = bool (fx.get("enabled",          c.flex_enabled))
        c.Pflex_max_kw      = float(fx.get("Pflex_max_kw",     c.Pflex_max_kw))
        c.flex_daily_kwh    = float(fx.get("daily_energy_kwh",  c.flex_daily_kwh))
        c.flex_time_start   = str  (fx.get("time_start",        c.flex_time_start))
        c.flex_time_end     = str  (fx.get("time_end",          c.flex_time_end))
        c.flex_ramp_up_kw   = float(fx.get("ramp_up_kw",        c.flex_ramp_up_kw))
        c.flex_ramp_down_kw = float(fx.get("ramp_down_kw",      c.flex_ramp_down_kw))
        c.flex_bl_mode       = str  (fx.get("baseline_mode",     c.flex_bl_mode))
        c.flex_bl_time_start = str  (fx.get("bl_time_start",     c.flex_bl_time_start))
        c.flex_bl_time_end   = str  (fx.get("bl_time_end",       c.flex_bl_time_end))
        c.flex_bl_power_kw   = float(fx.get("baseline_power_kw", c.flex_bl_power_kw))

        # ── Per-instance flex loads ────────────────────────────────────────
        # Pull every enabled "flex" entry from asset_instances so each
        # shiftable load becomes its own solver variable. This is what makes
        # multi-flex configurations behave correctly — without it, only the
        # first flex instance's params are seen by the MPC.
        c.flex_loads = []
        for _inst in d.get("asset_instances", []):
            if _inst.get("type") != "flex" or not _inst.get("enabled", True):
                continue
            c.flex_loads.append({
                "id":            str(_inst.get("id", "flex")),
                "name":          str(_inst.get("name", _inst.get("id", "Flex"))),
                "Pmax_kw":       float(_inst.get("Pflex_max_kw",     c.Pflex_max_kw)),
                "daily_kwh":     float(_inst.get("daily_energy_kwh", c.flex_daily_kwh)),
                "time_start":    str  (_inst.get("time_start",        c.flex_time_start)),
                "time_end":      str  (_inst.get("time_end",          c.flex_time_end)),
                "ramp_up_kw":    float(_inst.get("ramp_up_kw",        c.flex_ramp_up_kw)),
                "ramp_down_kw":  float(_inst.get("ramp_down_kw",      c.flex_ramp_down_kw)),
                "bl_mode":       str  (_inst.get("baseline_mode",     c.flex_bl_mode)),
                "bl_time_start": str  (_inst.get("baseline_time_start",
                                                 _inst.get("bl_time_start", c.flex_bl_time_start))),
                "bl_time_end":   str  (_inst.get("baseline_time_end",
                                                 _inst.get("bl_time_end",   c.flex_bl_time_end))),
                "bl_power_kw":   float(_inst.get("baseline_power_kw", c.flex_bl_power_kw)),
                # Day-of-week gates (Mon=0 … Sun=6). Empty / missing list
                # means “all days allowed” (backward compatible).
                "active_days":          list(_inst.get("active_days", []) or []),
                "baseline_active_days": list(_inst.get("baseline_active_days", []) or []),
            })
        if c.flex_loads:
            c.flex_enabled = True   # any per-instance flex load implies enabled

        hw = d.get("hot_water_tank", {})
        c.hw_enabled    = bool (hw.get("enabled",      c.hw_enabled))
        c.Ptank_max_kw  = float(hw.get("Ptank_max_kw", c.Ptank_max_kw))
        c.hw_volume_l   = float(hw.get("volume_l",     c.hw_volume_l))
        c.hw_T_min_c    = float(hw.get("T_min_c",      c.hw_T_min_c))
        c.hw_T_max_c    = float(hw.get("T_max_c",      c.hw_T_max_c))
        c.hw_T_init_c   = float(hw.get("T_init_c",     c.hw_T_init_c))
        c.hw_heat_loss_w = float(hw.get("heat_loss_w", c.hw_heat_loss_w))
        c.hw_draw_kw     = float(hw.get("draw_kw",     c.hw_draw_kw))
        c.hw_ramp_pct    = float(hw.get("ramp_pct",    c.hw_ramp_pct))
        c.hw_indoor_amb_c = float(hw.get("indoor_amb_c", c.hw_indoor_amb_c))
        c.hw_bl_mode     = str  (hw.get("baseline_mode",     c.hw_bl_mode))
        c.hw_bl_power_kw = float(hw.get("baseline_power_kw", c.hw_bl_power_kw))

        bld = d.get("building", {})
        c.Tset_c        = float(bld.get("Tset_c",        c.Tset_c))
        c.Tmin_c        = float(bld.get("Tmin_c",        c.Tmin_c))
        c.Tmax_c        = float(bld.get("Tmax_c",        c.Tmax_c))
        c.T_init_c      = float(bld.get("T_init_c",      c.T_init_c))
        c.Cth_kwh_per_c = float(bld.get("Cth_kwh_per_c", c.Cth_kwh_per_c))
        c.UA_kw_per_c   = float(bld.get("UA_kw_per_c",   c.UA_kw_per_c))
        c.use_night_setback = bool (bld.get("use_night_setback",  c.use_night_setback))
        c.T_set_night_c     = float(bld.get("T_set_night_c",      c.T_set_night_c))
        c.T_cool_night_c    = float(bld.get("T_cool_night_c",     c.T_cool_night_c))
        c.night_start_h     = float(bld.get("night_start_h",      c.night_start_h))
        c.night_end_h       = float(bld.get("night_end_h",        c.night_end_h))
        try:
            c.night_setback_days = [int(d) % 7 for d in (bld.get("night_setback_days") or [])]
        except Exception:
            c.night_setback_days = []
        c.base_load_day_kw   = float(bld.get("base_load_day_kw",   c.base_load_day_kw))
        c.base_load_night_kw = float(bld.get("base_load_night_kw", c.base_load_night_kw))

        return c

    def to_dict(self) -> dict:
        """Serialise to the ``'mpc'`` block format for dashboard_config.json."""
        return {
            "horizon": {"steps": self.horizon_steps, "dt_hours": self.dt_hours},
            "grid":    {
                "Pgrid_max_kw":       self.Pgrid_max_kw,
                "fee_eur_kwh":        self.fee_eur_kwh,
                "cap_tariff_Plim_kw":   self.cap_tariff_Plim_kw,
                "cap_tariff_epsilon_l": self.cap_tariff_epsilon_l,
            },
            "battery": {
                "enabled":     self.bat_enabled,
                "SOC_cap_kwh": self.SOC_cap_kwh,  "SOC_min_kwh": self.SOC_min_kwh,
                "SOC_init_kwh": self.SOC_init_kwh, "SOC_end_kwh": self.SOC_end_kwh,
                "Pch_max_kw":  self.Pch_max_kw,   "Pdis_max_kw": self.Pdis_max_kw,
                "eta_ch":      self.eta_ch,        "eta_dis":     self.eta_dis,
                "ramp_pct":    self.bat_ramp_pct,
            },
            "heat_pump": {
                "enabled":           self.hp_enabled,  "Php_max_kw": self.Php_max_kw,
                "COP0":              self.COP0,        "T0_c":       self.T0_c,
                "cop_alpha":         self.cop_alpha,   "COP_min":    self.COP_min,
                "ramp_pct":          self.hp_ramp_pct,
                "baseline_mode":     self.hp_bl_mode,
                "baseline_power_kw": self.hp_bl_power_kw,
                "cooling_enabled":   self.cooling_enabled,
                "Php_cool_max_kw":   self.Php_cool_max_kw,
                "COP_cool":          self.COP_cool,
            },
            "gas_boiler": {
                "enabled":           self.boiler_enabled,
                "Pgas_max_kw":       self.Pgas_max_kw,
                "eta_boiler":        self.eta_boiler,
                "gas_price_eur_kwh":  self.gas_price_boiler_eur_kwh,
                "ramp_pct":          self.boiler_ramp_pct,
                "baseline_mode":     self.boiler_bl_mode,
                "baseline_power_kw": self.boiler_bl_power_kw,
            },
            "chp": {
                "enabled":          self.chp_enabled,
                "Pchp_max_kw":      self.Pchp_max_kw,
                "Fchp_max_m3_h":    self.Fchp_max_m3_h,
                "eta_elec":         self.chp_eta_elec,
                "eta_heat":         self.chp_eta_heat,
                "startup_cost_eur": self.chp_startup_cost_eur,
                "gas_price_eur_kwh": self.gas_price_chp_eur_m3 / max(self.chp_gas_HV_kwh_m3, 1e-9),
                "gas_HV_kwh_m3":    self.chp_gas_HV_kwh_m3,
                "use_milp":         self.chp_use_milp,
                "ramp_pct":         self.chp_ramp_pct,
                "heat_dump_enabled": self.chp_heat_dump_enabled,
                "baseline_mode":    self.chp_bl_mode,
                "bl_Tamb_threshold_c": self.chp_bl_Tamb_threshold_c,
            },
            "pv": {"enabled": self.pv_enabled, "capacity_kwp": self.pv_capacity_kwp, "export_enabled": self.pv_export_enabled},
            "flexible_load": {
                "enabled":          self.flex_enabled,
                "Pflex_max_kw":     self.Pflex_max_kw,
                "daily_energy_kwh": self.flex_daily_kwh,
                "time_start":       self.flex_time_start,
                "time_end":         self.flex_time_end,
                "ramp_up_kw":       self.flex_ramp_up_kw,
                "ramp_down_kw":     self.flex_ramp_down_kw,
                "baseline_mode":    self.flex_bl_mode,
                "bl_time_start":    self.flex_bl_time_start,
                "bl_time_end":      self.flex_bl_time_end,
                "baseline_power_kw": self.flex_bl_power_kw,
            },
            "hot_water_tank": {
                "enabled":           self.hw_enabled,
                "Ptank_max_kw":      self.Ptank_max_kw,
                "volume_l":          self.hw_volume_l,
                "T_min_c":           self.hw_T_min_c,
                "T_max_c":           self.hw_T_max_c,
                "T_init_c":          self.hw_T_init_c,
                "heat_loss_w":       self.hw_heat_loss_w,
                "draw_kw":           self.hw_draw_kw,
                "ramp_pct":          self.hw_ramp_pct,
                "indoor_amb_c":      self.hw_indoor_amb_c,
                "baseline_mode":     self.hw_bl_mode,
                "baseline_power_kw": self.hw_bl_power_kw,
            },
            "building": {
                "Tset_c":        self.Tset_c,
                "Tmin_c":        self.Tmin_c,
                "Tmax_c":        self.Tmax_c,
                "T_init_c":      self.T_init_c,
                "Cth_kwh_per_c": self.Cth_kwh_per_c,
                "UA_kw_per_c":   self.UA_kw_per_c,
                "use_night_setback": self.use_night_setback,
                "T_set_night_c":     self.T_set_night_c,
                "T_cool_night_c":    self.T_cool_night_c,
                "night_start_h":     self.night_start_h,
                "night_end_h":       self.night_end_h,
                "night_setback_days": list(self.night_setback_days),
            },
        }


# ===========================================================================
# SECTION 2 — INPUT / OUTPUT DATA STRUCTURES
# ===========================================================================

@dataclass
class MPCInputs:
    """Everything needed for one MPC solve call."""

    # Electricity spot price forecast [€/kWh], shape (H,)
    price_eur_kwh: np.ndarray

    # Fixed (non-controllable) load forecast [kW], shape (H,)
    Pload_kw: np.ndarray

    # PV production forecast [kW], shape (H,) — upper bound on Ppv
    Ppv_forecast_kw: np.ndarray

    # Ambient (outdoor) temperature forecast [°C], shape (H,)
    # Used in COP[k] = COP0*(1 + cop_alpha*(Tamb[k]−T0)) and building physics
    Tamb_c: np.ndarray

    # Initial battery state of charge [kWh]
    SOC_init_kwh: float = 50.0

    # Initial building temperature [°C]
    T_building_init_c: float = 21.0

    # Initial hot water tank temperature [°C]
    T_tank_init_c: float = 55.0

    # Current month (1–12) — metadata only, not used in optimisation
    month: int = 1

    # Previous heat pump thermal output [kW] — used as initial condition for
    # the HP ramp constraint so the LP plans ramp-up BEFORE cheap periods
    Php_prev_kw: float = 0.0


@dataclass
class MPCOutputs:
    """
    Results from one MPC solve call.

    First-step values are the commands to execute NOW (receding horizon).
    Full-horizon plan arrays are for dashboard trend charts.
    """

    # ── First-step commands [kW unless noted] ───────────────────────────────
    Pgrid_kw:    float = 0.0     # Grid import
    Php_kw:      float = 0.0     # Heat pump thermal output
    Pgas_kw:     float = 0.0     # Gas boiler thermal output
    Pch_kw:      float = 0.0     # Battery charging
    Pdis_kw:     float = 0.0     # Battery discharging
    SOC_kwh:     float = 0.0     # Battery SOC after first step [kWh]
    Pflex_kw:    float = 0.0     # Flexible load
    Ppv_kw:      float = 0.0     # PV power used
    Ptank_kw:    float = 0.0     # Hot water heater
    Ttank_c:     float = 55.0    # Hot water tank temperature [°C]
    Fchp_m3_h:   float = 0.0     # CHP gas flow rate [m³/h]
    Pchp_kw:     float = 0.0     # CHP electrical output
    Qchp_kw:     float = 0.0     # CHP thermal output
    zchp:        float = 0.0     # CHP on/off (0 or 1)
    ychp:        float = 0.0     # CHP startup event (0 or 1)
    Tbuilding_c: float = 21.0    # Building indoor temperature [°C]
    Php_cool_kw: float = 0.0     # Heat pump cooling output first step [kW]
    COP_now:     float = 4.0     # COP at first step (derived from Tamb)

    # ── Derived KPIs ─────────────────────────────────────────────────────────
    total_heating_kw:   float = 0.0    # Php + Pgas + Qchp
    mpc_cost_eur:       float = 0.0    # Optimised cost this interval [€]
    baseline_cost_eur:  float = 0.0    # Baseline (no optimisation) [€]
    cost_saving_eur:    float = 0.0    # Saving vs baseline [€]

    # ── Legacy field aliases (for backward-compat with dashboard / SMPCOutputs) ──
    # These fields are set by the mapping layer in smpc_calculator.py.
    net_power_kwh:       float = 0.0   # Pgrid * dt
    wkk_elec_kwh:        float = 0.0   # Pchp * dt
    wkk_heat_kwh:        float = 0.0   # Qchp * dt
    wkk_gas_setpoint_m3: float = 0.0   # Fchp * dt  [m³ per step]
    ice_bank_charge_kwh: float = 0.0   # Pflex * dt  (legacy "ice bank")
    ice_bank_discharge_kwh: float = 0.0

    # ── Full-horizon plans [shape (H,) or (H+1,)] ───────────────────────────
    plan_Pgrid:     np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Php:       np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Pgas:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Pch:       np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Pdis:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_SOC:       np.ndarray = field(default_factory=lambda: np.array([]))   # H+1
    plan_Pflex:     np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-instance flex schedules — {instance_id: kW array of length H}.
    # Empty when no per-instance ``flex_loads`` are configured (legacy single-flex
    # config), in which case ``plan_Pflex`` holds the sole schedule.
    plan_Pflex_by_id: dict      = field(default_factory=dict)
    # Per-instance first-step power [kW] — {instance_id: kW scalar}.
    Pflex_by_id_kw:   dict      = field(default_factory=dict)
    plan_Ppv:       np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Pexport:   np.ndarray = field(default_factory=lambda: np.array([]))   # PV exported to grid [kW]
    plan_Ptank:     np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Ttank:     np.ndarray = field(default_factory=lambda: np.array([]))   # H+1
    plan_Fchp:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Pchp:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Qchp:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_zchp:      np.ndarray = field(default_factory=lambda: np.array([]))
    plan_Tbuilding: np.ndarray = field(default_factory=lambda: np.array([]))   # H+1
    plan_Php_cool:  np.ndarray = field(default_factory=lambda: np.array([]))
    plan_COP:       np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Solver metadata ───────────────────────────────────────────────────────
    solver_used:   str   = "none"
    solver_status: str   = "unknown"
    solve_time_ms: float = 0.0
    obj_value:     float = float("nan")


# ===========================================================================
# SECTION 3 — COP MODEL
# ===========================================================================

def compute_cop(Tamb: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Compute the heat pump COP profile for the optimisation horizon.

    Model
    -----
        COP[k] = COP0 × max(COP_min, 1 + cop_alpha × (Tamb[k] − T0))

    This is a first-order linear approximation valid around the rated
    operating point (Tamb = T0).  For Tamb > T0 the COP increases
    (warmer outside = smaller temperature lift = higher efficiency);
    for Tamb < T0 the COP decreases.  The COP_min clamp prevents
    non-physical values.

    Parameters
    ----------
    Tamb : ambient temperature forecast [°C], shape (H,)
    cfg  : MPCConfig with COP0, T0_c, cop_alpha, COP_min

    Returns
    -------
    COP : np.ndarray, shape (H,) — COP values ≥ COP_min
    """
    raw = cfg.COP0 * (1.0 + cfg.cop_alpha * (Tamb - cfg.T0_c))
    return np.maximum(cfg.COP_min, raw)


def build_pload_vector(cfg: "MPCConfig", start_hour: float, H: int) -> np.ndarray:
    """
    Return a (H,) array of base (non-controllable) electrical load [kW].

    Day window is ``[night_end_h, night_start_h)`` (default 07:00–22:00);
    daytime steps get ``base_load_day_kw``, all other steps ``base_load_night_kw``.
    """
    arr = np.empty(H)
    dt  = cfg.dt_hours
    ns  = float(cfg.night_start_h)
    ne  = float(cfg.night_end_h)
    for k in range(H):
        h = (start_hour + k * dt) % 24.0
        # night = h >= night_start_h or h < night_end_h
        is_night = (h >= ns) or (h < ne)
        arr[k] = cfg.base_load_night_kw if is_night else cfg.base_load_day_kw
    return arr


def _is_night_step(k: int, dt: float, cfg: MPCConfig, start_hour: float = 0.0,
                   start_weekday: int | None = None) -> bool:
    """Return whether horizon step k falls in configured night period.

    A day is treated as fully night if its weekday is listed in
    ``cfg.night_setback_days`` (Mon=0..Sun=6). Otherwise the time-of-day
    window [night_start_h, 24) ∪ [0, night_end_h) applies.
    """
    if not cfg.use_night_setback:
        return False
    # Full-day setback override (e.g. weekends in 'weekend mode').
    if start_weekday is not None and cfg.night_setback_days:
        try:
            _full_days = {int(d) % 7 for d in cfg.night_setback_days}
        except Exception:
            _full_days = set()
        if _full_days:
            day_off = int((start_hour + k * dt) // 24.0)
            wd = (int(start_weekday) + day_off) % 7
            if wd in _full_days:
                return True
    hour = (start_hour + k * dt) % 24.0
    return hour >= cfg.night_start_h or hour < cfg.night_end_h




def _active_heat_setpoint(k: int, dt: float, cfg: MPCConfig, start_hour: float = 0.0,
                          start_weekday: int | None = None) -> float:
    """Heating thermostat center used by rule-based controllers."""
    return (cfg.T_set_night_c
            if _is_night_step(k, dt, cfg, start_hour=start_hour, start_weekday=start_weekday)
            else cfg.Tset_c)


def _active_comfort_bounds(
    k: int,
    dt: float,
    cfg: MPCConfig,
    start_hour: float = 0.0,
    start_weekday: int | None = None,
) -> tuple[float, float]:
    """Comfort lower/upper bounds active at step k."""
    if _is_night_step(k, dt, cfg, start_hour=start_hour, start_weekday=start_weekday):
        return cfg.T_set_night_c, cfg.T_cool_night_c
    return cfg.Tmin_c, cfg.Tmax_c


# ===========================================================================
# SECTION 4 — HEURISTIC FALLBACK
# (used when cvxpy is unavailable or the solver fails)
# ===========================================================================

def _solve_heuristic(
    inputs: MPCInputs,
    cfg: MPCConfig,
    COP: np.ndarray,
    H: int,
    dt: float,
    start_hour: float = 0.0,
    start_weekday: int | None = None,
) -> dict:
    """
    Rule-based schedule when the LP solver is unavailable.

    Rules
    -----
    * PV: always use all available PV.
    * CHP: fire when spark spread is positive and heat buffer is not full.
    * Heat pump: maintain building temperature within deadband.
    * Gas boiler: supplement heat pump if thermal gap remains.
    * Battery: charge when cheap (lowest 30th percentile), discharge when
      expensive (top 30th percentile).
    * Hot water tank: heat when below T_min, stop at T_max.
    * Flexible load: spread evenly over cheapest intervals.
    * Grid: balancing residual.
    """
    price_total = inputs.price_eur_kwh + cfg.fee_eur_kwh  # (H,)

    Pgrid    = np.zeros(H)
    Pexport  = np.zeros(H)
    Php      = np.zeros(H)
    Pgas     = np.zeros(H)
    Pch      = np.zeros(H)
    Pdis     = np.zeros(H)
    SOC      = np.zeros(H + 1)
    Pflex    = np.zeros(H)
    Ppv      = np.zeros(H)
    Ptank    = np.zeros(H)
    Ttank    = np.zeros(H + 1)
    Fchp     = np.zeros(H)
    Pchp     = np.zeros(H)
    Qchp     = np.zeros(H)
    zchp     = np.zeros(H)
    Tbuilding = np.zeros(H + 1)
    Php_cool  = np.zeros(H)

    SOC[0]       = cfg.SOC_init_kwh
    Ttank[0]     = inputs.T_tank_init_c
    Tbuilding[0] = inputs.T_building_init_c

    # Pre-compute thresholds
    p_low  = np.percentile(price_total, 30)
    p_high = np.percentile(price_total, 70)

    # Flexible load(s): schedule each instance in its cheapest slots inside its
    # own active window.  When per-instance ``flex_loads`` is empty we fall back
    # to the legacy single-flex parameters.
    Pflex_by_id: dict = {}
    if cfg.flex_enabled:
        if cfg.flex_loads:
            _flex_specs = [
                (fl["id"], float(fl["Pmax_kw"]), float(fl["daily_kwh"]),
                 str(fl["time_start"]), str(fl["time_end"]))
                for fl in cfg.flex_loads
            ]
        else:
            _flex_specs = [(
                "flexible_load", float(cfg.Pflex_max_kw), float(cfg.flex_daily_kwh),
                str(cfg.flex_time_start), str(cfg.flex_time_end),
            )]
        horizon_h  = H * dt
        sorted_idx = np.argsort(price_total)
        # Per-instance days list (heuristic path): pull from flex_loads when
        # available so the greedy fallback honours the same Mon–Sun gating.
        _flex_days_per_id = {
            str(fl["id"]): list(fl.get("active_days", []))
            for fl in (cfg.flex_loads or [])
        }
        for _id, _pmax, _daily, _ts, _te in _flex_specs:
            if _daily <= 0 or _pmax <= 0:
                Pflex_by_id[_id] = np.zeros(H)
                continue
            _flex_active = _compute_flex_active_mask(
                _ts, _te, H, dt, start_hour=start_hour,
                start_weekday=start_weekday,
                active_days=_flex_days_per_id.get(str(_id)),
            )
            target_kwh = _daily * (horizon_h / 24.0)
            remaining  = target_kwh
            arr        = np.zeros(H)
            for idx in sorted_idx:
                if remaining <= 0:
                    break
                if not _flex_active[idx]:
                    continue
                slot_kwh   = min(_pmax * dt, remaining)
                arr[idx]   = slot_kwh / dt
                remaining -= slot_kwh
            Pflex_by_id[_id] = arr
            Pflex            = Pflex + arr     # sum into total Pflex

    Ctank  = max(cfg.hw_volume_l * 1.163e-3, 1e-9)  # kWh/°C
    # Thermal resistance of tank: R_tank [°C/kW] from nominal heat loss
    # evaluated at (T_init − indoor_amb).  Heat is exchanged with the
    # INDOOR plant-room air (cfg.hw_indoor_amb_c), not outdoor weather.
    _hl_nom_h  = cfg.hw_heat_loss_w / 1000.0          # nominal heat loss [kW]
    _T_ref_h   = float(cfg.hw_indoor_amb_c)           # indoor ambient [°C]
    R_tank_h   = (max(cfg.hw_T_init_c, _T_ref_h + 1.0) - _T_ref_h) / max(_hl_nom_h, 1e-9)
    gamma_t_h  = dt / (Ctank * R_tank_h)              # dimensionless step decay
    Qdraw_h    = float(cfg.hw_draw_kw)                 # constant hot-water draw [kW]
    alpha  = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)
    beta   = dt / max(cfg.Cth_kwh_per_c, 1e-9)

    for k in range(H):
        # ── PV ───────────────────────────────────────────────────────────────
        if cfg.pv_enabled:
            # When export is enabled, take the full forecast (excess sells
            # to grid at spot price). When disabled, curtail when price < 0.
            if cfg.pv_export_enabled:
                Ppv[k] = float(inputs.Ppv_forecast_kw[k])
            else:
                _lam_k = float(inputs.price_eur_kwh[k]) + cfg.fee_eur_kwh
                Ppv[k] = 0.0 if _lam_k < 0 else float(inputs.Ppv_forecast_kw[k])
        else:
            Ppv[k] = 0.0

        # ── CHP (spark-spread rule) ──────────────────────────────────────────
        if cfg.chp_enabled:
            _Fchp_max_h = cfg.Fchp_max_m3_h  # use the resolved gas cap directly
            elec_val_per_m3 = cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec * float(price_total[k])
            spark = elec_val_per_m3 - cfg.gas_price_chp_eur_m3
            if spark > 0:
                Fchp[k]  = _Fchp_max_h
                zchp[k]  = 1.0
            else:
                Fchp[k]  = 0.0
                zchp[k]  = 0.0
            Pchp[k] = Fchp[k] * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec
            Qchp[k] = Fchp[k] * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat

        T_prev = Tbuilding[k]
        _T_heat_set = _active_heat_setpoint(k, dt, cfg, start_hour=start_hour,
                                            start_weekday=start_weekday)
        # Natural drift (Newton's law)
        T_drift = (1.0 - alpha) * T_prev + alpha * float(inputs.Tamb_c[k])
        heat_deficit_kw = max(0.0, (_T_heat_set - T_drift) * cfg.Cth_kwh_per_c / dt)
        # If we are already in cooling territory the CHP waste heat is
        # unwanted: skip running it this step regardless of spark-spread.
        _, _T_cool_lim_now = _active_comfort_bounds(k, dt, cfg, start_hour=start_hour,
                                                    start_weekday=start_weekday)
        if cfg.cooling_enabled and cfg.hp_enabled and T_drift >= _T_cool_lim_now - 0.5:
            Fchp[k] = 0.0
            zchp[k] = 0.0
            Pchp[k] = 0.0
            Qchp[k] = 0.0
        heat_deficit_kw -= Qchp[k]   # CHP provides free heat
        heat_deficit_kw  = max(0.0, heat_deficit_kw)

        # Satisfy heat demand: use cheapest heater first (cost-aware)
        if heat_deficit_kw > 0:
            _c_boiler = (cfg.gas_price_boiler_eur_kwh
                         / max(cfg.eta_boiler, 1e-9))
            _c_hp = price_total[k] / max(COP[k], 1e-3)
            _hp_first = cfg.hp_enabled and (
                not cfg.boiler_enabled or _c_hp <= _c_boiler
            )
            if _hp_first:
                Php[k] = min(cfg.Php_max_kw, heat_deficit_kw)
                heat_deficit_kw -= Php[k]
                if cfg.boiler_enabled and heat_deficit_kw > 0:
                    Pgas[k] = min(cfg.Pgas_max_kw, heat_deficit_kw)
            else:
                if cfg.boiler_enabled:
                    Pgas[k] = min(cfg.Pgas_max_kw, heat_deficit_kw)
                    heat_deficit_kw -= Pgas[k]
                if cfg.hp_enabled and heat_deficit_kw > 0:
                    Php[k] = min(cfg.Php_max_kw, heat_deficit_kw)

        # Update building temperature
        Q_heat = Php[k] + Pgas[k] + Qchp[k]
        # Cooling: if building is above the comfort ceiling, run cooling
        if cfg.cooling_enabled and cfg.hp_enabled:
            _, _T_cool_lim = _active_comfort_bounds(k, dt, cfg, start_hour=start_hour,
                                                    start_weekday=start_weekday)
            if T_prev > _T_cool_lim:
                Php_cool[k] = min(cfg.Php_cool_max_kw,
                                  (T_prev - _T_cool_lim) * cfg.Cth_kwh_per_c / dt)
        Tbuilding[k + 1] = (1.0 - alpha) * T_prev + beta * (Q_heat - Php_cool[k]) + alpha * float(inputs.Tamb_c[k])

        # ── Hot water tank (Newton's cooling + hot water draw) ───────────────
        if cfg.hw_enabled:
            T_tank_prev = Ttank[k]
            _ta_k = _T_ref_h
            _Qd_k = Qdraw_h
            # Natural drift: cooling + draw with no heater
            T_nat = ((1.0 - gamma_t_h) * T_tank_prev
                     + (dt / Ctank) * (_ta_k / R_tank_h - _Qd_k))
            if T_nat < cfg.hw_T_min_c:
                # Heat toward T_max (pre-heat while it's available)
                heat_needed = (cfg.hw_T_max_c - T_tank_prev) * Ctank / dt
                Ptank[k] = min(cfg.Ptank_max_kw, max(0.0, heat_needed))
            Ttank[k + 1] = ((1.0 - gamma_t_h) * T_tank_prev
                            + (dt / Ctank) * (Ptank[k] + _ta_k / R_tank_h - _Qd_k))
            Ttank[k + 1] = float(np.clip(Ttank[k + 1], 0.0, cfg.hw_T_max_c))

        # ── Battery ──────────────────────────────────────────────────────────
        if cfg.bat_enabled:
            soc = SOC[k]
            if price_total[k] <= p_low and soc < cfg.SOC_cap_kwh:
                Pch[k] = min(cfg.Pch_max_kw, (cfg.SOC_cap_kwh - soc) / (cfg.eta_ch * dt))
            elif price_total[k] >= p_high and soc > cfg.SOC_min_kwh:
                Pdis[k] = min(cfg.Pdis_max_kw, (soc - cfg.SOC_min_kwh) * cfg.eta_dis / dt)
            SOC[k + 1] = soc + (cfg.eta_ch * Pch[k] - Pdis[k] / cfg.eta_dis) * dt
            SOC[k + 1] = float(np.clip(SOC[k + 1], cfg.SOC_min_kwh, cfg.SOC_cap_kwh))

        # ── Grid residual (power balance) ────────────────────────────────────
        Php_elec_k = Php[k] / max(COP[k], 1e-3) if cfg.hp_enabled else 0.0
        Php_cool_elec_k = (Php_cool[k] / max(cfg.COP_cool, 1e-3)
                           if cfg.cooling_enabled and cfg.hp_enabled else 0.0)
        net_load = (float(inputs.Pload_kw[k])
                    + Php_elec_k
                    + Php_cool_elec_k
                    + Ptank[k]
                    + Pflex[k]
                    + Pch[k]
                    - Ppv[k]
                    - Pchp[k]
                    - Pdis[k])
        if net_load >= 0.0:
            Pgrid[k]   = net_load
            Pexport[k] = 0.0
        else:
            Pgrid[k]   = 0.0
            if cfg.pv_enabled and cfg.pv_export_enabled:
                # Sell excess back to grid at spot price
                Pexport[k] = -net_load
            else:
                # No export contract → curtail PV to match demand
                Ppv[k] = float(Ppv[k]) + net_load

    return {
        "Pgrid": Pgrid, "Php": Php, "Pgas": Pgas,
        "Pch": Pch, "Pdis": Pdis, "SOC": SOC,
        "Pflex": Pflex, "Ppv": Ppv, "Pexport": Pexport,
        "Pflex_by_id": Pflex_by_id,
        "Ptank": Ptank, "Ttank": Ttank,
        "Fchp": Fchp, "Pchp": Pchp, "Qchp": Qchp,
        "zchp": zchp, "ychp": np.zeros(H),
        "Tbuilding": Tbuilding, "Php_cool": Php_cool,
        "solver": "heuristic", "status": "ok", "obj_value": float("nan"),
    }


# ===========================================================================
# SECTION 5 — CVXPY LP/MILP SOLVER
# ===========================================================================

def _solve_cvxpy(
    inputs: MPCInputs,
    cfg: MPCConfig,
    COP: np.ndarray,
    H: int,
    dt: float,
    start_hour: float = 0.0,
    start_weekday: int | None = None,
) -> Optional[dict]:
    """
    Build and solve the MPC LP/MILP with cvxpy.

    Returns a dict with all solution arrays, or None if the solver fails.

    PROBLEM STRUCTURE
    -----------------
    Continuous variables (always present):
        Pgrid, slack_peak, slack_Tmin, slack_Tmax, slack_tank, Tbuilding
    Conditional continuous variables (enabled by config):
        Php, Pgas, Pch, Pdis, SOC, Pflex, Ppv, Ptank, Ttank, Fchp
    Integer variables (only when CHP is enabled and use_milp=True):
        zchp, ychp

    Solver selection:
        No CHP    → CLARABEL (primal LP/QP, very fast)
        CHP MILP  → GLPK_MI → CBC → SCIP → (fallback: LP relaxation via CLARABEL)
        CHP relax → CLARABEL
    """

    # ── Decision variables ─────────────────────────────────────────────────
    Pgrid      = cp.Variable(H, nonneg=True, name="Pgrid")

    # Building temperature: H+1 values (index 0 = initial, fixed by constraint)
    Tbuilding  = cp.Variable(H + 1, name="Tbuilding")

    # Conditional variables — disabled subsystems use numpy zeros (constants)
    Php   = cp.Variable(H, nonneg=True, name="Php")   if cfg.hp_enabled      else np.zeros(H)
    Php_cool = cp.Variable(H, nonneg=True, name="Php_cool") if (cfg.cooling_enabled and cfg.hp_enabled) else np.zeros(H)
    Pgas  = cp.Variable(H, nonneg=True, name="Pgas")  if cfg.boiler_enabled  else np.zeros(H)
    Ppv   = cp.Variable(H, nonneg=True, name="Ppv")   if cfg.pv_enabled      else np.zeros(H)
    # PV export to grid (revenue at spot price, no grid fee)
    if cfg.pv_enabled and cfg.pv_export_enabled:
        Pexport = cp.Variable(H, nonneg=True, name="Pexport")
    else:
        Pexport = np.zeros(H)

    # ── Flexible load(s) ───────────────────────────────────────────────────
    # When per-instance ``flex_loads`` are configured, create one variable
    # per load so each gets its own window/energy/ramp constraints.  The
    # power balance sums them via ``Pflex``.  When no per-instance list is
    # given we fall back to a single legacy ``Pflex`` variable.
    Pflex_vars: list = []      # list of cp.Variable, one per flex load
    Pflex_ids:  list = []      # parallel list of instance ids
    if cfg.flex_enabled and cfg.flex_loads:
        for _i, _fl in enumerate(cfg.flex_loads):
            _v = cp.Variable(H, nonneg=True, name=f"Pflex_{_i}")
            Pflex_vars.append(_v)
            Pflex_ids.append(_fl.get("id", f"flex_{_i}"))
        Pflex = sum(Pflex_vars)   # cvxpy affine expression
    elif cfg.flex_enabled:
        Pflex = cp.Variable(H, nonneg=True, name="Pflex")
        Pflex_vars.append(Pflex)
        Pflex_ids.append("flexible_load")
    else:
        Pflex = np.zeros(H)

    if cfg.bat_enabled:
        Pch  = cp.Variable(H, nonneg=True, name="Pch")
        Pdis = cp.Variable(H, nonneg=True, name="Pdis")
        SOC  = cp.Variable(H + 1,          name="SOC")
        z_bat = cp.Variable(H, name="z_bat")  # LP relaxation: continuous [0,1], 1=charging
    else:
        Pch = Pdis = np.zeros(H)
        SOC = None
        z_bat = None

    if cfg.hw_enabled:
        Ptank = cp.Variable(H, nonneg=True, name="Ptank")
        Ttank = cp.Variable(H + 1,          name="Ttank")
    else:
        Ptank = np.zeros(H)
        Ttank = None

    if cfg.chp_enabled:
        Fchp = cp.Variable(H, nonneg=True, name="Fchp")
        if cfg.chp_use_milp:
            zchp = cp.Variable(H, boolean=True, name="zchp")
            ychp = cp.Variable(H, boolean=True, name="ychp")
        else:
            # LP relaxation: continuous [0, 1]
            zchp = cp.Variable(H, nonneg=True, name="zchp")
            ychp = cp.Variable(H, nonneg=True, name="ychp")
    else:
        Fchp = zchp = ychp = np.zeros(H)

    # ── Derived CHP power (linear in Fchp) ────────────────────────────────
    # Fchp [m³/h] × HV [kWh/m³] × η = [kW]
    if cfg.chp_enabled:
        Pchp = Fchp * (cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec)
        Qchp = Fchp * (cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat)
        # CHP heat-dump (external radiator): when CHP runs for electricity
        # revenue the building may not need (all of) the waste heat. The
        # nonneg dump variable lets the LP shed heat externally instead of
        # overheating the building. Disable via cfg.chp_heat_dump_enabled
        # to model installations without a dump radiator.
        if cfg.chp_heat_dump_enabled:
            Qchp_dump = cp.Variable(H, nonneg=True, name="Qchp_dump")
        else:
            Qchp_dump = np.zeros(H)
    else:
        Pchp = np.zeros(H)
        Qchp = np.zeros(H)
        Qchp_dump = np.zeros(H)

    # ── HP electrical input: Php_elec = Php / COP  (linear, COP is a param) ─
    # COP[k] is a precomputed numpy array → 1/COP[k] is a numpy array.
    COP_inv = 1.0 / COP  # shape (H,)
    if cfg.hp_enabled:
        Php_elec = cp.multiply(COP_inv, Php)   # elementwise product
    else:
        Php_elec = np.zeros(H)
    if cfg.cooling_enabled and cfg.hp_enabled:
        Php_cool_elec = Php_cool * (1.0 / max(cfg.COP_cool, 1e-3))
    else:
        Php_cool_elec = np.zeros(H)

    # ── Constraints ────────────────────────────────────────────────────────
    constraints = []

    # 1. Electrical power balance (vectorized):
    #    Pgrid + Ppv + Pchp + Pdis = Pload + Php_elec + Ptank + Pflex + Pch + Pexport
    lhs = Pgrid + Ppv + Pchp
    rhs = inputs.Pload_kw + Php_elec + Php_cool_elec + Ptank + Pflex
    if cfg.bat_enabled:
        lhs = lhs + Pdis
        rhs = rhs + Pch
    if cfg.pv_enabled and cfg.pv_export_enabled:
        rhs = rhs + Pexport
    constraints.append(lhs == rhs)

    # 2. Grid peak limit (hard):  Pgrid ≤ Pgrid_max
    constraints.append(Pgrid <= cfg.Pgrid_max_kw)

    # 3. PV upper bound
    if cfg.pv_enabled:
        constraints.append(Ppv <= inputs.Ppv_forecast_kw)
        # Export bounded by PV forecast (cannot resell battery-stored grid energy).
        # This also prevents grid→export arbitrage cycles when the LP is
        # numerically indifferent between several optima.
        if cfg.pv_export_enabled:
            constraints.append(Pexport <= inputs.Ppv_forecast_kw)

    # 4. Heat pump upper bound
    if cfg.hp_enabled:
        constraints.append(Php <= cfg.Php_max_kw)
        # Ramp-rate constraints [% of Php_max_kw per step]
        # The initial condition (Php_prev_kw) ties the first LP step to the
        # actual previous HP output so the solver anticipates ramp-up cost
        # and starts climbing *before* cheap windows, not during them.
        if H > 1 and cfg.hp_ramp_pct < 100.0:
            _hp_ramp_kw = cfg.hp_ramp_pct / 100.0 * cfg.Php_max_kw
            _php_prev   = float(getattr(inputs, 'Php_prev_kw', 0.0))
            constraints.append(Php[0]  - _php_prev  <=  _hp_ramp_kw)  # initial ramp up
            constraints.append(_php_prev  - Php[0]  <=  _hp_ramp_kw)  # initial ramp down
            constraints.append(Php[1:] - Php[:-1]   <=  _hp_ramp_kw)
            constraints.append(Php[:-1] - Php[1:]   <=  _hp_ramp_kw)

    # 4b. Cooling upper bound
    if cfg.cooling_enabled and cfg.hp_enabled:
        constraints.append(Php_cool <= cfg.Php_cool_max_kw)

    # 5. Gas boiler upper bound
    if cfg.boiler_enabled:
        constraints.append(Pgas <= cfg.Pgas_max_kw)
        # Ramp-rate constraints [% of Pgas_max_kw per step]
        if H > 1 and cfg.boiler_ramp_pct < 100.0:
            _boiler_ramp_kw = cfg.boiler_ramp_pct / 100.0 * cfg.Pgas_max_kw
            constraints.append(Pgas[1:] - Pgas[:-1] <=  _boiler_ramp_kw)
            constraints.append(Pgas[:-1] - Pgas[1:] <=  _boiler_ramp_kw)

    # 6. Battery SOC dynamics
    if cfg.bat_enabled:
        constraints.append(SOC[0] == cfg.SOC_init_kwh)
        constraints.append(SOC >= cfg.SOC_min_kwh)
        constraints.append(SOC <= cfg.SOC_cap_kwh)
        constraints.append(z_bat >= 0)                              # LP relaxation bounds
        constraints.append(z_bat <= 1)
        constraints.append(Pch  <= cfg.Pch_max_kw  * z_bat)        # Pch·Pdis = 0
        constraints.append(Pdis <= cfg.Pdis_max_kw * (1 - z_bat))  # complementarity
        # SOC[k+1] = SOC[k] + (η_ch·Pch[k] − Pdis[k]/η_dis)·dt
        constraints.append(
            SOC[1:] == SOC[:-1] + (cfg.eta_ch * Pch - Pdis / cfg.eta_dis) * dt
        )
        # Ramp-rate constraints [% of respective max power per step]
        if H > 1 and cfg.bat_ramp_pct < 100.0:
            _bat_ch_ramp  = cfg.bat_ramp_pct / 100.0 * cfg.Pch_max_kw
            _bat_dis_ramp = cfg.bat_ramp_pct / 100.0 * cfg.Pdis_max_kw
            constraints.append(Pch[1:]  - Pch[:-1]  <=  _bat_ch_ramp)
            constraints.append(Pch[:-1] - Pch[1:]   <=  _bat_ch_ramp)
            constraints.append(Pdis[1:] - Pdis[:-1] <=  _bat_dis_ramp)
            constraints.append(Pdis[:-1] - Pdis[1:] <=  _bat_dis_ramp)

    # 7. Flexible load(s): per-instance power cap, time window, energy target, ramp rates
    if cfg.flex_enabled and Pflex_vars:
        horizon_h = H * dt
        for _i, _var in enumerate(Pflex_vars):
            if cfg.flex_loads:
                _fl = cfg.flex_loads[_i]
                _pmax       = float(_fl["Pmax_kw"])
                _daily_kwh  = float(_fl["daily_kwh"])
                _t_start    = str  (_fl["time_start"])
                _t_end      = str  (_fl["time_end"])
                _ramp_up    = float(_fl["ramp_up_kw"])
                _ramp_down  = float(_fl["ramp_down_kw"])
            else:
                # Legacy single-flex (no per-instance list)
                _pmax       = cfg.Pflex_max_kw
                _daily_kwh  = cfg.flex_daily_kwh
                _t_start    = cfg.flex_time_start
                _t_end      = cfg.flex_time_end
                _ramp_up    = cfg.flex_ramp_up_kw
                _ramp_down  = cfg.flex_ramp_down_kw

            _flex_days = list(_fl.get("active_days", [])) if cfg.flex_loads else None
            _flex_active = _compute_flex_active_mask(
                _t_start, _t_end, H, dt, start_hour=start_hour,
                start_weekday=start_weekday, active_days=_flex_days,
            )
            constraints.append(_var <= _pmax)
            for _k in range(H):
                if not _flex_active[_k]:
                    constraints.append(_var[_k] == 0)
            target_kwh    = _daily_kwh * (horizon_h / 24.0)
            max_in_window = float(np.sum(_flex_active)) * _pmax * dt
            constraints.append(cp.sum(_var) * dt == min(target_kwh, max_in_window))
            if H > 1:
                if _ramp_up   < 9000.0:
                    constraints.append(_var[1:]  - _var[:-1] <= _ramp_up)
                if _ramp_down < 9000.0:
                    constraints.append(_var[:-1] - _var[1:]  <= _ramp_down)

    # 8. Hot water tank dynamics — Newton's cooling
    #    Ttank[k+1] = (1−γ)·Ttank[k] + (dt/C)·(Ptank[k] + Tamb_in/R_tank − Qdraw)
    #
    #    Heat is exchanged with INDOOR plant-room air (cfg.hw_indoor_amb_c),
    #    NOT outdoor weather.  Water draw is constant at cfg.hw_draw_kw.
    if cfg.hw_enabled:
        Ctank   = max(cfg.hw_volume_l * 1.163e-3, 1e-9)   # kWh/°C
        _hl_nom = cfg.hw_heat_loss_w / 1000.0             # nominal heat loss [kW]
        _T_ref  = float(cfg.hw_indoor_amb_c)              # indoor ambient [°C]
        # R_tank calibrated so that at T = T_init the standby loss = hw_heat_loss_w
        R_tank  = (max(cfg.hw_T_init_c, _T_ref + 1.0) - _T_ref) / max(_hl_nom, 1e-9)  # °C/kW
        gamma_t = dt / (Ctank * R_tank)                   # dimensionless step decay
        Qdraw   = float(cfg.hw_draw_kw)                   # constant hot-water draw [kW]
        _dt_c   = dt / Ctank
        constraints.append(Ttank[0] == inputs.T_tank_init_c)
        constraints.append(Ptank <= cfg.Ptank_max_kw)
        constraints.append(Ttank[1:] <= cfg.hw_T_max_c)
        # Ramp-rate constraints [% of Ptank_max_kw per step]
        if H > 1 and cfg.hw_ramp_pct < 100.0:
            _hw_ramp_kw = cfg.hw_ramp_pct / 100.0 * cfg.Ptank_max_kw
            constraints.append(Ptank[1:] - Ptank[:-1] <=  _hw_ramp_kw)
            constraints.append(Ptank[:-1] - Ptank[1:] <=  _hw_ramp_kw)
        # Dynamics: linear in Ttank and Ptank
        constraints.append(
            Ttank[1:] == (1.0 - gamma_t) * Ttank[:-1]
                       + _dt_c * (Ptank + _T_ref / R_tank - Qdraw)
        )
        # Per-step max-reachable lower bound (prevents infeasibility)
        _T_tk_max_r = np.empty(H + 1)
        _T_tk_max_r[0] = inputs.T_tank_init_c
        for _k in range(H):
            _T_tk_max_r[_k + 1] = min(
                (1.0 - gamma_t) * _T_tk_max_r[_k]
                    + _dt_c * (cfg.Ptank_max_kw + _T_ref / R_tank - Qdraw),
                cfg.hw_T_max_c,
            )
        _eff_tank_lb = np.minimum(cfg.hw_T_min_c, _T_tk_max_r[1:])
        constraints.append(Ttank[1:] >= _eff_tank_lb)

    # 9. Building temperature dynamics
    #    Tbuilding[k+1] = (1−α)·Tbuilding[k] + β·Q_in[k] + α·Tamb[k]
    alpha = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)   # dimensionless
    beta  = dt / max(cfg.Cth_kwh_per_c, 1e-9)                      # °C/kW/step
    constraints.append(Tbuilding[0] == inputs.T_building_init_c)
    Q_in = Php + Pgas + Qchp - Qchp_dump - Php_cool   # net thermal input (heating minus cooling) [kW]
    constraints.append(
        Tbuilding[1:] == (1.0 - alpha) * Tbuilding[:-1]
                         + beta * Q_in
                         + alpha * inputs.Tamb_c
    )
    # Per-step physically achievable comfort bounds.
    # Prevents LP infeasibility when T_init is outside [Tmin, Tmax] or when
    # heating/cooling capacity cannot reach the bound in one step.
    # T_max_r[k] = max reachable temperature at step k (full heating every step).
    # T_min_r[k] = min reachable temperature at step k (full cooling every step).
    _P_heat_max = ((cfg.Php_max_kw if cfg.hp_enabled else 0.0)
                   + (cfg.Pgas_max_kw if cfg.boiler_enabled else 0.0))
    _P_cool_max = cfg.Php_cool_max_kw if (cfg.cooling_enabled and cfg.hp_enabled) else 0.0
    _T_max_r = np.empty(H + 1)
    _T_min_r = np.empty(H + 1)
    _T_max_r[0] = _T_min_r[0] = inputs.T_building_init_c
    for _k in range(H):
        _ta = float(inputs.Tamb_c[_k])
        _T_max_r[_k + 1] = (1.0 - alpha) * _T_max_r[_k] + beta * _P_heat_max + alpha * _ta
        _T_min_r[_k + 1] = (1.0 - alpha) * _T_min_r[_k] - beta * _P_cool_max + alpha * _ta
    # Per-step comfort bounds — optionally tighter at night (night setback)
    _T_lb_k = np.empty(H)
    _T_ub_k = np.empty(H)
    for _k in range(H):
        _T_lb_k[_k], _T_ub_k[_k] = _active_comfort_bounds(_k, dt, cfg,
                                                          start_hour=start_hour,
                                                          start_weekday=start_weekday)
    _eff_lb = np.maximum(_T_lb_k, _T_min_r[1:])   # enforce comfort floor unless physically impossible
    # Upper bound is enforced as a SOFT constraint with a heavy quadratic
    # penalty (see ``comfort_pen`` in the objective). This forces the LP to
    # cool maximally and only overshoot when physically necessary, instead
    # of treating the bound as free to violate.
    slack_Tmax = cp.Variable(H, nonneg=True, name="slack_Tmax")
    constraints.append(Tbuilding[1:] >= _eff_lb)
    constraints.append(Tbuilding[1:] <= _T_ub_k + slack_Tmax)

    # 10. CHP constraints
    if cfg.chp_enabled:
        # Heat dump cannot exceed the actual CHP waste heat (Qchp_dump >= 0
        # is already enforced by nonneg=True at variable declaration).
        constraints.append(Qchp_dump <= Qchp)
        # Gas flow cap: Fchp[k] ≤ Fchp_max_m3_h·zchp[k]
        # Fchp_max_m3_h is the minimum of the explicit gas cap and the cap
        # derived from Pchp_max_kw, so both JSON fields are respected.
        constraints.append(Fchp <= cfg.Fchp_max_m3_h * zchp)
        constraints.append(zchp <= 1)
        constraints.append(ychp <= 1)
        # Startup indicator: ychp[k] ≥ zchp[k] − zchp[k−1]
        constraints.append(ychp[0] >= zchp[0])
        if H > 1:
            constraints.append(ychp[1:] >= zchp[1:] - zchp[:-1])
        # Ramp-rate constraints on CHP gas flow [% of Pchp_max_kw per step,
        # mapped to gas flow units via HV × η_elec]
        if H > 1 and cfg.chp_ramp_pct < 100.0:
            _hv_e = cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec
            _chp_ramp_m3 = cfg.chp_ramp_pct / 100.0 * cfg.Pchp_max_kw / max(_hv_e, 1e-9)
            constraints.append(Fchp[1:] - Fchp[:-1] <=  _chp_ramp_m3)
            constraints.append(Fchp[:-1] - Fchp[1:] <=  _chp_ramp_m3)

    # ── Objective ──────────────────────────────────────────────────────────
    # Electricity cost: (λ_spot[k] + fee) × Pgrid[k] × dt
    lambda_e_total = inputs.price_eur_kwh + cfg.fee_eur_kwh  # (H,)
    elec_cost = cp.sum(cp.multiply(lambda_e_total, Pgrid)) * dt

    # PV export revenue at spot price (no grid fee on injected energy).
    # When spot price is negative the LP naturally sets Pexport=0 (loss-making).
    if cfg.pv_enabled and cfg.pv_export_enabled:
        export_revenue = cp.sum(cp.multiply(inputs.price_eur_kwh, Pexport)) * dt
    else:
        export_revenue = 0.0

    # Capacity tariff — quadratic penalty: ε_L · Σ_k (max(0, Pgrid[k] − P_lim))²
    # Penalises grid import above the configurable limit P_lim with a quadratic
    # cost, turning the LP into a QP.  Both CLARABEL and SCS handle QPs natively.
    # Enabled only when both Plim and epsilon_L are > 0.
    if cfg.cap_tariff_Plim_kw > 0.0 and cfg.cap_tariff_epsilon_l > 0.0:
        cap_tariff_cost = (
            cfg.cap_tariff_epsilon_l
            * cp.sum(cp.square(cp.pos(Pgrid - cfg.cap_tariff_Plim_kw)))
        )
    else:
        cap_tariff_cost = 0.0

    # CHP gas cost: λ_gas_chp × Fchp[k] × dt  (m³/h → m³ per step)
    chp_gas_cost = (
        cp.sum(Fchp) * cfg.gas_price_chp_eur_m3 * dt
        if cfg.chp_enabled else 0.0
    )

    # Boiler gas cost: (λ_gas_boiler / η_boiler) × Pgas[k] × dt
    #   = marginal gas cost per kWh of thermal output
    c_boiler = cfg.gas_price_boiler_eur_kwh / max(
        cfg.eta_boiler, 1e-9
    )  # [€/kWh thermal]
    boiler_gas_cost = (
        cp.sum(Pgas) * c_boiler * dt
        if cfg.boiler_enabled else 0.0
    )

    # CHP startup cost
    startup_cost = (
        cp.sum(ychp) * cfg.chp_startup_cost_eur
        if cfg.chp_enabled else 0.0
    )

    # Comfort overshoot penalty: heavy linear cost on any Tmax violation.
    # Kept linear so the MILP-capable LP solvers (HiGHS / GLPK_MI / CBC)
    # remain applicable. Weight is large enough (1000 EUR per deg C per step)
    # to dominate any fuel revenue, so the LP always cools to its physical
    # limit and overshoots only when physically unavoidable.
    comfort_pen = 1000.0 * cp.sum(slack_Tmax)

    objective = cp.Minimize(
        elec_cost + cap_tariff_cost + chp_gas_cost + boiler_gas_cost + startup_cost
        + comfort_pen
        - export_revenue
    )

    problem = cp.Problem(objective, constraints)

    # ── Solve ──────────────────────────────────────────────────────────────
    use_milp = cfg.chp_enabled and cfg.chp_use_milp
    solver_name = "none"

    if use_milp:
        # Try MILP-capable solvers in order of preference (HIGHS is available by default)
        # time_limit prevents the solver hanging on hard instances.
        for _s, _kwargs in [
            (cp.HIGHS,   {"time_limit": 120.0}),   # correct HIGHS key (lowercase)
            (cp.GLPK_MI, {"tm_lim": 120000}),
            (cp.CBC,     {"maximumSeconds": 120}),
            (cp.SCIP,    {}),
        ]:
            try:
                problem.solve(solver=_s, verbose=False, **_kwargs)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    solver_name = str(_s)
                    break
            except Exception:
                continue
        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning("All MILP solvers failed (status=%s)", problem.status)
            return None
    else:
        # Pure LP / QP — CLARABEL is fastest
        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
            solver_name = "clarabel"
        except Exception:
            try:
                problem.solve(solver=cp.SCS)
                solver_name = "scs"
            except Exception:
                return None

    if problem.status not in ("optimal", "optimal_inaccurate"):
        logger.warning("MPC LP infeasible/unbounded (status=%s)", problem.status)
        return None

    # ── Extract solution ────────────────────────────────────────────────────
    def _v(var) -> np.ndarray:
        """Extract and clip to ≥ 0 (for power variables)."""
        if isinstance(var, np.ndarray):
            return var.copy()
        return np.maximum(0.0, np.array(var.value, dtype=float).ravel())

    def _vf(var) -> np.ndarray:
        """Extract without clipping (for temperature / SOC variables)."""
        if isinstance(var, np.ndarray):
            return var.copy()
        return np.array(var.value, dtype=float).ravel()

    Pgrid_sol     = _v(Pgrid)
    Php_sol       = _v(Php)
    Pgas_sol      = _v(Pgas)
    Pch_sol       = _v(Pch)
    Pdis_sol      = _v(Pdis)
    SOC_sol       = _vf(SOC)  if cfg.bat_enabled  else np.full(H + 1, cfg.SOC_init_kwh)
    Pflex_sol     = _v(Pflex) if isinstance(Pflex, np.ndarray) else _v(Pflex)
    # Per-instance flex schedules (parallel to Pflex_ids).  When the legacy
    # single-Pflex path is used this dict has one entry keyed "flexible_load".
    Pflex_by_id_sol: dict = {}
    if Pflex_vars:
        for _i, _var in enumerate(Pflex_vars):
            try:
                _arr = np.maximum(0.0, np.array(_var.value, dtype=float).ravel())
            except Exception:
                _arr = np.zeros(H)
            Pflex_by_id_sol[Pflex_ids[_i]] = _arr
    Ppv_sol       = _v(Ppv)
    Pexport_sol   = _v(Pexport) if (cfg.pv_enabled and cfg.pv_export_enabled) else np.zeros(H)
    Ptank_sol     = _v(Ptank)
    Ttank_sol     = _vf(Ttank) if cfg.hw_enabled  else np.full(H + 1, cfg.hw_T_init_c)
    Fchp_sol      = _v(Fchp)
    zchp_sol      = np.round(_v(zchp)) if cfg.chp_enabled else np.zeros(H)
    ychp_sol      = np.round(_v(ychp)) if cfg.chp_enabled else np.zeros(H)
    Tbuilding_sol = _vf(Tbuilding)
    Php_cool_sol  = _v(Php_cool) if (cfg.cooling_enabled and cfg.hp_enabled) else np.zeros(H)

    Pchp_sol = Fchp_sol * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec
    Qchp_sol = Fchp_sol * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat

    return {
        "Pgrid": Pgrid_sol, "Php": Php_sol, "Pgas": Pgas_sol,
        "Pch": Pch_sol, "Pdis": Pdis_sol, "SOC": SOC_sol,
        "Pflex": Pflex_sol, "Ppv": Ppv_sol, "Pexport": Pexport_sol,
        "Pflex_by_id": Pflex_by_id_sol,
        "Ptank": Ptank_sol, "Ttank": Ttank_sol,
        "Fchp": Fchp_sol, "Pchp": Pchp_sol, "Qchp": Qchp_sol,
        "zchp": zchp_sol, "ychp": ychp_sol,
        "Tbuilding": Tbuilding_sol, "Php_cool": Php_cool_sol,
        "solver": solver_name, "status": problem.status,
        "obj_value": float(problem.value) if problem.value is not None else float("nan"),
    }


# ===========================================================================
# SECTION 6 — BASELINE COST SIMULATION
# ===========================================================================

def compute_baseline_cost(inputs: MPCInputs, cfg: MPCConfig, start_hour: float = 0.0,
                          start_weekday: int | None = None) -> float:
    """
    Return total baseline cost [€] for the horizon.

    The baseline simulation logic is implemented in compute_baseline_arrays().
    This wrapper intentionally delegates to that function so KPI cost values
    always use the exact same physics and dispatch assumptions as the arrays
    shown in analysis views.
    """
    return float(compute_baseline_arrays(inputs, cfg, start_hour=start_hour,
                                         start_weekday=start_weekday)["total_cost"])


def compute_baseline_arrays(inputs: MPCInputs, cfg: MPCConfig, start_hour: float = 0.0,
                            start_weekday: int | None = None) -> dict:
    """
    Same simulation as compute_baseline_cost() but returns per-asset arrays.

    Returns
    -------
    dict with keys:
        total_cost  — float [€]
        Pgrid, Php, Pgas, Ptank, Pflex, Ppv, Fchp, Pchp — np.ndarray (H,)
        Tbuilding, Ttank — np.ndarray (H+1,)
    """
    H   = cfg.horizon_steps
    dt  = cfg.dt_hours

    COP      = compute_cop(inputs.Tamb_c[:H], cfg)
    lambda_e = inputs.price_eur_kwh[:H] + cfg.fee_eur_kwh
    c_boiler = cfg.gas_price_boiler_eur_kwh / max(cfg.eta_boiler, 1e-9)

    Pgrid = np.zeros(H)
    Pexport = np.zeros(H)
    Php   = np.zeros(H)
    Pgas  = np.zeros(H)
    Ptank = np.zeros(H)
    Pflex = np.zeros(H)
    Ppv   = np.zeros(H)
    Fchp  = np.zeros(H)

    Tbuilding    = np.zeros(H + 1)
    Ttank        = np.zeros(H + 1)
    Tbuilding[0] = inputs.T_building_init_c
    Ttank[0]     = inputs.T_tank_init_c

    alpha = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)
    beta  = dt / max(cfg.Cth_kwh_per_c, 1e-9)

    Ctank   = max(cfg.hw_volume_l * 1.163e-3, 1e-9)
    _hl_nom = cfg.hw_heat_loss_w / 1000.0
    _T_ref  = float(cfg.hw_indoor_amb_c)        # indoor plant-room temp (not outdoor)
    R_tank  = (max(cfg.hw_T_init_c, _T_ref + 1.0) - _T_ref) / max(_hl_nom, 1e-9)
    gamma_t = dt / (Ctank * R_tank)
    Qdraw   = float(cfg.hw_draw_kw)             # constant hot-water draw [kW]

    # on_off baseline: always use the asset's full rated capacity (Php_max_kw /
    # Pgas_max_kw / Ptank_max_kw) — a real thermostat runs at full power.
    # constant baseline: use the user-configured fixed power level.
    _php_bl   = cfg.hp_bl_power_kw      if cfg.hp_bl_power_kw      > 0 else cfg.Php_max_kw
    _pgas_bl  = cfg.boiler_bl_power_kw  if cfg.boiler_bl_power_kw  > 0 else cfg.Pgas_max_kw
    _phw_bl   = cfg.hw_bl_power_kw      if cfg.hw_bl_power_kw      > 0 else cfg.Ptank_max_kw

    if cfg.pv_enabled:
        if cfg.pv_export_enabled:
            # Excess sells back to grid at spot price — no need to curtail.
            Ppv = np.minimum(inputs.Ppv_forecast_kw[:H], cfg.pv_capacity_kwp)
        else:
            # Curtail PV at steps where the spot price is negative — injecting
            # surplus would increase cost rather than reduce it.
            _ppv_cap = np.where(
                inputs.price_eur_kwh[:H] + cfg.fee_eur_kwh < 0,
                0.0,
                cfg.pv_capacity_kwp,
            )
            Ppv = np.minimum(inputs.Ppv_forecast_kw[:H], _ppv_cap)

    # Baseline flex dispatch — sum the contribution from every configured load.
    # Each instance runs at its baseline power inside its own baseline window.
    if cfg.flex_enabled:
        if cfg.flex_loads:
            _flex_bl_specs = [
                (str(fl["bl_mode"]),
                 str(fl["bl_time_start"]), str(fl["bl_time_end"]),
                 float(fl["bl_power_kw"]), float(fl["Pmax_kw"]),
                 list(fl.get("baseline_active_days", [])))
                for fl in cfg.flex_loads
            ]
        else:
            _flex_bl_specs = [(
                str(cfg.flex_bl_mode),
                str(cfg.flex_bl_time_start), str(cfg.flex_bl_time_end),
                float(cfg.flex_bl_power_kw), float(cfg.Pflex_max_kw),
                None,
            )]
        for _bl_mode, _bl_ts, _bl_te, _bl_pwr, _pmax, _bl_days in _flex_bl_specs:
            if _bl_mode != "fixed_window":
                continue
            _bl_active = _compute_flex_active_mask(
                _bl_ts, _bl_te, H, dt, start_hour=start_hour,
                start_weekday=start_weekday, active_days=_bl_days,
            )
            _pflex_bl  = _bl_pwr if _bl_pwr > 0 else _pmax
            for _k in range(H):
                if _bl_active[_k]:
                    Pflex[_k] += min(_pflex_bl, _pmax)

    chp_heat_demand  = cfg.chp_enabled and cfg.chp_bl_mode == "heat_demand"
    chp_constant     = cfg.chp_enabled and cfg.chp_bl_mode == "constant"
    chp_ambient_temp = cfg.chp_enabled and cfg.chp_bl_mode == "ambient_temp"
    # Hysteresis state for heat-led CHP (start in the same state as building
    # heating: if T already below set-point, CHP turns on immediately).
    chp_on_bl = bool(
        chp_heat_demand and inputs.T_building_init_c < _active_heat_setpoint(
            0, dt, cfg, start_hour=start_hour, start_weekday=start_weekday) - 0.5
    )
    # Baseline follows configured day/night schedule, but uses the same lower
    # comfort floor as the LP for turn-on decisions.
    _Tset0_bl = _active_heat_setpoint(0, dt, cfg, start_hour=start_hour,
                                      start_weekday=start_weekday)
    _T_heat_lb0_bl, _ = _active_comfort_bounds(0, dt, cfg, start_hour=start_hour,
                                               start_weekday=start_weekday)
    heating_on = (
        inputs.T_building_init_c < _Tset0_bl - 0.5
        or inputs.T_building_init_c < _T_heat_lb0_bl
    )
    hw_on      = inputs.T_tank_init_c     < cfg.hw_T_min_c
    Php_cool   = np.zeros(H)

    for k in range(H):
        ta     = float(inputs.Tamb_c[k])
        T_prev = Tbuilding[k]

        # Active thermostat targets for this wall-clock step.
        _Tset_bl = _active_heat_setpoint(k, dt, cfg, start_hour=start_hour,
                                         start_weekday=start_weekday)
        _T_heat_lb_bl, _T_cool_lim_bl = _active_comfort_bounds(
            k, dt, cfg, start_hour=start_hour, start_weekday=start_weekday
        )

        if chp_heat_demand:
            # Hysteresis: on when T < Tset-0.5, off when T > Tset+0.5
            if T_prev < _Tset_bl - 0.5:
                chp_on_bl = True
            elif T_prev > _Tset_bl + 0.5:
                chp_on_bl = False
            Fchp[k] = cfg.Fchp_max_m3_h if chp_on_bl else 0.0
        elif chp_ambient_temp:
            # Heating-season operation: CHP runs at rated output whenever the
            # outdoor temperature is below the configured threshold.
            Fchp[k] = cfg.Fchp_max_m3_h if ta < cfg.chp_bl_Tamb_threshold_c else 0.0
        elif chp_constant:
            Fchp[k] = cfg.Fchp_max_m3_h

        Q_chp_k = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat
        Pchp_k  = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec

        # Keep baseline thermostat feasible vs LP: turn on at the same lower
        # comfort floor the LP enforces, or at setpoint deadband if tighter.
        if T_prev < max(_T_heat_lb_bl, _Tset_bl - 0.5):
            heating_on = True
        elif T_prev > _Tset_bl + 0.5:
            heating_on = False

        # Fair baseline vs MPC: if coasting would violate the active comfort floor
        # next step, force heating ON now (MPC also enforces per-step floor bounds).
        _T_next_idle = (
            (1.0 - alpha) * T_prev
            + beta * Q_chp_k
            + alpha * ta
        )
        if _T_next_idle < _T_heat_lb_bl:
            heating_on = True

        if heating_on:
            # Boiler-first (traditional building baseline): fire boiler as primary
            if cfg.boiler_bl_mode == "on_off" and cfg.boiler_enabled:
                Pgas[k] = min(_pgas_bl, cfg.Pgas_max_kw)
            elif cfg.boiler_bl_mode == "constant" and cfg.boiler_enabled:
                Pgas[k] = min(cfg.boiler_bl_power_kw, cfg.Pgas_max_kw)
            # HP supplements only if boiler alone cannot keep T above the
            # baseline turn-on threshold used above.
            if cfg.hp_bl_mode == "on_off" and cfg.hp_enabled:
                _T_next_boiler_only = (
                    (1.0 - alpha) * T_prev
                    + beta * (float(Pgas[k]) + Q_chp_k)
                    + alpha * ta
                )
                if _T_next_boiler_only < max(_T_heat_lb_bl, _Tset_bl - 0.5):
                    Php[k] = min(_php_bl, cfg.Php_max_kw)
            elif cfg.hp_bl_mode == "constant" and cfg.hp_enabled:
                Php[k] = min(cfg.hp_bl_power_kw, cfg.Php_max_kw)
        else:
            if cfg.hp_bl_mode == "constant" and cfg.hp_enabled:
                Php[k] = min(cfg.hp_bl_power_kw, cfg.Php_max_kw)
            if cfg.boiler_bl_mode == "constant" and cfg.boiler_enabled:
                Pgas[k] = min(cfg.boiler_bl_power_kw, cfg.Pgas_max_kw)

        Tbuilding[k + 1] = (
            (1.0 - alpha) * T_prev
            + beta * (float(Php[k]) + float(Pgas[k]) + Q_chp_k)
            + alpha * ta
        )
        # Cooling baseline: on_off at comfort ceiling
        if cfg.cooling_enabled and cfg.hp_enabled:
            _T_next_no_cool = Tbuilding[k + 1]
            if _T_next_no_cool > _T_cool_lim_bl:
                Php_cool[k] = cfg.Php_cool_max_kw
            Tbuilding[k + 1] -= beta * Php_cool[k]

        if cfg.hw_enabled:
            T_tank_prev = Ttank[k]
            if T_tank_prev <= cfg.hw_T_min_c:
                hw_on = True
            elif T_tank_prev >= cfg.hw_T_max_c:
                hw_on = False
            if cfg.hw_bl_mode == "on_off":
                if hw_on:
                    Ptank[k] = min(_phw_bl, cfg.Ptank_max_kw)
            elif cfg.hw_bl_mode == "constant":
                Ptank[k] = min(cfg.hw_bl_power_kw, cfg.Ptank_max_kw)
            Ttank[k + 1] = (
                (1.0 - gamma_t) * T_tank_prev
                + (dt / Ctank) * (float(Ptank[k]) + _T_ref / R_tank - Qdraw)
            )
            Ttank[k + 1] = float(np.clip(Ttank[k + 1], 0.0, cfg.hw_T_max_c))

        Php_elec_k = float(Php[k]) / max(float(COP[k]), 1e-3) if cfg.hp_enabled else 0.0
        Php_cool_elec_k_bl = (float(Php_cool[k]) / max(cfg.COP_cool, 1e-3)
                               if cfg.cooling_enabled and cfg.hp_enabled else 0.0)
        net = (
            float(inputs.Pload_kw[k])
            + Php_elec_k
            + Php_cool_elec_k_bl
            + float(Ptank[k])
            + float(Pflex[k])
            - float(Ppv[k])
            - Pchp_k
        )
        Pgrid[k] = max(0.0, net)
        # If PV exceeded demand, either export at spot price (preferred) or
        # curtail. Excess cannot be exported when no feed-in contract exists.
        if net < 0.0:
            if cfg.pv_enabled and cfg.pv_export_enabled:
                Pexport[k] = -net
            else:
                Ppv[k] = float(Ppv[k]) + net   # = min(Ppv[k], total_demand_k)

    Pchp_arr  = Fchp * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec
    elec_cost    = float(np.sum(lambda_e * Pgrid)) * dt
    # PV export revenue at spot price (no grid fee on injected energy).
    export_revenue = float(np.sum(inputs.price_eur_kwh[:H] * Pexport)) * dt
    boiler_cost  = float(np.sum(Pgas)) * c_boiler * dt
    chp_gas_cost = float(np.sum(Fchp)) * cfg.gas_price_chp_eur_m3 * dt
    bl_cap_tariff_cost = 0.0
    if cfg.cap_tariff_Plim_kw > 0.0 and cfg.cap_tariff_epsilon_l > 0.0:
        _bl_exc = np.maximum(0.0, Pgrid - cfg.cap_tariff_Plim_kw)
        bl_cap_tariff_cost = cfg.cap_tariff_epsilon_l * float(np.sum(_bl_exc ** 2))

    return {
        "total_cost": elec_cost + boiler_cost + chp_gas_cost + bl_cap_tariff_cost - export_revenue,
        # Gross cost = imports + gas + cap tariff (no export credit). Use as
        # an alternative denominator for "% saving of energy bill" so that
        # PV export revenue doesn't inflate the percentage.
        "gross_cost":    elec_cost + boiler_cost + chp_gas_cost + bl_cap_tariff_cost,
        "import_cost":   elec_cost,
        "export_revenue": export_revenue,
        "Pgrid": Pgrid,
        "Pexport": Pexport,
        "Php":   Php,
        "Pgas":  Pgas,
        "Ptank": Ptank,
        "Pflex": Pflex,
        "Ppv":   Ppv,
        "Fchp":  Fchp,
        "Pchp":  Pchp_arr,
        "Php_cool": Php_cool,
        "Tbuilding": Tbuilding,
        "Ttank":     Ttank,
    }


def compute_asset_savings(
    mpc_out: "MPCOutputs",
    bl: dict,
    inputs: "MPCInputs",
    cfg: "MPCConfig",
) -> dict:
    """
    Attribute the total MPC energy saving to individual assets and rules.

    Per-asset savings (hp, boiler, flex, battery, chp, hw, pv) sum to the
    total energy saving ``bl['total_cost'] − mpc_out.mpc_cost_eur``.
    The capacity-tariff peak-shaving saving is reported separately.

    Returns
    -------
    dict with keys:
        hp_eur, boiler_eur, flex_eur, battery_eur, chp_eur, hw_eur, pv_eur,
        thermal_building_eur, hw_thermal_eur, flex_shifting_eur,
        battery_arbitrage_eur, chp_spark_eur, pv_selfconsumption_eur,
        peak_shaving_eur, total_eur
    """
    H  = cfg.horizon_steps
    dt = cfg.dt_hours
    lambda_e = inputs.price_eur_kwh[:H] + cfg.fee_eur_kwh
    COP      = compute_cop(inputs.Tamb_c[:H], cfg)
    c_boiler = cfg.gas_price_boiler_eur_kwh / max(cfg.eta_boiler, 1e-9)

    def _plan(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        if len(a) >= H:
            return a[:H]
        return np.concatenate([a, np.zeros(H - len(a))])

    mpc_Php   = _plan(mpc_out.plan_Php)
    mpc_Pgas  = _plan(mpc_out.plan_Pgas)
    mpc_Pflex = _plan(mpc_out.plan_Pflex)
    mpc_Ptank = _plan(mpc_out.plan_Ptank)
    mpc_Pdis  = _plan(mpc_out.plan_Pdis)
    mpc_Pch   = _plan(mpc_out.plan_Pch)
    mpc_Ppv   = _plan(mpc_out.plan_Ppv)
    mpc_Pexport = _plan(getattr(mpc_out, "plan_Pexport", np.zeros(H)))
    mpc_Fchp  = _plan(mpc_out.plan_Fchp)
    mpc_Pchp  = _plan(mpc_out.plan_Pchp)
    mpc_Pgrid = _plan(mpc_out.plan_Pgrid)

    bl_Php   = np.asarray(bl["Php"],   dtype=float)[:H]
    bl_Pgas  = np.asarray(bl["Pgas"],  dtype=float)[:H]
    bl_Pflex = np.asarray(bl["Pflex"], dtype=float)[:H]
    bl_Ptank = np.asarray(bl["Ptank"], dtype=float)[:H]
    bl_Ppv   = np.asarray(bl["Ppv"],   dtype=float)[:H]
    bl_Pexport = np.asarray(bl.get("Pexport", np.zeros(H)), dtype=float)[:H]
    bl_Fchp  = np.asarray(bl["Fchp"],  dtype=float)[:H]
    bl_Pchp  = np.asarray(bl["Pchp"],  dtype=float)[:H]
    bl_Pgrid = np.asarray(bl["Pgrid"], dtype=float)[:H]

    # ── Per-asset savings (all in €) ──────────────────────────────────────
    bl_hp_elec  = bl_Php  / np.maximum(COP, 1e-3)
    mpc_hp_elec = mpc_Php / np.maximum(COP, 1e-3)
    hp_saving      = float(np.dot(lambda_e, bl_hp_elec  - mpc_hp_elec))  * dt
    boiler_saving  = float(np.sum(c_boiler * (bl_Pgas   - mpc_Pgas)))    * dt

    # ── Decompose thermal_building_eur = fuel_switching + thermal_storage ─
    # Counterfactual: supply the *same heat profile as the baseline* but
    # always pick the cheapest source at each step (no timing shift).
    #   fuel_switching_eur  = saving from always choosing the cheapest heater
    #                         for the baseline's heat demand pattern
    #   thermal_storage_eur = residual — saving from shifting heat demand in
    #                         time (pre-heating/coasting with building mass)
    bl_Q    = bl_Php + bl_Pgas              # total baseline heat demand [kW]
    c_hp_k  = lambda_e / np.maximum(COP, 1e-3)  # HP cost per kWh_th [€/kWh]
    # Baseline heat cost (boiler-first: almost all gas)
    bl_heat_cost = (float(np.dot(lambda_e / np.maximum(COP, 1e-3), bl_Php))
                    + float(np.sum(c_boiler * bl_Pgas))) * dt
    # Optimal-source cost for the same timing: pick cheaper of HP/boiler each step
    if cfg.hp_enabled and cfg.boiler_enabled:
        c_best   = np.minimum(c_hp_k, c_boiler)           # best cost per kWh_th
    elif cfg.hp_enabled:
        c_best   = c_hp_k
    else:
        c_best   = np.full(H, c_boiler)
    fs_heat_cost     = float(np.dot(c_best, bl_Q)) * dt
    fuel_switching_eur  = bl_heat_cost - fs_heat_cost
    thermal_storage_eur = (hp_saving + boiler_saving) - fuel_switching_eur
    flex_saving    = float(np.dot(lambda_e, bl_Pflex    - mpc_Pflex))    * dt
    battery_saving = float(np.dot(lambda_e, mpc_Pdis    - mpc_Pch))      * dt
    # CHP: marginal saving = (MPC net income) − (baseline net income).
    # Using absolute MPC income would double-count whenever the baseline
    # already runs the CHP (e.g. ambient_temp / heat_demand / constant modes),
    # making the per-rule sum exceed the true total saving.
    mpc_chp_income = (
        float(np.dot(lambda_e, mpc_Pchp)) * dt
        - float(np.sum(cfg.gas_price_chp_eur_m3 * mpc_Fchp)) * dt
    )
    bl_chp_income_pre = (
        float(np.dot(lambda_e, bl_Pchp)) * dt
        - float(np.sum(cfg.gas_price_chp_eur_m3 * bl_Fchp)) * dt
    )
    chp_saving = mpc_chp_income - bl_chp_income_pre
    hw_saving   = float(np.dot(lambda_e, bl_Ptank - mpc_Ptank)) * dt
    # PV saving — properly attributed.
    # The "raw" PV term λ_e·(mpc_Ppv − bl_Ppv) credits every extra self-
    # consumed PV kWh at the full retail price (spot + fee). But exported
    # PV only earns λ_spot (no fee). When MPC chooses to export MORE than
    # baseline, the raw term over-credits and needs to be corrected by
    # `fee·(bl_Pexport − mpc_Pexport)`. Folding the correction into pv_eur
    # gives a single, honest "PV" row whose value reflects the true
    # PV-attributable saving vs. the baseline (which also exports PV).
    # Derivation: total saving = Σ asset_savings(using λ_e) + fee·(bl_Pexport − mpc_Pexport)
    # so absorbing the residual into pv_eur closes the decomposition with
    # one fewer row.
    pv_raw_saving  = float(np.dot(lambda_e, mpc_Ppv - bl_Ppv)) * dt
    export_correct = float(cfg.fee_eur_kwh * np.sum(bl_Pexport - mpc_Pexport)) * dt
    pv_saving      = pv_raw_saving + export_correct
    # Cooling: MPC uses cooling more cleverly (e.g. pre-cools at cheap hours)
    mpc_Php_cool = _plan(getattr(mpc_out, "plan_Php_cool", np.zeros(H)))
    bl_Php_cool  = np.asarray(bl.get("Php_cool", np.zeros(H)), dtype=float)[:H]
    _cop_cool_inv = 1.0 / max(getattr(cfg, "COP_cool", 3.0), 1e-3)
    bl_cool_elec   = bl_Php_cool  * _cop_cool_inv
    mpc_cool_elec  = mpc_Php_cool * _cop_cool_inv
    cooling_saving = float(np.dot(lambda_e, bl_cool_elec - mpc_cool_elec)) * dt

    # ── Per-asset BASELINE costs (used for % display in charts) ──────────
    # Positive = the asset cost money in the baseline scenario.
    # PV: negative because it saves money; report as the amount PV contributed.
    bl_hp_cost      = float(np.dot(lambda_e, bl_hp_elec)) * dt
    bl_boiler_cost  = float(np.sum(c_boiler * bl_Pgas))   * dt
    bl_flex_cost    = float(np.dot(lambda_e, bl_Pflex))   * dt
    bl_hw_cost      = float(np.dot(lambda_e, bl_Ptank))   * dt
    bl_pv_saving    = float(np.dot(lambda_e, bl_Ppv))     * dt   # money PV saved in baseline
    bl_cooling_cost = float(np.dot(lambda_e, bl_cool_elec)) * dt   # electricity cooling used in baseline
    bl_chp_income   = (
        float(np.dot(lambda_e, bl_Pchp)) * dt
        - float(np.sum(cfg.gas_price_chp_eur_m3 * bl_Fchp)) * dt
    )

    # ── Capacity tariff peak-shaving (reported separately) ────────────────
    if cfg.cap_tariff_Plim_kw > 0.0 and cfg.cap_tariff_epsilon_l > 0.0:
        bl_exc  = np.maximum(0.0, bl_Pgrid  - cfg.cap_tariff_Plim_kw)
        mpc_exc = np.maximum(0.0, mpc_Pgrid - cfg.cap_tariff_Plim_kw)
        peak_saving = cfg.cap_tariff_epsilon_l * float(
            np.sum(bl_exc ** 2 - mpc_exc ** 2)
        ) * dt
    else:
        peak_saving = 0.0

    per_asset = (
        hp_saving + boiler_saving + flex_saving
        + battery_saving + chp_saving + hw_saving + pv_saving
        + cooling_saving
    )

    return {
        # Per-asset savings
        "hp_eur":      hp_saving,
        "boiler_eur":  boiler_saving,
        "flex_eur":    flex_saving,
        "battery_eur": battery_saving,
        "chp_eur":     chp_saving,
        "hw_eur":      hw_saving,
        "pv_eur":      pv_saving,
        "cooling_eur": cooling_saving,
        # Per-asset baseline costs (for % computation in charts)
        "bl_hp_cost":     bl_hp_cost,
        "bl_boiler_cost": bl_boiler_cost,
        "bl_flex_cost":   bl_flex_cost,
        "bl_hw_cost":     bl_hw_cost,
        "bl_pv_saving":   bl_pv_saving,   # >0 means PV already saved in baseline
        "bl_chp_income":  bl_chp_income,  # >0 means CHP generated net income in baseline
        "bl_battery_cost": 0.0,           # battery always idle in baseline
        "bl_cooling_cost": bl_cooling_cost,  # electricity used for cooling in baseline
        # Per-rule groupings
        "thermal_building_eur":   hp_saving + boiler_saving,
        "fuel_switching_eur":     fuel_switching_eur,
        "thermal_storage_eur":    thermal_storage_eur,
        "hw_thermal_eur":         hw_saving,
        "flex_shifting_eur":      flex_saving,
        "battery_arbitrage_eur":  battery_saving,
        "chp_spark_eur":          chp_saving,
        "pv_selfconsumption_eur": pv_saving,
        "peak_shaving_eur":       peak_saving,
        # Totals
        "total_eur": per_asset + peak_saving,
    }


# ===========================================================================
# SECTION 7 — PUBLIC API
# ===========================================================================

def solve_mpc(inputs: MPCInputs, cfg: MPCConfig, start_weekday: int | None = None) -> MPCOutputs:
    """
    Solve the MPC LP/MILP for one control interval.

    Implements the RECEDING HORIZON principle:
      1. Solve the full H-step problem with current measurements and forecasts.
      2. Return ALL decision variable trajectories (for dashboard charts).
      3. The dashboard executes only the first-step commands and calls again
         at the next interval with updated measurements.

    Parameters
    ----------
    inputs : MPCInputs — per-solve measurements and forecasts
    cfg    : MPCConfig — physical parameters and optimisation settings
    start_weekday : int | None — 0=Mon … 6=Sun for the *first* step of the
        horizon.  When ``None``, today's local weekday is used.  Passed to
        flex-load day-of-week gating; ignored when no per-flex ``active_days``
        list is configured.

    Returns
    -------
    MPCOutputs — first-step commands + full horizon plans + KPIs
    """
    t0 = time.perf_counter()
    H  = cfg.horizon_steps
    dt = cfg.dt_hours
    # Dashboard and analysis workflows build horizons from local midnight.
    # Keep setback schedule aligned to that shared timeline so graph overlays
    # and comfort constraints reference the same wall-clock steps.
    start_hour = 0.0
    if start_weekday is None:
        import datetime as _dt
        start_weekday = _dt.datetime.now().weekday()

    # ── Ensure all input arrays are padded/clipped to length H ────────────
    def _pad(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) >= H:
            return arr[:H]
        return np.concatenate([arr, np.full(H - len(arr), arr[-1] if len(arr) else fill)])

    price     = _pad(inputs.price_eur_kwh,      fill=0.15)
    Pload     = _pad(inputs.Pload_kw,           fill=50.0)
    Ppv_fc    = _pad(inputs.Ppv_forecast_kw,    fill=0.0)
    Tamb      = _pad(inputs.Tamb_c,             fill=10.0)

    # ── Override config initial states from inputs ─────────────────────────
    cfg_use = MPCConfig(**cfg.__dict__)
    cfg_use.SOC_init_kwh = inputs.SOC_init_kwh
    cfg_use.T_init_c     = inputs.T_building_init_c
    cfg_use.hw_T_init_c  = inputs.T_tank_init_c

    # ── Build padded MPCInputs ─────────────────────────────────────────────
    inp = MPCInputs(
        price_eur_kwh    = price,
        Pload_kw         = Pload,
        Ppv_forecast_kw  = Ppv_fc,
        Tamb_c           = Tamb,
        SOC_init_kwh     = cfg_use.SOC_init_kwh,
        T_building_init_c= cfg_use.T_init_c,
        T_tank_init_c    = cfg_use.hw_T_init_c,
        month            = inputs.month,
        Php_prev_kw      = inputs.Php_prev_kw,
    )

    # ── Precompute COP profile ──────────────────────────────────────────────
    COP = compute_cop(Tamb, cfg_use)

    # ── Solve ──────────────────────────────────────────────────────────────
    sol = None
    if CVXPY_AVAILABLE:
        try:
            sol = _solve_cvxpy(inp, cfg_use, COP, H, dt, start_hour=start_hour,
                               start_weekday=start_weekday)
        except Exception as exc:
            logger.warning("cvxpy solve failed (%s) — using heuristic", exc)

        # If the MILP path timed out / failed, automatically retry as LP
        # relaxation before falling back to the ramp-ignoring heuristic.
        if sol is None and cfg_use.chp_use_milp:
            import dataclasses as _dc
            _cfg_lp = _dc.replace(cfg_use, chp_use_milp=False)
            logger.info("MILP failed — retrying with CHP LP relaxation to preserve ramp constraints")
            try:
                sol = _solve_cvxpy(inp, _cfg_lp, COP, H, dt, start_hour=start_hour,
                                   start_weekday=start_weekday)
            except Exception as exc2:
                logger.warning("LP relaxation also failed (%s)", exc2)

    if sol is None:
        sol = _solve_heuristic(inp, cfg_use, COP, H, dt, start_hour=start_hour,
                               start_weekday=start_weekday)

    solve_ms = (time.perf_counter() - t0) * 1000.0

    # ── Extract first-step values ──────────────────────────────────────────
    def _f(key: str, default: float = 0.0) -> float:
        arr = sol.get(key, np.zeros(1))
        return float(arr[0]) if len(arr) > 0 else default

    Pgrid_0   = _f("Pgrid")
    Php_0     = _f("Php")
    Pgas_0    = _f("Pgas")
    Pch_0     = _f("Pch")
    Pdis_0    = _f("Pdis")
    SOC_0     = float(sol["SOC"][1]) if len(sol["SOC"]) > 1 else cfg_use.SOC_init_kwh
    Pflex_0   = _f("Pflex")
    Ppv_0     = _f("Ppv")
    Ptank_0   = _f("Ptank")
    Ttank_0   = float(sol["Ttank"][1]) if len(sol["Ttank"]) > 1 else cfg_use.hw_T_init_c
    Fchp_0    = _f("Fchp")
    Pchp_0    = _f("Pchp")
    Qchp_0    = _f("Qchp")
    zchp_0    = _f("zchp")
    ychp_0    = _f("ychp")
    Tbld_0    = float(sol["Tbuilding"][1]) if len(sol["Tbuilding"]) > 1 else cfg_use.T_init_c
    COP_0     = float(COP[0])

    total_heating_0 = Php_0 + Pgas_0 + Qchp_0

    # ── Cost KPIs ──────────────────────────────────────────────────────────
    lambda_e_0   = float(price[0]) + cfg_use.fee_eur_kwh  # [€/kWh]
    c_boiler_kw  = cfg_use.gas_price_boiler_eur_kwh / max(
        cfg_use.eta_boiler, 1e-9
    )  # €/kWh thermal

    # Full-horizon MPC cost from solution arrays (electricity + gas + capacity tariff)
    mpc_cap_tariff_cost = 0.0
    if cfg_use.cap_tariff_Plim_kw > 0.0 and cfg_use.cap_tariff_epsilon_l > 0.0:
        _exc = np.maximum(0.0, sol["Pgrid"] - cfg_use.cap_tariff_Plim_kw)
        mpc_cap_tariff_cost = cfg_use.cap_tariff_epsilon_l * float(np.sum(_exc ** 2))
    mpc_cost = (
        float(np.sum((price + cfg_use.fee_eur_kwh) * sol["Pgrid"])) * dt
        + float(np.sum(sol["Fchp"]))  * cfg_use.gas_price_chp_eur_m3 * dt
        + float(np.sum(sol["Pgas"]))  * c_boiler_kw * dt
        + mpc_cap_tariff_cost
        - float(np.sum(price * sol.get("Pexport", np.zeros_like(sol["Pgrid"])))) * dt
    )
    # Full-horizon baseline: rule-based simulation with same physics & forecasts
    baseline_cost = compute_baseline_cost(inp, cfg_use, start_hour=start_hour,
                                          start_weekday=start_weekday)
    saving        = baseline_cost - mpc_cost

    # ── Build plan arrays ──────────────────────────────────────────────────
    # Heating plan = Php + Pgas + Qchp per step
    plan_heat = sol["Php"] + sol["Pgas"] + sol["Qchp"]

    return MPCOutputs(
        # First-step commands
        Pgrid_kw    = Pgrid_0,
        Php_kw      = Php_0,
        Pgas_kw     = Pgas_0,
        Pch_kw      = Pch_0,
        Pdis_kw     = Pdis_0,
        SOC_kwh     = SOC_0,
        Pflex_kw    = Pflex_0,
        Ppv_kw      = Ppv_0,
        Ptank_kw    = Ptank_0,
        Ttank_c     = Ttank_0,
        Fchp_m3_h   = Fchp_0,
        Pchp_kw     = Pchp_0,
        Qchp_kw     = Qchp_0,
        zchp        = zchp_0,
        ychp        = ychp_0,
        Tbuilding_c = Tbld_0,
        Php_cool_kw = _f("Php_cool"),
        COP_now     = COP_0,
        # Derived KPIs
        total_heating_kw  = total_heating_0,
        mpc_cost_eur      = mpc_cost,
        baseline_cost_eur = baseline_cost,
        cost_saving_eur   = saving,
        # Legacy aliases (populated by smpc_calculator mapping layer)
        net_power_kwh        = Pgrid_0 * dt,
        wkk_elec_kwh         = Pchp_0  * dt,
        wkk_heat_kwh         = Qchp_0  * dt,
        wkk_gas_setpoint_m3  = Fchp_0  * dt,
        ice_bank_charge_kwh  = Pflex_0 * dt,
        # Full-horizon plans
        plan_Pgrid     = sol["Pgrid"],
        plan_Php       = sol["Php"],
        plan_Pgas      = sol["Pgas"],
        plan_Pch       = sol["Pch"],
        plan_Pdis      = sol["Pdis"],
        plan_SOC       = sol["SOC"],
        plan_Pflex     = sol["Pflex"],
        plan_Pflex_by_id = sol.get("Pflex_by_id", {}),
        Pflex_by_id_kw   = {
            _k: float(_v[0]) if len(_v) > 0 else 0.0
            for _k, _v in sol.get("Pflex_by_id", {}).items()
        },
        plan_Ppv       = sol["Ppv"],
        plan_Pexport   = sol.get("Pexport", np.zeros(H)),
        plan_Ptank     = sol["Ptank"],
        plan_Ttank     = sol["Ttank"],
        plan_Fchp      = sol["Fchp"],
        plan_Pchp      = sol["Pchp"],
        plan_Qchp      = sol["Qchp"],
        plan_zchp      = sol["zchp"],
        plan_Tbuilding = sol["Tbuilding"],
        plan_Php_cool  = sol.get("Php_cool", np.zeros(H)),
        plan_COP       = COP,
        # Metadata
        solver_used   = sol.get("solver", "unknown"),
        solver_status = sol.get("status", "unknown"),
        solve_time_ms = solve_ms,
        obj_value     = sol.get("obj_value", float("nan")),
    )


def load_mpc_config() -> MPCConfig:
    """Load MPCConfig from the ``'mpc'`` block in dashboard_config.json."""
    try:
        from dashboard_config import load_dashboard_config
        raw = load_dashboard_config()
        mpc_block = raw.get("mpc", {})
        if mpc_block:
            cfg = MPCConfig.from_dict(mpc_block)
            # Pull day/night base load from smpc.building if present, so the
            # values shown in Manage Assets ("Night load" / "Day load")
            # transparently drive the solver.
            smpc_b = raw.get("smpc", {}).get("building", {})
            if "base_load_kw" in smpc_b:
                cfg.base_load_night_kw = float(smpc_b["base_load_kw"])
            if "peak_load_kw" in smpc_b:
                cfg.base_load_day_kw = float(smpc_b["peak_load_kw"])
            return cfg
    except Exception as exc:
        logger.warning("Could not load MPC config (%s) — using defaults", exc)
    return MPCConfig()


# ===========================================================================
# QUICK SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    print("MPC LP self-test")
    print(f"  cvxpy available: {CVXPY_AVAILABLE}")

    cfg = MPCConfig(horizon_steps=24, dt_hours=1.0, hp_enabled=True,
                    boiler_enabled=True, chp_enabled=False, bat_enabled=False,
                    hw_enabled=True, flex_enabled=False, pv_enabled=True)

    rng = np.random.default_rng(42)
    inp = MPCInputs(
        price_eur_kwh     = 0.08 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, 24)),
        Pload_kw          = 50.0 + 30.0 * np.abs(np.sin(np.linspace(0, np.pi, 24))),
        Ppv_forecast_kw   = np.maximum(0.0, 80.0 * np.sin(np.linspace(0, np.pi, 24))),
        Tamb_c            = 5.0 + 5.0 * np.sin(np.linspace(0, np.pi, 24)),
        SOC_init_kwh      = 50.0,
        T_building_init_c = 20.0,
        T_tank_init_c     = 55.0,
    )

    out = solve_mpc(inp, cfg)
    print(f"  Solver: {out.solver_used}  status: {out.solver_status}")
    print(f"  Solve time: {out.solve_time_ms:.1f} ms")
    print(f"  First step: Pgrid={out.Pgrid_kw:.1f} kW  Php={out.Php_kw:.1f} kW"
          f"  Tbuilding={out.Tbuilding_c:.2f}°C  COP={out.COP_now:.2f}")
    print(f"  MPC cost: {out.mpc_cost_eur:.4f} €  Baseline: {out.baseline_cost_eur:.4f} €"
          f"  Saving: {out.cost_saving_eur:.4f} €")
