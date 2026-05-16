"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  THIS IS A CORE THESIS ALGORITHM                                 ║
║  MPC LP/MILP optimiser — all decision variables and physics      ║
║  models in one place.                                            ║
╚══════════════════════════════════════════════════════════════════╝

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
              + (λ_gas_boiler/(HV·η_boiler))·Pgas[k]]
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
    COP_cool:          float =  3.0    # Cooling COP (constant approximation)
    # Night setback schedule
    use_night_setback: bool  = False
    T_set_night_c:     float = 18.0    # Heating comfort floor at night [°C]
    T_cool_night_c:    float = 28.0    # Cooling comfort ceiling at night [°C]
    night_start_h:     float = 22.0    # Hour (0–24) when night period begins
    night_end_h:       float =  7.0    # Hour (0–24) when day resumes

    # ── Gas boiler ──────────────────────────────────────────────────────────
    boiler_enabled: bool  = True
    Pgas_max_kw:    float = 100.0  # Max thermal output [kW]
    eta_boiler:     float =  0.92  # Thermal efficiency (fraction of gas HHV)
    gas_price_boiler_eur_m3: float = 0.35  # Gas price [€/m³]
    gas_HV_kwh_m3:  float =  9.8   # Gas calorific value [kWh/m³]
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

    # ── PV ──────────────────────────────────────────────────────────────────
    pv_enabled:    bool  = True
    pv_capacity_kwp: float = 100.0  # Installed PV capacity [kWp]

    # ── Flexible / shiftable load ────────────────────────────────────────────
    flex_enabled:      bool  = False
    Pflex_max_kw:      float =  50.0   # Max instantaneous power [kW]
    flex_daily_kwh:    float = 400.0   # Daily energy requirement [kWh]
    flex_time_start:   str   = "00:00" # Active window start [HH:MM]
    flex_time_end:     str   = "23:59" # Active window end [HH:MM]
    flex_ramp_up_kw:   float = 9999.0  # Max power increase per step [kW]
    flex_ramp_down_kw: float = 9999.0  # Max power decrease per step [kW]

    # ── Hot water tank ───────────────────────────────────────────────────────
    hw_enabled:      bool  = True
    Ptank_max_kw:    float =  3.0    # Heater rated power [kW]  (COP = 1)
    hw_volume_l:     float = 200.0   # Tank volume [L]
    hw_T_min_c:      float =  45.0   # Minimum temperature (comfort/safety) [°C]
    hw_T_max_c:      float =  60.0   # Maximum temperature [°C]
    hw_T_init_c:     float =  55.0   # Initial temperature [°C]
    hw_heat_loss_w:  float =  50.0   # Standby heat loss [W] at nominal temperature
    hw_draw_kw:      float =   0.5   # Constant hot water draw demand [kW]
    hw_ramp_pct:     float = 100.0   # Max hot-water heater ramp [% of Ptank_max_kw per step]

    # ── Building ─────────────────────────────────────────────────────────────
    Tset_c:          float =  21.0    # Setpoint temperature [°C]
    Tmin_c:          float =  19.0    # Minimum comfort temperature [°C]
    Tmax_c:          float =  23.0    # Maximum comfort temperature [°C]
    T_init_c:        float =  20.0    # Initial building temperature [°C]
    Cth_kwh_per_c:   float = 2000.0   # Thermal mass [kWh/°C]
    UA_kw_per_c:     float =  100.0   # Envelope heat transfer coefficient [kW/°C]

    # ── Baseline simulation modes ─────────────────────────────────────────────
    # These define the rule-based 'no-optimisation' reference scenario used to
    # calculate energy cost savings attributable to the MPC.
    #
    # heat_pump / gas_boiler:
    #   "on_off"     — bang-bang controller: run at baseline_power_kw when building
    #                  T drops to Tmin_c, stop when T reaches Tmax_c. Both share
    #                  one heating_on state flag (same thermostat).
    #   "constant"   — fixed power (hp_bl_power_kw / boiler_bl_power_kw) every step.
    #   "always_off" — asset not used in the baseline.
    # battery:
    #   "always_off" — idle (no arbitrage without optimisation).
    # chp:
    #   "always_off"  — not used in baseline.
    #   "heat_demand" — run at max output whenever building T < Tset_c (heat-led
    #                   dispatch; electricity is a byproduct).
    #   "constant"    — always on at max output.
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
        c.gas_price_boiler_eur_m3 = float(bl.get("gas_price_eur_m3", c.gas_price_boiler_eur_m3))
        c.gas_HV_kwh_m3           = float(bl.get("gas_HV_kwh_m3",    c.gas_HV_kwh_m3))
        c.boiler_ramp_pct         = float(bl.get("ramp_pct",         c.boiler_ramp_pct))
        c.boiler_bl_mode          = str  (bl.get("baseline_mode",     c.boiler_bl_mode))
        c.boiler_bl_power_kw      = float(bl.get("baseline_power_kw", c.boiler_bl_power_kw))

        chp = d.get("chp", {})
        c.chp_enabled          = bool (chp.get("enabled",          c.chp_enabled))
        c.chp_eta_elec         = float(chp.get("eta_elec",         c.chp_eta_elec))
        c.chp_eta_heat         = float(chp.get("eta_heat",         c.chp_eta_heat))
        c.chp_startup_cost_eur = float(chp.get("startup_cost_eur", c.chp_startup_cost_eur))
        c.gas_price_chp_eur_m3 = float(chp.get("gas_price_eur_m3", c.gas_price_chp_eur_m3))
        c.chp_gas_HV_kwh_m3    = float(chp.get("gas_HV_kwh_m3",   c.chp_gas_HV_kwh_m3))
        c.chp_use_milp         = bool (chp.get("use_milp",         c.chp_use_milp))
        c.chp_ramp_pct         = float(chp.get("ramp_pct",         c.chp_ramp_pct))
        c.chp_bl_mode          = str  (chp.get("baseline_mode",    c.chp_bl_mode))
        # Both Pchp_max_kw and Fchp_max_m3_h are independent caps; the most
        # restrictive (minimum) gas flow wins.  This ensures changing either
        # value in dashboard_config.json has a real effect on the optimiser.
        _hv_e = c.chp_gas_HV_kwh_m3 * c.chp_eta_elec
        _fchp_from_pchp = (
            float(chp["Pchp_max_kw"]) / max(_hv_e, 1e-9)
            if "Pchp_max_kw" in chp
            else c.Fchp_max_m3_h
        )
        _fchp_from_cfg = float(chp.get("Fchp_max_m3_h", _fchp_from_pchp))
        c.Fchp_max_m3_h = min(_fchp_from_pchp, _fchp_from_cfg)
        c.Pchp_max_kw   = c.Fchp_max_m3_h * _hv_e

        pv = d.get("pv", {})
        c.pv_enabled      = bool (pv.get("enabled",      c.pv_enabled))
        c.pv_capacity_kwp = float(pv.get("capacity_kwp", c.pv_capacity_kwp))

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
                "gas_price_eur_m3":  self.gas_price_boiler_eur_m3,
                "gas_HV_kwh_m3":     self.gas_HV_kwh_m3,
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
                "gas_price_eur_m3": self.gas_price_chp_eur_m3,
                "gas_HV_kwh_m3":    self.chp_gas_HV_kwh_m3,
                "use_milp":         self.chp_use_milp,
                "ramp_pct":         self.chp_ramp_pct,
                "baseline_mode":    self.chp_bl_mode,
            },
            "pv": {"enabled": self.pv_enabled, "capacity_kwp": self.pv_capacity_kwp},
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
    # Used in COP[k] = COP0*(1 − cop_alpha*(Tamb[k]−T0)) and building physics
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
    plan_Ppv:       np.ndarray = field(default_factory=lambda: np.array([]))
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
        COP[k] = COP0 × max(COP_min, 1 − cop_alpha × (Tamb[k] − T0))

    This is a first-order linear approximation valid around the rated
    operating point (Tamb = T0).  For Tamb < T0 the COP increases
    (colder outside = better for heating in ground-source configurations);
    for Tamb > T0 the COP decreases.  The COP_min clamp prevents
    non-physical values.

    Parameters
    ----------
    Tamb : ambient temperature forecast [°C], shape (H,)
    cfg  : MPCConfig with COP0, T0_c, cop_alpha, COP_min

    Returns
    -------
    COP : np.ndarray, shape (H,) — COP values ≥ COP_min
    """
    raw = cfg.COP0 * (1.0 - cfg.cop_alpha * (Tamb - cfg.T0_c))
    return np.maximum(cfg.COP_min, raw)


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

    # Flexible load: schedule in cheapest slots inside the active time window
    if cfg.flex_enabled and cfg.flex_daily_kwh > 0:
        import datetime as _dt
        _flex_active = np.ones(H, dtype=bool)
        try:
            _t_on  = _dt.time.fromisoformat(cfg.flex_time_start)
            _t_off = _dt.time.fromisoformat(cfg.flex_time_end)
            _now   = _dt.datetime.now()
            for _k in range(H):
                _t = (_now + _dt.timedelta(hours=_k * dt)).time()
                if _t_on <= _t_off:
                    _flex_active[_k] = _t_on <= _t <= _t_off
                else:          # overnight window (e.g. 22:00–06:00)
                    _flex_active[_k] = _t >= _t_on or _t <= _t_off
        except ValueError:
            pass
        horizon_h  = H * dt
        target_kwh = cfg.flex_daily_kwh * (horizon_h / 24.0)
        sorted_idx = np.argsort(price_total)
        remaining  = target_kwh
        for idx in sorted_idx:
            if remaining <= 0:
                break
            if not _flex_active[idx]:
                continue
            slot_kwh = min(cfg.Pflex_max_kw * dt, remaining)
            Pflex[idx] = slot_kwh / dt  # kW
            remaining -= slot_kwh

    Ctank  = max(cfg.hw_volume_l * 1.163e-3, 1e-9)  # kWh/°C
    # Thermal resistance of tank: R_tank [°C/kW] from nominal heat loss at (T_init − 15°C)
    _hl_nom_h  = cfg.hw_heat_loss_w / 1000.0         # nominal heat loss [kW]
    _T_ref_h   = 15.0                                 # indoor ambient reference [°C]
    R_tank_h   = (max(cfg.hw_T_init_c, _T_ref_h + 1.0) - _T_ref_h) / max(_hl_nom_h, 1e-9)
    gamma_t_h  = dt / (Ctank * R_tank_h)             # dimensionless step decay
    Qdraw_h    = cfg.hw_draw_kw                       # constant hot water draw [kW]
    alpha  = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)
    beta   = dt / max(cfg.Cth_kwh_per_c, 1e-9)

    for k in range(H):
        # ── PV ───────────────────────────────────────────────────────────────
        if cfg.pv_enabled:
            # Curtail when price is negative (exporting / using PV increases cost)
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

        # ── Building thermal ─────────────────────────────────────────────────
        T_prev = Tbuilding[k]
        # Natural drift (Newton's law)
        T_drift = (1.0 - alpha) * T_prev + alpha * float(inputs.Tamb_c[k])
        heat_deficit_kw = max(0.0, (cfg.Tset_c - T_drift) * cfg.Cth_kwh_per_c / dt)
        heat_deficit_kw -= Qchp[k]   # CHP provides free heat
        heat_deficit_kw  = max(0.0, heat_deficit_kw)

        # Satisfy heat demand: use cheapest heater first (cost-aware)
        if heat_deficit_kw > 0:
            _c_boiler = (cfg.gas_price_boiler_eur_m3
                         / max(cfg.gas_HV_kwh_m3 * cfg.eta_boiler, 1e-9))
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
            _hour_k = (k * dt) % 24.0
            _is_night_k = (cfg.use_night_setback and
                           (_hour_k >= cfg.night_start_h or _hour_k < cfg.night_end_h))
            _T_cool_lim = cfg.T_cool_night_c if _is_night_k else cfg.Tmax_c
            if T_prev > _T_cool_lim:
                Php_cool[k] = min(cfg.Php_cool_max_kw,
                                  (T_prev - _T_cool_lim) * cfg.Cth_kwh_per_c / dt)
        Tbuilding[k + 1] = (1.0 - alpha) * T_prev + beta * (Q_heat - Php_cool[k]) + alpha * float(inputs.Tamb_c[k])

        # ── Hot water tank (Newton's cooling + hot water draw) ───────────────
        if cfg.hw_enabled:
            T_tank_prev = Ttank[k]
            _ta_k = float(inputs.Tamb_c[k])
            # Natural drift: cooling + draw with no heater
            T_nat = ((1.0 - gamma_t_h) * T_tank_prev
                     + (dt / Ctank) * (_ta_k / R_tank_h - Qdraw_h))
            if T_nat < cfg.hw_T_min_c:
                # Heat toward T_max (pre-heat while it's available)
                heat_needed = (cfg.hw_T_max_c - T_tank_prev) * Ctank / dt
                Ptank[k] = min(cfg.Ptank_max_kw, max(0.0, heat_needed))
            Ttank[k + 1] = ((1.0 - gamma_t_h) * T_tank_prev
                            + (dt / Ctank) * (Ptank[k] + _ta_k / R_tank_h - Qdraw_h))
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
        Pgrid[k] = max(0.0, net_load)

    return {
        "Pgrid": Pgrid, "Php": Php, "Pgas": Pgas,
        "Pch": Pch, "Pdis": Pdis, "SOC": SOC,
        "Pflex": Pflex, "Ppv": Ppv,
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
    Php      = cp.Variable(H, nonneg=True, name="Php")      if cfg.hp_enabled      else np.zeros(H)
    Php_cool = cp.Variable(H, nonneg=True, name="Php_cool") if (cfg.cooling_enabled and cfg.hp_enabled) else np.zeros(H)
    Pgas  = cp.Variable(H, nonneg=True, name="Pgas")  if cfg.boiler_enabled  else np.zeros(H)
    Ppv   = cp.Variable(H, nonneg=True, name="Ppv")   if cfg.pv_enabled      else np.zeros(H)
    Pflex = cp.Variable(H, nonneg=True, name="Pflex") if cfg.flex_enabled    else np.zeros(H)

    if cfg.bat_enabled:
        Pch  = cp.Variable(H, nonneg=True, name="Pch")
        Pdis = cp.Variable(H, nonneg=True, name="Pdis")
        SOC  = cp.Variable(H + 1,          name="SOC")
        z_bat = cp.Variable(H, boolean=True, name="z_bat")  # 1=charging, 0=discharging
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
    else:
        Pchp = np.zeros(H)
        Qchp = np.zeros(H)

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
    #    Pgrid + Ppv + Pchp + Pdis = Pload + Php_elec + Ptank + Pflex + Pch
    lhs = Pgrid + Ppv + Pchp
    rhs = inputs.Pload_kw + Php_elec + Php_cool_elec + Ptank + Pflex
    if cfg.bat_enabled:
        lhs = lhs + Pdis
        rhs = rhs + Pch
    constraints.append(lhs == rhs)

    # 2. Grid peak limit (hard):  Pgrid ≤ Pgrid_max
    constraints.append(Pgrid <= cfg.Pgrid_max_kw)

    # 3. PV upper bound
    if cfg.pv_enabled:
        constraints.append(Ppv <= inputs.Ppv_forecast_kw)

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

    # 7. Flexible load: power cap, active time window, energy target, ramp rates
    if cfg.flex_enabled:
        import datetime as _dt
        # Build per-step active mask from the configured time window
        _flex_active = np.ones(H, dtype=bool)
        try:
            _t_on  = _dt.time.fromisoformat(cfg.flex_time_start)
            _t_off = _dt.time.fromisoformat(cfg.flex_time_end)
            _now   = _dt.datetime.now()
            for _k in range(H):
                _t = (_now + _dt.timedelta(hours=_k * dt)).time()
                if _t_on <= _t_off:
                    _flex_active[_k] = _t_on <= _t <= _t_off
                else:          # overnight window
                    _flex_active[_k] = _t >= _t_on or _t <= _t_off
        except ValueError:
            pass
        constraints.append(Pflex <= cfg.Pflex_max_kw)
        # Force zero outside the allowed window
        for _k in range(H):
            if not _flex_active[_k]:
                constraints.append(Pflex[_k] == 0)
        # Energy target — capped so the problem stays feasible within the window
        horizon_h = H * dt
        target_kwh = cfg.flex_daily_kwh * (horizon_h / 24.0)
        max_in_window = float(np.sum(_flex_active)) * cfg.Pflex_max_kw * dt
        constraints.append(cp.sum(Pflex) * dt == min(target_kwh, max_in_window))
        # Ramp-rate constraints
        if H > 1:
            if cfg.flex_ramp_up_kw < 9000.0:
                constraints.append(Pflex[1:] - Pflex[:-1] <= cfg.flex_ramp_up_kw)
            if cfg.flex_ramp_down_kw < 9000.0:
                constraints.append(Pflex[:-1] - Pflex[1:] <= cfg.flex_ramp_down_kw)

    # 8. Hot water tank dynamics — proper Newton's cooling
    #    Ttank[k+1] = (1−γ)·Ttank[k] + (dt/C)·(Ptank[k] + Tamb[k]/R_tank − Qdraw)
    if cfg.hw_enabled:
        Ctank  = max(cfg.hw_volume_l * 1.163e-3, 1e-9)   # kWh/°C
        _hl_nom = cfg.hw_heat_loss_w / 1000.0             # nominal heat loss [kW]
        _T_ref  = 15.0                                    # indoor ambient reference [°C]
        R_tank  = (max(cfg.hw_T_init_c, _T_ref + 1.0) - _T_ref) / max(_hl_nom, 1e-9)  # °C/kW
        gamma_t = dt / (Ctank * R_tank)                   # dimensionless step decay
        Qdraw   = cfg.hw_draw_kw                          # constant hot water draw [kW]
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
                       + _dt_c * (Ptank + inputs.Tamb_c / R_tank - Qdraw)
        )
        # Per-step max-reachable lower bound (prevents infeasibility)
        _T_tk_max_r = np.empty(H + 1)
        _T_tk_max_r[0] = inputs.T_tank_init_c
        for _k in range(H):
            _ta = float(inputs.Tamb_c[_k])
            _T_tk_max_r[_k + 1] = min(
                (1.0 - gamma_t) * _T_tk_max_r[_k]
                    + _dt_c * (cfg.Ptank_max_kw + _ta / R_tank - Qdraw),
                cfg.hw_T_max_c,
            )
        _eff_tank_lb = np.minimum(cfg.hw_T_min_c, _T_tk_max_r[1:])
        constraints.append(Ttank[1:] >= _eff_tank_lb)

    # 9. Building temperature dynamics
    #    Tbuilding[k+1] = (1−α)·Tbuilding[k] + β·Q_in[k] + α·Tamb[k]
    alpha = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)   # dimensionless
    beta  = dt / max(cfg.Cth_kwh_per_c, 1e-9)                      # °C/kW/step
    constraints.append(Tbuilding[0] == inputs.T_building_init_c)
    Q_in = Php + Pgas + Qchp - Php_cool   # net thermal input (heating minus cooling) [kW]
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
    _T_lb_k = np.full(H, cfg.Tmin_c)
    _T_ub_k = np.full(H, cfg.Tmax_c)
    if cfg.use_night_setback:
        for _k in range(H):
            _hour = (_k * dt) % 24.0
            _is_night = _hour >= cfg.night_start_h or _hour < cfg.night_end_h
            if _is_night:
                _T_lb_k[_k] = cfg.T_set_night_c
                _T_ub_k[_k] = cfg.T_cool_night_c if cfg.cooling_enabled else cfg.Tmax_c
    _eff_lb = np.minimum(_T_lb_k, _T_max_r[1:])   # tightest lb still feasible
    _eff_ub = np.maximum(_T_ub_k, _T_min_r[1:])   # tightest ub still feasible
    constraints.append(Tbuilding[1:] >= _eff_lb)
    constraints.append(Tbuilding[1:] <= _eff_ub)

    # 10. CHP constraints
    if cfg.chp_enabled:
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

    # Boiler gas cost: (λ_gas_boiler / (HV × η_boiler)) × Pgas[k] × dt
    #   = marginal gas cost per kWh of thermal output
    c_boiler = cfg.gas_price_boiler_eur_m3 / max(
        cfg.gas_HV_kwh_m3 * cfg.eta_boiler, 1e-9
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

    objective = cp.Minimize(
        elec_cost + cap_tariff_cost + chp_gas_cost + boiler_gas_cost + startup_cost
    )

    problem = cp.Problem(objective, constraints)

    # ── Solve ──────────────────────────────────────────────────────────────
    use_milp = (cfg.chp_enabled and cfg.chp_use_milp) or cfg.bat_enabled
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
    Pflex_sol     = _v(Pflex)
    Ppv_sol       = _v(Ppv)
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
        "Pflex": Pflex_sol, "Ppv": Ppv_sol,
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

def compute_baseline_cost(inputs: MPCInputs, cfg: MPCConfig) -> float:
    """
    Simulate the 'no-optimisation' rule-based baseline over the full horizon.

    Each asset follows a simple, price-unaware rule according to its configured
    baseline mode (cfg.*_bl_mode).  The resulting grid import and gas consumption
    are costed at the same tariff as the MPC for an apples-to-apples comparison.

    Asset rules
    -----------
    heat_pump / gas_boiler — "on_off": classic bang-bang thermostat using the
        comfort band [Tmin_c, Tmax_c] as hysteresis.  Both share one heating_on
        state flag.  HP fires at hp_bl_power_kw (0 → Php_max_kw); boiler fires
        at boiler_bl_power_kw (0 → Pgas_max_kw).  No price awareness.
        "constant": fixed power every step.
        "always_off": asset not used.

    battery — always idle (no price-arbitrage dispatch without optimisation).

    chp — "always_off":  not used.
        "heat_demand": run at max output whenever building T < Tset_c (heat-led
        dispatch; electricity is a byproduct).  No spark-spread logic.
        "constant": always on at max output.

    hot_water_tank — "on_off": bang-bang on [hw_T_min_c, hw_T_max_c].  Heater
        fires at hw_bl_power_kw (0 → Ptank_max_kw) when tank drops below T_min,
        stops when it reaches T_max.
        "constant": fixed power every step.
        "always_off": not used.

    flexible_load — "fixed_window": run at baseline_power_kw (0 → Pflex_max_kw)
        between flex_bl_time_start and flex_bl_time_end every day.  This is a
        separate window from the MPC dispatch window — it models a dumb load
        that always runs during its fixed operating hours.
        "always_off": not used.

    pv — always use all available production (same as the MPC).

    Returns
    -------
    float  Total energy cost [€] for the horizon: electricity + boiler gas +
           CHP gas.
    """
    H   = cfg.horizon_steps
    dt  = cfg.dt_hours

    COP      = compute_cop(inputs.Tamb_c[:H], cfg)
    lambda_e = inputs.price_eur_kwh[:H] + cfg.fee_eur_kwh
    c_boiler = cfg.gas_price_boiler_eur_m3 / max(cfg.gas_HV_kwh_m3 * cfg.eta_boiler, 1e-9)

    Pgrid = np.zeros(H)
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

    # Building thermal coefficients
    alpha = cfg.UA_kw_per_c * dt / max(cfg.Cth_kwh_per_c, 1e-9)
    beta  = dt / max(cfg.Cth_kwh_per_c, 1e-9)

    # Hot water tank coefficients (same model as LP solver)
    Ctank   = max(cfg.hw_volume_l * 1.163e-3, 1e-9)
    _hl_nom = cfg.hw_heat_loss_w / 1000.0
    _T_ref  = 15.0
    R_tank  = (max(cfg.hw_T_init_c, _T_ref + 1.0) - _T_ref) / max(_hl_nom, 1e-9)
    gamma_t = dt / (Ctank * R_tank)
    Qdraw   = cfg.hw_draw_kw

    # ── Effective baseline powers (0 → use asset max) ─────────────────────
    _php_bl   = cfg.hp_bl_power_kw   if cfg.hp_bl_power_kw   > 0 else cfg.Php_max_kw
    _pgas_bl  = cfg.boiler_bl_power_kw if cfg.boiler_bl_power_kw > 0 else cfg.Pgas_max_kw
    _phw_bl   = cfg.hw_bl_power_kw   if cfg.hw_bl_power_kw   > 0 else cfg.Ptank_max_kw
    _pflex_bl = 0.0  # filled below per mode

    # ── PV: always use all available ──────────────────────────────────────
    if cfg.pv_enabled:
        Ppv = np.minimum(inputs.Ppv_forecast_kw[:H],
                         np.full(H, cfg.pv_capacity_kwp))

    # ── Flexible load: fixed window ───────────────────────────────────────
    if cfg.flex_enabled and cfg.flex_bl_mode == "fixed_window":
        import datetime as _dt
        _bl_active = np.zeros(H, dtype=bool)
        try:
            _t_on  = _dt.time.fromisoformat(cfg.flex_bl_time_start)
            _t_off = _dt.time.fromisoformat(cfg.flex_bl_time_end)
            _now   = _dt.datetime.now()
            for _k in range(H):
                _t = (_now + _dt.timedelta(hours=_k * dt)).time()
                if _t_on <= _t_off:
                    _bl_active[_k] = _t_on <= _t <= _t_off
                else:                                        # overnight window
                    _bl_active[_k] = _t >= _t_on or _t <= _t_off
        except ValueError:
            _bl_active[:] = True  # invalid time → run all day
        # Power: use baseline_power_kw if set, else Pflex_max_kw
        _pflex_bl = cfg.flex_bl_power_kw if cfg.flex_bl_power_kw > 0 else cfg.Pflex_max_kw
        for _k in range(H):
            if _bl_active[_k]:
                Pflex[_k] = min(_pflex_bl, cfg.Pflex_max_kw)

    # ── CHP: heat_demand (run at max when building needs heat) ────────────
    # Pre-compute only the on/off pattern; actual Fchp will be filled in the
    # step loop so it can respond to the current building temperature.
    chp_heat_demand = cfg.chp_enabled and cfg.chp_bl_mode == "heat_demand"
    chp_constant    = cfg.chp_enabled and cfg.chp_bl_mode == "constant"

    # ── Thermostat state flags (initialise from current temperatures) ─────
    heating_on = inputs.T_building_init_c < cfg.Tset_c  # building heating system
    hw_on      = inputs.T_tank_init_c     < cfg.hw_T_min_c  # hot water heater

    # ── Step-by-step simulation ───────────────────────────────────────────
    for k in range(H):
        ta      = float(inputs.Tamb_c[k])
        T_prev  = Tbuilding[k]

        # ── CHP ──────────────────────────────────────────────────────────
        if chp_heat_demand:
            # Fire when building is below setpoint (heat-led, electricity byproduct)
            Fchp[k] = cfg.Fchp_max_m3_h if T_prev < cfg.Tset_c else 0.0
        elif chp_constant:
            Fchp[k] = cfg.Fchp_max_m3_h

        Q_chp_k = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat
        Pchp_k  = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec

        # ── Building thermostat hysteresis (bang-bang on [Tmin, Tmax]) ───
        #    Both HP and boiler share one heating_on flag.
        if T_prev <= cfg.Tmin_c:
            heating_on = True
        elif T_prev >= cfg.Tmax_c:
            heating_on = False

        if heating_on:
            # Boiler-first (traditional building baseline): fire boiler as primary
            if cfg.boiler_bl_mode == "on_off" and cfg.boiler_enabled:
                Pgas[k] = min(_pgas_bl, cfg.Pgas_max_kw)
            elif cfg.boiler_bl_mode == "constant" and cfg.boiler_enabled:
                Pgas[k] = min(cfg.boiler_bl_power_kw, cfg.Pgas_max_kw)
            # HP supplements only if boiler alone cannot prevent T < Tmin
            if cfg.hp_bl_mode == "on_off" and cfg.hp_enabled:
                _T_next_boiler_only = (
                    (1.0 - alpha) * T_prev
                    + beta * (float(Pgas[k]) + Q_chp_k)
                    + alpha * ta
                )
                if _T_next_boiler_only < cfg.Tmin_c:
                    Php[k] = min(_php_bl, cfg.Php_max_kw)
            elif cfg.hp_bl_mode == "constant" and cfg.hp_enabled:
                Php[k] = min(cfg.hp_bl_power_kw, cfg.Php_max_kw)
        else:
            # "constant" mode ignores the thermostat flag — always on
            if cfg.hp_bl_mode == "constant" and cfg.hp_enabled:
                Php[k] = min(cfg.hp_bl_power_kw, cfg.Php_max_kw)
            if cfg.boiler_bl_mode == "constant" and cfg.boiler_enabled:
                Pgas[k] = min(cfg.boiler_bl_power_kw, cfg.Pgas_max_kw)

        # Update building temperature
        Tbuilding[k + 1] = (
            (1.0 - alpha) * T_prev
            + beta * (float(Php[k]) + float(Pgas[k]) + Q_chp_k)
            + alpha * ta
        )

        # ── Hot water tank bang-bang on [hw_T_min_c, hw_T_max_c] ─────────
        if cfg.hw_enabled:
            T_tank_prev = Ttank[k]
            # Update hysteresis state
            if T_tank_prev <= cfg.hw_T_min_c:
                hw_on = True
            elif T_tank_prev >= cfg.hw_T_max_c:
                hw_on = False

            if cfg.hw_bl_mode == "on_off":
                if hw_on:
                    Ptank[k] = min(_phw_bl, cfg.Ptank_max_kw)
            elif cfg.hw_bl_mode == "constant":
                Ptank[k] = min(cfg.hw_bl_power_kw, cfg.Ptank_max_kw)
            # "always_off" → Ptank[k] = 0

            Ttank[k + 1] = (
                (1.0 - gamma_t) * T_tank_prev
                + (dt / Ctank) * (float(Ptank[k]) + ta / R_tank - Qdraw)
            )
            Ttank[k + 1] = float(np.clip(Ttank[k + 1], 0.0, cfg.hw_T_max_c))

        # ── Grid residual (battery always idle) ───────────────────────────
        Php_elec_k = float(Php[k]) / max(float(COP[k]), 1e-3) if cfg.hp_enabled else 0.0
        net = (
            float(inputs.Pload_kw[k])
            + Php_elec_k
            + float(Ptank[k])
            + float(Pflex[k])
            - float(Ppv[k])
            - Pchp_k
        )
        Pgrid[k] = max(0.0, net)

    # ── Total horizon cost ────────────────────────────────────────────────
    Pchp_arr = Fchp * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec   # noqa: F841 (unused but mirrors LP pattern)
    elec_cost    = float(np.sum(lambda_e * Pgrid)) * dt
    boiler_cost  = float(np.sum(Pgas))  * c_boiler * dt
    chp_gas_cost = float(np.sum(Fchp))  * cfg.gas_price_chp_eur_m3 * dt
    return elec_cost + boiler_cost + chp_gas_cost


def compute_baseline_arrays(inputs: MPCInputs, cfg: MPCConfig) -> dict:
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
    c_boiler = cfg.gas_price_boiler_eur_m3 / max(cfg.gas_HV_kwh_m3 * cfg.eta_boiler, 1e-9)

    Pgrid = np.zeros(H)
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
    _T_ref  = 15.0
    R_tank  = (max(cfg.hw_T_init_c, _T_ref + 1.0) - _T_ref) / max(_hl_nom, 1e-9)
    gamma_t = dt / (Ctank * R_tank)
    Qdraw   = cfg.hw_draw_kw

    # on_off baseline: always use the asset's full rated capacity (Php_max_kw /
    # Pgas_max_kw / Ptank_max_kw) — a real thermostat runs at full power.
    # constant baseline: use the user-configured fixed power level.
    _pgas_bl  = cfg.boiler_bl_power_kw  if cfg.boiler_bl_power_kw  > 0 else cfg.Pgas_max_kw
    _phw_bl   = cfg.hw_bl_power_kw      if cfg.hw_bl_power_kw      > 0 else cfg.Ptank_max_kw

    if cfg.pv_enabled:
        # Curtail PV at steps where the spot price is negative — injecting
        # surplus would increase cost rather than reduce it.
        _ppv_cap = np.where(
            inputs.price_eur_kwh[:H] + cfg.fee_eur_kwh < 0,
            0.0,
            cfg.pv_capacity_kwp,
        )
        Ppv = np.minimum(inputs.Ppv_forecast_kw[:H], _ppv_cap)

    if cfg.flex_enabled and cfg.flex_bl_mode == "fixed_window":
        import datetime as _dt
        _bl_active = np.zeros(H, dtype=bool)
        try:
            _t_on  = _dt.time.fromisoformat(cfg.flex_bl_time_start)
            _t_off = _dt.time.fromisoformat(cfg.flex_bl_time_end)
            _now   = _dt.datetime.now()
            for _k in range(H):
                _t = (_now + _dt.timedelta(hours=_k * dt)).time()
                if _t_on <= _t_off:
                    _bl_active[_k] = _t_on <= _t <= _t_off
                else:
                    _bl_active[_k] = _t >= _t_on or _t <= _t_off
        except ValueError:
            _bl_active[:] = True
        _pflex_bl = cfg.flex_bl_power_kw if cfg.flex_bl_power_kw > 0 else cfg.Pflex_max_kw
        for _k in range(H):
            if _bl_active[_k]:
                Pflex[_k] = min(_pflex_bl, cfg.Pflex_max_kw)

    chp_heat_demand = cfg.chp_enabled and cfg.chp_bl_mode == "heat_demand"
    chp_constant    = cfg.chp_enabled and cfg.chp_bl_mode == "constant"
    heating_on = inputs.T_building_init_c < cfg.Tset_c
    hw_on      = inputs.T_tank_init_c     < cfg.hw_T_min_c
    Php_cool   = np.zeros(H)

    for k in range(H):
        ta     = float(inputs.Tamb_c[k])
        T_prev = Tbuilding[k]

        if chp_heat_demand:
            Fchp[k] = cfg.Fchp_max_m3_h if T_prev < cfg.Tset_c else 0.0
        elif chp_constant:
            Fchp[k] = cfg.Fchp_max_m3_h

        Q_chp_k = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_heat
        Pchp_k  = float(Fchp[k]) * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec

        if T_prev <= cfg.Tmin_c:
            heating_on = True
        elif T_prev >= cfg.Tmax_c:
            heating_on = False

        if heating_on:
            # Boiler-first (traditional building baseline): fire boiler as primary
            if cfg.boiler_bl_mode == "on_off" and cfg.boiler_enabled:
                Pgas[k] = min(_pgas_bl, cfg.Pgas_max_kw)
            elif cfg.boiler_bl_mode == "constant" and cfg.boiler_enabled:
                Pgas[k] = min(cfg.boiler_bl_power_kw, cfg.Pgas_max_kw)
            # HP supplements only if boiler alone cannot prevent T < Tmin
            if cfg.hp_bl_mode == "on_off" and cfg.hp_enabled:
                _T_next_boiler_only = (
                    (1.0 - alpha) * T_prev
                    + beta * (float(Pgas[k]) + Q_chp_k)
                    + alpha * ta
                )
                if _T_next_boiler_only < cfg.Tmin_c:
                    Php[k] = cfg.Php_max_kw
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
            _hour_bl = (k * dt) % 24.0
            _is_night_bl = (cfg.use_night_setback and
                            (_hour_bl >= cfg.night_start_h or _hour_bl < cfg.night_end_h))
            _T_cool_lim_bl = cfg.T_cool_night_c if _is_night_bl else cfg.Tmax_c
            if T_prev > _T_cool_lim_bl:
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
                + (dt / Ctank) * (float(Ptank[k]) + ta / R_tank - Qdraw)
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

    Pchp_arr  = Fchp * cfg.chp_gas_HV_kwh_m3 * cfg.chp_eta_elec
    elec_cost    = float(np.sum(lambda_e * Pgrid)) * dt
    boiler_cost  = float(np.sum(Pgas)) * c_boiler * dt
    chp_gas_cost = float(np.sum(Fchp)) * cfg.gas_price_chp_eur_m3 * dt

    return {
        "total_cost": elec_cost + boiler_cost + chp_gas_cost,
        "Pgrid": Pgrid,
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
    c_boiler = cfg.gas_price_boiler_eur_m3 / max(cfg.gas_HV_kwh_m3 * cfg.eta_boiler, 1e-9)

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
    mpc_Fchp  = _plan(mpc_out.plan_Fchp)
    mpc_Pchp  = _plan(mpc_out.plan_Pchp)
    mpc_Pgrid = _plan(mpc_out.plan_Pgrid)

    bl_Php   = np.asarray(bl["Php"],   dtype=float)[:H]
    bl_Pgas  = np.asarray(bl["Pgas"],  dtype=float)[:H]
    bl_Pflex = np.asarray(bl["Pflex"], dtype=float)[:H]
    bl_Ptank = np.asarray(bl["Ptank"], dtype=float)[:H]
    bl_Ppv   = np.asarray(bl["Ppv"],   dtype=float)[:H]
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
    chp_saving     = (
        float(np.dot(lambda_e, mpc_Pchp - bl_Pchp)) * dt
        - float(np.sum(cfg.gas_price_chp_eur_m3 * (mpc_Fchp - bl_Fchp))) * dt
    )
    hw_saving   = float(np.dot(lambda_e, bl_Ptank - mpc_Ptank)) * dt
    pv_saving   = float(np.dot(lambda_e, mpc_Ppv  - bl_Ppv))    * dt
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

def solve_mpc(inputs: MPCInputs, cfg: MPCConfig) -> MPCOutputs:
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

    Returns
    -------
    MPCOutputs — first-step commands + full horizon plans + KPIs
    """
    t0 = time.perf_counter()
    H  = cfg.horizon_steps
    dt = cfg.dt_hours

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
            sol = _solve_cvxpy(inp, cfg_use, COP, H, dt)
        except Exception as exc:
            logger.warning("cvxpy solve failed (%s) — using heuristic", exc)

        # If the MILP path timed out / failed, automatically retry as LP
        # relaxation before falling back to the ramp-ignoring heuristic.
        if sol is None and cfg_use.chp_use_milp:
            import dataclasses as _dc
            _cfg_lp = _dc.replace(cfg_use, chp_use_milp=False)
            logger.info("MILP failed — retrying with CHP LP relaxation to preserve ramp constraints")
            try:
                sol = _solve_cvxpy(inp, _cfg_lp, COP, H, dt)
            except Exception as exc2:
                logger.warning("LP relaxation also failed (%s)", exc2)

    if sol is None:
        sol = _solve_heuristic(inp, cfg_use, COP, H, dt)

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
    c_boiler_kw  = cfg_use.gas_price_boiler_eur_m3 / max(
        cfg_use.gas_HV_kwh_m3 * cfg_use.eta_boiler, 1e-9
    )  # €/kWh thermal

    # Full-horizon MPC cost from solution arrays (electricity + gas)
    mpc_cost = (
        float(np.sum((price + cfg_use.fee_eur_kwh) * sol["Pgrid"])) * dt
        + float(np.sum(sol["Fchp"]))  * cfg_use.gas_price_chp_eur_m3 * dt
        + float(np.sum(sol["Pgas"]))  * c_boiler_kw * dt
    )
    # Full-horizon baseline: rule-based simulation with same physics & forecasts
    baseline_cost = compute_baseline_cost(inp, cfg_use)
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
        plan_Ppv       = sol["Ppv"],
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
            return MPCConfig.from_dict(mpc_block)
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
