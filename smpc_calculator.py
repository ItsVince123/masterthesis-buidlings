"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKEND FILE — student is responsible for this module           ║
║                                                                  ║
║  THIS IS THE CORE THESIS ALGORITHM                               ║
║  Stochastic Model Predictive Control (SMPC) optimiser for        ║
║  building energy management.                                     ║
╚══════════════════════════════════════════════════════════════════╝

smpc_calculator.py — SMPC Live Calculator
==========================================
Single-solve Stochastic MPC that the dashboard calls every control interval.
All parameters are read from the "smpc" block in dashboard_config.json.

Typical integration with dashboard.py:

    from dashboard_config  import load_smpc_config
    from smpc_calculator   import SMPCCalculator, SMPCInputs

    calc = SMPCCalculator(load_smpc_config())   # reads dashboard_config.json

    inputs = SMPCInputs(
        electricity_price_eur_kwh = 0.12,
        price_forecast_eur_kwh    = price_array_96_steps,
        consumption_kwh           = 850.0,
        consumption_forecast_kwh  = consumption_array_96_steps,
        ice_bank_kwh              = 3200.0,
        heat_buffer_kwh           = 5000.0,
        heat_demand_forecast_kwh  = heat_array_96_steps,
        month                     = 7,
    )

    outputs = calc.solve(inputs)

    print(outputs.ice_bank_charge_kwh)    # → charge command this interval
    print(outputs.wkk_gas_setpoint_m3)   # → WKK gas setpoint this interval
    print(outputs.net_power_kwh)         # → expected net grid draw
    print(outputs.cost_saving_eur)       # → saving vs baseline this interval

The solver tries CLARABEL (fast, exact), falls back to SCS, then to a
rule-based heuristic if cvxpy is not installed at all.
"""

from __future__ import annotations

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
        "cvxpy not found — SMPCCalculator will use rule-based heuristics.\n"
        "Install with: pip install cvxpy",
        stacklevel=2,
    )

# MPC LP solver (new formulation)
try:
    from mpc_lp import (  # noqa: E402
        MPCConfig, MPCInputs, MPCOutputs,
        solve_mpc as _mpc_solve,
        load_mpc_config as _load_mpc_config,
    )
    MPC_LP_AVAILABLE = True
except ImportError as _mpc_err:
    MPC_LP_AVAILABLE = False
    warnings.warn(f"mpc_lp not found ({_mpc_err}) — solve_lp will raise.", stacklevel=2)


# ===========================================================================
# SECTION 1: CONFIGURATION
# All physical and cost parameters in one place.
# ===========================================================================

@dataclass
class SMPCConfig:
    """
    All tunable SMPC parameters.

    Construct directly (uses defaults) or load from dashboard_config.json:

        cfg = SMPCConfig.from_config_dict(raw["smpc"])

    The easiest way is via the dashboard_config helper:

        from dashboard_config import load_smpc_config
        cfg = load_smpc_config()
    """

    # --- Optimisation horizon -----------------------------------------------
    horizon_steps:      int   = 96      # Steps ahead to optimise (96 × 15 min = 24 h)

    # --- Ice bank (summer cooling) ------------------------------------------
    ice_bank_capacity_kwh:  float = 8_000.0   # Maximum stored cold energy
    ice_bank_min_kwh:       float =   200.0   # Safety reserve — never fully empty
    ice_charge_max_kwh:     float =   333.0   # Max electrical input per 15-min step (≈1.33 MW)
    ice_discharge_max_kwh:  float =   333.0   # Max thermal release per step
    ice_charge_efficiency:  float =     0.85  # COP-equivalent for charging
    ice_bank_initial_kwh:   float = 1_000.0   # Starting SOE when dashboard boots
    ice_bank_target_soc:    float =     0.30  # End-of-horizon SOC target (fraction of capacity)

    # --- WKK / CHP (winter heat + power) ------------------------------------
    gas_price_eur_m3:           float =    0.35   # Industrial gas tariff (not residential)
    gas_energy_kwh_m3:          float =    9.8    # Calorific value [kWh/m³]
    wkk_elec_efficiency:        float =    0.40   # Electrical efficiency
    wkk_heat_efficiency:        float =    0.45   # Thermal efficiency
    heat_buffer_capacity_kwh:   float = 15_000.0
    heat_buffer_initial_kwh:    float =  3_000.0  # Starting state when dashboard boots
    spark_spread_threshold:     float =    0.01   # Min spark spread [€/kWh] to run WKK
    wkk_max_gas_m3:             float =    0.0    # Max gas burn rate [m³/interval] (from building config)

    # --- Grid connection ----------------------------------------------------
    peak_limit_kwh:             float = 3_250.0   # Per 15-min step = 13 MW peak
    peak_tariff_eur_kw_month:   float =     8.0   # Monthly capacity tariff
    grid_fee_eur_kwh:           float =     0.05  # Grid/distribution charges [€/kWh] (50 €/MWh)

    # --- Objective weights --------------------------------------------------
    w_energy:       float = 1.0    # Weight on energy cost term
    w_peak:         float = 50.0   # Penalty weight on peak excess
    w_heat_comfort: float = 5.0    # Penalty on unmet heat demand
    w_buffer_end:   float = 0.05   # Penalty on missing end-of-horizon buffer target

    # --- Price scenario generation (stochastic MPC) -------------------------
    n_scenarios:        int   = 20     # Number of Monte Carlo price paths
    price_volatility:   float = 0.15   # Log-normal σ per step
    autocorrelation:    float = 0.70   # AR(1) coefficient for price noise

    # --- Season boundaries (month numbers) ----------------------------------
    summer_months:  tuple = (5, 6, 7, 8, 9)
    winter_months:  tuple = (11, 12, 1, 2, 3)
    # April and October are treated as transition months

    # --- Building thermal model ---------------------------------------------
    building_setpoint_c:        float =  21.0   # Target indoor temperature [°C]
    building_deadband_c:        float =   1.0   # Acceptable deviation ± [°C]
    building_initial_temp_c:    float =  18.0   # Start temperature when dashboard boots
    cooldown_rate_c_per_hour:   float =   0.5   # Natural heat loss rate [°C/h] (fallback)
    thermal_mass_kwh_per_c:     float = 500.0   # Thermal inertia: kWh to raise building 1 °C
    ua_kwh_per_c_per_h:         float =   2.5   # Envelope heat loss coefficient [kWh/°C/h]

    # -----------------------------------------------------------------------

    @classmethod
    def from_config_dict(cls, smpc_block: dict) -> "SMPCConfig":
        """
        Build an SMPCConfig from the 'smpc' block in dashboard_config.json.

        Keys that are absent fall back to the dataclass defaults, so you only
        need to specify the values you actually want to override.

        Example raw block (matches dashboard_config.json structure):
            {
              "horizon":   {"steps": 96},
              "ice_bank":  {"capacity_kwh": 8000, ...},
              "wkk":       {"gas_price_eur_m3": 0.35, ...},
              "grid":      {"peak_limit_kwh": 3250, ...},
              "objective_weights": {"energy": 1.0, ...},
              "scenarios": {"count": 20, ...},
              "seasons":   {"summer_months": [5,6,7,8,9], ...}
            }
        """
        def _get(section: str, key: str, default):
            """Safely read smpc_block[section][key], return default if missing."""
            return smpc_block.get(section, {}).get(key, default)

        d = cls()  # start with all defaults

        # horizon
        d.horizon_steps             = int(_get("horizon", "steps",                d.horizon_steps))

        # ice_bank
        d.ice_bank_capacity_kwh     = float(_get("ice_bank", "capacity_kwh",       d.ice_bank_capacity_kwh))
        d.ice_bank_min_kwh          = float(_get("ice_bank", "min_kwh",            d.ice_bank_min_kwh))
        d.ice_charge_max_kwh        = float(_get("ice_bank", "charge_max_kwh",     d.ice_charge_max_kwh))
        d.ice_discharge_max_kwh     = float(_get("ice_bank", "discharge_max_kwh",  d.ice_discharge_max_kwh))
        d.ice_charge_efficiency     = float(_get("ice_bank", "charge_efficiency",  d.ice_charge_efficiency))
        d.ice_bank_initial_kwh      = float(_get("ice_bank", "initial_kwh",        d.ice_bank_initial_kwh))
        d.ice_bank_target_soc       = float(_get("ice_bank", "target_soc_fraction",d.ice_bank_target_soc))

        # wkk
        d.gas_price_eur_m3          = float(_get("wkk", "gas_price_eur_m3",            d.gas_price_eur_m3))
        d.gas_energy_kwh_m3         = float(_get("wkk", "gas_energy_kwh_m3",           d.gas_energy_kwh_m3))
        d.wkk_elec_efficiency       = float(_get("wkk", "elec_efficiency",             d.wkk_elec_efficiency))
        d.wkk_heat_efficiency       = float(_get("wkk", "heat_efficiency",             d.wkk_heat_efficiency))
        d.heat_buffer_capacity_kwh  = float(_get("wkk", "heat_buffer_capacity_kwh",    d.heat_buffer_capacity_kwh))
        d.heat_buffer_initial_kwh   = float(_get("wkk", "heat_buffer_initial_kwh",     d.heat_buffer_initial_kwh))
        d.spark_spread_threshold    = float(_get("wkk", "spark_spread_threshold",      d.spark_spread_threshold))

        # grid
        d.peak_limit_kwh            = float(_get("grid", "peak_limit_kwh",             d.peak_limit_kwh))
        d.peak_tariff_eur_kw_month  = float(_get("grid", "peak_tariff_eur_kw_month",   d.peak_tariff_eur_kw_month))
        d.grid_fee_eur_kwh          = float(_get("grid", "fee_eur_kwh",               d.grid_fee_eur_kwh))

        # objective weights
        d.w_energy      = float(_get("objective_weights", "energy",       d.w_energy))
        d.w_peak        = float(_get("objective_weights", "peak",         d.w_peak))
        d.w_heat_comfort= float(_get("objective_weights", "heat_comfort", d.w_heat_comfort))
        d.w_buffer_end  = float(_get("objective_weights", "buffer_end",   d.w_buffer_end))

        # scenarios
        d.n_scenarios       = int  (_get("scenarios", "count",           d.n_scenarios))
        d.price_volatility  = float(_get("scenarios", "volatility",      d.price_volatility))
        d.autocorrelation   = float(_get("scenarios", "autocorrelation", d.autocorrelation))

        # seasons
        d.summer_months = tuple(_get("seasons", "summer_months", list(d.summer_months)))
        d.winter_months = tuple(_get("seasons", "winter_months", list(d.winter_months)))

        # building
        d.wkk_max_gas_m3 = float(_get("building", "wkk_max_gas_m3", d.wkk_max_gas_m3))

        # building thermal model
        thermal = smpc_block.get("building", {}).get("thermal", {})
        d.building_setpoint_c      = float(thermal.get("setpoint_c",              d.building_setpoint_c))
        d.building_deadband_c      = float(thermal.get("deadband_c",              d.building_deadband_c))
        d.building_initial_temp_c  = float(thermal.get("initial_temp_c",          d.building_initial_temp_c))
        d.cooldown_rate_c_per_hour = float(thermal.get("cooldown_rate_c_per_hour", d.cooldown_rate_c_per_hour))
        d.thermal_mass_kwh_per_c   = float(thermal.get("thermal_mass_kwh_per_c",  d.thermal_mass_kwh_per_c))
        d.ua_kwh_per_c_per_h       = float(thermal.get("ua_kwh_per_c_per_h",       d.ua_kwh_per_c_per_h))

        return d


# ===========================================================================
# SECTION 2: INPUT / OUTPUT DATA STRUCTURES
# ===========================================================================

@dataclass
class SMPCInputs:
    """
    Everything the calculator needs for one solve call.

    Scalar fields describe the CURRENT measurement (this 15-min interval).
    Array fields are FORECASTS for the optimisation horizon.

    If you only have the current price (no forecast), pass a flat array:
        price_forecast_eur_kwh = np.full(96, current_price)
    """

    # Current spot price [€/kWh]
    electricity_price_eur_kwh: float

    # Price forecast for the next `horizon_steps` intervals [€/kWh]
    # Shape: (horizon_steps,)   — padded/clipped inside solver if needed
    price_forecast_eur_kwh: np.ndarray

    # Current measured building consumption [kWh / 15-min interval]
    consumption_kwh: float

    # Consumption forecast [kWh / interval], shape (horizon_steps,)
    consumption_forecast_kwh: np.ndarray

    # Current ice bank state of energy [kWh]
    ice_bank_kwh: float = 1_000.0

    # Current heat buffer state [kWh]
    heat_buffer_kwh: float = 3_000.0

    # Heat demand forecast [kWh / interval], shape (horizon_steps,)
    # Set to zeros if no WKK / heat system present
    heat_demand_forecast_kwh: Optional[np.ndarray] = None

    # Max WKK gas rate this site can burn [m³ / interval]
    # Set to 0.0 if no WKK present
    wkk_max_gas_m3: float = 0.0

    # Current month (1–12) — used to determine season
    month: int = 7

    # Baseline costs this interval (for savings calculation)
    # Provide if known; otherwise the calculator estimates from inputs
    baseline_cost_eur: Optional[float] = None


@dataclass
class SMPCOutputs:
    """
    Optimal control actions and KPIs returned by one solve call.

    The FIRST-STEP values are what the controller should physically execute
    this interval.  The PLAN arrays are the full horizon schedule (useful
    for dashboard trend charts).
    """

    # --- First-step commands (execute NOW) ----------------------------------

    # Ice bank: electrical power to draw for charging [kWh]
    ice_bank_charge_kwh: float = 0.0

    # Ice bank: thermal energy to release for cooling [kWh]
    ice_bank_discharge_kwh: float = 0.0

    # WKK: gas consumption setpoint [m³]
    wkk_gas_setpoint_m3: float = 0.0

    # --- Derived first-step KPIs --------------------------------------------

    # Expected net grid draw this interval [kWh]
    net_power_kwh: float = 0.0

    # WKK electrical output this interval [kWh]
    wkk_elec_kwh: float = 0.0

    # WKK heat output this interval [kWh]
    wkk_heat_kwh: float = 0.0

    # Cost under SMPC this interval [€]
    smpc_cost_eur: float = 0.0

    # Cost under baseline (no optimisation) this interval [€]
    baseline_cost_eur: float = 0.0

    # Saving vs baseline [€] — positive means SMPC is cheaper
    cost_saving_eur: float = 0.0

    # Ice bank state after executing first-step command [kWh]
    ice_bank_next_kwh: float = 0.0

    # Heat buffer state after executing first-step command [kWh]
    heat_buffer_next_kwh: float = 0.0

    # --- Full-horizon plans (for dashboard charts) --------------------------

    # Shape: (horizon_steps,) — full charge schedule
    plan_ice_charge_kwh:    np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (horizon_steps,) — full discharge schedule
    plan_ice_discharge_kwh: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (horizon_steps,) — full WKK gas schedule
    plan_wkk_gas_m3:        np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (horizon_steps,) — expected net power over horizon
    plan_net_power_kwh:     np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Per-asset first-step power (kW) ------------------------------------
    asset_power_kw: dict = field(default_factory=dict)  # uid → kW

    # --- Per-asset full schedule (for slider) --------------------------------
    asset_schedules: dict = field(default_factory=dict)  # uid → np.ndarray (H,)

    # --- Solver metadata ----------------------------------------------------
    season:        str = "unknown"   # "summer" / "winter" / "transition"
    solver_used:   str = "none"      # "clarabel" / "scs" / "heuristic"
    solver_status: str = "unknown"
    solve_time_ms: float = 0.0

    # --- Building thermal state ---------------------------------------------
    building_temp_c:     float = 21.0    # Current simulated indoor temperature [°C]
    building_setpoint_c: float = 21.0    # Target setpoint [°C]
    heating_power_kw:    float = 0.0     # Total active heating this step [kW]
    beo_temp_c:          float = 12.0    # BEO-veld ground temperature [°C]

    # Shape: (horizon_steps,) — full building temperature profile
    plan_building_temp_c: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (horizon_steps,) — full heating power profile [kW]
    plan_heating_kw: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (horizon_steps,) — full BEO-veld temperature profile [°C]
    plan_beo_temp_c: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- Hot water tank state ------------------------------------------------
    hw_temp_c:       float = 55.0    # Current simulated tank temperature [°C]
    hw_heater_kw:    float = 0.0     # Heater power active this step [kW]

    # Shape: (horizon_steps,) — full HW temperature profile [°C]
    plan_hw_temp_c: np.ndarray = field(default_factory=lambda: np.array([]))

    # --- MPC LP extended outputs (new decision variables) --------------------
    battery_soc_kwh:       float = 0.0    # Battery SOC after first step [kWh]
    battery_charge_kw:     float = 0.0    # Battery charging power [kW]
    battery_discharge_kw:  float = 0.0    # Battery discharging power [kW]
    pv_used_kw:            float = 0.0    # PV power used this step [kW]
    gas_boiler_kw:         float = 0.0    # Gas boiler thermal output [kW]
    cop_now:               float = 4.0    # Heat pump COP at current ambient temp
    zchp_now:              float = 0.0    # CHP on/off state (0 or 1)
    chp_elec_kw:           float = 0.0    # CHP electrical output this step [kW]
    chp_heat_kw:           float = 0.0    # CHP thermal output this step [kW]
    pgrid_kw:              float = 0.0    # Grid import this step [kW] (direct from MPC Pgrid)

    # Shape: (H+1,) — full SOC profile (index 0 = initial state)
    plan_SOC: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full COP profile
    plan_COP: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full PV power profile [kW]
    plan_Ppv: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full CHP on/off profile
    plan_zchp: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full CHP electrical output profile [kW]
    plan_chp_elec_kw: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full CHP thermal output profile [kW]
    plan_chp_heat_kw: np.ndarray = field(default_factory=lambda: np.array([]))

    # Shape: (H,) — full grid import profile [kW] (direct from MPC plan_Pgrid)
    plan_pgrid_kw: np.ndarray = field(default_factory=lambda: np.array([]))


# ===========================================================================
# SECTION 3: PRICE SCENARIO GENERATOR
# ===========================================================================

def _generate_price_scenarios(
    base_forecast: np.ndarray,
    n_scenarios: int,
    volatility: float,
    autocorrelation: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate N correlated price scenarios around the base forecast.

    MODEL: AR(1) log-normal noise on top of the deterministic forecast.

    For each scenario s and time step t:
        noise[t] = ρ · noise[t-1] + σ_corr · ε_t       (AR(1) process)
        scenario[s, t] = base_forecast[t] · exp(noise[t])

    where:
        ρ           = autocorrelation coefficient (persistence of noise)
        σ_corr      = volatility × sqrt(1 - ρ²)         (innovations std)
        ε_t ~ N(0,1) i.i.d.

    The exp() ensures prices stay positive (log-normal distribution).
    In production, replace with ENTSO-E DAM distributions or an ARIMA/GARCH
    fit on historical EPEX data for better accuracy.

    Returns shape (n_scenarios, horizon_steps), all values ≥ 0.
    """
    rng = np.random.default_rng(seed)
    H = len(base_forecast)
    scenarios = np.zeros((n_scenarios, H))
    noise_scale = volatility * np.sqrt(1.0 - autocorrelation ** 2)

    for s in range(n_scenarios):
        noise = np.zeros(H)
        noise[0] = rng.normal(0, volatility)
        for t in range(1, H):
            noise[t] = autocorrelation * noise[t - 1] + rng.normal(0, noise_scale)
        scenarios[s] = base_forecast * np.exp(noise)

    return np.maximum(scenarios, 0.0)


# ===========================================================================
# SECTION 4: OPTIMISATION ROUTINES
# ===========================================================================


from lp_solver import (
    greedy_lp as _greedy_lp,
    greedy_lp_ramped as _greedy_lp_ramped,
    simulate_building_thermal as _simulate_thermal,
    _chp_optimal_schedule as _chp_schedule,
    _effective_prices as _eff_prices,
)


def _optimise_summer_cvxpy(
    mean_price: np.ndarray,
    consumption: np.ndarray,
    ice_bank_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """
    Convex QP for summer ice-bank optimisation (uses cvxpy).

    PROBLEM FORMULATION
    -------------------
    Decision variables:
        u_charge[t]    — electrical energy drawn to charge ice bank [kWh]
        u_discharge[t] — thermal energy released from ice bank [kWh]
        slack[t]       — peak constraint violation (soft, penalised) [kWh]

    State evolution (ice bank state of energy, SOE):
        soe[t] = ice_bank_kwh + Σ_{k=0}^{t} (u_charge[k]·η - u_discharge[k])
    Implemented via cumsum for numerical efficiency.

    Objective (minimise):
        J = w_energy  × Σ_t total_price[t] · P_net[t]     (energy cost)
          + w_peak    × Σ_t slack[t]²                       (soft peak penalty)
          + w_buffer  × (soe[H-1] - target)²                (end-of-horizon penalty)

    where:
        total_price[t] = spot_price[t] + grid_fee             (EUR/kWh)
        P_net[t]       = consumption[t] + u_charge[t] - u_discharge[t]
        target         = capacity × target_soc_fraction

    Constraints:
        0 ≤ u_charge[t]    ≤ ice_charge_max_kwh
        0 ≤ u_discharge[t] ≤ ice_discharge_max_kwh
        ice_bank_min_kwh   ≤ soe[t] ≤ ice_bank_capacity_kwh
        P_net[t]           ≤ peak_limit_kwh + slack[t]   (soft, slack ≥ 0)

    SOLVER: CLARABEL (first choice, fast exact QP), falls back to SCS.
    """
    H = len(mean_price)
    u_charge    = cp.Variable(H, nonneg=True)
    u_discharge = cp.Variable(H, nonneg=True)
    slack       = cp.Variable(H, nonneg=True)   # Peak limit softener

    # Vectorized state evolution via cumsum (avoids nested expression trees)
    delta         = u_charge * cfg.ice_charge_efficiency - u_discharge
    buffer_states = ice_bank_kwh + cp.cumsum(delta)   # shape (H,)

    p_net  = consumption + u_charge - u_discharge
    target = cfg.ice_bank_capacity_kwh * cfg.ice_bank_target_soc

    # Total cost per kWh = spot price + grid/distribution fee
    total_price = mean_price + cfg.grid_fee_eur_kwh

    objective = (
        cfg.w_energy    * cp.sum(cp.multiply(total_price, p_net))
        + cfg.w_peak    * cp.sum_squares(slack)
        + cfg.w_buffer_end * cp.square(buffer_states[-1] - target)
    )

    constraints = [
        u_charge      <= cfg.ice_charge_max_kwh,
        u_discharge   <= cfg.ice_discharge_max_kwh,
        buffer_states >= cfg.ice_bank_min_kwh,
        buffer_states <= cfg.ice_bank_capacity_kwh,
        p_net         <= cfg.peak_limit_kwh + slack,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        solver_name = "clarabel"
    except Exception:
        problem.solve(solver=cp.SCS)
        solver_name = "scs"

    if problem.status in ("optimal", "optimal_inaccurate"):
        return {
            "charge":    u_charge.value,
            "discharge": u_discharge.value,
            "solver":    solver_name,
            "status":    problem.status,
        }
    return None  # Signal to fall back to heuristic


def _optimise_summer_heuristic(
    mean_price: np.ndarray,
    consumption: np.ndarray,
    ice_bank_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """
    Rule-based fallback for summer when cvxpy is unavailable or infeasible.

    Logic:
      - Charge when price is in the lowest 35th percentile AND buffer has room
      - Discharge when price is in the top 40th percentile AND buffer is sufficient
      - Force-discharge if buffer > 95 % full
    """
    H = len(mean_price)
    charge    = np.zeros(H)
    discharge = np.zeros(H)
    buf       = ice_bank_kwh

    price_low  = np.percentile(mean_price, 35)
    price_high = np.percentile(mean_price, 60)
    base_load  = consumption.min() * 0.95   # Conservative base load estimate

    for t in range(H):
        price = mean_price[t]

        # Charge: price is cheap AND buffer has headroom AND won't breach peak
        if price <= price_low and buf < cfg.ice_bank_capacity_kwh * 0.90:
            headroom_peak = max(0.0, cfg.peak_limit_kwh - consumption[t])
            to_charge = min(
                cfg.ice_charge_max_kwh * 0.8,
                (cfg.ice_bank_capacity_kwh * 0.90 - buf) / cfg.ice_charge_efficiency,
                headroom_peak,
            )
            charge[t] = max(0.0, to_charge)
            buf += charge[t] * cfg.ice_charge_efficiency

        # Discharge: price is high AND buffer not depleted
        elif price >= price_high and buf > cfg.ice_bank_min_kwh:
            flex = max(0.0, consumption[t] - base_load)
            to_discharge = min(cfg.ice_discharge_max_kwh, buf - cfg.ice_bank_min_kwh, flex)
            discharge[t] = max(0.0, to_discharge)
            buf -= discharge[t]

        # Force discharge if buffer is nearly full
        if buf > cfg.ice_bank_capacity_kwh * 0.95 and consumption[t] > base_load:
            extra = min(cfg.ice_discharge_max_kwh * 0.5, buf - cfg.ice_bank_capacity_kwh * 0.80)
            discharge[t] += max(0.0, extra)
            buf -= max(0.0, extra)

        buf = float(np.clip(buf, cfg.ice_bank_min_kwh, cfg.ice_bank_capacity_kwh))

    return {"charge": charge, "discharge": discharge, "solver": "heuristic", "status": "ok"}


def _solve_summer(
    scenarios: np.ndarray,
    consumption: np.ndarray,
    ice_bank_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """Select cvxpy or heuristic solver for summer and return result dict."""
    mean_price = scenarios.mean(axis=0)

    if CVXPY_AVAILABLE:
        result = _optimise_summer_cvxpy(mean_price, consumption, ice_bank_kwh, cfg)
        if result is not None:
            return result

    return _optimise_summer_heuristic(mean_price, consumption, ice_bank_kwh, cfg)


def _optimise_winter_cvxpy(
    mean_price: np.ndarray,
    heat_demand: np.ndarray,
    wkk_max_gas: float,
    heat_buffer_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """
    Convex QP for winter WKK / CHP optimisation (uses cvxpy).

    PROBLEM FORMULATION
    -------------------
    Decision variables:
        u_gas[t]      — gas burned per interval [m³]
        heat_slack[t] — unmet heat demand (comfort penalty) [kWh]

    Derived quantities:
        heat_prod[t]  = u_gas[t] × calorific_value × heat_efficiency  [kWh heat]
        elec_prod[t]  = u_gas[t] × calorific_value × elec_efficiency  [kWh elec]

    Heat buffer evolution:
        buffer[t] = heat_buffer_kwh + Σ_{k=0}^{t} (heat_prod[k] − heat_demand[k] + heat_slack[k])

    Objective (minimise):
        J = Σ_t gas_cost_per_m3 · u_gas[t]       (gas cost)
          − Σ_t (mean_price[t] + grid_fee) · elec_prod[t]  (avoided grid import ← negative term)
          + w_heat_comfort × Σ_t heat_slack[t]²   (penalty for unmet heat demand)

    Economic intuition: WKK is run when the avoided electricity cost exceeds
    the gas cost — i.e. the spark spread is positive.

    Constraints:
        0 ≤ u_gas[t]   ≤ wkk_max_gas
        0 ≤ buffer[t]  ≤ heat_buffer_capacity_kwh

    SOLVER: CLARABEL, falls back to SCS.
    """
    H = len(mean_price)
    u_gas       = cp.Variable(H, nonneg=True)
    heat_slack  = cp.Variable(H, nonneg=True)   # Unmet heat demand (comfort penalty)

    heat_prod = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency
    elec_prod = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency

    # Vectorized state evolution via cumsum (avoids nested expression trees)
    heat_net      = heat_prod - heat_demand + heat_slack
    buffer_states = heat_buffer_kwh + cp.cumsum(heat_net)   # shape (H,)

    gas_cost         = cp.sum(u_gas) * cfg.gas_price_eur_m3
    # WKK electricity avoids grid import → value includes avoided grid fee
    avoided_price    = mean_price + cfg.grid_fee_eur_kwh
    electricity_rev  = cp.sum(cp.multiply(avoided_price, elec_prod))
    comfort_penalty  = cfg.w_heat_comfort * cp.sum_squares(heat_slack)

    objective = gas_cost - electricity_rev + comfort_penalty

    constraints = [
        u_gas         <= wkk_max_gas,
        buffer_states >= 0,
        buffer_states <= cfg.heat_buffer_capacity_kwh,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True)
        solver_name = "clarabel"
    except Exception:
        problem.solve(solver=cp.SCS)
        solver_name = "scs"

    if problem.status in ("optimal", "optimal_inaccurate"):
        return {
            "gas":    u_gas.value,
            "solver": solver_name,
            "status": problem.status,
        }
    return None


def _optimise_winter_heuristic(
    mean_price: np.ndarray,
    heat_demand: np.ndarray,
    wkk_max_gas: float,
    heat_buffer_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """
    Spark-spread heuristic for winter.

    Runs WKK at full capacity when spark spread is positive.
    Falls back to minimum gas needed to cover heat demand from buffer.
    """
    H   = len(mean_price)
    gas = np.zeros(H)
    buf = heat_buffer_kwh

    for t in range(H):
        price          = mean_price[t]
        heat_needed    = heat_demand[t]

        # Spark spread: revenue from 1 m³ gas minus the gas cost.
        # WKK electricity avoids grid import, so its value includes the
        # avoided grid fee (distribution/network charges), not just spot.
        effective_price = price + cfg.grid_fee_eur_kwh
        electricity_rev_per_m3 = cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency * effective_price
        spark_spread   = electricity_rev_per_m3 - cfg.gas_price_eur_m3

        if spark_spread > cfg.spark_spread_threshold and buf < cfg.heat_buffer_capacity_kwh:
            # Run WKK hard — profitable AND buffer has room
            buffer_headroom = cfg.heat_buffer_capacity_kwh - buf
            max_gas_by_buffer = buffer_headroom / (cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency)
            gas[t] = min(wkk_max_gas, max_gas_by_buffer)
        else:
            # Run WKK only as needed to cover heat demand buffer cannot meet
            if buf >= heat_needed:
                gas[t] = 0.0
            else:
                heat_shortfall = heat_needed - buf
                gas[t] = min(
                    wkk_max_gas,
                    heat_shortfall / (cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency),
                )

        heat_produced = gas[t] * cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency
        buf = float(np.clip(buf + heat_produced - heat_needed, 0, cfg.heat_buffer_capacity_kwh))

    return {"gas": gas, "solver": "heuristic", "status": "ok"}


def _solve_winter(
    scenarios: np.ndarray,
    heat_demand: np.ndarray,
    wkk_max_gas: float,
    heat_buffer_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """Select cvxpy or heuristic solver for winter and return result dict."""
    mean_price = scenarios.mean(axis=0)

    if CVXPY_AVAILABLE:
        result = _optimise_winter_cvxpy(mean_price, heat_demand, wkk_max_gas, heat_buffer_kwh, cfg)
        if result is not None:
            return result

    return _optimise_winter_heuristic(mean_price, heat_demand, wkk_max_gas, heat_buffer_kwh, cfg)


# ===========================================================================
# SECTION 5: SEASON HELPER
# ===========================================================================

def _get_season(month: int, cfg: SMPCConfig) -> str:
    """Return 'summer', 'winter', or 'transition' based on the current month."""
    if month in cfg.summer_months:
        return "summer"
    if month in cfg.winter_months:
        return "winter"
    return "transition"


# ===========================================================================
# SECTION 6: MAIN CALCULATOR CLASS
# ===========================================================================

class SMPCCalculator:
    """
    Live SMPC calculator for the dashboard.

    Usage
    -----
    Instantiate once (reads dashboard_config.json automatically), then call
    `solve()` on every dashboard tick:

        calc    = SMPCCalculator()          # loads config from dashboard_config.json
        outputs = calc.solve(inputs)

    You can also pass an explicit config:

        calc = SMPCCalculator(SMPCConfig.from_config_dict(raw_smpc_block))

    The calculator is stateless between calls — all state (ice bank level,
    heat buffer level) is passed in via SMPCInputs each time.  This keeps
    the dashboard responsible for state tracking, which is cleaner.
    """

    def __init__(self, config=None):
        if isinstance(config, str):
            # config is a filesystem path to dashboard_config.json
            self.config_path = config
            try:
                from dashboard_config import load_smpc_config
                self.cfg = load_smpc_config(config)
            except Exception:
                self.cfg = SMPCConfig()
        elif config is not None:
            self.cfg = config
            self.config_path = "dashboard_config.json"
        else:
            # Auto-load from dashboard_config.json sitting next to this file.
            # Falls back to hardcoded defaults if the file is missing.
            self.config_path = "dashboard_config.json"
            try:
                from dashboard_config import load_smpc_config
                self.cfg = load_smpc_config()
            except Exception:
                self.cfg = SMPCConfig()

    @property
    def mpc_cfg(self) -> "MPCConfig":
        """
        MPC LP configuration (MPCConfig).

        Reloads from dashboard_config.json on every access so that GUI
        changes (MpcConfigDialog) take effect at the next solve call
        without restarting the dashboard.
        """
        if not MPC_LP_AVAILABLE:
            raise RuntimeError("mpc_lp.py is not available")
        return _load_mpc_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, inputs: SMPCInputs) -> SMPCOutputs:
        """
        Run one SMPC solve and return optimal control outputs.

        Steps:
          1. Validate and pad/clip forecast arrays to the horizon length
          2. Determine season from current month
          3. Generate stochastic price scenarios
          4. Run the appropriate optimiser (summer / winter / transition)
          5. Extract the first-step action (receding horizon principle)
          6. Compute KPIs (cost, savings, buffer states)
          7. Return SMPCOutputs
        """
        import time
        t_start = time.perf_counter()

        cfg = self.cfg
        H   = cfg.horizon_steps

        # 1. Prepare forecasts (ensure correct length)
        price_fc       = self._pad_forecast(inputs.price_forecast_eur_kwh,       H)
        consumption_fc = self._pad_forecast(inputs.consumption_forecast_kwh,     H)
        heat_fc        = self._pad_forecast(
            inputs.heat_demand_forecast_kwh if inputs.heat_demand_forecast_kwh is not None
            else np.zeros(H),
            H,
        )

        wkk_max_gas = max(0.0, inputs.wkk_max_gas_m3)

        # 2. Season
        season = _get_season(inputs.month, cfg)

        # 3. Price scenarios
        # Seed varies by month and current price level so scenarios are not
        # identical across every solve in the same month.
        seed = abs(hash((inputs.month, int(price_fc[0] * 10_000)))) % (2**31)
        scenarios = _generate_price_scenarios(
            base_forecast  = price_fc,
            n_scenarios    = cfg.n_scenarios,
            volatility     = cfg.price_volatility,
            autocorrelation= cfg.autocorrelation,
            seed           = seed,
        )

        # 4. Solve
        charge_plan    = np.zeros(H)
        discharge_plan = np.zeros(H)
        gas_plan       = np.zeros(H)
        solver_used    = "none"
        solver_status  = "skipped"

        if season == "summer":
            res = _solve_summer(scenarios, consumption_fc, inputs.ice_bank_kwh, cfg)
            charge_plan    = np.array(res["charge"],    dtype=float)
            discharge_plan = np.array(res["discharge"], dtype=float)
            solver_used    = res["solver"]
            solver_status  = res["status"]

        elif season == "winter" and wkk_max_gas > 0:
            res = _solve_winter(scenarios, heat_fc, wkk_max_gas, inputs.heat_buffer_kwh, cfg)
            gas_plan      = np.array(res["gas"], dtype=float)
            solver_used   = res["solver"]
            solver_status = res["status"]

        else:
            # Transition: run both systems at reduced capacity
            cfg_half = SMPCConfig(**self.cfg.__dict__)
            cfg_half.ice_charge_max_kwh    = cfg.ice_charge_max_kwh    * 0.6
            cfg_half.ice_discharge_max_kwh = cfg.ice_discharge_max_kwh * 0.6
            cfg_half.spark_spread_threshold = cfg.spark_spread_threshold * 1.5  # Stricter WKK

            res_s = _solve_summer(scenarios, consumption_fc, inputs.ice_bank_kwh, cfg_half)
            charge_plan    = np.array(res_s["charge"],    dtype=float)
            discharge_plan = np.array(res_s["discharge"], dtype=float)
            solver_used    = res_s["solver"]
            solver_status  = res_s["status"]

            if wkk_max_gas > 0:
                res_w = _solve_winter(scenarios, heat_fc, wkk_max_gas, inputs.heat_buffer_kwh, cfg_half)
                gas_plan = np.array(res_w["gas"], dtype=float)

        # 5. Extract first-step action (receding horizon)
        u_charge    = float(np.clip(charge_plan[0],    0, cfg.ice_charge_max_kwh))
        u_discharge = float(np.clip(
            discharge_plan[0], 0,
            min(cfg.ice_discharge_max_kwh, inputs.ice_bank_kwh - cfg.ice_bank_min_kwh),
        ))
        u_gas = float(np.clip(gas_plan[0], 0, wkk_max_gas))

        # 6. Compute KPIs
        wkk_elec = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency
        wkk_heat = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency
        net_power = inputs.consumption_kwh + u_charge - u_discharge - wkk_elec

        # Total electricity cost includes spot price + grid/distribution fee
        total_price = inputs.electricity_price_eur_kwh + cfg.grid_fee_eur_kwh
        smpc_elec_cost = net_power * total_price
        smpc_gas_cost  = u_gas     * cfg.gas_price_eur_m3
        smpc_cost      = smpc_elec_cost + smpc_gas_cost

        if inputs.baseline_cost_eur is not None:
            baseline_cost = inputs.baseline_cost_eur
        else:
            # Estimate: building draws full consumption from grid, no optimisation
            baseline_cost = inputs.consumption_kwh * total_price

        saving = baseline_cost - smpc_cost

        # Buffer states after first-step action
        ice_next = float(np.clip(
            inputs.ice_bank_kwh + u_charge * cfg.ice_charge_efficiency - u_discharge,
            cfg.ice_bank_min_kwh,
            cfg.ice_bank_capacity_kwh,
        ))
        heat_next = float(np.clip(
            inputs.heat_buffer_kwh + wkk_heat - (heat_fc[0] if len(heat_fc) else 0),
            0,
            cfg.heat_buffer_capacity_kwh,
        ))

        # Full horizon net power plan (for dashboard charts)
        wkk_elec_plan = gas_plan * cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency
        net_power_plan = consumption_fc + charge_plan - discharge_plan - wkk_elec_plan

        solve_ms = (time.perf_counter() - t_start) * 1000

        return SMPCOutputs(
            # First-step commands
            ice_bank_charge_kwh    = u_charge,
            ice_bank_discharge_kwh = u_discharge,
            wkk_gas_setpoint_m3    = u_gas,

            # KPIs
            net_power_kwh       = net_power,
            wkk_elec_kwh        = wkk_elec,
            wkk_heat_kwh        = wkk_heat,
            smpc_cost_eur       = smpc_cost,
            baseline_cost_eur   = baseline_cost,
            cost_saving_eur     = saving,
            ice_bank_next_kwh   = ice_next,
            heat_buffer_next_kwh= heat_next,

            # Full-horizon plans
            plan_ice_charge_kwh    = charge_plan,
            plan_ice_discharge_kwh = discharge_plan,
            plan_wkk_gas_m3        = gas_plan,
            plan_net_power_kwh     = net_power_plan,

            # Metadata
            season        = season,
            solver_used   = solver_used,
            solver_status = solver_status,
            solve_time_ms = solve_ms,
        )

    # ------------------------------------------------------------------
    # MPC LP -- receding-horizon solver (replaces greedy LP)
    # ------------------------------------------------------------------

    def solve_lp(
        self,
        price_forecast_eur_kwh: np.ndarray,
        base_load_kwh: np.ndarray | None = None,
        solar_pred_kwh: np.ndarray | None = None,
        month: int = 7,
        building_temp_c: float | None = None,
        outside_temp_c: np.ndarray | None = None,
        hw_temp_c: float | None = None,
    ) -> SMPCOutputs:
        """
        MPC LP/MILP solve -- receding-horizon building energy optimisation.

        Calls mpc_lp.solve_mpc() which optimises all decision variables
        simultaneously over the full horizon H:

            Pgrid[k]      grid import power            [kW]
            Php[k]        heat pump thermal output     [kW]
            Pgas[k]       gas boiler thermal output    [kW]
            Pch[k]        battery charging power       [kW]
            Pdis[k]       battery discharging power    [kW]
            SOC[k]        battery state of charge      [kWh]
            Pflex[k]      flexible / shiftable load    [kW]
            Ppv[k]        PV power actually used       [kW]
            Ptank[k]      hot water heater power       [kW]
            Ttank[k]      hot water tank temperature   [C]
            Fchp[k]       CHP gas flow rate            [m3/h]
            Pchp[k]       CHP electrical output        [kW]
            Qchp[k]       CHP thermal output           [kW]
            zchp[k]       CHP on/off binary            {0,1}
            ychp[k]       CHP startup binary           {0,1}
            Tbuilding[k]  building temperature         [C]

        COP model (precomputed parameter):
            COP[k] = COP0 x max(COP_min, 1 - cop_alpha x (Tamb[k] - T0))

        All efficiencies and parameters are read from the 'mpc' block
        in dashboard_config.json (editable via MpcConfigDialog).

        Parameters match the legacy signature so the dashboard needs only
        the change:  H = self.lp.mpc_cfg.horizon_steps
        """
        import time as _time

        if not MPC_LP_AVAILABLE:
            raise RuntimeError(
                "mpc_lp.py is not importable. "
                "Install cvxpy: pip install cvxpy"
            )

        t0  = _time.perf_counter()
        cfg = self.mpc_cfg          # reloads from JSON on each call
        H   = cfg.horizon_steps
        dt  = cfg.dt_hours

        def _pad(arr, fallback: float = 0.0) -> np.ndarray:
            if arr is None:
                return np.full(H, fallback)
            a = np.asarray(arr, dtype=float).ravel()
            if len(a) >= H:
                return a[:H]
            tail = float(a[-1]) if len(a) else fallback
            return np.concatenate([a, np.full(H - len(a), tail)])

        price_fc = _pad(price_forecast_eur_kwh, 0.15)

        # DataManager provides kWh/step; convert to kW for the MPC
        if base_load_kwh is not None:
            Pload = _pad(np.asarray(base_load_kwh, float) / max(dt, 1e-9), 50.0)
        else:
            t_h   = np.arange(H, dtype=float) * dt
            Pload = 50.0 + 30.0 * np.abs(np.sin(t_h * np.pi / 12.0))

        Ppv_fc = (_pad(np.asarray(solar_pred_kwh, float) / max(dt, 1e-9), 0.0)
                  if solar_pred_kwh is not None else np.zeros(H))

        Tamb = (_pad(outside_temp_c, cfg.T_init_c - 5.0)
                if outside_temp_c is not None
                else np.full(H, cfg.T_init_c - 5.0))

        inp = MPCInputs(
            price_eur_kwh     = price_fc,
            Pload_kw          = Pload,
            Ppv_forecast_kw   = Ppv_fc,
            Tamb_c            = Tamb,
            SOC_init_kwh      = cfg.SOC_init_kwh,
            T_building_init_c = (building_temp_c if building_temp_c is not None
                                 else cfg.T_init_c),
            T_tank_init_c     = (hw_temp_c if hw_temp_c is not None
                                 else cfg.hw_T_init_c),
            month             = month,
        )

        # Solve via MPC LP/MILP
        mpc = _mpc_solve(inp, cfg)

        # Map MPCOutputs -> SMPCOutputs (backward-compatible)
        plan_Tbld = (mpc.plan_Tbuilding[1:]
                     if len(mpc.plan_Tbuilding) > 1
                     else np.full(H, mpc.Tbuilding_c))
        plan_heat  = mpc.plan_Php + mpc.plan_Pgas + mpc.plan_Qchp
        plan_Ttank = (mpc.plan_Ttank[1:]
                      if len(mpc.plan_Ttank) > 1
                      else np.full(H, mpc.Ttank_c))

        plan_grid = mpc.plan_Pgrid * dt
        plan_flex = mpc.plan_Pflex * dt
        plan_pdis = mpc.plan_Pdis  * dt
        plan_fchp = mpc.plan_Fchp  * dt   # m3/h x h = m3/step

        asset_power: dict = {}
        asset_sched: dict = {}
        if cfg.pv_enabled:
            asset_power["pv"]          = mpc.Ppv_kw
            asset_sched["pv"]          = mpc.plan_Ppv * dt
        if cfg.hp_enabled:
            asset_power["heat_pump"]   = mpc.Php_kw
            asset_sched["heat_pump"]   = mpc.plan_Php * dt
        if cfg.boiler_enabled:
            asset_power["gas_boiler"]  = mpc.Pgas_kw
            asset_sched["gas_boiler"]  = mpc.plan_Pgas * dt
        if cfg.chp_enabled:
            asset_power["chp"]         = mpc.Pchp_kw
            asset_sched["chp"]         = mpc.plan_Pchp * dt
        if cfg.bat_enabled:
            asset_power["battery_charge"]    = mpc.Pch_kw
            asset_power["battery_discharge"] = mpc.Pdis_kw
            asset_sched["battery_charge"]    = mpc.plan_Pch  * dt
            asset_sched["battery_discharge"] = mpc.plan_Pdis * dt
        if cfg.flex_enabled:
            asset_power["flexible_load"] = mpc.Pflex_kw
            asset_sched["flexible_load"] = plan_flex
        if cfg.hw_enabled:
            asset_power["hot_water_tank"] = mpc.Ptank_kw
            asset_sched["hot_water_tank"] = mpc.plan_Ptank * dt

        solve_ms = (_time.perf_counter() - t0) * 1000.0

        return SMPCOutputs(
            ice_bank_charge_kwh    = mpc.Pflex_kw  * dt,
            ice_bank_discharge_kwh = mpc.Pdis_kw   * dt,
            wkk_gas_setpoint_m3    = mpc.Fchp_m3_h * dt,
            net_power_kwh          = mpc.Pgrid_kw   * dt,
            wkk_elec_kwh           = mpc.Pchp_kw    * dt,
            wkk_heat_kwh           = mpc.Qchp_kw    * dt,
            smpc_cost_eur          = mpc.mpc_cost_eur,
            baseline_cost_eur      = mpc.baseline_cost_eur,
            cost_saving_eur        = mpc.cost_saving_eur,
            ice_bank_next_kwh      = mpc.SOC_kwh,
            heat_buffer_next_kwh   = 0.0,
            asset_power_kw         = asset_power,
            asset_schedules        = asset_sched,
            plan_ice_charge_kwh    = plan_flex,
            plan_ice_discharge_kwh = plan_pdis,
            plan_wkk_gas_m3        = plan_fchp,
            plan_net_power_kwh     = plan_grid,
            season        = "mpc",
            solver_used   = mpc.solver_used,
            solver_status = mpc.solver_status,
            solve_time_ms = solve_ms,
            building_temp_c      = mpc.Tbuilding_c,
            building_setpoint_c  = cfg.Tset_c,
            heating_power_kw     = mpc.total_heating_kw,
            beo_temp_c           = 12.0,
            plan_building_temp_c = plan_Tbld,
            plan_heating_kw      = plan_heat,
            plan_beo_temp_c      = np.full(len(plan_Tbld), 12.0),
            hw_temp_c      = mpc.Ttank_c,
            hw_heater_kw   = mpc.Ptank_kw,
            plan_hw_temp_c = plan_Ttank,
            battery_soc_kwh      = mpc.SOC_kwh,
            battery_charge_kw    = mpc.Pch_kw,
            battery_discharge_kw = mpc.Pdis_kw,
            pv_used_kw           = mpc.Ppv_kw,
            gas_boiler_kw        = mpc.Pgas_kw,
            cop_now              = mpc.COP_now,
            zchp_now             = mpc.zchp,
            chp_elec_kw          = mpc.Pchp_kw,
            chp_heat_kw          = mpc.Qchp_kw,
            pgrid_kw             = mpc.Pgrid_kw,
            plan_SOC             = mpc.plan_SOC,
            plan_COP             = mpc.plan_COP,
            plan_Ppv             = mpc.plan_Ppv,
            plan_zchp            = mpc.plan_zchp,
            plan_chp_elec_kw     = mpc.plan_Pchp,
            plan_chp_heat_kw     = mpc.plan_Qchp,
            plan_pgrid_kw        = mpc.plan_Pgrid,
        )

    # ------------------------------------------------------------------
    # Legacy greedy LP (renamed -- kept as internal reference only)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_forecast(arr: np.ndarray, target_length: int) -> np.ndarray:
        """
        Ensure forecast has exactly `target_length` elements.
        Pads by repeating the last value, or clips if too long.
        """
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) >= target_length:
            return arr[:target_length]
        pad = np.full(target_length - len(arr), arr[-1] if len(arr) else 0.0)
        return np.concatenate([arr, pad])

    @staticmethod
    def outputs_to_dashboard_dict(outputs: SMPCOutputs, step: int = 0) -> dict:
        """
        Flatten SMPCOutputs into a flat key→value dict suitable for
        updating dashboard tag labels directly.

        *step* selects which horizon step to read (0 = current, >0 = future).
        """
        if step > 0:
            # Read from full-horizon plan arrays
            def _plan_val(arr, idx, decimals=1):
                if arr is not None and len(arr) > idx:
                    return round(float(arr[idx]), decimals)
                return 0.0

            def _asset_kw(uid):
                """Convert kWh/15-min schedule entry → kW."""
                sched = outputs.asset_schedules.get(uid)
                if sched is not None and len(sched) > step:
                    return round(float(sched[step]) * 4.0, 1)
                return 0.0

            ice_charge    = _plan_val(outputs.plan_ice_charge_kwh, step)
            ice_discharge = _plan_val(outputs.plan_ice_discharge_kwh, step)
            net_power     = _plan_val(outputs.plan_net_power_kwh, step)
            wkk_gas       = _plan_val(outputs.plan_wkk_gas_m3, step)

            cfg      = SMPCConfig()  # defaults for efficiency calcs
            wkk_elec = round(wkk_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency, 1)
            wkk_heat = round(wkk_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency, 1)

            heating_kw = _plan_val(outputs.plan_heating_kw, step)
            d = {
                "ice_bank_charge_kwh":     ice_charge,
                "ice_bank_discharge_kwh":  ice_discharge,
                "wkk_gas_setpoint_m3":     round(wkk_gas, 2),
                "net_power_kwh":           net_power,
                "wkk_elec_kwh":            wkk_elec,
                "wkk_heat_kwh":            wkk_heat,
                "heating_power_kw":        heating_kw,
                "smpc_cost_eur":           0.0,
                "baseline_cost_eur":       0.0,
                "cost_saving_eur":         0.0,
                "ice_bank_next_kwh":       0.0,
                "heat_buffer_next_kwh":    0.0,
                "season":                  outputs.season,
                "solver_used":             outputs.solver_used,
                "solve_time_ms":           round(outputs.solve_time_ms, 1),
                # ── MPC-specific fields (from asset schedules / plan arrays) ──
                "pv_used_kw":             _asset_kw("pv"),
                "gas_boiler_kw":          _asset_kw("gas_boiler"),
                "hw_heater_kw":           _asset_kw("hot_water_tank"),
                "battery_charge_kw":      _asset_kw("battery_charge"),
                "battery_discharge_kw":   _asset_kw("battery_discharge"),
                "cop_now":                _plan_val(outputs.plan_COP,              step, 2),
                "zchp_now":               _plan_val(outputs.plan_zchp,             step, 2),
                "hw_temp_c":              _plan_val(outputs.plan_hw_temp_c,        step, 1),
                "building_temp_c":        _plan_val(outputs.plan_building_temp_c,  step, 1),
                "building_setpoint_c":    round(outputs.building_setpoint_c, 1),
                "chp_elec_kw":            _plan_val(outputs.plan_chp_elec_kw,      step, 1),
                "chp_heat_kw":            _plan_val(outputs.plan_chp_heat_kw,      step, 1),
                "pgrid_kw":               _plan_val(outputs.plan_pgrid_kw,         step, 1),
            }
            for uid, sched in outputs.asset_schedules.items():
                kw = round(float(sched[step]) * 4.0, 1) if len(sched) > step else 0.0
                d[f"asset_{uid}_kw"] = kw
            return d

        d = {
            "ice_bank_charge_kwh":     round(outputs.ice_bank_charge_kwh,     1),
            "ice_bank_discharge_kwh":  round(outputs.ice_bank_discharge_kwh,  1),
            "wkk_gas_setpoint_m3":     round(outputs.wkk_gas_setpoint_m3,     2),
            "net_power_kwh":           round(outputs.net_power_kwh,           1),
            "wkk_elec_kwh":            round(outputs.wkk_elec_kwh,            1),
            "wkk_heat_kwh":            round(outputs.wkk_heat_kwh,            1),
            "heating_power_kw":        round(outputs.heating_power_kw,        1),
            "smpc_cost_eur":           round(outputs.smpc_cost_eur,           4),
            "baseline_cost_eur":       round(outputs.baseline_cost_eur,       4),
            "cost_saving_eur":         round(outputs.cost_saving_eur,         4),
            "ice_bank_next_kwh":       round(outputs.ice_bank_next_kwh,       1),
            "heat_buffer_next_kwh":    round(outputs.heat_buffer_next_kwh,    1),
            "season":                  outputs.season,
            "solver_used":             outputs.solver_used,
            "solve_time_ms":           round(outputs.solve_time_ms,           1),
            # ── New MPC fields ──────────────────────────────────────
            "pv_used_kw":              round(outputs.pv_used_kw,              1),
            "battery_soc_kwh":         round(outputs.battery_soc_kwh,         1),
            "battery_charge_kw":       round(outputs.battery_charge_kw,       1),
            "battery_discharge_kw":    round(outputs.battery_discharge_kw,    1),
            "gas_boiler_kw":           round(outputs.gas_boiler_kw,           1),
            "cop_now":                 round(outputs.cop_now,                 2),
            "zchp_now":                round(outputs.zchp_now,                2),
            "building_temp_c":         round(outputs.building_temp_c,         1),
            "building_setpoint_c":     round(outputs.building_setpoint_c,     1),
            "hw_temp_c":               round(outputs.hw_temp_c,               1),
            "hw_heater_kw":            round(outputs.hw_heater_kw,            1),
            "chp_elec_kw":             round(outputs.chp_elec_kw,             1),
            "chp_heat_kw":             round(outputs.chp_heat_kw,             1),
            "pgrid_kw":                round(outputs.pgrid_kw,                1),
        }
        for uid, kw in outputs.asset_power_kw.items():
            d[f"asset_{uid}_kw"] = round(kw, 1)
        return d


# ===========================================================================
# SECTION 7: SUBPROCESS ENTRY POINT
# Called by dashboard.py via multiprocessing.Pool to isolate the HIGHS
# MILP solver from Qt's DLL space (they conflict on Windows).
# Must be a module-level function so it can be pickled by multiprocessing.
# ===========================================================================

def _subprocess_solve_lp(
    config_path: str,
    price_forecast_eur_kwh: np.ndarray,
    base_load_kwh,
    solar_pred_kwh,
    month: int,
    building_temp_c,
    outside_temp_c,
    hw_temp_c,
) -> "SMPCOutputs":
    """Run solve_lp() in a subprocess (no Qt DLLs loaded)."""
    calc = SMPCCalculator(config_path)
    return calc.solve_lp(
        price_forecast_eur_kwh=price_forecast_eur_kwh,
        base_load_kwh=base_load_kwh,
        solar_pred_kwh=solar_pred_kwh,
        month=month,
        building_temp_c=building_temp_c,
        outside_temp_c=outside_temp_c,
        hw_temp_c=hw_temp_c,
    )


# ===========================================================================
# SECTION 8: QUICK SELF-TEST
# Runs when you execute this file directly: python smpc_calculator.py
# ===========================================================================

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  SMPC Calculator — self-test")
    print("=" * 55)

    config = SMPCConfig()
    calc   = SMPCCalculator(config)

    # Simulate a summer noon scenario (high price, buffer half full)
    H = config.horizon_steps
    t_of_day = np.linspace(0, 24, H)

    price_forecast = np.clip(
        0.08 + 0.04 * np.sin((t_of_day - 9) * np.pi / 12)
        + 0.02 * np.random.default_rng(1).normal(0, 1, H),
        0.02, None,
    )
    consumption_fc = np.clip(
        900 + 200 * np.sin((t_of_day - 7) * np.pi / 12)
        + np.random.default_rng(2).normal(0, 30, H),
        200, None,
    )

    test_inputs = SMPCInputs(
        electricity_price_eur_kwh = float(price_forecast[0]),
        price_forecast_eur_kwh    = price_forecast,
        consumption_kwh           = float(consumption_fc[0]),
        consumption_forecast_kwh  = consumption_fc,
        ice_bank_kwh              = 3_500.0,
        heat_buffer_kwh           = 5_000.0,
        heat_demand_forecast_kwh  = np.zeros(H),
        wkk_max_gas_m3            = 6.0,
        month                     = 7,   # July → summer
    )

    t0 = time.perf_counter()
    result = calc.solve(test_inputs)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Season          : {result.season}")
    print(f"  Solver          : {result.solver_used} ({result.solver_status})")
    print(f"  Solve time      : {elapsed:.1f} ms")
    print(f"\n  --- First-step commands ---")
    print(f"  Ice charge      : {result.ice_bank_charge_kwh:.1f} kWh")
    print(f"  Ice discharge   : {result.ice_bank_discharge_kwh:.1f} kWh")
    print(f"  WKK gas         : {result.wkk_gas_setpoint_m3:.2f} m³")
    print(f"\n  --- KPIs ---")
    print(f"  Net grid draw   : {result.net_power_kwh:.1f} kWh")
    print(f"  WKK electricity : {result.wkk_elec_kwh:.1f} kWh")
    print(f"  SMPC cost       : €{result.smpc_cost_eur:.4f}")
    print(f"  Baseline cost   : €{result.baseline_cost_eur:.4f}")
    print(f"  Saving          : €{result.cost_saving_eur:.4f}")
    print(f"\n  --- Next states ---")
    print(f"  Ice bank        : {result.ice_bank_next_kwh:.0f} kWh")
    print(f"  Heat buffer     : {result.heat_buffer_next_kwh:.0f} kWh")

    print(f"\n  Dashboard dict  :")
    kpis = SMPCCalculator.outputs_to_dashboard_dict(result)
    for k, v in kpis.items():
        print(f"    {k:<28} {v}")

    print("\n  Self-test complete.")
