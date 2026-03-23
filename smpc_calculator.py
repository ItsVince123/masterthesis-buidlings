"""
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

    # --- Grid connection ----------------------------------------------------
    peak_limit_kwh:             float = 3_250.0   # Per 15-min step = 13 MW peak
    peak_tariff_eur_kw_month:   float =     8.0   # Monthly capacity tariff

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

    # --- Solver metadata ----------------------------------------------------
    season:        str = "unknown"   # "summer" / "winter" / "transition"
    solver_used:   str = "none"      # "clarabel" / "scs" / "heuristic"
    solver_status: str = "unknown"
    solve_time_ms: float = 0.0


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

    Uses a simple AR(1) log-normal model.  In production, replace with
    ENTSO-E DAM distributions or an ARIMA/GARCH fit on historical EPEX data.

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

def _optimise_summer_cvxpy(
    mean_price: np.ndarray,
    consumption: np.ndarray,
    ice_bank_kwh: float,
    cfg: SMPCConfig,
) -> dict:
    """
    Convex QP for summer ice-bank optimisation.

    Minimises:
        J = w_energy  × Σ mean_price × P_net
          + w_peak    × Σ slack²              (soft peak constraint)
          + w_buffer  × (buffer_end − target)²
    """
    H = len(mean_price)
    u_charge    = cp.Variable(H, nonneg=True)
    u_discharge = cp.Variable(H, nonneg=True)
    slack       = cp.Variable(H, nonneg=True)   # Peak limit softener

    # State evolution (list of scalars / affine expressions)
    buffer = [ice_bank_kwh]
    for t in range(H):
        buffer.append(
            buffer[t] + u_charge[t] * cfg.ice_charge_efficiency - u_discharge[t]
        )

    p_net  = consumption + u_charge - u_discharge
    target = cfg.ice_bank_capacity_kwh * cfg.ice_bank_target_soc

    objective = (
        cfg.w_energy    * cp.sum(cp.multiply(mean_price, p_net))
        + cfg.w_peak    * cp.sum_squares(slack)
        + cfg.w_buffer_end * cp.square(buffer[-1] - target)
    )

    constraints = [
        u_charge    <= cfg.ice_charge_max_kwh,
        u_discharge <= cfg.ice_discharge_max_kwh,
        cp.vstack(buffer[1:]) >= cfg.ice_bank_min_kwh,
        cp.vstack(buffer[1:]) <= cfg.ice_bank_capacity_kwh,
        p_net <= cfg.peak_limit_kwh + slack,
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
    Convex QP for winter WKK optimisation.

    Minimises:
        J = Σ gas_cost  −  Σ electricity_revenue  +  w_comfort × Σ heat_shortfall²

    WKK runs when spark spread (electricity revenue − gas cost per m³) > threshold.
    Heat buffer absorbs excess heat; prevents WKK from over-running.
    """
    H = len(mean_price)
    u_gas       = cp.Variable(H, nonneg=True)
    heat_slack  = cp.Variable(H, nonneg=True)   # Unmet heat demand (comfort penalty)

    heat_prod = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_heat_efficiency
    elec_prod = u_gas * cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency

    buffer = [heat_buffer_kwh]
    for t in range(H):
        buffer.append(buffer[t] + heat_prod[t] - heat_demand[t] + heat_slack[t])

    gas_cost         = cp.sum(u_gas) * cfg.gas_price_eur_m3
    electricity_rev  = cp.sum(cp.multiply(mean_price, elec_prod))
    comfort_penalty  = cfg.w_heat_comfort * cp.sum_squares(heat_slack)

    objective = gas_cost - electricity_rev + comfort_penalty

    constraints = [
        u_gas <= wkk_max_gas,
        cp.vstack(buffer[1:]) >= 0,
        cp.vstack(buffer[1:]) <= cfg.heat_buffer_capacity_kwh,
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

        # Spark spread: revenue from 1 m³ gas minus the gas cost
        electricity_rev_per_m3 = cfg.gas_energy_kwh_m3 * cfg.wkk_elec_efficiency * price
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

    def __init__(self, config: SMPCConfig = None):
        if config is not None:
            self.cfg = config
        else:
            # Auto-load from dashboard_config.json sitting next to this file.
            # Falls back to hardcoded defaults if the file is missing.
            try:
                from dashboard_config import load_smpc_config
                self.cfg = load_smpc_config()
            except Exception:
                self.cfg = SMPCConfig()

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
        scenarios = _generate_price_scenarios(
            base_forecast  = price_fc,
            n_scenarios    = cfg.n_scenarios,
            volatility     = cfg.price_volatility,
            autocorrelation= cfg.autocorrelation,
            seed           = inputs.month,   # Reproducible per month
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

        smpc_elec_cost = net_power * inputs.electricity_price_eur_kwh
        smpc_gas_cost  = u_gas     * cfg.gas_price_eur_m3
        smpc_cost      = smpc_elec_cost + smpc_gas_cost

        if inputs.baseline_cost_eur is not None:
            baseline_cost = inputs.baseline_cost_eur
        else:
            # Estimate: building draws full consumption from grid, no optimisation
            baseline_cost = inputs.consumption_kwh * inputs.electricity_price_eur_kwh

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
    def outputs_to_dashboard_dict(outputs: SMPCOutputs) -> dict:
        """
        Flatten SMPCOutputs into a flat key→value dict suitable for
        updating dashboard tag labels directly.

        Example usage in dashboard.py:
            kpis = SMPCCalculator.outputs_to_dashboard_dict(outputs)
            self.value_labels[("output", "ice_charge")].setText(
                f"{kpis['ice_bank_charge_kwh']:.1f} kWh"
            )
        """
        return {
            "ice_bank_charge_kwh":     round(outputs.ice_bank_charge_kwh,     1),
            "ice_bank_discharge_kwh":  round(outputs.ice_bank_discharge_kwh,  1),
            "wkk_gas_setpoint_m3":     round(outputs.wkk_gas_setpoint_m3,     2),
            "net_power_kwh":           round(outputs.net_power_kwh,           1),
            "wkk_elec_kwh":            round(outputs.wkk_elec_kwh,            1),
            "wkk_heat_kwh":            round(outputs.wkk_heat_kwh,            1),
            "smpc_cost_eur":           round(outputs.smpc_cost_eur,           4),
            "baseline_cost_eur":       round(outputs.baseline_cost_eur,       4),
            "cost_saving_eur":         round(outputs.cost_saving_eur,         4),
            "ice_bank_next_kwh":       round(outputs.ice_bank_next_kwh,       1),
            "heat_buffer_next_kwh":    round(outputs.heat_buffer_next_kwh,    1),
            "season":                  outputs.season,
            "solver":                  outputs.solver_used,
            "solve_time_ms":           round(outputs.solve_time_ms,           1),
        }


# ===========================================================================
# SECTION 7: QUICK SELF-TEST
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
