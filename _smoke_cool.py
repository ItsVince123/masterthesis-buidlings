"""Cooling smoke test: building starts hot, CHP profitable, Tmax=24 C."""
import numpy as np
from mpc_lp import MPCConfig, MPCInputs, solve_mpc

H = 24
cfg = MPCConfig()
cfg.horizon_steps = H
cfg.dt_hours = 1.0
cfg.hp_enabled = True
cfg.cooling_enabled = True
cfg.Php_max_kw = 30.0
cfg.Php_cool_max_kw = 50.0
cfg.COP_cool = 3.0
cfg.boiler_enabled = False
cfg.chp_enabled = True
cfg.chp_use_milp = False
cfg.bat_enabled = False
cfg.pv_enabled = False
cfg.hw_enabled = False
cfg.flex_enabled = False
cfg.UA_kw_per_c = 1.5
cfg.Cth_kwh_per_c = 8.0
cfg.Tmin_c = 20.0
cfg.Tmax_c = 24.0

inputs = MPCInputs(
    price_eur_kwh=np.full(H, 0.30),
    Pload_kw=np.full(H, 10.0),
    Ppv_forecast_kw=np.zeros(H),
    Tamb_c=np.full(H, 28.0),
    T_building_init_c=25.0,
    T_tank_init_c=60.0,
    SOC_init_kwh=0.0,
)

for dump_enabled in (True, False):
    cfg.chp_heat_dump_enabled = dump_enabled
    out = solve_mpc(inputs, cfg)
    if out is None:
        print(f"dump={dump_enabled}: SOLVER FAILED")
        continue
    # MPCOutputs has trajectory arrays as attributes
    Tbld = np.asarray(out.plan_Tbuilding)
    Pcool = np.asarray(out.plan_Php_cool)
    Pchp = np.asarray(getattr(out, 'plan_Pchp', np.zeros(H)))
    Qchp = np.asarray(getattr(out, 'plan_Qchp', np.zeros(H)))
    print(
        f"dump={dump_enabled}: "
        f"Tbld_max={Tbld.max():.3f} Tbld_end={Tbld[-1]:.3f} "
        f"Pcool_mean={Pcool.mean():.2f} "
        f"Pchp_mean={Pchp.mean():.2f} "
        f"Qchp_mean={Qchp.mean():.2f}"
    )
