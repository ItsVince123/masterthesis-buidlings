import mpc_lp, numpy as np

cfg = mpc_lp.MPCConfig()
cfg.hp_enabled = True
cfg.boiler_enabled = True
cfg.Php_max_kw = 300.0
cfg.Pgas_max_kw = 300.0
cfg.Tset_c = 21.0
cfg.Tmin_c = 20.5
cfg.Tmax_c = 22.0

H = 96
np.random.seed(42)
prices = np.clip(0.15 + 0.06*np.sin(np.linspace(0, 4*3.14159, H)) + 0.02*np.random.randn(H), 0.03, 0.30)

COP = max(cfg.COP0, cfg.COP_min)
c_boiler = cfg.gas_price_boiler_eur_m3 / (cfg.gas_HV_kwh_m3 * cfg.eta_boiler)
breakeven = COP * c_boiler
cheap_steps = int((prices < breakeven).sum())
print(f"Price min/max/mean: {prices.min():.3f} / {prices.max():.3f} / {prices.mean():.3f} EUR/kWh")
print(f"c_boiler:           {c_boiler:.4f} EUR/kWh_th")
print(f"HP breakeven price: {breakeven:.4f} EUR/kWh  (HP cheaper when price < this)")
print(f"Steps where HP < boiler cost: {cheap_steps} of {H}")
print()

inp = mpc_lp.MPCInputs(
    price_eur_kwh=prices,
    Pload_kw=np.full(H, 50.0),
    Ppv_forecast_kw=np.zeros(H),
    Tamb_c=np.full(H, 3.0),
)
sol = mpc_lp.solve_mpc(inp, cfg)
bl  = mpc_lp.compute_baseline_arrays(inp, cfg)
sav = mpc_lp.compute_asset_savings(sol, bl, inp, cfg)

th = sav["thermal_building_eur"]
fs = sav["fuel_switching_eur"]
ts = sav["thermal_storage_eur"]
print(f"thermal_building_eur:  {th:+.4f} EUR")
print(f"  fuel_switching_eur:  {fs:+.4f} EUR  ({100*fs/max(abs(th),1e-9):.1f}% of thermal total)")
print(f"  thermal_storage_eur: {ts:+.4f} EUR  ({100*ts/max(abs(th),1e-9):.1f}% of thermal total)")
print(f"  sum check:           {fs+ts:+.4f} EUR  (should == {th:+.4f})")

# Show baseline gas vs HP heat
bl_arrays = bl
bl_Php_total  = float(np.sum(bl_arrays["Php"])) * cfg.dt_hours
bl_Pgas_total = float(np.sum(bl_arrays["Pgas"])) * cfg.dt_hours
print()
print(f"Baseline heat (HP):    {bl_Php_total:.1f} kWh_th")
print(f"Baseline heat (boiler):{bl_Pgas_total:.1f} kWh_th")
