"""
Analysis day batch solver — executed as a subprocess by analysis_dialog.py.

Reads pickled args from stdin, solves all days, writes pickled results to
stdout.  This script intentionally imports NOTHING from PyQt6 so that
HIGHS DLLs and Qt DLLs never coexist in the same process.

Input  (stdin):  pickle of dict {
    "days":    list of {date, price, ppv, temperature, pload_kw},
    "cfg":     MPCConfig dataclass,
}
Output (stdout): pickle of list of result dicts (one per day).
"""
from __future__ import annotations

import logging
import os
import pickle
import sys

# Must be set before cvxpy / HIGHS loads any DLL
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HIGHS_NUM_THREADS", "1")

# Keep subprocess quiet — warnings still reach stderr so the parent can log them
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

if __name__ == "__main__":
    import numpy as np

    from mpc_lp import (
        MPCInputs,
        compute_asset_savings,
        compute_baseline_arrays,
        solve_mpc,
    )

    payload = pickle.loads(sys.stdin.buffer.read())
    days: list[dict] = payload["days"]
    cfg = payload["cfg"]

    results: list[dict] = []
    for day in days:
        date_str: str = day.get("date", "?")
        try:
            H   = cfg.horizon_steps
            dt  = cfg.dt_hours

            price = np.asarray(day["price"],       dtype=float)[:H]
            ppv   = np.asarray(day["ppv"],         dtype=float)[:H]
            temps = np.asarray(day["temperature"], dtype=float)[:H]

            # Pad if shorter than H (shouldn't happen, but defensive)
            if len(price) < H:
                price = np.pad(price, (0, H - len(price)), "edge")
            if len(ppv) < H:
                ppv = np.pad(ppv, (0, H - len(ppv)), "edge")
            if len(temps) < H:
                temps = np.pad(temps, (0, H - len(temps)), "edge")

            inp = MPCInputs(
                price_eur_kwh     = price,
                Pload_kw          = np.full(H, float(day["pload_kw"])),
                Ppv_forecast_kw   = ppv,
                Tamb_c            = temps,
                SOC_init_kwh      = cfg.SOC_init_kwh,
                T_building_init_c = cfg.T_init_c,
                T_tank_init_c     = cfg.hw_T_init_c,
            )

            sol = solve_mpc(inp, cfg)
            bl  = compute_baseline_arrays(inp, cfg)
            sav = compute_asset_savings(sol, bl, inp, cfg)

            results.append({
                "ok":                True,
                "date":              date_str,
                "baseline_cost_eur": bl["total_cost"],
                "mpc_cost_eur":      sol.mpc_cost_eur,
                "total_saving_eur":  bl["total_cost"] - sol.mpc_cost_eur,
                "savings":           sav,
                "dt":                dt,
                # MPC dispatch arrays
                "plan_Pgrid":  sol.plan_Pgrid.tolist(),
                "plan_Php":    sol.plan_Php.tolist(),
                "plan_Ppv":    sol.plan_Ppv.tolist(),
                "plan_Pflex":  sol.plan_Pflex.tolist(),
                "plan_Pch":    sol.plan_Pch.tolist(),
                "plan_Pdis":   sol.plan_Pdis.tolist(),
                "plan_Pgas":   sol.plan_Pgas.tolist(),
                "plan_Ptank":  sol.plan_Ptank.tolist(),
                "plan_Php_cool": sol.plan_Php_cool.tolist() if len(sol.plan_Php_cool) > 0 else [],
                "plan_Tbuilding": sol.plan_Tbuilding[:H].tolist() if len(sol.plan_Tbuilding) > 0 else [],
                # Baseline dispatch arrays
                "bl_Pgrid":    bl["Pgrid"].tolist(),
                "bl_Php":      bl["Php"].tolist(),
                "bl_Ppv":      bl["Ppv"].tolist(),
                "bl_Pflex":    bl["Pflex"].tolist(),
                "bl_Ptank":    bl["Ptank"].tolist(),
                "prices":      price.tolist(),
                "temperature": temps.tolist(),
            })

        except Exception as exc:
            results.append({
                "ok":    False,
                "date":  date_str,
                "error": f"{type(exc).__name__}: {exc}",
            })

    sys.stdout.buffer.write(pickle.dumps(results))
