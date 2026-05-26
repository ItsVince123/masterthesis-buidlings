"""
Analysis day batch solver — executed as a subprocess by analysis_dialog.py.

Reads pickled args from stdin, solves all days, writes pickled results to
stdout.  This script intentionally imports NOTHING from PyQt6 so that
HIGHS DLLs and Qt DLLs never coexist in the same process.

For multi-day analyses (full year / range), days are solved in parallel
across CPU cores using ``multiprocessing.Pool`` — each child process is
fully independent (own HIGHS DLL load, own BLAS threads pinned to 1).

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

# Must be set before cvxpy / HIGHS loads any DLL — applies to this process
# AND is inherited by spawned worker children (initializer re-asserts it).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HIGHS_NUM_THREADS", "1")

# Keep subprocess quiet — warnings still reach stderr so the parent can log them
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


# ────────────────────────────────────────────────────────────────────
# Per-day solve — must be top-level so multiprocessing can pickle it
# ────────────────────────────────────────────────────────────────────
def _solve_one_day(args: tuple) -> dict:
    """Solve a single day's MPC + baseline; return the result dict.

    `args` is a 2-tuple of (day_dict, cfg) so it can be mapped directly.
    """
    import numpy as np

    from mpc_lp import (
        MPCInputs,
        build_pload_vector,
        compute_asset_savings,
        compute_baseline_arrays,
        solve_mpc,
    )

    day, cfg = args
    date_str: str = day.get("date", "?")
    try:
        H  = cfg.horizon_steps
        dt = cfg.dt_hours

        price = np.asarray(day["price"],       dtype=float)[:H]
        ppv   = np.asarray(day["ppv"],         dtype=float)[:H]
        temps = np.asarray(day["temperature"], dtype=float)[:H]

        if len(price) < H:
            price = np.pad(price, (0, H - len(price)), "edge")
        if len(ppv) < H:
            ppv = np.pad(ppv, (0, H - len(ppv)), "edge")
        if len(temps) < H:
            temps = np.pad(temps, (0, H - len(temps)), "edge")

        inp = MPCInputs(
            price_eur_kwh     = price,
            Pload_kw          = build_pload_vector(cfg, 0.0, H),
            Ppv_forecast_kw   = ppv,
            Tamb_c            = temps,
            SOC_init_kwh      = cfg.SOC_init_kwh,
            T_building_init_c = cfg.T_init_c,
            T_tank_init_c     = cfg.hw_T_init_c,
        )

        # Derive weekday from the simulated date so flex day-of-week gating
        # uses the historical day, not "today" (Mon=0 … Sun=6).
        _start_wd = None
        try:
            from datetime import datetime as _dtcls
            _start_wd = _dtcls.strptime(date_str, "%Y-%m-%d").weekday()
        except Exception:
            _start_wd = None

        sol = solve_mpc(inp, cfg, start_weekday=_start_wd)
        bl  = compute_baseline_arrays(inp, cfg, start_weekday=_start_wd)
        sav = compute_asset_savings(sol, bl, inp, cfg)

        return {
            "ok":                True,
            "date":              date_str,
            "baseline_cost_eur": bl["total_cost"],
            "mpc_cost_eur":      sol.mpc_cost_eur,
            "total_saving_eur":  bl["total_cost"] - sol.mpc_cost_eur,
            # Gross baseline (imports+gas, no PV export credit) + the export
            # revenue itself — for the dashboard's "% of gross bill" KPI.
            "baseline_gross_eur":          bl.get("gross_cost", bl["total_cost"]),
            "baseline_export_revenue_eur": bl.get("export_revenue", 0.0),
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
            "plan_Pchp":   sol.plan_Pchp.tolist() if len(sol.plan_Pchp) > 0 else [],
            "plan_Qchp":   sol.plan_Qchp.tolist() if len(sol.plan_Qchp) > 0 else [],
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
        }

    except Exception as exc:
        return {
            "ok":    False,
            "date":  date_str,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _pool_initializer():
    """Re-assert thread-limiting env vars inside each worker process."""
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["HIGHS_NUM_THREADS"]    = "1"


if __name__ == "__main__":
    payload = pickle.loads(sys.stdin.buffer.read())
    days: list[dict] = payload["days"]
    cfg = payload["cfg"]

    tasks = [(day, cfg) for day in days]

    # For a single day, parallelism overhead isn't worth it.
    if len(tasks) <= 1:
        results = [_solve_one_day(t) for t in tasks]
    else:
        import multiprocessing as _mp
        # Leave one core for the OS; cap at 8 to limit HIGHS DLL memory churn.
        n_workers = min(8, max(1, (os.cpu_count() or 2) - 1), len(tasks))
        try:
            ctx = _mp.get_context("spawn")
            with ctx.Pool(processes=n_workers, initializer=_pool_initializer) as pool:
                # chunksize=1 → finer-grained load balancing across uneven days
                results = pool.map(_solve_one_day, tasks, chunksize=1)
        except Exception as exc:
            # Fall back to sequential on any pool error
            logging.warning("Parallel pool failed (%s); falling back to sequential", exc)
            results = [_solve_one_day(t) for t in tasks]

    sys.stdout.buffer.write(pickle.dumps(results))
