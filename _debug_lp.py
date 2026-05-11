"""Quick diagnostic: trace LP → tag rendering pipeline."""
import sys, time, logging
logging.basicConfig(level=logging.WARNING)
from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)
from dashboard import ScadaWindow

w = ScadaWindow()
w.timer.stop()
w.show()

# Tick 1 — submits LP to subprocess
w._on_tick()
print(f"After tick1: async_result={w._lp_async_result}  last_outputs={w.last_lp_outputs is not None}")

# Poll until subprocess result is ready (max 10 s)
for i in range(100):
    time.sleep(0.1)
    if w._lp_async_result is not None and w._lp_async_result.ready():
        print(f"Result ready after {(i+1)*100} ms")
        break
else:
    print("TIMEOUT — subprocess never finished!")
    # Try to get the exception
    if w._lp_async_result is not None:
        try:
            w._lp_async_result.get(timeout=2)
        except Exception as e:
            print("  pool exception:", e)
    app.quit()
    sys.exit(1)

# Collect the result
w._collect_lp_result()
print(f"After collect: last_outputs={w.last_lp_outputs is not None}")

if w.last_lp_outputs is not None:
    out = w.last_lp_outputs
    print(f"  solver_status={out.solver_status}  solver_used={out.solver_used}")
    print(f"  asset_schedules keys={list(out.asset_schedules.keys())}")
    print(f"  plan_net_power_kwh len={len(out.plan_net_power_kwh)}")
    print(f"  _lp_now_idx={w._lp_now_idx}")
    print()
    print("Tag rendering:")
    for ttype in ("input", "output"):
        for tid, defn in w.tag_definitions[ttype].items():
            txt, col = w._sim_value(defn)
            sim = defn.get("simulation", {})
            mode = sim.get("mode", "?")
            field = sim.get("field", "")
            ok = txt not in ("--", "") and not txt.startswith("--")
            flag = "OK" if ok else "!!"
            print(f"  [{flag}][{ttype}] {tid}: mode={mode} field={field} => {txt!r}")
else:
    print("  last_lp_outputs is STILL None!")

app.quit()
