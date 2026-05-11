"""
LP solver worker — executed as a subprocess by dashboard.py.

Reads pickled args from stdin, runs the MPC solve, writes pickled result
to stdout.  This script intentionally imports NOTHING from PyQt6 so that
HIGHS DLLs and Qt DLLs never coexist in the same process.

Called by _AsyncSubprocessResult in dashboard.py:
    python _lp_worker.py  (stdin=pickle(args), stdout=pickle(result))
"""
import os
import pickle
import sys

# Must be set before cvxpy / HIGHS loads any DLL
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HIGHS_NUM_THREADS", "1")

if __name__ == "__main__":
    from smpc_calculator import _subprocess_solve_lp

    args = pickle.loads(sys.stdin.buffer.read())
    result = _subprocess_solve_lp(*args)
    sys.stdout.buffer.write(pickle.dumps(result))
