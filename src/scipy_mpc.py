import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)

from well_model         import WellModel
from constraint_manager import ConstraintManager
from brute_force_mpc    import BruteForceMPC


class ScipyMPC:

    def __init__(self, model, constraints, horizon=10):
        self.model      = model
        self.cm         = constraints
        self.horizon    = horizon
        self.fallback   = BruteForceMPC(model, constraints, horizon)

        # tuning weights
        self.w_Q  = 100.0   # Q tracking error
        self.w_du = 2.0     # move suppression
        self.w_u  = 0.5     # deviation from steady-state target

        self.current_choke  = 0.0
        self.prev_solution  = None

        # cache feasibility result — computed once, reused every step
        self.max_feasible_Q      = None
        self.limiting_constraint = None

        self.stats = {
            "scipy_calls":   0,
            "scipy_success": 0,
            "scipy_failed":  0,
            "fallback_used": 0,
            "total_iters":   0,
            "solve_times_ms": [],
        }

    def _cost(self, U, state, target_Q, prev_choke):
        ss_choke = self.model.invert_for_choke(target_Q)
        traj = self.model.predict_trajectory(state, U.tolist(), self.horizon)

        cost   = 0.0
        u_prev = prev_choke
        for k in range(self.horizon):
            tw  = 0.95 ** k
            q_k = traj["Q"][k]
            u_k = U[k]

            cost += self.w_Q  * tw * (q_k - target_Q) ** 2
            cost += self.w_du * (u_k - u_prev) ** 2
            cost += self.w_u  * (u_k - ss_choke) ** 2
            u_prev = u_k

        return cost

    def _build_constraints(self, state, prev_choke):
        lim = self.cm.limits

        # cache trajectory so all constraints share a single call per U evaluation
        _cache = {"U_key": None, "traj": None}

        def get_traj(U):
            key = U.tobytes()
            if _cache["U_key"] != key:
                _cache["traj"]  = self.model.predict_trajectory(state, U.tolist(), self.horizon)
                _cache["U_key"] = key
            return _cache["traj"]

        cons = []
        for k in range(self.horizon):
            def make_cons(k_):
                def bhp_min(U):
                    return get_traj(U)["BHP"][k_] - lim["BHP_min"]
                def whp_max(U):
                    return lim["WHP_max"] - get_traj(U)["WHP"][k_]
                def whp_min(U):
                    return get_traj(U)["WHP"][k_] - lim["WHP_min"]
                def flp_max(U):
                    return lim["FLP_max"] - get_traj(U)["FLP"][k_]
                def ramp_up(U):
                    u_prev_ = U[k_ - 1] if k_ > 0 else prev_choke
                    return lim["delta_choke_max"] - (U[k_] - u_prev_)
                def ramp_dn(U):
                    u_prev_ = U[k_ - 1] if k_ > 0 else prev_choke
                    return lim["delta_choke_max"] + (U[k_] - u_prev_)
                return [bhp_min, whp_max, whp_min, flp_max, ramp_up, ramp_dn]

            for fn in make_cons(k):
                cons.append({"type": "ineq", "fun": fn})

        return cons

    def _initial_guess(self, prev_choke, target_Q):
        ss_choke = self.model.invert_for_choke(target_Q)

        # target-directed ramp
        directed = [prev_choke]
        cur = prev_choke
        for _ in range(self.horizon - 1):
            move = float(np.clip(ss_choke - cur, -5.0, 5.0))
            cur  = float(np.clip(cur + move, 0.0, 100.0))
            directed.append(cur)
        directed = np.array(directed)

        if self.prev_solution is not None:
            # warm start: shift left, repeat last value
            warm = np.append(self.prev_solution[1:], self.prev_solution[-1])
            return 0.7 * warm + 0.3 * directed

        return directed

    def _build_sequence(self, u0, prev_choke, target_Q):
        ss_choke = self.model.invert_for_choke(target_Q)
        seq = [float(np.clip(u0, 0.0, 100.0))]
        cur = seq[0]
        for _ in range(self.horizon - 1):
            move = float(np.clip(ss_choke - cur, -5.0, 5.0))
            cur  = float(np.clip(cur + move, 0.0, 100.0))
            seq.append(cur)
        return seq

    def _solve(self, state, target_Q, prev_choke):
        bounds  = Bounds(lb=0.0, ub=100.0)
        cons    = self._build_constraints(state, prev_choke)
        U0      = self._initial_guess(prev_choke, target_Q)

        try:
            res = minimize(
                self._cost,
                U0,
                args=(state, target_Q, prev_choke),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 50, "ftol": 1e-4, "disp": False},
            )
            u_first = float(np.clip(res.x[0], 0.0, 100.0))
            return u_first, res.success, res.nit, res.message, res.x
        except Exception as e:
            return prev_choke, False, 0, str(e), None

    def get_feasible_target(self, target_Q):
        # compute once and cache — model doesn't change between calls
        if self.max_feasible_Q is None:
            self.max_feasible_Q, _, self.limiting_constraint = \
                self.cm.compute_max_safe_production(self.model)

        if target_Q > self.max_feasible_Q:
            return self.max_feasible_Q, True, self.limiting_constraint
        return target_Q, False, None

    def _get_status(self, pred_Q, target_Q, solver_ok, clamped, limit_str):
        if clamped:
            return "CONSTRAINED_MAX", limit_str

        if not solver_ok:
            return "FALLBACK_ACTIVE", None

        rel_err = abs(pred_Q - target_Q) / max(abs(target_Q), 1e-6)
        if rel_err < 0.03:
            return "TRACKING", None
        if pred_Q < target_Q:
            return "CONVERGING_UP", None
        return "CONVERGING_DOWN", None

    def reset(self, choke=0.0):
        self.current_choke = choke
        self.prev_solution = None

    def compute(self, state, target_Q):
        t0         = time.perf_counter()
        prev_choke = self.current_choke
        self.stats["scipy_calls"] += 1

        # clamp to feasible range before doing anything else
        working_target, clamped, limit_str = self.get_feasible_target(target_Q)

        u_first, success, n_iters, msg, sol_array = self._solve(state, working_target, prev_choke)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats["solve_times_ms"].append(elapsed_ms)
        self.stats["total_iters"] += n_iters

        solver_ok = False
        method    = "BRUTE_FORCE"
        choke_cmd = prev_choke
        fallback_reason = ""

        if success:
            seq  = self._build_sequence(u_first, prev_choke, working_target)
            traj = self.model.predict_trajectory(state, seq, self.horizon)
            traj_safe, viol = self.cm.is_trajectory_safe(traj)

            if traj_safe:
                self.stats["scipy_success"] += 1
                solver_ok        = True
                method           = "SCIPY_SLSQP"
                choke_cmd        = u_first
                self.prev_solution = sol_array
            else:
                self.stats["scipy_failed"] += 1
                fallback_reason = f"post-validation failed: {viol}"
        else:
            self.stats["scipy_failed"] += 1
            fallback_reason = msg

        # use brute-force fallback if scipy didn't pan out
        if not solver_ok:
            self.stats["fallback_used"] += 1
            self.prev_solution = None
            self.fallback.current_choke = prev_choke
            fb_result  = self.fallback.compute(state, working_target)
            choke_cmd  = fb_result["choke_command"]
            method     = f"BRUTE_FORCE ({fallback_reason[:40]})" if fallback_reason else "BRUTE_FORCE"

        # final ramp-rate enforcement
        move      = float(np.clip(choke_cmd - prev_choke, -5.0, 5.0))
        choke_cmd = float(np.clip(prev_choke + move, 0.0, 100.0))
        self.current_choke = choke_cmd

        # one-step-ahead prediction for reporting
        one_step = self.model.predict_trajectory(state, [choke_cmd], 1)
        pred_Q   = float(one_step["Q"][0])

        status, active_limit = self._get_status(pred_Q, working_target, solver_ok, clamped, limit_str)

        return {
            "choke_command":    round(choke_cmd, 2),
            "status":           status,
            "method":           method,
            "predicted_Q":      round(pred_Q, 2),
            "solver_iters":     n_iters,
            "solve_time_ms":    round(elapsed_ms, 1),
            "active_limit":     active_limit,
            "choke_delta":      round(choke_cmd - prev_choke, 2),
            "requested_target": target_Q,
            "working_target":   round(working_target, 2),
            "is_clamped":       clamped,
        }

    def report(self):
        s = self.stats
        total = s["scipy_calls"]
        pct   = 100 * s["scipy_success"] / total if total > 0 else 0.0
        avg_t = np.mean(s["solve_times_ms"]) if s["solve_times_ms"] else 0.0
        avg_i = s["total_iters"] / total if total > 0 else 0.0

        print("\n--- ScipyMPC Report ---")
        print(f"  Total MPC calls    : {total}")
        print(f"  Scipy success      : {s['scipy_success']}  ({pct:.1f}%)")
        print(f"  Scipy failed       : {s['scipy_failed']}")
        print(f"  Fallback used      : {s['fallback_used']}")
        print(f"  Avg solve time     : {avg_t:.1f} ms")
        print(f"  Avg iterations     : {avg_i:.1f}")


if __name__ == "__main__":
    from oil_well_simulator import OilWellSimulator

    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))
    model = WellModel()
    model.fit(df_sweep, verbose=False)

    cm  = ConstraintManager()
    mpc = ScipyMPC(model, cm, horizon=10)
    sim = OilWellSimulator()
    sim.reset()

    target_Q      = 60.0
    n_steps       = 25
    any_violation = False
    choke         = 0.0

    print(f"\n{'t':>4}  {'target':>7}  {'Q':>7}  {'choke':>7}  {'WHP':>7}  "
          f"{'FLP':>7}  {'BHP':>7}  {'method':<16}  status")
    print("-" * 95)

    for t in range(n_steps):
        q, whp, flp, bhp = sim.step(choke)

        state  = {"Q": q, "WHP": whp, "FLP": flp, "BHP": bhp, "choke": choke}
        result = mpc.compute(state, target_Q)
        choke  = result["choke_command"]

        safe, reason = cm.is_safe({"WHP": whp, "FLP": flp, "BHP": bhp})
        if not safe:
            any_violation = True
            disp_status = f"VIOLATION: {reason}"
        else:
            disp_status = result["status"]

        method_short = result["method"][:16]
        print(f"{t:>4}  {target_Q:>7.1f}  {q:>7.2f}  {choke:>7.2f}  {whp:>7.1f}  "
              f"{flp:>7.1f}  {bhp:>7.1f}  {method_short:<16}  {disp_status}")

    print()
    if any_violation:
        print("RESULT: Constraint violation(s) occurred.")
    else:
        print("RESULT: No constraint violations — all steps safe.")

    mpc.report()
