import os
import sys
import time
import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)

from well_model        import WellModel
from constraint_manager import ConstraintManager


class BruteForceMPC:

    def __init__(self, model, constraint_manager, horizon=10):
        self.model   = model
        self.cm      = constraint_manager
        self.horizon = horizon
        self.current_choke = 0.0

    def build_sequence(self, u0, target_choke):
        seq = [u0]
        cur = u0
        for _ in range(self.horizon - 1):
            move = np.clip(target_choke - cur, -5.0, 5.0)
            cur  = float(np.clip(cur + move, 0.0, 100.0))
            seq.append(cur)
        return seq

    def score_trajectory(self, trajectory, target_Q):
        score = 0.0
        for k in range(self.horizon):
            w      = 0.95 ** k          # discount near-term errors more heavily
            q_pred = trajectory["Q"][k]
            score -= w * (q_pred - target_Q) ** 2
        return score

    def reset(self, choke=0.0):
        self.current_choke = choke

    def compute(self, state, target_Q):
        t0 = time.time()
        current_choke = float(state.get("choke", self.current_choke))
        candidates    = self.cm.get_choke_candidates(current_choke)
        target_choke  = self.model.invert_for_choke(target_Q)

        best_score     = -np.inf
        best_candidate = None
        n_evaluated    = 0
        n_safe         = 0

        for u_cand in candidates:
            seq  = self.build_sequence(u_cand, target_choke)
            traj = self.model.predict_trajectory(state, seq, self.horizon)
            n_evaluated += 1

            safe, _ = self.cm.is_trajectory_safe(traj)
            if not safe:
                continue
            n_safe += 1

            score = self.score_trajectory(traj, target_Q)
            if score > best_score:
                best_score     = score
                best_candidate = u_cand

        # emergency fallback if nothing is safe
        if best_candidate is None:
            choke_cmd = float(np.clip(current_choke - 5.0, 0.0, 100.0))
            self.current_choke = choke_cmd
            return {
                "choke_command":        round(choke_cmd, 2),
                "status":               "EMERGENCY_CLOSE",
                "method":               "BRUTE_FORCE",
                "predicted_Q":          None,
                "candidates_evaluated": n_evaluated,
                "candidates_safe":      n_safe,
                "active_limit":         None,
                "solve_time_ms":        round((time.time() - t0) * 1000, 1),
                "solver_iters":         n_evaluated,
            }

        # final ramp-rate enforcement
        move      = np.clip(best_candidate - current_choke, -5.0, 5.0)
        choke_cmd = float(np.clip(current_choke + move, 0.0, 100.0))
        self.current_choke = choke_cmd

        # one-step-ahead Q prediction for reporting
        one_step = self.model.predict_trajectory(state, [choke_cmd], 1)
        pred_Q   = round(float(one_step["Q"][0]), 3)

        return {
            "choke_command":        round(choke_cmd, 2),
            "status":               "OK",
            "method":               "BRUTE_FORCE",
            "predicted_Q":          pred_Q,
            "candidates_evaluated": n_evaluated,
            "candidates_safe":      n_safe,
            "active_limit":         None,
            "solve_time_ms":        round((time.time() - t0) * 1000, 1),
            "solver_iters":         n_evaluated,
        }


if __name__ == "__main__":
    from oil_well_simulator import OilWellSimulator

    # load sweep data and fit model
    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))
    model = WellModel()
    model.fit(df_sweep, verbose=False)

    cm  = ConstraintManager()
    mpc = BruteForceMPC(model, cm, horizon=10)
    sim = OilWellSimulator()
    sim.reset()

    target_Q = 60.0
    n_steps  = 20
    any_violation = False

    print(f"\n{'t':>4}  {'target':>7}  {'Q':>7}  {'choke':>7}  {'WHP':>7}  {'FLP':>7}  {'BHP':>7}  status")
    print("-" * 72)

    choke = 0.0
    for t in range(n_steps):
        q, whp, flp, bhp = sim.step(choke)

        state = {"Q": q, "WHP": whp, "FLP": flp, "BHP": bhp, "choke": choke}
        result = mpc.compute(state, target_Q)
        choke  = result["choke_command"]

        # check constraints on actual measured state
        safe, reason = cm.is_safe({"WHP": whp, "FLP": flp, "BHP": bhp})
        if not safe:
            any_violation = True
            status_str = f"VIOLATION: {reason}"
        else:
            status_str = result["status"]

        print(f"{t:>4}  {target_Q:>7.1f}  {q:>7.2f}  {choke:>7.2f}  {whp:>7.1f}  {flp:>7.1f}  {bhp:>7.1f}  {status_str}")

    print()
    if any_violation:
        print("RESULT: Constraint violation(s) occurred during the test.")
    else:
        print("RESULT: No constraint violations — all steps safe.")
