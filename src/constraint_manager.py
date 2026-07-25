import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


class ConstraintManager:

    def __init__(self):
        self.limits = {
            "BHP_min":        2200.0,
            "WHP_min":        600.0,
            "WHP_max":        3000.0,
            "FLP_max":        700.0,
            "choke_min":      0.0,
            "choke_max":      100.0,
            "delta_choke_max": 5.0,
        }
        self.margin = 0.02  # 2% safety buffer

    def is_safe(self, state, use_margin=True):
        m = self.margin if use_margin else 0.0
        violations = []

        bhp_min = self.limits["BHP_min"] * (1 + m)
        whp_min = self.limits["WHP_min"] * (1 + m)
        whp_max = self.limits["WHP_max"] * (1 - m)
        flp_max = self.limits["FLP_max"] * (1 - m)

        # BHP is highest priority — reservoir integrity
        if state["BHP"] < bhp_min:
            violations.append(f"BHP={state['BHP']:.1f} < BHP_min={bhp_min:.1f}")

        if state["WHP"] < whp_min:
            violations.append(f"WHP={state['WHP']:.1f} < WHP_min={whp_min:.1f}")
        if state["WHP"] > whp_max:
            violations.append(f"WHP={state['WHP']:.1f} > WHP_max={whp_max:.1f}")

        if state["FLP"] > flp_max:
            violations.append(f"FLP={state['FLP']:.1f} > FLP_max={flp_max:.1f}")

        if violations:
            return False, " | ".join(violations)
        return True, "SAFE"

    def is_trajectory_safe(self, trajectory):
        n = len(trajectory["Q"])
        for k in range(n):
            state = {var: trajectory[var][k] for var in ["Q", "WHP", "FLP", "BHP"]}
            safe, reason = self.is_safe(state, use_margin=True)
            if not safe:
                return False, f"step {k}: {reason}"
        return True, "SAFE"

    def get_choke_candidates(self, current_choke, resolution=0.5):
        dmax = self.limits["delta_choke_max"]
        offsets = np.arange(-dmax, dmax + resolution, resolution)
        candidates = current_choke + offsets
        candidates = np.clip(candidates, self.limits["choke_min"], self.limits["choke_max"])
        candidates = np.unique(np.round(candidates, 6))
        return candidates

    def compute_max_safe_production(self, model):
        choke_vals = np.arange(0, 100.5, 0.5)
        max_Q      = 0.0
        best_choke = 0.0
        limit_str  = "none"
        found_safe = False

        for u in choke_vals:
            ss    = model.predict_steady_state(u)
            state = {var: float(ss[var][0]) for var in ["Q", "WHP", "FLP", "BHP"]}
            safe, reason = self.is_safe(state, use_margin=True)

            if safe:
                found_safe = True
                if state["Q"] > max_Q:
                    max_Q      = state["Q"]
                    best_choke = u
            elif found_safe:
                # first unsafe point after a run of safe ones — record and stop
                limit_str = reason
                break

        return max_Q, best_choke, limit_str

    def plot_safe_envelope(self, model, save_path):
        max_Q, best_choke, limit_str = self.compute_max_safe_production(model)
        print(f"\nMax safe production : {max_Q:.2f} bbl/hr")
        print(f"Optimal choke       : {best_choke:.1f} %")
        print(f"Limiting constraint : {limit_str}")

        choke_vals = np.arange(0, 100.5, 0.5)
        Q_vals, BHP_vals, WHP_vals, FLP_vals = [], [], [], []
        colors = []

        for u in choke_vals:
            ss    = model.predict_steady_state(u)
            state = {var: float(ss[var][0]) for var in ["Q", "WHP", "FLP", "BHP"]}
            safe, _ = self.is_safe(state, use_margin=True)
            Q_vals.append(state["Q"])
            BHP_vals.append(state["BHP"])
            WHP_vals.append(state["WHP"])
            FLP_vals.append(state["FLP"])
            colors.append("green" if safe else "red")

        Q_vals   = np.array(Q_vals)
        BHP_vals = np.array(BHP_vals)
        WHP_vals = np.array(WHP_vals)
        FLP_vals = np.array(FLP_vals)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # left: Q vs choke coloured by safety
        for c_val in ["green", "red"]:
            mask = np.array(colors) == c_val
            label = "safe" if c_val == "green" else "unsafe"
            ax1.scatter(choke_vals[mask], Q_vals[mask], c=c_val, s=10, label=label)
        ax1.axhline(max_Q,      color="blue",   linestyle="--", label=f"max safe Q={max_Q:.1f}")
        ax1.axvline(best_choke, color="orange", linestyle=":",  label=f"opt choke={best_choke:.1f}%")
        ax1.set_xlabel("Choke (%)")
        ax1.set_ylabel("Q (bbl/hr)")
        ax1.set_title("Production vs Choke")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)

        # right: pressure profiles vs choke
        ax2.plot(choke_vals, BHP_vals, color="purple",     label="BHP")
        ax2.plot(choke_vals, WHP_vals, color="darkorange",  label="WHP")
        ax2.plot(choke_vals, FLP_vals, color="green",       label="FLP")
        ax2.axhline(self.limits["BHP_min"], color="purple",     linestyle="--", alpha=0.7, label=f"BHP_min={self.limits['BHP_min']}")
        ax2.axhline(self.limits["WHP_max"], color="darkorange",  linestyle="--", alpha=0.7, label=f"WHP_max={self.limits['WHP_max']}")
        ax2.axhline(self.limits["FLP_max"], color="green",       linestyle="--", alpha=0.7, label=f"FLP_max={self.limits['FLP_max']}")
        ax2.axvline(best_choke,             color="orange",       linestyle=":",              label=f"opt choke={best_choke:.1f}%")
        ax2.set_xlabel("Choke (%)")
        ax2.set_ylabel("Pressure (psi)")
        ax2.set_title("Pressure Profiles vs Choke")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)

        fig.suptitle("Safe Operating Envelope", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    from well_model import WellModel

    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))
    model = WellModel()
    model.fit(df_sweep, verbose=False)

    cm = ConstraintManager()
    cm.plot_safe_envelope(model, os.path.join(RESULTS, "safe_envelope.png"))
