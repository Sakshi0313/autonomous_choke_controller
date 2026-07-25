import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

OUTPUTS = ["Q", "WHP", "FLP", "BHP"]
UNITS   = {"Q": "bbl/hr", "WHP": "psi", "FLP": "psi", "BHP": "psi"}


class WellModel:

    def __init__(self, Ts=1.0):
        self.Ts = Ts
        self.ss_coeffs = {}          # quadratic coeffs for each output
        self.taus = {"Q": 2.0, "WHP": 1.5, "FLP": 1.5, "BHP": 3.5}
        self.r_squared = {}
        self.rmse = {}
        self.PI = None               # productivity index
        self.p_res = None            # reservoir pressure
        self.is_fitted = False

    def extract_steady_state(self, df):
        rows = []
        for choke, grp in df.groupby("choke", sort=True):
            tail = grp.tail(5)
            rows.append({"choke": choke,
                         "Q":   tail["Q"].mean(),
                         "WHP": tail["WHP"].mean(),
                         "FLP": tail["FLP"].mean(),
                         "BHP": tail["BHP"].mean()})
        return pd.DataFrame(rows)

    def fit_steady_state(self, ss_df):
        u = ss_df["choke"].values
        print("\n-- Steady-State Quadratic Fits --")
        for var in OUTPUTS:
            y = ss_df[var].values
            # degree-2 poly: coeffs = [c, b, a] -> y = a + b*u + c*u^2
            coeffs = np.polyfit(u, y, 2)
            self.ss_coeffs[var] = coeffs
            c, b, a = coeffs
            print(f"  {var:4s} = {a:.4f} + {b:.4f}*u + {c:.6f}*u^2")

    def estimate_time_constants(self, df):
        # find rows where choke jumps by more than 5%
        choke = df["choke"].values
        step_idxs = np.where(np.abs(np.diff(choke)) > 5)[0]

        if len(step_idxs) == 0:
            print("  No step changes found, keeping default time constants.")
            return

        tau_estimates = {v: [] for v in OUTPUTS}
        window = 20

        for idx in step_idxs:
            if idx + window >= len(df):
                continue
            for var in OUTPUTS:
                y = df[var].values
                y0  = y[idx]
                y_end = y[idx + window]
                delta = y_end - y0
                if abs(delta) < 0.1:
                    continue
                # find first time it crosses 63.2% of the total change
                target = y0 + 0.632 * delta
                crossed = False
                for k in range(1, window + 1):
                    if delta > 0 and y[idx + k] >= target:
                        tau_estimates[var].append(float(k))
                        crossed = True
                        break
                    elif delta < 0 and y[idx + k] <= target:
                        tau_estimates[var].append(float(k))
                        crossed = True
                        break
                if not crossed:
                    # didn't cross within window — tau is larger than window
                    tau_estimates[var].append(float(window))

        print("\n-- Estimated Time Constants --")
        for var in OUTPUTS:
            ests = tau_estimates[var]
            if len(ests) > 0:
                tau = np.mean(ests)
                tau = float(np.clip(tau, 0.5, 20.0))
                self.taus[var] = tau

        # BHP diffusion is physically slower than wellbore flow
        if self.taus["BHP"] < self.taus["Q"]:
            self.taus["BHP"] = self.taus["Q"] * 1.3

        for var in OUTPUTS:
            print(f"  tau_{var:3s} = {self.taus[var]:.2f} hr")

    def derive_physics_params(self, ss_df):
        # only use flowing rows for the fit
        mask = ss_df["Q"].values > 0.5
        q   = ss_df["Q"].values[mask]
        bhp = ss_df["BHP"].values[mask]

        coeffs = np.polyfit(q, bhp, 1)
        m, c = coeffs
        self.PI    = -1.0 / m       # darcy's law: BHP = p_res - Q/PI
        self.p_res = c

        print(f"\n-- Derived Physics Params --")
        print(f"  PI         = {self.PI:.4f} bbl/hr/psi")
        print(f"  P_reservoir = {self.p_res:.2f} psi")

    def predict_steady_state(self, choke):
        u = np.atleast_1d(np.asarray(choke, dtype=float))
        return {var: np.polyval(self.ss_coeffs[var], u) for var in OUTPUTS}

    def predict_trajectory(self, current_state, choke_sequence, horizon):
        state = {var: float(current_state[var]) for var in OUTPUTS}
        traj  = {var: np.zeros(horizon) for var in OUTPUTS}

        choke_seq = list(choke_sequence)

        for k in range(horizon):
            u = choke_seq[k] if k < len(choke_seq) else choke_seq[-1]
            u = float(np.clip(u, 0.0, 100.0))

            ss = self.predict_steady_state(u)
            for var in OUTPUTS:
                alpha = min(self.Ts / self.taus[var], 1.0)   # cap for stability
                state[var] += alpha * (float(ss[var][0]) - state[var])
                traj[var][k] = state[var]

        return traj

    def invert_for_choke(self, target_Q):
        def obj(u):
            ss = self.predict_steady_state(u)
            return (ss["Q"][0] - target_Q) ** 2

        res = minimize_scalar(obj, bounds=(0.0, 100.0), method="bounded")
        return float(np.clip(res.x, 0.0, 100.0))

    def compute_fit_quality(self, ss_df):
        u    = ss_df["choke"].values
        print("\n-- Fit Quality --")
        print(f"  {'Var':4s}  {'R²':>8}  {'RMSE':>10}")
        for var in OUTPUTS:
            y_true = ss_df[var].values
            y_pred = np.polyval(self.ss_coeffs[var], u)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            self.r_squared[var] = r2
            self.rmse[var]      = rmse
            print(f"  {var:4s}  {r2:8.4f}  {rmse:10.4f} {UNITS[var]}")

    def fit(self, df, verbose=True):
        if verbose:
            print("=" * 45)
            print("  WellModel.fit()")
            print("=" * 45)

        ss_df = self.extract_steady_state(df)
        self.fit_steady_state(ss_df)
        self.estimate_time_constants(df)
        self.derive_physics_params(ss_df)
        self.compute_fit_quality(ss_df)
        self.is_fitted = True
        return ss_df

    def plot_steady_state_fit(self, ss_df, save_path):
        u_data = ss_df["choke"].values
        u_fine = np.linspace(0, 100, 200)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()

        for i, var in enumerate(OUTPUTS):
            ax = axes[i]
            y_data = ss_df[var].values
            y_fit  = np.polyval(self.ss_coeffs[var], u_fine)

            ax.scatter(u_data, y_data, color="steelblue", zorder=3, label="data")
            ax.plot(u_fine, y_fit, color="tomato", label="fit")
            ax.set_xlabel("Choke (%)")
            ax.set_ylabel(f"{var} ({UNITS[var]})")
            ax.set_title(f"{var}  —  R²={self.r_squared[var]:.4f}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle("Steady-State Quadratic Fits", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {save_path}")

    def plot_dynamic_validation(self, df, save_path):
        # use first row as initial state, then simulate the full choke sequence
        init = {var: df[var].iloc[0] for var in OUTPUTS}
        choke_seq = df["choke"].values.tolist()
        horizon   = len(choke_seq)

        traj = self.predict_trajectory(init, choke_seq, horizon)
        t    = df["time"].values

        fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

        axes[0].step(t, df["choke"], where="post", color="black")
        axes[0].set_ylabel("Choke (%)")
        axes[0].grid(alpha=0.3)

        colors = ["steelblue", "darkorange", "green", "purple"]
        for i, var in enumerate(OUTPUTS):
            ax = axes[i + 1]
            y_actual = df[var].values
            y_pred   = traj[var]

            # r2 on dynamic data
            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            ax.plot(t, y_actual, color=colors[i], label=f"actual")
            ax.plot(t, y_pred,   color="black", linestyle="--", label=f"predicted (R²={r2:.3f})")
            ax.set_ylabel(f"{var} ({UNITS[var]})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        axes[-1].set_xlabel("Time (hours)")
        fig.suptitle("Dynamic Validation — Actual vs Predicted", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {save_path}")


if __name__ == "__main__":
    np.random.seed(0)

    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))

    model = WellModel()
    ss_df = model.fit(df_sweep)

    model.plot_steady_state_fit(ss_df, os.path.join(RESULTS, "model_steady_state_fit.png"))

    df_dynamic = pd.read_csv(os.path.join(RESULTS, "step_test_dynamic.csv"))
    model.plot_dynamic_validation(df_dynamic, os.path.join(RESULTS, "model_dynamic_validation.png"))

    # final summary table
    print("\n" + "=" * 65)
    print(f"  {'Var':4s}  {'Equation (a + b*u + c*u²)':35s}  {'R²':>6}  {'tau (hr)':>8}")
    print("=" * 65)
    for var in OUTPUTS:
        c, b, a = model.ss_coeffs[var]
        eq = f"{a:.3f} + {b:.4f}*u + {c:.6f}*u²"
        print(f"  {var:4s}  {eq:35s}  {model.r_squared[var]:.4f}  {model.taus[var]:8.2f}")
