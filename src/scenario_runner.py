import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, HERE)

from oil_well_simulator  import OilWellSimulator
from well_model          import WellModel
from constraint_manager  import ConstraintManager
from scipy_mpc           import ScipyMPC

# status colour map for subplot 6
STATUS_COLORS = {
    "TRACKING":        "green",
    "CONVERGING":      "orange",   # matched by substring
    "CONSTRAINED_MAX": "purple",
    "FALLBACK_ACTIVE": "red",
    "EMERGENCY_CLOSE": "darkred",
    "OTHER":           "grey",
}


def _status_color(s):
    if s == "TRACKING":           return "green"
    if "CONVERGING" in s:         return "orange"
    if s == "CONSTRAINED_MAX":    return "purple"
    if s == "FALLBACK_ACTIVE":    return "red"
    if s == "EMERGENCY_CLOSE":    return "darkred"
    return "grey"


class ScenarioRunner:

    def __init__(self, simulator, controller, constraints, model):
        self.sim   = simulator
        self.ctrl  = controller
        self.cm    = constraints
        self.model = model

    def run(self, name, target_schedule, n_steps, initial_choke=0.0):
        # build step-index -> target lookup
        sched = sorted(target_schedule, key=lambda x: x[0])
        def get_target(t):
            target = sched[0][1]
            for idx, val in sched:
                if t >= idx:
                    target = val
            return target

        self.sim.reset()
        self.ctrl.reset(initial_choke)

        # warm up with one step at initial choke to get a valid starting state
        q, whp, flp, bhp = self.sim.step(initial_choke)
        choke = initial_choke

        rows = []
        print(f"\n{'='*80}")
        print(f"  {name}")
        print(f"{'='*80}")
        print(f"{'t':>4}  {'target':>7}  {'Q':>7}  {'choke':>7}  {'WHP':>7}  {'FLP':>7}  {'BHP':>7}  {'safe':>5}  status")
        print("-" * 80)

        for t in range(n_steps):
            target = get_target(t)
            state  = {"Q": q, "WHP": whp, "FLP": flp, "BHP": bhp, "choke": choke}

            result = self.ctrl.compute(state, target)
            choke  = result["choke_command"]

            q, whp, flp, bhp = self.sim.step(choke)

            safe, viol = self.cm.is_safe({"WHP": whp, "FLP": flp, "BHP": bhp}, use_margin=False)
            safe_sym   = "OK" if safe else "!!"

            row = {
                "time":         t,
                "target_Q":     target,
                "Q":            q,
                "WHP":          whp,
                "FLP":          flp,
                "BHP":          bhp,
                "choke":        choke,
                "status":       result["status"],
                "method":       result["method"],
                "solve_time_ms": result["solve_time_ms"],
                "solver_iters": result["solver_iters"],
                "active_limit": result["active_limit"],
                "is_safe":      safe,
                "violation":    None if safe else viol,
            }
            rows.append(row)

            # print every 5th step and every unsafe step
            if t % 5 == 0 or not safe:
                print(f"{t:>4}  {target:>7.1f}  {q:>7.2f}  {choke:>7.2f}  "
                      f"{whp:>7.1f}  {flp:>7.1f}  {bhp:>7.1f}  {safe_sym:>5}  {result['status']}")

        df = pd.DataFrame(rows)
        n_viol = (~df["is_safe"]).sum()
        print(f"\n  Steps: {n_steps}  |  Violations: {n_viol}")
        return df

    def compute_kpis(self, df, name):
        kpis = {}

        n_viol = int((~df["is_safe"]).sum())
        pct_viol = 100 * n_viol / len(df)
        kpis["violations"]     = n_viol
        kpis["pct_violations"] = pct_viol

        # per-target metrics
        iae_total = 0.0
        for target, grp in df.groupby("target_Q"):
            grp = grp.reset_index(drop=True)
            n   = len(grp)
            steady = grp.iloc[max(0, n - n // 3):]
            mean_Q = steady["Q"].mean()
            iae    = (grp["Q"] - target).abs().sum()
            iae_total += iae

            err_frac = (grp["Q"] - target).abs() / max(abs(target), 1e-6)
            exceeded = grp.index[err_frac > 0.05].tolist()
            settling = (exceeded[-1] + 1) if exceeded else 0

            kpis[f"mean_Q_{target}"]    = mean_Q
            kpis[f"iae_{target}"]       = iae
            kpis[f"settling_{target}"]  = settling

        kpis["iae_total"] = iae_total

        choke_diff   = df["choke"].diff().abs().fillna(0)
        kpis["choke_travel"]    = choke_diff.sum()
        kpis["n_large_moves"]   = int((choke_diff > 0.05).sum())
        kpis["max_choke_move"]  = choke_diff.max()
        kpis["avg_solve_ms"]    = df["solve_time_ms"].mean()
        kpis["pct_scipy"]       = 100 * df["method"].str.contains("SCIPY").mean()

        print(f"\n--- KPIs: {name} ---")
        print(f"  Violations         : {n_viol}  ({pct_viol:.1f}%)")
        for target, grp in df.groupby("target_Q"):
            print(f"  Target {target:.0f} bbl/hr:")
            print(f"    Steady-state Q   : {kpis[f'mean_Q_{target}']:.2f} bbl/hr")
            print(f"    IAE              : {kpis[f'iae_{target}']:.1f}")
            print(f"    Settling time    : {kpis[f'settling_{target}']} steps")
        print(f"  Total IAE          : {iae_total:.1f}")
        print(f"  Choke travel       : {kpis['choke_travel']:.2f} %")
        print(f"  Large moves (>0.05): {kpis['n_large_moves']}")
        print(f"  Max single move    : {kpis['max_choke_move']:.2f} %")
        print(f"  Avg solve time     : {kpis['avg_solve_ms']:.1f} ms")
        print(f"  Scipy solve %      : {kpis['pct_scipy']:.1f}%")
        return kpis

    def plot_scenario(self, df, name, save_path):
        lim = self.cm.limits
        t   = df["time"].values

        # find unsafe time regions for shading
        unsafe_mask = ~df["is_safe"].values

        fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
        fig.suptitle(name, fontsize=14, fontweight="bold")

        def shade_unsafe(ax):
            """shade unsafe steps light red across an axis"""
            if unsafe_mask.any():
                for i, bad in enumerate(unsafe_mask):
                    if bad:
                        ax.axvspan(t[i] - 0.5, t[i] + 0.5, color="red", alpha=0.15, zorder=0)

        # --- subplot 1: Q tracking ---
        ax = axes[0]
        target_vals = df["target_Q"].values
        ax.step(t, target_vals, where="post", color="red",   linestyle="--", label="Target Q", zorder=3)
        ax.plot(t, df["Q"].values, color="green", label="Actual Q", zorder=4)
        ax.fill_between(t, df["Q"].values, target_vals, alpha=0.15, color="grey", step="post")
        ax.set_ylabel("Oil Rate (bbl/hr)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        shade_unsafe(ax)

        # --- subplot 2: WHP ---
        ax = axes[1]
        ax.plot(t, df["WHP"].values, color="darkorange", label="WHP")
        ax.axhline(lim["WHP_min"], color="red", linestyle="--", alpha=0.8, label=f"WHP_min={lim['WHP_min']}")
        ax.axhline(lim["WHP_max"], color="red", linestyle="--", alpha=0.8, label=f"WHP_max={lim['WHP_max']}")
        ax.set_ylabel("WHP (psi)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        shade_unsafe(ax)

        # --- subplot 3: FLP ---
        ax = axes[2]
        ax.plot(t, df["FLP"].values, color="purple", label="FLP")
        ax.axhline(lim["FLP_max"], color="red", linestyle="--", alpha=0.8, label=f"FLP_max={lim['FLP_max']}")
        ax.set_ylabel("FLP (psi)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        shade_unsafe(ax)

        # --- subplot 4: BHP ---
        ax = axes[3]
        ax.plot(t, df["BHP"].values, color="red", label="BHP")
        ax.axhline(lim["BHP_min"], color="darkred", linestyle="--", alpha=0.8, label=f"BHP_min={lim['BHP_min']}")
        ax.fill_between(t, df["BHP"].values, lim["BHP_min"],
                        where=df["BHP"].values < lim["BHP_min"],
                        color="red", alpha=0.2, label="BHP violation zone")
        ax.set_ylabel("BHP (psi)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        shade_unsafe(ax)

        # --- subplot 5: choke ---
        ax = axes[4]
        choke_vals = df["choke"].values
        ax.step(t, choke_vals, where="post", color="steelblue", label="Choke")
        ax.fill_between(t, choke_vals, 0, alpha=0.15, color="lightblue", step="post")
        ax.set_ylim(-5, 110)
        ax.set_ylabel("Choke (%)")
        ax.grid(alpha=0.3)
        max_move = df["choke"].diff().abs().max()
        pass_fail = "PASS" if max_move <= lim["delta_choke_max"] else "FAIL"
        ax.text(0.02, 0.08,
                f"Max move: {max_move:.2f}% (limit {lim['delta_choke_max']}%) — {pass_fail}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
        ax.legend(fontsize=8)

        # --- subplot 6: status bars ---
        ax = axes[5]
        for i, row in df.iterrows():
            c = _status_color(row["status"])
            ax.bar(row["time"], 1, bottom=0, color=c, width=1.0, align="center")
            if not row["is_safe"]:
                ax.bar(row["time"], 1, bottom=0, color="black", width=1.0, align="center", alpha=0.4)

        ax.set_yticks([])
        ax.set_ylabel("Status")
        ax.set_xlabel("Time (control intervals, 1 hr each)")

        # build legend patches
        legend_patches = [
            mpatches.Patch(color="green",   label="TRACKING"),
            mpatches.Patch(color="orange",  label="CONVERGING"),
            mpatches.Patch(color="purple",  label="CONSTRAINED_MAX"),
            mpatches.Patch(color="red",     label="FALLBACK_ACTIVE"),
            mpatches.Patch(color="darkred", label="EMERGENCY_CLOSE"),
            mpatches.Patch(color="grey",    label="OTHER"),
            mpatches.Patch(color="black",   label="UNSAFE", alpha=0.4),
        ]
        ax.legend(handles=legend_patches, fontsize=7, ncol=4, loc="upper right")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {save_path}")

    def infeasibility_report(self, model, requested_target):
        max_Q, opt_choke, limit_str = self.cm.compute_max_safe_production(model)
        gap     = requested_target - max_Q
        gap_pct = 100 * gap / max(abs(requested_target), 1e-6)

        print(f"\n{'='*55}")
        print(f"  Infeasibility Report")
        print(f"{'='*55}")
        print(f"  Requested target   : {requested_target:.1f} bbl/hr")
        print(f"  Max safe rate      : {max_Q:.2f} bbl/hr")
        print(f"  Shortfall          : {gap:.2f} bbl/hr  ({gap_pct:.1f}% of target)")
        print(f"  Optimal choke      : {opt_choke:.1f} %")
        print(f"  Limiting constraint: {limit_str}")
        print(f"{'='*55}")
        return max_Q


if __name__ == "__main__":
    np.random.seed(42)

    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))
    model = WellModel()
    model.fit(df_sweep, verbose=False)

    sim  = OilWellSimulator()
    cm   = ConstraintManager()
    ctrl = ScipyMPC(model, cm, horizon=5)
    runner = ScenarioRunner(sim, ctrl, cm, model)

    summary_rows = []

    # ---- Scenario A: Startup to target ----
    df_A = runner.run("Scenario A - Startup to Target",
                      target_schedule=[(0, 60)],
                      n_steps=40, initial_choke=0.0)
    kpis_A = runner.compute_kpis(df_A, "Scenario A")
    runner.plot_scenario(df_A, "Scenario A - Startup to Target",
                         os.path.join(RESULTS, "scenario_A.png"))
    df_A.to_csv(os.path.join(RESULTS, "scenario_A.csv"), index=False)
    summary_rows.append({
        "Scenario":    "A - Startup to 60",
        "Target":      "60 bbl/hr",
        "Steady Q":    round(next((v for k, v in kpis_A.items() if k.startswith("mean_Q_")), float("nan")), 2),
        "Violations":  f"{kpis_A['violations']} {'OK' if kpis_A['violations']==0 else 'XX'}",
        "IAE":         round(kpis_A["iae_total"], 1),
        "Choke Travel": round(kpis_A["choke_travel"], 2),
        "Max Move":    round(kpis_A["max_choke_move"], 2),
        "Scipy %":     round(kpis_A["pct_scipy"], 1),
    })

    # ---- Scenario B: Target tracking (step change) ----
    df_B = runner.run("Scenario B - Target Tracking",
                      target_schedule=[(0, 50), (25, 70)],
                      n_steps=60, initial_choke=0.0)
    kpis_B = runner.compute_kpis(df_B, "Scenario B")
    runner.plot_scenario(df_B, "Scenario B - Target Tracking",
                         os.path.join(RESULTS, "scenario_B.png"))
    df_B.to_csv(os.path.join(RESULTS, "scenario_B.csv"), index=False)
    summary_rows.append({
        "Scenario":    "B - Target Tracking",
        "Target":      "50->70 bbl/hr",
        "Steady Q":    round(next((v for k, v in kpis_B.items() if k.startswith("mean_Q_70")), float("nan")), 2),
        "Violations":  f"{kpis_B['violations']} {'OK' if kpis_B['violations']==0 else 'XX'}",
        "IAE":         round(kpis_B["iae_total"], 1),
        "Choke Travel": round(kpis_B["choke_travel"], 2),
        "Max Move":    round(kpis_B["max_choke_move"], 2),
        "Scipy %":     round(kpis_B["pct_scipy"], 1),
    })

    # ---- Scenario C: Infeasible target ----
    runner.infeasibility_report(model, requested_target=100.0)
    df_C = runner.run("Scenario C - Infeasible Target",
                      target_schedule=[(0, 100)],
                      n_steps=50, initial_choke=0.0)
    kpis_C = runner.compute_kpis(df_C, "Scenario C")
    runner.plot_scenario(df_C, "Scenario C - Infeasible Target",
                         os.path.join(RESULTS, "scenario_C.png"))
    df_C.to_csv(os.path.join(RESULTS, "scenario_C.csv"), index=False)
    summary_rows.append({
        "Scenario":    "C - Infeasible 100",
        "Target":      "100 bbl/hr",
        "Steady Q":    round(next((v for k, v in kpis_C.items() if k.startswith("mean_Q_")), float("nan")), 2),
        "Violations":  f"{kpis_C['violations']} {'OK' if kpis_C['violations']==0 else 'XX'}",
        "IAE":         round(kpis_C["iae_total"], 1),
        "Choke Travel": round(kpis_C["choke_travel"], 2),
        "Max Move":    round(kpis_C["max_choke_move"], 2),
        "Scipy %":     round(kpis_C["pct_scipy"], 1),
    })

    # ---- Combined summary ----
    print(f"\n{'='*80}")
    print("  Combined Scenario Summary")
    print(f"{'='*80}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    ctrl.report()
