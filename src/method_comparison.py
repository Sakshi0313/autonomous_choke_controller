import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, HERE)

from oil_well_simulator  import OilWellSimulator
from well_model          import WellModel
from constraint_manager  import ConstraintManager
from brute_force_mpc     import BruteForceMPC
from scipy_mpc           import ScipyMPC
from scenario_runner     import ScenarioRunner

SCHEDULE = [(0, 50), (25, 70)]
N_STEPS  = 60


def compute_metrics(df_bf, df_sc):
    """Compare both controllers on key metrics, return a summary DataFrame."""

    def choke_reversals(choke_vals):
        diffs = np.diff(choke_vals)
        diffs = diffs[diffs != 0]   # ignore zero moves
        if len(diffs) < 2:
            return 0
        return int(np.sum(np.diff(np.sign(diffs)) != 0))

    results = {}
    for label, df in [("Brute Force", df_bf), ("Scipy SLSQP", df_sc)]:
        choke  = df["choke"].values
        q      = df["Q"].values
        target = df["target_Q"].values

        iae         = np.sum(np.abs(q - target))
        ss_error    = abs(df["Q"].tail(10).mean() - df["target_Q"].iloc[-1])
        travel      = np.sum(np.abs(np.diff(choke)))
        reversals   = choke_reversals(choke)
        distinct    = len(np.unique(np.round(choke, 2)))
        avg_solve   = df["solve_time_ms"].mean()
        max_move    = np.abs(np.diff(choke)).max() if len(choke) > 1 else 0.0

        results[label] = {
            "Total IAE":          round(iae, 2),
            "SS Error (bbl/hr)":  round(ss_error, 3),
            "Choke Travel (%)":   round(travel, 2),
            "Reversals":          reversals,
            "Distinct Positions": distinct,
            "Avg Solve (ms)":     round(avg_solve, 1),
            "Max Move (%)":       round(max_move, 3),
        }

    # lower is better for all except Distinct Positions (more = finer resolution)
    higher_better = {"Distinct Positions"}

    rows = []
    for metric in results["Brute Force"]:
        bf_val = results["Brute Force"][metric]
        sc_val = results["Scipy SLSQP"][metric]
        if metric in higher_better:
            winner = "Scipy SLSQP" if sc_val > bf_val else "Brute Force" if bf_val > sc_val else "Tie"
        else:
            winner = "Scipy SLSQP" if sc_val < bf_val else "Brute Force" if bf_val < sc_val else "Tie"
        rows.append({"Metric": metric, "Brute Force": bf_val, "Scipy SLSQP": sc_val, "Winner": winner})

    cmp_df = pd.DataFrame(rows)
    print("\n--- Method Comparison Metrics ---")
    print(cmp_df.to_string(index=False))
    return cmp_df


def comparison_plot(df_bf, df_sc, cmp_df, save_path):
    t = df_bf["time"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Brute Force vs Scipy SLSQP — Method Comparison", fontsize=13, fontweight="bold")

    # top left: Q tracking
    ax = axes[0, 0]
    ax.plot(t, df_bf["Q"].values,  color="steelblue",  label="Brute Force Q")
    ax.plot(t, df_sc["Q"].values,  color="darkorange",  label="Scipy Q")
    ax.step(t, df_bf["target_Q"].values, where="post", color="black", linestyle="--", label="Target")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Q (bbl/hr)")
    ax.set_title("Oil Rate Tracking")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # top right: choke position
    ax = axes[0, 1]
    ax.step(t, df_bf["choke"].values, where="post", color="steelblue",  label="Brute Force")
    ax.step(t, df_sc["choke"].values, where="post", color="darkorange",  label="Scipy SLSQP")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Choke (%)")
    ax.set_title("Choke Position")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # bottom left: histogram of choke values in last 20 steps
    ax = axes[1, 0]
    last_bf = df_bf["choke"].tail(20).values
    last_sc = df_sc["choke"].tail(20).values
    bins = np.linspace(
        min(last_bf.min(), last_sc.min()) - 1,
        max(last_bf.max(), last_sc.max()) + 1,
        21
    )
    ax.hist(last_bf, bins=bins, alpha=0.6, color="steelblue",  label="Brute Force")
    ax.hist(last_sc, bins=bins, alpha=0.6, color="darkorange",  label="Scipy SLSQP")
    ax.set_xlabel("Choke (%)")
    ax.set_ylabel("Count")
    ax.set_title("Choke Resolution in Steady State (last 20 steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # bottom right: normalised bar chart for IAE, choke travel, reversals
    ax = axes[1, 1]
    metrics_to_plot = ["Total IAE", "Choke Travel (%)", "Reversals"]
    bf_vals = [cmp_df.loc[cmp_df["Metric"] == m, "Brute Force"].values[0] for m in metrics_to_plot]
    sc_vals = [cmp_df.loc[cmp_df["Metric"] == m, "Scipy SLSQP"].values[0] for m in metrics_to_plot]

    # normalise so the larger value = 1
    bf_norm = [bf / max(bf, sc, 1e-9) for bf, sc in zip(bf_vals, sc_vals)]
    sc_norm = [sc / max(bf, sc, 1e-9) for bf, sc in zip(bf_vals, sc_vals)]

    x      = np.arange(len(metrics_to_plot))
    width  = 0.35
    bars_bf = ax.bar(x - width / 2, bf_norm, width, color="steelblue",  alpha=0.8, label="Brute Force")
    bars_sc = ax.bar(x + width / 2, sc_norm, width, color="darkorange",  alpha=0.8, label="Scipy SLSQP")

    # value labels on bars
    for bar, raw in zip(bars_bf, bf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                str(raw), ha="center", va="bottom", fontsize=7)
    for bar, raw in zip(bars_sc, sc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                str(raw), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(["IAE", "Choke Travel", "Reversals"], fontsize=9)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Normalised Metric Comparison (lower = better)")
    ax.set_ylim(0, 1.25)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def grid_resolution_demo(df_bf, df_sc):
    last_bf = np.sort(np.unique(np.round(df_bf["choke"].tail(20).values, 2)))
    last_sc = np.sort(np.unique(np.round(df_sc["choke"].tail(20).values, 2)))

    print("\n--- Grid Resolution Demo (last 20 steps) ---")
    print(f"  Brute Force unique choke values ({len(last_bf)}): {last_bf}")
    print(f"  Scipy SLSQP unique choke values ({len(last_sc)}): {last_sc}")
    print(f"  Brute Force used {len(last_bf)} distinct positions "
          f"(constrained to 0.5% search grid).")
    print(f"  Scipy SLSQP used {len(last_sc)} distinct positions "
          f"(continuous — arbitrary precision).")


def print_summary(df_bf, df_sc):
    iae_bf = np.sum(np.abs(df_bf["Q"].values - df_bf["target_Q"].values))
    iae_sc = np.sum(np.abs(df_sc["Q"].values - df_sc["target_Q"].values))
    travel_bf = np.sum(np.abs(np.diff(df_bf["choke"].values)))
    travel_sc = np.sum(np.abs(np.diff(df_sc["choke"].values)))
    distinct_bf = len(np.unique(np.round(df_bf["choke"].tail(20).values, 2)))
    distinct_sc = len(np.unique(np.round(df_sc["choke"].tail(20).values, 2)))
    avg_ms_bf = df_bf["solve_time_ms"].mean()
    avg_ms_sc = df_sc["solve_time_ms"].mean()

    print("\n--- Summary ---")
    lower_iae    = "Scipy SLSQP" if iae_sc < iae_bf else "Brute Force"
    lower_travel = "Scipy SLSQP" if travel_sc < travel_bf else "Brute Force"
    print(f"  Lower tracking error (IAE)  : {lower_iae}  "
          f"(BF={iae_bf:.1f}, SC={iae_sc:.1f})")
    print(f"  Less choke travel           : {lower_travel}  "
          f"(BF={travel_bf:.2f}%, SC={travel_sc:.2f}%)")
    print(f"  Distinct choke positions    : "
          f"Brute Force={distinct_bf}, Scipy={distinct_sc} (last 20 steps)")
    print(f"  Avg solve time              : "
          f"Brute Force={avg_ms_bf:.1f} ms, Scipy={avg_ms_sc:.1f} ms")


if __name__ == "__main__":
    np.random.seed(42)

    df_sweep = pd.read_csv(os.path.join(RESULTS, "step_test_sweep.csv"))
    model = WellModel()
    model.fit(df_sweep, verbose=False)

    cm = ConstraintManager()

    # separate simulator instances so each controller gets a clean well
    sim_bf = OilWellSimulator()
    sim_sc = OilWellSimulator()

    ctrl_bf = BruteForceMPC(model, cm, horizon=5)
    ctrl_sc = ScipyMPC(model, cm, horizon=5)

    runner_bf = ScenarioRunner(sim_bf, ctrl_bf, cm, model)
    runner_sc = ScenarioRunner(sim_sc, ctrl_sc, cm, model)

    print("\n=== Running Brute Force ===")
    df_bf = runner_bf.run("Brute Force", SCHEDULE, N_STEPS, initial_choke=0.0)
    df_bf.to_csv(os.path.join(RESULTS, "comparison_brute_force.csv"), index=False)

    print("\n=== Running Scipy SLSQP ===")
    df_sc = runner_sc.run("Scipy SLSQP", SCHEDULE, N_STEPS, initial_choke=0.0)
    df_sc.to_csv(os.path.join(RESULTS, "comparison_scipy.csv"), index=False)

    cmp_df = compute_metrics(df_bf, df_sc)
    comparison_plot(df_bf, df_sc, cmp_df,
                    os.path.join(RESULTS, "method_comparison.png"))
    grid_resolution_demo(df_bf, df_sc)
    print_summary(df_bf, df_sc)
