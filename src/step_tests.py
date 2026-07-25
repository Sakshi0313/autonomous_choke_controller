import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# resolve paths relative to this file so it works regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, HERE)
from oil_well_simulator import OilWellSimulator

# grab constraint limits from class
BHP_MIN = OilWellSimulator.BHP_MIN
WHP_MIN = OilWellSimulator.WHP_MIN
WHP_MAX = OilWellSimulator.WHP_MAX
FLP_MAX = OilWellSimulator.FLP_MAX


def run_sequence(sim, choke_sequence, reset_first=True):
    """Run a list of (choke, hold_hours) pairs and return a DataFrame."""
    if reset_first:
        sim.reset()

    rows = []
    t = 0
    for choke, hours in choke_sequence:
        for _ in range(hours):
            q, whp, flp, bhp = sim.step(choke)
            rows.append({"time": t, "choke": choke, "Q": q, "WHP": whp, "FLP": flp, "BHP": bhp})
            t += 1

    return pd.DataFrame(rows)


def plot_step_test(df, title, filename):
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].step(df["time"], df["choke"], where="post", color="black")
    axes[0].set_ylabel("Choke (%)")

    axes[1].plot(df["time"], df["Q"], color="steelblue")
    axes[1].set_ylabel("Q (bbl/hr)")

    axes[2].plot(df["time"], df["WHP"], color="darkorange")
    axes[2].axhline(WHP_MIN, color="red", linestyle="--", alpha=0.8, label=f"WHP_MIN={WHP_MIN}")
    axes[2].axhline(WHP_MAX, color="red", linestyle="--", alpha=0.8, label=f"WHP_MAX={WHP_MAX}")
    axes[2].set_ylabel("WHP (psi)")
    axes[2].legend(fontsize=8)

    axes[3].plot(df["time"], df["FLP"], color="green")
    axes[3].axhline(FLP_MAX, color="red", linestyle="--", alpha=0.8, label=f"FLP_MAX={FLP_MAX}")
    axes[3].set_ylabel("FLP (psi)")
    axes[3].legend(fontsize=8)

    axes[4].plot(df["time"], df["BHP"], color="purple")
    axes[4].axhline(BHP_MIN, color="red", linestyle="--", alpha=0.8, label=f"BHP_MIN={BHP_MIN}")
    axes[4].set_ylabel("BHP (psi)")
    axes[4].legend(fontsize=8)
    axes[4].set_xlabel("Time (hours)")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> results/{filename}")


def extract_steady_state(df):
    # take last 5 rows at each choke position and average them
    ss_rows = []
    for choke, group in df.groupby("choke", sort=True):
        tail = group.tail(5)
        ss_rows.append({
            "choke": choke,
            "Q":     tail["Q"].mean(),
            "WHP":   tail["WHP"].mean(),
            "FLP":   tail["FLP"].mean(),
            "BHP":   tail["BHP"].mean(),
        })
    return pd.DataFrame(ss_rows)


def validate_physics(ss_df):
    q   = ss_df["Q"].values
    bhp = ss_df["BHP"].values
    whp = ss_df["WHP"].values
    flp = ss_df["FLP"].values

    print("\n--- Physics Validation ---")

    # 1. Q vs BHP should be negatively correlated (darcy's law)
    corr_q_bhp = np.corrcoef(q, bhp)[0, 1]
    status = "PASS" if corr_q_bhp < 0 else "FAIL"
    print(f"[{status}] Q-BHP correlation = {corr_q_bhp:.3f}  (expect negative, Darcy's Law)")

    # 2. Q vs (BHP - WHP) should be positive — more drawdown across tubing means more flow
    tubing_dp = bhp - whp
    corr_q_dp = np.corrcoef(q, tubing_dp)[0, 1]
    status = "PASS" if corr_q_dp > 0 else "FAIL"
    print(f"[{status}] Q vs (BHP-WHP) correlation = {corr_q_dp:.3f}  (expect positive, tubing friction)")

    # 3. Q vs FLP should be positive — more flow means more flowline friction
    corr_q_flp = np.corrcoef(q, flp)[0, 1]
    status = "PASS" if corr_q_flp > 0 else "FAIL"
    print(f"[{status}] Q-FLP correlation = {corr_q_flp:.3f}  (expect positive, Darcy-Weisbach)")

    # 4. fit BHP = m*Q + c, PI = -1/m, p_res = intercept
    # only use rows where Q > 0 to avoid the shut-in point skewing the fit
    mask = q > 0.5
    coeffs = np.polyfit(q[mask], bhp[mask], 1)
    m, c = coeffs
    pi_est   = -1.0 / m
    pres_est = c
    print(f"\n  Linear fit BHP = {m:.4f}*Q + {c:.2f}")
    print(f"  Estimated PI         = {pi_est:.4f} bbl/hr/psi  (true = 0.08)")
    print(f"  Estimated P_reservoir = {pres_est:.2f} psi        (true = 3200)")


if __name__ == "__main__":
    np.random.seed(0)
    sim = OilWellSimulator()

    # ---- Test 1: Steady-state sweep (upward) ----
    print("\n=== Test 1: Steady-State Sweep ===")
    sweep_seq = [(c, 25) for c in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    # don't reset between positions — let it transition naturally
    df_sweep = run_sequence(sim, sweep_seq, reset_first=True)
    df_sweep.to_csv(os.path.join(RESULTS, "step_test_sweep.csv"), index=False)
    print(f"  rows: {len(df_sweep)}, saved -> results/step_test_sweep.csv")
    plot_step_test(df_sweep, "Steady-State Sweep (0→100%)", "step_test_sweep.png")

    # ---- Test 2: Dynamic step test ----
    print("\n=== Test 2: Dynamic Step Test ===")
    dynamic_seq = [(20, 25), (55, 30), (20, 25), (80, 30), (40, 25)]
    df_dynamic = run_sequence(sim, dynamic_seq, reset_first=True)
    df_dynamic.to_csv(os.path.join(RESULTS, "step_test_dynamic.csv"), index=False)
    print(f"  rows: {len(df_dynamic)}, saved -> results/step_test_dynamic.csv")
    plot_step_test(df_dynamic, "Dynamic Step Test", "step_test_dynamic.png")

    # ---- Test 3: Reverse sweep ----
    print("\n=== Test 3: Reverse Sweep (hysteresis check) ===")
    rev_seq = [(c, 25) for c in [100, 80, 60, 40, 20, 0]]
    df_reverse = run_sequence(sim, rev_seq, reset_first=True)
    df_reverse.to_csv(os.path.join(RESULTS, "step_test_reverse.csv"), index=False)
    print(f"  rows: {len(df_reverse)}, saved -> results/step_test_reverse.csv")
    plot_step_test(df_reverse, "Reverse Sweep (100→0%)", "step_test_reverse.png")

    # ---- Steady-state extraction and physics validation ----
    print("\n=== Steady-State Extraction (from sweep) ===")
    ss_df = extract_steady_state(df_sweep)
    print(ss_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    validate_physics(ss_df)
