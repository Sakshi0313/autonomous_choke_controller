# Autonomous Production Choke Controller for a Naturally Flowing Oil Well

Model Predictive Control (MPC) solution that autonomously optimises the
production choke of a single naturally flowing oil well — maximising oil
production while guaranteeing that Wellhead Pressure (WHP), Flowline
Pressure (FLP) and Bottom Hole Pressure (BHP) never leave their safe
operating envelope.

**Headline result: 150 control steps across 3 scenarios, zero constraint
violations, 99.3% optimiser convergence.**

---

## 1. Problem

A production choke is the only manipulated variable on a naturally flowing
well. Opening it increases production but reduces BHP (reservoir drawdown),
reduces WHP (tubing friction) and raises FLP (flowline friction). Operators
adjust chokes manually across large well counts, which does not scale and
produces inconsistent operation.

The controller must:
- Track a target oil rate whenever it is safely achievable
- Settle at the maximum safe rate when the target is infeasible
- Never violate WHP, FLP or BHP limits
- Respect a choke ramp-rate limit of ±5% per 1-hour control interval

## 2. Note on the Simulator

The problem statement referenced a provided simulator. HirePro confirmed by
email that no simulator would be supplied and that an open-source equivalent
should be used. Rather than adapt a full reservoir simulator, a
**first-principles well simulator was built from published petroleum
production engineering relationships** (Guo, *Petroleum Production
Engineering*, Ch. 5). This is included as a deliverable in
`src/oil_well_simulator.py`.

## 3. Architecture

![System Architecture](https://github.com/user-attachments/assets/your-generated-url)

## 4. Physics Basis

| Simulator relationship | Physical law | Reference |
|---|---|---|
| Q from choke opening and pressure drop | Choke flow equation, q proportional to sqrt(dP) | Guo Eq. 5.3 |
| BHP = P_res - Q/PI | Darcy's Law / straight-line IPR | Guo Sec. 5.1 |
| WHP = BHP - tubing friction - hydrostatic head | Tubing pressure loss | Guo Sec. 5.1 |
| FLP rises with Q squared | Darcy-Weisbach pipe friction | Standard pipe flow |

Constraint justifications, from Guo Sec. 5.1 — chokes exist to *"limit
production rates for regulations, protect surface equipment from slugging,
avoid sand problems due to high drawdown, and control flow rate to avoid
water or gas coning."*

| Constraint | Value | Physical reason |
|---|---|---|
| BHP_min | 2200 psi | Sand production and water coning avoidance (31% max drawdown) |
| WHP_min | 600 psi | Slugging / unstable flow regime prevention |
| WHP_max | 3000 psi | Wellhead equipment rating |
| FLP_max | 700 psi | Flowline integrity |
| Choke ramp | ±5%/hr | Valve mechanical limit |

## 5. System Identification Results

Open-loop step tests: 275-row steady-state sweep, 135-row dynamic step
test, 150-row reverse sweep (no hysteresis detected).

**Reservoir parameters recovered from data alone:**

| Parameter | True | Identified | Error |
|---|---|---|---|
| Productivity Index | 0.0800 bbl/hr/psi | 0.0798 | 0.25% |
| Reservoir Pressure | 3200 psi | 3201.63 | 0.05% |

**Grey-box model — quadratic steady state, first-order dynamics:**

| Output | Fitted steady-state model | R² | RMSE | tau (hr) |
|---|---|---|---|---|
| Q | 0.296 + 1.2578u - 0.004253u² | 1.0000 | 0.19 bbl/hr | 2.00 |
| WHP | 2799.45 - 15.8883u + 0.053882u² | 1.0000 | 2.09 psi | 1.10 |
| FLP | 184.49 + 2.7049u + 0.017956u² | 0.9956 | 9.57 psi | 1.20 |
| BHP | 3198.20 - 15.7775u + 0.053487u² | 0.9999 | 2.43 psi | 3.70 |

tau_BHP is the largest, consistent with reservoir pressure diffusion being
slower than surface hydraulics.

**Physics validation from step-test data:**
- corr(Q, BHP) = -1.000 — Darcy's Law confirmed
- corr(Q, FLP) = +0.966 — Darcy-Weisbach confirmed (non-unity due to quadratic form)
- corr(Q, BHP-WHP) = +0.858 — tubing friction confirmed (hydrostatic head dominates)

## 6. Safe Operating Envelope

| Quantity | Value |
|---|---|
| Physical max production (BHP = 2200) | ~80.0 bbl/hr at ~91.9% choke |
| Controller max (2% operating margin) | 76.21 bbl/hr at 84.5% choke |
| Sole binding constraint | BHP_min |

FLP peaks at 89% of its limit and WHP stays well inside bounds across the
full choke range, so BHP is the only constraint that ever becomes active —
consistent with drawdown-limited operation.

## 7. Results

| Scenario | Target | Achieved | Violations | Settling | Max move | Optimiser |
|---|---|---|---|---|---|---|
| A — Startup | 60 bbl/hr | 59.99 | **0** | 12 steps | 5.00% | 100% |
| B — Tracking | 50 -> 70 bbl/hr | 50.05 / 69.85 | **0** | 9 / 5 steps | 5.00% | 100% |
| C — Infeasible | 100 bbl/hr | 76.14 (max safe) | **0** | n/a | 5.00% | 98% |

**Scenario C** requested 100 bbl/hr. The steady-state optimiser identified
76.21 bbl/hr as the maximum safe rate, named BHP_min as the binding
constraint, and the closed loop delivered 76.14 bbl/hr — capturing 99.9% of
achievable production. Minimum observed BHP was 2240.8 psi against a hard
limit of 2200 psi, retaining 40.8 psi of true margin.

**Solver performance across all 150 control steps:**
- SLSQP convergence: 149/150 (99.3%)
- Fallback activations: 1
- Average solve time: 417 ms against a 3600 s control interval (0.012% utilisation)
- Average iterations: 4.9 (warm-started)

## 8. Method Comparison — Why an Optimiser

Identical scenario (50 -> 70 bbl/hr, 60 steps) run through both controllers:

| Metric | Brute Force | Scipy SLSQP |
|---|---|---|
| Total IAE | 311.12 | 306.65 |
| Steady-state error | 0.130 bbl/hr | 0.119 bbl/hr |
| Total choke travel | 105.00% | **95.87%** |
| Distinct choke positions (last 20 steps) | 5 | **18** |
| Average solve time | **14.7 ms** | 461.2 ms |
| Constraint violations | 0 | 0 |

Tracking performance is equivalent — the 1.4% IAE difference is within
measurement noise. The real difference is **search resolution**. Brute force
produced only 5 distinct choke values in steady state, all multiples of its
0.5% search grid (73.0, 73.5, 74.0, 74.5, 75.0). SLSQP produced 18
continuous values (73.27, 73.35, 73.46, ...). Despite using more distinct
positions, SLSQP required **8.7% less total choke travel**, because it makes
many small precise corrections instead of fewer quantised jumps. Less valve
travel means less wear.

Brute force is 31× faster and is retained as the fallback controller,
guaranteeing a valid safe action if the optimiser ever fails.

## 9. Design Decisions

**Grey-box over black-box.** A pure reactive controller cannot see that
opening the choke now will violate BHP three hours later. The identified
model lets the MPC check the full 10-step predicted trajectory before
committing to any move — this is what makes zero violations achievable.

**Trajectory rejection, not penalisation.** Any candidate whose predicted
trajectory violates a constraint at any horizon step is discarded entirely
rather than soft-penalised. This makes safety a hard guarantee.

**2% operating margin.** Constraints are tightened by 2% inside the
controller. This absorbs measurement noise and prevents chattering on the
constraint boundary, at the cost of 3.8 bbl/hr of theoretical production.

**Steady-state optimiser above the dynamic controller.** Without it, an
infeasible target makes the optimiser aim at a non-existent operating point,
every trajectory fails the safety check, and the controller stalls
conservatively. Adding this layer recovered 5.6 bbl/hr in Scenario C and
raised optimiser convergence from 30% to 98%. This mirrors the steady-state
optimisation layer found in industrial MPC packages.

**Fallback retained despite never firing in normal operation.** Defence in
depth — the system must produce a valid control action even under solver
failure.

## 10. Development Log — Issues Found and Fixed

| Issue | Symptom | Fix | Result |
|---|---|---|---|
| Wrong trajectory validated | 24% spurious fallbacks | Validate the actual SLSQP solution, not a rebuilt sequence | 76% -> 100% convergence |
| Infeasible target stalled controller | Scenario C settled at 70.59 vs 76.21 achievable | Added steady-state target optimiser | 76.14 bbl/hr, 98% convergence |
| Envelope rescanned every step | 4381 ms average solve | Cached the feasibility solution | 417 ms, 10× faster |

## 11. Repository Structure

## 12. How to Run

pip install numpy pandas scipy matplotlib
cd src
python step_tests.py # generates step-test data and plots
python well_model.py # fits and validates the model
python constraint_manager.py # safe operating envelope
python scenario_runner.py # all three scenarios (~2 min)
python method_comparison.py # optimiser comparison (~1 min)

## 13. Assumptions and Limitations

- Single naturally flowing well, single choke, no artificial lift
- Constant reservoir properties, GOR and water cut (per problem statement)
- The simulator uses a subcritical choke flow formulation. Observed FLP/WHP
  ratios span 0.07 to 0.36, below the critical pressure ratio of ~0.55
  (Guo Eq. 5.1), indicating the well operates in critical flow. Explicit
  critical/subcritical regime switching is identified as future work.
- The MPC prediction model is fitted to this simulator; plant-model mismatch
  is not exercised

## 14. Future Work

- Explicit critical/subcritical flow regime switching (Guo Eq. 5.1)
- Extended Kalman Filter for reservoir decline and plant-model mismatch
- Gilbert correlation as an alternative critical-flow model (reported
  accurate to 6.19% mean error over 155 well tests, Al-Attar & Abdul-Majeed 1988)
- Economic MPC weighting oil price against valve maintenance cost
- Multi-well extension with shared flowline network constraints
