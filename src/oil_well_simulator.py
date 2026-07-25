import numpy as np


class OilWellSimulator:

    # constraint limits
    BHP_MIN = 2200.0
    WHP_MIN = 600.0
    WHP_MAX = 3000.0
    FLP_MAX = 700.0
    CHOKE_MIN = 0.0
    CHOKE_MAX = 100.0
    DELTA_CHOKE_MAX = 5.0

    def __init__(self):
        # reservoir and well parameters
        self.p_res = 3200.0   # reservoir pressure, psi
        self.PI = 0.08        # productivity index, bbl/hr/psi
        self.Cv = 2.5         # choke valve coefficient

        # time constants (hours)
        self.tau_Q   = 2.0
        self.tau_WHP = 1.5
        self.tau_FLP = 1.5
        self.tau_BHP = 3.5

        # noise std devs
        self.noise_Q   = 0.3
        self.noise_WHP = 1.5
        self.noise_FLP = 1.5
        self.noise_BHP = 3.0

        self.reset()

    def reset(self):
        # shut-in initial conditions: no flow, BHP = reservoir pressure
        self.q   = 0.0
        self.whp = 2800.0
        self.flp = 200.0
        self.bhp = 3200.0

    def _steady_state(self, u):
        # solve for steady-state values given choke opening u (0-100)
        # iterative fixed-point: start with a guess and converge
        q_ss = 0.0
        for _ in range(200):
            # flowline and wellhead pressures from q
            flp_ss = 200.0 + 0.06 * q_ss**2   # separator backpressure + friction
            bhp_ss = self.p_res - q_ss / self.PI  # darcy's law
            whp_ss = bhp_ss - 0.0008 * q_ss**2 - 400.0  # friction loss in tubing

            # choke flow equation
            dp = max(whp_ss - flp_ss, 0.0)
            q_new = self.Cv * (u / 100.0) * np.sqrt(dp)

            if abs(q_new - q_ss) < 1e-6:
                q_ss = q_new
                break
            q_ss = 0.5 * q_ss + 0.5 * q_new  # damped update to help convergence

        flp_ss = 200.0 + 0.06 * q_ss**2
        bhp_ss = self.p_res - q_ss / self.PI
        whp_ss = bhp_ss - 0.0008 * q_ss**2 - 400.0

        return q_ss, whp_ss, flp_ss, bhp_ss

    def step(self, choke_position):
        u = np.clip(choke_position, self.CHOKE_MIN, self.CHOKE_MAX)

        q_ss, whp_ss, flp_ss, bhp_ss = self._steady_state(u)

        # first-order lag dynamics: y_new = y + (1/tau) * (y_ss - y)
        self.q   += (1.0 / self.tau_Q)   * (q_ss   - self.q)
        self.whp += (1.0 / self.tau_WHP) * (whp_ss - self.whp)
        self.flp += (1.0 / self.tau_FLP) * (flp_ss - self.flp)
        self.bhp += (1.0 / self.tau_BHP) * (bhp_ss - self.bhp)

        # add measurement noise
        q_meas   = self.q   + np.random.normal(0, self.noise_Q)
        whp_meas = self.whp + np.random.normal(0, self.noise_WHP)
        flp_meas = self.flp + np.random.normal(0, self.noise_FLP)
        bhp_meas = self.bhp + np.random.normal(0, self.noise_BHP)

        # clip to non-negative
        q_meas   = max(q_meas,   0.0)
        whp_meas = max(whp_meas, 0.0)
        flp_meas = max(flp_meas, 0.0)
        bhp_meas = max(bhp_meas, 0.0)

        return (q_meas, whp_meas, flp_meas, bhp_meas)

    def get_constraints(self):
        return {
            "BHP_MIN":        self.BHP_MIN,
            "WHP_MIN":        self.WHP_MIN,
            "WHP_MAX":        self.WHP_MAX,
            "FLP_MAX":        self.FLP_MAX,
            "CHOKE_MIN":      self.CHOKE_MIN,
            "CHOKE_MAX":      self.CHOKE_MAX,
            "DELTA_CHOKE_MAX": self.DELTA_CHOKE_MAX,
        }


if __name__ == "__main__":
    np.random.seed(42)
    sim = OilWellSimulator()

    print(f"{'Choke':>8} {'Q (bbl/hr)':>12} {'WHP (psi)':>11} {'FLP (psi)':>11} {'BHP (psi)':>11}")
    print("-" * 58)

    for choke in range(0, 101, 10):
        sim.reset()
        # run 15 steps to let dynamics settle
        for _ in range(15):
            q, whp, flp, bhp = sim.step(choke)
        print(f"{choke:>8.1f} {q:>12.2f} {whp:>11.2f} {flp:>11.2f} {bhp:>11.2f}")
