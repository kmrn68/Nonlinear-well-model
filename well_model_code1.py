"""
Well Model Simulation and Analysis using CasADi
STABLE VERSION – Three-Stage Warm-Up with Gradual WGC Transition
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# =======================
# Configuration
# =======================

T_s      = 1
T_final  = 86400
EQUILIBRIUM_FILE = "equilibrium_points.npy"

R       = 8314
g       = 9.81
T       = 298
M       = 18

B_sw      = 84.9
rho_l     = 0.01 * (1000 * B_sw + 885 * (100 - B_sw))
alfa_gw   = 0.0189
rho_mres  = 968
L_t       = 1890
L_a       = 1100
L_fl      = 2928
L_r       = 1639
D_t       = 0.1524
D_a       = 0.1016
D_ss      = 0.15
theta     = 1.0004
H_vgl     = 1012
H_pdg     = 1827
H_t       = 1973
P_s       = 1.0e6
W_gc      = (132130 * 101325.0 * M / (293.0 * R) / 3600.0 / 24.0)
M_lstill  = 7.1098207999143210e2
C_g       = 2.3460789880222731e-5
C_out     = 5.8137935670717683e-3
V_eb      = 9.0159519351419036e1
E         = 3.5822558226225404e-2
K_w       = 1.0212053760436238e-3
K_a       = 1.7666249329670688e-4
K_r       = 2.4671647300003354e2
Z_op      = 0.2          # choke opening (0-1 or 0-100 depending on convention)
omega_u   = 1
P_r       = 2.5e7

# Nominal initial guess (may be infeasible – warm-up handles this)
X0 = np.array([
    8984.49438066,   # mga
    5698.96901788,   # mgt
    18832.03682361,  # mlt
    948.00058044,    # mgb
    2432.76378415,   # mgr
    8039.14607023    # mlr
], dtype=float)

# Warm-up durations
T_WARMUP_PRE  = 3600   # Stage 1: very large WGC  (find any feasible region)
T_WARMUP_MAIN = 7200   # Stage 2: transition + settle at real WGC
WGC_SCALE_PRE = 100.0   # multiplier for the pre-warm-up WGC
N_TRANSITION  = 300    # steps over which WGC is ramped (filter-like transition)


# =======================
# Model
# =======================

class WellModelCasADi:

    def __init__(self):

        self.mga  = ca.MX.sym('mga')
        self.mgt  = ca.MX.sym('mgt')
        self.mlt  = ca.MX.sym('mlt')
        self.mgb  = ca.MX.sym('mgb')
        self.mgr  = ca.MX.sym('mgr')
        self.mlr  = ca.MX.sym('mlr')
        self.Z    = ca.MX.sym('Z')
        self.W_gc = ca.MX.sym('W_gc')

        self.x = ca.vertcat(self.mga, self.mgt, self.mlt,
                            self.mgb, self.mgr, self.mlr)
        self.u = ca.vertcat(self.Z, self.W_gc)

        self.V_t  = np.pi * (D_t**2) * L_t / 4
        self.V_a  = np.pi * (D_a**2) * L_a / 4
        self.V_ss = (np.pi * (D_ss**2) * L_r / 4) + (np.pi * (D_ss**2) * L_fl / 4)
        self.A_ss = np.pi * (D_ss**2) / 4

        self.f          = self._create_dynamics()
        self.integrator = self._create_integrator()
        self.jacobian   = ca.Function('jacobian_f', [self.x, self.u],
                                      [ca.jacobian(self.f(self.x, self.u), self.x)])

    # =======================
    # Dynamics
    # =======================

    def _create_dynamics(self):

        V_gt   = ca.fmax(self.V_t - (self.mlt / rho_l), 1e-8)
        rho_gt = self.mgt / V_gt
        rho_mt = (self.mgt + self.mlt) / self.V_t

        den_rt = ca.fmax((omega_u * self.V_ss) -
                         ((self.mlr + M_lstill) / rho_l), 1e-8)

        P_rt = self.mgr * R * T / (M * den_rt)
        P_eb = self.mgb * R * T / (M * V_eb)
        P_rb = P_rt + (((self.mlr + M_lstill) * g * ca.sin(theta)) / self.A_ss)

        P_tt  = rho_gt * R * T / M
        P_tb  = P_tt  + (rho_mt  * g * H_vgl)
        P_pdg = P_tb  + (rho_mres * g * (H_pdg - H_vgl))
        P_bh  = P_pdg + (rho_mres * g * (H_t   - H_pdg))

        P_ai  = ((R * T / (self.V_a * M)) + (g * L_a) / self.V_a) * self.mga
        rho_ai = M * P_ai / (R * T)

        den_gr = ca.fmax(self.mgr + self.mlr, 1e-8)
        alfa_gr = self.mgr / den_gr
        alfa_lr = 1 - alfa_gr

        den_gt  = ca.fmax(self.mgt + self.mlt, 1e-8)
        alfa_gt = self.mgt / den_gt

        W_lout = alfa_lr * C_out * self.Z * ca.sqrt(ca.fmax(0, rho_l * (P_rt - P_s)))
        W_gout = alfa_gr * C_out * self.Z * ca.sqrt(ca.fmax(0, rho_l * (P_rt - P_s)))
        W_g    = C_g * ca.fmax(0, (P_eb - P_rb))
        W_whl  = K_w * ca.sqrt(ca.fmax(0, rho_l * (P_tt - P_rb))) * (1 - alfa_gt)
        W_whg  = K_w * ca.sqrt(ca.fmax(0, rho_l * (P_tt - P_rb))) * alfa_gt
        W_r    = K_r * ca.fmax(0, (1 - 0.2 * (P_bh / P_r) - (0.8 * (P_bh / P_r))**2))
        W_iv   = K_a * ca.sqrt(ca.fmax(0, rho_ai * (P_ai - P_tb)))

        xdot = ca.vertcat(
            self.W_gc - W_iv,
            W_r * alfa_gw  + W_iv - W_whg,
            W_r * (1 - alfa_gw) - W_whl,
            (1 - E) * W_whg - W_g,
            E * W_whg + W_g - W_gout,
            W_whl - W_lout
        )

        return ca.Function('f', [self.x, self.u], [xdot])

    def _create_integrator(self):
        dae = {'x': self.x, 'p': self.u, 'ode': self.f(self.x, self.u)}
        return ca.integrator('F', 'rk', dae, 0, T_s)

    # =======================
    # Safe Integration Step
    # =======================

    def _safe_step(self, x, Z, wgc):
        """Advance one step; return (new_x, ok)."""
        try:
            res = self.integrator(x0=x, p=[Z, wgc])
            xn  = res['xf'].full().flatten()
            if np.any(np.isnan(xn)) or np.any(np.isinf(xn)) or np.any(xn < 0):
                return x, False
            return xn, True
        except Exception:
            return x, False

    # =======================
    # Three-Stage Warm-Up
    # =======================

    def warmup(self, x0, Z, wgc_target,
               t_pre=T_WARMUP_PRE,
               t_main=T_WARMUP_MAIN,
               wgc_scale_pre=WGC_SCALE_PRE,
               n_transition=N_TRANSITION):
        """
        Three-stage warm-up to obtain a feasible initial condition.

        Stage 1 – Pre-warm-up  : run with a very large WGC (wgc_scale_pre × wgc_target).
                                  A high gas-injection rate makes the system tolerant of
                                  poor initial conditions (empirical rule of thumb).

        Stage 2 – Transition   : ramp WGC linearly from the large value back down to
                                  wgc_target over n_transition steps.  This avoids the
                                  abrupt change that causes crashes.

        Stage 3 – Main warm-up : settle at wgc_target for t_main seconds.
        """

        x         = x0.copy()
        wgc_pre   = wgc_scale_pre * wgc_target
        n_pre     = int(t_pre  / T_s)
        n_main    = int(t_main / T_s)

        # ---- Stage 1: pre-warm-up with large WGC ----
        print(f"  [Warmup] Stage 1 – pre-warm-up  (WGC = {wgc_pre:.4f}, {n_pre} steps)")
        ok_count = 0
        for _ in range(n_pre):
            x, ok = self._safe_step(x, Z, wgc_pre)
            if ok:
                ok_count += 1
        print(f"  [Warmup] Stage 1 done  ({ok_count}/{n_pre} steps succeeded)")
        

        # ---- Stage 2: gradual transition WGC_pre → WGC_target ----
        print(f"  [Warmup] Stage 2 – ramp WGC  ({wgc_pre:.4f} → {wgc_target:.4f}, {n_transition} steps)")
        wgc_ramp = np.linspace(wgc_pre, wgc_target, n_transition)
        ok_count = 0
        for wgc in wgc_ramp:
            x, ok = self._safe_step(x, Z, wgc)
            if ok:
                ok_count += 1
        print(f"  [Warmup] Stage 2 done  ({ok_count}/{n_transition} steps succeeded)")
        
        # ---- Stage 3: main warm-up at target WGC ----
        print(f"  [Warmup] Stage 3 – main warm-up (WGC = {wgc_target:.4f}, {n_main} steps)")
        ok_count = 0
        
        for _ in range(n_main):
            x, ok = self._safe_step(x, Z, wgc_target)
            if ok:
                ok_count += 1
        print(f"  [Warmup] Stage 3 done  ({ok_count}/{n_main} steps succeeded)")
        return x

    # =======================
    # Equilibrium Finder
    # =======================

    def find_equilibrium(self, Z, wgc):
        print("  [EQ] Running warm-up to obtain feasible initial guess ...")
        x_guess = self.warmup(X0, Z, wgc)
        print(f"  [EQ] Warm-up result: {np.round(x_guess, 2)}")
        

        opti  = ca.Opti()
        x_eq  = opti.variable(6)

        opti.subject_to(self.f(x_eq, [Z, wgc]) == 0)
        opti.subject_to(x_eq >= 1e-6)
        opti.subject_to(x_eq <= 1e8)

        opti.set_initial(x_eq, x_guess)
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False})
        
        sol = opti.solve()
        x_eq_val = sol.value(x_eq)
        print(f"  [EQ] Equilibrium found: {np.round(x_eq_val, 2)}")
        
        return x_eq_val
        

        # try:
        #     sol = opti.solve()
        #     x_eq_val = sol.value(x_eq)
        #     print(f"  [EQ] Equilibrium found: {np.round(x_eq_val, 2)}")
        #     return x_eq_val
        # except Exception as e:
        #     print(a)
        #     print(f"  [EQ] IPOPT failed ({e}). Using warm-up endpoint as fallback.")
        #     return x_guess

    # =======================
    # Simulation
    # =======================

    def simulate(self, x0, Z, wgc):
        n_steps = int(T_final / T_s)
        X = np.zeros((n_steps + 1, 6))
        X[0, :] = x0

        crashed_at = None
        for i in range(n_steps):
            xn, ok = self._safe_step(X[i, :], Z, wgc)
            X[i + 1, :] = xn
            if not ok and crashed_at is None:
                crashed_at = i
                print(f"  [SIM] Warning: simulation became infeasible at step {i} (t={i*T_s} s). "
                      f"Holding last valid state.")

        if crashed_at is None:
            print("  [SIM] Simulation completed without issues.")
            
        
        return X


# =======================
# MAIN
# =======================

def main():
    model = WellModelCasADi()

    Z_val   = Z_op   # choke opening
    wgc_val = W_gc   # nominal gas-lift injection rate

    print(f"\nOperating point:  Z = {Z_val:.4f},  W_gc = {wgc_val:.6f} kg/s\n")

    print("Step 1 – Finding equilibrium ...")
    eq = model.find_equilibrium(Z_val, wgc_val)
    np.save(EQUILIBRIUM_FILE, eq)
    print(f"Equilibrium saved to '{EQUILIBRIUM_FILE}'\n")

    print("Step 2 – Open-loop simulation from equilibrium ...")
    
    Z_val = 0.11
    X = model.simulate(eq, Z_val, wgc_val)

    t = np.linspace(0, T_final, int(T_final / T_s) + 1)

    state_labels = ['mga', 'mgt', 'mlt', 'mgb', 'mgr', 'mlr']
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for idx, ax in enumerate(axes.flat):
        ax.plot(t / 3600, X[:, idx])
        ax.set_ylabel(f'{state_labels[idx]} [kg]')
        ax.grid(True)
    for ax in axes[-1, :]:
        ax.set_xlabel('Time [h]')
    fig.suptitle('Open-Loop Simulation – State Trajectories')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
