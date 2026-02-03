

import numpy as np
import casadi as ca

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LinearBicycleMPC:
    def __init__(self, v=4.0, L=2.5, dt=0.1, N=10, q_y=10.0, q_psi=1.0, r=1e-2, delta_max=0.5):
        self.v = v
        self.L = L
        self.dt = dt
        self.N = N
        self.q_y = q_y
        self.q_psi = q_psi
        self.r = r
        self.delta_max = delta_max
        self.setup_solver()
        self.last_u = np.zeros(N)

    def setup_solver(self):
        opti = ca.Opti()

        # States [e_y, e_psi] over N+1 steps
        x = opti.variable(2, self.N + 1)
        e_y = x[0, :]
        e_psi = x[1, :]

        # Controls delta over N steps
        u = opti.variable(self.N)

        # Cost (matches original structure - cost on predicted states after each step)
        cost = 0
        for k in range(self.N):
            cost += self.q_y * e_y[k + 1]**2 + self.q_psi * e_psi[k + 1]**2 + self.r * u[k]**2

        opti.minimize(cost)

        # Initial state parameter
        p_x0 = opti.parameter(2)
        opti.subject_to(x[:, 0] == p_x0)

        # Linearized discrete dynamics
        A = ca.DM([[1.0, self.dt * self.v],
                   [0.0, 1.0]])
        B = ca.DM([[0.0],
                   [self.dt * self.v / self.L]])

        for k in range(self.N):
            opti.subject_to(x[:, k + 1] == A @ x[:, k] + B * u[k])

        # Control bounds
        opti.subject_to(opti.bounded(-self.delta_max, u, self.delta_max))

        # Solver options (silent)
        opts = {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": 0}
        opti.solver("ipopt", opts)

        self.opti = opti
        self.u = u
        self.p_x0 = p_x0

    def control(self, e_init):
        e_init = np.array(e_init).reshape(2,)
        self.opti.set_value(self.p_x0, e_init)

        # Warm-start: shift previous solution and repeat last input
        u_warm = np.append(self.last_u[1:], self.last_u[-1])
        self.opti.set_initial(self.u, u_warm)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.u)
            # Shift for next warm-start
            self.last_u = np.append(u_opt[1:], u_opt[-1])
            return float(u_opt[0])
        except Exception as e:
            print(f"MPC solver failed: {e}")
            return 0.0


# Global singleton controller (fixed parameters matching your original defaults)
# mpc_controller = LinearBicycleMPC(v=4.0, L=2.5, dt=0.1, N=10)

mpc_controller = LinearBicycleMPC(
    v=0.3,          # Start with 0.5 m/s (matches your ~0.3 commanded speed; tune up/down)
    L=0.25,         # JetRacer wheelbase ≈0.25 m (measure yours exactly)
    dt=0.1,         # Keep 10 Hz control rate
    N=10,           # Slightly longer horizon helps stability
    q_y=5.0,       # Increase: stronger lateral error correction
    q_psi=1.0,      # Slight increase: better heading alignment
    r=0.5,         # Decrease: allow more aggressive steering (less input penalty)
    delta_max=0.4   # Keep your bound
)


def compute_steering_from_binary(poly_coeffs, lookahead_x=0.0):
    """
    Compute steering angle from fitted polynomial coefficients.
    poly_coeffs: output from np.polyfit (highest degree first)
    lookahead_x: optional lookahead distance in meters (0.0 matches original behavior)
    """
    if not hasattr(poly_coeffs, "__len__") or len(poly_coeffs) == 0:
        return 0.0

    y_ref = np.polyval(poly_coeffs, lookahead_x)
    dy_ref = np.polyval(np.polyder(poly_coeffs), lookahead_x)
    psi_ref = np.arctan2(dy_ref, 1.0)

    e_y = -y_ref
    e_psi = wrap_angle(-psi_ref)

    delta = mpc_controller.control([e_y, e_psi])
    return delta
