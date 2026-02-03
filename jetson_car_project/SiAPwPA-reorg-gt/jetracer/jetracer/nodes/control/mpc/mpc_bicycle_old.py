import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
from sensor_msgs.msg import Image


def linear_mpc_control(e_init, v, L, dt, N, ref_ys=None, q_y=10.0, q_psi=1.0, r=1e-2,
                       delta_bounds=(-0.5, 0.5)):
    """Solve MPC for a horizon N given initial error state e_init=[e_y, e_psi].

    ref_ys optionally gives reference lateral positions per horizon step
    (not used directly here since we target zero lateral error after
    shifting). We use a linearized discrete model for prediction.
    Returns first steering angle delta (radians) and the full optimal
    steering sequence.
    """
    e_y0, e_psi0 = e_init

    def simulate_errors(deltas):
        e_y = e_y0
        e_psi = e_psi0
        traj = []
        for k in range(N):
            delta_k = deltas[k]
            e_y = e_y + dt * v * e_psi
            e_psi = e_psi + dt * v / L * delta_k
            traj.append((e_y, e_psi))
        return np.array(traj)

    def cost(deltas):
        traj = simulate_errors(deltas)
        e_y_traj = traj[:, 0]
        e_psi_traj = traj[:, 1]
        J = q_y * np.sum(e_y_traj ** 2) + q_psi * np.sum(e_psi_traj ** 2) + r * np.sum(deltas ** 2)
        return J

    x0 = np.zeros(N)
    bounds = [delta_bounds] * N
    res = minimize(cost, x0, bounds=bounds, method="SLSQP", options={"maxiter": 200, "ftol":1e-4})
    deltas_opt = res.x if res.success else x0
    return deltas_opt[2], deltas_opt


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi 


def compute_steering_from_binary(poly, v=4.0, L=2.5, dt=0.1, horizon=10,):
    """Compute steering angle (delta, radians) from a binary bird-eye image.

    Steps:
      - extract centerline points (meters) using `extract_points_from_binary`
      - fit a reference polynomial
      - compute lateral and heading error at vehicle origin (x=0, y=0, psi=0)
      - run linear_mpc_control and return first steering command

    Returns:
      delta0 (float): steering angle in radians (first control returned by MPC).
      If not enough data to fit a polynomial, returns 0.0.
        Extra:
            If `img_publisher` (ROS2 publisher for sensor_msgs/Image) is passed,
            publishes the binary image on that topic (encoding mono8).
    """

    if not hasattr(poly, "__len__"):
        return 0.0
    cv2.imwrite("poly.png", poly)  

    x_fwd = 0.0
    y_ref = np.polyval(poly, x_fwd)
    dy_ref = np.polyval(np.polyder(poly), x_fwd)
    psi_ref = np.arctan2(dy_ref, 1.0)

    e_y = 0.0 - y_ref
    e_psi = wrap_angle(0.0 - psi_ref)

    delta0, seq = linear_mpc_control((e_y, e_psi), v, L, dt, horizon)
    return float(delta0)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2
from sensor_msgs.msg import Image


def linear_mpc_control(e_init, v, L, dt, N, ref_ys=None, q_y=10.0, q_psi=1.0, r=1e-2,
                       delta_bounds=(-0.5, 0.5)):
    """Solve MPC for a horizon N given initial error state e_init=[e_y, e_psi].

    ref_ys optionally gives reference lateral positions per horizon step
    (not used directly here since we target zero lateral error after
    shifting). We use a linearized discrete model for prediction.
    Returns first steering angle delta (radians) and the full optimal
    steering sequence.
    """
    e_y0, e_psi0 = e_init

    def simulate_errors(deltas):
        e_y = e_y0
        e_psi = e_psi0
        traj = []
        for k in range(N):
            delta_k = deltas[k]
            e_y = e_y + dt * v * e_psi
            e_psi = e_psi + dt * v / L * delta_k
            traj.append((e_y, e_psi))
        return np.array(traj)

    def cost(deltas):
        traj = simulate_errors(deltas)
        e_y_traj = traj[:, 0]
        e_psi_traj = traj[:, 1]
        J = q_y * np.sum(e_y_traj ** 2) + q_psi * np.sum(e_psi_traj ** 2) + r * np.sum(deltas ** 2)
        return J

    x0 = np.zeros(N)
    bounds = [delta_bounds] * N
    res = minimize(cost, x0, bounds=bounds, method="SLSQP", options={"maxiter": 200, "ftol":1e-4})
    deltas_opt = res.x if res.success else x0
    return deltas_opt[2], deltas_opt


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi 


def compute_steering_from_binary(poly, v=4.0, L=2.5, dt=0.1, horizon=10,):
    """Compute steering angle (delta, radians) from a binary bird-eye image.

    Steps:
      - extract centerline points (meters) using `extract_points_from_binary`
      - fit a reference polynomial
      - compute lateral and heading error at vehicle origin (x=0, y=0, psi=0)
      - run linear_mpc_control and return first steering command

    Returns:
      delta0 (float): steering angle in radians (first control returned by MPC).
      If not enough data to fit a polynomial, returns 0.0.
        Extra:
            If `img_publisher` (ROS2 publisher for sensor_msgs/Image) is passed,
            publishes the binary image on that topic (encoding mono8).
    """

    if not hasattr(poly, "__len__"):
        return 0.0
    cv2.imwrite("poly.png", poly)  

    x_fwd = 0.0
    y_ref = np.polyval(poly, x_fwd)
    dy_ref = np.polyval(np.polyder(poly), x_fwd)
    psi_ref = np.arctan2(dy_ref, 1.0)

    e_y = 0.0 - y_ref
    e_psi = wrap_angle(0.0 - psi_ref)

    delta0, seq = linear_mpc_control((e_y, e_psi), v, L, dt, horizon)
    return float(delta0)

