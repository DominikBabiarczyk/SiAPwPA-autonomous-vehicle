"""mpc_bicycle.py

Simple MPC for a kinematic bicycle-like (linearized) model.

Assumptions / Inputs:
- You are given a bird's-eye binary matrix where '1' pixels indicate
  points along the center of the road. This script simulates that
  matrix (as an example) and shows how you would extract a polynomial
  reference from it.
- The vehicle uses a simple linearized lateral-error model for MPC:
    e_y_{k+1} = e_y_k + dt * v * e_psi_k
    e_psi_{k+1} = e_psi_k + dt * v/L * delta_k
  with Euler integration for the actual vehicle state using the
  full kinematic equations.

Run with: python3 mpc_bicycle.py

Dependencies: numpy, matplotlib, scipy
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
try:
    import cv2
except Exception:
    cv2 = None
try:
    from sensor_msgs.msg import Image
except Exception:
    Image = None  # allows use outside ROS2 context



def binary_path_demo(width=200, height=400, lane_offset=0.0):
    """Create a demo binary bird-eye image with a center line.

    Image coordinates: x increases to the right, y increases downward.
    We'll create a centerline represented by some polynomial y = f(x)
    but we then rotate/translate so that 'forward' is up (decreasing y)
    and the vehicle sits near bottom-center.
    Returns a binary 2D numpy array (height x width).
    """
    img = np.zeros((height, width), dtype=np.uint8)
    # Build x coordinates in meters (assume pixel scale 0.05 m/pixel)
    px2m = 0.05
    xs = np.arange(0, height) * px2m  # forward distance
    # Create a gentle curved path in lateral coordinate (meters)
    # use a polynomial in forward coordinate
    a, b, c = 0.0008, -0.01, 0.0
    ys_m = a * (xs ** 2) + b * xs + c + lane_offset
    # Convert to pixel coordinates with origin bottom-center
    center_x = width // 2
    for i, xm in enumerate(xs):
        y_pix = i  # forward along rows
        x_m = ys_m[i]
        x_pix = int(center_x + x_m / px2m)
        if 0 <= x_pix < width and 0 <= y_pix < height:
            img[y_pix, x_pix] = 1
    return img, px2m


def extract_points_from_binary(binary_img, px2m):
    """Return arrays of (x_forward_m, y_lateral_m) from binary image.

    We assume image origin is top-left; forward is +y (downwards in image).
    Vehicle is placed at bottom center (y = height-1), so we convert
    image row to forward distance measured from vehicle position.
    """
    h, w = binary_img.shape
    ys, xs = np.where(binary_img > 0)
    # forward distance from vehicle (vehicle at bottom row)
    x_forward_px = (h - 1) - ys
    x_forward_m = x_forward_px * px2m
    center_x = w // 2
    y_lateral_m = (xs - center_x) * px2m
    # keep points sorted by forward distance
    order = np.argsort(x_forward_m)
    return x_forward_m[order], y_lateral_m[order]


def fit_reference_poly(x, y, deg=3):
    if len(x) < deg + 1:
        deg = max(1, len(x) - 1)
    coeffs = np.polyfit(x, y, deg)
    return coeffs


def draw_points_on_image(binary_img, x_pts, y_pts, px2m, poly=None,
                         max_poly_x=20.0, point_color=(0, 0, 255),
                         poly_color=(0, 255, 0), origin_color=(255, 0, 0)):
    """Build a BGR visualization image from a binary bird-eye image.

    Draws:
      - original binary pixels (gray)
      - extracted path points (small circles, point_color)
      - fitted polynomial (poly_color) if coeffs provided
      - vehicle origin (bottom-center) as a triangle/origin_color
      - heading reference arrow at origin if poly provided

    Parameters:
      binary_img : 2D np.ndarray (binary 0/1 or 0/255)
      x_pts, y_pts : arrays of forward (m) and lateral (m) coordinates
      px2m : meters per pixel scale used for extraction
      poly : optional polynomial coeffs (highest degree first)
      max_poly_x : forward distance (m) up to which polynomial is drawn
      *_color : BGR colors

    Returns: HxWx3 uint8 BGR image.
    """
    if binary_img is None:
        return None
    img = np.asarray(binary_img)
    if img.ndim != 2:
        # reduce to first channel if multi-channel
        img = img[..., 0]
    # normalize to 0/255 uint8
    if img.dtype != np.uint8:
        img = (img > 0).astype(np.uint8) * 255
    else:
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

    # make 3-channel BGR base
    if cv2 is not None:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = np.stack([img, img, img], axis=-1)

    h, w = base.shape[:2]
    center_x = w // 2

    # Draw path points
    for xf, yf in zip(x_pts, y_pts):
        col = int(round(center_x + yf / px2m))
        row = int((h - 1) - round(xf / px2m))
        if 0 <= row < h and 0 <= col < w:
            if cv2 is not None:
                cv2.circle(base, (col, row), 3, point_color, -1)
            else:
                r0 = max(0, row - 2); r1 = min(h, row + 3)
                c0 = max(0, col - 2); c1 = min(w, col + 3)
                base[r0:r1, c0:c1, :] = point_color

    # Draw polynomial curve if available
    if poly is not None and len(x_pts) > 0:
        xs_curve = np.linspace(0.0, max(max_poly_x, min(max_poly_x, np.max(x_pts))), 300)
        ys_curve = np.polyval(poly, xs_curve)
        prev_pixel = None
        for xf, yf in zip(xs_curve, ys_curve):
            col = int(round(center_x + yf / px2m))
            row = int((h - 1) - round(xf / px2m))
            if 0 <= row < h and 0 <= col < w:
                if cv2 is not None:
                    if prev_pixel is not None:
                        cv2.line(base, prev_pixel, (col, row), poly_color, 1)
                    prev_pixel = (col, row)
                else:
                    base[row, col, :] = poly_color

        # Heading arrow at origin using derivative at x=0
        try:
            dy0 = np.polyval(np.polyder(poly), 0.0)
            # direction vector in pixel space (forward negative rows, lateral columns)
            heading_len_m = 3.0
            vx_m = heading_len_m  # forward distance
            vy_m = dy0 * heading_len_m
            tip_col = int(round(center_x + vy_m / px2m))
            tip_row = int((h - 1) - round(vx_m / px2m))
            if cv2 is not None and 0 <= tip_row < h and 0 <= tip_col < w:
                cv2.arrowedLine(base, (center_x, h - 1), (tip_col, tip_row), (0, 255, 255), 2, tipLength=0.3)
        except Exception:
            pass

    # Vehicle origin marker (triangle)
    if cv2 is not None:
        p0 = (center_x, h - 1)
        p1 = (center_x - 8, h - 25)
        p2 = (center_x + 8, h - 25)
        cv2.fillPoly(base, [np.array([p0, p1, p2])], origin_color)
    else:
        row0 = h - 1
        base[row0-25:row0, center_x-2:center_x+3, :] = origin_color

    return base

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
            # discrete linear update
            e_y = e_y + dt * v * e_psi
            e_psi = e_psi + dt * v / L * delta_k
            traj.append((e_y, e_psi))
        return np.array(traj)

    def cost(deltas):
        traj = simulate_errors(deltas)
        # Penalize lateral error and heading error and control effort
        e_y_traj = traj[:, 0]
        e_psi_traj = traj[:, 1]
        J = q_y * np.sum(e_y_traj ** 2) + q_psi * np.sum(e_psi_traj ** 2) + r * np.sum(deltas ** 2)
        return J

    x0 = np.zeros(N)
    bounds = [delta_bounds] * N
    res = minimize(cost, x0, bounds=bounds, method="SLSQP", options={"maxiter": 200, "ftol":1e-4})
    deltas_opt = res.x if res.success else x0
    return deltas_opt[0], deltas_opt


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi 


def compute_steering_from_binary(binary_img, px2m, img_publisher, v=4.0, L=2.5, dt=0.1, horizon=10, ):
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

    try:
        x_pts, y_pts = extract_points_from_binary(binary_img, px2m)
    except Exception:
        return 0.0

    if len(x_pts) < 4:
        # za mało punktów do dopasowania sensownego polinomu
        return 0.0

    # dopasuj wielomian referencyjny (używamy stopnia 3 tak jak w symulacji)
    try:
        poly = fit_reference_poly(x_pts, y_pts, deg=3)
    except Exception:
        return 0.0
    

    # Build visualization image with points & polynomial and publish
    try:
        viz = draw_points_on_image(binary_img, x_pts, y_pts, px2m, poly)
        if viz is not None:
            # prefer direct color frame update
            if hasattr(img_publisher, 'update_frame'):
                img_publisher.update_frame(viz)
            else:
                img_publisher.update_binary_frame(viz)
            # publish immediately if method exists
            if hasattr(img_publisher, 'publish_now'):
                img_publisher.publish_now()
    except Exception:
        pass

    # vehicle at origin: x_fwd = 0
    x_fwd = 0.0
    # referencyjna lateralna pozycja i jej pochodna w punkcie 0
    y_ref = np.polyval(poly, x_fwd)
    dy_ref = np.polyval(np.polyder(poly), x_fwd)
    psi_ref = np.arctan2(dy_ref, 1.0)

    # zakładamy, że pojazd ma y=0 i psi=0
    e_y = 0.0 - y_ref
    e_psi = wrap_angle(0.0 - psi_ref)

    delta0, seq = linear_mpc_control((e_y, e_psi), v, L, dt, horizon)
    return float(delta0)


def run_simulation():
    # Car parameters
    L = 2.5  # wheelbase (m)
    v = 4.0  # constant forward speed (m/s)
    dt = 0.1
    horizon = 10
    sim_time = 20.0

    # Create synthetic binary image and extract ref polynomial
    binary, px2m = binary_path_demo(width=220, height=600, lane_offset=0.0)

    # show binary image using OpenCV (or matplotlib fallback)
    def show_binary_opencv(binary_img, px2m, scale=3, window_name='binary_path'):
        """Display binary image using OpenCV. Scales image for visibility and waits for key press.

        If OpenCV is not available, falls back to matplotlib.imshow.
        """
        if cv2 is None:
            try:
                plt.figure(figsize=(6, 10))
                plt.imshow(binary_img, cmap='gray', origin='upper')
                plt.title('binary path (matplotlib fallback)')
                plt.axis('off')
                plt.show()
            except Exception:
                print('[show_binary_opencv] Cannot display image: cv2 and matplotlib unavailable')
            return

        img = (binary_img * 255).astype('uint8')
        h, w = img.shape
        img_resized = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        cv2.imshow(window_name, img_bgr)
        print('[show_binary_opencv] Press any key in the image window to continue...')
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    show_binary_opencv(binary, px2m, scale=3)


    x_pts, y_pts = extract_points_from_binary(binary, px2m)
    poly = fit_reference_poly(x_pts, y_pts, deg=3)

    # Initial vehicle state (x forward, y lateral, psi heading)
    state = np.array([0.0, 0.0, 0.0])
    states = [state.copy()]
    deltas = []
    times = [0.0]

    fig, ax = plt.subplots(figsize=(6, 9))
    plt.ion()

    t = 0.0
    steps = int(sim_time / dt)
    for i in range(steps):
        # Compute lateral & heading errors with respect to polynomial reference
        x_fwd = state[0]
        y_lat = state[1]
        psi = state[2]
        y_ref = np.polyval(poly, x_fwd)
        dy_ref = np.polyval(np.polyder(poly), x_fwd)
        psi_ref = np.arctan2(dy_ref, 1.0)
        e_y = y_lat - y_ref
        e_psi = wrap_angle(psi - psi_ref)

        # Solve MPC for steering sequence
        delta0, seq = linear_mpc_control((e_y, e_psi), v, L, dt, horizon)

        # Apply first control to actual vehicle (kinematic bicycle with Euler)
        delta_apply = np.clip(delta0, -0.6, 0.6)
        # update full kinematic
        state[0] += dt * v * np.cos(state[2])
        state[1] += dt * v * np.sin(state[2])
        state[2] += dt * v / L * np.tan(delta_apply)
        state[2] = wrap_angle(state[2])

        # store data
        states.append(state.copy())
        deltas.append(delta_apply)
        t += dt
        times.append(t)

        # Visualization
        ax.clear()
        ax.set_title(f"MPC bicycle demo t={t:.2f}s")
        # plot binary path (converted to meters)
        h, w = binary.shape
        # show path points as scatter in world coordinates
        ax.scatter(x_pts, y_pts, c='k', s=3, label='path points')

        # plot fitted polynomial
        xs_plot = np.linspace(0, max(20.0, state[0] + 20), 200)
        ys_plot = np.polyval(poly, xs_plot)
        ax.plot(xs_plot, ys_plot, 'r--', label='ref poly')

        # plot vehicle position and past trajectory
        St = np.array(states)
        ax.plot(St[:, 0], St[:, 1], 'b-', label='vehicle traj')
        ax.scatter([state[0]], [state[1]], c='b', s=40)

        # predicted error trajectory in world coords using linear predictions
        # We'll map predicted e_y,e_psi to approximate world points
        # (simple integration assuming small angles)
        e_y_pred = e_y
        e_psi_pred = e_psi
        pred_world_x = []
        pred_world_y = []
        for k in range(horizon):
            # predict next using same linear model used in MPC
            e_y_pred = e_y_pred + dt * v * e_psi_pred
            e_psi_pred = e_psi_pred + dt * v / L * seq[k]
            pred_x = state[0] + (k + 1) * dt * v * np.cos(state[2])
            # map lateral error back to world y by adding to reference at that x
            yref_k = np.polyval(poly, pred_x)
            pred_world_x.append(pred_x)
            pred_world_y.append(yref_k + e_y_pred)
        if pred_world_x:
            ax.plot(pred_world_x, pred_world_y, 'g-.', label='predicted')

        ax.set_xlabel('x forward (m)')
        ax.set_ylabel('y lateral (m)')
        ax.legend(loc='upper left')
        ax.set_xlim(max(0, state[0] - 5), state[0] + 25)
        ax.set_ylim(min(-6, state[1] - 5), max(6, state[1] + 5))
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_simulation()
