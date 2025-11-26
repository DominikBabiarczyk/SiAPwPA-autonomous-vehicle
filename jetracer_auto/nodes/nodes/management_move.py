
from jetracer_auto.perception.splain_tracking.get_error import ImageErrorCalculator
from jetracer_auto.perception.splain_tracking.get_main_lines import MainLines
from jetracer_auto.perception.splain_tracking.get_splain_from_lines import LaneSpline
from jetracer_auto.control.pid.pid import PIDController
from jetracer_auto.control.vehicle_go_continuous import ContinuousVehicleCommander
from jetracer_auto.control.mpc.mpc_bicycle import compute_steering_from_binary
from jetracer_auto.nodes.publish_image import ImagePublisher
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading



def imshow_nonblocking(img, window_name='image', scale=1):
    """Show `img` in a non-blocking OpenCV window.

    - If OpenCV is available: uses cv2.imshow + cv2.waitKey(1) so the call returns
      immediately and the window updates. Call repeatedly to update the image.
    - If OpenCV is not available: falls back to matplotlib interactive draw and pause(0.001).

    img: numpy array (grayscale 2D or BGR 3-channel). If values are 0/1 it will be scaled.
    scale: integer scale factor to enlarge small images for visibility.
    """
    # prepare image
    if img is None:
        return
    arr = img.copy()
    # scale binary 0/1 to 0-255
    if arr.dtype == np.bool_ or arr.max() == 1 and arr.min() >= 0:
        arr = (arr * 255).astype('uint8')
    if arr.dtype != np.uint8:
        arr = arr.astype('uint8')

    # if single channel, convert to BGR for consistent display
    if arr.ndim == 2:
        disp = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if cv2 is not None else arr
    else:
        disp = arr

    if scale != 1:
        h, w = disp.shape[:2]
        disp = cv2.resize(disp, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) if cv2 is not None else disp

    if cv2 is not None:
        cv2.imshow(window_name, disp)
        # waitKey with 1ms so it's non-blocking but GUI can refresh
        cv2.waitKey(1)
    else:
        try:
            plt.ion()
            plt.imshow(disp, cmap='gray') if disp.ndim == 2 else plt.imshow(disp[:,:,::-1])
            plt.title(window_name)
            plt.axis('off')
            plt.pause(0.001)
            plt.clf()
        except Exception:
            # last-resort: do nothing
            pass


class ManagementMove:
    def __init__(self):
        self.current_speed = 0.3
        self.current_steering = 0.0
        self.max_speed = 1.0
        self.min_speed = -1.0
        self.max_steering = 1.0
        self.min_steering = -1.0
        self.speed_increment = 0.1
        self.steering_increment = 0.1
        self.main_lines = MainLines()
        # self.spline = LaneSpline(smooth=5.0, step=3)
        self.commander = ContinuousVehicleCommander()
        self.error = ImageErrorCalculator()
        self.pid_controller = PIDController(Kp=0.01, Ki=0.0, Kd=0.0)
        self.error_calculator = ImageErrorCalculator()
        self.image_publisher = ImagePublisher()
        # start spinning the publisher node in a background thread so its timer callbacks run
        # def spin_pub():
        #     try:
        #         rclpy.spin(self.image_publisher)
        #     except Exception:
        #         pass
        # self._pub_thread = threading.Thread(target=spin_pub, daemon=True)
        # self._pub_thread.start()

    def adjust_movement(self):
        steps = 20000
        for i in range(steps):
            rclpy.spin_once(self.main_lines, timeout_sec=0.1)
            self.main_lines.wait_for_image()
            splain = self.main_lines.get_main_lines()
            # roi = self.main_lines.get_main_lines()


            #PID
            # error_tuple = self.error_calculator.calculate(roi)
            # if isinstance(error_tuple, (tuple, list)) and len(error_tuple) == 3 and error_tuple[2] is not None:
            #     error_x = error_tuple[2]
            # elif isinstance(error_tuple, (float, int, np.floating)):
            #     error_x = float(error_tuple)
            # else:
            #     error_x = 0.0

            # steering = self.pid_controller.compute(error_x)

            # print(steering)


            #MCP
            # show_binary_opencv(splain, 0.05, scale=3)
            # update published frame (splain is binary image)
            # self.image_publisher.update_binary_frame(splain)
            # self.image_publisher.publish_now()

            steering = compute_steering_from_binary(splain, 0.05, self.image_publisher)

            

            print(steering)
            self.commander.go_vehicle(0.35, -steering)


if __name__ == "__main__":
    rclpy.init()
    manager = ManagementMove()
    manager.adjust_movement()