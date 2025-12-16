from jetracer.nodes.perception.splain_tracking.get_error import ImageErrorCalculator
from jetracer.nodes.perception.splain_tracking.get_main_lines import MainLines
from jetracer.nodes.perception.splain_tracking.get_splain_from_lines import LaneSpline
from jetracer.nodes.control.pid.pid import PIDController
from jetracer.nodes.control.vehicle_go_continuous import ContinuousVehicleCommander
from jetracer.nodes.control.mpc.mpc_bicycle import compute_steering_from_binary
from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.vision.transform import BirdView
from jetracer.nodes.perception.splain_tracking.get_depth_image import DepthImageSubscriber
from jetracer.nodes.perception.splain_tracking.make_splain import get_poly_from_binary_image
import rclpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node


def get_centroid(points):
    """Zwraca środek ciężkości (centroid) dla listy punktów [(x, y), ...]."""
    if not points or len(points) == 0:
        return None
    points_arr = np.array(points)
    centroid = np.mean(points_arr, axis=0)
    return tuple(centroid)


class ImageReceiver(Node):
    def __init__(self, topic="camera/depth_diff"):
        super().__init__('image_receiver')
        self.bridge = CvBridge()
        self.last_image = None
        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, 10)
    
    def image_callback(self, msg):
        # Konwersja ROS Image -> numpy
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.last_image = img
    
    def get_last_image(self):
        return self.last_image


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
        self.commander = ContinuousVehicleCommander()
        self.error = ImageErrorCalculator()
        self.pid_controller = PIDController(Kp=0.01, Ki=0.0, Kd=0.0)
        self.error_calculator = ImageErrorCalculator()
        self.image_publisher = ImagePublisher()
        self.image_publisher2 = ImagePublisher("camera/merged_view", name_node="merged_view_publisher")
        self.image_publisher3 = ImagePublisher("camera/original_bird_view", name_node="original_bird_view")

        self.image_receiver = ImageReceiver()
        self.depth_subscriber = DepthImageSubscriber()
        self.transformed_points = BirdView()
        # transformed_points = self.image_transform.transform_points(points)

    def adjust_movement(self):
        steps = 20000
        for i in range(steps):
            rclpy.spin_once(self.main_lines, timeout_sec=0.1)
            rclpy.spin_once(self.depth_subscriber, timeout_sec=0.1)

            # Ensure images are available before processing
            if not self.main_lines.wait_for_image(timeout_sec=0.1):
                continue
            if not self.depth_subscriber.wait_for_image(timeout_sec=0.1):
                continue

            # Get detected obstacle base points from depth subscriber
            points = self.depth_subscriber.get_obstacle_base_points()
            points_after_transform = self.transformed_points.transform_points(points)
            one_point = get_centroid(points_after_transform)
            # Pass points into main_lines to visualize/transform them
            splain, panorama, bird_view_image = self.main_lines.get_main_lines()
            # roi = self.main_lines.get_main_lines()


            if hasattr(self.image_publisher2, 'update_frame'):
                self.image_publisher2.update_frame(panorama)
            else:
                self.image_publisher2.update_binary_frame(panorama)
            # publish immediately if method exists
            if hasattr(self.image_publisher2, 'publish_now'):
                self.image_publisher2.publish_now()



            if hasattr(self.image_publisher3, 'update_frame'):
                self.image_publisher3.update_frame(bird_view_image)
            else:
                self.image_publisher3.update_binary_frame(bird_view_image)
            # publish immediately if method exists
            if hasattr(self.image_publisher3, 'publish_now'):
                self.image_publisher3.publish_now()


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
            # obstacle = self.image_receiver.get_last_image()

            cv2.imwrite("splain.png", splain)

            poly = get_poly_from_binary_image(splain, [one_point],  0.05, self.image_publisher)

            steering = compute_steering_from_binary(poly)

            self.commander.go_vehicle(0.2, -steering)


def main(args=None):
    rclpy.init(args=args)
    try:
        manager = ManagementMove()
        manager.adjust_movement()
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()