from jetracer.nodes.perception.splain_tracking.get_error import ImageErrorCalculator
from jetracer.nodes.perception.splain_tracking.get_main_lines import MainLines
from jetracer.nodes.control.pid.pid import PIDController
from jetracer.nodes.control.vehicle_go_continuous import ContinuousVehicleCommander
from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.vision.transform import BirdView
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading


def get_centroid(points):
    """Returns centroid for a list of points [(x, y), ...]."""
    if not points or len(points) == 0:
        return None
    points_arr = np.array(points)
    centroid = np.mean(points_arr, axis=0)
    return [tuple(centroid)]


class ObstacleAvoidanceDecider:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def decide(self, obstacle_points):
        if not obstacle_points:
            return None

        print(obstacle_points)

        max_point = max(obstacle_points, key=lambda p: p[1])
        if max_point[1] > self.image_height * 0.2:
            avg_x = sum(x for x, y in obstacle_points) / len(obstacle_points)
            if avg_x < self.image_width / 2:
                return -1
            else:
                return 1
        return None


class TurnDirection:
    def __init__(self, commander):
        self.commander = commander
        self.max_on_state = [9, 2, 8, 1, 1]#[20, 16, 16, 20]
        self.direction = 0
        self.number_of_iterations = np.zeros(5, dtype=int)
        self.actual_state = 0
        self.state_steering = []

    def initialize(self, direction):
        self.direction = direction
        self.number_of_iterations = np.zeros(5, dtype=int)
        self.actual_state = 0
        self.state_steering = [direction, 0, -direction, -direction, direction]

    def move(self, velocity=0.15):
        if self.actual_state >= 3:
            return False

        if self.number_of_iterations[self.actual_state] > self.max_on_state[self.actual_state]:
            self.actual_state += 1
            if self.actual_state >= 3:
                print("Turn maneuver complete")
                return False

        self.commander.go_vehicle(
            velocity,
            self.state_steering[self.actual_state] * 2.0
        )
        self.number_of_iterations[self.actual_state] += 1
        return True


class DepthImageSubscriber(Node):
    def __init__(
        self,
        topic="/depth/image_rect_raw"
    ):
        super().__init__('depth_image_subscriber')

        self.bridge = CvBridge() #if _HAS_CV_BRIDGE else None
        self.last_image = None
        self.reference_image = None

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f"Subskrypcja: {topic}")

        #self.diff_publisher = ImagePublisher(
        #    topic='camera/depth_diff',
        #    name_node='depth_diff_publisher'
        #)

        #import cv2

        #self.bird_view_transform = BirdView()
        self.roi_mean = None
        self.turn = 1

    def image_callback(self, msg):
        if self.bridge:
            depth_img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        else:
            arr = np.frombuffer(msg.data, dtype=np.float32)
            depth_img = arr.reshape((msg.height, msg.width))

        if depth_img is None:
            return
        if not isinstance(depth_img, np.ndarray):
            return

        if depth_img.dtype != np.float32:
            try:
                depth_img = depth_img.astype(np.float32)
            except Exception:
                return

        self.last_image = depth_img

        x, y, w, h = 600, 80, 80, 200

        x_end = min(x + w, depth_img.shape[1])
        y_end = min(y + h, depth_img.shape[0])

        roi = depth_img[y:y_end, x:x_end]
        roi = roi[::10, ::10]
        #print("roi_size", roi.size)
        h, w = roi.shape
        mid = w//2
        left_mean = roi[:, :mid].mean()
        right_mean = roi[:, mid:].mean()

        self.turn = 1
        if left_mean > right_mean:
            self.turn=-1
         
        #roi_sum = np.sum(roi)
            #print("ROI_SUM",roi_sum)
        self.roi_mean = left_mean + right_mean

        self.roi_mean_brightness = self.roi_mean
        #print(f"ROI mean brightness: {self.roi_mean}")
        self.get_logger().info(f" mean={self.roi_mean}")

        #self.diff_publisher.update_frame(roi)
        #self.diff_publisher.publish_now()

    def wait_for_image(self, timeout_sec=5.0):
        import time
        start = time.time()
        while self.last_image is None and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_image is not None

    def get_depth_image(self):
        if self.last_image is not None:
            return self.last_image.copy()
        return None

    def get_turn_decision(self): 
        if self.roi_mean is None:
            return False
        print(self.roi_mean)
        return (self.roi_mean < 850.0), self.turn


class JetRacerController(Node):
    def __init__(self, depth_subscriber):
        super().__init__('jetracer_controller')

        self.sensor_cb_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._binary_lines = None
        self._depth_image = None
        self._last_error = 0.0
        self._last_steering = 0.0

        self.base_speed = 0.30
        self.flag_turn = False
        self.break_iterations=0

        self.error_calculator = ImageErrorCalculator()
        self.pid_controller = PIDController(Kp=1.01, Ki=0.01, Kd=0.52)
        self.commander = ContinuousVehicleCommander()
        self.turn_direction = TurnDirection(self.commander)
        #self.turn_direction.initialize(1)
        #self.transformed_points = BirdView()
        self.depth_subscriber = depth_subscriber

        self.line_subscription = self.create_subscription(
            Image,
            'camera/binary_lines',
            self.line_callback,
            10,
            callback_group=self.sensor_cb_group
        )

        #self.depth_subscription = self.create_subscription(
        #    Image,
        #    'camera/depth_diff',
        #    self.depth_callback,
        #    10,
        #    callback_group=self.sensor_cb_group
        #)

        self.control_timer = self.create_timer(
            0.033,
            self.control_loop,
            callback_group=self.control_cb_group
        )

        self.get_logger().info('JetRacer Controller initialized')

    def line_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._binary_lines = img
        except Exception as e:
            self.get_logger().warn(f'Line callback error: {e}')

    #def depth_callback(self, msg):
    #    try:
    #        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #        with self._lock:
    #            self._depth_image = img
    #    except Exception as e:
    #        self.get_logger().warn(f'Depth callback error: {e}')

    def control_loop(self):
        
        with self._lock:
            binary_lines = self._binary_lines

        #if binary_lines is None:
        #    return
        
        #if self.flag_turn:
        #    self.flag_turn = self.turn_direction.move(velocity=0.2)
        #    return

        try:
            error_tuple = self.error_calculator.calculate(binary_lines)

            if (
                isinstance(error_tuple, (tuple, list))
                and len(error_tuple) >= 3
                and error_tuple[2] is not None
            ):
                error_x = float(error_tuple[2])
            elif isinstance(error_tuple, (float, int, np.floating)):
                error_x = float(error_tuple)
            else:
                error_x = self._last_error

            self._last_error = error_x

        except Exception as e:
            self.get_logger().warn(f'Error calculation failed: {e}')
            error_x = self._last_error

        steering = self.pid_controller.compute(error_x)
        #self._last_steering = steering
        isTurn, turn = self.depth_subscriber.get_turn_decision()
        
        if isTurn and not self.flag_turn and (self.break_iterations > 100):
            self.flag_turn = True
            self.turn_direction.initialize(-turn)
            self.break_iterations=0

        if self.flag_turn:
            self.flag_turn = self.turn_direction.move(velocity=0.16)
            #self.commander.go_vehicle(0.14, -steering)
        else:
            self.commander.go_vehicle(0.15, -0.0)
        self.break_iterations += 1
    def stop(self):
        self.commander.go_vehicle(0.0, 0.0)


class MainLinesNode(Node):
    def __init__(self):
        super().__init__('main_lines_node')

        self.main_lines = MainLines()
        self.bridge = CvBridge()

        self.line_publisher = self.create_publisher(
            Image,
            'camera/binary_lines',
            10
        )

        self.process_timer = self.create_timer(
            0.033,
            self.process_callback
        )

    def process_callback(self):
        rclpy.spin_once(self.main_lines, timeout_sec=0.001)

        if not self.main_lines.wait_for_image(timeout_sec=0.001):
            return

        binary_lines = self.main_lines.get_main_lines()

        if binary_lines is not None:
            if binary_lines.dtype == bool:
                binary_lines = binary_lines.astype(np.uint8) * 255

            msg = self.bridge.cv2_to_imgmsg(
                binary_lines,
                encoding='mono8'
            )
            self.line_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    depth_node = DepthImageSubscriber()
    controller = JetRacerController(depth_node)
    lines_node = MainLinesNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    executor.add_node(lines_node)
    executor.add_node(depth_node)

    try:
        controller.get_logger().info('Starting JetRacer control system...')
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    except Exception as e:
        controller.get_logger().error(f'Error: {e}')
    finally:
        controller.stop()
        controller.destroy_node()
        lines_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

