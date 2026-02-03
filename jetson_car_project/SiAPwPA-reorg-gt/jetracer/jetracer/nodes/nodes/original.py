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


class TurnDirection:
    def __init__(self, commander):
        self.commander = commander
        self.max_on_state = [30, 30, 10, 30, 40]
        self.direction = 0
        self.number_of_iterations = np.zeros(5, dtype=int)
        self.actual_state = 0
        self.state_steering = []

    def initialize(self, direction):
        self.direction = direction
        self.number_of_iterations = np.zeros(5, dtype=int)
        self.actual_state = 0
        self.state_steering = [direction, -direction, 0, -direction, direction, 0]

    def move(self, velocity=0.15):
        if self.actual_state >= 5:
            return False

        if self.number_of_iterations[self.actual_state] > self.max_on_state[self.actual_state]:
            self.actual_state += 1
            if self.actual_state >= 5:
                print("Turn maneuver complete")
                return False

        self.commander.go_vehicle(velocity, self.state_steering[self.actual_state] * 0.6)
        self.number_of_iterations[self.actual_state] += 1
        return True


class JetRacerController(Node):
    """
    Unified controller node that handles image processing and vehicle control
    using ROS2 timer-based callbacks for efficient execution.
    """

    def __init__(self):
        super().__init__('jetracer_controller')

        # Callback groups for concurrent execution
        self.sensor_cb_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # CV Bridge
        self.bridge = CvBridge()

        # State variables (thread-safe access)
        self._lock = threading.Lock()
        self._binary_lines = None
        self._depth_image = None
        self._last_error = 0.0
        self._last_steering = 0.0

        # Control parameters
        self.base_speed = 0.30
        self.flag_turn = False

        # Initialize components
        self.error_calculator = ImageErrorCalculator()
        self.pid_controller = PIDController(Kp=1.01, Ki=0.01, Kd=0.52)
        self.commander = ContinuousVehicleCommander()
        self.turn_direction = TurnDirection(self.commander)
        self.transformed_points = BirdView()

        # Subscribe to camera/line detection topic
        # Adjust topic name based on what MainLines publishes
        self.line_subscription = self.create_subscription(
            Image,
            'camera/binary_lines',  # Adjust to your actual topic
            self.line_callback,
            10,
            callback_group=self.sensor_cb_group
        )

        # Optional: depth difference subscription
        self.depth_subscription = self.create_subscription(
            Image,
            'camera/depth_diff',
            self.depth_callback,
            10,
            callback_group=self.sensor_cb_group
        )

        # Control loop timer - runs at ~30Hz (adjust based on your needs)
        # Lower frequency = less CPU load on Jetson Nano
        self.control_timer = self.create_timer(
            0.033,  # ~30Hz, increase to 0.05 for 20Hz if needed
            self.control_loop,
            callback_group=self.control_cb_group
        )

        self.get_logger().info('JetRacer Controller initialized')

    def line_callback(self, msg):
        """Process incoming line detection images."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._binary_lines = img
        except Exception as e:
            self.get_logger().warn(f'Line callback error: {e}')

    def depth_callback(self, msg):
        """Process incoming depth images."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._lock:
                self._depth_image = img
        except Exception as e:
            self.get_logger().warn(f'Depth callback error: {e}')

    def control_loop(self):
        """
        Main control loop - called by timer at fixed rate.
        This replaces the blocking while loop.
        """
        # Get latest sensor data (thread-safe)
        with self._lock:
            binary_lines = self._binary_lines

        if binary_lines is None:
            # No data yet, skip this cycle
            return

        # Handle turn maneuver if active
        if self.flag_turn:
            self.flag_turn = self.turn_direction.move(velocity=0.2)
            return

        # Calculate error from line detection
        try:
            error_tuple = self.error_calculator.calculate(binary_lines)

            if isinstance(error_tuple, (tuple, list)) and len(error_tuple) >= 3 and error_tuple[2] is not None:
                error_x = float(error_tuple[2])
            elif isinstance(error_tuple, (float, int, np.floating)):
                error_x = float(error_tuple)
            else:
                # Use last known error if calculation fails
                error_x = self._last_error

            self._last_error = error_x

        except Exception as e:
            self.get_logger().warn(f'Error calculation failed: {e}')
            error_x = self._last_error

        # Compute steering with PID
        steering = self.pid_controller.compute(error_x)
        self._last_steering = steering

        # Send command to vehicle
        self.commander.go_vehicle(self.base_speed, -steering)

    def stop(self):
        """Emergency stop."""
        self.commander.go_vehicle(0.0, 0.0)


class MainLinesNode(Node):
    """
    Wrapper node for MainLines that publishes processed images.
    This separates perception from control for better modularity.
    """

    def __init__(self):
        super().__init__('main_lines_node')

        self.main_lines = MainLines()
        self.bridge = CvBridge()

        # Publisher for processed line images
        self.line_publisher = self.create_publisher(
            Image,
            'camera/binary_lines',
            10
        )

        # Timer for processing - adjust rate as needed
        self.process_timer = self.create_timer(0.033, self.process_callback)

    def process_callback(self):
        """Process and publish line detection."""
        # Spin the MainLines node to get fresh data
        rclpy.spin_once(self.main_lines, timeout_sec=0.001)

        if not self.main_lines.wait_for_image(timeout_sec=0.001):
            return

        binary_lines = self.main_lines.get_main_lines()

        if binary_lines is not None:
            # Convert to ROS message and publish
            if binary_lines.dtype == bool:
                binary_lines = binary_lines.astype(np.uint8) * 255

            msg = self.bridge.cv2_to_imgmsg(binary_lines, encoding='mono8')
            self.line_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    controller = JetRacerController()
    lines_node = MainLinesNode()

    # Use MultiThreadedExecutor for concurrent callback execution
    # This is more efficient than manual spin_once in a loop
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(controller)
    executor.add_node(lines_node)

    try:
        controller.get_logger().info('Starting JetRacer control system...')
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    except Exception as e:
        controller.get_logger().error(f'Error: {e}')
    finally:
        # Clean shutdown
        controller.stop()
        controller.destroy_node()
        lines_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
