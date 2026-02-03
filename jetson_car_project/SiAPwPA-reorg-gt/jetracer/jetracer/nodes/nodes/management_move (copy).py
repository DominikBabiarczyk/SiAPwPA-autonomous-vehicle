from jetracer.nodes.perception.splain_tracking.make_splain import PollynomialFit
from jetracer.nodes.perception.splain_tracking.get_error import ImageErrorCalculator
from jetracer.nodes.perception.splain_tracking.get_main_lines import MainLines
from jetracer.nodes.perception.splain_tracking.get_splain_from_lines import LaneSpline
from jetracer.nodes.control.pid.pid import PIDController
from jetracer.nodes.control.vehicle_go_continuous import ContinuousVehicleCommander
from jetracer.nodes.control.mpc.mpc_bicycle import compute_steering_from_binary
from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.vision.transform import BirdView
from jetracer.nodes.perception.splain_tracking.get_depth_image import DepthImageSubscriber
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
import cv2
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
import time


class TurnDirection:
    def __init__(self, commander):
        self.commander = commander
        self.max_on_state = [30, 30, 10, 30, 40]

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
            self.direction *= -1
            print("Finished turn maneuver")
            if self.actual_state >= 5:
                return False
        self.commander.go_vehicle(velocity, self.state_steering[self.actual_state] * 0.6)
        self.number_of_iterations[self.actual_state] += 1
        return True


class ManagementMove(Node):
    """
    Single unified node for vehicle control.
    Uses callbacks instead of polling for better performance.
    """

    def __init__(self):
        super().__init__('management_move')

        # Callback groups for parallel execution
        self.processing_cb_group = ReentrantCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # Control parameters
        self.current_speed = 0.05
        self.current_steering = 0.0
        self.last_steering = 0.0

        # Thread-safe data sharing
        self._lock = threading.Lock()
        self._latest_splain = None
        self._new_frame_available = False

        # Performance tracking
        self._frame_times = deque(maxlen=30)
        self._last_process_time = time.time()

        # Initialize components (lazy initialization where possible)
        self.commander = ContinuousVehicleCommander()
        self.poly_fitter = PollynomialFit()
        self.image_publisher = ImagePublisher()
        self.turn_direction = TurnDirection(self.commander)
        self.flag_turn = False

        # MainLines handles its own subscription - we'll integrate with it
        self.main_lines = MainLines()

        # Control loop timer - runs at fixed rate (e.g., 30 Hz)
        self.control_timer = self.create_timer(
            0.033,  # ~30 Hz control loop
            self._control_loop,
            callback_group=self.control_cb_group
        )

        # Processing timer - processes frames as fast as possible
        self.process_timer = self.create_timer(
            0.01,  # 100 Hz check rate
            self._process_frame,
            callback_group=self.processing_cb_group
        )

        # Stats timer
        self.stats_timer = self.create_timer(2.0, self._print_stats)

        self.get_logger().info("ManagementMove node initialized")

    def _process_frame(self):
        """Process incoming frames - runs in processing callback group"""
        # Quick spin to get latest data from MainLines
        rclpy.spin_once(self.main_lines, timeout_sec=0.001)

        if not self.main_lines.wait_for_image(timeout_sec=0.001):
            return

        try:
            splain = self.main_lines.get_main_lines()
            if splain is None:
                return

            # Compute polynomial fit
            poly = self.poly_fitter.get_poly_from_binary_image(
                splain, 0.05, self.image_publisher
            )

            # Compute steering
            if hasattr(poly, "__len__"):
                steering = compute_steering_from_binary(poly)
            else:
                steering = self.last_steering  # Use last good value

            # Thread-safe update
            with self._lock:
                self._latest_splain = splain
                self.current_steering = steering
                self._new_frame_available = True

            # Track timing
            now = time.time()
            self._frame_times.append(now - self._last_process_time)
            self._last_process_time = now

        except Exception as e:
            self.get_logger().warn(f"Frame processing error: {e}")

    def _control_loop(self):
        """Fixed-rate control loop - sends commands to vehicle"""
        with self._lock:
            steering = self.current_steering
            new_frame = self._new_frame_available
            self._new_frame_available = False

        # Apply steering smoothing to reduce jitter
        alpha = 0.7  # Smoothing factor
        smoothed_steering = alpha * steering + (1 - alpha) * self.last_steering
        self.last_steering = smoothed_steering

        # Send command
        if not self.flag_turn:
            self.commander.go_vehicle(self.current_speed, -smoothed_steering)
        else:
            self.flag_turn = self.turn_direction.move(velocity=0.2)

    def _print_stats(self):
        """Print performance statistics"""
        if self._frame_times:
            avg_time = np.mean(self._frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(f"Processing FPS: {fps:.1f}")


def main(args=None):
    rclpy.init(args=args)

    try:
        manager = ManagementMove()

        # Use MultiThreadedExecutor for parallel callback execution
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(manager)
        executor.add_node(manager.main_lines)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            manager.commander.go_vehicle(0.0, 0.0)  # Stop vehicle
            executor.shutdown()
            manager.destroy_node()

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
