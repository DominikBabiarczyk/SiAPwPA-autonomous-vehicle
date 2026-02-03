from jetracer.nodes.perception.splain_tracking.make_splain import PollynomialFit
#from jetracer.nodes.perception.obstacle_avoiddance.obstacle_avoidance_decider import ObstacleAvoidanceDecider
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
    return [tuple(centroid)]


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

class TurnDirection:
    def __init__(self, commander):
        self.commander = commander
        self.max_on_state = 30
        self.max_on_state = [30, 30, 10, 30, 40]

    def initialize(self, direction):
        self.direction = direction
        self.number_of_iterations = np.zeros(5, dtype=int)
        self.actual_state = 0
        self.state_steerin = [direction, -direction, 0, -direction, direction, 0]

    def move(self, velocity=0.15):
        if self.number_of_iterations[self.actual_state] > self.max_on_state[self.actual_state]:
            self.actual_state += 1
            self.direction *= -1
            print("Finished turn maneuver")
            if self.actual_state >= 5:
                print("koniec") 

                return False  
        self.commander.go_vehicle(velocity, self.state_steerin[self.actual_state] * 0.6)
        self.number_of_iterations[self.actual_state] += 1
        return True



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
        self.pid_controller = PIDController(Kp=0.01, Ki=0.0, Kd=0.0)
        self.error_calculator = ImageErrorCalculator()
        self.image_publisher = ImagePublisher()
        #self.image_publisher2 = ImagePublisher("camera/merged_view", name_node="merged_view_publisher")
        #self.obstacle_avoider = ObstacleAvoidanceDecider(image_width=640, image_height=480)
        self.image_receiver = ImageReceiver()
        #self.depth_subscriber = DepthImageSubscriber()
        self.transformed_points = BirdView()
        self.flag_turn = False
        self.turn_direction = TurnDirection(self.commander)
        self.poly_fitter = PollynomialFit()

        # transformed_points = self.image_transform.transform_points(points)

    def adjust_movement(self):
        steps = 2000000
        for i in range(steps):
            
            rclpy.spin_once(self.main_lines, timeout_sec=0.1)
            #rclpy.spin_once(self.depth_subscriber, timeout_sec=0.1)
            # Ensure images are available before processing
            if not self.main_lines.wait_for_image(timeout_sec=0.1):
                continue
            #if not self.depth_subscriber.wait_for_image(timeout_sec=0.1):
            #    continue

            # Get detected obstacle base points from depth subscriber
            #points = self.depth_subscriber.get_obstacle_base_points()
            # if points:
            #     points_after_transform = self.transformed_points.transform_points(points)
            #     one_point = get_centroid(points_after_transform)
            # else:
            #     one_point = None
            
            splain = self.main_lines.get_main_lines()



            #PID
            #error_tuple = self.error_calculator.calculate(roi)
            #if isinstance(error_tuple, (tuple, list)) and len(error_tuple) == 3 and error_tuple[2] is not None:
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


            poly = self.poly_fitter.get_poly_from_binary_image(splain, 0.05, self.image_publisher)
            #cv2.imwrite("splain.png", splain)
            # Skip MPC if polynomial fit failed (function returns 0.0)
            if hasattr(poly, "__len__"):
                steering = compute_steering_from_binary(poly)
            else:
                steering = 0.0
            

            #is_needing_avoidance = self.obstacle_avoider.decide(points if points else [])
            #print(is_needing_avoidance)
            #if is_needing_avoidance is not None and not self.flag_turn:
            #    self.flag_turn = True
            #    self.turn_direction.initialize(is_needing_avoidance)

            
            #if self.flag_turn:
            #    self.flag_turn = self.turn_direction.move(velocity=0.2)
            #else:   
            self.commander.go_vehicle(0.05, -steering)


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
