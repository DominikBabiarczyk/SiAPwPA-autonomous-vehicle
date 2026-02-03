import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading
import time

class ContinuousVehicleCommander:
    def __init__(self):
        # rclpy.init()
        self.node = Node('vehicle_commander')
        self.pub = self.node.create_publisher(Twist, '/cmd_vel', 100)
        # self.checker = ColisionChecker()

    def go_vehicle(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        #start = time.time()
        
        #while time.time() - start < 0.02:
            #self.pub.publish(msg)
            #rclpy.spin_once(self.node, timeout_sec=0.1)
            #print(f"Published at {time.time():.6f}")
        # self.stop_vehicle()
        # print(self.checker.is_green_in_rect())

    def stop_vehicle(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        stop_start = time.time()
        while time.time() - stop_start < 0.2:
            self.pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def shutdown(self):
        self.stop_vehicle()
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    commander = ContinuousVehicleCommander()
    commander.go_vehicle(0.7, 1.0)  # Przykład: jedź prosto przez 0.5s, potem zatrzymaj
    commander.shutdown()
