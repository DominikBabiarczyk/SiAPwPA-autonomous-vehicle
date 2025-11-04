from rclpy.node import Node
import rclpy
from geometry_msgs.msg import Twist
import time


class SetAction:
    def __init__(self):
        self.node = Node('vehicle_commander')
        self.pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

    def map_value_control(self, linear_x_idx, angular_z_idx):
        linear_x_values = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        angular_z_values = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        linear_x = linear_x_values[linear_x_idx]
        angular_z = angular_z_values[angular_z_idx]
        return linear_x, angular_z
    
    def go_vehicle(self, linear_x_idx, angular_z_idx):
        msg = Twist()
        linear_x, angular_z = self.map_value_control(linear_x_idx, angular_z_idx)
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        start = time.time()
        while time.time() - start < 0.3:
            self.pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.stop_vehicle()
        
    def stop_vehicle(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        stop_start = time.time()
        while time.time() - stop_start < 0.2:
            self.pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)

if __name__ == '__main__':
    commander = SetAction()
    commander.go_vehicle(7, 4)  # Przykład: jedź prosto przez 0.2s, potem zatrzymaj
    commander.node.destroy_node()
    rclpy.shutdown()