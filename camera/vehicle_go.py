import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

def main():
    rclpy.init()
    node = Node('simple_cmd_vel_publisher')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)

    msg = Twist()
    msg.linear.x = 0.5  # prędkość do przodu
    msg.angular.z = 0.0 # bez skrętu

    # Jedź przez 3 sekundy
    start = time.time()
    while time.time() - start < 3.0:
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)

    # Zatrzymaj pojazd
    msg.linear.x = 0.0
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()