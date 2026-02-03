
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from jetracer.nvidia_racecar import NvidiaRacecar

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        self.car = NvidiaRacecar()
        # Optional: Set gains if needed (e.g., steering_gain=-0.65, throttle_gain=0.8)
        self.steering_gain = 1.0  # Invert if steering direction is wrong
        self.throttle_gain = 1.0
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            100
        )
        self.get_logger().info('Subscribed to /cmd_vel')

    def listener_callback(self, msg):
        throttle = msg.linear.x * self.throttle_gain
        steering = msg.angular.z * self.steering_gain
        # Clamp values to safe ranges (-1 to 1)
        throttle = max(min(throttle, 1.0), -1.0)
        steering = max(min(steering, 1.0), -1.0)
        self.car.throttle_gain = 0.9
        self.car.throttle = throttle
        self.car.steering = -steering
        self.get_logger().info(f'Set throttle: {throttle}, steering: {steering}')

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.car.throttle = 0.0  # Safety stop
        node.car.steering = 0.0
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
