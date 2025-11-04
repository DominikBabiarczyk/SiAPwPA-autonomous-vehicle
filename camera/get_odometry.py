import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time


class OdometrySubscriber(Node):
	def get_position(self, pose):
		"""Zwraca pozycję (x, y) z obiektu pose.pose."""
		x = pose.position.x
		y = pose.position.y
		return x, y

	def wait_for_odometry(self, timeout_sec=5.0):
		start = time.time()
		while self.last_odom is None and (time.time() - start) < timeout_sec:
				rclpy.spin_once(self, timeout_sec=0.1)
		return self.last_odom is not None

	def get_yaw_from_quaternion(self, orientation):
			import math
			z = orientation.z
			w = orientation.w
			return 2 * math.atan2(z, w)

	def __init__(self):
			super().__init__('odometry_subscriber')
			self.subscription = self.create_subscription(
				Odometry,
				'/model/saye_1/odometry',
				self.odom_callback,
				10)
			self.last_odom = None
			# Origin (punkt odniesienia) do logicznego resetu odometrii
			self._origin_pos = None  # (x0, y0)
			self._origin_yaw = None

	def reset_origin(self):
		"""Ustawia bieżącą odometrię jako nowy punkt odniesienia.
		Zwraca True jeśli się udało."""
		if not self.wait_for_odometry():
			return False
		pose = self.last_odom.pose.pose
		x0 = pose.position.x
		y0 = pose.position.y
		yaw0 = self.get_yaw_from_quaternion(pose.orientation)
		self._origin_pos = (x0, y0)
		self._origin_yaw = yaw0
		self.get_logger().info(f"Reset odometrii: origin=({x0:.2f},{y0:.2f}), yaw0={yaw0:.2f}")
		return True

	def get_actual_position(self):
		"""Zwraca pozycję (x, y) względną do origin jeśli ustawiony."""
		rclpy.spin_once(self, timeout_sec=0.0)
		if self.last_odom is None:
			return None, None
		pose = self.last_odom.pose.pose
		x = pose.position.x
		y = pose.position.y
		if self._origin_pos is not None:
			ox, oy = self._origin_pos
			return x - ox, y - oy
		return x, y

	def get_actual_yaw(self):
		"""Zwraca bieżący yaw względnie do origin yaw jeśli ustawiony."""
		if self.last_odom is None:
			return None
		pose = self.last_odom.pose.pose
		yaw = self.get_yaw_from_quaternion(pose.orientation)
		if self._origin_yaw is not None:
			return yaw - self._origin_yaw
		return yaw

	def odom_callback(self, msg):
		import math
		self.last_odom = msg
		z = msg.pose.pose.orientation.z
		w = msg.pose.pose.orientation.w
		yaw = 2 * math.atan2(z, w)
		# Loguj zarówno globalne jak i (jeśli origin) względne
		if self._origin_pos is not None:
			gx = msg.pose.pose.position.x
			gy = msg.pose.pose.position.y
			ox, oy = self._origin_pos
			self.get_logger().info(f"Odometria: global=({gx:.2f},{gy:.2f}) rel=({gx-ox:.2f},{gy-oy:.2f}) yaw={yaw:.2f} rad")
		else:
			self.get_logger().info(f"Odometria: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}, yaw={yaw:.2f} rad")

def main():
	rclpy.init()
	node = OdometrySubscriber()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
