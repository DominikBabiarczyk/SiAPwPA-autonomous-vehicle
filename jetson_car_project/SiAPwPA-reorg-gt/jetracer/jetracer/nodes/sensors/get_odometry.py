import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time
import os


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
			# Tryb pracy odometrii: 'relative_zero' (zerowanie po reset_origin) lub 'global'
			self.mode = os.getenv('ODOM_MODE', 'relative_zero')  # można też zamienić na parametry ROS jeśli potrzeba
			self.get_logger().info(f"Odometry mode: {self.mode}")
			self.subscription = self.create_subscription(
				Odometry,
				'/model/saye_1/odometry',
				self.odom_callback,
				10)
			# Publisher dla odometrii względnej / lub skopiowanej globalnej jeśli mode='global'
			self.relative_pub = self.create_publisher(Odometry, '/relative_odometry', 10)
			self.last_odom = None
			# Origin (punkt odniesienia) do logicznego resetu odometrii
			self._origin_pos = None  # (x0, y0)
			self._origin_yaw = None
			self.rel_x = 0
			self.rel_y = 0

	def set_mode(self, mode: str):
		"""Pozwala dynamicznie zmienić tryb pracy odometrii."""
		self.mode = mode
		self.get_logger().info(f"Zmieniono odometry mode na: {self.mode}")

	
	def reset_odometry(self):
		rclpy.spin_once(self, timeout_sec=0.0)
		if self.last_odom is None:
			return None, None
		pose = self.last_odom.pose.pose
		x = pose.position.x
		y = pose.position.y
		self.rel_x = x
		self.rel_y = y 


	def get_actual_position(self):
		"""
		Zwraca pozycję (x, y) zgodnie z trybem odometrii:
		- 'relative_zero': względem origin (zerowana po teleporcie)
		- 'global': globalna pozycja w świecie
		"""
		rclpy.spin_once(self, timeout_sec=0.0)
		if self.last_odom is None:
			return None, None
		pose = self.last_odom.pose.pose
		x = pose.position.x
		y = pose.position.y

		new_x, new_y = x - self.rel_x, y - self.rel_y
		# print(f"Actual position (global): {new_x}, {new_y}")
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
		# pass
		import math
		self.last_odom = msg
		# z = msg.pose.pose.orientation.z
		# w = msg.pose.pose.orientation.w
		# yaw = 2 * math.atan2(z, w)
		# # Loguj zarówno globalne jak i (jeśli origin) względne
		# gx = msg.pose.pose.position.x
		# gy = msg.pose.pose.position.y
		# # Publikacja zależy od trybu
		# if self.mode == 'relative_zero' and self._origin_pos is not None:
		# 		ox, oy = self._origin_pos
		# 		rel = Odometry()
		# 		rel.header = msg.header
		# 		rel.child_frame_id = 'base_link'
		# 		rel.pose.pose = msg.pose.pose
		# 		rel.pose.pose.position.x = gx - ox
		# 		rel.pose.pose.position.y = gy - oy
		# 		rel.twist = msg.twist
		# 		self.relative_pub.publish(rel)
		# elif self.mode == 'global':
		# 		# Publikuj kopię globalnej odometrii jako 'relative' aby klienci mieli jeden topic
		# 		rel = Odometry()
		# 		rel.header = msg.header
		# 		rel.child_frame_id = 'base_link'
		# 		rel.pose.pose = msg.pose.pose
		# 		rel.twist = msg.twist
		# 		self.relative_pub.publish(rel)

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
