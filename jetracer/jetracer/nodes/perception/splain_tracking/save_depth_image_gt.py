from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import numpy as np

_HAS_CV_BRIDGE = True

class DepthImageSubscriber(Node):
	def __init__(self, topic="/rs_front/depth_image"):
		super().__init__('depth_image_subscriber')
		self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
		self.last_image = None
		self.subscription = self.create_subscription(
			Image, topic, self.image_callback, 10)
		self.get_logger().info(f"Subskrypcja: {topic}")

	def image_callback(self, msg):
		if self.bridge:
			self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		else:
			arr = np.frombuffer(msg.data, dtype=np.float32)
			self.last_image = arr.reshape((msg.height, msg.width))

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

def main():
	rclpy.init()
	node = DepthImageSubscriber()
	try:
		if node.wait_for_image():
			depth = node.get_depth_image()
			if depth is not None:
				print(f"Odebrano obraz głębi o kształcie: {depth.shape}")
				# Normalizacja i zapis do PNG
				import cv2
				norm_depth = (255 * (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth) + 1e-8)).astype(np.uint8)
				cv2.imwrite("depth_image_with_obstacle.png", norm_depth)
		else:
			print("Nie otrzymano obrazu głębi w zadanym czasie")
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()