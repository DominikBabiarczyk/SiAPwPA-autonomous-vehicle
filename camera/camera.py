#!/usr/bin/env python3
"""
ROS 2 node do pobierania obrazu z topicu /rs_front/image, prostej analizy
(średnia jasność, histogram kanału Y / szarości) oraz opcjonalnego podglądu.

Uruchomienie (po zmostkowaniu Gazebo -> ROS 2):
  ros2 run ros_gz_bridge parameter_bridge \
	/rs_front/image@sensor_msgs/msg/Image@gz.msgs.Image
  python3 image_preprocesing.py --show

Parametry:
  --topic /inny/topic   (domyślnie /rs_front/image)
  --show                (okno OpenCV; wymaga środowiska graficznego)
  --hist-every N        (co N ramek wypisz histogram)

Jeśli cv_bridge jest dostępne, używa go. W przeciwnym razie konwertuje ręcznie
zakładając encoding rgb8 lub bgr8.
"""

import argparse
import sys
import time
from typing import Optional
import os

from transform import BirdView
from image_preprocesing import ImageProcessor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
	from cv_bridge import CvBridge
	_HAS_CV_BRIDGE = True
except ImportError:  # Fallback jeśli brak cv_bridge
	_HAS_CV_BRIDGE = False

try:
	import cv2
	_HAS_CV2 = True
except ImportError:
	_HAS_CV2 = False

import numpy as np


class ImageAnalyzer(Node):
	def __init__(self, topic: str, show: bool, hist_every: int):
		super().__init__('image_analyzer')
		self.topic = topic
		self.show = show and _HAS_CV2
		self.hist_every = max(1, hist_every)
		self.bridge: Optional[CvBridge] = CvBridge() if _HAS_CV_BRIDGE else None
		self.frame_count = 0
		self.last_stats_time = time.time()
		self.last_save_time = 0.0
		self.image_procesor = ImageProcessor()
		self.subscription = self.create_subscription(
			Image, self.topic, self.image_callback, 10)
		self.get_logger().info(
			f"Subskrypcja: {self.topic} | cv_bridge={_HAS_CV_BRIDGE} | cv2={_HAS_CV2} | show={self.show}")

	def image_callback(self, msg: Image):
		
		self.frame_count += 1
		cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		# FPS counter
		if self.frame_count == 1:
			self.start_time = time.time()
		elif self.frame_count % 30 == 0:
			elapsed = time.time() - self.start_time
			fps = self.frame_count / elapsed
			print(f"FPS: {fps:.2f}")

		gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if _HAS_CV2 else np.mean(cv_img, axis=2).astype(np.uint8)

		now = time.time()

		# cv_img = self.image_procesor.canny_edges(cv_img)
		# cv_img = self.image_procesor.get_lines(cv_img)


		transformation = BirdView()
		bird_view = transformation.apply_transform(cv_img)
		bird_line_image = self.image_procesor.get_lines(bird_view)

		if _HAS_CV2:
			if now - self.last_save_time >= 1.0:
				self.last_save_time = now
				ts = time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
				fname = os.path.join("frames", f'frame_{ts}.png')
				cv2.imwrite(fname, bird_line_image)


def main(argv=None):
	rclpy.init(args=argv)
	node = ImageAnalyzer("/rs_front/image", False, 30)
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		node.get_logger().info('Zatrzymano (Ctrl+C)')
	finally:
		if node.show and _HAS_CV2:
			cv2.destroyAllWindows()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main(sys.argv)

