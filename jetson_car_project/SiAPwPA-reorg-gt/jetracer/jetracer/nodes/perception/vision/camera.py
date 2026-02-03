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


from jetracer.nodes.control.vehicle_go import VehicleCommander
from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.qlearning.preprocess_sensor import PreprocessSensor
from transform import BirdView
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2

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




if __name__ == '__main__':
	main(sys.argv)

