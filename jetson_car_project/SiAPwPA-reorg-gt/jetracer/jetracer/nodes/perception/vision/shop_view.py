#!/usr/bin/env python3
from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.perception.splain_tracking.main_line_preprocessing import OrangeBinaryProcessor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
from jetracer.nodes.perception.vision.transform import BirdView

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False


if __name__ == '__main__':
    main()
