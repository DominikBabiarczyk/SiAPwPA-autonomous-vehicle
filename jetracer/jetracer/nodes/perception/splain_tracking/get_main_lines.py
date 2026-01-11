from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.vision.transform import BirdView
from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.perception.splain_tracking.main_line_preprocessing import OrangeBinaryProcessor
from rclpy.node import Node
from cv_bridge import CvBridge
_HAS_CV_BRIDGE = True
import rclpy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from collections import deque


class MainLines(Node):
  def __init__(self, topic="/rs_front/image"):
    super().__init__('preprocess_sensor')
    self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
    self.last_image = None
    self.frame_counter = 0  # licznik zapisanych ramek
    self.subscription = self.create_subscription(
        Image, topic, self.image_callback, 10)
    self.get_logger().info(f"Subskrypcja: {topic}")
    self.image_processor = ImageProcessor()
    self.orange_processor = OrangeBinaryProcessor()
    # self.lane_spline = LaneSpline(smooth=5.0, step=3)
    self.bufor = deque(maxlen=10)
    self.image_transform = BirdView()
    self.image_publisher3 = ImagePublisher("camera/original_bird_view", name_node="original_bird_view")



  def image_callback(self, msg):
      if self.bridge:
          self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
      else:
          # fallback: konwersja ręczna jeśli nie ma cv_bridge
          arr = np.frombuffer(msg.data, dtype=np.uint8)
          self.last_image = arr.reshape((msg.height, msg.width, 3))

  def wait_for_image(self, timeout_sec=5.0):
    """Czeka na pierwszą wiadomość z obrazem"""
    import time
    start = time.time()
    while self.last_image is None and (time.time() - start) < timeout_sec:
        rclpy.spin_once(self, timeout_sec=0.1)
    return self.last_image is not None

  def get_main_lines(self):
    image = self.last_image.copy()

    bird_view_image_roi = self.get_roi(image)

    lines = self.orange_processor.to_binary(bird_view_image_roi)
    self.image_publisher3.update_frame(lines)
    self.image_publisher3.publish_now()
    return lines

  def get_roi(self, image):
      height, width = image.shape[:2]
      roi = image[200:height, :]
      return roi
  


