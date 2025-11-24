from camera.splain_tracking.main_lines.get_splain_from_lines import LaneSpline
from rclpy.node import Node
from camera.transform import BirdView
from cv_bridge import CvBridge
_HAS_CV_BRIDGE = True
import rclpy
from sensor_msgs.msg import Image
from camera.image_preprocesing import ImageProcessor
from camera.splain_tracking.main_lise_preprocessing import OrangeBinaryProcessor
import numpy as np
import cv2

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
    self.lane_spline = LaneSpline(smooth=5.0, step=3)

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
    image_transform = BirdView()
    bird_view_image = image_transform.apply_transform(image)

    lines = self.orange_processor.to_binary(bird_view_image)
    roi = self.get_roi(lines)

    # splain = self.lane_spline.process(lines)
    #Zapisz splaina jako obraz
    
    # self.lane_spline.visualize(splain)
    return roi

  def get_roi(self, image):
      # Zwraca region zainteresowania (ROI) z obrazu
      height, width = image.shape[:2]
      roi = image[250:height, 70:width-70]
      return roi
  

def main():
  rclpy.init()
  node = MainLines()
  try:
    if node.wait_for_image():
      node.get_main_lines()
      print("Zapisano main_lines.png")
    else:
      print("Nie otrzymano obrazu w zadanym czasie")
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
  main()