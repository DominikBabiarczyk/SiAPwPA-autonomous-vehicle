from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.splain_tracking.get_splain_from_lines import LaneSpline
from jetracer.nodes.perception.vision.transform import BirdView
from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.perception.splain_tracking.main_line_preprocessing import OrangeBinaryProcessor
from rclpy.node import Node
from cv_bridge import CvBridge
_HAS_CV_BRIDGE = True
import rclpy
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import cv2
from collections import deque


class MainLines(Node):
  def __init__(self, topic="/color/image_raw"): #/rs_front/image/compressed
    super().__init__('preprocess_sensor')
    self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
    self.last_image = None
    self.frame_counter = 0  # licznik zapisanych ramek
    self.subscription = self.create_subscription(
        Image, topic, self.image_callback, 10)   #CompressedImage
    self.get_logger().info(f"Subskrypcja: {topic}")
    self.image_processor = ImageProcessor()
    self.orange_processor = OrangeBinaryProcessor()
    self.lane_spline = LaneSpline(smooth=5.0, step=3)
    self.bufor = deque(maxlen=10)
    self.image_transform = BirdView()
    self.image_publisher3 = ImagePublisher("camera/original_bird_view", name_node="original_bird_view")



  def image_callback(self, msg):
      if self.bridge:
          self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
      #else:
           #fallback: konwersja ręczna jeśli nie ma cv_bridge
      #np_arr = np.frombuffer(msg.data, dtype=np.uint8)
      #img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
      #self.last_image = img
          #arr = np.frombuffer(msg.data, dtype=np.uint8)
          #self.last_image = arr.reshape((msg.height, msg.width, 3))

  def wait_for_image(self, timeout_sec=5.0):
    """Czeka na pierwszą wiadomość z obrazem"""
    import time
    start = time.time()
    while self.last_image is None and (time.time() - start) < timeout_sec:
        rclpy.spin_once(self, timeout_sec=0.1)
    return self.last_image is not None

  def get_main_lines(self):
    #image = self.last_image.copy()
    bird_view_image = cv2.resize(self.last_image[:,320:,:], (160, 90), interpolation=cv2.INTER_AREA)
    #bird_view_image = self.image_transform.apply_transform(image)
    
    #bird_view_image_roi = self.get_roi(bird_view_image)
    #self.bufor.append(bird_view_image_roi)
    #panorama = stitch_first_last_from_buffer(buffer=self.bufor)

    lines = self.orange_processor.to_binary(bird_view_image)

    #self.image_publisher3.update_frame(bird_view_image)
    #self.image_publisher3.publish_now()
    return lines

  def get_roi(self, image):
      height, width = image.shape[:2]
      roi = image[200:height, width//2 - 70: width//2 + 70]
      return roi
  
  def point_to_roi(self, pt, image_shape):
    height, width = image_shape[:2]
    x_offset = width // 2 - 70
    y_offset = 200
    x_roi = pt[0] - x_offset
    y_roi = pt[1] - y_offset
    return (x_roi, y_roi)

  def get_point_in_roi(self, points):
    roi_points = [self.point_to_roi(pt, self.last_image.shape) for pt in points]
    return roi_points
  
  def get_set_points_in_roi(self, points):
    points_after_transform = self.image_transform.transform_points(points)
    roi_points = [self.point_to_roi(pt, self.last_image.shape) for pt in points_after_transform]
    return roi_points


def stitch_first_last_from_buffer(buffer):
    """
    Tworzy panoramę tylko z pierwszego i ostatniego obrazu w buforze.
    Jeśli jest jeden obraz -> zwraca go.
    Jeśli bufor pusty -> None.
    """

    if len(buffer) == 0:
        return None
    if len(buffer) == 1:
        return buffer[0]

    img1 = buffer[0]
    img2 = buffer[-1]

    orb = cv2.ORB_create(40)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return img1  # brak punktów, zwracamy pierwszy obraz

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return img1  # za mało punktów do homografii

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    all_corners = np.concatenate([
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
        warped_corners
    ])

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 10)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 10)

    translation = [-xmin, -ymin]
    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(img2, T @ H, (xmax - xmin, ymax - ymin))
    result[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = img1

    return result

