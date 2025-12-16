from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
import numpy as np
from jetracer.nodes.nodes.publish_image import ImagePublisher
from jetracer.nodes.perception.vision.shop_view import BirdView
import cv2

_HAS_CV_BRIDGE = True


import cv2
import numpy as np

def extract_largest_blob(binary_img, min_area=500):
    """
    Zwraca obraz zawierający tylko największy biały obiekt.
    
    Parametry:
        binary_img – obraz binarny (0/255)
        min_area – minimalna wielkość obiektu, aby nie brać pod uwagę linii / szumu

    Zwraca:
        mask – obraz 0/255 z największym spójnym obiektem
    """
    # Upewnij się, że wartości to 0/255
    bin_img = (binary_img > 127).astype(np.uint8) * 255

    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # stats[:, cv2.CC_STAT_AREA] -> lista pól obiektów
    areas = stats[1:, cv2.CC_STAT_AREA]  # pomiń tło (index 0)
    if len(areas) == 0:
        return np.zeros_like(binary_img)

    # filtr anty-linia: odrzucamy obiekty poniżej min_area
    valid_areas = [(i+1, area) for i, area in enumerate(areas) if area >= min_area]
    
    if not valid_areas:
        return np.zeros_like(binary_img)

    # wybór największego obiektu
    largest_label, largest_area = max(valid_areas, key=lambda x: x[1])

    # maska dla największego obiektu
    mask = (labels == largest_label).astype(np.uint8) * 255

    return mask


def get_obstacle_base_points(binary):
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	bottom_points = []

	for cnt in contours:
			# każdy punkt konturu ma postać [[x, y]]
			ys = cnt[:, 0, 1]          # bierzemy współrzędne y
			idx = np.argmax(ys)        # index największego y
			bottom_point = cnt[idx][0] # [x, y]
			bottom_points.append(tuple(bottom_point))
	return bottom_points

def blackout_rows_threshold_5_6(img):
    """
    Zastępuje rzędy obrazu binarnego (0/255) czarnymi pikselami,
    jeśli liczba białych pikseli przekracza 5/6 długości wiersza.
    """
    out = img.copy()
    height, width = img.shape

    # próg = 5/6 szerokości wiersza
    threshold = (5 * width) / 8

    # policz białe piksele w każdym rzędzie
    white_count = np.sum(img == 255, axis=1)

    # znajdź rzędy do wyczyszczenia
    rows_to_blackout = white_count > threshold

    # zamień całe wiersze na czarne (0)
    out[rows_to_blackout, :] = 0

    return out

ABSOLUTE_PATH = "/home/developer/ros2_ws/src/jetracer/resource/depth_image.png"

class DepthImageSubscriber(Node):
	def __init__(self, 
			     topic="/rs_front/depth_image", 
				#  reference_path="camera/splain_tracking/depth_image.png",
				 reference_path=ABSOLUTE_PATH,):
		super().__init__('depth_image_subscriber')
		self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
		self.last_image = None
		self.reference_image = None
		self.subscription = self.create_subscription(
			Image, topic, self.image_callback, 10)
		self.get_logger().info(f"Subskrypcja: {topic}")
		# Publisher for difference image (visible in rqt)
		self.diff_publisher = ImagePublisher(topic='camera/depth_diff', name_node='depth_diff_publisher')
		# Load reference image from file
		import cv2
		try:
			ref = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
			if ref is not None:
				# If loaded as 8-bit, convert to float32 for diff
				if ref.dtype != np.float32:
					ref = ref.astype(np.float32)
				self.reference_image = ref
				self.get_logger().info(f"Wczytano obraz referencyjny: {reference_path}, shape={ref.shape}")
			else:
				self.get_logger().warning(f"Nie udało się wczytać obrazu referencyjnego: {reference_path}")
		except Exception as e:
			self.get_logger().warning(f"Błąd przy wczytywaniu obrazu referencyjnego: {e}")
		self.bird_view_transform = BirdView()
		self.points = None

	def image_callback(self, msg):
		# Convert ROS Image message to numpy array
		if self.bridge:
			depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
		else:
			arr = np.frombuffer(msg.data, dtype=np.float32)
			depth_img = arr.reshape((msg.height, msg.width))

		# Ensure float32 depth
		if depth_img is None:
			return
		if not isinstance(depth_img, np.ndarray):
			return
		if depth_img.dtype != np.float32:
			try:
				depth_img = depth_img.astype(np.float32)
			except Exception:
				pass

		self.last_image = depth_img

		# diff and binarization
		# img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
		# vmin = float(np.min(img))
		# vmax = float(np.max(img))
		# if vmax - vmin < 1e-12:
		# 	img_norm = np.zeros_like(img, dtype=np.uint8)
		# else:
		# 	img_norm = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
		# diff = np.abs(self.reference_image - img_norm)
		# _, diff_bin = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
		# diff_bin = blackout_rows_threshold_5_6(diff_bin)
		# largest_object = extract_largest_blob(diff_bin, min_area=500)
		# cv2.imwrite("received_depth.png", largest_object)
		# bird_view_image = self.bird_view_transform.apply_transform(largest_object.astype(np.uint8))

		img_uint8 = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
		img_uint8 = img_uint8.astype(np.uint8)


		sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, dx=1, dy=0, ksize=3)
		sobel_x = cv2.convertScaleAbs(sobel_x)

		_, diff_bin = cv2.threshold(sobel_x, 58, 255, cv2.THRESH_BINARY)

		self.points = get_obstacle_base_points(diff_bin)
		for point in self.points:
			diff_bin[point[1]-5:point[1]+5, point[0]-5:point[0]+5] = 128  # oznaczenie punktu na szaro

		self.diff_publisher.update_frame(diff_bin)
		self.diff_publisher.publish_now()

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
	
	def get_obstacle_base_points(self):
		return self.points

def main():
	rclpy.init()
	node = DepthImageSubscriber()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()