from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.perception.vision.transform import BirdView
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False

class PreprocessSensor(Node):
    def __init__(self, topic="/rs_front/image"):
        super().__init__('preprocess_sensor')
        self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
        self.last_image = None
        self.frame_counter = 0  # licznik zapisanych ramek
        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, 10)
        self.get_logger().info(f"Subskrypcja: {topic}")
        self.image_procesor = ImageProcessor()

    def image_callback(self, msg):
        if self.bridge:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            # fallback: konwersja ręczna jeśli nie ma cv_bridge
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.last_image = arr.reshape((msg.height, msg.width, 3))

        # Zapisz każdą odebraną ramkę do pliku PNG
        if self.last_image is not None:
            cv2.imwrite(f"simple_i_{self.frame_counter:06d}.png", self.last_image)
            self.frame_counter += 1   

    def wait_for_image(self, timeout_sec=5.0):
        """Czeka na pierwszą wiadomość z obrazem"""
        import time
        start = time.time()
        while self.last_image is None and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_image is not None

    def get_sensor_detect(self):
        # Odbierz nowe wiadomości przed przetworzeniem
        rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.last_image is None:
            self.get_logger().warning("sensor nie dziala: brak obrazu!")
            print("Brak obrazu do przetworzenia.")
            return None

        # Pracuj na kopii najnowszego obrazu
        image = self.last_image.copy()
        
        image_transform = BirdView()
        bird_view_image = image_transform.apply_transform(image)

        lines = self.image_procesor.get_lines(bird_view_image)

        if lines is not None:
            # cv2.imwrite(f"frame_{self.frame_counter:06d}.png", lines)
            self.frame_counter += 1

        height, width = lines.shape[:2]
        r = 50
        mid = width // 2
        roi1 = lines[height-1:, mid-r:mid+r]
        roi2 = lines[height-41:height-40, mid-r:mid+r]
        roi3 = lines[height-81:height-80, mid-r:mid+r]
        return np.squeeze(np.array([roi1, roi2, roi3]))

def main():
    rclpy.init()
    node = PreprocessSensor()
    try:
        if node.wait_for_image():  # Poczekaj na pierwszy obraz
            roi = node.get_sensor_detect()
            if roi is not None:
                cv2.imwrite("ROI.png", roi[0])
        else:
            print("Nie otrzymano obrazu w zadanym czasie")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()