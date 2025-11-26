from time import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class CollisionChecker(Node):
    # Pole przechowujące ostatni odebrany obraz BGR
    last_image = None

    def wait_for_odometry(self, timeout_sec=5.0):
        start = time.time()
        while self.last_image is None and (time.time() - start) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_image is not None

    def is_green_in_rect(self, timeout=5, continuous=False):
        """
        Czeka na pierwszy obraz do timeoutu (sekundy), potem sprawdza obecność trawy w prostokącie.
        Jeśli continuous=True, działa w pętli i sprawdza na bieżąco (Ctrl+C aby przerwać).
        """
        rclpy.spin_once(self, timeout_sec=0.1)
        import time
        start = time.time()
        while self.last_image is None and (time.time() - start < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.last_image is None:
            print("Nie odebrano obrazu w zadanym czasie!")
            return False
        def check():
            mask = self.binarize_green(self.last_image)
            x1, y1 = (185, 239)
            x2, y2 = (356, 542)
            roi = mask[y1:y2, x1:x2]
            return cv2.countNonZero(roi) > 0
        if not continuous:
            return check()
        else:
            print("Tryb ciągły: sprawdzanie obecności trawy w prostokącie (Ctrl+C aby przerwać)")
            try:
                while True:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    result = check()
                    print(f"Trawa w prostokącie: {result}")
            except KeyboardInterrupt:
                print("Zakończono tryb ciągły.")
    
    def binarize_green(self, image):
        # Konwertuj obraz do przestrzeni HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Zakresy dla koloru zielonego z dużym zapasem
        hsv = self.draw_black_rectangle(hsv, (234, 461), (295, 516))
        lower_green = (30, 40, 40)
        upper_green = (90, 255, 255)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask
    
    def draw_black_rectangle(self, image, pt1, pt2):
        # Rysuje czarny prostokąt o zadanych rogach na obrazie
        cv2.rectangle(image, pt1, pt2, (0, 0, 0), thickness=-1)
        return image
    
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/chase_camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.counter = 0
        import time
        self.last_save_time = time.time()

    def listener_callback(self, msg):
        import time
        # current_time = time.time()
        # Zapisz ostatni odebrany obraz BGR jako pole klasy
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Błąd przy konwersji obrazu: {e}')
            return
        # if current_time - self.last_save_time >= 3.0:
        try:
            # Rysuj czarny prostokąt przed zapisem
            bin_img = self.binarize_green(self.last_image)
            filename = f'chase_frame_{self.counter:04d}.png'
            # cv2.imwrite(filename, bin_img)
            # self.get_logger().info(f'Zapisano obraz: {filename}')
            self.counter += 1
            # self.last_save_time = current_time
        except Exception as e:
            self.get_logger().error(f'Błąd przy zapisie obrazu: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CollisionChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()