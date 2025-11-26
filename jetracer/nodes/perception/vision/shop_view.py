#!/usr/bin/env python3
from jetracer.nodes.perception.vision.image_preprocessing import ImageProcessor
from jetracer.nodes.perception.splain_tracking.main_line_preprocessing import OrangeBinaryProcessor
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from jetracer.nodes.perception.vision.transform import BirdView

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_CV_BRIDGE = False

class BirdViewSaver(Node):
    def __init__(self, topic='/rs_front/image'):
        super().__init__('bird_view_saver')
        self.bridge = CvBridge() if _HAS_CV_BRIDGE else None
        self.subscription = self.create_subscription(
            Image, topic, self.image_callback, 10)
        self.get_logger().info(f'Subskrypcja: {topic}')
        self.saved = False
        self.image_procesor = ImageProcessor()
        self.orange_processor = OrangeBinaryProcessor()


    def image_callback(self, msg):
        if self.saved:
            return
        if self.bridge:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            cv_img = arr.reshape((msg.height, msg.width, 3))
        cv2.imshow('Kamera', cv_img)
        cv2.waitKey(1)
        transformation = BirdView()
        bird_view = transformation.apply_transform(cv_img)
        # bird_line_image = self.image_procesor.get_lines(bird_view)
        lines = self.orange_processor.to_binary(bird_view)
        
        cv2.imwrite('bird_line_image.png', lines)

        self.get_logger().info('Zapisano bird_view.png')
        self.saved = True
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = BirdViewSaver()
    try:
        while rclpy.ok() and not node.saved:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
