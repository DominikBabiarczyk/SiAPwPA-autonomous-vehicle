import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImagePublisher(Node):
    def __init__(self, topic='camera/splain', name_node='image_publisher'):
        super().__init__(name_node)
        self.publisher_ = self.create_publisher(Image, topic, 10)

        self.bridge = CvBridge()

        self.timer = self.create_timer(0.1, self.publish_image)

        self.frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    def update_frame(self, new_frame: np.ndarray):
        """Replace the current frame to be published.

        Accepts BGR (H,W,3) uint8 or grayscale (H,W) arrays. Function will
        convert grayscale to BGR automatically.
        """
        if new_frame is None:
            return
        arr = np.asarray(new_frame)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if hasattr(cv2, 'cvtColor') else np.stack([arr]*3, axis=-1)

        if arr.dtype != np.uint8:
            arr = (255 * (arr - arr.min()) / max(1e-8, (arr.max() - arr.min()))).astype(np.uint8)

        self.frame = arr


    # def update_binary_frame(self, binary_img: np.ndarray):
    #     """Accept binary image (0/1 or 0/255) and convert to BGR uint8 for publishing."""
    #     if binary_img is None:
    #         return
    #     b = np.asarray(binary_img)
    #     if b.ndim == 3 and b.shape[2] == 3:
    #         if cv2 is not None:
    #             gray = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #         else:
    #             gray = (b.astype(np.float32).mean(axis=2)).astype(np.uint8)
    #         mask = (gray > 0).astype(np.uint8) * 255
    #         bgr = np.stack([mask, mask, mask], axis=-1)
    #     else:
    #         if b.ndim == 3:
    #             b = b[..., 0]
    #         b_bin = (b > 0).astype(np.uint8) * 255
    #         bgr = np.stack([b_bin, b_bin, b_bin], axis=-1)

    #     self.update_frame(bgr)

    def publish_now(self):
        """Publish current frame immediately (safe, avoids extra timer handling)."""
        try:
            frame = np.asarray(self.frame)
            if frame.dtype != np.uint8:
                frame = (255 * (frame - frame.min()) / max(1e-8, (frame.max() - frame.min()))).astype(np.uint8)
            if frame.ndim == 2:
                encoding = 'mono8'
            else:
                encoding = 'bgr8'
            msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
            self.publisher_.publish(msg)
            #s = int(self.frame.sum())
            #self.get_logger().info(f"Published image now sum={s}")
        except Exception:
            pass

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
        self.publisher_.publish(msg)
        try:
            s = int(self.frame.sum())
        except Exception:
            s = 0
        self.get_logger().info(f"Published image sum={s}")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
