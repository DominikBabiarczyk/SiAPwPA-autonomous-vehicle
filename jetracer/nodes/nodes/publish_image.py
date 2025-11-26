import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/splain', 10)

        self.bridge = CvBridge()

        # Timer: wywołanie co 0.1 s (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_image)

        # Stały biały obraz 480x640
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
            # gray -> BGR
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if hasattr(cv2, 'cvtColor') else np.stack([arr]*3, axis=-1)

        # ensure uint8
        if arr.dtype != np.uint8:
            # try to normalize/convert
            arr = (255 * (arr - arr.min()) / max(1e-8, (arr.max() - arr.min()))).astype(np.uint8)

        self.frame = arr


    def update_binary_frame(self, binary_img: np.ndarray):
        """Accept binary image (0/1 or 0/255) and convert to BGR uint8 for publishing."""
        if binary_img is None:
            return
        b = np.asarray(binary_img)
        # If it's already color (H,W,3), compute a mask from luminance
        if b.ndim == 3 and b.shape[2] == 3:
            # convert to gray for thresholding (use simple average if cv2 missing)
            if cv2 is not None:
                gray = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = (b.astype(np.float32).mean(axis=2)).astype(np.uint8)
            mask = (gray > 0).astype(np.uint8) * 255
            bgr = np.stack([mask, mask, mask], axis=-1)
        else:
            # collapse to 2D if multi-channel but not 3 (take first channel)
            if b.ndim == 3:
                b = b[..., 0]
            # threshold to 0/1
            b_bin = (b > 0).astype(np.uint8) * 255
            # convert to BGR
            bgr = np.stack([b_bin, b_bin, b_bin], axis=-1)

        self.update_frame(bgr)

    def publish_now(self):
        """Publish current frame immediately (safe, avoids extra timer handling)."""
        try:
            frame = np.asarray(self.frame)
            # ensure uint8
            if frame.dtype != np.uint8:
                frame = (255 * (frame - frame.min()) / max(1e-8, (frame.max() - frame.min()))).astype(np.uint8)
            if frame.ndim == 2:
                encoding = 'mono8'
            else:
                encoding = 'bgr8'
            msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
            self.publisher_.publish(msg)
            s = int(self.frame.sum())
            self.get_logger().info(f"Published image now sum={s}")
        except Exception:
            # avoid raising errors from publish helper
            pass

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
        self.publisher_.publish(msg)
        # log a small debug value so we can see when the image changes
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
