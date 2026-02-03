#!/usr/bin/env python3
"""Publish a binary numpy image (2D) as a ROS2 Image message.

This node accepts a 2D numpy array (values 0/1 or 0/255) and publishes it
as `mono8` (or `rgb8` if requested). It can be used from Python by
instantiating `BinaryImagePublisher` and calling `publish_array()` or via
the CLI to publish a .npy or image file once or in a loop.

Examples:
  # publish a PNG once
  python3 binary_image_publisher.py --file bird_line_image.png --topic /binary_image --rate 1

  # publish an .npy array in loop
  python3 binary_image_publisher.py --file /tmp/binary.npy --loop --rate 2
"""
import argparse
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
except Exception:
    _HAS_CV_BRIDGE = False

try:
    import cv2
except Exception:
    cv2 = None


class BinaryImagePublisher(Node):
    def __init__(self, topic: str = '/binary_image'):
        super().__init__('binary_image_publisher')
        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge() if _HAS_CV_BRIDGE else None

    def publish_array(self, binary_img: np.ndarray, encoding: str = 'mono8', frame_id: str = 'camera'):
        """Publish a numpy array as a ROS2 Image message.

        binary_img: 2D (H,W) or 3D (H,W,3) numpy array. If values are 0/1 they
        will be scaled to 0/255. The function is robust to uint8/float inputs.
        encoding: 'mono8' or 'rgb8'. If a 2D array is provided and 'rgb8' is
        requested, the image will be stacked into 3 channels.
        """
        if not isinstance(binary_img, np.ndarray):
            raise TypeError('binary_img must be a numpy ndarray')

        arr = binary_img.copy()
        # normalize to uint8 0..255
        if arr.dtype != np.uint8:
            # assume in [0,1] or floats
            try:
                arr = (arr * 255).astype('uint8')
            except Exception:
                arr = arr.astype('uint8')
        else:
            if arr.max() <= 1:
                arr = (arr * 255).astype('uint8')

        # Prepare image for publishing
        if arr.ndim == 2:
            if encoding == 'rgb8':
                if cv2 is not None:
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                else:
                    img = np.stack([arr, arr, arr], axis=-1)
                enc = 'rgb8'
            else:
                img = arr
                enc = 'mono8'
        elif arr.ndim == 3:
            # assume already color HxWx3
            img = arr
            enc = 'rgb8'
        else:
            raise ValueError('Unsupported array shape for image publishing')

        # Create ROS Image message
        if self.bridge:
            msg = self.bridge.cv2_to_imgmsg(img, encoding=enc)
        else:
            msg = Image()
            msg.height = img.shape[0]
            msg.width = img.shape[1]
            msg.encoding = enc
            msg.step = img.shape[1] * (3 if enc.startswith('rgb') else 1)
            msg.data = img.tobytes()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        self.pub.publish(msg)
        self.get_logger().debug(f'Published binary image ({img.shape}, enc={enc})')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file', '-f', required=True, help='Path to .npy or image file to publish')
    p.add_argument('--topic', '-t', default='/binary_image', help='ROS topic to publish to')
    p.add_argument('--rate', '-r', type=float, default=1.0, help='Publish rate in Hz')
    p.add_argument('--loop', action='store_true', help='Keep publishing in a loop')
    p.add_argument('--rgb', action='store_true', help='Publish as rgb8 (stacks gray to 3 channels)')
    return p.parse_args()


def load_file(path: str) -> np.ndarray:
    if path.lower().endswith('.npy'):
        return np.load(path)
    if cv2 is None:
        raise RuntimeError('cv2 is required to load non-npy images')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    return img


def main():
    args = parse_args()
    if not os.path.exists(args.file):
        raise FileNotFoundError(args.file)

    img = load_file(args.file)
    # Ensure binary-like values: threshold if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8')
    # If not strictly binary, threshold at >0
    if img.max() > 1 and img.max() <= 255:
        # keep as-is
        pass
    elif img.max() <= 1:
        img = (img * 255).astype('uint8')

    rclpy.init()
    node = BinaryImagePublisher(topic=args.topic)
    try:
        rate = float(args.rate)
        period = 1.0 / max(rate, 1e-3)
        node.get_logger().info(f'Publishing {args.file} -> {args.topic} (loop={args.loop})')
        while rclpy.ok():
            enc = 'rgb8' if args.rgb else 'mono8'
            node.publish_array(img, encoding=enc)
            if not args.loop:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
