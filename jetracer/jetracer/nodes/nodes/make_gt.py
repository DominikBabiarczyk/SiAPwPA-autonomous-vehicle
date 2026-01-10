from jetracer.nodes.perception.splain_tracking.make_splain import fit_reference_poly, extract_points_from_binary
from jetracer.nodes.perception.splain_tracking.main_line_preprocessing import OrangeBinaryProcessor
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def draw_points_on_image(binary_img, x_pts, y_pts, px2m, poly=None,
                         max_poly_x=20.0, point_color=(0, 0, 255),
                         poly_color=(0, 255, 0), origin_color=(255, 0, 0)):
    """Build a BGR visualization image from a binary bird-eye image.

    Draws:
      - original binary pixels (gray)
      - extracted path points (small circles, point_color)
      - fitted polynomial (poly_color) if coeffs provided
      - vehicle origin (bottom-center) as a triangle/origin_color

    NOTE: Heading arrow has been removed.
    """

    if binary_img is None:
        return None

    img = np.asarray(binary_img)
    if img.ndim != 2:
        img = img[..., 0]

    # normalize to 0/255 uint8
    if img.dtype != np.uint8:
        img = (img > 0).astype(np.uint8) * 255
    else:
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)

    # make 3-channel BGR base
    if cv2 is not None:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = np.stack([img, img, img], axis=-1)

    h, w = base.shape[:2]
    center_x = w // 2

    for xf, yf in zip(x_pts, y_pts):
        col = int(round(center_x + yf / px2m))
        row = int((h - 1) - round(xf / px2m))
        if 0 <= row < h and 0 <= col < w:
            if cv2 is not None:
                cv2.circle(base, (col, row), 3, point_color, -1)
            else:
                r0 = max(0, row - 2); r1 = min(h, row + 3)
                c0 = max(0, col - 2); c1 = min(w, col + 3)
                base[r0:r1, c0:c1, :] = point_color

    if poly is not None and len(x_pts) > 0:
        xs_curve = np.linspace(0.0, max(max_poly_x, min(max_poly_x, np.max(x_pts))), 300)
        ys_curve = np.polyval(poly, xs_curve)

        prev_pixel = None
        for xf, yf in zip(xs_curve, ys_curve):
            col = int(round(center_x + yf / px2m))
            row = int((h - 1) - round(xf / px2m))
            if 0 <= row < h and 0 <= col < w:
                if cv2 is not None:
                    if prev_pixel is not None:
                        cv2.line(base, prev_pixel, (col, row), poly_color, 1)
                    prev_pixel = (col, row)
                else:
                    base[row, col, :] = poly_color

    return base



class GreenMaskPublisher(Node):
	def __init__(self):
		super().__init__('green_mask_publisher')
		self.bridge = CvBridge()

        # Accumulators for frame-averaged distance from image center
		self.total_distance_m = 0.0
		self.processed_frames = 0

		# Subscribe to the raw chase camera images
		self.subscription = self.create_subscription(
			Image,
			'/chase_camera/image_raw',
			self.image_callback,
			qos_profile_sensor_data,
		)

		# Publish the binarized green mask for viewing in rqt_image_view
		self.publisher = self.create_publisher(
			Image,
			'/chase_camera/green_mask',
			qos_profile_sensor_data,
		)
		self.orange_processor = OrangeBinaryProcessor()
              
	def compute_and_overlay_distance_to_line(self, viz, px2m, poly):
		"""Compute Euclidean distance from image center to the fitted line (poly)
		and overlay the measurement and running average on the viz image.
		Distance is measured in meters by sampling the polynomial across the
		visible forward range and finding the nearest sampled point to image center.
		"""
		try:
			if viz is None or poly is None:
				return
			h, w = viz.shape[:2]
			center_x = w // 2
			center_y = h // 2

			# Sample the fitted line across the image height (in meters)
			x_max_m = (h - 1) * px2m
			xs_curve = np.linspace(0.0, x_max_m, 300)
			ys_curve = np.polyval(poly, xs_curve)

			# Convert sampled curve to pixel coords
			cols = center_x + (ys_curve / px2m)
			rows = (h - 1) - (xs_curve / px2m)

			# Nearest sampled point to image center
			dx = cols - center_x
			dy = rows - center_y
			dists_px = np.sqrt(dx * dx + dy * dy)
			idx = int(np.argmin(dists_px))

			# Distance in meters
			dist_m = float(dists_px[idx] * px2m)
			self.total_distance_m += dist_m
			self.processed_frames += 1
			avg_dist_m = self.total_distance_m / max(1, self.processed_frames)

			# Overlay line from center to nearest point and labels
			row = int(round(rows[idx]))
			col_line = int(round(cols[idx]))
			cv2.circle(viz, (center_x, center_y), 4, (200, 200, 200), -1)
			if 0 <= row < h and 0 <= col_line < w:
				cv2.line(viz, (center_x, center_y), (col_line, row), (255, 255, 0), 2)
				cv2.circle(viz, (col_line, row), 5, (0, 255, 255), -1)

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(viz, f"dist_from_center (2D): {dist_m:.3f} m", (10, 30), font, 0.7, (255, 255, 0), 2)
			cv2.putText(viz, f"avg_dist: {avg_dist_m:.3f} m over {self.processed_frames}", (10, 60), font, 0.7, (255, 255, 0), 2)
		except Exception as e:
			self.get_logger().warn(f'Overlaying distance failed: {e}')

	

	def image_callback(self, msg: Image):
		try:
			bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		except Exception as e:
			self.get_logger().error(f'Image conversion failed: {e}')
			return

		# Convert to HSV and threshold for green regions
		# hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		# import numpy as np
		# zeros = np.zeros(hsv.shape[:2], dtype=np.uint8)
		mask = self.orange_processor.to_binary(bgr)
		#237:270, 367:507
		mask[367:507, 237:270] = 0
		

		px2m = 0.05
		x_pts, y_pts = extract_points_from_binary(mask, px2m)
		poly = fit_reference_poly(x_pts, y_pts, deg=3, anchor_origin=True)
		viz = draw_points_on_image(mask, x_pts, y_pts, px2m, poly)

		# Compute and overlay distance to fitted line
		self.compute_and_overlay_distance_to_line(viz, px2m, poly)

		cv2.imwrite('green_mask_viz.png', viz)

		try:
			mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
			mask_msg.header = msg.header  # keep original timestamp/frame
			self.publisher.publish(mask_msg)
		except Exception as e:
			self.get_logger().error(f'Publishing mask failed: {e}')


def main(args=None):
	rclpy.init(args=args)
	node = GreenMaskPublisher()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
