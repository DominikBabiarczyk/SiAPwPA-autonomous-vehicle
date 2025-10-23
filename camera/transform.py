import cv2
import numpy as np

class BirdView:
  def __init__(self, src_points=None, dst_points=None):
    self.src_points = np.float32([
      [91, 359],
      [560, 359],
      [370, 217],
      [271, 217]
    ])

    self.dst_points = np.float32([
      [0 + 100, 300],
      [87 + 100, 300],
      [87 + 100, 0],
      [0 + 100, 0 ]
    ])
    self.set_transform(self.src_points, self.dst_points)

  def set_transform(self, src_points, dst_points):
    # Oblicz macierz transformacji na podstawie punktów źródłowych i docelowych
    self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

  def apply_transform(self, img):
    if self.transform_matrix is not None:
      # Zastosuj transformację do obrazu
      width = int(np.max(self.dst_points[:, 0]))
      height = int(np.max(self.dst_points[:, 1]))
      return cv2.warpPerspective(img, self.transform_matrix, (width+100, height))
    return img