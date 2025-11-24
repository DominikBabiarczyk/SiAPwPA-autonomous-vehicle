import numpy as np
import cv2

class OrangeBinaryProcessor:
	"""
	Klasa do binaryzacji obrazu po kolorze pomarańczowym (HSV).
	Zakres jest szeroki, by wykryć różne odcienie pomarańczowego.
	"""
	def __init__(self):
		self.lower_orange = np.array([10, 100, 100], dtype=np.uint8)
		self.upper_orange = np.array([30, 255, 255], dtype=np.uint8)

	def to_binary(self, img):
		if len(img.shape) == 3 and img.shape[2] == 3:
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		else:
			hsv = img  
		mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
		return mask
