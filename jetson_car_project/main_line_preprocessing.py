import numpy as np
import cv2

#class OrangeBinaryProcessor:
#	"""
#	Klasa do binaryzacji obrazu po kolorze pomarańczowym (HSV).
#	Zakres jest szeroki, by wykryć różne odcienie pomarańczowego.
#	"""
#	def __init__(self, lower_orange=None, upper_orange=None):
#		if lower_orange is not None:
#			self.lower_orange = np.array(lower_orange, dtype=np.uint8)
#		else:
#			self.lower_orange = np.array([10, 100, 100], dtype=np.uint8)
#		if upper_orange is not None:
#			self.upper_orange = np.array(upper_orange, dtype=np.uint8)
#		else:
#			self.upper_orange = np.array([30, 255, 255], dtype=np.uint8)
#
#	def to_binary(self, img):
#		if len(img.shape) == 3 and img.shape[2] == 3:
#			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#		else:
#			hsv = img  
#		mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
#		return mask

class OrangeBinaryProcessor:
        def __init__(self):
                self.one = 1
        def to_binary(self, img):
                if len(img.shape) == 3 and img.shape[2] ==3:
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                else:
                        hsv = img
                mask1 = cv2.inRange(hsv, np.array([0, 100, 100], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
                mask2 = cv2.inRange(hsv, np.array([160, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
                mask = cv2.bitwise_or(mask1, mask2)
                return mask
                
