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
                self.lower_red1 = (0, 50, 50)
                self.upper_red1 = (5, 255, 255)

                # Upper red range (155-180)
                self.lower_red2 = (175, 50, 50)
                self.upper_red2 = (180, 255, 255)
        def to_binary(self, img):
                ## optional smoothing of image
                img  = cv2.blur(img,(3,3))
                img = cv2.medianBlur(img,3)
                img= cv2.GaussianBlur(img,(3,3),0)
                img= cv2.bilateralFilter(img,9,75,75)
                if len(img.shape) == 3 and img.shape[2] ==3:
                        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                else:
                        hsv_image = img

		
                #mask1 = cv2.inRange(hsv, np.array([0, 50, 50], dtype=np.uint8), np.array([5, 255, 255], dtype=np.uint8))
                #mask2 = cv2.inRange(hsv, np.array([160, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
                #mask = cv2.bitwise_or(mask1, mask2)
                mask = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
                mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
                mask = cv2.bitwise_or(mask, mask2)

                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

                return mask
                
