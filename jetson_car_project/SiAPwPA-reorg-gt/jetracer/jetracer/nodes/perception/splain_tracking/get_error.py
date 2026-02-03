import numpy as np

class ImageErrorCalculator:
	def __init__(self):
		pass

	def calculate(self, binary_img):
		M = np.sum(binary_img)
		if M == 0:
			h, w = binary_img.shape
			return None, None, None
		indices = np.argwhere(binary_img > 0)
		cy = np.mean(indices[:, 0])
		cx = np.mean(indices[:, 1])
		h, w = binary_img.shape
		center_x = 0.5
                #center_x = 0.5
                #cx = cx / w
		error_x = (cx/w) - center_x
		return error_x
