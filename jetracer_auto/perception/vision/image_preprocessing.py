import cv2
import numpy as np

# Prosta klasa do przetwarzania obrazu: binarizacja i detekcja krawędzi
class ImageProcessor:
	def __init__(self, threshold=128, canny1=100, canny2=200):
		self.threshold = threshold
		self.canny1 = canny1
		self.canny2 = canny2

	def to_binary(self, img):
		# Założenie: img w formacie RGB
		# Parametry progowe
		min_brightness = self.threshold  # np. 128
		max_diff = 20  

		# Rozdziel kanały
		R = img[:, :, 0].astype(np.int16)
		G = img[:, :, 1].astype(np.int16)
		B = img[:, :, 2].astype(np.int16)

		diff_rg = np.abs(R - G)
		diff_rb = np.abs(R - B)
		diff_gb = np.abs(G - B)
		diff_mask = (diff_rg < max_diff) & (diff_rb < max_diff) & (diff_gb < max_diff)

		mean_rgb = ((R + G + B) / 3)
		bright_mask = mean_rgb > min_brightness

		mask = diff_mask & bright_mask
		binary = np.zeros_like(R, dtype=np.uint8)
		binary[mask] = 255
		return binary

	def canny_edges(self, img):
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		edges = cv2.Canny(gray_img, self.canny1, self.canny2)
		return edges

	def get_lines(self, img):
		kernel = np.ones((3,3), np.uint8)
		mask_binary = self.to_binary(img)
		mask_canny = self.canny_edges(img)
		mask_canny = cv2.dilate(mask_canny, kernel, iterations=1)

		
		contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
		result = np.zeros_like(mask_binary)

		for cnt in contours:
        # Utwórz tymczasową maskę dla danego obiektu
				obj_mask = np.zeros_like(mask_binary)
				cv2.drawContours(obj_mask, [cnt], -1, 255, -1)

        # Sprawdź, czy ten obiekt styka się z mask2
				if np.any(cv2.bitwise_and(obj_mask, mask_canny)):
					result = cv2.bitwise_or(result, obj_mask)

		return result
