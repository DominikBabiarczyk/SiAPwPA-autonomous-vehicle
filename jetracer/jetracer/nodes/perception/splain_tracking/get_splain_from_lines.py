import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class LaneSpline:
    def __init__(self, smooth=5.0, step=4, curvature_scale=1.0):
        """
        :param smooth: parametr wygładzenia dla splajnu
        :param step: co ile wierszy pobierać punkty linii
        :param curvature_scale: skalowanie krzywizny (np. do sterowania)
        """
        self.smooth = smooth
        self.step = step
        self.curvature_scale = curvature_scale
        self.points = None
        self.tck = None
        self.u = None
        self.x_smooth = None
        self.y_smooth = None
        self.curvature = None

    def extract_centerline_points(self, binary_img):
        """Wyznacza punkty środkowe linii (centroid w każdej linii obrazu)."""
        h, w = binary_img.shape
        pts = []
        for y in range(h-1, -1, -self.step):
            xs = np.where(binary_img[y, :] > 0)[0]
            if len(xs) > 0:
                cx = np.mean(xs)
                pts.append((cx, y))
        self.points = np.array(pts)
        return self.points

    def filter_points(self, window_size=5):
        """Wygładza punkty np. filtrem medianowym po współrzędnej x."""
        if self.points is None or len(self.points) < 5:
            return None
        x = self.points[:, 0].astype(np.float32)
        y = self.points[:, 1].astype(np.float32)
        x_med = cv2.medianBlur(x.astype(np.uint8), window_size).astype(np.float32)
        self.points = np.column_stack([x_med, y])
        return self.points

    def fit_spline(self):
        """Dopasowuje splajn parametryczny (B-spline)."""
        if self.points is None or len(self.points) < 4:
            return None
        x = self.points[:, 0]
        y = self.points[:, 1]
        self.tck, self.u = interpolate.splprep([x, y], s=self.smooth)
        return self.tck

    def sample_spline(self, num=200):
        """Generuje gładki tor (x, y) z dopasowanego splajnu."""
        if self.tck is None:
            return None
        unew = np.linspace(0, 1, num)
        x_smooth, y_smooth = interpolate.splev(unew, self.tck)
        self.x_smooth, self.y_smooth = np.array(x_smooth), np.array(y_smooth)
        return self.x_smooth, self.y_smooth

    def spline_mask(self, binary_img=None, shape=None):
        """Tworzy binarną maskę (0/1) z linią splajnu o grubości 1 piksela.

        Parametry:
          binary_img: opcjonalny obraz wejściowy (używany do uzyskania wymiarów maski)
          shape: alternatywnie krotka (h, w) jeśli nie przekazano obrazu

        Zwraca:
          mask: numpy.ndarray uint8 (h, w) z wartościami 0 lub 1; None jeśli brak danych splajnu.
        """
        if self.x_smooth is None or self.y_smooth is None:
            return None
        if binary_img is not None:
            h, w = binary_img.shape[:2]
        elif shape is not None:
            h, w = shape
        else:
            h = int(np.ceil(np.max(self.y_smooth))) + 2
            w = int(np.ceil(np.max(self.x_smooth))) + 2
            h = max(h, 1); w = max(w, 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        xs = np.rint(self.x_smooth).astype(int)
        ys = np.rint(self.y_smooth).astype(int)
        for i in range(len(xs) - 1):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i+1], ys[i+1]
            if (x1 < 0 and x2 < 0) or (y1 < 0 and y2 < 0) or (x1 >= w and x2 >= w) or (y1 >= h and y2 >= h):
                continue
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
            cv2.line(mask, (x1, y1), (x2, y2), color=1, thickness=1)
        return mask

    def compute_curvature(self):
        """Oblicza krzywiznę splajnu wzdłuż toru."""
        if self.tck is None:
            return None
        dx, dy = interpolate.splev(self.u, self.tck, der=1)
        ddx, ddy = interpolate.splev(self.u, self.tck, der=2)
        curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)
        self.curvature = curvature * self.curvature_scale
        return self.curvature

    def visualize(self, binary_img, show_points=True, show_curvature=False):
        """Rysuje wynik na obrazie."""
        plt.figure(figsize=(6, 8))
        plt.imshow(binary_img, cmap='gray')
        if show_points and self.points is not None:
            plt.plot(self.points[:, 0], self.points[:, 1], 'ro', markersize=3, label='punkty środkowe')
        if self.x_smooth is not None:
            plt.plot(self.x_smooth, self.y_smooth, 'b-', linewidth=2, label='splajn')
        if show_curvature and self.curvature is not None:
            plt.title(f'Max curvature: {np.max(np.abs(self.curvature)):.4f}')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(f"camera/splain_tracking/main_lines/main_lines_spline.png")
        plt.close()

    def process(self, binary_img):
        """Główna funkcja łącząca wszystkie kroki."""
        self.extract_centerline_points(binary_img)
        self.filter_points()
        self.fit_spline()
        self.sample_spline()
        self.compute_curvature()
        mask = self.spline_mask(binary_img=binary_img)
        return mask
        # return {
        #     "points": self.points,
        #     "spline": (self.x_smooth, self.y_smooth),
        #     "curvature": self.curvature,
        #     "mask": mask
        # }
