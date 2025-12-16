import numpy as np
import cv2


def draw_obstacle_points_on_image(image, points, color=(255,255,0), radius=4):
    """Rysuje wokół każdego punktu biały okrąg o promieniu 40px (współrzędne w układzie obrazu: (col, row))."""
    circle_radius = 40
    circle_color = (255, 255, 255)  # biały
    thickness = 2  # grubość linii okręgu
    if image is not None and points is not None:
        h, w = image.shape[:2]
        center_x = w // 2
        center_y = h // 2
        for point in points:
            col, row = int(round(point[0])), int(round(point[1]))
            if 0 <= row < h and 0 <= col < w:
                # Wektor odległości od środka obrazu
                dx = col - center_x
                dy = row - center_y
                # Przesunięcie środka półokręgu w kierunku przeciwnym do położenia punktu
                new_col = int(round(col - dx))
                new_row = int(round(row - dy))
                # pionowa oś symetrii: półokrąg po tej samej stronie co punkt względem środka
                if col < center_x:
                    # punkt po lewej stronie obrazu -> półokrąg na prawo (od -90 do 90 stopni)
                    start_angle = -90
                    end_angle = 90
                else:
                    # punkt po prawej stronie obrazu -> półokrąg na lewo (od 90 do 270 stopni)
                    start_angle = 90
                    end_angle = 270
                if 0 <= new_row < h and 0 <= new_col < w:
                    cv2.ellipse(image, (new_col, new_row), (circle_radius, circle_radius), 0, start_angle, end_angle, circle_color, thickness)
    return image

def extract_points_from_binary(binary_img, px2m):
    """Return arrays of (x_forward_m, y_lateral_m) from binary image.

    We assume image origin is top-left; forward is +y (downwards in image).
    Vehicle is placed at bottom center (y = height-1), so we convert
    image row to forward distance measured from vehicle position.
    """
    h, w = binary_img.shape
    ys, xs = np.where(binary_img > 0)
    # forward distance from vehicle (vehicle at bottom row)
    x_forward_px = (h - 1) - ys
    x_forward_m = x_forward_px * px2m
    center_x = w // 2
    y_lateral_m = (xs - center_x) * px2m
    # keep points sorted by forward distance
    order = np.argsort(x_forward_m)
    return x_forward_m[order], y_lateral_m[order]

def fit_reference_poly(x, y, deg=5):
    if len(x) < deg + 1:
        deg = max(1, len(x) - 1)
    coeffs = np.polyfit(x, y, deg)
    return coeffs

def draw_points_on_image(binary_img, x_pts, y_pts, px2m, poly=None,
                         max_poly_x=20.0, point_color=(0, 0, 255),
                         poly_color=(0, 255, 0), origin_color=(255, 0, 0)):
    """Build a BGR visualization image from a binary bird-eye image.

    Draws:
      - original binary pixels (gray)
      - extracted path points (small circles, point_color)
      - fitted polynomial (poly_color) if coeffs provided
      - vehicle origin (bottom-center) as a triangle/origin_color
      - heading reference arrow at origin if poly provided

    Parameters:
      binary_img : 2D np.ndarray (binary 0/1 or 0/255)
      x_pts, y_pts : arrays of forward (m) and lateral (m) coordinates
      px2m : meters per pixel scale used for extraction
      poly : optional polynomial coeffs (highest degree first)
      max_poly_x : forward distance (m) up to which polynomial is drawn
      *_color : BGR colors

    Returns: HxWx3 uint8 BGR image.
    """
    if binary_img is None:
        return None
    img = np.asarray(binary_img)
    if img.ndim != 2:
        # reduce to first channel if multi-channel
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

    # Draw path points
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

    # Draw polynomial curve if available
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

        # Heading arrow at origin using derivative at x=0
        try:
            dy0 = np.polyval(np.polyder(poly), 0.0)
            # direction vector in pixel space (forward negative rows, lateral columns)
            heading_len_m = 3.0
            vx_m = heading_len_m  # forward distance
            vy_m = dy0 * heading_len_m
            tip_col = int(round(center_x + vy_m / px2m))
            tip_row = int((h - 1) - round(vx_m / px2m))
            if cv2 is not None and 0 <= tip_row < h and 0 <= tip_col < w:
                cv2.arrowedLine(base, (center_x, h - 1), (tip_col, tip_row), (0, 255, 255), 2, tipLength=0.3)
        except Exception:
            pass

    # Vehicle origin marker (triangle)
    if cv2 is not None:
        p0 = (center_x, h - 1)
        p1 = (center_x - 8, h - 25)
        p2 = (center_x + 8, h - 25)
        cv2.fillPoly(base, [np.array([p0, p1, p2])], origin_color)
    else:
        row0 = h - 1
        base[row0-25:row0, center_x-2:center_x+3, :] = origin_color

    return base

def get_poly_from_binary_image(binary_img, set_points_in_obstacle, px2m, img_publisher):
    try:
      binary_img = draw_obstacle_points_on_image(binary_img, set_points_in_obstacle, color=(255,255,0), radius=4)

      x_pts, y_pts = extract_points_from_binary(binary_img, px2m)
    except Exception:
        return 0.0

    if len(x_pts) < 4:
        # za mało punktów do dopasowania sensownego polinomu
        return 0.0

    # dopasuj wielomian referencyjny (używamy stopnia 3 tak jak w symulacji)
    try:
        poly = fit_reference_poly(x_pts, y_pts, deg=2)
    except Exception:
        return 0.0
    

    # Build visualization image with points & polynomial and publish
    try:
        viz = draw_points_on_image(binary_img, x_pts, y_pts, px2m, poly)
        print("punkty na splaine", set_points_in_obstacle)
        print("wysokość obrazu", binary_img.shape)
        # Rysuj punkty obstacle przez osobną funkcję
        if viz is not None:
            # prefer direct color frame update
            if hasattr(img_publisher, 'update_frame'):
                img_publisher.update_frame(viz)
            else:
                img_publisher.update_binary_frame(viz)
            # publish immediately if method exists
            if hasattr(img_publisher, 'publish_now'):
                img_publisher.publish_now()
    except Exception as e:
        print(f"Błąd podczas wizualizacji punktów: {e}")
        pass
    return poly