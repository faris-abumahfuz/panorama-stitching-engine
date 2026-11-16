import cv2
import numpy as np


def compute_canvas_bounds(shapes: list[tuple[int, int]], homographies: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    """Computes unified bounding box and translation offset matrix for warped images."""
    all_corners = []
    for (h, w), H in zip(shapes, homographies):
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
        warped = H @ corners
        warped /= warped[2:3, :]
        all_corners.append(warped[:2, :].T)

    pts = np.vstack(all_corners)
    min_x, min_y = np.floor(pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(pts.max(axis=0)).astype(int)

    T_offset = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    canvas_size = (int(max_x - min_x), int(max_y - min_y))
    return T_offset, canvas_size


def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Trims outer zero-padded regions around the stitched panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]