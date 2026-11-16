import cv2
import numpy as np


def project_points(pts: np.ndarray, img_shape: tuple[int, int], focal_length: float, mode: int) -> np.ndarray:
    """Analytically maps 2D keypoint coordinates into the target projection."""
    if mode == 0 or len(pts) == 0:
        return pts.copy()

    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy

    if mode == 1:  # Cylindrical Outward (Convex / Barrel)
        x_p = focal_length * np.arctan2(x, focal_length) + cx
        y_p = focal_length * (y / np.sqrt(x**2 + focal_length**2)) + cy
    elif mode == 2:  # Cylindrical Inward (Concave Ribbon)
        x_p = focal_length * np.arctan2(x, focal_length) + cx
        y_p = y * (np.sqrt(x**2 + focal_length**2) / focal_length) + cy
    elif mode == 3:  # Spherical
        x_p = focal_length * np.arctan2(x, focal_length) + cx
        y_p = focal_length * np.arctan2(y, np.sqrt(x**2 + focal_length**2)) + cy
    elif mode == 4:  # Stereographic (Circular)
        u, v = x / focal_length, y / focal_length
        r = np.sqrt(u**2 + v**2)
        theta = np.arctan(r)
        r_p = 2.0 * np.tan(theta / 2.0)
        scale = np.where(r > 1e-6, r_p / r, 1.0)
        x_p = focal_length * u * scale + cx
        y_p = focal_length * v * scale + cy
    else:
        return pts.copy()

    return np.column_stack([x_p, y_p]).astype(np.float32)


def project_image(img: np.ndarray, focal_length: float, mode: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """C++-accelerated inverse projection via cv2.remap."""
    h, w = img.shape[:2]
    if mode == 0:
        return img.copy(), np.full((h, w), 255, dtype=np.uint8)

    cx, cy = w / 2.0, h / 2.0
    y_i, x_i = np.indices((h, w), dtype=np.float32)
    u = (x_i - cx) / focal_length
    v = (y_i - cy) / focal_length

    if mode == 1:  # Cylindrical Outward (Convex / Barrel)
        map_x = focal_length * np.tan(u) + cx
        map_y = focal_length * (v / np.cos(u)) + cy
    elif mode == 2:  # Cylindrical Inward (Concave Ribbon)
        map_x = focal_length * np.tan(u) + cx
        map_y = focal_length * (v * np.cos(u)) + cy
    elif mode == 3:  # Spherical
        map_x = focal_length * np.tan(u) + cx
        map_y = focal_length * (np.tan(v) / np.cos(u)) + cy
    elif mode == 4:  # Stereographic (Circular)
        r = np.sqrt(u**2 + v**2)
        theta = 2.0 * np.arctan(r / 2.0)
        scale = np.where(r > 1e-6, np.tan(theta) / r, 1.0)
        map_x = focal_length * u * scale + cx
        map_y = focal_length * v * scale + cy
    else:
        return img.copy(), np.full((h, w), 255, dtype=np.uint8)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
    mask = (valid.astype(np.uint8)) * 255
    return warped, mask


def cylindrical_warp(img: np.ndarray, focal_length: float) -> tuple[np.ndarray, np.ndarray]:
    return project_image(img, focal_length, mode=1)