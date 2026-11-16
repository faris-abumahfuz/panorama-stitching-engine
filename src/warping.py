import numpy as np


def cylindrical_warp(img: np.ndarray, focal_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Projects a planar image onto a cylindrical surface using inverse mapping."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y_i, x_i = np.indices((h, w), dtype=np.float32)
    theta = (x_i - cx) / focal_length
    h_cyl = (y_i - cy) / focal_length

    x_src = focal_length * np.tan(theta) + cx
    y_src = focal_length * (h_cyl / np.cos(theta)) + cy

    valid = (x_src >= 0) & (x_src < w - 1) & (y_src >= 0) & (y_src < h - 1)
    warped = np.zeros_like(img)

    x0 = np.floor(x_src[valid]).astype(np.int32)
    y0 = np.floor(y_src[valid]).astype(np.int32)
    dx = (x_src[valid] - x0)[:, None]
    dy = (y_src[valid] - y0)[:, None]

    top = (1 - dx) * img[y0, x0] + dx * img[y0, x0 + 1]
    bot = (1 - dx) * img[y0 + 1, x0] + dx * img[y0 + 1, x0 + 1]
    warped[valid] = np.clip((1 - dy) * top + dy * bot, 0, 255).astype(np.uint8)

    mask = (valid.astype(np.uint8)) * 255
    return warped, mask