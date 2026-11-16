import cv2
import numpy as np


def rectify_panorama_edges(img: np.ndarray, smooth_sigma: float = 22.0, max_stretch: float = 1.22) -> np.ndarray:
    """
    Gently levels wavy top/bottom panorama boundaries while clamping maximum local
    stretch (<= 1.22x) so buildings on the far left and right edges never distort.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 2).astype(np.uint8)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return img

    bx, by, bw, bh = cv2.boundingRect(coords)
    if bw < 20 or bh < 20:
        return img

    sub_img = img[by : by + bh, bx : bx + bw]
    sub_mask = mask[by : by + bh, bx : bx + bw]

    # 1. Detect top y_top(x) and bottom y_bot(x) contour across X
    y_top = np.zeros(bw, dtype=np.float32)
    y_bot = np.full(bw, bh - 1, dtype=np.float32)

    for x in range(bw):
        ys = np.where(sub_mask[:, x] > 0)[0]
        if len(ys) > 0:
            y_top[x] = float(ys[0])
            y_bot[x] = float(ys[-1])
        else:
            y_top[x] = y_top[x - 1] if x > 0 else 0.0
            y_bot[x] = y_bot[x - 1] if x > 0 else float(bh - 1)

    ksize = max(5, int(smooth_sigma * 4) | 1)
    y_top_smooth = cv2.GaussianBlur(y_top.reshape(1, -1), (ksize, 1), smooth_sigma).ravel()
    y_bot_smooth = cv2.GaussianBlur(y_bot.reshape(1, -1), (ksize, 1), smooth_sigma).ravel()

    # Clamp target top/bottom to the 12th/88th percentile height so tapered outer tips aren't over-stretched
    target_top = float(np.percentile(y_top_smooth, 12))
    target_bot = float(np.percentile(y_bot_smooth, 88))
    target_h = max(target_bot - target_top, 16.0)

    # Blend each column toward target_top/target_bot, capped by max_stretch to prevent edge distortion
    src_h = np.maximum(y_bot_smooth - y_top_smooth, 8.0)
    allowed_ratio = np.minimum(target_h / src_h, max_stretch)
    actual_dest_h = src_h * allowed_ratio

    # Center the gentler vertical mapping inside [0 .. target_h]
    grid_y, grid_x = np.indices((int(target_h), bw), dtype=np.float32)
    v_norm = grid_y / max(target_h - 1.0, 1.0)
    map_y = y_top_smooth[None, :] + v_norm * src_h[None, :]

    # If a column would exceed max_stretch, pull from a slightly narrower vertical window instead of distorting
    over_stretched = (target_h / src_h) > max_stretch
    if np.any(over_stretched):
        clamped_span = target_h / max_stretch
        center_y = 0.5 * (y_top_smooth + y_bot_smooth)
        safe_top = np.clip(center_y - 0.5 * clamped_span, 0.0, float(bh - 1))
        safe_span = np.minimum(clamped_span, float(bh - 1) - safe_top)
        map_y[:, over_stretched] = safe_top[None, over_stretched] + v_norm[:, over_stretched] * safe_span[None, over_stretched]

    rectified = cv2.remap(
        sub_img,
        grid_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified