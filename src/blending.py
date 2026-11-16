import cv2
import numpy as np


def match_exposure(src_img: np.ndarray, ref_img: np.ndarray, src_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Adjusts per-channel gain of src_img to match ref_img brightness in the overlap region."""
    overlap = (src_mask > 0) & (ref_mask > 0)
    if np.count_nonzero(overlap) < 50:
        return src_img

    corrected = src_img.astype(np.float32)
    ref_f = ref_img.astype(np.float32)

    for c in range(3):
        src_mean = float(np.mean(corrected[:, :, c][overlap]))
        ref_mean = float(np.mean(ref_f[:, :, c][overlap]))
        if src_mean > 5.0:
            gain = np.clip(ref_mean / src_mean, 0.72, 1.38)
            corrected[:, :, c] *= gain

    return np.clip(corrected, 0, 255).astype(np.uint8)


def alpha_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Fast distance-transform feather blending across overlapping regions."""
    img2_adj = match_exposure(img2, img1, mask2, mask1)
    d1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)

    weight1 = d1 / (d1 + d2 + 1e-8)
    weight2 = 1.0 - weight1

    blended = img1.astype(np.float32) * weight1[..., None] + img2_adj.astype(np.float32) * weight2[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def multiband_blend(
    img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, levels: int = 6
) -> np.ndarray:
    """
    Exposure-compensated Laplacian pyramid blending computed strictly inside the
    active union ROI (5x-10x faster on wide multi-image canvases) with soft seam masks.
    """
    img2_adj = match_exposure(img2, img1, mask2, mask1)

    union = ((mask1 > 0) | (mask2 > 0)).astype(np.uint8)
    coords = cv2.findNonZero(union)
    if coords is None:
        return img1

    x, y, w, h = cv2.boundingRect(coords)
    # Pad ROI to a multiple of 2^levels so pyrDown/pyrUp never lose alignment
    step = 1 << levels
    pad_w = (step - (w % step)) % step
    pad_h = (step - (h % step)) % step

    x2 = min(img1.shape[1], x + w + pad_w)
    y2 = min(img1.shape[0], y + h + pad_h)
    x1 = max(0, x2 - (w + pad_w))
    y1 = max(0, y2 - (h + pad_h))

    sub1 = img1[y1:y2, x1:x2].astype(np.float32)
    sub2 = img2_adj[y1:y2, x1:x2].astype(np.float32)
    m1 = mask1[y1:y2, x1:x2]
    m2 = mask2[y1:y2, x1:x2]

    d1 = cv2.distanceTransform((m1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform((m2 > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # Smooth sigmoid-like transition mask instead of hard binary step (eliminates vertical seam lines)
    diff = (d1 - d2) / (np.maximum(d1 + d2, 1.0) * 0.35)
    seam_mask = np.clip(0.5 + 0.5 * np.tanh(diff), 0.0, 1.0).astype(np.float32)
    seam_mask[m2 == 0] = 1.0
    seam_mask[m1 == 0] = 0.0

    max_levels = min(levels, int(np.log2(max(min(sub1.shape[:2]), 8))) - 1)
    if max_levels < 1:
        out_sub = sub1 * seam_mask[..., None] + sub2 * (1.0 - seam_mask[..., None])
    else:
        gp_mask = [seam_mask]
        gp1 = [sub1]
        gp2 = [sub2]

        for _ in range(max_levels):
            gp_mask.append(cv2.pyrDown(gp_mask[-1]))
            gp1.append(cv2.pyrDown(gp1[-1]))
            gp2.append(cv2.pyrDown(gp2[-1]))

        lp1 = [gp1[-1]]
        lp2 = [gp2[-1]]
        for i in range(max_levels, 0, -1):
            size = (gp1[i - 1].shape[1], gp1[i - 1].shape[0])
            lp1.append(gp1[i - 1] - cv2.pyrUp(gp1[i], dstsize=size))
            lp2.append(gp2[i - 1] - cv2.pyrUp(gp2[i], dstsize=size))

        gp_mask.reverse()
        blended_pyr = [l1 * m[..., None] + l2 * (1.0 - m[..., None]) for l1, l2, m in zip(lp1, lp2, gp_mask)]

        out_sub = blended_pyr[0]
        for i in range(1, len(blended_pyr)):
            size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
            out_sub = cv2.pyrUp(out_sub, dstsize=size) + blended_pyr[i]

    result = img1.copy()
    result[y1:y2, x1:x2] = np.clip(out_sub, 0, 255).astype(np.uint8)
    return result