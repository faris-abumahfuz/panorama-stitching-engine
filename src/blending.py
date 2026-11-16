import cv2
import numpy as np


def alpha_blend(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Distance-transform feather blending across overlapping regions."""
    d1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)

    weight1 = d1 / (d1 + d2 + 1e-8)
    weight2 = 1.0 - weight1

    blended = img1.astype(np.float32) * weight1[..., None] + img2.astype(np.float32) * weight2[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def multiband_blend(
    img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, levels: int = 5
) -> np.ndarray:
    """Laplacian pyramid multi-band blending for seamless exposure transitions."""
    d1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    seam_mask = (d1 >= d2).astype(np.float32)

    gp_mask = [seam_mask]
    gp1 = [img1.astype(np.float32)]
    gp2 = [img2.astype(np.float32)]

    for _ in range(levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))

    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(levels, 0, -1):
        size = (gp1[i - 1].shape[1], gp1[i - 1].shape[0])
        lp1.append(gp1[i - 1] - cv2.pyrUp(gp1[i], dstsize=size))
        lp2.append(gp2[i - 1] - cv2.pyrUp(gp2[i], dstsize=size))

    gp_mask.reverse()
    blended_pyr = []
    for l1, l2, m in zip(lp1, lp2, gp_mask):
        m3 = m[..., None]
        blended_pyr.append(l1 * m3 + l2 * (1.0 - m3))

    out = blended_pyr[0]
    for i in range(1, len(blended_pyr)):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + blended_pyr[i]

    return np.clip(out, 0, 255).astype(np.uint8)