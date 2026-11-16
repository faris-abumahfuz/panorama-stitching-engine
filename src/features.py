import cv2
import numpy as np


def extract_and_match_features(
    img1: np.ndarray, img2: np.ndarray, ratio_thresh: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """Detects SIFT keypoints and filters matches using Lowe's ratio test."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_src, good_dst = [], []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_src.append(kp1[m.queryIdx].pt)
            good_dst.append(kp2[m.trainIdx].pt)

    return np.float32(good_src), np.float32(good_dst)