import random
import numpy as np


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / (mean_dist + 1e-8)

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    norm_pts = (T @ pts_h.T).T[:, :2]
    return norm_pts, T


def compute_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Vectorized Normalized Direct Linear Transform (DLT)."""
    norm_src, T_src = _normalize_points(src_pts)
    norm_dst, T_dst = _normalize_points(dst_pts)

    n = len(norm_src)
    x, y = norm_src[:, 0], norm_src[:, 1]
    u, v = norm_dst[:, 0], norm_dst[:, 1]
    zeros = np.zeros(n, dtype=np.float64)
    ones = np.ones(n, dtype=np.float64)

    A = np.empty((2 * n, 9), dtype=np.float64)
    A[0::2] = np.column_stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u])
    A[1::2] = np.column_stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v])

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / (H[2, 2] + 1e-12)


def ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    iterations: int = 350,
    threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast RANSAC with early stopping and vectorized DLT."""
    num_pts = len(src_pts)
    if num_pts < 4:
        raise ValueError("Need at least 4 matched keypoints for homography.")

    best_inliers = np.zeros(num_pts, dtype=bool)
    max_inliers = 0
    rng = random.Random(42)
    indices = list(range(num_pts))
    src_h = np.hstack([src_pts, np.ones((num_pts, 1))])
    thresh_sq = threshold * threshold
    early_stop = int(0.85 * num_pts)

    for _ in range(iterations):
        idx = rng.sample(indices, 4)
        H_cand = compute_homography_dlt(src_pts[idx], dst_pts[idx])

        proj_h = src_h @ H_cand.T
        denom = proj_h[:, 2:3]
        valid = np.abs(denom[:, 0]) > 1e-8
        proj = proj_h[:, :2] / np.where(valid[:, None], denom, 1.0)

        diff = proj - dst_pts
        inliers = valid & ((diff[:, 0] ** 2 + diff[:, 1] ** 2) < thresh_sq)
        count = int(np.sum(inliers))

        if count > max_inliers:
            max_inliers = count
            best_inliers = inliers
            if max_inliers >= early_stop:
                break

    best_H = compute_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
    return best_H, best_inliers