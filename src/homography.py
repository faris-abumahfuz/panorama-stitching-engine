import numpy as np


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: zero-mean and average distance sqrt(2) from origin."""
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
    """Normalized Direct Linear Transform (DLT) for 3x3 homography estimation."""
    norm_src, T_src = _normalize_points(src_pts)
    norm_dst, T_dst = _normalize_points(dst_pts)

    A = []
    for (x, y), (u, v) in zip(norm_src, norm_dst):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    _, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2]


def ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    iterations: int = 2000,
    threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly estimates homography H (src -> dst) using RANSAC."""
    num_pts = len(src_pts)
    if num_pts < 4:
        raise ValueError("Need at least 4 matched keypoints for homography.")

    best_inliers = np.zeros(num_pts, dtype=bool)
    max_inliers = 0
    rng = np.random.default_rng(seed=42)
    src_h = np.hstack([src_pts, np.ones((num_pts, 1))])

    for _ in range(iterations):
        idx = rng.choice(num_pts, size=4, replace=False)
        H_cand = compute_homography_dlt(src_pts[idx], dst_pts[idx])

        proj_h = (H_cand @ src_h.T).T
        proj = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-8)

        errors = np.linalg.norm(proj - dst_pts, axis=1)
        inliers = errors < threshold
        count = int(np.sum(inliers))

        if count > max_inliers:
            max_inliers = count
            best_inliers = inliers

    best_H = compute_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])
    return best_H, best_inliers