import numpy as np

from src.homography import ransac_homography
from src.warping import project_image, project_points
from src.utils import compute_canvas_bounds


def level_homographies(shapes: list[tuple[int, int]], homographies: list[np.ndarray]) -> list[np.ndarray]:
    """
    Removes vertical camera pitch/roll drift by aligning the start and end image
    centers horizontally without cropping any image pixels.
    """
    centers = []
    for (h, w), H in zip(shapes, homographies):
        c = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float64)
        p = H @ c
        centers.append(p[:2] / p[2])

    centers = np.asarray(centers)
    if len(centers) < 2:
        return homographies

    dx = centers[-1, 0] - centers[0, 0]
    dy = centers[-1, 1] - centers[0, 1]
    if abs(dx) < 1e-5:
        return homographies

    angle = -np.arctan2(dy, dx)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)

    return [R @ H for H in homographies]


def find_optimal_parameters(images: list[np.ndarray], cached_matches: list) -> dict:
    """
    Finds the (Shape, Focal, RANSAC) settings that minimize pixel reprojection error
    and maximize SIFT inlier consensus while keeping 100% of all input images intact.
    """
    best_score = -1e9
    best_cfg = {"shape": 2, "focal_val": 95, "blend": True, "ransac": 4.0, "error_px": 0.0, "inlier_pct": 0.0}

    # Evaluate across Outward (1), Inward Concave (2), Spherical (3), and Circular (4)
    candidates = [
        (1, 75, 4.0), (1, 95, 4.0), (1, 120, 4.0),
        (2, 75, 4.0), (2, 95, 4.0), (2, 120, 4.0),
        (3, 85, 4.0), (3, 110, 4.0),
        (4, 75, 4.0), (4, 95, 4.0),
    ]

    mid = len(images) // 2
    img_w = images[0].shape[1]

    for s_mode, f_val, r_px in candidates:
        focal = (f_val / 100.0) * img_w
        try:
            warped_pairs = [project_image(im, focal, mode=s_mode) for im in images]
            w_imgs = [w[0] for w in warped_pairs]
            homographies = [np.eye(3, dtype=np.float64) for _ in w_imgs]

            total_err = 0.0
            total_inlier_ratio = 0.0
            pair_count = 0
            match_idx = 0

            for i in range(mid - 1, -1, -1):
                raw_src, raw_dst = cached_matches[match_idx]
                match_idx += 1
                src_pts = project_points(raw_src, images[i].shape, focal, s_mode)
                dst_pts = project_points(raw_dst, images[i + 1].shape, focal, s_mode)
                H, inliers = ransac_homography(src_pts, dst_pts, iterations=250, threshold=r_px)
                homographies[i] = homographies[i + 1] @ H

                src_h = np.hstack([src_pts[inliers], np.ones((int(np.sum(inliers)), 1))])
                proj_h = src_h @ H.T
                proj = proj_h[:, :2] / proj_h[:, 2:3]
                err = float(np.mean(np.linalg.norm(proj - dst_pts[inliers], axis=1)))
                total_err += err
                total_inlier_ratio += float(np.mean(inliers))
                pair_count += 1

            for i in range(mid + 1, len(w_imgs)):
                raw_src, raw_dst = cached_matches[match_idx]
                match_idx += 1
                src_pts = project_points(raw_src, images[i].shape, focal, s_mode)
                dst_pts = project_points(raw_dst, images[i - 1].shape, focal, s_mode)
                H, inliers = ransac_homography(src_pts, dst_pts, iterations=250, threshold=r_px)
                homographies[i] = homographies[i - 1] @ H

                src_h = np.hstack([src_pts[inliers], np.ones((int(np.sum(inliers)), 1))])
                proj_h = src_h @ H.T
                proj = proj_h[:, :2] / proj_h[:, 2:3]
                err = float(np.mean(np.linalg.norm(proj - dst_pts[inliers], axis=1)))
                total_err += err
                total_inlier_ratio += float(np.mean(inliers))
                pair_count += 1

            homographies = level_homographies([im.shape[:2] for im in w_imgs], homographies)
            _, canvas_size = compute_canvas_bounds([im.shape[:2] for im in w_imgs], homographies)
            cw, ch = canvas_size

            if cw <= 0 or ch <= 0 or cw > 6500 or ch > 3500:
                continue

            mean_err = total_err / max(pair_count, 1)
            mean_inliers = total_inlier_ratio / max(pair_count, 1)

            # Pure geometric precision score (no bias toward shrinking width)
            score = (mean_inliers * 100.0) - (mean_err * 22.0)

            if score > best_score:
                best_score = score
                best_cfg = {
                    "shape": s_mode,
                    "focal_val": f_val,
                    "blend": True,
                    "ransac": r_px,
                    "error_px": mean_err,
                    "inlier_pct": mean_inliers * 100.0,
                }
        except Exception:
            continue

    return best_cfg