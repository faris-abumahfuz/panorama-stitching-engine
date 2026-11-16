import argparse
from pathlib import Path
import cv2
import numpy as np

from src.features import extract_and_match_features
from src.homography import ransac_homography
from src.warping import cylindrical_warp
from src.blending import alpha_blend, multiband_blend
from src.utils import compute_canvas_bounds, crop_black_borders


def load_images(input_dir: Path) -> list[np.ndarray]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if len(paths) < 2:
        raise ValueError(f"Found {len(paths)} images in {input_dir}. Add at least 2 overlapping photos.")
    return [cv2.imread(str(p)) for p in paths]


def stitch_sequence(images: list[np.ndarray], blend_mode: str = "multiband", use_cylindrical: bool = True) -> np.ndarray:
    if use_cylindrical:
        focal = 0.85 * images[0].shape[1]
        warped_pairs = [cylindrical_warp(img, focal) for img in images]
        imgs = [w[0] for w in warped_pairs]
        masks = [w[1] for w in warped_pairs]
    else:
        imgs = images
        masks = [np.full(img.shape[:2], 255, dtype=np.uint8) for img in images]

    mid = len(imgs) // 2
    homographies = [np.eye(3, dtype=np.float64) for _ in imgs]

    for i in range(mid - 1, -1, -1):
        src_pts, dst_pts = extract_and_match_features(imgs[i], imgs[i + 1])
        H, _ = ransac_homography(src_pts, dst_pts)
        homographies[i] = homographies[i + 1] @ H

    for i in range(mid + 1, len(imgs)):
        src_pts, dst_pts = extract_and_match_features(imgs[i], imgs[i - 1])
        H, _ = ransac_homography(src_pts, dst_pts)
        homographies[i] = homographies[i - 1] @ H

    T_offset, canvas_size = compute_canvas_bounds([im.shape[:2] for im in imgs], homographies)

    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)

    blend_fn = multiband_blend if blend_mode == "multiband" else alpha_blend

    for img, mask, H in zip(imgs, masks, homographies):
        M = T_offset @ H
        warped_img = cv2.warpPerspective(img, M, canvas_size)
        warped_mask = cv2.warpPerspective(mask, M, canvas_size)

        if np.any(canvas_mask > 0):
            canvas = blend_fn(canvas, warped_img, canvas_mask, warped_mask)
        else:
            canvas = warped_img
        canvas_mask = np.maximum(canvas_mask, warped_mask)

    return crop_black_borders(canvas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical Panorama Stitching Pipeline")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/output/panorama.jpg"))
    parser.add_argument("--blend", choices=["alpha", "multiband"], default="multiband")
    parser.add_argument("--no-cylindrical", action="store_true")
    args = parser.parse_args()

    imgs = load_images(args.input_dir)
    pano = stitch_sequence(imgs, blend_mode=args.blend, use_cylindrical=not args.no_cylindrical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), pano)
    print(f"Saved panorama ({pano.shape[1]}x{pano.shape[0]}) -> {args.output}")