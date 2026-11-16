# Classical Image Stitching & Interactive Panorama Studio

A modular computer vision panorama stitching engine implemented in Python using classical projective geometry, SIFT feature matching, normalized DLT + RANSAC homography estimation, multi-surface projections, exposure-compensated Laplacian pyramid blending, and an interactive Tkinter studio.

## Pipeline Architecture
1. **Feature Extraction & Ratio Test (`src/features.py`):** SIFT keypoint detection with FLANN/BFMatcher and Lowe's ratio test ($r = 0.75$).
2. **Robust Homography Estimation (`src/homography.py`):** Hartley coordinate normalization ($\mu = 0, \bar{d} = \sqrt{2}$) paired with vectorized Direct Linear Transform (DLT) and RANSAC outlier rejection.
3. **Multi-Surface Projection (`src/warping.py`):** C++-accelerated inverse mapping (`cv2.remap`) supporting **Planar**, **Cylindrical (Outward & Inward)**, **Spherical**, and **Stereographic** projections, plus analytical keypoint projection.
4. **Horizon Leveling & Clamped Edge Rectangling (`src/optimizer.py`, `src/rectangling.py`):** Global rotational drift correction, multi-objective parameter auto-fitting, and bounded 2D mesh straightening.
5. **Exposure-Compensated Multi-Band Blending (`src/blending.py`):** Per-channel gain compensation in overlap regions followed by 6-level Gaussian/Laplacian pyramid blending computed inside active ROIs.
6. **Interactive Studio (`interactive.py`):** Multi-threaded GUI with a selectable photo gallery filmstrip, real-time parameter diagnostics, zoom/pan inspection, and HD export.

## Quick Start
```bash
pip install -r requirements.txt
python interactive.py