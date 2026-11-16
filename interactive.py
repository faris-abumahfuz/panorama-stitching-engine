import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import cv2
import numpy as np

from main import load_images
from src.features import extract_and_match_features
from src.homography import ransac_homography
from src.warping import project_image, project_points
from src.blending import multiband_blend, match_exposure
from src.utils import compute_canvas_bounds, crop_black_borders
from src.optimizer import level_homographies, find_optimal_parameters
from src.rectangling import rectify_panorama_edges

SHAPE_NAMES = {
    0: "Planar (Flat)",
    1: "Cyl Outward",
    2: "Cyl Inward",
    3: "Spherical",
    4: "Circular",
}


def center_out_order(n: int) -> list[int]:
    mid = n // 2
    order = [mid]
    for offset in range(1, n):
        if mid - offset >= 0:
            order.append(mid - offset)
        if mid + offset < n:
            order.append(mid + offset)
    return order


def fast_alpha_blend(layers, canvas_size, straighten_edges: bool = False) -> np.ndarray:
    order = center_out_order(len(layers))
    acc_img = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    acc_w = np.zeros((canvas_size[1], canvas_size[0], 1), dtype=np.float32)
    running_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)

    for idx in order:
        w_img, w_mask, w_dist = layers[idx]
        if np.any(running_mask > 0):
            cur_preview = np.clip(acc_img / np.maximum(acc_w, 1e-6), 0, 255).astype(np.uint8)
            w_img = match_exposure(w_img, cur_preview, w_mask, running_mask)

        clean_w = np.where(w_mask > 0, np.maximum(w_dist, 1e-2), 0.0).astype(np.float32)[..., None]
        acc_img += w_img.astype(np.float32) * clean_w
        acc_w += clean_w
        running_mask = np.maximum(running_mask, w_mask)

    out = crop_black_borders(np.clip(acc_img / np.maximum(acc_w, 1e-6), 0, 255).astype(np.uint8))
    return rectify_panorama_edges(out) if straighten_edges else out


def fast_multiband_blend(layers, canvas_size, straighten_edges: bool = False) -> np.ndarray:
    order = center_out_order(len(layers))
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)

    for idx in order:
        w_img, w_mask, _ = layers[idx]
        if np.any(canvas_mask > 0):
            canvas = multiband_blend(canvas, w_img, canvas_mask, w_mask, levels=6)
        else:
            canvas = w_img.copy()
        canvas_mask = np.maximum(canvas_mask, w_mask)

    out = crop_black_borders(canvas)
    return rectify_panorama_edges(out) if straighten_edges else out


def estimate_homographies(
    images, pair_matches: list, focal_mult: float, shape_mode: int, ransac_thresh: float, level_horizon: bool = True
) -> tuple[list[np.ndarray], tuple[int, int]]:
    focal = focal_mult * images[0].shape[1]
    mid = len(images) // 2
    homographies = [np.eye(3, dtype=np.float64) for _ in images]

    kept_inliers = 0
    total_matches = 0
    match_idx = 0

    for i in range(mid - 1, -1, -1):
        raw_src, raw_dst = pair_matches[match_idx]
        match_idx += 1
        if len(raw_src) < 4:
            raise ValueError(f"Not enough SIFT matches between selected frame #{i+1} and #{i+2}.")
        src_pts = project_points(raw_src, images[i].shape, focal, shape_mode)
        dst_pts = project_points(raw_dst, images[i + 1].shape, focal, shape_mode)
        H, inliers = ransac_homography(src_pts, dst_pts, iterations=350, threshold=ransac_thresh)
        homographies[i] = homographies[i + 1] @ H
        kept_inliers += int(np.sum(inliers))
        total_matches += len(inliers)

    for i in range(mid + 1, len(images)):
        raw_src, raw_dst = pair_matches[match_idx]
        match_idx += 1
        if len(raw_src) < 4:
            raise ValueError(f"Not enough SIFT matches between selected frame #{i+1} and #{i}.")
        src_pts = project_points(raw_src, images[i].shape, focal, shape_mode)
        dst_pts = project_points(raw_dst, images[i - 1].shape, focal, shape_mode)
        H, inliers = ransac_homography(src_pts, dst_pts, iterations=350, threshold=ransac_thresh)
        homographies[i] = homographies[i - 1] @ H
        kept_inliers += int(np.sum(inliers))
        total_matches += len(inliers)

    if level_horizon:
        homographies = level_homographies([im.shape[:2] for im in images], homographies)

    return homographies, (kept_inliers, total_matches)


def scale_homographies(homographies: list[np.ndarray], scale: float) -> list[np.ndarray]:
    S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    S_inv = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return [S @ H @ S_inv for H in homographies]


def render_from_homographies(images, homographies: list[np.ndarray], focal_mult: float, shape_mode: int, max_canvas_dim: int = 3200):
    focal = focal_mult * images[0].shape[1]
    warped_pairs = [project_image(img, focal, mode=shape_mode) for img in images]
    imgs = [w[0] for w in warped_pairs]
    masks = [w[1] for w in warped_pairs]

    dists = []
    for m in masks:
        d = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)
        max_d = float(d.max())
        dists.append((d / max_d) if max_d > 0 else d)

    T_offset, canvas_size = compute_canvas_bounds([im.shape[:2] for im in imgs], homographies)
    cw, ch = canvas_size
    if cw <= 0 or ch <= 0 or cw > 30000 or ch > 18000:
        raise ValueError(f"Extreme stretch ({cw}x{ch}) — try Cylindrical shape or higher Focal.")

    downscale = min(1.0, max_canvas_dim / max(cw, ch))
    if downscale < 1.0:
        S = np.array([[downscale, 0.0, 0.0], [0.0, downscale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_offset = S @ T_offset
        canvas_size = (max(1, int(cw * downscale)), max(1, int(ch * downscale)))

    layers = []
    for img, mask, dist, H in zip(imgs, masks, dists, homographies):
        M = T_offset @ H
        w_img = cv2.warpPerspective(img, M, canvas_size)
        w_mask = cv2.warpPerspective(mask, M, canvas_size)
        w_dist = np.clip(cv2.warpPerspective(dist, M, canvas_size), 0.0, 1.0)
        w_dist[w_mask == 0] = 0.0
        layers.append((w_img, w_mask, w_dist))

    return layers, canvas_size


def crop_to_photoimage(bgr_crop: np.ndarray, out_w: int, out_h: int) -> tk.PhotoImage:
    interp = cv2.INTER_NEAREST if out_w > bgr_crop.shape[1] * 2 else cv2.INTER_AREA
    resized = cv2.resize(bgr_crop, (max(1, out_w), max(1, out_h)), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    ppm_header = f"P6 {resized.shape[1]} {resized.shape[0]} 255\n".encode("ascii")
    return tk.PhotoImage(data=ppm_header + rgb.tobytes(), format="PPM")


class PanoramaStudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Panorama Stitching Studio")
        self.root.geometry("1500x880")
        self.root.configure(bg="#18191f")

        self.full_images = load_images(Path("data/raw"))
        self.preview_images = [
            cv2.resize(im, (560, int(im.shape[0] * 560 / im.shape[1])), interpolation=cv2.INTER_AREA)
            if im.shape[1] > 560 else im
            for im in self.full_images
        ]

        self.selected_indices = list(range(len(self.preview_images)))
        self.match_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.thumb_widgets = []
        self.thumb_photo_refs = []

        self.is_busy = False
        self.applied_params = None
        self.last_homographies = None
        self.last_inlier_stats = None
        self.current_bgr = None
        self.tk_img_ref = None

        self.zoom_factor = 1.0
        self.pan_cx = 0.5
        self.pan_cy = 0.5
        self._drag_start = None

        self._build_layout()
        self._refresh_gallery_ui()
        self.update_ui_state()
        self.trigger_auto_fit()

    def _get_pair_matches_for_selection(self, indices: list[int]) -> list:
        mid = len(indices) // 2
        ordered_pairs = []
        for i in range(mid - 1, -1, -1):
            ordered_pairs.append((indices[i], indices[i + 1]))
        for i in range(mid + 1, len(indices)):
            ordered_pairs.append((indices[i], indices[i - 1]))

        results = []
        for idx_src, idx_dst in ordered_pairs:
            key = (idx_src, idx_dst)
            if key not in self.match_cache:
                self.match_cache[key] = extract_and_match_features(
                    self.preview_images[idx_src], self.preview_images[idx_dst]
                )
            results.append(self.match_cache[key])
        return results

    def _build_layout(self):
        sidebar = tk.Frame(self.root, width=445, bg="#202128", padx=16, pady=12)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="PANORAMA CONTROLS", font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#202128").pack(anchor="w", pady=(0, 6))

        preset_row = tk.Frame(sidebar, bg="#202128")
        preset_row.pack(fill=tk.X, pady=(0, 8))

        self.btn_auto = tk.Button(
            preset_row, text="AUTO-FIT BEST", font=("Segoe UI", 9, "bold"),
            bg="#8e44ad", fg="white", activebackground="#732d91", activeforeground="white",
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2", command=self.trigger_auto_fit
        )
        self.btn_auto.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        self.btn_phone = tk.Button(
            preset_row, text="PHONE PANO PRESET", font=("Segoe UI", 9, "bold"),
            bg="#16a085", fg="white", activebackground="#1abc9c", activeforeground="white",
            relief=tk.FLAT, padx=10, pady=6, cursor="hand2", command=self.trigger_phone_pano
        )
        self.btn_phone.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.var_shape = tk.IntVar(value=1)
        self.var_focal = tk.IntVar(value=155)
        self.var_blend = tk.IntVar(value=1)
        self.var_ransac = tk.IntVar(value=5)
        self.var_level = tk.BooleanVar(value=True)
        self.var_straighten = tk.BooleanVar(value=False)

        self.card_shape = self._create_slider_card(sidebar, "1. PROJECTION SHAPE (0 - 4)", "#9b59b6", self.var_shape, 0, 4)
        self.card_focal = self._create_slider_card(sidebar, "2. FOCAL LENGTH (x100)", "#1abc9c", self.var_focal, 25, 250)
        self.card_blend = self._create_slider_card(sidebar, "3. BLEND MODE", "#3498db", self.var_blend, 0, 1)
        self.card_ransac = self._create_slider_card(sidebar, "4. RANSAC THRESHOLD (px)", "#e67e22", self.var_ransac, 1, 15)

        tk.Checkbutton(
            sidebar, text="Level Horizon Tilt (Aligns Start & End Height)",
            variable=self.var_level, font=("Segoe UI", 9, "bold"),
            fg="#ecf0f1", bg="#202128", selectcolor="#17181e",
            activebackground="#202128", activeforeground="#ffffff",
            command=self.update_ui_state
        ).pack(anchor="w", pady=(3, 1))

        tk.Checkbutton(
            sidebar, text="Soft-Flatten Outer Edges (Clamped 1.22x Anti-Distort)",
            variable=self.var_straighten, font=("Segoe UI", 9, "bold"),
            fg="#48c9b0", bg="#202128", selectcolor="#17181e",
            activebackground="#202128", activeforeground="#ffffff",
            command=self.update_ui_state
        ).pack(anchor="w", pady=(1, 4))

        btn_frame = tk.Frame(sidebar, bg="#202128")
        btn_frame.pack(fill=tk.X, pady=(4, 6))

        self.btn_apply = tk.Button(
            btn_frame, text="APPLY CHANGES (Space)", font=("Segoe UI", 11, "bold"),
            bg="#e67e22", fg="white", activebackground="#d35400", activeforeground="white",
            relief=tk.FLAT, pady=8, cursor="hand2", command=self.trigger_apply
        )
        self.btn_apply.pack(fill=tk.X, pady=(0, 6))

        sub_btns = tk.Frame(btn_frame, bg="#202128")
        sub_btns.pack(fill=tk.X)

        self.btn_save = tk.Button(
            sub_btns, text="SAVE HD", font=("Segoe UI", 10, "bold"),
            bg="#2980b9", fg="white", relief=tk.FLAT, pady=6, cursor="hand2", command=self.trigger_save
        )
        self.btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        tk.Button(
            sub_btns, text="QUIT", font=("Segoe UI", 10, "bold"),
            bg="#c0392b", fg="white", relief=tk.FLAT, pady=6, cursor="hand2", command=self.root.destroy
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.progress = ttk.Progressbar(sidebar, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(4, 4))

        self.lbl_status = tk.Label(sidebar, text="Ready", font=("Segoe UI", 9, "bold"), fg="#2ecc71", bg="#202128", anchor="w", wraplength=415)
        self.lbl_status.pack(fill=tk.X)

        # Right Panel
        right_panel = tk.Frame(self.root, bg="#121317")
        right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        gallery_container = tk.Frame(right_panel, bg="#171820", padx=12, pady=8)
        gallery_container.pack(side=tk.TOP, fill=tk.X)

        gal_header = tk.Frame(gallery_container, bg="#171820")
        gal_header.pack(fill=tk.X, pady=(0, 6))

        self.lbl_gallery = tk.Label(
            gal_header,
            text=f"PHOTO GALLERY ({len(self.preview_images)}/{len(self.preview_images)} Selected)",
            font=("Segoe UI", 9, "bold"), fg="#ecf0f1", bg="#171820"
        )
        self.lbl_gallery.pack(side=tk.LEFT)

        tk.Button(
            gal_header, text="First 2", font=("Segoe UI", 8, "bold"), bg="#34495e", fg="white",
            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self._set_selection_preset("first2")
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            gal_header, text="Middle 3", font=("Segoe UI", 8, "bold"), bg="#34495e", fg="white",
            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self._set_selection_preset("mid3")
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            gal_header, text=f"Select All ({len(self.preview_images)})", font=("Segoe UI", 8, "bold"), bg="#27ae60", fg="white",
            relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self._set_selection_preset("all")
        ).pack(side=tk.RIGHT, padx=2)

        strip_frame = tk.Frame(gallery_container, bg="#171820")
        strip_frame.pack(fill=tk.X)

        for idx, img in enumerate(self.preview_images):
            thumb_img = crop_to_photoimage(img, 96, 62)
            self.thumb_photo_refs.append(thumb_img)

            card = tk.Frame(strip_frame, bg="#2ecc71", padx=2, pady=2, cursor="hand2")
            card.pack(side=tk.LEFT, padx=4)

            lbl_pic = tk.Label(card, image=thumb_img, bg="#121317", cursor="hand2")
            lbl_pic.pack()

            lbl_cap = tk.Label(card, text=f"#{idx+1} ON", font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#1e8449", cursor="hand2")
            lbl_cap.pack(fill=tk.X)

            for w in (card, lbl_pic, lbl_cap):
                w.bind("<Button-1>", lambda e, i=idx: self._toggle_photo(i))

            self.thumb_widgets.append((card, lbl_cap))

        zoom_bar = tk.Frame(right_panel, bg="#1e2029", height=34, padx=12)
        zoom_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_zoom = tk.Label(zoom_bar, text="Zoom: Fit (1.0x)  |  Scroll wheel to zoom • Drag to pan • Double-click to reset", font=("Segoe UI", 9, "bold"), fg="#bdc3c7", bg="#1e2029")
        self.lbl_zoom.pack(side=tk.LEFT, pady=5)

        tk.Button(zoom_bar, text="Fit", font=("Segoe UI", 8, "bold"), bg="#34495e", fg="white", relief=tk.FLAT, padx=8, command=self.reset_zoom).pack(side=tk.RIGHT, padx=3, pady=4)
        tk.Button(zoom_bar, text="100% Detail", font=("Segoe UI", 8, "bold"), bg="#34495e", fg="white", relief=tk.FLAT, padx=8, command=lambda: self.set_zoom(2.5)).pack(side=tk.RIGHT, padx=3, pady=4)
        tk.Button(zoom_bar, text="+", font=("Segoe UI", 9, "bold"), bg="#2c3e50", fg="white", relief=tk.FLAT, padx=8, command=lambda: self.set_zoom(self.zoom_factor * 1.35)).pack(side=tk.RIGHT, padx=2, pady=4)
        tk.Button(zoom_bar, text="-", font=("Segoe UI", 9, "bold"), bg="#2c3e50", fg="white", relief=tk.FLAT, padx=8, command=lambda: self.set_zoom(self.zoom_factor / 1.35)).pack(side=tk.RIGHT, padx=2, pady=4)

        self.viewport = tk.Frame(right_panel, bg="#121317")
        self.viewport.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)
        self.viewport.bind("<Configure>", lambda e: self._refresh_viewport())

        self.lbl_image = tk.Label(self.viewport, bg="#121317", text="Computing optimal mathematical fit...", fg="#bdc3c7", font=("Segoe UI", 14), cursor="fleur")
        self.lbl_image.pack(expand=True)

        for widget in (self.viewport, self.lbl_image):
            widget.bind("<MouseWheel>", self._on_mouse_wheel)
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_move)
            widget.bind("<Double-Button-1>", lambda e: self.reset_zoom())

        self.root.bind("<space>", lambda e: self.trigger_apply())
        self.root.bind("<Return>", lambda e: self.trigger_apply())

    def _toggle_photo(self, idx: int):
        if self.is_busy:
            return
        if idx in self.selected_indices:
            if len(self.selected_indices) <= 2:
                self.lbl_status.config(text="At least 2 photos must remain selected!", fg="#e74c3c")
                return
            self.selected_indices.remove(idx)
        else:
            self.selected_indices.append(idx)
            self.selected_indices.sort()
        self._refresh_gallery_ui()
        self.update_ui_state()

    def _set_selection_preset(self, preset: str):
        if self.is_busy:
            return
        n = len(self.preview_images)
        if preset == "all":
            self.selected_indices = list(range(n))
        elif preset == "mid3":
            mid = n // 2
            self.selected_indices = [i for i in (mid - 1, mid, mid + 1) if 0 <= i < n]
        elif preset == "first2":
            self.selected_indices = list(range(min(2, n)))
        self._refresh_gallery_ui()
        self.update_ui_state()

    def _refresh_gallery_ui(self):
        n = len(self.preview_images)
        sel_count = len(self.selected_indices)
        self.lbl_gallery.config(
            text=f"PHOTO GALLERY ({sel_count}/{n} Selected: {[i+1 for i in self.selected_indices]} — Click thumbnails to toggle)"
        )
        for idx, (card, lbl_cap) in enumerate(self.thumb_widgets):
            if idx in self.selected_indices:
                card.config(bg="#2ecc71")
                lbl_cap.config(text=f"#{idx+1} ON", bg="#1e8449", fg="#ffffff")
            else:
                card.config(bg="#3b3e4a")
                lbl_cap.config(text=f"#{idx+1} OFF", bg="#262830", fg="#7f8c8d")

    def set_zoom(self, new_zoom: float):
        self.zoom_factor = float(np.clip(new_zoom, 1.0, 8.0))
        if self.zoom_factor == 1.0:
            self.pan_cx, self.pan_cy = 0.5, 0.5
        self._refresh_viewport()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_cx, self.pan_cy = 0.5, 0.5
        self._refresh_viewport()

    def _on_mouse_wheel(self, event):
        if self.current_bgr is None:
            return
        factor = 1.25 if event.delta > 0 else (1.0 / 1.25)
        self.set_zoom(self.zoom_factor * factor)

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y, self.pan_cx, self.pan_cy)

    def _on_drag_move(self, event):
        if self._drag_start is None or self.zoom_factor <= 1.0 or self.current_bgr is None:
            return
        sx, sy, start_cx, start_cy = self._drag_start
        vw = max(1, self.viewport.winfo_width())
        vh = max(1, self.viewport.winfo_height())
        dx = (event.x - sx) / (vw * self.zoom_factor)
        dy = (event.y - sy) / (vh * self.zoom_factor)
        self.pan_cx = float(np.clip(start_cx - dx, 0.0, 1.0))
        self.pan_cy = float(np.clip(start_cy - dy, 0.0, 1.0))
        self._refresh_viewport()

    def _create_slider_card(self, parent, title: str, accent: str, var: tk.IntVar, frm: int, to: int):
        outer = tk.Frame(parent, bg="#17181e", pady=1)
        outer.pack(fill=tk.X, pady=4)

        bar = tk.Frame(outer, width=5, bg=accent)
        bar.pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(outer, bg="#17181e", padx=11, pady=5)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_row = tk.Frame(body, bg="#17181e")
        top_row.pack(fill=tk.X)

        tk.Label(top_row, text=title, font=("Segoe UI", 10, "bold"), fg="#bdc3c7", bg="#17181e").pack(side=tk.LEFT)
        badge_lbl = tk.Label(top_row, text="", font=("Segoe UI", 10, "bold"), fg=accent, bg="#17181e")
        badge_lbl.pack(side=tk.RIGHT)

        scale = tk.Scale(
            body, variable=var, from_=frm, to=to, orient=tk.HORIZONTAL,
            showvalue=False, width=15, sliderlength=24,
            bg="#17181e", troughcolor="#0e0f13", activebackground=accent, highlightthickness=0,
            command=lambda _: self.update_ui_state()
        )
        scale.pack(fill=tk.X, pady=(2, 2))

        desc_lbl = tk.Label(
            body, text="", font=("Segoe UI", 9), fg="#ecf0f1", bg="#17181e",
            justify=tk.LEFT, anchor="w", wraplength=380
        )
        desc_lbl.pack(fill=tk.X)

        return (badge_lbl, desc_lbl, scale)

    def _current_params(self):
        return (
            self.var_shape.get(),
            self.var_focal.get(),
            self.var_blend.get() == 1,
            float(self.var_ransac.get()),
            self.var_level.get(),
            self.var_straighten.get(),
            tuple(self.selected_indices),
        )

    def update_ui_state(self):
        s_mode, f_val, u_mb, r_px, lvl, str_edge, sel_tup = self._current_params()
        f_mult = f_val / 100.0
        a_s, a_f, a_b, a_r, _, _, _ = self.applied_params if self.applied_params else (s_mode, f_val, u_mb, r_px, lvl, str_edge, sel_tup)

        shape_descs = {
            0: "Planar (Flat): direct 3x3 homography — keeps straight lines straight.",
            1: "Cylindrical Outward: standard wide strip (vertical walls stay straight).",
            2: "Cylindrical Inward: concave ribbon curve (pinches inward in center).",
            3: "Spherical: 2D globe curve across both horizon and verticals.",
            4: "Circular (Stereo): fisheye lens look curving into a circle.",
        }
        s_badge = f"[{s_mode}] {SHAPE_NAMES[s_mode]}"
        if self.applied_params and s_mode != a_s:
            s_badge = f"{SHAPE_NAMES[a_s]} → {SHAPE_NAMES[s_mode]}"
        self.card_shape[0].config(text=s_badge, fg="#f39c12" if (self.applied_params and s_mode != a_s) else "#bb8fce")
        self.card_shape[1].config(text=shape_descs[s_mode], fg="#ecf0f1")

        # Focal Length Card: make explicit when Planar mode ignores focal length
        if s_mode == 0:
            self.card_focal[0].config(text="N/A (Planar)", fg="#7f8c8d")
            self.card_focal[1].config(text="Focal length only affects curved projections (Shapes 1–4).", fg="#95a5a6")
        else:
            f_badge = f"{f_mult:.2f}"
            if self.applied_params and f_val != a_f:
                f_badge = f"{a_f/100:.2f} → {f_mult:.2f}"

            if f_mult < 0.45:
                f_text = "⚠ Too low: extreme curvature may fail to render."
                f_color = "#ff6b6b"
            elif f_mult < 1.15:
                f_text = "Tight radius: pronounced circular / barrel curvature."
                f_color = "#ecf0f1"
            elif f_mult <= 1.85:
                f_text = "Balanced wide-angle lens: natural left-to-right panorama."
                f_color = "#2ecc71"
            else:
                f_text = "ℹ High focal: nearly flat projection (close to Planar)."
                f_color = "#f9e79f"

            self.card_focal[0].config(text=f_badge, fg="#f39c12" if (self.applied_params and f_val != a_f) else "#48c9b0")
            self.card_focal[1].config(text=f_text, fg=f_color)

        b_name = "Multi-Band" if u_mb else "Alpha"
        b_badge = f"[{int(u_mb)}] {b_name}"
        if self.applied_params and u_mb != a_b:
            b_badge = f"{'Multi-Band' if a_b else 'Alpha'} → {b_name}"
        b_text = (
            "6-level pyramid + exposure gain match: erases vertical seam lines."
            if u_mb else
            "Exposure-matched alpha feather: very fast preview."
        )
        self.card_blend[0].config(text=b_badge, fg="#f39c12" if (self.applied_params and u_mb != a_b) else "#5dade2")
        self.card_blend[1].config(text=b_text)

        r_badge = f"{r_px:.0f} px"
        if self.applied_params and r_px != a_r:
            r_badge = f"{a_r:.0f}px → {r_px:.0f}px"

        inlier_note = ""
        if self.last_inlier_stats is not None:
            kept, total = self.last_inlier_stats
            inlier_note = f" ({kept}/{total} matches kept)"

        if r_px < 2.0:
            r_text = f"Strict (<2px): drops noisy matches{inlier_note}."
            r_color = "#f9e79f"
        elif r_px <= 6.0:
            r_text = f"Balanced outlier filter{inlier_note}."
            r_color = "#ecf0f1"
        else:
            r_text = f"Forgiving (>6px): keeps looser matches{inlier_note}."
            r_color = "#f9e79f"

        self.card_ransac[0].config(text=r_badge, fg="#f39c12" if (self.applied_params and r_px != a_r) else "#eb984e")
        self.card_ransac[1].config(text=r_text, fg=r_color)

        if not self.is_busy and self.applied_params is not None:
            dirty = self._current_params() != self.applied_params
            if dirty:
                self.btn_apply.config(bg="#e67e22", text=f"APPLY ({len(sel_tup)} PHOTOS)")
            else:
                self.btn_apply.config(bg="#27ae60", text="APPLIED ✓")

    def _set_busy(self, busy: bool, msg: str):
        self.is_busy = busy
        self.lbl_status.config(text=msg, fg="#f1c40f" if busy else "#2ecc71")
        for btn in (self.btn_apply, self.btn_save, self.btn_auto, self.btn_phone):
            btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        if busy:
            self.btn_apply.config(bg="#7f8c8d", text="RENDERING...")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.update_ui_state()

    def _refresh_viewport(self):
        if self.current_bgr is None:
            return
        vw = max(200, self.viewport.winfo_width() - 20)
        vh = max(200, self.viewport.winfo_height() - 20)
        ih, iw = self.current_bgr.shape[:2]

        base_scale = min(vw / max(iw, 1), vh / max(ih, 1))
        eff_scale = base_scale * self.zoom_factor

        vis_src_w = min(iw, max(10, int(vw / eff_scale)))
        vis_src_h = min(ih, max(10, int(vh / eff_scale)))

        cx_px = int(self.pan_cx * iw)
        cy_px = int(self.pan_cy * ih)

        x1 = int(np.clip(cx_px - vis_src_w // 2, 0, max(0, iw - vis_src_w)))
        y1 = int(np.clip(cy_px - vis_src_h // 2, 0, max(0, ih - vis_src_h)))
        x2 = min(iw, x1 + vis_src_w)
        y2 = min(ih, y1 + vis_src_h)

        crop = self.current_bgr[y1:y2, x1:x2]
        disp_w = min(vw, max(1, int((x2 - x1) * eff_scale)))
        disp_h = min(vh, max(1, int((y2 - y1) * eff_scale)))

        self.tk_img_ref = crop_to_photoimage(crop, disp_w, disp_h)
        self.lbl_image.config(image=self.tk_img_ref, text="")
        self.lbl_zoom.config(
            text=f"Zoom: {self.zoom_factor:.1f}x ({iw}x{ih}px)  |  Scroll wheel to zoom • Drag to pan • Double-click to reset"
        )

    def trigger_phone_pano(self):
        if self.is_busy:
            return
        self.var_shape.set(1)
        self.var_focal.set(155)
        self.var_blend.set(1)
        self.var_ransac.set(5)
        self.var_level.set(True)
        self.var_straighten.set(True)
        self.update_ui_state()
        self.trigger_apply()

    def trigger_auto_fit(self):
        if self.is_busy:
            return
        sel = list(self.selected_indices)
        self._set_busy(True, f"Optimizing geometry for {len(sel)} selected photos...")

        def worker():
            t0 = time.perf_counter()
            try:
                sub_imgs = [self.preview_images[i] for i in sel]
                sub_matches = self._get_pair_matches_for_selection(sel)
                best = find_optimal_parameters(sub_imgs, sub_matches)
                s_mode = best["shape"]
                f_val = best["focal_val"]
                u_mb = best["blend"]
                r_px = best["ransac"]
                lvl = self.var_level.get()
                str_edge = self.var_straighten.get()

                Hs, inlier_stats = estimate_homographies(sub_imgs, sub_matches, f_val / 100.0, s_mode, r_px, lvl)
                layers, c_size = render_from_homographies(sub_imgs, Hs, f_val / 100.0, s_mode, max_canvas_dim=2800)
                pano = fast_multiband_blend(layers, c_size, str_edge) if u_mb else fast_alpha_blend(layers, c_size, str_edge)
                ms = (time.perf_counter() - t0) * 1000.0

                def on_done():
                    self.var_shape.set(s_mode)
                    self.var_focal.set(f_val)
                    self.var_blend.set(int(u_mb))
                    self.var_ransac.set(int(r_px))
                    self.last_homographies = Hs
                    self.last_inlier_stats = inlier_stats
                    self.current_bgr = pano
                    self.applied_params = (s_mode, f_val, u_mb, r_px, lvl, str_edge, tuple(sel))
                    self.reset_zoom()
                    self._set_busy(
                        False,
                        f"Auto-Fit ({len(sel)} pics, {ms:.0f}ms): {SHAPE_NAMES[s_mode]} | Err: {best['error_px']:.2f}px"
                    )

                self.root.after(0, on_done)
            except Exception as e:
                def on_err():
                    self.lbl_image.config(image="", text=f"Auto-Fit error:\n{e}", fg="#e74c3c")
                    self._set_busy(False, "Optimization failed — check selected frames")

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def trigger_apply(self):
        if self.is_busy:
            return
        params = self._current_params()
        sel = list(params[6])
        self._set_busy(True, f"Stitching {len(sel)} selected photos {[i+1 for i in sel]}...")

        def worker():
            s_mode, f_val, u_mb, r_px, lvl, str_edge, _ = params
            t0 = time.perf_counter()
            try:
                sub_imgs = [self.preview_images[i] for i in sel]
                sub_matches = self._get_pair_matches_for_selection(sel)
                Hs, inlier_stats = estimate_homographies(sub_imgs, sub_matches, f_val / 100.0, s_mode, r_px, lvl)
                layers, c_size = render_from_homographies(sub_imgs, Hs, f_val / 100.0, s_mode, max_canvas_dim=2800)
                pano = fast_multiband_blend(layers, c_size, str_edge) if u_mb else fast_alpha_blend(layers, c_size, str_edge)
                ms = (time.perf_counter() - t0) * 1000.0

                def on_done():
                    self.last_homographies = Hs
                    self.last_inlier_stats = inlier_stats
                    self.current_bgr = pano
                    self.applied_params = params
                    self._refresh_viewport()
                    self._set_busy(False, f"Stitched {len(sel)} photos in {ms:.0f} ms ({pano.shape[1]}x{pano.shape[0]})")

                self.root.after(0, on_done)
            except Exception as e:
                def on_err():
                    self.lbl_image.config(image="", text=f"Cannot render:\n{e}", fg="#e74c3c")
                    self._set_busy(False, "Render failed — adjust selection or sliders")

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def trigger_save(self):
        if self.is_busy:
            return
        params = self._current_params()
        sel = list(params[6])
        self._set_busy(True, f"Exporting HD panorama ({len(sel)} photos)...")

        def worker():
            s_mode, f_val, u_mb, r_px, lvl, str_edge, _ = params
            t0 = time.perf_counter()
            try:
                sub_full = [self.full_images[i] for i in sel]
                hd_images = [
                    cv2.resize(im, (1400, int(im.shape[0] * 1400 / im.shape[1])), interpolation=cv2.INTER_AREA)
                    if im.shape[1] > 1400 else im
                    for im in sub_full
                ]

                if self.last_homographies is not None and self.applied_params == params:
                    scale = hd_images[0].shape[1] / float(self.preview_images[sel[0]].shape[1])
                    hd_Hs = scale_homographies(self.last_homographies, scale)
                else:
                    sub_matches = self._get_pair_matches_for_selection(sel)
                    prev_sub = [self.preview_images[i] for i in sel]
                    prev_Hs, _ = estimate_homographies(prev_sub, sub_matches, f_val / 100.0, s_mode, r_px, lvl)
                    scale = hd_images[0].shape[1] / float(prev_sub[0].shape[1])
                    hd_Hs = scale_homographies(prev_Hs, scale)

                hd_layers, hd_size = render_from_homographies(hd_images, hd_Hs, f_val / 100.0, s_mode, max_canvas_dim=5000)
                hd_pano = fast_multiband_blend(hd_layers, hd_size, str_edge) if u_mb else fast_alpha_blend(hd_layers, hd_size, str_edge)

                safe_name = SHAPE_NAMES[s_mode].lower().replace(" ", "_").replace("(", "").replace(")", "")
                tag = "".join(str(i + 1) for i in sel)
                rect_tag = "_rect" if str_edge else ""
                out_path = Path(f"data/output/hd_pics{tag}_{safe_name}_f{f_val}{rect_tag}.jpg")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), hd_pano)
                ms = (time.perf_counter() - t0) * 1000.0
                self.root.after(0, lambda: self._set_busy(False, f"Saved {out_path.name} ({hd_pano.shape[1]}x{hd_pano.shape[0]} in {ms:.0f}ms)"))
            except Exception as e:
                self.root.after(0, lambda: self._set_busy(False, f"Save error: {e}"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramaStudioApp(root)
    root.mainloop()