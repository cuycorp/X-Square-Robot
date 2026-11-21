"""Real-time Rubik's cube detector and optional Claude multi-face reader.

Pure numpy + Pillow core (works without OpenCV), using cv2 when installed for speed.
Supports both oblique 3D multi-face views (top + side faces on cluttered tabletops)
and single-face frontal / stickerless / white-majority views.
"""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.request
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from claude_policy import DEFAULT_CLAUDE_MODEL, extract_json_object, load_anthropic_api_key
from cube_state import FACES, empty_grid, normalize_grid, rotate_grid_cw

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

CUBE_SIZE_M = 0.057
DEFAULT_HFOV_DEG = 60.0


def _decode_image(image_input: bytes | str | Image.Image) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        s = image_input.strip()
        if "," in s and s.startswith("data:image/"):
            s = s.split(",", 1)[1]
        raw = base64.b64decode(s)
    elif isinstance(image_input, (bytes, bytearray)):
        raw = bytes(image_input)
    else:
        raise ValueError("Unsupported image input type")
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("RGB")


def _rgb_to_hsv_np(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB (H, W, 3) to H in [0, 360), S in [0, 1], V in [0, 1]."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 1e-6
    mask_r = nz & (cmax == r)
    mask_g = nz & (cmax == g)
    mask_b = nz & (cmax == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)
    h = np.mod(h, 360.0)

    s = np.zeros_like(cmax)
    pos = cmax > 1e-6
    s[pos] = delta[pos] / cmax[pos]
    return h, s, cmax


def _connected_blobs(
    mask: np.ndarray,
    use_cv2: bool = True,
    *,
    connectivity: int = 8,
    min_area: int = 8,
) -> list[dict[str, Any]]:
    h_img, w_img = mask.shape
    blobs: list[dict[str, Any]] = []
    if use_cv2 and HAS_CV2:
        conn = 8 if connectivity == 8 else 4
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=conn
        )
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            blobs.append({
                "label": lbl,
                "area": area,
                "bbox": (x, y, x + bw, y + bh),
                "bw": bw,
                "bh": bh,
                "cx": cx,
                "cy": cy,
                "labels_ref": labels,
            })
        return blobs

    visited = np.zeros((h_img, w_img), dtype=np.bool_)
    labels = np.zeros((h_img, w_img), dtype=np.int32)
    ys, xs = np.nonzero(mask)
    nbr_offsets = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 4
        else ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    )
    lbl = 0
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        lbl += 1
        q = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pts_y: list[int] = []
        pts_x: list[int] = []
        while q:
            cy, cx = q.popleft()
            labels[cy, cx] = lbl
            pts_y.append(cy)
            pts_x.append(cx)
            for dy, dx in nbr_offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h_img and 0 <= nx < w_img and not visited[ny, nx] and mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        area = len(pts_x)
        if area < min_area:
            continue
        x0, x1 = min(pts_x), max(pts_x) + 1
        y0, y1 = min(pts_y), max(pts_y) + 1
        blobs.append({
            "label": lbl,
            "area": area,
            "bbox": (x0, y0, x1, y1),
            "bw": x1 - x0,
            "bh": y1 - y0,
            "cx": float(sum(pts_x)) / area,
            "cy": float(sum(pts_y)) / area,
            "labels_ref": labels,
        })
    return blobs


def _box_any(mask: np.ndarray, ksize: int, use_cv2: bool = True) -> np.ndarray:
    if use_cv2 and HAS_CV2:
        k = np.ones((ksize, ksize), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    rad = ksize // 2
    padded = np.pad(mask.astype(np.int32), ((rad, rad), (rad, rad)), mode="constant", constant_values=0)
    cs = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(axis=0).cumsum(axis=1)
    h, w = mask.shape
    sums = cs[ksize:ksize + h, ksize:ksize + w] - cs[:h, ksize:ksize + w] - cs[ksize:ksize + h, :w] + cs[:h, :w]
    return sums > 0


def _sliding_max_2d(arr: np.ndarray, ksize: int) -> np.ndarray:
    rad = ksize // 2
    p = np.pad(arr, ((0, 0), (rad, rad)), mode="edge")
    out_h = p[:, :arr.shape[1]].copy()
    for d in range(1, ksize):
        out_h = np.maximum(out_h, p[:, d:d + arr.shape[1]])
    p2 = np.pad(out_h, ((rad, rad), (0, 0)), mode="edge")
    out = p2[:arr.shape[0], :].copy()
    for d in range(1, ksize):
        out = np.maximum(out, p2[d:d + arr.shape[0], :])
    return out


def _blackhat_2d(arr: np.ndarray, ksize: int, use_cv2: bool = True) -> np.ndarray:
    if use_cv2 and HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, k)
    closed = -_sliding_max_2d(-_sliding_max_2d(arr, ksize), ksize)
    return np.maximum(0.0, closed - arr)


def _color_grad_mag(patch: np.ndarray, use_cv2: bool = True) -> np.ndarray:
    pf = patch.astype(np.float32)
    if use_cv2 and HAS_CV2:
        gx = cv2.Sobel(pf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(pf, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(np.max(gx * gx + gy * gy, axis=2))
    p = np.pad(pf, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    gx = (p[0:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]) - (p[0:-2, 0:-2] + 2.0 * p[1:-1, 0:-2] + p[2:, 0:-2])
    gy = (p[2:, 0:-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]) - (p[0:-2, 0:-2] + 2.0 * p[0:-2, 1:-1] + p[0:-2, 2:])
    return np.sqrt(np.max(gx * gx + gy * gy, axis=2))


def _order_quad_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 2D points clockwise starting from top-left."""
    pts = np.asarray(pts, dtype=np.float64)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(diff))]
    bl = pts[int(np.argmax(diff))]
    return np.array([tl, tr, br, bl], dtype=np.float64)


def _convex_hull_2d(pts_xy: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain 2D convex hull in pure NumPy/Python."""
    pts = np.unique(np.round(pts_xy, 3), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order].tolist()

    def cross(o: list[float], a: list[float], b: list[float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[list[float]] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[list[float]] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _oriented_box_corners(pts_xy: np.ndarray, use_cv2: bool = True) -> np.ndarray:
    """Compute minimum-area oriented bounding quad [TL, TR, BR, BL] around (N, 2) pixel points."""
    if use_cv2 and HAS_CV2:
        rect = cv2.minAreaRect(pts_xy.astype(np.float32))
        box = cv2.boxPoints(rect)
        return _order_quad_tl_tr_br_bl(box)

    hull = _convex_hull_2d(pts_xy)
    if len(hull) < 3:
        x0, y0 = pts_xy.min(axis=0)
        x1, y1 = pts_xy.max(axis=0)
        return _order_quad_tl_tr_br_bl(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))

    edges = np.roll(hull, -1, axis=0) - hull
    norms = np.linalg.norm(edges, axis=1)
    valid = norms > 1e-6
    edges = edges[valid]
    norms = norms[valid]
    if len(edges) == 0:
        x0, y0 = pts_xy.min(axis=0)
        x1, y1 = pts_xy.max(axis=0)
        return _order_quad_tl_tr_br_bl(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))

    best_area = float("inf")
    best_corners = None
    for edge, norm in zip(edges, norms):
        ux, uy = edge[0] / norm, edge[1] / norm
        vx, vy = -uy, ux
        proj_u = hull[:, 0] * ux + hull[:, 1] * uy
        proj_v = hull[:, 0] * vx + hull[:, 1] * vy
        u0, u1 = float(proj_u.min()), float(proj_u.max())
        v0, v1 = float(proj_v.min()), float(proj_v.max())
        area = (u1 - u0) * (v1 - v0)
        if area < best_area:
            best_area = area
            best_corners = np.array([
                [u0 * ux + v0 * vx, u0 * uy + v0 * vy],
                [u1 * ux + v0 * vx, u1 * uy + v0 * vy],
                [u1 * ux + v1 * vx, u1 * uy + v1 * vy],
                [u0 * ux + v1 * vx, u0 * uy + v1 * vy],
            ], dtype=np.float64)
    return _order_quad_tl_tr_br_bl(best_corners)


def _split_if_elongated(quad: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """If an oriented quad spans 2 side-by-side faces (aspect > 1.45), crop to the densest square sub-face."""
    tl, tr, br, bl = quad
    top_vec = tr - tl
    left_vec = bl - tl
    w = float(np.linalg.norm(top_vec))
    h = float(np.linalg.norm(left_vec))
    if w < 1e-3 or h < 1e-3:
        return quad
    ratio = max(w, h) / min(w, h)
    if ratio <= 1.45:
        return quad

    if w > h:
        e_long = top_vec / w
        e_short = left_vec / h
        origin = tl
        long_len, short_len = w, h
    else:
        e_long = left_vec / h
        e_short = top_vec / w
        origin = tl
        long_len, short_len = h, w

    rel = pts_xy - origin
    u = rel @ e_long
    v = rel @ e_short
    best_u0 = 0.0
    best_count = -1
    steps = max(5, int((long_len - short_len) / 3.0) + 1)
    for idx in range(steps):
        u0 = (long_len - short_len) * (idx / max(1, steps - 1))
        cnt = int(np.sum((u >= u0) & (u <= u0 + short_len) & (v >= -2) & (v <= short_len + 2)))
        if cnt > best_count:
            best_count = cnt
            best_u0 = u0

    c0 = origin + e_long * best_u0
    c1 = c0 + e_long * short_len
    c2 = c1 + e_short * short_len
    c3 = c0 + e_short * short_len
    return _order_quad_tl_tr_br_bl(np.array([c0, c1, c2, c3]))


def classify_rgb_sticker(rgb_triplet: list[int] | np.ndarray) -> tuple[str, float]:
    """Classify one median [R, G, B] patch into W/Y/R/O/B/G with confidence."""
    r, g, b = [float(x) for x in rgb_triplet]
    arr = np.array([[[r, g, b]]], dtype=np.float32)
    h_arr, s_arr, v_arr = _rgb_to_hsv_np(arr)
    h = float(h_arr[0, 0])
    s = float(s_arr[0, 0])
    v = float(v_arr[0, 0])

    if v < 0.16:
        return "?", 0.15

    if s <= 0.25 and v >= 0.45:
        conf = min(1.0, (0.28 - s) / 0.28 * 0.6 + v * 0.4)
        return "W", round(max(0.45, conf), 3)

    if 175.0 <= h < 268.0:
        if v < 0.20 or s < 0.24:
            return "?", 0.15
        return "B", round(min(1.0, 0.55 + 0.45 * s), 3)
    if 76.0 <= h < 175.0:
        if v < 0.20 or s < 0.28:
            return "?", 0.15
        return "G", round(min(1.0, 0.55 + 0.45 * s), 3)

    # Warm hue sector (R/O/Y): reject dark brown rubber/shadow glare
    if v < 0.36 or s < 0.25:
        return "?", 0.15

    if 38.0 <= h < 76.0:
        return "Y", round(min(1.0, 0.55 + 0.45 * s), 3)
    if h >= 315.0 or h < 17.0:
        return "R", round(min(1.0, 0.55 + 0.45 * s), 3)
    return "O", round(min(1.0, 0.50 + 0.40 * s), 3)


def _bilinear_point(quad: np.ndarray, u: float, v: float) -> tuple[float, float]:
    tl, tr, br, bl = quad
    top = tl * (1.0 - u) + tr * u
    bot = bl * (1.0 - u) + br * u
    pt = top * (1.0 - v) + bot * v
    return float(pt[0]), float(pt[1])


def _estimate_pose(quad: np.ndarray, w_img: int, h_img: int) -> dict[str, Any]:
    fx = (0.5 * w_img) / math.tan(math.radians(DEFAULT_HFOV_DEG * 0.5))
    top_w = float(np.linalg.norm(quad[1] - quad[0]))
    bot_w = float(np.linalg.norm(quad[2] - quad[3]))
    left_h = float(np.linalg.norm(quad[3] - quad[0]))
    right_h = float(np.linalg.norm(quad[2] - quad[1]))
    apparent_size = max(4.0, 0.5 * (top_w + bot_w + left_h + right_h) * 0.5)
    z_m = (fx * CUBE_SIZE_M) / apparent_size
    center = quad.mean(axis=0)
    x_m = ((float(center[0]) - w_img * 0.5) * z_m) / fx
    y_m = ((float(center[1]) - h_img * 0.5) * z_m) / fx
    roll_rad = math.atan2(float(quad[1][1] - quad[0][1]), float(quad[1][0] - quad[0][0]))
    aspect = min(1.0, (0.5 * (left_h + right_h)) / max(1e-3, 0.5 * (top_w + bot_w)))
    tilt_rad = math.acos(max(-1.0, min(1.0, aspect)))
    return {
        "frame": "camera_optical_uncalibrated",
        "position_m": [round(x_m, 3), round(y_m, 3), round(z_m, 3)],
        "roll_deg": round(math.degrees(roll_rad), 1),
        "tilt_deg": round(math.degrees(tilt_rad), 1),
    }


def _find_seed_candidates(rgb_small: np.ndarray, use_cv2: bool = True, top_k: int = 3) -> list[tuple[int, int, int, int]]:
    """Locate compact regions where dark plastic grid seams co-occur with >=2 non-wood vivid colors."""
    h_img, w_img = rgb_small.shape[:2]
    h, s, v = _rgb_to_hsv_np(rgb_small)
    ksize = max(9, int(round(max(w_img, h_img) / 38.0)) | 1)

    dark = v < 0.25
    blue = (h >= 180.0) & (h <= 260.0) & (s >= 0.38) & (v >= 0.18)
    green = (h >= 78.0) & (h <= 170.0) & (s >= 0.28) & (v >= 0.15)
    yellow = (h >= 49.0) & (h <= 78.0) & (s >= 0.35) & (v >= 0.38)
    red = ((h < 20.0) | (h > 330.0)) & (s >= 0.42) & (v >= 0.36)

    votes = (
        _box_any(blue, ksize, use_cv2).astype(np.int32)
        + _box_any(green, ksize, use_cv2).astype(np.int32)
        + _box_any(yellow, ksize, use_cv2).astype(np.int32)
        + _box_any(red, ksize, use_cv2).astype(np.int32)
    )
    seed = _box_any(dark, ksize, use_cv2) & (votes >= 2)
    blobs = _connected_blobs(seed, use_cv2=use_cv2, connectivity=8, min_area=max(40, int(0.0003 * w_img * h_img)))

    vivid = blue | green | yellow | red
    cands: list[tuple[float, tuple[int, int, int, int]]] = []
    for b in blobs:
        x0, y0, x1, y1 = b["bbox"]
        bw, bh = b["bw"], b["bh"]
        if bw > 0.42 * w_img or bh > 0.42 * h_img:
            continue
        asp = max(bw, bh) / max(1.0, min(bw, bh))
        if asp > 2.2:
            continue
        sub = b["labels_ref"][y0:y1, x0:x1] == b["label"]
        v_cnt = (
            int(np.sum(blue[y0:y1, x0:x1] & sub) >= 4)
            + int(np.sum(green[y0:y1, x0:x1] & sub) >= 4)
            + int(np.sum(yellow[y0:y1, x0:x1] & sub) >= 4)
            + int(np.sum(red[y0:y1, x0:x1] & sub) >= 4)
        )
        dc = int(np.sum(dark[y0:y1, x0:x1] & sub))
        vc = int(np.sum(vivid[y0:y1, x0:x1] & sub))
        if v_cnt < 2 or dc < 12 or vc < 20:
            continue
        sc = v_cnt * 220.0 + min(dc, 600) * 0.4 + min(vc, 1000) * 0.25 - (asp - 1.0) * 90.0
        cands.append((sc, (x0, y0, bw, bh)))
    cands.sort(key=lambda item: -item[0])
    return [c[1] for c in cands[:top_k]]


def _gripper_occlusion_mask(patch: np.ndarray, use_cv2: bool = True) -> np.ndarray:
    """Detect large bright-white or dark-rubber robot arm/gripper regions touching the ROI."""
    ph, pw = patch.shape[:2]
    _h, s, v = _rgb_to_hsv_np(patch)
    # Large bright-white arm/gripper shells or dark rubber grip pads
    bright_arm = (s <= 0.24) & (v >= 0.62)
    dark_pad = v < 0.16
    occ = np.zeros((ph, pw), dtype=np.bool_)
    min_arm_area = max(320, int(0.008 * ph * pw))
    for comp_mask in (bright_arm, dark_pad):
        for b in _connected_blobs(comp_mask, use_cv2=use_cv2, connectivity=4, min_area=min_arm_area):
            x0, y0, x1, y1 = b["bbox"]
            touches_border = (x0 <= 2) or (y0 <= 2) or (x1 >= pw - 2) or (y1 >= ph - 2)
            if touches_border or b["area"] >= max(600, int(0.025 * ph * pw)):
                occ[y0:y1, x0:x1] |= (b["labels_ref"][y0:y1, x0:x1] == b["label"])
    return _box_any(occ, 5, use_cv2=use_cv2)


def _extract_enclosed_tiles(
    patch: np.ndarray,
    use_cv2: bool = True,
    occlusion_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Segment dark-seam-bounded sticker tiles inside an ROI patch."""
    ph, pw = patch.shape[:2]
    val = patch.max(axis=2).astype(np.float32)
    bh_v = _blackhat_2d(val, 11, use_cv2=use_cv2)
    gmag = _color_grad_mag(patch, use_cv2=use_cv2)

    seams = (val < 58.0) | (bh_v > 26.0) | ((val < 95.0) & (bh_v > 14.0)) | (gmag > 95.0)
    non_seam = ~seams
    blobs = _connected_blobs(non_seam, use_cv2=use_cv2, connectivity=4, min_area=18)

    max_tile_area = max(1200, int(0.06 * ph * pw))
    max_side = max(45, int(0.28 * max(pw, ph)))
    tiles: list[dict[str, Any]] = []
    for b in blobs:
        tx0, ty0, tx1, ty1 = b["bbox"]
        tbw, tbh, tarea = b["bw"], b["bh"], b["area"]
        if tx0 <= 1 or ty0 <= 1 or tx1 >= pw - 1 or ty1 >= ph - 1:
            continue
        if not (18 <= tarea <= max_tile_area and 4 <= tbw <= max_side and 4 <= tbh <= max_side):
            continue
        extent = tarea / float(tbw * tbh)
        asp = max(tbw, tbh) / float(min(tbw, tbh))
        if extent < 0.40 or asp > 3.0:
            continue
        sub = b["labels_ref"][ty0:ty1, tx0:tx1] == b["label"]
        px_vals = patch[ty0:ty1, tx0:tx1][sub]
        if float(np.std(px_vals, axis=0).mean()) >= 42.0:
            continue
        med_rgb = np.median(px_vals, axis=0)
        code, _ = classify_rgb_sticker(med_rgb)
        if code == "?":
            continue
        if code == "W" and occlusion_mask is not None:
            if float(np.mean(occlusion_mask[ty0:ty1, tx0:tx1][sub])) > 0.15:
                continue
        tiles.append({
            "cx": b["cx"],
            "cy": b["cy"],
            "area": tarea,
            "bw": tbw,
            "bh": tbh,
            "rgb": med_rgb.astype(int).tolist(),
            "code": code,
        })

    colored = [t for t in tiles if t["code"] != "W"]
    ref_tiles = colored if len(colored) >= 4 else tiles
    if len(ref_tiles) >= 3:
        med_a = float(np.median([t["area"] for t in ref_tiles]))
        tiles = [t for t in tiles if 0.16 * med_a <= t["area"] <= 2.6 * med_a]
    return tiles


def _fit_best_lattice(tiles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]] | None:
    """Fit the strongest 3x3 planar parallelogram lattice to enclosed sticker centroids."""
    n = len(tiles)
    if n < 4:
        return None
    pts = np.array([[t["cx"], t["cy"]] for t in tiles], dtype=np.float64)
    med_sz = float(np.median([math.sqrt(max(1.0, float(t["area"]))) for t in tiles]))
    min_step = max(6.0, 0.90 * med_sz)
    max_step = max(18.0, 2.6 * med_sz)

    best_model: tuple[np.ndarray, list[tuple[int, int]]] | None = None
    best_score = -1.0

    for i in range(n):
        diffs = pts - pts[i]
        dists = np.linalg.norm(diffs, axis=1)
        nbrs = sorted(
            (j for j in range(n) if j != i and min_step <= dists[j] <= max_step),
            key=lambda j: float(dists[j]),
        )[:6]
        if len(nbrs) < 2:
            continue
        for a_idx in range(len(nbrs)):
            u0 = diffs[nbrs[a_idx]]
            nu = float(np.linalg.norm(u0))
            for b_idx in range(a_idx + 1, len(nbrs)):
                v0 = diffs[nbrs[b_idx]]
                nv = float(np.linalg.norm(v0))
                if max(nu, nv) / min(nu, nv) > 2.2:
                    continue
                det = u0[0] * v0[1] - u0[1] * v0[0]
                if abs(det) / (nu * nv) < 0.35:
                    continue
                u_cand, v_cand = (u0, v0) if det > 0 else (v0, u0)
                mat = np.column_stack([u_cand, v_cand])
                coords = np.linalg.solve(mat, (pts - pts[i]).T).T
                rounded = np.round(coords)
                errs = np.linalg.norm((coords - rounded) @ mat.T, axis=1)
                tol = 0.32 * min(nu, nv)
                inliers = np.nonzero(errs <= tol)[0]
                if len(inliers) < 4:
                    continue
                ic = rounded[inliers, 0].astype(int)
                ir = rounded[inliers, 1].astype(int)
                for c0 in range(int(ic.min()), int(ic.max()) + 1):
                    for r0 in range(int(ir.min()), int(ir.max()) + 1):
                        sel = (ic >= c0) & (ic <= c0 + 2) & (ir >= r0) & (ir <= r0 + 2)
                        cnt = int(np.sum(sel))
                        if cnt < 4:
                            continue
                        idx_sel = inliers[sel]
                        cr_pairs = list(zip((ic[sel] - c0).tolist(), (ir[sel] - r0).tolist()))
                        if len(set(cr_pairs)) != len(cr_pairs):
                            continue
                        mean_err = float(np.mean(errs[idx_sel]))
                        sc = cnt * 100.0 - mean_err
                        if sc > best_score:
                            best_score = sc
                            best_model = (idx_sel, cr_pairs)

    if best_model is None:
        return None

    idx_sel, cr_pairs = best_model
    mat_a = np.array([[1.0, float(c), float(r)] for c, r in cr_pairs], dtype=np.float64)
    coef, _, _, _ = np.linalg.lstsq(mat_a, pts[idx_sel], rcond=None)
    p0, u_ref, v_ref = coef[0], coef[1], coef[2]

    mat_ref = np.column_stack([u_ref, v_ref])
    coords = np.linalg.solve(mat_ref, (pts - p0).T).T
    matched: dict[tuple[int, int], tuple[int, float]] = {}
    tol = 0.36 * min(float(np.linalg.norm(u_ref)), float(np.linalg.norm(v_ref)))
    for ti in range(n):
        c_rnd = int(round(coords[ti, 0]))
        r_rnd = int(round(coords[ti, 1]))
        if 0 <= c_rnd <= 2 and 0 <= r_rnd <= 2:
            pred = p0 + c_rnd * u_ref + r_rnd * v_ref
            err = float(np.linalg.norm(pts[ti] - pred))
            if err <= tol:
                if (r_rnd, c_rnd) not in matched or err < matched[(r_rnd, c_rnd)][1]:
                    matched[(r_rnd, c_rnd)] = (ti, err)
    if len(matched) >= 4:
        mat_a2 = np.array([[1.0, float(c), float(r)] for (r, c) in matched.keys()], dtype=np.float64)
        pts_y2 = np.array([pts[ti] for (ti, _) in matched.values()], dtype=np.float64)
        coef2, _, _, _ = np.linalg.lstsq(mat_a2, pts_y2, rcond=None)
        p0, u_ref, v_ref = coef2[0], coef2[1], coef2[2]

    corners_c = np.array([p0, p0 + 2 * u_ref, p0 + 2 * u_ref + 2 * v_ref, p0 + 2 * v_ref])
    tl_idx = int(np.argmin(corners_c[:, 0] + corners_c[:, 1]))
    if tl_idx == 1:
        p0 = corners_c[1]
        u_ref, v_ref = v_ref, -u_ref
    elif tl_idx == 2:
        p0 = corners_c[2]
        u_ref, v_ref = -u_ref, -v_ref
    elif tl_idx == 3:
        p0 = corners_c[3]
        u_ref, v_ref = -v_ref, u_ref
    if u_ref[0] * v_ref[1] - u_ref[1] * v_ref[0] < 0:
        u_ref, v_ref = v_ref, u_ref

    return p0, u_ref, v_ref, {ti for ti, _ in matched.values()}


def _quad_from_lattice(p0: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array([
        p0 - 0.5 * u - 0.5 * v,
        p0 + 2.5 * u - 0.5 * v,
        p0 + 2.5 * u + 2.5 * v,
        p0 - 0.5 * u + 2.5 * v,
    ], dtype=np.float64)


def _find_adjacent_side_faces(
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rem_tiles: list[dict[str, Any]],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], tuple[int, float]]]]:
    """Recover adjacent side face(s) sharing an edge with the primary face via 1-DOF vertical step w."""
    if not rem_tiles:
        return []
    rem_pts = np.array([[t["cx"], t["cy"]] for t in rem_tiles], dtype=np.float64)
    has_colored = any(t["code"] != "W" for t in rem_tiles)
    fit_indices = [i for i, t in enumerate(rem_tiles) if (not has_colored or t["code"] != "W")]

    edges = [
        ("bottom", p0 + 2.0 * v, u, v),
        ("right", p0 + 2.0 * u, v, u),
        ("left", p0, v, -u),
    ]
    scored_edges = []
    for ename, origin_pt, e_along, e_out in edges:
        edge_len = float(np.linalg.norm(e_along))
        out_len = float(np.linalg.norm(e_out))
        mat_e = np.column_stack([e_along, e_out])
        if abs(float(np.linalg.det(mat_e))) < 1e-6:
            continue
        rel = np.linalg.solve(mat_e, (rem_pts - origin_pt).T).T
        cands = []
        for ti in fit_indices:
            cc, oc = rel[ti]
            c_rnd = int(round(cc))
            if 0 <= c_rnd <= 2 and abs(cc - c_rnd) <= 0.35 and 0.45 <= oc <= 4.5:
                edge_pt = origin_pt + c_rnd * e_along + 0.5 * e_out
                delta = rem_pts[ti] - edge_pt
                if delta[1] > 0.25 * out_len:
                    for r_guess in (0, 1, 2):
                        w_cand = delta / (r_guess + 0.5)
                        wn = float(np.linalg.norm(w_cand))
                        if 0.45 * edge_len <= wn <= 1.25 * edge_len and w_cand[1] > 0.65 * wn:
                            cands.append(w_cand)
        best_local = None
        for w_cand in cands:
            wn = float(np.linalg.norm(w_cand))
            tol = max(3.5, 0.28 * min(edge_len, wn))
            hits: dict[tuple[int, int], tuple[int, float]] = {}
            for ti in fit_indices:
                for r in range(3):
                    for c in range(3):
                        pred = origin_pt + c * e_along + 0.5 * e_out + (r + 0.5) * w_cand
                        err = float(np.linalg.norm(rem_pts[ti] - pred))
                        if err <= tol:
                            if (r, c) not in hits or err < hits[(r, c)][1]:
                                hits[(r, c)] = (ti, err)
            if len(hits) >= 2:
                num = np.zeros(2, dtype=np.float64)
                den = 0.0
                for (r, c), (ti, _) in hits.items():
                    edge_pt = origin_pt + c * e_along + 0.5 * e_out
                    k_r = r + 0.5
                    num += k_r * (rem_pts[ti] - edge_pt)
                    den += k_r * k_r
                w_ref = num / max(1e-6, den)
                hits2: dict[tuple[int, int], tuple[int, float]] = {}
                for ti in fit_indices:
                    for r in range(3):
                        for c in range(3):
                            pred = origin_pt + c * e_along + 0.5 * e_out + (r + 0.5) * w_ref
                            err = float(np.linalg.norm(rem_pts[ti] - pred))
                            if err <= tol:
                                if (r, c) not in hits2 or err < hits2[(r, c)][1]:
                                    hits2[(r, c)] = (ti, err)
                sc = len(hits2) * 100.0 - sum(val[1] for val in hits2.values())
                if best_local is None or sc > best_local[0]:
                    best_local = (sc, ename, origin_pt, e_along, e_out, w_ref, hits2)
        if best_local is not None:
            scored_edges.append(best_local)

    scored_edges.sort(key=lambda x: -x[0])
    out_faces = []
    used_global: set[int] = set()
    w_shared = None
    claimed_edges: set[str] = set()
    for _sc, ename, origin_pt, e_along, e_out, w_ref, hits in scored_edges:
        fresh_hits = {rc: val for rc, val in hits.items() if val[0] not in used_global}
        if len(fresh_hits) < 2:
            continue
        if w_shared is None:
            w_shared = w_ref
        else:
            w_ref = w_shared
        for _, (ti, _) in fresh_hits.items():
            used_global.add(ti)
        claimed_edges.add(ename)
        if e_along[0] < 0 and abs(e_along[0]) > 0.25 * abs(e_along[1]):
            p0_side = origin_pt + 2.0 * e_along + 0.5 * e_out + 0.5 * w_ref
            u_side = -e_along
        else:
            p0_side = origin_pt + 0.5 * e_out + 0.5 * w_ref
            u_side = e_along
        out_faces.append((ename, p0_side, u_side, w_ref, fresh_hits))

    if w_shared is not None and len(out_faces) == 1:
        for ename, origin_pt, e_along, e_out in edges:
            if ename in claimed_edges:
                continue
            if e_out[1] <= 0.1 * float(np.linalg.norm(e_out)):
                continue
            edge_len = float(np.linalg.norm(e_along))
            tol = max(4.0, 0.35 * min(edge_len, float(np.linalg.norm(w_shared))))
            hits_extra: dict[tuple[int, int], tuple[int, float]] = {}
            for ti in fit_indices:
                if ti in used_global:
                    continue
                for r in range(3):
                    for c in range(3):
                        pred = origin_pt + c * e_along + 0.5 * e_out + (r + 0.5) * w_shared
                        err = float(np.linalg.norm(rem_pts[ti] - pred))
                        if err <= tol:
                            if (r, c) not in hits_extra or err < hits_extra[(r, c)][1]:
                                hits_extra[(r, c)] = (ti, err)
            if len(hits_extra) >= 2:
                if e_along[0] < 0 and abs(e_along[0]) > 0.25 * abs(e_along[1]):
                    p0_side = origin_pt + 2.0 * e_along + 0.5 * e_out + 0.5 * w_shared
                    u_side = -e_along
                else:
                    p0_side = origin_pt + 0.5 * e_out + 0.5 * w_shared
                    u_side = e_along
                out_faces.append((ename, p0_side, u_side, w_shared, hits_extra))
                break
    return out_faces


def _sample_lattice_face(
    patch: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    tiles: list[dict[str, Any]],
    *,
    x_off: int,
    y_off: int,
    full_w: int,
    full_h: int,
    is_side: bool = False,
    occlusion_mask: np.ndarray | None = None,
) -> tuple[list[list[str]], list[list[list[int]]], list[list[list[float]]], list[float]]:
    ph, pw = patch.shape[:2]
    grid = empty_grid()
    cell_rgb: list[list[list[int]]] = [[[] for _ in range(3)] for _ in range(3)]
    cell_centers: list[list[list[float]]] = [[[] for _ in range(3)] for _ in range(3)]
    confs: list[float] = []
    step_min = min(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    step_max = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    half = max(1, int(round(step_min * 0.22)))
    tile_pts = np.array([[t["cx"], t["cy"]] for t in tiles], dtype=np.float64) if tiles else None

    for r in range(3):
        for c in range(3):
            pt = p0 + c * u + r * v
            px, py = float(pt[0]), float(pt[1])
            cell_centers[r][c] = [round((px + x_off) / full_w, 4), round((py + y_off) / full_h, 4)]
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= pw or iy < 0 or iy >= ph:
                cell_rgb[r][c] = [0, 0, 0]
                confs.append(0.15)
                continue

            matched_tile = None
            if tile_pts is not None and len(tile_pts) > 0:
                dists = np.linalg.norm(tile_pts - pt, axis=1)
                idx_m = int(np.argmin(dists))
                if float(dists[idx_m]) <= 0.45 * step_max:
                    matched_tile = tiles[idx_m]

            sub = patch[max(0, iy - half):min(ph, iy + half + 1), max(0, ix - half):min(pw, ix + half + 1)].reshape(-1, 3)
            bright = sub[sub.max(axis=1) > 45]
            samp = bright if len(bright) >= max(2, len(sub) // 3) else sub
            med = np.median(samp, axis=0)
            med_int = [int(round(med[0])), int(round(med[1])), int(round(med[2]))]
            cell_rgb[r][c] = med_int
            code, c_conf = classify_rgb_sticker(med_int)

            if is_side:
                if occlusion_mask is not None and occlusion_mask[iy, ix]:
                    code, c_conf = "?", 0.15
                elif code == "W" and (matched_tile is None or matched_tile["code"] != "W"):
                    code, c_conf = "?", 0.15
                elif matched_tile is None and code != "?":
                    _ha, sa, va = _rgb_to_hsv_np(np.array([[med_int]], dtype=np.uint8))
                    if float(sa[0, 0]) < 0.40 or float(va[0, 0]) < 0.25:
                        code, c_conf = "?", 0.15

            grid[r][c] = code
            confs.append(c_conf)

    return grid, cell_rgb, cell_centers, confs


def _rotate_quad_centers_grid(
    quad_px: np.ndarray,
    grid: list[list[str]],
    cell_rgb: list[list[list[int]]],
    cell_centers: list[list[list[float]]],
    turns_cw: int,
) -> tuple[np.ndarray, list[list[str]], list[list[list[int]]], list[list[list[float]]]]:
    t = turns_cw % 4
    if t == 0:
        return quad_px, grid, cell_rgb, cell_centers
    q = quad_px.copy()
    g = [list(row) for row in grid]
    cr = [list(row) for row in cell_rgb]
    cc = [list(row) for row in cell_centers]
    for _ in range(t):
        q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        g = [[g[2 - c][r] for c in range(3)] for r in range(3)]
        cr = [[cr[2 - c][r] for c in range(3)] for r in range(3)]
        cc = [[cc[2 - c][r] for c in range(3)] for r in range(3)]
    return q, g, cr, cc


def _detect_cube_lattice_multiface(
    img: Image.Image,
    *,
    source: str,
    use_cv2: bool,
) -> dict[str, Any] | None:
    """Detect tabletop Rubik's cube using dark-seam + multi-color ROI seeding and 3D lattice fitting."""
    orig_w, orig_h = img.size
    seed_scale = min(1.0, 480.0 / max(orig_w, orig_h))
    sw = max(32, int(round(orig_w * seed_scale)))
    sh = max(32, int(round(orig_h * seed_scale)))
    small_rgb = np.asarray(img.resize((sw, sh), Image.BILINEAR) if seed_scale < 1.0 else img, dtype=np.uint8)

    cands = _find_seed_candidates(small_rgb, use_cv2=use_cv2, top_k=3)
    if not cands:
        return None

    # Cap native ROI processing at <=1280px max image scale (preserves full 1280x720 front camera detail)
    work_scale = min(1.0, 1280.0 / max(orig_w, orig_h))
    if work_scale < 1.0:
        work_w = max(32, int(round(orig_w * work_scale)))
        work_h = max(32, int(round(orig_h * work_scale)))
        work_arr = np.asarray(img.resize((work_w, work_h), Image.BILINEAR), dtype=np.uint8)
    else:
        work_w, work_h = orig_w, orig_h
        work_arr = np.asarray(img, dtype=np.uint8)

    rel_scale = seed_scale / work_scale
    best_result = None
    for sx, sy, sbw, sbh in cands:
        cx = int(round((sx + sbw * 0.5) / rel_scale))
        cy = int(round((sy + sbh * 0.5) / rel_scale))
        half = int(round(max(70.0, max(sbw, sbh) * 2.25) / rel_scale))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(work_w, cx + half), min(work_h, cy + half)
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        patch = work_arr[y0:y1, x0:x1]
        occ_mask = _gripper_occlusion_mask(patch, use_cv2=use_cv2)
        tiles = _extract_enclosed_tiles(patch, use_cv2=use_cv2, occlusion_mask=occ_mask)
        fit = _fit_best_lattice(tiles)
        if fit is None:
            continue
        p0, u, v, used = fit
        if len(used) < 5:
            continue
        rem = [tiles[k] for k in range(len(tiles)) if k not in used]
        adj = _find_adjacent_side_faces(p0, u, v, rem)
        total_matched = len(used) + sum(len(a[4]) for a in adj)
        if best_result is None or total_matched > best_result[0]:
            best_result = (total_matched, x0, y0, patch, occ_mask, tiles, rem, p0, u, v, used, adj)

    if best_result is None:
        return None

    _, x0, y0, patch, occ_mask, tiles, rem, p0, u, v, used, adj = best_result

    q1 = _quad_from_lattice(p0, u, v) + np.array([x0, y0])
    g1, rgb1, cc1, confs1 = _sample_lattice_face(
        patch, p0, u, v, tiles, x_off=x0, y_off=y0, full_w=work_w, full_h=work_h, is_side=False
    )
    known1 = sum(1 for row in g1 for cell in row if cell != "?")
    distinct1 = {cell for row in g1 for cell in row if cell != "?"}
    if known1 < 5 or (len(distinct1) < 2 and g1[1][1] != "W"):
        return None

    visible_faces: list[dict[str, Any]] = []
    if not adj:
        q1_norm = [[round(float(x) / work_w, 4), round(float(y) / work_h, 4)] for x, y in q1]
        visible_faces.append({
            "face": "F",
            "quad": q1_norm,
            "cell_centers": cc1,
            "grid": g1,
            "cell_rgb": rgb1,
            "matched_tiles": len(used),
            "known_stickers": known1,
            "confidence": round(float(np.mean(confs1)) * (known1 / 9.0), 3),
        })
        primary_quad_px = q1
        primary_face_entry = visible_faces[0]
    else:
        # Align U so row 2 touches the primary side face F's row 0!
        side_entries: list[tuple[str, np.ndarray, list[list[str]], list[list[list[int]]], list[list[list[float]]], list[float], int, int]] = []
        for ename, ps, us, vs, hits in adj:
            qs = _quad_from_lattice(ps, us, vs) + np.array([x0, y0])
            gs, rgbs, ccs, confss = _sample_lattice_face(
                patch, ps, us, vs, rem,
                x_off=x0, y_off=y0, full_w=work_w, full_h=work_h,
                is_side=True, occlusion_mask=occ_mask,
            )
            ks = sum(1 for row in gs for cell in row if cell != "?")
            if ks >= 2:
                side_entries.append((ename, qs, gs, rgbs, ccs, confss, len(hits), ks))

        if not side_entries:
            q1_norm = [[round(float(x) / work_w, 4), round(float(y) / work_h, 4)] for x, y in q1]
            visible_faces.append({
                "face": "F",
                "quad": q1_norm,
                "cell_centers": cc1,
                "grid": g1,
                "cell_rgb": rgb1,
                "matched_tiles": len(used),
                "known_stickers": known1,
                "confidence": round(float(np.mean(confs1)) * (known1 / 9.0), 3),
            })
            primary_quad_px = q1
            primary_face_entry = visible_faces[0]
        else:
            # Sort side faces left-to-right by x centroid; leftmost becomes F, rightmost R
            side_entries.sort(key=lambda item: float(item[1][:, 0].mean()))
            f_edge_name = side_entries[0][0]
            turns_u = {"bottom": 0, "right": 1, "top": 2, "left": 3}.get(f_edge_name, 0)
            q1_rot, g1_rot, rgb1_rot, cc1_rot = _rotate_quad_centers_grid(q1, g1, rgb1, cc1, turns_u)
            q1_norm = [[round(float(x) / work_w, 4), round(float(y) / work_h, 4)] for x, y in q1_rot]
            u_entry = {
                "face": "U",
                "quad": q1_norm,
                "cell_centers": cc1_rot,
                "grid": g1_rot,
                "cell_rgb": rgb1_rot,
                "matched_tiles": len(used),
                "known_stickers": known1,
                "confidence": round(float(np.mean(confs1)) * (known1 / 9.0), 3),
            }
            visible_faces.append(u_entry)

            side_labels = ["F", "R"] if len(side_entries) >= 2 else ["F"]
            for s_idx, (_ename, qs, gs, rgbs, ccs, confss, n_hits, ks) in enumerate(side_entries[:2]):
                qs_norm = [[round(float(x) / work_w, 4), round(float(y) / work_h, 4)] for x, y in qs]
                visible_faces.append({
                    "face": side_labels[s_idx],
                    "quad": qs_norm,
                    "cell_centers": ccs,
                    "grid": gs,
                    "cell_rgb": rgbs,
                    "matched_tiles": n_hits,
                    "known_stickers": ks,
                    "confidence": round(float(np.mean(confss)) * (ks / 9.0), 3),
                })

            # Pick F if mostly complete (>=7), otherwise U (most complete face) as primary backward-compatible quad
            f_entry = visible_faces[1]
            if f_entry["known_stickers"] >= 7:
                primary_face_entry = f_entry
                primary_quad_px = side_entries[0][1]
            else:
                primary_face_entry = u_entry
                primary_quad_px = q1_rot

    all_quads = np.array([pt for vf in visible_faces for pt in vf["quad"]], dtype=np.float64)
    bbox_norm = [
        round(float(all_quads[:, 0].min()), 4),
        round(float(all_quads[:, 1].min()), 4),
        round(float(all_quads[:, 0].max()), 4),
        round(float(all_quads[:, 1].max()), 4),
    ]
    total_known = sum(vf["known_stickers"] for vf in visible_faces)
    return {
        "detected": True,
        "confidence": primary_face_entry["confidence"],
        "source": source,
        "bbox": bbox_norm,
        "quad": primary_face_entry["quad"],
        "cell_centers": primary_face_entry["cell_centers"],
        "grid": primary_face_entry["grid"],
        "cell_rgb": primary_face_entry["cell_rgb"],
        "estimated_pose": _estimate_pose(primary_quad_px, work_w, work_h),
        "visible_faces": visible_faces,
        "total_known_stickers": total_known,
    }


def _blobs_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check spatial proximity and sticker size/shape compatibility."""
    sa = float(max(a["bw"], a["bh"]))
    sb = float(max(b["bw"], b["bh"]))
    if max(sa, sb) / max(1.0, min(sa, sb)) > 2.2:
        return False
    area_ratio = max(a["area"], b["area"]) / max(1.0, min(a["area"], b["area"]))
    if area_ratio > 3.2:
        return False
    asp_a = max(a["bw"], a["bh"]) / max(1.0, min(a["bw"], a["bh"]))
    asp_b = max(b["bw"], b["bh"]) / max(1.0, min(b["bw"], b["bh"]))
    if max(asp_a, asp_b) / max(1.0, min(asp_a, asp_b)) > 1.9:
        return False
    dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
    return dist <= max(12.0, 3.6 * max(sa, sb))


def _cluster_compatible_blobs(blobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    n = len(blobs)
    visited = [False] * n
    groups: list[list[dict[str, Any]]] = []
    for i in range(n):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        group: list[dict[str, Any]] = []
        while q:
            idx = q.popleft()
            bi = blobs[idx]
            group.append(bi)
            for j in range(n):
                if not visited[j] and _blobs_compatible(bi, blobs[j]):
                    visited[j] = True
                    q.append(j)
        groups.append(group)
    return groups


def _detect_cube_blob_cluster(
    img: Image.Image,
    *,
    source: str,
    use_cv2: bool,
) -> dict[str, Any]:
    """Fallback single-face cluster detector for stickerless and all-white cube views."""
    orig_w, orig_h = img.size
    target_max = 640.0
    scale = min(1.0, target_max / max(orig_w, orig_h))
    work_w = max(32, int(round(orig_w * scale)))
    work_h = max(32, int(round(orig_h * scale)))
    work_img = img.resize((work_w, work_h), Image.BILINEAR) if scale < 1.0 else img

    rgb = np.asarray(work_img, dtype=np.uint8)
    _h, s, v = _rgb_to_hsv_np(rgb)
    img_area = work_w * work_h
    min_sticker_area = max(12.0, 0.00008 * img_area)
    max_sticker_area = 0.20 * img_area

    white_mask = (s <= 0.30) & (v >= 0.42)
    white_blobs_all: list[dict[str, Any]] | None = None
    cluster_blobs: list[dict[str, Any]] = []
    chrom_blobs_all: list[dict[str, Any]] = []

    for s_min, v_min in ((0.18, 0.14), (0.12, 0.12)):
        chrom_mask = (s >= s_min) & (v >= v_min)
        chrom_blobs_all = _connected_blobs(chrom_mask, use_cv2=use_cv2)
        sticker_blobs = [
            b for b in chrom_blobs_all
            if min_sticker_area <= b["area"] <= max_sticker_area
            and (b["area"] / max(1, b["bw"] * b["bh"])) >= 0.35
            and max(b["bw"], b["bh"]) / max(1, min(b["bw"], b["bh"])) <= 3.2
        ]
        if len(sticker_blobs) >= 2:
            best_group: list[dict[str, Any]] = []
            best_score = -1.0
            for group in _cluster_compatible_blobs(sticker_blobs):
                if len(group) < 2:
                    continue
                xs0 = [g["bbox"][0] for g in group]
                ys0 = [g["bbox"][1] for g in group]
                xs1 = [g["bbox"][2] for g in group]
                ys1 = [g["bbox"][3] for g in group]
                span_w = max(xs1) - min(xs0)
                span_h = max(ys1) - min(ys0)
                span = max(span_w, span_h, 1)
                density = sum(g["area"] for g in group) / float(span * span)
                aspect_penalty = max(span_w, span_h) / max(1.0, min(span_w, span_h))
                eff_count = min(len(group), 9)
                score = eff_count * 1.8 + density * 4.5 - max(0.0, aspect_penalty - 1.35) * 2.0
                if score > best_score:
                    best_score = score
                    best_group = group
            if best_group:
                cluster_blobs = list(best_group)
                c_xs0 = min(b["bbox"][0] for b in cluster_blobs)
                c_ys0 = min(b["bbox"][1] for b in cluster_blobs)
                c_xs1 = max(b["bbox"][2] for b in cluster_blobs)
                c_ys1 = max(b["bbox"][3] for b in cluster_blobs)
                unit_s = float(np.median([max(b["bw"], b["bh"]) for b in cluster_blobs]))
                med_area = float(np.median([b["area"] for b in cluster_blobs]))
                pad = max(8.0, 1.3 * unit_s)

                if white_blobs_all is None:
                    white_blobs_all = _connected_blobs(white_mask, use_cv2=use_cv2)
                for wb in white_blobs_all:
                    if not (min_sticker_area <= wb["area"] <= 0.06 * img_area):
                        continue
                    if max(wb["bw"], wb["bh"]) / max(1, min(wb["bw"], wb["bh"])) > 2.2:
                        continue
                    if wb["area"] / max(1, wb["bw"] * wb["bh"]) < 0.45:
                        continue
                    if (c_xs0 - pad <= wb["cx"] <= c_xs1 + pad) and (c_ys0 - pad <= wb["cy"] <= c_ys1 + pad):
                        if 0.4 * med_area <= wb["area"] <= 2.2 * med_area and max(wb["bw"], wb["bh"]) <= 1.8 * unit_s:
                            cluster_blobs.append(wb)
                if len(cluster_blobs) >= 3:
                    break

    white_only_mode = False
    if len(cluster_blobs) < 3:
        if white_blobs_all is None:
            white_blobs_all = _connected_blobs(white_mask, use_cv2=use_cv2)
        cand = [
            wb for wb in white_blobs_all
            if min_sticker_area <= wb["area"] <= 0.045 * img_area
            and max(wb["bw"], wb["bh"]) / max(1, min(wb["bw"], wb["bh"])) <= 1.8
            and (wb["area"] / max(1, wb["bw"] * wb["bh"])) >= 0.55
        ]
        if len(cand) >= 4:
            groups = [g for g in _cluster_compatible_blobs(cand) if len(g) >= 4]
            if groups:
                cluster_blobs = max(groups, key=len)
                white_only_mode = True

    used_merged_blob = False
    if len(cluster_blobs) < 3:
        merged_cands = [
            b for b in chrom_blobs_all
            if 0.002 * img_area <= b["area"] <= 0.35 * img_area
            and max(b["bw"], b["bh"]) / max(1, min(b["bw"], b["bh"])) <= 1.65
            and (b["area"] / max(1, b["bw"] * b["bh"])) >= 0.45
        ]
        if merged_cands:
            cluster_blobs = [max(merged_cands, key=lambda b: b["area"])]
            used_merged_blob = True
        else:
            return {
                "detected": False,
                "confidence": 0.0,
                "source": source,
                "bbox": None,
                "quad": None,
                "cell_centers": None,
                "grid": empty_grid(),
                "cell_rgb": None,
                "estimated_pose": None,
                "visible_faces": [],
            }

    pts_list = []
    for b in cluster_blobs:
        x0, y0, x1, y1 = b["bbox"]
        sub = b["labels_ref"][y0:y1, x0:x1] == b["label"]
        ry, rx = np.nonzero(sub)
        pts_list.append(np.column_stack((rx + x0, ry + y0)))
    pts_xy = np.vstack(pts_list)

    quad = _oriented_box_corners(pts_xy, use_cv2=use_cv2)
    quad = _split_if_elongated(quad, pts_xy)

    top_w = float(np.linalg.norm(quad[1] - quad[0]))
    left_h = float(np.linalg.norm(quad[3] - quad[0]))
    if top_w < 14 or left_h < 14 or max(top_w, left_h) / max(1.0, min(top_w, left_h)) > 1.85:
        return {
            "detected": False,
            "confidence": 0.0,
            "source": source,
            "bbox": None,
            "quad": None,
            "cell_centers": None,
            "grid": empty_grid(),
            "cell_rgb": None,
            "estimated_pose": None,
            "visible_faces": [],
        }

    grid = empty_grid()
    cell_rgb: list[list[list[int]]] = [[[] for _ in range(3)] for _ in range(3)]
    cell_centers: list[list[list[float]]] = [[[] for _ in range(3)] for _ in range(3)]
    cell_confs: list[float] = []
    patch_half = max(1, int(round(min(top_w, left_h) / 16.0)))

    for r_idx in range(3):
        for c_idx in range(3):
            u = (c_idx + 0.5) / 3.0
            v_coord = (r_idx + 0.5) / 3.0
            px, py = _bilinear_point(quad, u, v_coord)
            ix = max(0, min(work_w - 1, int(round(px))))
            iy = max(0, min(work_h - 1, int(round(py))))
            cell_centers[r_idx][c_idx] = [round(px / work_w, 4), round(py / work_h, 4)]

            x_lo = max(0, ix - patch_half)
            x_hi = min(work_w, ix + patch_half + 1)
            y_lo = max(0, iy - patch_half)
            y_hi = min(work_h, iy + patch_half + 1)
            patch = rgb[y_lo:y_hi, x_lo:x_hi].reshape(-1, 3)
            bright = patch[patch.mean(axis=1) > 42]
            sample = bright if len(bright) >= max(2, len(patch) // 3) else patch
            med = np.median(sample, axis=0)
            med_int = [int(round(med[0])), int(round(med[1])), int(round(med[2]))]
            cell_rgb[r_idx][c_idx] = med_int

            color_code, c_conf = classify_rgb_sticker(med_int)
            grid[r_idx][c_idx] = color_code
            cell_confs.append(c_conf)

    known = sum(1 for row in grid for cell in row if cell != "?")
    distinct_colors = {cell for row in grid for cell in row if cell != "?"}
    has_cool_or_red = bool(distinct_colors & {"R", "G", "B"})
    # Guard against wood-table or single-color non-cube false positives
    conf_est = float(np.mean(cell_confs)) * (known / 9.0)
    if (
        known < 5
        or conf_est < 0.42
        or (used_merged_blob and (len(distinct_colors) < 3 or not has_cool_or_red))
        or (not white_only_mode and (len(distinct_colors) < 2 or (not has_cool_or_red and known < 8)))
    ):
        return {
            "detected": False,
            "confidence": 0.0,
            "source": source,
            "bbox": None,
            "quad": None,
            "cell_centers": None,
            "grid": empty_grid(),
            "cell_rgb": None,
            "estimated_pose": None,
            "visible_faces": [],
        }

    quad_norm = [[round(float(x) / work_w, 4), round(float(y) / work_h, 4)] for x, y in quad]
    xs = [p[0] for p in quad_norm]
    ys = [p[1] for p in quad_norm]
    bbox_norm = [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
    conf = round(float(np.mean(cell_confs)) * (known / 9.0), 3)

    return {
        "detected": True,
        "confidence": conf,
        "source": source,
        "bbox": bbox_norm,
        "quad": quad_norm,
        "cell_centers": cell_centers,
        "grid": grid,
        "cell_rgb": cell_rgb,
        "estimated_pose": _estimate_pose(quad, work_w, work_h),
        "visible_faces": [
            {
                "face": "F",
                "quad": quad_norm,
                "cell_centers": cell_centers,
                "grid": grid,
                "cell_rgb": cell_rgb,
                "matched_tiles": len(cluster_blobs),
                "known_stickers": known,
                "confidence": conf,
            }
        ],
        "total_known_stickers": known,
    }


def detect_cube_cv(
    image_input: bytes | str | Image.Image,
    *,
    source: str = "",
    use_cv2: bool = True,
) -> dict[str, Any]:
    """Detect visible Rubik's cube face(s) and extract 3x3 sticker grids, quads, and pose."""
    img = _decode_image(image_input)
    lattice_res = _detect_cube_lattice_multiface(img, source=source, use_cv2=use_cv2)
    if lattice_res is not None:
        return lattice_res
    return _detect_cube_blob_cluster(img, source=source, use_cv2=use_cv2)


CLAUDE_CUBE_PROMPT = """Inspect the camera image(s) of a 3x3x3 Rubik's cube.
Identify every clearly visible 3x3 face of the cube and output ONE JSON object strictly in this format:
{
  "detected": true,
  "confidence": 0.0 to 1.0,
  "reason": "brief visual summary of which cube faces are visible",
  "primary_quad": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "faces": {
    "F": [["W","R","G"], ["B","Y","O"], ["W","G","R"]]
  }
}
Rules:
- Color codes MUST be one of: W (White), Y (Yellow), R (Red), O (Orange), B (Blue), G (Green), or ? if occluded.
- Face keys in "faces" must be from U, R, F, D, L, B matching standard relative cube geometry (F facing the camera, U on top, R on right, L on left, D on bottom, B in back).
- primary_quad coordinates are normalized [0.0..1.0] [TL, TR, BR, BL] of the primary visible face in the first image (or null if uncertain).
- Read each 3x3 grid strictly row-by-row from top-left to bottom-right as viewed from outside that face.
- If no Rubik's cube is visible, return {"detected": false, "confidence": 0.0, "reason": "No cube visible", "primary_quad": null, "faces": {}}."""


def scan_cube_with_claude(
    images_b64: dict[str, str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    """Use Anthropic Claude Vision to read 1 or more visible 3x3 faces of the Rubik's cube."""
    key = load_anthropic_api_key(api_key)
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    if not images_b64:
        raise ValueError("At least one camera image is required")

    content: list[dict[str, Any]] = []
    for name, raw_b64 in images_b64.items():
        clean = raw_b64.split(",", 1)[1] if "," in raw_b64 and raw_b64.startswith("data:image/") else raw_b64
        content.append({"type": "text", "text": f"Camera view: {name}"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": clean.strip()},
        })

    base = (base_url or "https://api.anthropic.com").rstrip("/")
    url = base + "/v1/messages"
    chosen_model = model or DEFAULT_CLAUDE_MODEL

    def _post(m: str) -> dict[str, Any]:
        body = {
            "model": m,
            "max_tokens": 768,
            "system": CLAUDE_CUBE_PROMPT,
            "messages": [{"role": "user", "content": content}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.load(resp)

    try:
        payload = _post(chosen_model)
    except urllib.error.HTTPError as err:
        if err.code != 404 or model:
            raise
        models_req = urllib.request.Request(
            base + "/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(models_req, timeout=min(timeout_s, 8.0)) as mresp:
            mdata = json.load(mresp)
        ids = [str(item.get("id", "")) for item in mdata.get("data", []) if item.get("id")]
        fallback = next((mid for p in ("claude-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-") for mid in ids if mid.startswith(p)), ids[0] if ids else "")
        if not fallback:
            raise
        payload = _post(fallback)

    text_parts = [blk.get("text", "") for blk in payload.get("content", []) if blk.get("type") == "text"]
    parsed = extract_json_object("\n".join(text_parts))

    faces_out: dict[str, list[list[str]]] = {}
    raw_faces = parsed.get("faces") or {}
    if isinstance(raw_faces, dict):
        for face_key, grid in raw_faces.items():
            fk = str(face_key).strip().upper()
            if fk in FACES:
                faces_out[fk] = normalize_grid(grid)

    return {
        "detected": bool(parsed.get("detected", bool(faces_out))),
        "confidence": float(parsed.get("confidence", 0.85 if faces_out else 0.0)),
        "reason": str(parsed.get("reason", "")),
        "primary_quad": parsed.get("primary_quad"),
        "faces": faces_out,
    }
