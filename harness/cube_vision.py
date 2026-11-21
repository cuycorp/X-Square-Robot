"""Real-time Rubik's cube detector and optional Claude multi-face reader.

Pure numpy + Pillow core (works without OpenCV), using cv2 when installed for speed.
"""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.request
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from claude_policy import DEFAULT_CLAUDE_MODEL, extract_json_object, load_anthropic_api_key
from cube_state import FACES, empty_grid, normalize_grid

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


def _connected_blobs(mask: np.ndarray, use_cv2: bool = True) -> list[dict[str, Any]]:
    h_img, w_img = mask.shape
    blobs: list[dict[str, Any]] = []
    if use_cv2 and HAS_CV2:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < 8:
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
            for ny in (cy - 1, cy, cy + 1):
                if 0 <= ny < h_img:
                    for nx in (cx - 1, cx, cx + 1):
                        if 0 <= nx < w_img and not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
        area = len(pts_x)
        if area < 8:
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

    if v < 0.14:
        return "?", 0.15

    if s <= 0.25 and v >= 0.45:
        conf = min(1.0, (0.28 - s) / 0.28 * 0.6 + v * 0.4)
        return "W", round(max(0.45, conf), 3)

    if 175.0 <= h < 268.0:
        return "B", round(min(1.0, 0.55 + 0.45 * s), 3)
    if 74.0 <= h < 175.0:
        return "G", round(min(1.0, 0.55 + 0.45 * s), 3)
    if 38.0 <= h < 74.0:
        return "Y", round(min(1.0, 0.55 + 0.45 * s), 3)

    if h >= 315.0:
        return "R", round(min(1.0, 0.55 + 0.45 * s), 3)

    g_over_r = g / max(1.0, r)
    if h < 13.0 and g_over_r < 0.28:
        return "R", round(min(1.0, 0.55 + 0.45 * s), 3)
    if 13.0 <= h < 38.0 or g_over_r >= 0.28:
        if g_over_r >= 0.64:
            return "Y", round(min(1.0, 0.50 + 0.40 * s), 3)
        return "O", round(min(1.0, 0.50 + 0.40 * s), 3)
    return "R", round(min(1.0, 0.50 + 0.40 * s), 3)


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


def detect_cube_cv(
    image_input: bytes | str | Image.Image,
    *,
    source: str = "",
    use_cv2: bool = True,
) -> dict[str, Any]:
    """Detect a Rubik's cube face and extract its 3x3 sticker grid + quad."""
    img = _decode_image(image_input)
    orig_w, orig_h = img.size
    scale = min(1.0, 320.0 / max(orig_w, orig_h))
    work_w = max(32, int(round(orig_w * scale)))
    work_h = max(32, int(round(orig_h * scale)))
    work_img = img.resize((work_w, work_h), Image.BILINEAR) if scale < 1.0 else img

    rgb = np.asarray(work_img, dtype=np.uint8)
    _h, s, v = _rgb_to_hsv_np(rgb)
    img_area = work_w * work_h

    chrom_mask = (s >= 0.28) & (v >= 0.20)
    white_mask = (s <= 0.24) & (v >= 0.65)

    chrom_blobs_all = _connected_blobs(chrom_mask, use_cv2=use_cv2)
    sticker_blobs = [
        b for b in chrom_blobs_all
        if 0.00045 * img_area <= b["area"] <= 0.20 * img_area
        and (b["area"] / max(1, b["bw"] * b["bh"])) >= 0.38
    ]

    cluster_blobs: list[dict[str, Any]] = []
    white_blobs_all: list[dict[str, Any]] = []

    if len(sticker_blobs) >= 2:
        sizes = [min(b["bw"], b["bh"]) for b in sticker_blobs]
        med_size = float(np.median(sizes))
        dist_thresh = max(14.0, 3.8 * med_size)

        n = len(sticker_blobs)
        visited = [False] * n
        best_group: list[dict[str, Any]] = []
        best_score = -1.0
        for i in range(n):
            if visited[i]:
                continue
            q = deque([i])
            visited[i] = True
            group = []
            while q:
                idx = q.popleft()
                b = sticker_blobs[idx]
                group.append(b)
                for j in range(n):
                    if not visited[j]:
                        bj = sticker_blobs[j]
                        d = math.hypot(b["cx"] - bj["cx"], b["cy"] - bj["cy"])
                        if d <= dist_thresh:
                            visited[j] = True
                            q.append(j)
            xs0 = [g["bbox"][0] for g in group]
            ys0 = [g["bbox"][1] for g in group]
            xs1 = [g["bbox"][2] for g in group]
            ys1 = [g["bbox"][3] for g in group]
            span = max(max(xs1) - min(xs0), max(ys1) - min(ys0), 1)
            density = sum(g["area"] for g in group) / float(span * span)
            score = len(group) * 1.5 + density * 4.0
            if len(group) >= 2 and score > best_score:
                best_score = score
                best_group = group
        cluster_blobs = best_group

        if cluster_blobs:
            c_xs0 = min(b["bbox"][0] for b in cluster_blobs)
            c_ys0 = min(b["bbox"][1] for b in cluster_blobs)
            c_xs1 = max(b["bbox"][2] for b in cluster_blobs)
            c_ys1 = max(b["bbox"][3] for b in cluster_blobs)
            unit_s = float(np.median([min(b["bw"], b["bh"]) for b in cluster_blobs]))
            pad = max(12.0, 2.6 * unit_s)

            white_blobs_all = _connected_blobs(white_mask, use_cv2=use_cv2)
            for wb in white_blobs_all:
                if not (0.00045 * img_area <= wb["area"] <= 0.06 * img_area):
                    continue
                if max(wb["bw"], wb["bh"]) / max(1, min(wb["bw"], wb["bh"])) > 2.2:
                    continue
                if wb["area"] / max(1, wb["bw"] * wb["bh"]) < 0.45:
                    continue
                if (c_xs0 - pad <= wb["cx"] <= c_xs1 + pad) and (c_ys0 - pad <= wb["cy"] <= c_ys1 + pad):
                    if min(wb["bw"], wb["bh"]) <= 2.4 * unit_s:
                        cluster_blobs.append(wb)
    else:
        white_blobs_all = _connected_blobs(white_mask, use_cv2=use_cv2)
        cand = [
            wb for wb in white_blobs_all
            if 0.0005 * img_area <= wb["area"] <= 0.045 * img_area
            and max(wb["bw"], wb["bh"]) / max(1, min(wb["bw"], wb["bh"])) <= 1.8
            and (wb["area"] / max(1, wb["bw"] * wb["bh"])) >= 0.55
        ]
        if len(cand) >= 6:
            cluster_blobs = cand

    used_merged_blob = False
    if len(cluster_blobs) < 3:
        merged_cands = [
            b for b in chrom_blobs_all
            if 0.008 * img_area <= b["area"] <= 0.35 * img_area
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
    if top_w < 18 or left_h < 18 or max(top_w, left_h) / max(1.0, min(top_w, left_h)) > 1.85:
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
    if known < 5 or (used_merged_blob and len(distinct_colors) < 2):
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
    }


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

    url = (base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
    body = {
        "model": model or DEFAULT_CLAUDE_MODEL,
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
        payload = json.load(resp)

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
