"""Canonical 6-face 3D Rubik's cube state accumulator, validator, and multi-camera correlator."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

FACES = ("U", "R", "F", "D", "L", "B")
AUTO_FACE_ORDER = ("F", "R", "U", "L", "B", "D")
SIDE_RING = ("F", "R", "B", "L")
VALID_COLORS = ("W", "Y", "R", "O", "B", "G")
UNKNOWN = "?"

COLOR_NAMES = {
    "W": "White",
    "Y": "Yellow",
    "R": "Red",
    "O": "Orange",
    "B": "Blue",
    "G": "Green",
    "?": "Unknown",
}


def empty_grid() -> list[list[str]]:
    return [[UNKNOWN, UNKNOWN, UNKNOWN] for _ in range(3)]


def rotate_grid_cw(grid: list[list[str]], turns: int = 1) -> list[list[str]]:
    out = [list(row) for row in grid]
    for _ in range(turns % 4):
        out = [[out[2 - c][r] for c in range(3)] for r in range(3)]
    return out


def normalize_grid(grid: Any) -> list[list[str]]:
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        raise ValueError("Grid must be a 3x3 list")
    out: list[list[str]] = []
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Each grid row must have 3 elements")
        norm_row: list[str] = []
        for cell in row:
            c = str(cell).strip().upper()
            if c not in VALID_COLORS and c != UNKNOWN:
                raise ValueError(f"Invalid sticker color: {cell!r}")
            norm_row.append(c)
        out.append(norm_row)
    return out


def _source_role(source: str) -> str:
    s = str(source or "").lower()
    if "front" in s or "head" in s:
        return "front"
    if "camera1" in s or "cam1" in s or "right_wrist" in s:
        return "cam1"
    if "camera3" in s or "cam3" in s or "left_wrist" in s:
        return "cam3"
    return s


def _source_priority(source: str) -> int:
    role = _source_role(source)
    if role == "front":
        return 0
    if role == "cam1":
        return 1
    if role == "cam3":
        return 2
    return 3


def _default_u_rotation_for_role(role: str) -> int:
    # Left wrist camera (cam3) views across from the right (+90 deg CW on U relative to front).
    if role == "cam3":
        return 1
    return 0


def _best_u_rotation(u_ref: list[list[str]], u_obs: list[list[str]], fallback_rot: int = 0) -> tuple[int, int, int]:
    """Return (k_cw, matches, mismatches) such that rotate_grid_cw(u_ref, k_cw) ~= u_obs."""
    best_k = fallback_rot % 4
    best_score = (-1, 99, -1)
    best_stats = (0, 0)
    ties: list[tuple[int, int, int]] = []
    for k in range(4):
        cand = rotate_grid_cw(u_ref, k)
        matches = 0
        mismatches = 0
        for r in range(3):
            for c in range(3):
                a, b = cand[r][c], u_obs[r][c]
                if a != UNKNOWN and b != UNKNOWN:
                    if a == b:
                        matches += 1
                    else:
                        mismatches += 1
        score = (matches - 2 * mismatches, -mismatches, 1 if k == (fallback_rot % 4) else 0)
        if score[:2] > best_score[:2]:
            best_score = score
            best_k = k
            best_stats = (matches, mismatches)
            ties = [(k, matches, mismatches)]
        elif score[:2] == best_score[:2]:
            ties.append((k, matches, mismatches))
            if k == (fallback_rot % 4):
                best_k = k
                best_stats = (matches, mismatches)
    if len(ties) > 1:
        for k, m, mm in ties:
            if k == (fallback_rot % 4):
                return k, m, mm
    return best_k, best_stats[0], best_stats[1]


class CubeStateManager:
    """Thread-safe 3D Rubik's cube state tracker and multi-camera correlator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faces: dict[str, dict[str, Any]] = {}
        self._live: dict[str, Any] = {
            "detected": False,
            "confidence": 0.0,
            "source": "",
            "bbox": None,
            "quad": None,
            "cell_centers": None,
            "grid": empty_grid(),
            "cell_rgb": None,
            "estimated_pose": None,
            "visible_faces": [],
            "total_known_stickers": 0,
            "updated_at": None,
        }
        self._live_by_source: dict[str, dict[str, Any]] = {}
        self._correlation: dict[str, Any] = {
            "active_cameras": [],
            "camera_mappings": {},
            "correlated_faces": [],
            "total_known_stickers": 0,
            "updated_at": None,
        }
        self._reset_faces_locked()

    def _reset_faces_locked(self) -> None:
        self._faces = {
            face: {
                "grid": empty_grid(),
                "scanned": False,
                "source": "",
                "sources": [],
                "mode": "",
                "confidence": 0.0,
                "updated_at": None,
            }
            for face in FACES
        }
        self._correlation = {
            "active_cameras": [],
            "camera_mappings": {},
            "correlated_faces": [],
            "total_known_stickers": 0,
            "updated_at": None,
        }

    def reset(self) -> None:
        with self._lock:
            self._reset_faces_locked()

    def update_live(
        self,
        detection: dict[str, Any],
        source: str = "",
        *,
        auto_correlate: bool = False,
        auto_fill: bool = True,
    ) -> None:
        src = source or str(detection.get("source", ""))
        entry = {
            "detected": bool(detection.get("detected", False)),
            "confidence": round(float(detection.get("confidence", 0.0)), 3),
            "source": src,
            "bbox": detection.get("bbox"),
            "quad": detection.get("quad"),
            "cell_centers": detection.get("cell_centers"),
            "grid": detection.get("grid") or empty_grid(),
            "cell_rgb": detection.get("cell_rgb"),
            "estimated_pose": detection.get("estimated_pose"),
            "visible_faces": copy.deepcopy(detection.get("visible_faces") or []),
            "total_known_stickers": int(detection.get("total_known_stickers", 0)),
            "updated_at": time.time(),
        }
        with self._lock:
            if src:
                self._live_by_source[src] = copy.deepcopy(entry)
            now = float(entry["updated_at"] or 0.0)
            cur = self._live
            keep_active_other = (
                not entry["detected"]
                and bool(cur.get("detected"))
                and bool(src)
                and bool(cur.get("source"))
                and src != cur.get("source")
                and (now - float(cur.get("updated_at") or 0.0)) < 4.0
            )
            if not keep_active_other:
                self._live = entry
            if auto_correlate:
                self._correlate_multicam_locked(auto_fill=auto_fill)

    def correlate_multicam(self, *, auto_fill: bool = True) -> dict[str, Any]:
        with self._lock:
            corr = self._correlate_multicam_locked(auto_fill=auto_fill)
            return {
                "correlation": copy.deepcopy(corr),
                "state": self._snapshot_locked(),
            }

    def _correlate_multicam_locked(self, *, auto_fill: bool = True) -> dict[str, Any]:
        now = time.time()
        # Use all active detected cameras (recent within 12s, or all detected in offline/test snapshots).
        active: list[tuple[str, dict[str, Any]]] = []
        for src, det in self._live_by_source.items():
            if not det.get("detected"):
                continue
            ts = float(det.get("updated_at") or now)
            if (now - ts) <= 12.0:
                active.append((src, det))
        if not active and self._live.get("detected"):
            src = str(self._live.get("source") or "live")
            active.append((src, self._live))

        if not active:
            self._correlation = {
                "active_cameras": [],
                "camera_mappings": {},
                "correlated_faces": [],
                "total_known_stickers": 0,
                "updated_at": now,
            }
            return self._correlation

        active.sort(key=lambda item: (_source_priority(item[0]), -int(item[1].get("total_known_stickers", 0))))

        # Extract normalized visible_faces list per camera.
        cam_faces: dict[str, list[dict[str, Any]]] = {}
        for src, det in active:
            vfs = det.get("visible_faces") or []
            if not vfs and det.get("grid"):
                vfs = [{
                    "face": "F",
                    "grid": det["grid"],
                    "confidence": det.get("confidence", 0.0),
                    "known_stickers": sum(1 for r in det["grid"] for c in r if c != UNKNOWN),
                }]
            cam_faces[src] = vfs

        # Establish canonical U reference from the highest-priority camera seeing U, or stored U.
        anchor_src = None
        anchor_u: list[list[str]] | None = None
        for src, _det in active:
            for vf in cam_faces[src]:
                if vf.get("face") == "U":
                    g = normalize_grid(vf.get("grid"))
                    if sum(1 for r in g for c in r if c != UNKNOWN) >= 5:
                        anchor_src = src
                        anchor_u = g
                        break
            if anchor_u is not None:
                break

        k_table = 0
        stored_u = self._faces["U"]["grid"] if self._faces["U"]["scanned"] else None
        if anchor_u is not None and anchor_src is not None:
            anchor_role = _source_role(anchor_src)
            anchor_default = _default_u_rotation_for_role(anchor_role)
            if anchor_default != 0:
                anchor_u = rotate_grid_cw(anchor_u, (4 - anchor_default) % 4)
            # Check whether the physical cube was turned horizontally on the table vs already stored U.
            if stored_u is not None:
                u_distinct = {c for row in stored_u for c in row if c != UNKNOWN}
                if len(u_distinct) >= 2:
                    k_best, m_best, mm_best = _best_u_rotation(stored_u, anchor_u, fallback_rot=0)
                    if m_best >= 6 and mm_best <= 1:
                        k_table = k_best
                        anchor_u = rotate_grid_cw(anchor_u, (4 - k_table) % 4)
                    elif stored_u[1][1] != anchor_u[1][1] and self._faces["U"].get("mode") == "auto":
                        # Cube was flipped onto a different top face; clear stale auto faces.
                        for fkey in FACES:
                            if self._faces[fkey].get("mode") == "auto":
                                self._faces[fkey] = {
                                    "grid": empty_grid(),
                                    "scanned": False,
                                    "source": "",
                                    "sources": [],
                                    "mode": "",
                                    "confidence": 0.0,
                                    "updated_at": None,
                                }
        elif stored_u is not None:
            anchor_u = [list(row) for row in stored_u]

        # Keep the distinct colors observed at each canonical sticker position.
        observations: dict[str, list[list[set[str]]]] = {
            f: [[set() for _ in range(3)] for _ in range(3)] for f in FACES
        }
        face_sources: dict[str, list[str]] = {f: [] for f in FACES}
        face_confs: dict[str, list[float]] = {f: [] for f in FACES}
        camera_mappings: dict[str, Any] = {}

        for src, det in active:
            role = _source_role(src)
            vfs = cam_faces[src]
            u_vf = next((vf for vf in vfs if vf.get("face") == "U"), None)
            side_vfs = [vf for vf in vfs if vf.get("face") != "U"]

            k_cam_rel = _default_u_rotation_for_role(role)
            if u_vf is not None and anchor_u is not None:
                u_obs = normalize_grid(u_vf.get("grid"))
                u_distinct = {c for row in anchor_u for c in row if c != UNKNOWN}
                if len(u_distinct) >= 2:
                    k_fit, m_fit, mm_fit = _best_u_rotation(anchor_u, u_obs, fallback_rot=(k_table + k_cam_rel) % 4)
                    if m_fit >= 4 and mm_fit <= 1:
                        k_total = k_fit
                    else:
                        k_total = (k_table + k_cam_rel) % 4
                else:
                    k_total = (k_table + k_cam_rel) % 4
            else:
                k_total = (k_table + k_cam_rel) % 4

            per_src_map: dict[str, str] = {}
            # Annotate live_by_source[src] visible_faces with canonical_face for UI overlays
            live_vfs = self._live_by_source.get(src, {}).get("visible_faces") or []

            if u_vf is not None:
                u_obs = normalize_grid(u_vf.get("grid"))
                u_canon = rotate_grid_cw(u_obs, (4 - k_total) % 4)
                conf_u = float(u_vf.get("confidence", det.get("confidence", 0.5)))
                for r in range(3):
                    for c in range(3):
                        col = u_canon[r][c]
                        if col != UNKNOWN:
                            observations["U"][r][c].add(col)
                if src not in face_sources["U"]:
                    face_sources["U"].append(src)
                face_confs["U"].append(conf_u)
                per_src_map["U"] = "U"
                for lvf in live_vfs:
                    if lvf.get("face") == "U":
                        lvf["canonical_face"] = "U"
                        lvf["rotation_cw"] = k_total

            for s_idx, svf in enumerate(side_vfs):
                local_face = str(svf.get("face") or "F")
                if len(vfs) == 1:
                    # Single-face view without U: match by center color if known, else F.
                    s_grid = normalize_grid(svf.get("grid"))
                    canon_face = self._pick_face_by_center_locked(s_grid[1][1], preferred=local_face if local_face in FACES else "F")
                else:
                    offset = 1 if local_face == "R" else (3 if local_face == "L" else s_idx)
                    canon_face = SIDE_RING[(k_total + offset) % 4]
                s_grid = normalize_grid(svf.get("grid"))
                conf_s = float(svf.get("confidence", det.get("confidence", 0.5)))
                for r in range(3):
                    for c in range(3):
                        col = s_grid[r][c]
                        if col != UNKNOWN:
                            observations[canon_face][r][c].add(col)
                if src not in face_sources[canon_face]:
                    face_sources[canon_face].append(src)
                face_confs[canon_face].append(conf_s)
                per_src_map[local_face] = canon_face
                for lvf in live_vfs:
                    if lvf.get("face") == local_face:
                        lvf["canonical_face"] = canon_face

            camera_mappings[src] = {
                "role": role,
                "u_rot_cw": k_total,
                "faces": per_src_map,
            }

        correlated_faces: list[str] = []
        for fkey in FACES:
            if not face_sources[fkey]:
                continue
            voted_grid = empty_grid()
            voted_known = 0
            for r in range(3):
                for c in range(3):
                    colors = observations[fkey][r][c]
                    if len(colors) == 1:
                        voted_grid[r][c] = next(iter(colors))
                        voted_known += 1
            if voted_known:
                correlated_faces.append(fkey)

            if auto_fill:
                cur_face = self._faces[fkey]
                cur_mode = cur_face.get("mode", "")
                if cur_mode in ("manual", "claude"):
                    continue

                avg_conf = sum(face_confs[fkey]) / max(1, len(face_confs[fkey]))
                eff_conf = round(avg_conf * (0.5 + 0.5 * (voted_known / 9.0)), 3)
                self._faces[fkey] = {
                    "grid": voted_grid,
                    "scanned": voted_known > 0,
                    "source": "+".join(face_sources[fkey]),
                    "sources": list(face_sources[fkey]),
                    "mode": "auto",
                    "confidence": min(1.0, eff_conf),
                    "updated_at": now,
                }

        total_known = sum(
            1 for fkey in FACES for row in self._faces[fkey]["grid"] for cell in row if cell != UNKNOWN
        )
        self._correlation = {
            "active_cameras": [src for src, _ in active],
            "camera_mappings": camera_mappings,
            "correlated_faces": correlated_faces,
            "total_known_stickers": total_known,
            "updated_at": now,
        }
        return self._correlation

    def capture_face(
        self,
        face: str | None = None,
        grid: list[list[str]] | None = None,
        source: str = "",
        confidence: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            is_auto = face is None or str(face).strip().upper() == "AUTO"
            if grid is None:
                if not self._live.get("detected"):
                    raise ValueError("No Rubik's cube face currently detected in live view")
                used_source = source or str(self._live.get("source", "live"))
                if is_auto and len(self._live_by_source) >= 2:
                    corr = self._correlate_multicam_locked(auto_fill=True)
                    captured = list(corr.get("correlated_faces") or [])
                    if captured:
                        for fkey in captured:
                            if self._faces[fkey]["scanned"]:
                                self._faces[fkey]["mode"] = "manual"
                        return {
                            "captured_face": captured[0],
                            "captured_faces": captured,
                            "state": self._snapshot_locked(),
                        }
                visible_faces = self._live.get("visible_faces") or []
                if is_auto and len(visible_faces) >= 2:
                    captured_faces: list[str] = []
                    for vf in visible_faces:
                        vf_grid = normalize_grid(vf.get("grid"))
                        known = sum(1 for r in vf_grid for c in r if c != UNKNOWN)
                        if known < 4:
                            continue
                        role = str(vf.get("canonical_face") or vf.get("face") or "").strip().upper()
                        target = self._pick_face_by_center_locked(vf_grid[1][1], preferred=role if role in FACES else None)
                        existing_known = sum(1 for r in self._faces[target]["grid"] for c in r if c != UNKNOWN)
                        if self._faces[target]["scanned"] and self._faces[target].get("mode") == "manual" and existing_known > known:
                            continue
                        vf_conf = float(vf.get("confidence", self._live.get("confidence", 0.0))) if confidence is None else float(confidence)
                        self._faces[target] = {
                            "grid": vf_grid,
                            "scanned": True,
                            "source": used_source,
                            "sources": [used_source],
                            "mode": "manual",
                            "confidence": round(vf_conf, 3),
                            "updated_at": time.time(),
                        }
                        captured_faces.append(target)
                    if captured_faces:
                        return {
                            "captured_face": captured_faces[0],
                            "captured_faces": captured_faces,
                            "state": self._snapshot_locked(),
                        }
                norm_grid = normalize_grid(self._live["grid"])
                used_conf = float(self._live.get("confidence", 0.0)) if confidence is None else float(confidence)
            else:
                norm_grid = normalize_grid(grid)
                used_source = source or "manual"
                used_conf = 1.0 if confidence is None else float(confidence)

            if is_auto:
                target_face = self._pick_face_by_center_locked(norm_grid[1][1])
            else:
                target_face = str(face).strip().upper()
                if target_face not in FACES:
                    raise ValueError(f"Face must be one of {FACES} or AUTO")

            mode = "claude" if used_source == "claude" else "manual"
            self._faces[target_face] = {
                "grid": norm_grid,
                "scanned": True,
                "source": used_source,
                "sources": [used_source],
                "mode": mode,
                "confidence": round(used_conf, 3),
                "updated_at": time.time(),
            }
            return {
                "captured_face": target_face,
                "captured_faces": [target_face],
                "state": self._snapshot_locked(),
            }

    def _pick_face_by_center_locked(self, center_color: str, preferred: str | None = None) -> str:
        if center_color in VALID_COLORS:
            for face in AUTO_FACE_ORDER:
                entry = self._faces[face]
                if entry["scanned"] and entry["grid"][1][1] == center_color:
                    return face
        if preferred and preferred in FACES and not self._faces[preferred]["scanned"]:
            return preferred
        for face in AUTO_FACE_ORDER:
            if not self._faces[face]["scanned"]:
                return face
        return preferred if preferred in FACES else "F"

    def rotate_face(self, face: str, turns_cw: int = 1) -> dict[str, Any]:
        target = str(face).strip().upper()
        if target not in FACES:
            raise ValueError(f"Face must be one of {FACES}")
        with self._lock:
            self._faces[target]["grid"] = rotate_grid_cw(self._faces[target]["grid"], turns_cw)
            self._faces[target]["mode"] = "manual"
            self._faces[target]["updated_at"] = time.time()
            return self._snapshot_locked()

    def set_sticker(self, face: str, row: int, col: int, color: str) -> dict[str, Any]:
        target = str(face).strip().upper()
        if target not in FACES:
            raise ValueError(f"Face must be one of {FACES}")
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError("Row and col must be in 0..2")
        c = str(color).strip().upper()
        if c not in VALID_COLORS and c != UNKNOWN:
            raise ValueError(f"Invalid color: {color!r}")
        with self._lock:
            self._faces[target]["grid"][row][col] = c
            self._faces[target]["scanned"] = any(
                cell != UNKNOWN for r in self._faces[target]["grid"] for cell in r
            )
            self._faces[target]["mode"] = "manual" if self._faces[target]["scanned"] else ""
            self._faces[target]["updated_at"] = time.time()
            return self._snapshot_locked()

    def _validation_locked(self) -> dict[str, Any]:
        counts = {c: 0 for c in VALID_COLORS}
        unknown_count = 0
        scanned_faces = 0
        centers: dict[str, str] = {}
        duplicate_centers: list[str] = []

        for face in FACES:
            entry = self._faces[face]
            grid = entry["grid"]
            if entry["scanned"]:
                scanned_faces += 1
                center = grid[1][1]
                if center in VALID_COLORS:
                    if center in centers.values():
                        duplicate_centers.append(center)
                    centers[face] = center
            for row in grid:
                for cell in row:
                    if cell in counts:
                        counts[cell] += 1
                    else:
                        unknown_count += 1

        issues: list[str] = []
        if duplicate_centers:
            issues.append(f"Duplicate center colors: {', '.join(sorted(set(duplicate_centers)))}")
        over_nine = [f"{c}:{n}" for c, n in counts.items() if n > 9]
        if over_nine:
            issues.append(f"More than 9 stickers detected for {', '.join(over_nine)}")
        if scanned_faces == 6 and unknown_count == 0:
            not_nine = [f"{c}:{n}" for c, n in counts.items() if n != 9]
            if not_nine:
                issues.append(f"Expected 9 of each color, got {', '.join(not_nine)}")
            if len(set(centers.values())) != 6:
                issues.append("All 6 faces must have distinct center colors")

        ready = scanned_faces == 6 and unknown_count == 0 and not issues
        facelet_string = self._facelet_string_locked(centers) if ready else None
        return {
            "scanned_faces": scanned_faces,
            "total_faces": 6,
            "known_stickers": 54 - unknown_count,
            "unknown_stickers": unknown_count,
            "color_counts": counts,
            "centers": centers,
            "issues": issues,
            "ready_to_solve": ready,
            "facelet_string": facelet_string,
        }

    def _facelet_string_locked(self, centers: dict[str, str]) -> str | None:
        color_to_face = {color: face for face, color in centers.items()}
        if len(color_to_face) != 6:
            return None
        chars: list[str] = []
        for face in FACES:
            for row in self._faces[face]["grid"]:
                for cell in row:
                    mapped = color_to_face.get(cell)
                    if not mapped:
                        return None
                    chars.append(mapped)
        return "".join(chars)

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "live": copy.deepcopy(self._live),
            "live_by_source": copy.deepcopy(self._live_by_source),
            "correlation": copy.deepcopy(self._correlation),
            "faces": copy.deepcopy(self._faces),
            "validation": self._validation_locked(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()
