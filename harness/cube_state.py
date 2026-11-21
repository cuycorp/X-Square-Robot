"""Canonical 6-face 3D Rubik's cube state accumulator and validator."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

FACES = ("U", "R", "F", "D", "L", "B")
AUTO_FACE_ORDER = ("F", "R", "U", "L", "B", "D")
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


class CubeStateManager:
    """Thread-safe 3D Rubik's cube state tracker for live detection and multi-side scanning."""

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
        self._reset_faces_locked()

    def _reset_faces_locked(self) -> None:
        self._faces = {
            face: {
                "grid": empty_grid(),
                "scanned": False,
                "source": "",
                "confidence": 0.0,
                "updated_at": None,
            }
            for face in FACES
        }

    def reset(self) -> None:
        with self._lock:
            self._reset_faces_locked()

    def update_live(self, detection: dict[str, Any], source: str = "") -> None:
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
                visible_faces = self._live.get("visible_faces") or []
                if is_auto and len(visible_faces) >= 2:
                    captured_faces: list[str] = []
                    for vf in visible_faces:
                        vf_grid = normalize_grid(vf.get("grid"))
                        known = sum(1 for r in vf_grid for c in r if c != UNKNOWN)
                        if known < 4:
                            continue
                        role = str(vf.get("face", "")).strip().upper()
                        target = self._pick_face_by_center_locked(vf_grid[1][1], preferred=role if role in FACES else None)
                        existing_known = sum(1 for r in self._faces[target]["grid"] for c in r if c != UNKNOWN)
                        if self._faces[target]["scanned"] and existing_known > known:
                            continue
                        vf_conf = float(vf.get("confidence", self._live.get("confidence", 0.0))) if confidence is None else float(confidence)
                        self._faces[target] = {
                            "grid": vf_grid,
                            "scanned": True,
                            "source": used_source,
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

            self._faces[target_face] = {
                "grid": norm_grid,
                "scanned": True,
                "source": used_source,
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
            "faces": copy.deepcopy(self._faces),
            "validation": self._validation_locked(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()
