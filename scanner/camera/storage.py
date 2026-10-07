"""Saving and reloading captures.

Each capture is three files sharing one name:

* ``<name>.npz``  the raw camera payload, its scale/offsets and the validity
  mask. This is the record of truth; everything else can be rebuilt from it.
* ``<name>.json`` human-readable metadata: camera, settings, timestamps and,
  once the robot is integrated, the pose the frame was taken from.
* ``<name>.ply``  the valid points in metres, for viewing and exchange.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from .decode import decode_abcy16
from .types import PIXEL_FORMAT, CameraInfo, CoordinateTransform, Frame


@dataclass(frozen=True)
class CapturePaths:
    npz: Path
    metadata: Path
    ply: Path | None = None
    native_ply: Path | None = None


def capture_name(when: float | None = None) -> str:
    """A unique, sortable capture name based on the local time."""
    stamp = datetime.fromtimestamp(time.time() if when is None else when)
    return "capture_" + stamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def write_ply(path: Path, points_m: np.ndarray, intensity: np.ndarray | None = None,
              comments: tuple[str, ...] = ()) -> None:
    """Write an (N, 3) point array to a binary PLY file, in metres."""
    points_m = np.asarray(points_m, dtype="<f4").reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0",
              *(f"comment {line}" for line in ("units metres", *comments)),
              f"element vertex {len(points_m)}",
              "property float x", "property float y", "property float z"]
    if intensity is not None:
        fields.append(("intensity", "<u2"))
        header.append("property ushort intensity")
    header.append("end_header")

    vertices = np.empty(len(points_m), dtype=fields)
    vertices["x"], vertices["y"], vertices["z"] = points_m.T
    if intensity is not None:
        vertices["intensity"] = np.asarray(intensity).reshape(-1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        file.write(vertices.tobytes())


def save_capture(frame: Frame, directory: Path, name: str | None = None, *,
                 camera_info: CameraInfo | None = None,
                 settings: dict[str, Any] | None = None,
                 robot_pose: Any = None, ply: bool = True,
                 native_ply: Path | None = None) -> CapturePaths:
    """Save one frame and its metadata, returning the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = name or capture_name()
    paths = CapturePaths(npz=directory / f"{name}.npz",
                         metadata=directory / f"{name}.json",
                         ply=directory / f"{name}.ply" if ply else None,
                         native_ply=native_ply)

    np.savez_compressed(paths.npz, raw=frame.raw, valid=frame.valid,
                        scale_mm=np.asarray(frame.transform.scale_mm),
                        offset_mm=np.asarray(frame.transform.offset_mm),
                        frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns,
                        host_time=frame.host_time)

    if paths.ply is not None:
        write_ply(paths.ply, frame.points_m(), frame.intensity[frame.valid],
                  comments=("frame camera", f"capture {name}"))

    metadata = {
        "capture_id": name,
        "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame": {
            "frame_id": frame.frame_id,
            "camera_timestamp_ns": frame.timestamp_ns,
            "host_time_unix": frame.host_time,
            "width": frame.width,
            "height": frame.height,
            "pixel_format": PIXEL_FORMAT,
            "valid_points": frame.valid_count,
            "valid_fraction": round(frame.valid_fraction, 6),
        },
        "coordinates": {
            "frame": "camera",
            "raw_to_mm": "mm = raw * scale_mm + offset_mm, per axis (X, Y, Z)",
            "scale_mm": list(frame.transform.scale_mm),
            "offset_mm": list(frame.transform.offset_mm),
            "ply_units": "metres",
            "native_ply_units": "millimetres" if native_ply else None,
        },
        "camera": asdict(camera_info) if camera_info else None,
        "settings": settings or {},
        "robot_pose": robot_pose,
        "files": {key: Path(value).name
                  for key, value in asdict(paths).items() if value is not None},
        "software": _software_versions(),
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return paths


def load_capture(npz_path: Path) -> Frame:
    """Rebuild a :class:`Frame` from a saved ``.npz``; no camera or SDK needed."""
    with np.load(Path(npz_path)) as data:
        raw = data["raw"]
        valid = data["valid"]
        transform = CoordinateTransform(tuple(float(v) for v in data["scale_mm"]),
                                        tuple(float(v) for v in data["offset_mm"]))
        frame_id = int(data["frame_id"])
        timestamp_ns = int(data["timestamp_ns"])
        host_time = float(data["host_time"])
    xyz_m, intensity, _ = decode_abcy16(raw, transform.scale_mm, transform.offset_mm)
    return Frame(raw=raw, transform=transform, xyz_m=xyz_m, intensity=intensity, valid=valid,
                 frame_id=frame_id, timestamp_ns=timestamp_ns, host_time=host_time)


def _software_versions() -> dict[str, str]:
    versions = {"scanner": __version__, "python": platform.python_version(),
                "numpy": np.__version__}
    try:
        from importlib.metadata import version
        versions["arena_api"] = version("arena_api")
    except Exception:
        pass
    return versions
