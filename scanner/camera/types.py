"""Data types and errors shared by every camera backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PIXEL_FORMAT = "Coord3D_ABCY16"
"""Four unsigned 16-bit channels per pixel: X, Y, Z and intensity."""

INVALID_RAW_Z = 65535
"""Raw Z value the camera writes for a pixel with no valid depth."""


class CameraError(RuntimeError):
    """Base class for camera failures."""


class CameraNotFoundError(CameraError):
    """No matching camera was discovered."""


class FrameTimeoutError(CameraError, TimeoutError):
    """No frame arrived before the timeout expired."""


class IncompleteFrameError(CameraError):
    """A frame arrived with missing or unexpected data."""


@dataclass(frozen=True)
class CoordinateTransform:
    """Converts raw channel counts to millimetres: ``mm = raw * scale + offset``.

    Scales and offsets are stored per axis in X, Y, Z order.
    """

    scale_mm: tuple[float, float, float]
    offset_mm: tuple[float, float, float]


@dataclass(frozen=True)
class CameraInfo:
    """Identity of a camera."""

    model: str
    serial: str
    firmware: str = ""
    ip: str = ""
    mac: str = ""
    simulated: bool = False


@dataclass(frozen=True)
class ControlSpec:
    """One adjustable camera feature, described so a UI can build a widget for it."""

    name: str
    kind: str  # "enum", "bool", "int" or "float"
    value: Any
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    writable: bool = True


@dataclass
class Frame:
    """One decoded capture, in the camera's own coordinate frame.

    ``raw`` is the payload exactly as the camera sent it and is never modified;
    everything else is derived from it with ``transform``.
    """

    raw: np.ndarray  # (H, W, 4) uint16
    transform: CoordinateTransform
    xyz_m: np.ndarray  # (H, W, 3) float64, metres
    intensity: np.ndarray  # (H, W) uint16
    valid: np.ndarray  # (H, W) bool
    frame_id: int = 0
    timestamp_ns: int = 0  # camera clock, not synchronised to the host
    host_time: float = 0.0  # time.time() when the frame was received
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.raw.shape[0]

    @property
    def width(self) -> int:
        return self.raw.shape[1]

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def valid_fraction(self) -> float:
        return self.valid_count / self.valid.size

    def points_m(self) -> np.ndarray:
        """Valid points as an (N, 3) array in metres."""
        return self.xyz_m[self.valid]
