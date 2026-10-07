"""Camera acquisition: open, configure, trigger, decode and save.

All use of LUCID's ``arena_api`` is confined to :mod:`scanner.camera.helios`.
"""

from .base import Camera
from .config import CameraSettings
from .decode import decode_abcy16, frame_from_raw
from .helios import HeliosCamera
from .simulated import SimulatedCamera
from .storage import CapturePaths, capture_name, load_capture, save_capture, write_ply
from .types import (CameraError, CameraInfo, CameraNotFoundError, ControlSpec,
                    CoordinateTransform, Frame, FrameTimeoutError, IncompleteFrameError)

__all__ = [
    "Camera", "CameraSettings", "HeliosCamera", "SimulatedCamera",
    "Frame", "CameraInfo", "ControlSpec", "CoordinateTransform",
    "CameraError", "CameraNotFoundError", "FrameTimeoutError", "IncompleteFrameError",
    "decode_abcy16", "frame_from_raw",
    "CapturePaths", "capture_name", "save_capture", "load_capture", "write_ply",
]
