"""A synthetic camera for developing and testing without hardware.

It produces raw Coord3D_ABCY16 frames with the same shape, dtype and
invalid-value convention as the real camera, so the decode, storage and UI code
run unchanged. The scene is a flat table seen from above with a box on it and
a sphere that circles over it. Noise levels are illustrative, not a model of
the Helios2+, and the on-camera settings (HDR, filters, distance limits) have
simple stand-ins so the UI can be exercised without hardware.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Camera
from .config import CameraSettings
from .decode import frame_from_raw
from .types import (INVALID_RAW_Z, CameraError, CameraInfo, ControlSpec,
                    CoordinateTransform, Frame, FrameTimeoutError)

WIDTH, HEIGHT = 640, 480
FOV_DEG = (31.0, 24.0)
TABLE_Z_M = 0.90
BOX_MIN_M = np.array([-0.14, -0.09, 0.78])
BOX_MAX_M = np.array([0.00, 0.05, TABLE_Z_M])
SPHERE_RADIUS_M = 0.045
BASE_NOISE_M = 0.0015
# Neighbouring pixels further apart than this straddle an edge and partly see
# both surfaces, so they return a depth in between: "flying pixels".
MIXED_PIXEL_STEP_M = 0.02

# Scan3dHDRMode entries: (exposures averaged, highest frame rate). Illustrative.
HDR_MODES = {
    "Off": (1, None),
    "StandardHDR": (1, 10.0),
    "LowNoiseHDRX4": (4, 5.0),
    "LowNoiseHDRX8": (8, 3.0),
    "LowNoiseHDRX16": (16, 1.67),
    "LowNoiseHDRX32": (32, 0.88),
}

_TRANSFORM = CoordinateTransform(scale_mm=(0.25, 0.25, 0.25),
                                 offset_mm=(-8192.0, -8192.0, 0.0))


class SimulatedCamera(Camera):
    """Drop-in stand-in for :class:`HeliosCamera`."""

    def __init__(self, settings: CameraSettings | None = None, fps: float = 15.0,
                 seed: int | None = None):
        super().__init__(settings)
        self.fps = fps
        self._rng = np.random.default_rng(seed)
        self._open = False
        self._streaming = False
        self._triggered = False
        self._frame_id = 0
        self._next_frame_time = 0.0
        self._start_time = 0.0
        self._values: dict[str, Any] = {name: spec[-1] for name, spec in _CONTROLS.items()}
        requested = self.settings.node_values()
        for name in _CONTROLS:  # in order, so HDR is set before accumulation
            if name in requested:
                self.set_control(name, requested[name])

        u = (np.arange(WIDTH) - (WIDTH - 1) / 2) / (WIDTH / 2)
        v = (np.arange(HEIGHT) - (HEIGHT - 1) / 2) / (HEIGHT / 2)
        x = u[None, :] * np.tan(np.radians(FOV_DEG[0] / 2))
        y = v[:, None] * np.tan(np.radians(FOV_DEG[1] / 2))
        rays = np.stack(np.broadcast_arrays(x, y, 1.0), axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        static_range, albedo = self._trace_static_scene(rays)
        self._rays = rays.astype(np.float32)
        self._static_range = static_range.astype(np.float32)
        self._static_albedo = albedo.astype(np.float32)
        scale = np.asarray(_TRANSFORM.scale_mm)
        self._ray_counts = (rays * 1000.0 / scale).astype(np.float32)
        # The extra 0.5 turns the integer truncation below into rounding.
        self._offset_counts = (-np.asarray(_TRANSFORM.offset_mm) / scale + 0.5).astype(np.float32)

    # lifecycle ---------------------------------------------------------------

    def open(self) -> None:
        self.info = CameraInfo(model="Simulated Helios2+", serial="SIM0001",
                               firmware="simulated", simulated=True)
        self._open = True

    def close(self) -> None:
        self._streaming = False
        self._open = False

    def start(self, triggered: bool = False) -> None:
        self._require_open()
        self._streaming = True
        self._triggered = triggered
        self._start_time = time.monotonic()
        self._next_frame_time = self._start_time

    def stop(self) -> None:
        self._streaming = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    # acquisition -------------------------------------------------------------

    def grab(self, timeout_s: float = 2.0, discard: int = 0,
             native_ply: Path | None = None) -> Frame:
        self._require_streaming()
        if self._triggered:
            raise CameraError("The stream is waiting for triggers; call trigger() instead")
        for _ in range(discard + 1):
            delay = self._next_frame_time - time.monotonic()
            if delay > timeout_s:
                time.sleep(timeout_s)
                raise FrameTimeoutError(f"No frame within {timeout_s} s")
            if delay > 0:
                time.sleep(delay)
            self._next_frame_time = (max(self._next_frame_time, time.monotonic())
                                     + self._frame_interval())
            self._frame_id += 1
        return self._render()

    def trigger(self, timeout_s: float = 2.0, native_ply: Path | None = None) -> Frame:
        self._require_streaming()
        if not self._triggered:
            raise CameraError("Call start(triggered=True) before trigger()")
        self._frame_id += 1
        return self._render()

    # settings ----------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        hdr_on = self._values["Scan3dHDRMode"] != "Off"
        return [ControlSpec(name, kind, self._values[name], options=options,
                            minimum=minimum, maximum=maximum,
                            writable=not (hdr_on and name == "Scan3dImageAccumulation"))
                for name, (kind, options, minimum, maximum, _) in _CONTROLS.items()]

    def set_control(self, name: str, value: Any) -> None:
        if name not in _CONTROLS:
            raise CameraError(f"The simulated camera has no '{name}' feature")
        kind, options, minimum, maximum, _ = _CONTROLS[name]
        if name == "Scan3dImageAccumulation" and self._values["Scan3dHDRMode"] != "Off":
            raise CameraError("Scan3dImageAccumulation is locked while Scan3dHDRMode is on")
        if kind == "enum":
            if str(value) not in options:
                raise CameraError(f"{name} has no option {value!r}. "
                                  f"Available: {', '.join(options)}")
            value = str(value)
        elif kind == "bool":
            value = bool(value)
        else:
            value = int(np.clip(int(value), minimum, maximum))
        self._values[name] = value

    def read_settings(self) -> dict[str, Any]:
        return {"PixelFormat": "Coord3D_ABCY16", **self._values}

    def _frame_interval(self) -> float:
        hdr_rate = HDR_MODES[self._values["Scan3dHDRMode"]][1]
        return max(1 / self.fps, 1 / hdr_rate if hdr_rate else 0.0)

    # scene -------------------------------------------------------------------

    @staticmethod
    def _trace_static_scene(rays):
        """Range along each ray to the table or the box, whichever is nearer."""
        table_range = TABLE_Z_M / rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_low = BOX_MIN_M / rays
            t_high = BOX_MAX_M / rays
        t_enter = np.nanmax(np.minimum(t_low, t_high), axis=-1)
        t_exit = np.nanmin(np.maximum(t_low, t_high), axis=-1)
        hits_box = (t_enter <= t_exit) & (t_enter > 0)
        static_range = np.where(hits_box, t_enter, table_range)
        albedo = np.where(hits_box, 0.75, 0.45)
        # A dark patch on the table: weak return, so many pixels drop out.
        table_point = rays * table_range[..., None]
        dark = (~hits_box & (np.abs(table_point[..., 0] - 0.16) < 0.04)
                & (np.abs(table_point[..., 1] + 0.10) < 0.03))
        albedo = np.where(dark, 0.04, albedo)
        return static_range, albedo

    def _render(self) -> Frame:
        shape = (HEIGHT, WIDTH)
        angle = 0.8 * (time.monotonic() - self._start_time)
        centre = np.array([0.11 + 0.07 * np.cos(angle), 0.02 + 0.07 * np.sin(angle),
                           TABLE_Z_M - SPHERE_RADIUS_M - 0.03], dtype=np.float32)
        b = self._rays @ centre
        discriminant = b * b - (centre @ centre - SPHERE_RADIUS_M ** 2)
        sphere_range = b - np.sqrt(np.maximum(discriminant, 0.0))
        hits_sphere = (discriminant > 0) & (sphere_range < self._static_range)

        distance = np.where(hits_sphere, sphere_range, self._static_range)
        albedo = np.where(hits_sphere, np.float32(0.9), self._static_albedo)

        distance = self._mix_edges(distance)

        values = self._values
        if values["Scan3dHDRMode"] == "Off":
            averaged = values["Scan3dImageAccumulation"]
        else:
            averaged = HDR_MODES[values["Scan3dHDRMode"]][0]
        # Weak returns (dark surfaces) are noisier.
        noise = BASE_NOISE_M / np.sqrt(averaged) * np.sqrt(0.45 / albedo)
        if values["Scan3dSpatialFilterEnable"]:
            noise *= 0.5
        distance += self._rng.standard_normal(shape, dtype=np.float32) * noise.astype(np.float32)

        intensity = np.float32(9000.0) * albedo / (distance * distance)
        intensity += self._rng.standard_normal(shape, dtype=np.float32) * np.float32(40.0)

        dropout = self._rng.random(shape, dtype=np.float32) < 0.004
        if values["Scan3dConfidenceThresholdEnable"]:
            dropout |= intensity < values["Scan3dConfidenceThresholdMin"]
        if values["Scan3dFlyingPixelsRemovalEnable"]:
            threshold_m = values["Scan3dFlyingPixelsDistanceThreshold"] / 1000
            dropout |= _flying_pixels(distance, threshold_m)
        dropout |= ((distance < values["Scan3dDistanceMin"] / 1000)
                    | (distance > values["Scan3dDistanceMax"] / 1000))

        # Encode the way the camera does: counts = (mm - offset) / scale.
        raw = np.empty(shape + (4,), dtype=np.uint16)
        raw[..., :3] = self._ray_counts * distance[..., None] + self._offset_counts
        raw[..., 3] = np.clip(intensity, 0, 65535)
        raw[dropout, :3] = INVALID_RAW_Z
        raw.setflags(write=False)
        return frame_from_raw(raw, _TRANSFORM, frame_id=self._frame_id,
                              timestamp_ns=time.monotonic_ns(), host_time=time.time(),
                              z_min_m=self.settings.z_min_m, z_max_m=self.settings.z_max_m)

    def _mix_edges(self, distance: np.ndarray) -> np.ndarray:
        """Give each pixel just past a depth step a depth partway across the step."""
        mixed = distance.copy()
        share = self._rng.random(distance.shape, dtype=np.float32)
        for axis in (0, 1):
            ahead = [slice(None)] * 2
            behind = [slice(None)] * 2
            ahead[axis], behind[axis] = slice(1, None), slice(None, -1)
            ahead, behind = tuple(ahead), tuple(behind)
            step = np.abs(distance[ahead] - distance[behind]) > MIXED_PIXEL_STEP_M
            target = mixed[ahead]
            blend = distance[ahead] + share[ahead] * (distance[behind] - distance[ahead])
            target[step] = blend[step]
        return mixed

    def _require_open(self) -> None:
        if not self._open:
            raise CameraError("The camera is not open")

    def _require_streaming(self) -> None:
        self._require_open()
        if not self._streaming:
            raise CameraError("The camera is not streaming; call start() first")


# Simulated features: kind, enum options, minimum, maximum, default.
_CONTROLS = {
    "Scan3dHDRMode": ("enum", tuple(HDR_MODES), None, None, "Off"),
    "Scan3dImageAccumulation": ("int", (), 1, 8, 1),
    "Scan3dSpatialFilterEnable": ("bool", (), None, None, False),
    "Scan3dFlyingPixelsRemovalEnable": ("bool", (), None, None, False),
    "Scan3dFlyingPixelsDistanceThreshold": ("int", (), 0, 1000, 20),
    "Scan3dConfidenceThresholdEnable": ("bool", (), None, None, True),
    "Scan3dConfidenceThresholdMin": ("int", (), 0, 10000, 1000),
    "Scan3dDistanceMin": ("int", (), 0, 5000, 0),
    "Scan3dDistanceMax": ("int", (), 0, 5000, 5000),
}


def _flying_pixels(distance: np.ndarray, threshold_m: float) -> np.ndarray:
    """Pixels further than the threshold from both neighbours along a row or column.

    A pixel on a clean edge differs from one side only, so it is kept.
    """
    flying = np.zeros(distance.shape, dtype=bool)
    middle = distance[:, 1:-1]
    flying[:, 1:-1] |= ((np.abs(middle - distance[:, :-2]) > threshold_m)
                        & (np.abs(middle - distance[:, 2:]) > threshold_m))
    middle = distance[1:-1]
    flying[1:-1] |= ((np.abs(middle - distance[:-2]) > threshold_m)
                     & (np.abs(middle - distance[2:]) > threshold_m))
    return flying
