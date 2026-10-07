"""A synthetic camera for developing and testing without hardware.

It produces raw Coord3D_ABCY16 frames with the same shape, dtype and
invalid-value convention as the real camera, so the decode, storage and UI code
run unchanged. The scene is a flat table seen from above with a box on it and
a sphere that circles over it. Noise levels are illustrative, not a model of
the Helios2+.
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
                    CoordinateTransform, Frame)

WIDTH, HEIGHT = 640, 480
FOV_DEG = (31.0, 24.0)
TABLE_Z_M = 0.90
BOX_MIN_M = np.array([-0.14, -0.09, 0.78])
BOX_MAX_M = np.array([0.00, 0.05, TABLE_Z_M])
SPHERE_RADIUS_M = 0.045
BASE_NOISE_M = 0.0015

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
        self._accumulation = int(self.settings.image_accumulation or 1)
        self._spatial_filter = bool(self.settings.spatial_filter)

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
        self._dropout_rate = np.where(albedo < 0.1, 0.6, 0.004).astype(np.float32)
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
            if delay > 0:
                time.sleep(delay)
            self._next_frame_time = max(self._next_frame_time, time.monotonic()) + 1 / self.fps
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
        return [
            ControlSpec("Scan3dImageAccumulation", "int", self._accumulation,
                        minimum=1, maximum=8),
            ControlSpec("Scan3dSpatialFilterEnable", "bool", self._spatial_filter),
        ]

    def set_control(self, name: str, value: Any) -> None:
        if name == "Scan3dImageAccumulation":
            self._accumulation = int(np.clip(int(value), 1, 8))
        elif name == "Scan3dSpatialFilterEnable":
            self._spatial_filter = bool(value)
        else:
            raise CameraError(f"The simulated camera has no '{name}' feature")

    def read_settings(self) -> dict[str, Any]:
        return {"PixelFormat": "Coord3D_ABCY16",
                "Scan3dImageAccumulation": self._accumulation,
                "Scan3dSpatialFilterEnable": self._spatial_filter}

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

        noise = BASE_NOISE_M / np.sqrt(self._accumulation)
        if self._spatial_filter:
            noise *= 0.5
        distance += self._rng.standard_normal(shape, dtype=np.float32) * np.float32(noise)

        intensity = np.float32(9000.0) * albedo / (distance * distance)
        intensity += self._rng.standard_normal(shape, dtype=np.float32) * np.float32(40.0)
        dropout = self._rng.random(shape, dtype=np.float32) < self._dropout_rate

        # Encode the way the camera does: counts = (mm - offset) / scale.
        raw = np.empty(shape + (4,), dtype=np.uint16)
        raw[..., :3] = self._ray_counts * distance[..., None] + self._offset_counts
        raw[..., 3] = np.clip(intensity, 0, 65535)
        raw[dropout, :3] = INVALID_RAW_Z
        raw.setflags(write=False)
        return frame_from_raw(raw, _TRANSFORM, frame_id=self._frame_id,
                              timestamp_ns=time.monotonic_ns(), host_time=time.time(),
                              z_min_m=self.settings.z_min_m, z_max_m=self.settings.z_max_m)

    def _require_open(self) -> None:
        if not self._open:
            raise CameraError("The camera is not open")

    def _require_streaming(self) -> None:
        self._require_open()
        if not self._streaming:
            raise CameraError("The camera is not streaming; call start() first")
